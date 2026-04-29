import atexit
from dataclasses import fields
from time import perf_counter
import threading
from tqdm.auto import tqdm
from transformers import AutoTokenizer, Qwen2Tokenizer
import torch.multiprocessing as mp
from sparsevllm.utils.log import logger
import sys

from sparsevllm.config import Config
from sparsevllm.sampling_params import SamplingParams
from sparsevllm.engine.sequence import Sequence
from sparsevllm.engine.scheduler import Scheduler
from sparsevllm.engine.model_runner import ModelRunner
from sparsevllm.utils.profiler import profiler

class _ThroughputIntervalLogger: # 定时打印日志的守护线程
    def __init__(self, interval_s: float):
        self._interval_s = float(interval_s)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prefill_tokens = 0
        self._decode_tokens = 0
        self._running_seqs = 0
        self._prefill_seqs = 0
        self._decode_seqs = 0
        self._prefill_long_seqs = 0
        self._prefill_short_seqs = 0
        self._decode_long_seqs = 0
        self._decode_short_seqs = 0
        self._last_batch = "idle"  # "pf-L", "pf-S", "dc-L", "dc-S", "idle"
        self._last_report_t = perf_counter()

    def start(self): # 专门开一个后台线程，每隔固定的时间（比如 10-30 秒）醒来一次，计算并打印当前引擎的吞吐量（平均处理了多少 token）
        if self._interval_s <= 0:
            return
        if self._thread is not None:
            return
        with self._lock:
            self._last_report_t = perf_counter()
        self._thread = threading.Thread(target=self._run, name="svllm-throughput-logger", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=self._interval_s + 1.0)

    def record_step(self, num_tokens: int):
        if num_tokens == 0:
            return
        with self._lock:
            if num_tokens > 0:
                self._prefill_tokens += int(num_tokens)
            else:
                self._decode_tokens += int(-num_tokens)

    def record_state(
        self,
        running_seqs: int,
        prefill_seqs: int,
        decode_seqs: int,
        prefill_long_seqs: int,
        prefill_short_seqs: int,
        decode_long_seqs: int,
        decode_short_seqs: int,
        last_batch: str,
    ):
        with self._lock:
            self._running_seqs = int(running_seqs)
            self._prefill_seqs = int(prefill_seqs)
            self._decode_seqs = int(decode_seqs)
            self._prefill_long_seqs = int(prefill_long_seqs)
            self._prefill_short_seqs = int(prefill_short_seqs)
            self._decode_long_seqs = int(decode_long_seqs)
            self._decode_short_seqs = int(decode_short_seqs)
            self._last_batch = str(last_batch)

    def _run(self): #  # 定时唤醒打印日志
        while not self._stop.wait(self._interval_s):
            now = perf_counter()
            with self._lock:
                prefill_tokens = self._prefill_tokens
                decode_tokens = self._decode_tokens
                running_seqs = self._running_seqs
                prefill_seqs = self._prefill_seqs
                decode_seqs = self._decode_seqs
                prefill_long_seqs = self._prefill_long_seqs
                prefill_short_seqs = self._prefill_short_seqs
                decode_long_seqs = self._decode_long_seqs
                decode_short_seqs = self._decode_short_seqs
                last_batch = self._last_batch
                self._prefill_tokens = 0
                self._decode_tokens = 0
                last_t = self._last_report_t
                self._last_report_t = now

            dt = max(now - last_t, 1e-9)
            prefill_tp = prefill_tokens / dt
            decode_tp = decode_tokens / dt
            logger.info(
                "Avg TP (last {dt:.1f}s): prefill_tp={prefill_tp:.0f} tok/s, decode_tp={decode_tp:.0f} tok/s "
                "| seq(run/prf/dc)={running_seqs}/{prefill_seqs}/{decode_seqs} "
                "| prf(L/S)={prefill_long_seqs}/{prefill_short_seqs} dc(L/S)={decode_long_seqs}/{decode_short_seqs} "
                "| last_batch={last_batch} "
                "(prefill_tokens={prefill_tokens}, decode_tokens={decode_tokens})",
                dt=dt,
                prefill_tokens=prefill_tokens,
                prefill_tp=prefill_tp,
                decode_tokens=decode_tokens,
                decode_tp=decode_tp,
                running_seqs=running_seqs,
                prefill_seqs=prefill_seqs,
                decode_seqs=decode_seqs,
                prefill_long_seqs=prefill_long_seqs,
                prefill_short_seqs=prefill_short_seqs,
                decode_long_seqs=decode_long_seqs,
                decode_short_seqs=decode_short_seqs,
                last_batch=last_batch,
            )

class LLMEngine:
    """
    Sparse-vLLM 推理引擎的核心入口类。
    负责协调 Tokenizer、调度器 (Scheduler) 和模型执行器 (ModelRunner)。
    管理多进程张量并行 (Tensor Parallelism) 的生命周期。
    """

    def __init__(self, model, **kwargs):
        """
        初始化 LLMEngine，完成以下任务：
        1. 加载配置和模型分词器
        2. 启动多进程张量并行 (TP) 环境
        3. 创建调度器和缓存管理器
        4. 预热模型，确保所有算子就绪
        """
        # === 第1步：基础配置初始化 ===
        # 从 Config dataclass 中过滤出有效参数，避免传入不识别的 kwargs
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        
        # 启用/禁用性能分析器
        profiler.set_enabled(config.enable_profiler)
        
        # === 第2步：启动多进程张量并行 (TP) 环境 ===
        # Rank 0 在主进程运行，Rank 1..N-1 各启动独立的子进程
        # 这样可以充分利用多 GPU 的计算能力
        self.ps = []  # 子进程对象列表，用于最后的清理和等待
        self.events = []  # 进程间同步事件，用于协调多进程启动
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            # 为每个非零 Rank 启动独立的 ModelRunner 进程
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        
        # === 第3步：初始化 Rank 0 的 ModelRunner ===
        # 必须先于 Scheduler 创建，以便在 GPU 上初始化 KV Cache 物理内存账本
        self.model_runner = ModelRunner(config, 0, self.events)
        
        # === 第4步：加载分词器 ===
        self.tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        
        # === 第5步：初始化调度器 ===
        # 关键设计：将主进程的 CacheManager 传给 Scheduler
        # Scheduler 通过 CacheManager 感知剩余显存，实现智能调度和抢占策略
        self.scheduler = Scheduler(config, self.model_runner.cache_manager)
        
        # === 第6步：设置日志和资源清理 ===
        self._exited = False
        self._throughput_logger = _ThroughputIntervalLogger(config.throughput_log_interval_s)
        # 注册退出钩子，确保异常退出时仍能正确清理多进程资源
        atexit.register(self.exit)

        # === 第7步：预热模型 ===
        # 触发算子编译和内存分配，预热后系统可投入使用
        self._warmup()
        self._throughput_logger.start()

    def _warmup(self):
        """预热模型，确保所有算子和显存都已就绪"""
        logger.info("Warming up the engine...")
        
        # 预热只需触发算子编译，使用固定短长度即可
        if self.config.vllm_sparse_method == "deltakv-snapkv":
            warmup_len = (
                self.config.num_sink_tokens
                + self.config.num_recent_tokens
                + self.config.snapkv_window_size
                + self.config.chunk_prefill_size
                + 1024
            )
        elif self.config.vllm_sparse_method == "deltakv-standalone":
            warmup_len = (
                self.config.num_sink_tokens
                + self.config.num_recent_tokens
                + self.config.chunk_prefill_size
                + 1024
            )
        else:
            warmup_len = self.config.num_sink_tokens + self.config.num_top_tokens_in_prefill\
                         + self.config.num_recent_tokens + self.config.chunk_prefill_size + 1024
        # DeepSeek MLA paths often use large chunk_prefill_size to keep prefill non-chunked; keep warmup short.
        if getattr(getattr(self.config, "hf_config", None), "model_type", "") in ("deepseek_v32", "deepseek_v2"):
            warmup_len = min(1024, int(self.config.chunk_prefill_size))
        num_seqs = 1
        
        # 预热 1 个 Token 的生成（包含 Prefill 和 Decode）
        sampling_params = SamplingParams(max_tokens=1)
        max_prompt_len = max(1, int(self.config.max_model_len) - int(sampling_params.max_tokens))
        if warmup_len > max_prompt_len:
            logger.warning(
                f"Warmup prompt length ({warmup_len}) exceeds max_model_len - max_tokens "
                f"({max_prompt_len}). Clamping warmup_len to {max_prompt_len}."
            )
            warmup_len = max_prompt_len
        dummy_prompt = [0] * warmup_len
        
        for _ in range(num_seqs):
            self.add_request(dummy_prompt, sampling_params)
            
        while not self.is_finished():
            self.step()
        logger.info("Warmup finished.")

    def exit(self):
        """优雅地退出所有子进程并清理共享内存"""
        if self._exited:
            return
        self._exited = True

        profiler.print_stats()
        if hasattr(self, "_throughput_logger"):
            self._throughput_logger.stop()
        if hasattr(self, "model_runner"):
            self.model_runner.call("exit")
            del self.model_runner
        if hasattr(self, "ps"):
            for p in self.ps:
                if p.is_alive():
                    p.terminate()
                p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """将一个新的推理请求加入系统"""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        prompt_len = len(prompt)
        max_tokens = sampling_params.max_tokens
        if prompt_len + max_tokens > self.config.max_model_len:
            raise ValueError(
                "Prompt length + max_tokens exceeds max_model_len: "
                f"{prompt_len} + {max_tokens} > {self.config.max_model_len}. "
                "Reduce prompt/decoding length or increase max_model_len if the model supports it."
            )
        logger.debug(f'add prompt with {len(prompt)} tokens.')
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """
        执行引擎的单次推理迭代（心跳循环）。
        每次调用都会挑出一批序列，让 GPU 执行前向计算（预填充提示词，或者生成一个新字），并自动管理显存分配。
        
        返回值：
        - finished_outputs: 遇到终止符、已完成生成的序列结果 [(seq_id, token_ids), ...]
        - num_tokens: 本次迭代处理的总 token 数（方便外部算吞吐量）
        """
        with profiler.record("step"):
            
            # --- 1. 队列调度 (Scheduling) ---
            # 不混跑：要么全是 prefill，要么全是 decode
            # 可能返回被抢占的序列 (显存不足时)
            with profiler.record("schedule"):
                seqs, is_prefill, preempted_seqs = self.scheduler.schedule()
            
            # --- 2. 抢占释放 (Preemption) ---
            # 如果显存不够，调度器会牺牲掉部分序列 (preempted_seqs)。
            # 跨进程释放被抢占序列的 KV Cache 物理 slot
            with profiler.record("preempt_free"):
                for seq in preempted_seqs:
                    self.model_runner.call("free_slots", seq.seq_id)
                
            # --- 兜底检查：如果本轮无事可做 ---
            # 如果什么序列都没拿到，正常情况是因为都跑完了，或者刚好在做大清理。
            # 如果不是这俩原因，说明死锁卡住了，直接报错。
            if not seqs:
                if preempted_seqs or self.is_finished():
                    # 把排队状况上报给监控，然后本回合空跑结束
                    prefill_seqs = len(self.scheduler.waiting)
                    decode_seqs = len(self.scheduler.decoding)
                    prefill_threshold = self.scheduler._long_text_threshold(is_prefill=True)
                    decode_threshold = self.scheduler._long_text_threshold(is_prefill=False)
                    prefill_long = sum(1 for s in self.scheduler.waiting if int(s.num_prompt_tokens) > int(prefill_threshold))
                    decode_long = sum(1 for s in self.scheduler.decoding if int(s.num_tokens) > int(decode_threshold))
                    self._throughput_logger.record_state(
                        prefill_seqs + decode_seqs, prefill_seqs, decode_seqs,
                        prefill_long, prefill_seqs - prefill_long, decode_long, decode_seqs - decode_long, "idle",
                    )
                    return [], 0
                
                raise RuntimeError(
                    f"死锁：调度器拿不到活儿，也没释放资源！方法={self.config.vllm_sparse_method} "
                    f"空闲显存块={self.model_runner.cache_manager.num_free_slots}"
                )
                
            # --- 3. 模型执行 (Model Forward) ---
            # 所有 Rank (含子进程) 同步执行模型前向 + 采样
            with profiler.record("model_run_call"):
                token_ids, attn_score = self.model_runner.call("run", seqs, is_prefill)
            
            # --- 4. 后处理状态更新 (Postprocessing) ---
            '''→ 更新序列状态：prefill 进度递增、decode token 追加
            → 序列在 waiting/decoding 之间迁移
            → 标记完成的序列'''
            with profiler.record("postprocess"):
                self.scheduler.postprocess(seqs, token_ids, is_prefill)
            
            # --- 5. 清理完成序列 (Cleanup) ---
            # 对于刚才判断已经全剧终的序列，马上通知显卡释放它们的显存区，让位给别的请求。
            with profiler.record("finished_free"):
                finished_outputs = []
                for seq in seqs:
                    if seq.is_finished:
                        self.model_runner.call("free_slots", seq.seq_id)
                        finished_outputs.append((seq.seq_id, seq.completion_token_ids))
        
        # --- 6. 记账与性能上报 (Throughput Logging) ---
        # 算一下刚刚这一回合干了多少活，喂给后台线程去打日志和画吞吐图。
        num_tokens = sum(seq.current_chunk_size for seq in seqs) if is_prefill else -len(seqs)
        self._throughput_logger.record_step(num_tokens)
        
        prefill_seqs = len(self.scheduler.waiting)
        decode_seqs = len(self.scheduler.decoding)
        prefill_threshold = self.scheduler._long_text_threshold(is_prefill=True)
        decode_threshold = self.scheduler._long_text_threshold(is_prefill=False)
        prefill_long = sum(1 for s in self.scheduler.waiting if int(s.num_prompt_tokens) > int(prefill_threshold))
        decode_long = sum(1 for s in self.scheduler.decoding if int(s.num_tokens) > int(decode_threshold))
        
        if is_prefill:
            batch_is_long = bool(int(seqs[0].num_prompt_tokens) > int(prefill_threshold))
            stage = "pf"
        else:
            batch_is_long = bool(int(seqs[0].num_tokens) > int(decode_threshold))
            stage = "dc"
            
        self._throughput_logger.record_state(
            prefill_seqs + decode_seqs, prefill_seqs, decode_seqs, prefill_long,
            prefill_seqs - prefill_long, decode_long, decode_seqs - decode_long,
            f"{stage}-{'L' if batch_is_long else 'S'}",
        )
        
        return finished_outputs, num_tokens

    def is_finished(self):
        """检查是否所有请求都已处理完毕"""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        """
        高层 API：批量输入 Prompt，阻塞直至全部生成完成。
        返回包含生成的 text 和 token_ids 的字典列表。
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # 提交所有请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
            
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        
        # 主推理循环
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            
            # 更新吞吐量统计
            if use_tqdm:
                dt = perf_counter() - t
                if num_tokens > 0:
                    prefill_throughput = num_tokens / dt
                else:
                    decode_throughput = -num_tokens / dt
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            
            # 收集已完成的输出
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
                    
        # 按照请求提交顺序排序并解码
        results = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        results = [{"text": self.tokenizer.decode(tids, skip_special_tokens=True), "token_ids": tids} for tids in results]
        
        if use_tqdm:
            pbar.close()
        return results
