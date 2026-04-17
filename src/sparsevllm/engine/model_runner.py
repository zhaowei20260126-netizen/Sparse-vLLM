import os
import pickle
import torch
import torch.distributed as dist
from sparsevllm.utils.log import logger
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence
from sparsevllm.models.qwen2 import Qwen2ForCausalLM
from sparsevllm.models.qwen3 import Qwen3ForCausalLM
from sparsevllm.models.deepseek_v2 import DeepSeekV2ForCausalLM
from sparsevllm.layers.sampler import Sampler
from sparsevllm.utils.context import set_context, get_context, reset_context
from sparsevllm.utils.loader import load_model, sync_deltakv_config_from_checkpoint

from sparsevllm.engine.cache_manager import CacheManager
from sparsevllm.engine.sparse_controller import SparseController
from sparsevllm.utils.profiler import profiler

class ModelRunner:
    """
    负责模型执行的类。每个 GPU Rank 进程都拥有一个 ModelRunner 实例。
    主要职责：权重加载、显存分配 (KV Cache)、槽位管理 (Rank-Local)、前向计算。
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        初始化 ModelRunner。每个 GPU Rank 进程都有一个独立的 ModelRunner 实例。
        主要流程：分布式初始化 → 加载模型权重 → 初始化 KV Cache 和调度结构 → 多进程协调
        """
        self.config = config
        
        # === 准备基础环境 ===
        # 禁用自动求导，这个进程是推理专用，不需要梯度计算
        torch.set_grad_enabled(False)
        
        # 将当前进程的 rank 编号告诉分析器，用于性能统计时标识哪个进程
        profiler.set_rank(rank)
        
        # 启用/禁用性能分析。仅 Rank 0 输出统计信息，避免多进程重复输出
        profiler.set_enabled(config.enable_profiler and rank == 0)
        
        # 从全局配置中提取 HuggingFace 模型配置（包含模型架构、dtype、各层大小等）
        hf_config = config.hf_config
        
        # 是否强制使用 Eager 执行模式。如果为 True，禁用 torch.compile/Inductor 优化
        # 用于调试性能问题或在不支持编译的硬件上运行
        self.enforce_eager = config.enforce_eager
        
        # 张量并行的总进程数。例如 TP=4 表示模型权重分片到 4 个 GPU 上
        # 后续用于确定是否需要启用多进程通信机制
        self.world_size = config.tensor_parallel_size
        
        # 当前进程在张量并行组中的编号，从 0 开始
        # Rank 0 创建共享内存用于多进程 RPC 通信，所有 Rank 都独立加载权重分片进行并行计算
        # 在推理执行时，只有 Rank 0 负责 Token 采样和向其他 Rank 广播方法指令
        self.rank = rank
        
        # 多进程同步事件对象，用于 Rank 0 通知其他 Rank 有新的方法调用指令
        # 类型为 Event 或 list[Event]（单进程时为 Event，多进程时为 Event 列表）
        self.event = event

        # === 第1步：初始化分布式环境 ===
        # 多 GPU 通过 NCCL 进行通信协调，每个进程绑定到对应的 GPU
        # 目的：建立一个分布式进程组，让多个 GPU 上的进程能互相发送消息和同步
        if not dist.is_initialized():
            # 从环境变量读取主节点通信端口，默认 2333
            # 所有进程都会连接到 localhost:master_port 来建立通信
            master_port = int(os.getenv("SPARSEVLLM_MASTER_PORT", "2333"))
        
            # 初始化分布式进程组，这个调用会进行 TCP 握手，让所有标记相同 world_size 和 rank 的进程相互发现
            dist.init_process_group(
                "nccl",  # 使用 NCCL 后端。NCCL 是 NVIDIA 的集合通信库，优化了 GPU 间的高速通信
                         # 比 CPU 间的 TCP/gloo 快得多
                f"tcp://localhost:{master_port}",  # 所有进程通过这个地址进行初始化握手
                                                    # 将通过 TCP 连接到这个端口完成进程发现和同步
                world_size=self.world_size,  # 总进程数。例如 TP=4 表示有 4 个进程参与通信
                                             # 所有进程都必须指定相同的 world_size，否则握手失败
                rank=rank,  # 当前进程的编号，从 0 开始。Rank 0 通常作为主进程
                            # 每个进程必须指定唯一的 rank（0 到 world_size-1），不能重复
            )
        
        # 绑定当前进程到对应的 GPU
        # rank 0 → GPU 0, rank 1 → GPU 1, 以此类推
        # 这样可以确保：
        #   1. 每个进程的 GPU 操作都在其对应的 GPU 上执行
        #   2. GPU 内存分配不会超出对应 GPU 的显存
        #   3. NCCL 通信会使用最优的 GPU 对应的高速互联（e.g., NVLink）
        torch.cuda.set_device(rank)
        
        # === 第2步：设置设备和数据类型 ===
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)  # 模型权重数据类型（通常 fp16 或 bf16）
        torch.set_default_device("cuda")  # 后续分配的张量都在 GPU 上
        
        # === 第3步：加载模型架构并初始化权重 ===
        # 根据模型类型创建对应的模型类，然后从 checkpoint 加载权重
        # 在 TP 场景下，只加载该进程负责的权重分片
        if hf_config.model_type == "qwen2":
            self.model = Qwen2ForCausalLM(hf_config)
        elif hf_config.model_type == "deepseek_v2":
            self.model = DeepSeekV2ForCausalLM(
                hf_config,
                dsa_topk=config.dsa_topk,
                use_flash_mla=config.dsa_use_flash_mla,
            )
        elif hf_config.model_type == "deepseek_v32":
            raise NotImplementedError(
                "DeepSeek-V3.2 sparsevllm support is disabled. "
                "Use DeepSeek-V2 or another backend for now."
            )
        else:
            self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model, rank=rank, world_size=self.world_size)
        
        # === 第4步：初始化采样器 ===
        self.sampler = Sampler()

        # === 第5步：初始化 KV Cache 和稀疏方法相关结构 ===
        # 先同步 DeltaKV 压缩器的架构配置（如果使用 DeltaKV 方法）
        sync_deltakv_config_from_checkpoint(config)
        
        # 创建 CacheManager —— 这是关键的内存管理器，负责 KV Cache 物理内存的分配和回收
        # 支持多种稀疏方法（vanilla、DeltaKV、OmniKV 等）
        self.cache_manager = CacheManager.create(config, rank, self.world_size)

        # 初始化稀疏控制器 —— 决定每一步中哪些 token 被保留或丢弃
        self.sparse_controller = SparseController(config, self.cache_manager)
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # 将稀疏控制器注入到模型中（仅 Qwen 风格的模型支持；DeepSeek MLA 使用不同机制）
            self.model.model.sparse_controller = self.sparse_controller
            self.sparse_controller.set_modules(self.model.model.layers)

        # 加载 DeltaKV 压缩器权重（如果使用 DeltaKV 方法）
        self.load_deltakv_compressors()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # === 第6步：多进程协调（仅在 TP > 1 时启用） ===
        # TP 场景下的 RPC 通信：Rank 0 通过共享内存向其他 Rank 发送方法调用指令
        # 这样可以让所有进程在收到相同的方法调用时，都执行本地逻辑并保持同步
        if self.world_size > 1:
            if rank == 0:
                # ===== Rank 0 的初始化流程 =====
                # Rank 0 创建共享内存对象
                # create=True: 如果内存不存在则创建，存在则抛错。这确保只有一个创建者
                # name="sparsevllm": 共享内存的唯一标识符，其他 Rank 用这个名字连接
                # size=2**20: 1MB 大小，足够存放序列化的方法名和参数（通常只有几 KB）
                self.shm = SharedMemory(name="sparsevllm", create=True, size=2**20)
                
                # Rank 0 创建完共享内存后，等待其他 Rank 都连接上
                # barrier() 会阻塞所有 Rank，直到所有 Rank 都到达这一点
                # 这确保 Rank 0 不会在其他 Rank 开始监听前就发送指令
                dist.barrier()
            else:
                # ===== 其他 Rank (Rank > 0) 的初始化流程 =====
                # 其他 Rank 先在 barrier() 处等待 Rank 0 创建好共享内存
                # 当 Rank 0 创建完成并调用第二个 barrier() 时，这些 Rank 会被唤醒
                dist.barrier()
                
                # 连接 Rank 0 创建的共享内存
                # create=False: 连接已存在的内存，如果不存在则抛错
                # name 必须与 Rank 0 的相同，这样才能连接到同一块内存
                self.shm = SharedMemory(name="sparsevllm")
                
                # 进入子进程的主监听循环
                # loop() 会阻塞在这里，持续监听共享内存中的新指令
                # 每当 Rank 0 写入新指令并设置 event 时，此进程就会读取并执行
                # 直到收到 "exit" 指令才会跳出循环并返回（进程退出）
                self.loop()

    def exit(self):
        """
        釋放資源並正確清理多進程環境。
        包括：共享內存的關閉/取消鏈接，與分布式進程組的注銷。
        """
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()  # 所有進程同步
            if self.rank == 0:
                self.shm.unlink()  # 只有 Rank 0 取消鏈接共享內存資源
        torch.cuda.synchronize()  # 等待 GPU 執行完所有操作
        dist.destroy_process_group()  # 銷毀分布式進程組

    def loop(self):
        """
        子進程 (Rank > 0) 的主循環。
        不斷從共享內存讀取來自 Rank 0 的方法調用指令，執行後繼續等待下一個指令。
        當接收到 "exit" 指令時才會退出。
        """
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """
        从共享内存读取并反序列化来自 Rank 0 的方法指令。
        
        流程：
        1. 等待 event 被设置（表示 Rank 0 已写入新指令）
        2. 从共享内存前 4 字节读取数据长度
        3. 反序列化方法名和参数
        4. 清空 event 标志，等待下次指令
        """
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()  # 等待 Rank 0 的信号
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()  # 重置事件标志
        return method_name, args

    def write_shm(self, method_name, *args):
        """
        序列化方法名和参数，并写入共享内存供其他 Rank 读取。
        
        流程：
        1. 序列化 [method_name, *args]
        2. 将数据长度写入前 4 字节
        3. 将序列化数据写入后续字节
        4. 设置所有 event 标志，通知各 Rank 有新指令
        """
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:  # 逐个通知每个子进程
            event.set()

    def call(self, method_name, *args):
        """
        RPC 风格的方法调用。在 TP 场景下实现所有 Rank 的同步执行。
        
        关键点：
        1. 如果是 Rank 0，先通过共享内存广播指令给其他 Rank
        2. 所有 Rank 都在 inference_mode 下执行本地方法逻辑
        3. inference_mode 禁用自动求导，避免激活值图过大导致 OOM
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)  # 广播指令给其他 Rank
        method = getattr(self, method_name, None)
        # 在 inference_mode 下运行，避免积累激活值图（特别是 DeltaKV 驱逐等后处理操作）
        with torch.inference_mode():
            return method(*args)

    def load_deltakv_compressors(self):
        """加载 DeltaKV 压缩器权重"""
        method = str(self.config.vllm_sparse_method or "")
        if not method.startswith('deltakv') or self.config.deltakv_path is None:
            return
        
        logger.info(f"Loading DeltaKV compressors from {self.config.deltakv_path}")
        from sparsevllm.utils.loader import load_deltakv_compressors_to_cache_manager

        load_deltakv_compressors_to_cache_manager(self.cache_manager, self.config.deltakv_path)

    def free_slots(self, seq_id: int):
        """通知 CacheManager 释放该序列占用的物理显存位子"""
        with profiler.record("model_free_slots"):
            if os.getenv("SPARSEVLLM_DEBUG_SLOTS", "0") == "1":
                before = self.cache_manager.free_slot_stats()
                logger.info("model_runner.free_slots seq_id={} before={}", seq_id, before)
            self.cache_manager.free_seq(seq_id)
            if os.getenv("SPARSEVLLM_DEBUG_SLOTS", "0") == "1":
                after = self.cache_manager.free_slot_stats()
                logger.info("model_runner.free_slots seq_id={} after={}", seq_id, after)

    def _long_text_threshold(self, is_prefill: bool) -> int:
        """
        根据稀疏方法和阶段，计算区分「长文本」和「短文本」的阈值。
        
        设计理念：
        - 对于 Vanilla、SnapKV 等需要保留更多 token 的方法，短文本不触发稀疏优化
        - Prefill 阶段的 chunk_prefill_size 需要额外计入（因为分块处理）
        - DeltaKV-Standalone 方案中没有 SnapKV 的滑动窗口，所以阈值更低
        """
        if self.config.vllm_sparse_method == "deltakv-snapkv":
            base = (
                self.config.num_sink_tokens  # 开头 token（类似 StreamingLLM 的 Sink）
                + self.config.num_recent_tokens  # 最近 token
                + self.config.snapkv_window_size  # SnapKV 滑动窗口
            )
        elif self.config.vllm_sparse_method == "deltakv-standalone":
            base = self.config.num_sink_tokens + self.config.num_recent_tokens
        elif self.config.vllm_sparse_method in ("streamingllm", "attention-sink", "attention_sink"):
            base = self.config.num_sink_tokens + self.config.num_recent_tokens
        else:
            # Vanilla 或其他方法：需要保留 top-k tokens
            base = (
                self.config.num_sink_tokens
                + self.config.num_recent_tokens
                + self.config.num_top_tokens
            )
        return base + (self.config.chunk_prefill_size if is_prefill else 0)

    def _is_long_text_batch(self, seqs: list[Sequence], is_prefill: bool) -> bool:
        """
        判断当前 batch 是否属于「长文本」。
        
        重要说明：
        - 这是一个批级别的标志，用于 gate 稀疏逻辑（整个 batch 要么都触发，要么都不触发）
        - 短文本可能在 decode 过程中变成长文本，所以动态计算而不是缓存
        - 如果 batch 中存在长短混合，会抛错（调度器应该已保证分离）
        """
        threshold = self._long_text_threshold(is_prefill)
        if not seqs:
            return False
        if is_prefill:
            flags = [int(seq.num_prompt_tokens) > int(threshold) for seq in seqs]
        else:
            flags = [int(seq.num_tokens) > int(threshold) for seq in seqs]
        is_long = bool(flags[0])
        if any(bool(flag) != is_long for flag in flags):
            raise ValueError("Mixed long/short batch detected; scheduler should enforce separation.")
        return is_long

    def prepare_step(self, seqs: list[Sequence], is_prefill: bool):
        """
        为模型前向计算准备输入数据和执行上下文。
        
        任务：
        1. 从 CacheManager 获取 token ID、位置和序列长度累积和
        2. 设置全局上下文，包含 KV Cache 管理器和稀疏化标志
        3. 返回准备好的张量供 run_model 使用
        """
        input_ids, positions, cu_seqlens_q = self.cache_manager.prepare_step(seqs, is_prefill)
        set_context(
            is_prefill,
            cu_seqlens_q=cu_seqlens_q,
            cache_manager=self.cache_manager,
            is_long_text=self._is_long_text_batch(seqs, is_prefill),
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """
        为采样阶段准备温度参数张量。
        这些参数控制 softmax 的「热度」，影响 token 采样的随机性。
        """
        temperatures = [seq.temperature for seq in seqs]
        return torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        执行模型前向计算。
        在 inference_mode 下禁用自动求导，直接计算 logits。
        """
        _stage = 'prefill' if is_prefill else 'decode'
        with profiler.record(f"model_run_model_{_stage}"):
            return self.model.compute_logits(self.model(input_ids, positions))

    def run(self, seqs: list[Sequence], is_prefill: bool) -> tuple[list[int], torch.Tensor | None]:
        """
        单步执行主逻辑。包含数据准备、前向计算、采样和后处理等 5 个阶段。
        
        返回值：
        - token_ids: 采样得到的 token ID 列表（仅 Rank 0 有效；其他 Rank 返回 None）
        - attn_score: 暂未使用（为后续扩展预留）
        """
        name = "model_run_prefill" if is_prefill else "model_run_decode"
        with profiler.record(name):
            # 第1步：准备前向上下文（获取 input_ids、positions 等）
            ctx = get_context()
            input_ids, positions = self.prepare_step(seqs, is_prefill)
            
            # 第2步：准备稀疏化状态
            # 根据文本长度和稀疏方法，动态决定如何处理 token（保留或压缩）
            with profiler.record("model_sparse_prepare"):
                ctx.sparse_controller = self.sparse_controller
                self.sparse_controller.prepare_forward(seqs, is_prefill)
            
            # 第3步：准备采样参数（仅 Rank 0 需要）
            temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
            
            # 第4步：模型前向计算
            # 所有 Rank 都参与计算，但数据分片分布在各自 GPU 上
            logits = self.run_model(input_ids, positions, is_prefill)
            
            # 第5步：Token 采样
            # 只在 Rank 0 执行，其他 Rank 返回 None
            with profiler.record("model_sampler"):
                token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

            # 第6步：后置稀疏处理
            # 例如 SnapKV 的 KV 驱逐、DeltaKV 的压缩等
            with profiler.record("model_sparse_post"):
                self.sparse_controller.post_forward(seqs, is_prefill)

            reset_context()  # 清理全局上下文
            return token_ids, None  # attn_score 暂未使用
