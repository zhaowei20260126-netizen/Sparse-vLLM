import os
import time
import torch
import argparse
import multiprocessing as mp
import traceback
import json
from pathlib import Path
from typing import Any
from time import perf_counter


def get_peak_memory():
    return torch.cuda.max_memory_allocated() / (1024 ** 3) # GB


def _load_json_arg(value: str) -> dict[str, Any]:
    """Load a JSON object from a CLI arg.

    Supports:
      - Inline JSON: '{"gpu_memory_utilization": 0.9}'
    """
    if value is None:
        return {}
    value = str(value).strip()

    try:
        parsed = json.loads(value)
    except Exception as e:
        raise ValueError(f"Invalid JSON for --hyper_params: {e}") from e

    if not isinstance(parsed, dict):
        raise ValueError("--hyper_params must be a JSON object (dict).")
    return parsed


def _build_engine_hyper_params(args) -> dict[str, Any]:
    # Keep benchmark defaults stable (do not rely on sparsevllm.Config defaults).
    hyper_params: dict[str, Any] = {
        "enforce_eager": True, # 强制走 eager 执行，不走 torch.compile
        "gpu_memory_utilization": 0.8, # 
        "chunk_prefill_size": 4096, # 预填充阶段按多少 token 分块处理
        "tensor_parallel_size": 1, # 预填充阶段按多少 token 分块处理
    }

    hyper_params.update(_load_json_arg(args.hyper_params))

    return hyper_params


def benchmark_task(method, length, bs, args, results_dict):
    """
    为单个 (method, context_length, batch_size) 组合执行性能基准测试。
    
    参数：
        method (str): 稀疏方法名称 (vanilla, snapkv, omnikv, deltakv 等)
        length (int): 提示词 token 长度（上下文长度）
        bs (int): 批处理大小（同时推理的序列数）
        args: 命令行参数对象（包含模型路径、超参数等）
        results_dict: 多进程共享的字典，用于存储测试结果
    
    测试流程：
        1. 初始化 GPU 显存
        2. 创建 LLM 引擎（带指定的稀疏方法）
        3. 准备虚拟提示词和采样参数
        4. 执行推理循环并精确测量时间
        5. 收集吞吐量、延迟、显存等性能指标
        6. 将结果存储到共享字典
    """
    # ==================== 第1步：GPU 初始化 ====================
    # 为每个子进程重置显存统计（确保每个测试都从干净状态开始）
    torch.cuda.reset_peak_memory_stats() 
    torch.cuda.empty_cache()
    
    print(f"\n>>> Starting: {method.upper()} | Context: {length} | Batch: {bs}...")
    
    # ==================== 第2步：确定稀疏方法 ====================
    base_hyper_params = args.hyper_params_dict
    sparse_kwargs: dict[str, Any] = {"vllm_sparse_method": ""}  # 默认为 vanilla（全注意力）
    
    # 根据 method 参数设置稀疏方法
    # vanilla 对应空字符串，其他方法名和设置值一致
    if method == "vanilla":
        sparse_kwargs["vllm_sparse_method"] = ""  # 空字符串表示不使用稀疏
    elif method in ("streamingllm", "attention-sink", "attention_sink", "snapkv", "pyramidkv", "omnikv", "quest", "deltakv"):
        sparse_kwargs["vllm_sparse_method"] = method  # 直接使用方法名
    elif "deltakv" in method:
        # 支持 deltakv-triton, deltakv-triton-v2, deltakv-triton-v3 等变体
        sparse_kwargs["vllm_sparse_method"] = method
    
    llm = None
    try:
        # ==================== 第3步：创建 LLM 引擎 ====================
        # 计算模型支持的最大序列长度
        # 公式：实际输入长度 + 生成输出长度 + 安全缓冲（100 token）
        # 缓冲用于防止方法特定的边界处理导致 OOM
        m_len = length + args.output_len + 100
        
        # 准备引擎的超参数
        # 注意：max_model_len 会被计算的 m_len 覆盖，以保证基准测试的一致性
        hyper_params = dict(base_hyper_params)  # 复制基础超参数
        hyper_params.pop("max_model_len", None)  # 移除以避免被覆盖
        
        # 设置 batch 相关的参数
        # max_num_seqs_in_batch: prefill 阶段最多处理的序列数
        # max_decoding_seqs: decode 阶段最多处理的序列数
        hyper_params.setdefault("max_num_seqs_in_batch", int(bs))
        hyper_params.setdefault("max_decoding_seqs", int(bs))
        
        # 提取分块 prefill 大小（默认 4096）
        # 用于长提示词的分块处理，降低峰值显存
        chunk_prefill_size = int(hyper_params.get("chunk_prefill_size", 4096))
        
        from sparsevllm import LLM, SamplingParams
        
        # 合并所有引擎配置参数
        engine_kwargs = {
            **hyper_params,      # 基础超参数（显存占用、chunk 大小等）
            "max_model_len": m_len,  # 最大序列长度
            **sparse_kwargs,     # 稀疏方法选择
        }
        llm = LLM(args.model_path, **engine_kwargs)  # 创建 LLM 推理引擎

        # ==================== 第4步：准备测试数据 ====================
        # 生成虚拟提示词：bs 个序列，每个 length 个 token（都是 100）
        # 注意：测试中使用固定 token ID 是为了复现性，而不是真实的单词
        prompt_token_ids = [[100] * length for _ in range(bs)]
        
        # 为每个序列创建采样参数
        # temperature: 采样的随机性（0=greedy，小值接近greedy，大值更随机）
        # ignore_eos: 忽略结束符，继续生成直到 max_tokens
        # max_tokens: 每个序列最多生成多少个 token
        sampling_params = [
            SamplingParams(
                temperature=float(args.temperature),
                ignore_eos=True,  # 为基准测试强制生成固定长度
                max_tokens=args.output_len,
            )
            for _ in range(bs)
        ]
        # ==================== 第5步：准备分波次准入策略（可选） ====================
        # 分波次准入（Staged Admission）：将 bs 个序列分多波加入，以测试调度器性能
        # 一般的基准测试中不用这个，但支持更细粒度的测试
        admission_wave_size = int(getattr(args, "admission_wave_size", 0) or 0)
        staged_admission = 0 < admission_wave_size < bs  # 是否启用分波次模式
        wave_decode_gap_steps = int(getattr(args, "wave_decode_gap_steps", 0) or 0)  # 两波之间间隔多少 decode 步

        # ==================== 第6步：重置计时和性能统计 ====================
        # 重置 profiler（清空之前的统计数据）
        from sparsevllm.utils.profiler import profiler
        profiler.reset()
        
        # GPU 同步：确保所有之前的 GPU 操作完成再开始计时
        # 这是准确测量的必要步骤
        torch.cuda.synchronize()

        # ==================== 第7步：初始化性能统计变量 ====================
        # === Prefill 阶段统计 ===
        prefill_tokens = 0            # 处理的 prefill token 总数
        prefill_times = []            # 每个 prefill step 的耗时列表
        
        # === Decode 阶段统计 ===
        decode_tokens = 0             # 生成的 token 总数
        decode_times = []             # 每个 decode step 的耗时列表
        ttft = None                   # Time To First Token（首字延迟）
        decode_started = False        # 是否已开始 decode
        
        # === 分波次准入相关的统计 ===
        # 在分波次模式下，只计算"全进"后的 decode 指标，以排除波次准入的影响
        decode_tokens_after_full = 0        # "全进"后生成的 token 数
        decode_times_after_full = []        # "全进"后每个 step 的耗时
        decode_bs_after_full = []           # "全进"后每个 step 的活跃 batch size
        full_admission_reached = not staged_admission  # 是否已全部准入
        impossible_full_admission = False              # 是否因显存不足无法全进
        decode_steps_after_full = 0         # "全进"后的 decode step 数
        
        # === 计时 ===
        t_start = perf_counter()      # 记录测试开始时间

        # ==================== 第8步：准备推理循环 ====================
        # 手动运行生成循环以收集详细的步级统计信息
        # 不能用 llm.generate()，因为无法获得每一步的时间细节
        
        next_request_idx = 0           # 下一个待准入的请求索引
        decode_steps_since_last_wave = 0  # 上一波准入后经过的 decode step 数

        def add_wave(max_new_requests: int):
            """将最多 max_new_requests 个请求加入引擎（用于分波次准入）"""
            nonlocal next_request_idx, decode_steps_since_last_wave
            end_idx = min(bs, next_request_idx + max_new_requests)
            # 逐个提交请求到引擎
            for req_idx in range(next_request_idx, end_idx):
                llm.add_request(prompt_token_ids[req_idx], sampling_params[req_idx])
            added = end_idx - next_request_idx
            next_request_idx = end_idx
            decode_steps_since_last_wave = 0  # 重置 wave 后的 step 计数
            return added

        # 添加第一波请求（如果分波次则只添加 admission_wave_size 个，否则全部）
        add_wave(admission_wave_size if staged_admission else bs)

        # ==================== 第9步：主推理循环 ====================
        has_queued = False     # 是否出现过 prefill 和 decode 混合的情况
        zero_steps = 0         # 连续返回 0 tokens 的次数（用于检测死锁）
        
        while not llm.is_finished():  # 直到所有序列完成
            # === 分波次准入：在合适的时间添加下一波请求 ===
            if (
                staged_admission
                and next_request_idx < bs                           # 还有未准入的请求
                and len(llm.scheduler.waiting) == 0                 # 等待队列已空（上波已全部开始处理或完成）
                and len(llm.scheduler.decoding) > 0                 # 当前还有 decode 任务
                and decode_steps_since_last_wave >= wave_decode_gap_steps  # 间隔足够
            ):
                # 添加下一波请求
                add_wave(admission_wave_size)

            # === 执行一个推理步骤（prefill 或 decode） ===
            step_start = perf_counter()
            finished_outputs, num_tokens = llm.step()  # 执行单个 step
            step_dt = perf_counter() - step_start      # 记录耗时
            
            # === 根据 num_tokens 的符号判断是 prefill 还是 decode ===
            # Sparse-vLLM 的设计：
            #   - prefill step: num_tokens > 0（处理了多个提示词 token）
            #   - decode step: num_tokens < 0（生成了新的 token，负数表示序列数）
            #   - idle step: num_tokens == 0（无进展，可能是调度问题）
            
            if num_tokens > 0:  # ===== Prefill 阶段 =====
                prefill_tokens += num_tokens  # 累计处理的 token 数
                prefill_times.append(step_dt) # 记录此步耗时
                
                # 如果 decode 已开始但又有 prefill，说明新请求插入了
                if decode_started:
                    has_queued = True
                
                zero_steps = 0  # 重置死锁检测计数
                
                # === 计算 TTFT（首字延迟） ===
                # Sparse-vLLM 设计：第一个生成 token 在 prefill 的最后一步产生
                # 所以 TTFT 应该在 prefill 阶段有序列进入 decode 时记录
                if ttft is None and (llm.scheduler.decoding or any(tids for _, tids in finished_outputs)):
                    ttft = perf_counter() - t_start
            elif num_tokens < 0:  # ===== Decode 阶段 =====
                # num_tokens 是负数，负值等于当前在 decode 的序列数
                decode_started = True
                decode_steps_since_last_wave += 1  # 用于分波决策
                decode_times.append(step_dt)       # 记录此 decode step 的耗时
                decode_tokens += (-num_tokens)     # 累计生成的 token 数（转正）
                
                # === 如果已全部准入，则收集用于算吞吐的专用统计 ===
                # 分波次模式下只计算"全进"后的 decode 吞吐
                if full_admission_reached:
                    decode_times_after_full.append(step_dt)
                    decode_tokens_after_full += (-num_tokens)
                    decode_bs_after_full.append(len(llm.scheduler.decoding))  # 当前活跃序列数
                    decode_steps_after_full += 1
                
                zero_steps = 0  # 重置死锁检测计数
                
                # === 备选方案：如果之前没有捕获 TTFT，在第一个 decode step 时设置 ===
                # （正常情况下 TTFT 应该在 prefill 最后一步捕获）
                if ttft is None:
                    ttft = perf_counter() - t_start
            else:  # ===== Idle / 无进展 =====
                # num_tokens == 0 可能表示调度器卡住或等待
                zero_steps += 1  # 记录连续无进展的次数
                
                # 如果连续 50 步无进展，说明有问题（死锁）
                if zero_steps >= 50:
                    raise RuntimeError("llm.step() returned 0 tokens repeatedly; scheduler may be stuck.")

            # === 检测"全进"状态（分波次模式） ===
            if staged_admission and not full_admission_reached and next_request_idx == bs:
                # 所有请求都已准入
                if len(llm.scheduler.waiting) == 0 and len(llm.scheduler.decoding) == bs:
                    # 并且全部都已进入 decode
                    full_admission_reached = True
                elif finished_outputs:
                    # 或者有序列提前完成（无法全进）
                    impossible_full_admission = True
                    break

            # === 提前停止条件（用于调试） ===
            # 如果指定了 max_decode_steps_after_full，在"全进"后达到该步数时停止
            max_decode_steps_after_full = int(getattr(args, "max_decode_steps_after_full", 0) or 0)
            if full_admission_reached and max_decode_steps_after_full > 0 and decode_steps_after_full >= max_decode_steps_after_full:
                break

        # ==================== 第10步：结束和同步 ====================
        print(f'@@@ {decode_tokens=}')
        
        # GPU 同步：确保所有 GPU 操作完成
        torch.cuda.synchronize()
        t_end = perf_counter()  # 记录测试结束时间
        
        # === 收集基础性能指标 ===
        duration = t_end - t_start        # 总耗时
        peak_mem = get_peak_memory()      # 峰值显存（GB）

        # === 聚合时间数据 ===
        ttft = float(ttft or 0.0)         # 首字延迟（秒）
        prefill_s = sum(prefill_times)    # 所有 prefill step 的总耗时
        decode_s = sum(decode_times)      # 所有 decode step 的总耗时

        # === 调试输出 ===
        print(f'[debug] {prefill_tokens=} {prefill_s=} {ttft=} {decode_tokens=} {decode_s=} {has_queued=}')
        
        # ==================== 第11步：计算性能吞吐指标 ====================
        # === Prefill 吞吐量 ===
        # 单位：token/sec（每秒处理多少个 prefill token）
        prefill_tp = prefill_tokens / prefill_s if prefill_s > 0 else 0
        
        # === Decode 吞吐量的选择与计算 ===
        # 分波次模式：只计算"全进"后的 decode 吞吐（排除准入时间的影响）
        # 非分波次模式：使用全部 decode 数据
        used_full_admission_window = bool(decode_times_after_full)
        decode_s_effective = sum(decode_times_after_full) if used_full_admission_window else decode_s
        decode_tokens_effective = decode_tokens_after_full if used_full_admission_window else decode_tokens
        
        # Decode 吞吐量：生成的 token 总数 / 总耗时
        # 单位：token/sec
        decode_tp = decode_tokens_effective / decode_s_effective if decode_s_effective > 0 else 0
        
        # === ITL（Inter-Token Latency，词间延迟） ===
        # 用户感知的生成速度：平均每个序列每生成一个 token 需要多长时间
        # 公式：总 decode 时间 / (生成的 token 总数 / batch_size) * 1000 (转毫秒)
        # 意义：如果 batch_size=2，生成 100 个 token 用时 10s，则 ITL = 10s / (100/2) * 1000 = 100ms
        avg_itl = (decode_s_effective / (decode_tokens_effective / bs) * 1000) if decode_tokens_effective > 0 else 0
        
        # === 平均活跃 Batch Size ===
        # 在分波次模式下：计算"全进"后各 decode step 的活跃序列数平均值
        # 非分波次模式：简单用 decode token 数 / decode step 数（近似）
        avg_active_bs = (
            sum(decode_bs_after_full) / len(decode_bs_after_full)
            if decode_bs_after_full
            else (decode_tokens / len(decode_times) if decode_times else 0)
        )
        
        # ==================== 第12步：输出结果 ====================
        # 分波次模式的额外信息
        stage_mode = (
            f" | AdmissionWave: {admission_wave_size}"
            f" | WaveGapSteps: {wave_decode_gap_steps}"
            f" | FullAdmit: {'yes' if full_admission_reached else 'no'}"
            f" | DecodeScope: {'full' if used_full_admission_window else 'fallback'}"
            if staged_admission
            else ""
        )
        
        # 打印简要结果
        # TTFT: 首字延迟（秒）
        # Prefill: 预填充吞吐（token/sec）
        # Decode: 生成吞吐（token/sec）
        # ITL: 词间延迟（毫秒）
        # AvgBS: 平均活跃 batch size
        # Mem: 峰值显存（GB）
        print(f"[{method.upper()}] TTFT: {ttft:.2f}s | Prefill: {prefill_tp:.2f} tok/s | Decode: {decode_tp:.2f} tok/s | ITL: {avg_itl:.2f}ms | AvgBS: {avg_active_bs:.1f} | Mem: {peak_mem:.2f} GB{stage_mode}")
        
        # ==================== 第13步：存储结果 ====================
        # 将所有计算的性能指标存储到共享字典中
        # 键为 (method, length, bs) 三元组，值为结果字典
        results_dict[(method, length, bs)] = {
            "prefill_tp": prefill_tp,                          # 预填充吞吐（token/sec）
            "decode_tp": decode_tp,                            # 生成吞吐（token/sec）
            "ttft": ttft,                                      # 首字延迟（秒）
            "itl": avg_itl,                                    # 词间延迟（毫秒）
            "avg_bs": avg_active_bs,                           # 平均活跃 batch size
            "mem": peak_mem,                                   # 峰值显存（GB）
            "has_queued": has_queued,                          # 是否出现混合 prefill/decode
            "full_admission_reached": full_admission_reached,  # 分波次模式是否全进
            "impossible_full_admission": impossible_full_admission,  # 是否无法全进
            "decode_scope": "full" if used_full_admission_window else "fallback",  # 使用的 decode 数据范围
            "staged_admission": staged_admission,              # 是否使用分波次模式
            "admission_wave_size": admission_wave_size if staged_admission else None,
            "status": "SUCCESS"
        }

    except Exception as e:
        # === 错误处理 ===
        print(f"Error at {method}/{length}/{bs}: {e}")
        traceback.print_exc()
        results_dict[(method, length, bs)] = {"status": "FAILED"}
    
    finally:
        # === 清理资源 ===
        # 必须调用 llm.exit() 来释放 GPU 显存和终止子进程
        if llm is not None and hasattr(llm, "exit"):
            llm.exit()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description="Professional benchmark for sparsevllm.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--lengths", type=str, default="16000,32000,64000", help="Context lengths to test")
    parser.add_argument("--batch_sizes", type=str, default="4", help="Batch sizes to test")
    parser.add_argument(
        "--methods",
        type=str,
        default="vanilla,snapkv,omnikv",
        help="Methods to test (vanilla, streamingllm, attention-sink, snapkv, pyramidkv, omnikv, quest, deltakv, deltakv-triton, deltakv-triton-v2, deltakv-triton-v3, deltakv-triton-v3-offload, deltakv-triton-v3-cuda-offload)",
    )
    parser.add_argument("--output_len", type=int, default=512, help="Output tokens per request")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for generation. Default 0.0 (greedy) for throughput benchmarking.",
    )
    parser.add_argument(
        "--admission_wave_size",
        type=int,
        default=0,
        help="If >0 and < batch size, only admit this many sequences at a time. Decode throughput is then measured after the final wave has fully entered decode.",
    )
    parser.add_argument(
        "--max_decode_steps_after_full",
        type=int,
        default=0,
        help="If >0 in staged mode, stop after this many decode steps after full admission is reached.",
    )
    parser.add_argument(
        "--wave_decode_gap_steps",
        type=int,
        default=0,
        help="In staged admission mode, require this many decode steps before admitting the next wave.",
    )
    parser.add_argument(
        "--hyper_params",
        type=str,
        default="{}",
        help=(
            "LLMEngine/Config hyper-params as JSON (string or @file.json). "
            'Example: \'{"gpu_memory_utilization":0.9,"c":4096,"tensor_parallel_size":1,"num_top_tokens":2048}\''
        ),
    )
    # Deprecated (kept for backward compatibility; prefer --hyper_params)
    parser.add_argument("--gpu_util", type=float, default=None, help="[DEPRECATED] use --hyper_params.gpu_memory_utilization")
    parser.add_argument("--chunk_size", type=int, default=None, help="[DEPRECATED] use --hyper_params.chunk_prefill_size")
    parser.add_argument("--tp", type=int, default=None, help="[DEPRECATED] use --hyper_params.tensor_parallel_size")
    parser.add_argument("--enforce_eager", type=str2bool, default=None, help="[DEPRECATED] use --hyper_params.enforce_eager")
    
    args = parser.parse_args()
    try:
        args.hyper_params_dict = _build_engine_hyper_params(args)
    except ValueError as e:
        parser.error(str(e))
    
    test_lengths = [int(x) for x in args.lengths.split(",")]
    test_methods = args.methods.split(",")
    test_batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    manager = mp.Manager() # 启动一个专门的管理进程，用来协调多个子进程之间共享数据
    results_dict = manager.dict() # 这个字典可以被多个子进程安全地读写，子进程将测试结果写入这个字典，主进程在所有子进程完成后读取并汇总结果。

    for method in test_methods:
        for length in test_lengths:
            for bs in test_batch_sizes:
                p = mp.Process(target=benchmark_task, args=(method, length, bs, args, results_dict)) # 创建一个子进程对象，指定要执行的函数和参数
                p.start() # 真正启动子进程
                p.join() # 主进程在这里阻塞，等 p 执行完再继续执行其他 method、input_len、batch_size的任务

    # 打印最终报表
    print(f"\n\n{'='*140}")
    print(f"{ 'Method':<12} {'Len':<8} {'BS':<4} {'TTFT(s)':<10} {'PreTP':<12} {'DecTP':<12} {'ITL(ms)':<10} {'AvgBS':<8} {'Mem(GB)':<10} {'Speedup'}")
    print("-" * 140)
    
    # 获取 Vanilla 作为基准计算加速比 (按 length 和 BS 匹配)
    vanilla_stats = {}
    for length in test_lengths:
        for bs in test_batch_sizes:
            v_res = results_dict.get(("vanilla", length, bs))
            if v_res and v_res["status"] == "SUCCESS":
                vanilla_stats[(length, bs)] = v_res["decode_tp"]

    for method in test_methods:
        for length in test_lengths:
            for bs in test_batch_sizes:
                res = results_dict.get((method, length, bs))
                if not res or res["status"] in ["FAILED", "OOM"]:
                    status_str = res["status"] if res else "UNKNOWN"
                    print(f"{method:<12} {length:<8} {bs:<4} {status_str:<10} {'-':<12} {'-':<12} {'-':<10} {'-':<8} {'-':<10} {'-'}")
                    continue
                
                ttft = res["ttft"] # Time To First Token，首字延迟。衡量处理长 Prompt 的速度
                pre_tp = res["prefill_tp"] # 预填充吞吐量（prefill_tokens / prefill_s）
                dec_tp = res["decode_tp"]  #  解码吞吐量，反映了每秒能生成多少个 Token。
                itl = res["itl"] #  (Inter-Token Latency)，词间延迟。公式为 (decode_s / (decode_tokens / bs) * 1000)
                avg_bs = res["avg_bs"] #平均每个 decode step 里有多少条序列同时在解码，反应了并行度
                mem = res["mem"] # 显存峰值。用于验证显存占用是否缩减
                has_queued = res.get("has_queued", False) # 用来表示这次 benchmark 过程中是否出现过“decode 已经开始了，但后面还有 prefill / 新波次在继续进入”的情况。
                
                bs_str = f"{bs}*" if has_queued else f"{bs}"
                
                speedup = 1.0
                if (length, bs) in vanilla_stats:
                    speedup = dec_tp / vanilla_stats[(length, bs)]
                
                speedup_str = f"{speedup:.2f}x"
                print(f"{method:<12} {length:<8} {bs_str:<4} {ttft:<10.2f} {pre_tp:<12.1f} {dec_tp:<12.1f} {itl:<10.2f} {avg_bs:<8.1f} {mem:<10.2f} {speedup_str}")
    print(f"{ '='*140}\n")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) # 使用 spawn 确保了每个子测试任务都是在一个干净的 GPU 环境下冷启动，避免了显存泄漏或上一个测试的 CUDA 状态干扰下一次测评。
    main()
