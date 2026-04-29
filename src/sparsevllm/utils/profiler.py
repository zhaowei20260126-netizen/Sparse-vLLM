"""
轻量级性能分析器，用于测量 Sparse-vLLM 推理管线中各阶段的耗时。
通过 with profiler.record("name"): 语法包裹代码块即可记录执行时长，无需侵入业务逻辑。
"""

import time
import os
from collections import defaultdict
from contextlib import contextmanager

import torch
from sparsevllm.utils.log import logger

class Profiler:
    """全局单例性能分析器，按名称记录代码块的累计耗时和调用次数。"""

    def __init__(self):
        # defaultdict 确保首次记录时不需要判断 key 是否存在
        self.times = defaultdict(float)   # name -> 累计秒数
        self.counts = defaultdict(int)    # name -> 调用次数
        self.enabled = False              # 默认关闭，避免生产环境引入测量开销
        self.rank = 0                     # GPU rank，多卡场景区分输出来源
        # CUDA 操作是异步的：不 synchronize 则 perf_counter 只测量了 kernel launch 时间，而非实际 GPU 执行时间。
        # 开启后每次 record 前后均同步 GPU 流，使测量值包含 GPU 真实耗时（但会引入同步开销）。
        self.cuda_sync = os.environ.get("CUDA_SYNC_SVLLM", "0") == "1"

    def set_enabled(self, enabled: bool):
        """动态开关 profiler，关闭时 record() 为空操作。"""
        self.enabled = enabled

    def set_rank(self, rank: int):
        """设置当前进程的 GPU rank，用于多卡报告输出标识。"""
        self.rank = rank

    @contextmanager
    def record(self, name):
        """上下文管理器：进入时计时开始，退出时累加耗时。
        用法: with profiler.record("attn_forward"):
                  ...
        """
        # 未启用时直接透传，零开销
        if not self.enabled:
            yield
            return

        # 同步 GPU 流，确保之前提交的 kernel 都执行完成后再计时
        if self.cuda_sync:
            torch.cuda.synchronize()
        t1 = time.perf_counter()  # perf_counter 精度最高，不受系统时间调整影响
        yield
        # 再次同步 GPU 流，确保被测代码块的 kernel 也执行完成
        if self.cuda_sync:
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        # 累加时间和调用次数，支持同一 name 被多次 record 的统计
        self.times[name] += (t2 - t1)
        self.counts[name] += 1

    def reset(self):
        """清空所有累计的耗时和计数。"""
        self.times.clear()
        self.counts.clear()

    def print_stats(self):
        """打印性能报告：各代码块的总耗时、调用次数、平均耗时和占比。
        仅在 enabled=True 且有统计数据时输出。
        """
        if not self.enabled or not self.times:
            return

        logger.info(f"\n=== Sparse-vLLM Profiler Report (Rank {self.rank}) ===")
        # 按总耗时降序排列，最耗时的在前
        sorted_keys = sorted(self.times.keys(), key=lambda x: self.times[x], reverse=True)

        # 以 "step" 的耗时作为总量基准（如果没有记录 step，则用所有项之和）
        total_time = self.times.get("step", sum(self.times.values()))
        if total_time == 0:
            total_time = 1e-9  # 避免除零

        # 固定列宽表格输出
        print(f"{'Category':<30} {'Calls':<10} {'Avg (ms)':<15} {'Total (s)':<15} {'Percentage':<10}")
        print("-" * 80)
        for key in sorted_keys:
            t = self.times[key]
            c = self.counts[key]
            avg = (t / c) * 1000 if c > 0 else 0        # 毫秒/次
            pct = (t / total_time) * 100                 # 占 step 总耗时的百分比
            print(f"{key:<30} {c:<10} {avg:<15.4f} {t:<15.4f} {pct:<10.2f}%")
        print("-" * 80)

# 全局单例，模块被 import 即创建，整个进程内共享同一实例
profiler = Profiler()
