from __future__ import annotations

import os
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import torch

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence
from sparsevllm.utils.context import get_context
from sparsevllm.utils.log import logger
from sparsevllm.utils.profiler import profiler

from .attnpredict import AttnPredictCacheManager
from .base import CacheManager, LayerBatchStates


class AttnPredictOffloadCacheManager(AttnPredictCacheManager):
    """AttentionPredictor 的 offload 版本。

    核心思路：
    1. GPU 只保存“当前要参与 attention 的 active KV”，也就是一个可复用的 active slot 池。
    2. CPU 保存完整历史 KV，作为 full backing。某个历史 token 被 GPU 驱逐后，
       之后 predictor 又选中了它，就可以从 CPU 拿回来。
    3. decode 时每层 attention 结束后，后台线程在独立 CUDA stream 上预测下一步 mask，
       并预取下一步需要的 KV；下一步走到同一层时再等待该层预取完成。

    注意：这里刻意不调用 ``StandardCacheManager.__init__``。标准 cache manager 假设
    “每个 token 在 GPU 上长期占一个 slot”，而 offload 版本需要 per-layer active pool
    和 CPU full backing 两套元数据，所以只复用基类的模型尺寸/配置初始化。
    """

    def __init__(self, config: Config, rank: int, world_size: int):
        """初始化 GPU active pool、CPU full backing、predictor 和异步预取资源。"""
        CacheManager.__init__(self, config, rank, world_size)
        # CPU backing 和 per-layer active pool 元数据尚未做 TP 分片。
        assert world_size == 1, "attnpredict-offload currently supports tensor_parallel_size=1."
        # 分配 GPU active KV pool 和 CPU full KV backing
        self.allocate_kv_cache()

        # ------------------------------------------------------------------
        # GPU active pool 元数据
        # ------------------------------------------------------------------
        # 每层都有一套独立的 GPU slot 池。slot 里只放当前驻留 GPU 的 token KV，
        # 不再表示“这个序列的完整历史都在 GPU 上”。
        num_slots = config.num_kvcache_slots
        self.layer_num_slots = [num_slots for _ in range(self.num_layers)]
        self.free_slots_stack: list[list[int]] = [list(range(num_slots)) for _ in range(self.num_layers)]
        self._num_free_slots = [num_slots for _ in range(self.num_layers)]

        # GPU 侧 row -> slot 映射。prefill 用完整映射；decode attention
        # 只读 build_decode_view() 产出的 packed slots。给 GPU / attention / 通用接口用。
        self.buffer_req_to_token_slots = [
            torch.full((self.max_buffer_rows, self.max_model_len), -1, dtype=torch.int32, device="cuda")
            for _ in range(self.num_layers)
        ]

        # CPU 侧镜像，方便 Python 后台线程快速判断某个 token 是否已驻留 GPU。
        # 语义与 buffer_req_to_token_slots 相同，但存在 CPU numpy 数组里。
        self.gpu_req_to_token_slots_cpu = [
            np.full((self.max_buffer_rows, self.max_model_len), -1, dtype=np.int32)
            for _ in range(self.num_layers)
        ]
        self.layer_batch_states = [LayerBatchStates() for _ in range(self.num_layers)]

        # ------------------------------------------------------------------
        # 序列行号元数据
        # ------------------------------------------------------------------
        # seq_id_to_row 将外部序列 id 映射到 cache manager 内部的 row。
        # row_seq_lens 记录每个 row 当前完整逻辑长度，也就是 CPU full backing 已保存的 token 数。
        self.seq_id_to_row: dict[int, int] = {}
        self.free_rows = deque(range(self.max_buffer_rows))
        self.row_seq_lens = np.zeros((self.max_buffer_rows,), dtype=np.int32)

        # ------------------------------------------------------------------
        # CPU full KV backing 元数据
        # ------------------------------------------------------------------
        # CPU 侧 row -> cpu_slot 映射。只要 token 进入过 KV cache，这里就应该能找到
        # 它完整 K/V 在 CPU full backing 中的位置。
        self.cpu_req_to_token_slots = np.full(
            (self.max_buffer_rows, self.max_model_len),
            -1,
            dtype=np.int64,
        )
        self.cpu_free_slots_stack: list[int] = list(range(self.cpu_num_slots))
        self._num_free_cpu_slots = self.cpu_num_slots

        # 本轮 forward 中每个 token 对应的 CPU slot。attention.py 写完 GPU KV 后，
        # on_kv_stored() 会用它把同一批 K/V 复制到 CPU full backing。
        self._layer_cpu_slot_mapping: list[np.ndarray | None] = [None for _ in range(self.num_layers)]
        # 连续 CPU slot 用 slice copy 写入；元素为 (token_start, token_end, slot_start, slot_end)。
        self._cpu_store_segments: list[tuple[int, int, int, int]] | None = None

        # decode 当前 step 的 row 和当前位置。get_layer_store_view() 逐层消费时会用它
        # 构造“本层需要先确保驻留 GPU 的历史位置”。
        self._decode_current_positions = np.empty((0,), dtype=np.int64)
        self._decode_rows = np.empty((0,), dtype=np.int64)

        # 每层本轮 decode 实际要读的逻辑 token positions。get_layer_store_view() 先准备，
        # build_decode_view() 再把它转换成 packed GPU slots 交给 decode kernel。
        self._decode_view_positions: list[list[np.ndarray] | None] = [None for _ in range(self.num_layers)]
        # 每层当前正在复用的 hot positions；sink/recent/current 每步动态拼接。
        self._lease_hot_positions: list[dict[int, np.ndarray] | None] = [None for _ in range(self.num_layers)]
        # row -> 这份预测结果从哪个 decode 位置开始使用。
        self._lease_start_positions: list[dict[int, int]] = [dict() for _ in range(self.num_layers)]
        # 每隔多少个 decode step 重新预测。
        self._reuse_steps = config.attnpredict_reuse_steps
        # 后台预测过慢时最多复用几步旧 lease。
        self._max_stale_steps = config.attnpredict_max_stale_steps
        # 每 4 层复用组首层 predictor 结果。
        self._layer_reuse_stride = 4

        # AttentionPredictor 算法状态与普通 attnpredict 完全一致，复用父类 helper。
        # StandardCacheManager 的全局 GPU slot 结构，和 offload 的 per-layer active pool 冲突。
        self._init_attnpredictor_state(config)
        # pooled seq len 会随 decode 增长；动态图避免每个长度重新编译 CNN。
        self.cnn = torch.compile(
            self.cnn,
            dynamic=True,
            options={"triton.cudagraphs": False},
        )
        pooled_len = (self.max_model_len + self.pooling_block_size - 1) // self.pooling_block_size
        pred_len = max(
            3,
            pooled_len - self.sink_token // self.pooling_block_size - self.local_token // self.pooling_block_size,
        )
        dummy = torch.zeros(
            (self.hf_config.num_attention_heads // self.world_size, self.history_step, pred_len),
            dtype=self.cnn_dtype,
            device="cuda",
        )
        with torch.inference_mode():
            # 预热 torch.compile，避免第一次正式 decode 才触发编译。
            self.cnn(dummy)
        # 等预热相关 GPU 工作完成后再进入推理资源初始化。
        torch.cuda.synchronize()

        # ------------------------------------------------------------------
        # 异步预取资源
        # ------------------------------------------------------------------
        # 后台 CPU 线程负责提交 predictor/prefetch 任务；每层一个 CUDA stream，
        # 用于让预测和 H2D 拷贝尽量与主计算流 overlap。
        cpu_threads = config.attnpredict_offload_cpu_threads or 1
        # 控制 CPU 侧 PyTorch 算子线程数，不影响 GPU kernel 线程数。
        torch.set_num_threads(cpu_threads)
        self._pin_staging = bool(config.attnpredict_offload_pin_staging)
        # 后台线程池负责提交 predictor、prefetch 和 CPU residency 任务。
        self._prefetch_executor = ThreadPoolExecutor(max_workers=cpu_threads)
        # 单条默认优先级预测流，避免多层 predictor 并发抢占主计算流。
        self._prefetch_stream = torch.cuda.Stream(priority=0)
        self._prefetch_streams = [self._prefetch_stream for _ in range(self.num_layers)]
        # 每层一个 future 句柄；结果可能覆盖组内多个 layer。
        self._prefetch_futures: list[Future | None] = [None for _ in range(self.num_layers)]
        # 异步 decode 切到新 lease 后，各层到达自身写入点时再释放旧 resident。
        self._pending_residency_cleanup = [False for _ in range(self.num_layers)]
        self._pending_prefill_views: list[dict[str, object] | None] = [None for _ in range(self.num_layers)]
        # 主线程和后台线程都会改同层 active pool 元数据。
        self._layer_locks = [threading.RLock() for _ in range(self.num_layers)]
        # CNN 模块共享，串行化 forward 提交，避免 stream/编译缓存状态并发风险。
        self._cnn_lock = threading.RLock()
        # decode 新 KV 先留在 GPU；只有驱逐 dirty token 时才补写 CPU backing。
        self._dirty_gpu_positions: list[set[tuple[int, int]]] = [set() for _ in range(self.num_layers)]
        # GPU 驻留位置集合，避免每次从 mirror[row, :full_len] 扫完整长上下文。
        self._resident_positions: list[dict[int, set[int]]] = [dict() for _ in range(self.num_layers)]
        # row -> (lease_start, static_len, static_slots_gpu, static_positions_gpu)，recent/current 动态补。
        self._static_view_cache: list[dict[int, tuple[int, int, torch.Tensor, torch.Tensor | None]]] = [
            dict() for _ in range(self.num_layers)
        ]
        # 记录 decode view 最大长度，避免 hot path 从 GPU Tensor 上 .item()。
        self._decode_view_max_lens = [1 for _ in range(self.num_layers)]
        # 复用 packed slots/positions/local req indices，减少每层每步分配。
        self._packed_slots_cache: list[torch.Tensor | None] = [None for _ in range(self.num_layers)]
        self._packed_positions_cache: list[torch.Tensor | None] = [None for _ in range(self.num_layers)]
        self._local_req_indices_cache: torch.Tensor | None = None
        # 本轮 decode 是否需要 positions metadata；只在预测步为 source layer 构造。
        self._collect_decode_positions = [False for _ in range(self.num_layers)]
        # top-k block 展开成 token positions 时复用 GPU offsets。
        self._pooling_offsets_gpu = torch.arange(int(self.pooling_block_size), device="cuda")

        logger.info(
            "AttnPredict offload allocation: gpu_active_slots={} cpu_full_slots={} "
            "layers={} async_prefetch=always cpu_threads={} reuse_steps={} max_stale_steps={} layer_reuse_stride={}".format(
                num_slots,
                self.cpu_num_slots,
                self.num_layers,
                cpu_threads,
                self._reuse_steps,
                self._max_stale_steps,
                self._layer_reuse_stride,
            )
        )

    def allocate_kv_cache(self):
        """分配 GPU active KV pool 和 CPU full KV backing。

        GPU cache 仍然是标准形状：
            [2, num_layers, num_gpu_slots, num_kv_heads, head_dim]

        但它的语义和 vanilla 不同：这里的 GPU slot 是 active pool，
        只保存当前驻留 GPU、会被 attention kernel 读取的 token。
        CPU cache 则为每层分别保存完整 K/V，用于恢复已驱逐 token。
        """
        available_memory, slot_bytes_per_layer = self._get_available_slots_info()
        slot_bytes = self.num_layers * slot_bytes_per_layer
        capacity_slots = max(1, int(available_memory // slot_bytes))
        full_resident_target = self.max_model_len * max(1, self.config.max_num_seqs_in_batch) # 全量驻留目标容量，即“最坏情况下要在 GPU 里容纳多少个 token 的 KV slot”
        self.config.num_kvcache_slots = max(1, min(capacity_slots, full_resident_target))

        # GPU active KV pool。每层都有 config.num_kvcache_slots 个可复用 slot。
        self.kv_cache = torch.empty(
            2,
            self.num_layers,
            self.config.num_kvcache_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.torch_dtype,
            device="cuda",
        )

        # CPU full KV backing。布局和 GPU kv_cache 保持一致：
        #   cpu_kv_cache[0, layer, slot] 是 K
        #   cpu_kv_cache[1, layer, slot] 是 V
        self.cpu_num_slots = self._compute_cpu_num_slots()
        self.cpu_kv_cache = torch.empty(
            2,
            self.num_layers,
            self.cpu_num_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.torch_dtype,
            device="cpu",
        )

    def _compute_cpu_num_slots(self) -> int:
        """计算 CPU full backing 可以容纳多少 token slot。

        如果用户配置 attnpredict_offload_cpu_slots > 0，就完全尊重用户配置。
        否则自动估算：
        1. 目标值先取 max_model_len * max_num_seqs_in_batch。
        2. 再根据系统可用 CPU 内存和 attnpredict_offload_cpu_memory_utilization
           估算最多能放多少完整 KV。
        3. 取二者较小值，避免默认情况下把 CPU 内存吃满。
        """
        explicit = int(getattr(self.config, "attnpredict_offload_cpu_slots", -1) or -1)
        if explicit > 0:
            return explicit

        # 一个 token 跨所有层的完整 KV 占用：
        # num_layers * 2(K/V) * num_kv_heads * head_dim * dtype_size。
        dtype_size = torch.tensor([], dtype=self.hf_config.torch_dtype).element_size()
        bytes_per_slot_all_layers = (
            self.num_layers * 2 * self.num_kv_heads * self.head_dim * dtype_size
        )

        # CPU backing 保存完整逻辑上下文，不按 GPU active pool 容量裁剪。
        desired = self.max_model_len * max(1, self.config.max_num_seqs_in_batch)
        mem_available = self._cpu_mem_available_bytes()
        if mem_available > 0 and bytes_per_slot_all_layers > 0:
            # 只用可用内存的一部分做预算，给系统、dataloader、Python 对象等留余量。
            cpu_mem_ratio = float(self.config.attnpredict_offload_cpu_memory_utilization)
            by_mem = int((mem_available * cpu_mem_ratio) // bytes_per_slot_all_layers)
            desired = min(desired, max(1, by_mem))
        return max(1, desired)

    @staticmethod
    def _cpu_mem_available_bytes() -> int:
        """读取当前机器可用 CPU 内存，返回字节数。

        Linux 下优先读取 /proc/meminfo 的 MemAvailable。它比 MemFree 更合理，
        因为它把可回收 page cache 也算进“可用”。
        """
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages) * int(page_size)
        except (ValueError, OSError, AttributeError):
            return 0

    def get_layer_batch_states(self, layer_idx: int) -> LayerBatchStates:
        """返回某一层当前 batch 的基础元数据：写入 slot、context_lens、row indices。"""
        return self.layer_batch_states[layer_idx]

    def get_layer_kv_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回某层 GPU active pool 中的 K/V cache 张量。"""
        return self.kv_cache[0, layer_idx], self.kv_cache[1, layer_idx]

    def get_layer_buffer_req_to_token_slots(self, layer_idx: int) -> torch.Tensor:
        """返回某层 GPU 侧 row/pos -> active slot 映射表。"""
        return self.buffer_req_to_token_slots[layer_idx]

    @property
    def num_free_slots(self) -> int:
        """调度器看到的可用 slot 数。

        decode 新 token 既需要 CPU full backing slot，也可能需要 GPU active slot；
        因此这里取 GPU 剩余和 CPU 剩余的较小值，作为保守准入依据。
        """
        gpu_free = min(self._num_free_slots)
        return min(gpu_free, self._num_free_cpu_slots)

    def prompt_admission_free_slots(self) -> int:
        """prompt 准入时使用的可用容量。

        prefill token 必须同时写入 GPU active pool 和 CPU full backing，
        num_free_slots 已经取了 GPU/CPU 两侧的较小值。
        """
        return self.num_free_slots

    def free_slot_stats(self) -> dict[str, int]:
        """返回调试日志用的 GPU/CPU 剩余 slot 统计。"""
        gpu_free = min(self._num_free_slots)
        return {
            "free_slots": self.num_free_slots,
            "free_gpu_active_slots": gpu_free,
            "free_cpu_full_slots": self._num_free_cpu_slots,
        }

    def prefill_batched_tokens_margin(self) -> int:
        """给 chunk prefill 预留的 token 预算余量。

        AttentionPredictor 需要最后 history_step 个 query 来初始化历史，
        因此调度时留一点 batch token 余量，避免最后一段被切得太小。
        """
        return self.config.attnpredict_history_steps

    def remaining_prefill_tokens(self, seq: Sequence) -> int:
        """返回调度器本轮还应该继续 prefill 的 token 数。

        这里沿用 SnapKV/StreamingLLM 的“保留尾部窗口”写法：
        如果剩余 token 多于 history_step，就先只推进到尾部 history_step 之前，
        让最后一块有足够 query 用于初始化 predictor history。
        """
        remaining = seq.num_prompt_tokens - seq.num_prefilled_tokens
        history = self.config.attnpredict_history_steps
        if history > 0 and remaining > history:
            return remaining - history
        return remaining

    def _get_free_row(self, seq_id: int) -> int:
        """获取序列在元数据表里的 row；没有则分配一个新 row。"""
        if seq_id in self.seq_id_to_row:
            return self.seq_id_to_row[seq_id]
        if not self.free_rows:
            raise RuntimeError("No free rows in AttnPredict offload cache manager.")
        row_idx = self.free_rows.popleft()
        self.seq_id_to_row[seq_id] = row_idx
        return row_idx

    def _allocate_cpu_slots(self, seq_id: int, size: int) -> np.ndarray:
        """为某个序列的新 token 分配 CPU full backing slots。

        CPU slot 是跨层共享的逻辑 token slot：同一个 cpu_slot 在每一层的
        cpu_kv_cache[0/1, layer, slot] 中保存该 token 对应层的 K/V。
        """
        if self._num_free_cpu_slots < size:
            raise RuntimeError(
                f"Out of AttnPredict CPU full-KV slots: need={size}, free={self._num_free_cpu_slots}"
            )
        row_idx = self._get_free_row(seq_id)
        cur_len = self.row_seq_lens[row_idx]

        # row_seq_lens 是完整逻辑长度，不能超过 max_model_len 对应的映射表宽度。
        if cur_len + size > self.max_model_len:
            raise RuntimeError(
                f"AttnPredict offload sequence exceeds max_model_len: "
                f"seq_id={seq_id} cur_len={cur_len} add={size} max={self.max_model_len}"
            )

        # 从 free stack 末尾弹出 size 个 CPU slot，并写入 row/pos -> cpu_slot 表。
        slots = np.asarray(self.cpu_free_slots_stack[-size:], dtype=np.int64)
        del self.cpu_free_slots_stack[-size:]
        self._num_free_cpu_slots -= size
        self.cpu_req_to_token_slots[row_idx, cur_len:cur_len + size] = slots
        self.row_seq_lens[row_idx] += size
        return slots

    def _allocate_gpu_slots_locked(self, layer_idx: int, size: int) -> np.ndarray:
        """在指定层的 GPU active pool 中分配 slots。

        调用者必须已经持有该层 lock。原因是后台 prefetch、同步路径、
        free_seq 都可能同时修改同一层的 free stack 和 residency map。
        """
        if self._num_free_slots[layer_idx] < size:
            raise RuntimeError(
                f"Out of AttnPredict GPU active slots: layer={layer_idx} need={size} "
                f"free={self._num_free_slots[layer_idx]}"
            )
        slots = np.asarray(self.free_slots_stack[layer_idx][-size:], dtype=np.int32)
        del self.free_slots_stack[layer_idx][-size:]
        self._num_free_slots[layer_idx] -= size
        return slots

    def _copy_cpu_to_gpu(
        self,
        layer_idx: int,
        cpu_slots: list[int],
        gpu_slots: list[int],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """把一批 CPU full backing slots 对应的 K/V 拷贝到 GPU active slots。

        两阶段：
        1. CPU index_select：从 CPU full KV backing 中按 cpu_slots gather 出 host_k/host_v。
        2. H2D + index_copy_：把 host_k/host_v 搬到 GPU，并写入对应 gpu_slots。

        如果启用 pin staging，会先复制到 pinned memory，使后续 non_blocking H2D
        更容易和主计算流重叠。
        """
        if not cpu_slots:
            return
        with profiler.record("attnpredict_offload_cpu_gather_background"):
            # CPU gather：这里仍在 CPU 上做 index_select，避免先把完整 CPU cache 搬上 GPU。
            cpu_idx = torch.tensor(cpu_slots, dtype=torch.long, device="cpu")
            host_k = self.cpu_kv_cache[0, layer_idx].index_select(0, cpu_idx)
            host_v = self.cpu_kv_cache[1, layer_idx].index_select(0, cpu_idx)
            if self._pin_staging:
                # pinned memory 是异步 H2D 的常见 staging 区。这里多一次 CPU copy，
                # 换取后续 host_k.to("cuda", non_blocking=True) 更容易异步执行。
                pinned_k = torch.empty(host_k.shape, dtype=host_k.dtype, device="cpu", pin_memory=True)
                pinned_v = torch.empty(host_v.shape, dtype=host_v.dtype, device="cpu", pin_memory=True)
                pinned_k.copy_(host_k, non_blocking=False)
                pinned_v.copy_(host_v, non_blocking=False)
                host_k, host_v = pinned_k, pinned_v

        ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with ctx, profiler.record("attnpredict_offload_h2d_prefetch_stream"):
            # H2D 拷贝和 GPU active pool 写入。如果 stream 是预取流，
            # 这些操作不会阻塞主计算流，直到下一步同层消费时 wait event。
            gpu_idx = torch.tensor(gpu_slots, dtype=torch.long, device="cuda")
            gpu_k = host_k.to(device="cuda", non_blocking=True)
            gpu_v = host_v.to(device="cuda", non_blocking=True)
            k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
            k_cache.index_copy_(0, gpu_idx, gpu_k)
            v_cache.index_copy_(0, gpu_idx, gpu_v)

    def _copy_dirty_gpu_to_cpu(
        self,
        layer_idx: int,
        cpu_slots: list[int],
        gpu_slots: list[int],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """驱逐 dirty KV 前写回 CPU backing。"""
        if not cpu_slots:
            return
        ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with ctx, profiler.record("attnpredict_offload_dirty_d2h_writeback"):
            gpu_idx = torch.tensor(gpu_slots, dtype=torch.long, device="cuda")
            cpu_idx = torch.tensor(cpu_slots, dtype=torch.long, device="cpu")
            k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
            host_k = k_cache.index_select(0, gpu_idx).to(device="cpu", dtype=self.hf_config.torch_dtype)
            host_v = v_cache.index_select(0, gpu_idx).to(device="cpu", dtype=self.hf_config.torch_dtype)
            self.cpu_kv_cache[0, layer_idx].index_copy_(0, cpu_idx, host_k)
            self.cpu_kv_cache[1, layer_idx].index_copy_(0, cpu_idx, host_v)

    def _ensure_positions_resident(
        self,
        layer_idx: int,
        row_positions: dict[int, np.ndarray],# row -> 这行需要读的 token 的逻辑位置列表
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """
        给某一层，把下一次 attention 要读的 token 位置准备到 GPU active pool 里，同时把不再需要的 token 从 GPU active pool 释放掉
        """
        # CPU 侧的 GPU 驻留映射,如果是 -1，表示这个 token 当前不在 GPU active pool
        mirror = self.gpu_req_to_token_slots_cpu[layer_idx]

        load_cpu_slots: list[int] = []  # 待加载 token 在 CPU full backing 中的 slot。
        load_gpu_slots: list[int] = []  # 为待加载 token 新分配的 GPU active slot。
        dirty_cpu_slots: list[int] = [] # 驱逐前需要补写 CPU backing 的 dirty token。
        dirty_gpu_slots: list[int] = []

        with profiler.record("attnpredict_offload_residency_plan_cpu"):
            with self._layer_locks[layer_idx]:
                dirty = self._dirty_gpu_positions[layer_idx]
                for row_idx, positions in row_positions.items():
                    full_len = self.row_seq_lens[row_idx]

                    # 清理非法位置并去重，避免重复分配或访问越界。
                    # positions = np.asarray(positions, dtype=np.int64)
                    positions = positions[(positions >= 0) & (positions < full_len)]
                    positions_list = positions.tolist()
                    target = set(positions_list)
                    resident = self._resident_positions[layer_idx].setdefault(row_idx, set())
                    if not resident and full_len > 0: # 如果这个 row 的 resident set 还没初始化，就从 mirror 扫一遍，找出当前已经在 GPU 的位置。
                        resident_positions = np.flatnonzero(mirror[row_idx, :full_len] >= 0) # 返回所有为 True 的下标
                        row_has_dirty = any(dirty_row == row_idx for dirty_row, _ in dirty) # 检查这个 row 有没有 dirty KV
                        if resident_positions.size and not row_has_dirty: #扫描到当前 row 有 GPU resident，并且这个 row 没有 dirty token，就可以走快速 shrink。
                            keep = np.isin(resident_positions, positions)
                            evict_positions = resident_positions[~keep]
                            if evict_positions.size:
                                evict_slots = mirror[row_idx, evict_positions].astype(np.int32, copy=False)
                                mirror[row_idx, evict_positions] = -1
                                evict_slots_list = evict_slots.tolist()
                                self.free_slots_stack[layer_idx].extend(evict_slots_list)
                                self._num_free_slots[layer_idx] += len(evict_slots_list)
                            # 初次 shrink 后只保留小集合，避免长期维护 128k Python set。快速 shrink 后，resident set 只记录目标 positions 里仍然在 GPU 的位置
                            resident.update(pos for pos in positions_list if mirror[row_idx, pos] >= 0)
                        else:
                            resident.update(resident_positions.tolist()) # 不能走快速 shrink，只把 mirror 里已有的 resident 全部登记进 set

                    # 第一步：释放已经驻留 GPU、但下一步不再需要的 token。
                    # resident - target:当前在 GPU，但下一次 attention 不需要了
                    for pos in list(resident - target):
                        slot = mirror[row_idx, pos]
                        is_dirty = (row_idx, pos) in dirty
                        if is_dirty:
                            cpu_slot = self.cpu_req_to_token_slots[row_idx, pos]
                            if cpu_slot < 0:
                                raise RuntimeError(
                                    f"Missing CPU backing slot for dirty KV: layer={layer_idx} row={row_idx} pos={pos}"
                                )
                            dirty.remove((row_idx, pos))
                            dirty_cpu_slots.append(cpu_slot)
                            dirty_gpu_slots.append(slot)
                        mirror[row_idx, pos] = -1
                        resident.remove(pos)
                        if not is_dirty:
                            self.free_slots_stack[layer_idx].append(slot)
                            self._num_free_slots[layer_idx] += 1

                    # 第二步：找出目标集合里还没有 GPU slot 的 positions。
                    missing = [pos for pos in positions_list if pos not in resident]
                    if missing:
                        gpu_slots = self._allocate_gpu_slots_locked(layer_idx, len(missing))
                        for pos, gpu_slot in zip(missing, gpu_slots.tolist()):
                            # CPU backing 是最终真实来源；如果这里没有 cpu_slot，说明生命周期账本坏了。
                            cpu_slot = self.cpu_req_to_token_slots[row_idx, pos]
                            if cpu_slot < 0:
                                raise RuntimeError(
                                    f"Missing CPU backing slot: layer={layer_idx} row={row_idx} pos={pos}"
                                )
                            mirror[row_idx, pos] = gpu_slot
                            resident.add(pos)
                            load_cpu_slots.append(cpu_slot) # 从 CPU 哪些 slot 读
                            load_gpu_slots.append(gpu_slot) # 写到 GPU 哪些 active slot

        # 第三步：attention 只读 packed slots，GPU 全局映射不参与 decode 寻址。
        self._copy_dirty_gpu_to_cpu(layer_idx, dirty_cpu_slots, dirty_gpu_slots, stream=stream)
        if dirty_gpu_slots:
            with self._layer_locks[layer_idx]:
                self.free_slots_stack[layer_idx].extend(dirty_gpu_slots)
                self._num_free_slots[layer_idx] += len(dirty_gpu_slots)
        self._copy_cpu_to_gpu(layer_idx, load_cpu_slots, load_gpu_slots, stream=stream)

    def _ensure_positions_loaded(
        self,
        layer_idx: int,
        row_positions: dict[int, np.ndarray],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """后台预取只补缺失 KV，不释放旧 lease 正在读的 resident。"""
        mirror = self.gpu_req_to_token_slots_cpu[layer_idx]
        load_cpu_slots: list[int] = []
        load_gpu_slots: list[int] = []

        with profiler.record("attnpredict_offload_load_plan_cpu"):
            with self._layer_locks[layer_idx]:
                resident_rows = self._resident_positions[layer_idx]
                for row_idx, positions in row_positions.items():
                    full_len = self.row_seq_lens[row_idx]
                    positions = np.asarray(positions, dtype=np.int64)
                    positions = positions[(positions >= 0) & (positions < full_len)]
                    resident = resident_rows.setdefault(row_idx, set())
                    missing = [pos for pos in positions.tolist() if pos not in resident]
                    if not missing:
                        continue
                    if len(missing) > self._num_free_slots[layer_idx]:
                        # 空闲 slot 不足时跳过后台预取；消费 lease 时会同步补齐。
                        continue
                    gpu_slots = self._allocate_gpu_slots_locked(layer_idx, len(missing))
                    for pos, gpu_slot in zip(missing, gpu_slots.tolist()):
                        cpu_slot = self.cpu_req_to_token_slots[row_idx, pos]
                        if cpu_slot < 0:
                            raise RuntimeError(
                                f"Missing CPU backing slot: layer={layer_idx} row={row_idx} pos={pos}"
                            )
                        mirror[row_idx, pos] = gpu_slot
                        resident.add(pos)
                        load_cpu_slots.append(cpu_slot)
                        load_gpu_slots.append(gpu_slot)

        self._copy_cpu_to_gpu(layer_idx, load_cpu_slots, load_gpu_slots, stream=stream)

    def _create_tsp_mask(
        self,
        tsp_attn: torch.Tensor,
        *,
        seq_len: int,
        start_block: int,
        device: torch.device,
    ) -> torch.Tensor:
        """offload 只缓存 CNN hot mask；sink/recent 在 decode view 动态补。"""
        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        pred_len = tsp_attn.shape[-1]
        middle_budget = max(0, self.topk - self.sink_token - self.local_token)
        block_budget = min(middle_budget // self.pooling_block_size, pred_len)
        if block_budget <= 0:
            return keep_mask
        block_scores = tsp_attn.max(dim=0).values if tsp_attn.dim() == 2 else tsp_attn
        _, topk_indices = torch.topk(block_scores, block_budget, dim=-1)
        token_indices = (
            (topk_indices + start_block).unsqueeze(-1) * self.pooling_block_size
            + torch.arange(self.pooling_block_size, device=device)
        ).reshape(-1)
        sink_end = min(self.sink_token, seq_len)
        recent_start = max(sink_end, seq_len - self.local_token)
        token_indices = token_indices[(token_indices >= sink_end) & (token_indices < recent_start)]
        keep_mask[token_indices] = True
        return keep_mask

    def _compose_positions_from_hot(self, hot_positions: np.ndarray, full_len: int) -> np.ndarray:
        """把复用 hot tokens 和当前步动态 sink/recent 合成完整可见 positions。"""
        sink_end = min(self.sink_token, full_len)
        recent_start = max(sink_end, full_len - self.local_token)
        parts = [np.arange(0, sink_end, dtype=np.int64)]
        if hot_positions.size:
            hot = hot_positions[(hot_positions >= sink_end) & (hot_positions < recent_start)]
            if hot.size:
                parts.append(hot.astype(np.int64, copy=False))
        parts.append(np.arange(recent_start, full_len, dtype=np.int64))
        # sink/hot/recent 已经有序且互不重叠，避免每层每步重新排序去重。
        return np.concatenate(parts) if parts else np.empty((0,), dtype=np.int64)

    def _compose_static_positions_from_hot(self, hot_positions: np.ndarray, full_len: int) -> np.ndarray:
        """返回 lease 内稳定的 sink/hot positions；recent/current 每步动态补。"""
        sink_end = min(self.sink_token, full_len)
        recent_start = max(sink_end, full_len - self.local_token)
        parts = [np.arange(0, sink_end, dtype=np.int64)]
        if hot_positions.size:
            hot = hot_positions[(hot_positions >= sink_end) & (hot_positions < recent_start)]
            if hot.size:
                parts.append(hot.astype(np.int64, copy=False))
        return np.concatenate(parts) if parts else np.empty((0,), dtype=np.int64)

    def _hot_positions_from_scores(
        self,
        tsp_attn: torch.Tensor,
        *,
        seq_len: int,
        start_block: int,
    ) -> np.ndarray:
        """从 block score 直接生成中间 hot positions。"""
        pred_len = int(tsp_attn.shape[-1])
        middle_budget = int(self.topk) - int(self.sink_token) - int(self.local_token)
        block_budget = max(0, min(middle_budget // int(self.pooling_block_size), pred_len))
        if seq_len <= 0 or block_budget <= 0:
            return np.empty((0,), dtype=np.int64)

        block_scores = tsp_attn.max(dim=0).values if tsp_attn.dim() == 2 else tsp_attn
        _, topk_indices = torch.topk(block_scores, block_budget, dim=-1)
        token_indices = (
            (topk_indices + int(start_block)).unsqueeze(-1) * int(self.pooling_block_size)
            + self._pooling_offsets_gpu
        ).reshape(-1)
        sink_end = min(int(self.sink_token), int(seq_len))
        recent_start = max(sink_end, int(seq_len) - int(self.local_token))
        valid = (token_indices >= sink_end) & (token_indices < recent_start)
        # topk block 不重复，展开后的 token 也不重复；排序后一次性拷回 CPU。
        token_indices = token_indices[valid].sort().values
        return token_indices.to(device="cpu", dtype=torch.long).numpy().astype(np.int64, copy=False)

    def _commit_lease(
        self,
        layer_idx: int,
        hot_positions: dict[int, np.ndarray],
        lease_start_positions: dict[int, int],
    ) -> None:
        """提交已完成预取的 lease；之后 decode 才会切到新预测结果。"""
        lease = self._lease_hot_positions[layer_idx]
        if lease is None:
            lease = {}
            self._lease_hot_positions[layer_idx] = lease
        # 连续 batching 中新 row 可能单独完成 prefill，不能覆盖仍在 decode 的旧 row。
        lease.update(hot_positions)
        self._lease_start_positions[layer_idx].update(lease_start_positions) 
        #保存的是该层sink + hot 这部分稳定位置对应的 GPU slots
        #因为 sink + hot 在同一个 lease 内不变，所以 build_decode_view() 不想每一步都重新构造它，就缓存起来
        cache = self._static_view_cache[layer_idx] 
        for row_idx in lease_start_positions:
            cache.pop(row_idx, None)

    def _is_layer_reuse_source(self, layer_idx: int) -> bool:
        """本层是否真正跑 predictor；组内其他层复用它。"""
        return layer_idx % self._layer_reuse_stride == 0

    def _layer_reuse_targets(self, layer_idx: int) -> range:
        """返回组首层预测结果覆盖的层范围。"""
        end = min(self.num_layers, layer_idx + self._layer_reuse_stride)
        return range(layer_idx, end)

    def _expand_layer_results(
        self,
        layer_idx: int,
        hot_positions: dict[int, np.ndarray],
        lease_starts: dict[int, int],
    ) -> dict[int, dict[str, dict[int, np.ndarray] | dict[int, int]]]:
        """把复用组首层预测结果映射到组内层。"""
        return {
            i: {
                "hot_positions": hot_positions,
                "lease_start_positions": lease_starts,
            }
            for i in self._layer_reuse_targets(layer_idx)
        }

    def _ensure_layer_results_resident(
        self,
        layer_results: dict[int, dict[str, dict[int, np.ndarray] | dict[int, int]]],
        stream: torch.cuda.Stream | None,
        *,
        release_old: bool = True,
    ) -> None:
        """按层准备复用结果的 GPU residency。"""
        for layer_idx, result in layer_results.items():
            hot_positions = result["hot_positions"]
            lease_start_positions = result["lease_start_positions"]
            row_positions = {
                row_idx: self._compose_positions_from_hot(hot_positions[row_idx], full_len) #将 hot positions 和 sink/recent 合成完整 positions
                for row_idx, full_len in lease_start_positions.items()
            }
            if release_old:
                self._ensure_positions_resident(layer_idx, row_positions, stream=stream) # 确保给定 row 的seq 的 positions 都驻留在指定层的 GPU active pool 中
            else:
                self._ensure_positions_loaded(layer_idx, row_positions, stream=stream)

    def _commit_layer_results(
        self,
        layer_results: dict[int, dict[str, dict[int, np.ndarray] | dict[int, int]]],
    ) -> None:
        """提交组内各层 lease。"""
        for layer_idx, result in layer_results.items(): #TODO：既然复用层组里的lease一样，直接赋值不就可以了，为什么还要单独循环，循环里的步骤都是一模一样的，除了layer_idx不一样
            self._commit_lease(layer_idx, result["hot_positions"], result["lease_start_positions"])

    def _get_packed_slots_buffer(self, layer_idx: int, batch_size: int, max_keep: int, device: torch.device) -> torch.Tensor:
        """复用 decode packed slots；kernel 只读 view_lens 内的有效区域。"""
        cached = self._packed_slots_cache[layer_idx]
        if cached is None or cached.shape[0] < batch_size or cached.shape[1] < max_keep or cached.device != device:
            cached = torch.empty((batch_size, max_keep), dtype=torch.int32, device=device)
            self._packed_slots_cache[layer_idx] = cached
        return cached[:batch_size, :max_keep]

    def _get_packed_positions_buffer(self, layer_idx: int, batch_size: int, max_keep: int, device: torch.device) -> torch.Tensor:
        """复用 source layer 的 packed positions。"""
        cached = self._packed_positions_cache[layer_idx]
        if cached is None or cached.shape[0] < batch_size or cached.shape[1] < max_keep or cached.device != device:
            cached = torch.empty((batch_size, max_keep), dtype=torch.int32, device=device)
            self._packed_positions_cache[layer_idx] = cached
        return cached[:batch_size, :max_keep]

    def _get_local_req_indices(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """local req indices 只和 batch size 有关，避免每层重复 arange。"""
        cached = self._local_req_indices_cache
        if cached is None or cached.numel() < batch_size or cached.device != device:
            cached = torch.arange(batch_size, dtype=torch.int32, device=device)
            self._local_req_indices_cache = cached
        return cached[:batch_size]

    def _consume_prefetch(self, layer_idx: int, *, wait: bool) -> bool:
        """消费后台预测；wait=False 时没完成就继续使用旧 lease。
        没有后台任务：consumed=False，继续用旧 lease
        后台没完成，且没到 max_stale：consumed=False，继续用旧 lease
        后台完成了：consumed=True，切换到新 lease
        """
        future = self._prefetch_futures[layer_idx]
        if future is None: # 如果这一层没有后台任务，直接返回 False，表示没有新预测结果可消费
            return False 
        if not wait and not future.done():
            return False # # decode 正常步不会强等后台预测。如果还没完成，就返回 False，继续用旧预测结果。
        with profiler.record("attnpredict_offload_prefetch_wait"):#真正消费的新lease的地方
            result = future.result() # {"event": done,  "layer_results": layer_results }
            # for i, known_future in enumerate(self._prefetch_futures):
            #     if known_future is future:
            #         self._prefetch_futures[i] = None #  跨层复用时，多个 layer 可能共享同一个 future。这里把所有指向同一个 future 的槽位清空，避免后面重复消费同一个后台结果
            self._prefetch_futures[layer_idx] = None
            marker = (
                "attnpredict_offload_prefetch_event_ready"
                if result["event"].query() # event.query() 是检查 GPU event 是否已经完成。如果返回 True，说明后台 stream 上的 GPU 工作已经做完
                else "attnpredict_offload_prefetch_event_pending"
            )
            with profiler.record(marker):
                pass
            torch.cuda.current_stream().wait_event(result["event"])#当前主计算流要等后台 event，确保后面 decode 真正读取 KV 之前，后台搬运到 GPU 的 KV 已经可用。
            self._commit_layer_results(result["layer_results"])#把新预测结果提交为当前 lease
            if result.get("needs_cleanup", False):
                for cleanup_layer_idx in result["layer_results"]:
                    self._pending_residency_cleanup[cleanup_layer_idx] = True
        return True

    def _wait_prefetch(self, layer_idx: int) -> None:
        """等待指定层上一轮异步预取完成。

        只在“下一次同层真正要消费 KV”时等待，而不是在提交预取后立刻等待。
        这就是 offload 版本 overlap 的关键：主线程可以先继续算后续层。
        """
        self._consume_prefetch(layer_idx, wait=True)

    def _lease_missing_decode_rows(self, layer_idx: int) -> bool:
        """当前 lease 是否缺少本步 decode 的 row。"""
        lease = self._lease_hot_positions[layer_idx]
        return lease is None or any(row not in lease for row in self._decode_rows.tolist())

    def _lease_age_reached(self, layer_idx: int, steps: int) -> bool:
        """当前 lease 是否已被任一 row 复用到指定步数。"""
        starts = self._lease_start_positions[layer_idx]
        for row_idx, cur_pos in zip(self._decode_rows.tolist(), self._decode_current_positions.tolist()):
            if cur_pos - starts[row_idx] + 1 >= steps:
                return True
        return False

    def should_collect_decode_attn_score(self, layer_idx: int) -> bool:
        """每 4 步刷新一次；后台未完成时继续复用旧 lease。"""
        collect = False
        if not self._is_layer_reuse_source(layer_idx):
            self._collect_decode_positions[layer_idx] = collect
            return collect
        if self._prefetch_futures[layer_idx] is None and not self._lease_missing_decode_rows(layer_idx):
            collect = self._lease_age_reached(layer_idx, self._reuse_steps)
        self._collect_decode_positions[layer_idx] = collect
        return collect

    def decode_attn_score_max_len(self, layer_idx: int, context_lens: torch.Tensor) -> int:
        """decode 只需为 packed sparse view 分配 score buffer。"""
        middle_budget = max(0, int(self.topk) - int(self.sink_token) - int(self.local_token))
        block_budget = middle_budget // max(1, int(self.pooling_block_size))
        keep_bound = int(self.sink_token) + int(self.local_token) + block_budget * int(self.pooling_block_size)
        return max(1, keep_bound)

    def decode_view_max_len(self, layer_idx: int, context_lens: torch.Tensor) -> int:
        """build_decode_view 已在 CPU 侧算出 max_keep，避免每层 .item() 同步。"""
        return self._decode_view_max_lens[layer_idx]

    def get_layer_store_view(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回当前层写入 K/V 的目标 cache 和 slot_mapping。

        这是 offload decode 的“按层消费点”：
        - prefill：所有 token 已经在 _prepare_prefill 中分配好 GPU slots，直接返回。
        - decode：到达某一层时，才等待该层上一轮 prefetch；然后直接消费
          predictor 已保存的下一步 positions；最后给当前新 token 分配本层 GPU slot。

        这样不会在 decode step 一开始就等待所有层的预取，而是让前面层的计算
        和后面层的后台预取自然 overlap。
        """
        ctx = get_context()
        if not ctx.is_prefill:
            # 若新 lease 还没好，继续用旧 lease；没有旧 lease 时必须等待初始化完成。
            force_wait = self._lease_missing_decode_rows(layer_idx) or (
                self._prefetch_futures[layer_idx] is not None
                and self._lease_age_reached(layer_idx, self._max_stale_steps)
            )
            self._consume_prefetch(layer_idx, wait=force_wait) #预测步后，lease刷新，需要像复用层组提交新的lease
            lease_hot_positions = self._lease_hot_positions[layer_idx]

            state = self.layer_batch_states[layer_idx]

            rows = self._decode_rows
            current_positions = self._decode_current_positions

            # 普通复用步不扫全量 residency；切到新 lease 后，各层到达自身写入点时清理旧 resident。
            if self._pending_residency_cleanup[layer_idx]:
                with profiler.record("attnpredict_offload_compose_decode_positions"):
                    resident_positions: dict[int, np.ndarray] = {}#row_idx -> 这个 row 当前 attention 需要保证驻留 GPU 的历史位置
                    for row_idx, cur_pos in zip(rows.tolist(), current_positions.tolist()): #遍历本轮 decode batch 里的每条请求
                        positions = self._compose_positions_from_hot( #把当前 lease 里的 hot positions 拼成完整可见历史:sink + hot + recent
                            lease_hot_positions[row_idx],
                            self.row_seq_lens[row_idx],
                        )
                        resident_positions[row_idx] = positions[:-1]# current 是 recent 的最后一个位置；这里只保留历史 KV
                self._ensure_positions_resident(layer_idx, resident_positions, stream=None)
                self._pending_residency_cleanup[layer_idx] = False

            # current KV 尚未写入 CPU backing，先处理历史驻留，再为 current 分配 GPU slot。
            current_slots = self._ensure_current_decode_slots(layer_idx, rows, current_positions)
            state.slot_mapping = torch.tensor(current_slots, dtype=torch.int32, device="cuda")

        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        return k_cache, v_cache, self.layer_batch_states[layer_idx].slot_mapping

    def _ensure_current_decode_slots(
        self,
        layer_idx: int,
        rows: np.ndarray,
        positions: np.ndarray,
    ) -> list[int]:
        """确保当前 decode token 在本层有 GPU active slot。

        decode 新 token 是逐层写入的：同一个 token 在每一层的 K/V 不同，
        因此每层走到 attention 前，才为该层分配 GPU slot 并写入当前 K/V。
        """
        with profiler.record("attnpredict_offload_ensure_current_slots"):
            mirror = self.gpu_req_to_token_slots_cpu[layer_idx]
            result: list[int] = []
            with self._layer_locks[layer_idx]:
                resident_rows = self._resident_positions[layer_idx]
                for row_idx, pos in zip(rows.tolist(), positions.tolist()):
                    slot = mirror[row_idx, pos]
                    if slot < 0:
                        slot = self._allocate_gpu_slots_locked(layer_idx, 1)[0]
                        mirror[row_idx, pos] = slot
                    resident_rows.setdefault(row_idx, set()).add(pos)
                    result.append(slot)
            # decode kernel 使用 packed slots，不再读全局 GPU row/pos map。
            return result

    def on_kv_stored(
        self,
        layer_idx: int,
        k: torch.Tensor,
        slot_mapping: torch.Tensor,
        v: torch.Tensor | None = None,
    ):
        """GPU KV 写入后维护 CPU full backing。

        prefill 立即写 CPU；decode 只标记 dirty，驱逐前再写回，减少每步 D2H。
        """
        
        if not get_context().is_prefill:
            with self._layer_locks[layer_idx]:
                dirty = self._dirty_gpu_positions[layer_idx]
                for row_idx, pos in zip(self._decode_rows.tolist(), self._decode_current_positions.tolist()):
                    dirty.add((row_idx, pos))
            return
        cpu_slots_np = self._layer_cpu_slot_mapping[layer_idx]
        with profiler.record("attnpredict_offload_store_cpu_full_kv"):
            # prefill 是 chunk 内所有 token；decode 是 batch 内每个 seq 的当前 token。
            host_k = k.detach().to(device="cpu", dtype=self.hf_config.torch_dtype)
            host_v = v.detach().to(device="cpu", dtype=self.hf_config.torch_dtype)
            segments = self._cpu_store_segments
            if segments is None:
                cpu_slots = torch.tensor(cpu_slots_np, dtype=torch.long, device="cpu")
                self.cpu_kv_cache[0, layer_idx].index_copy_(0, cpu_slots, host_k)
                self.cpu_kv_cache[1, layer_idx].index_copy_(0, cpu_slots, host_v)
            else:
                for token_start, token_end, slot_start, slot_end in segments:
                    self.cpu_kv_cache[0, layer_idx, slot_start:slot_end].copy_(host_k[token_start:token_end])
                    self.cpu_kv_cache[1, layer_idx, slot_start:slot_end].copy_(host_v[token_start:token_end])

    @torch.no_grad()
    def prepare_prefill_predictor_inputs(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        cu_seqlens_q: torch.Tensor | None,
        *,
        num_heads: int,
        num_kv_heads: int,
        prefill_is_last_chunk: list[bool] | None = None,
    ) -> torch.Tensor | None:
        """Prefill 最后一块chunk 的 predictor 初始化。

        返回一个 4D tail-score buffer 给 prefill kernel 填写：
            [batch, num_heads, history_step, ceil(max_context_len / block_size)]

        on_prefill_layer_end() 会在 attention kernel 完成后消费这个 buffer，
        同步或异步初始化首个 decode 所需的 predictor mask。
        """
        if not self._is_layer_reuse_source(layer_idx):
            self._pending_prefill_views[layer_idx] = None
            return None
        if prefill_is_last_chunk is None:
            self._pending_prefill_views[layer_idx] = None
            return None
        last_batch_indices = [i for i, is_last in enumerate(prefill_is_last_chunk) if bool(is_last)]
        if not last_batch_indices:
            self._pending_prefill_views[layer_idx] = None
            return None

        view = self._make_prefill_tail_score_view(
            q=q,
            req_indices=req_indices,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens_q,
            num_heads=num_heads,
            last_batch_indices=last_batch_indices,
        )
        if view is None:
            self._pending_prefill_views[layer_idx] = None
            return None

        self._pending_prefill_views[layer_idx] = view
        return view["tail_score"]

    def prefill_attn_score_block_size(self, layer_idx: int) -> int | None:
        """offload 的 prefill tail-score 按 pooling block 写出，避免 token 级大缓冲。"""
        return self.pooling_block_size

    def _make_prefill_tail_score_view(
        self,
        *,
        q: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        cu_seqlens_q: torch.Tensor | None,
        num_heads: int,
        last_batch_indices: list[int],
    ) -> dict[str, object] | None:
        if cu_seqlens_q is None or cu_seqlens_q.numel() <= 1:
            return None

        batch_size = req_indices.numel()
        history = self.history_step
        max_len = context_lens.max().item()
        block_size = self.pooling_block_size
        if block_size <= 0:
            raise ValueError("attnpredict_pooling_block_size must be > 0")
        pooled_len = (max_len + block_size - 1) // block_size
        tail_score = torch.full(
            (batch_size, num_heads, history, pooled_len),
            0.0,
            dtype=torch.float32,
            device=q.device,
        )
        return {
            "tail_score": tail_score,
            "req_indices": req_indices.detach().to(device="cpu", dtype=torch.long).numpy().astype(np.int64, copy=True),
            "context_lens": context_lens.detach().to(device="cpu", dtype=torch.long).numpy().astype(np.int64, copy=True),
            "cu_seqlens_q": cu_seqlens_q.detach().to(device="cpu", dtype=torch.long).numpy().astype(np.int64, copy=True),
            "last_batch_indices": last_batch_indices,
            "num_heads": num_heads,
            "block_size": block_size,
            "pooled_len": pooled_len,
        }

    def _predict_prefill_positions(
        self,
        layer_idx: int,
        view: dict[str, object],
    ) -> dict[int, np.ndarray]:
        tail_score = view["tail_score"]
        req_indices = view["req_indices"]
        context_lens = view["context_lens"]
        cu_seqlens_q = view["cu_seqlens_q"]
        last_batch_indices = view["last_batch_indices"]
        block_size = view["block_size"]
        pooled_len = view["pooled_len"]

        row_positions: dict[int, np.ndarray] = {}
        for b in last_batch_indices:
            row_idx = int(req_indices[b])
            q_end = cu_seqlens_q[b + 1]
            q_start = cu_seqlens_q[b]
            q_len = q_end - q_start
            full_len = context_lens[b]
            if q_len <= 0 or full_len <= 0:
                continue

            take = min(self.history_step, q_len)
            active_pooled_len = min(pooled_len, (full_len + block_size - 1) // block_size)#pooled_len是按批序列里的最长的算的，短的实际pooled len要小于他
            attn_pooling = tail_score[b, :, :take, :active_pooled_len].to(self.hf_config.torch_dtype)

            self._update_row_prediction_from_pooled(
                layer_idx,
                row_idx,
                attn_pooling,
                seq_len=full_len,
            )
            hot = self.tsp_mask[layer_idx][row_idx].nonzero(as_tuple=False).squeeze(-1)
            row_positions[row_idx] = hot.to(device="cpu", dtype=torch.long).numpy().astype(np.int64, copy=False)
        return row_positions

    def build_decode_view(
        self,
        layer_idx: int,
        q: torch.Tensor,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        *,
        num_heads: int,
        num_kv_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """构造 decode kernel 需要读取的 packed GPU slots。

        SparseController.get_read_view() 对 attnpredict-offload 返回的是完整映射表，
        但真正 decode 前还需要把“本层本 batch 的可见 positions”压成二维 packed slots：
            packed_slots[b, j] = 第 b 个请求第 j 个可见 token 的 GPU slot
            view_lens[b]      = 第 b 个请求实际可见 token 数

        local_req_indices 使用 [0, batch_size) 是因为 packed_slots 已经是局部 batch 表，
        kernel 不再需要全局 row_idx 去查完整 mapping。
        """
        with profiler.record("attnpredict_offload_build_decode_view"):
            batch_size = req_indices.numel()
            rows = self._decode_rows
            mirror = self.gpu_req_to_token_slots_cpu[layer_idx]
            lease = self._lease_hot_positions[layer_idx]
            starts = self._lease_start_positions[layer_idx]
            static_cache = self._static_view_cache[layer_idx]

            row_views: list[tuple[int, torch.Tensor, torch.Tensor | None, np.ndarray, int, int]] = []
            keep_counts: list[int] = []
            rows_cpu = rows.tolist()
            full_lens: list[int] = []
            need_positions = self._collect_decode_positions[layer_idx]
            with self._layer_locks[layer_idx]:
                for row_idx in rows_cpu:
                    full_len = int(self.row_seq_lens[row_idx])
                    full_lens.append(full_len)
                    sink_end = min(self.sink_token, full_len)
                    recent_start = max(sink_end, full_len - self.local_token)
                    recent_len = full_len - recent_start
                    recent_slots = mirror[row_idx, recent_start:full_len].astype(np.int32, copy=True)

                    lease_start = starts[row_idx]
                    cached = static_cache.get(row_idx)
                    if cached is None or cached[0] != lease_start or (need_positions and cached[3] is None):
                        static_positions = self._compose_static_positions_from_hot(lease[row_idx], full_len)# 只包含sink+hot
                        static_slots = mirror[row_idx, static_positions].astype(np.int32, copy=True)
                        static_slots_gpu = torch.as_tensor(static_slots, dtype=torch.int32, device=q.device)
                        static_positions_gpu = (
                            torch.as_tensor(
                                static_positions.astype(np.int32, copy=False),
                                dtype=torch.int32,
                                device=q.device,
                            )
                            if need_positions
                            else None
                        )
                        cached = (lease_start, int(static_slots.size), static_slots_gpu, static_positions_gpu)
                        static_cache[row_idx] = cached
                    _, static_len, static_slots_gpu, static_positions_gpu = cached
                    row_views.append(
                        (static_len, static_slots_gpu, static_positions_gpu, recent_slots, recent_start, recent_len)
                    )
                    keep_counts.append(static_len + recent_len)

            max_keep = max(keep_counts)
            self._decode_view_max_lens[layer_idx] = max_keep

            packed_slots = self._get_packed_slots_buffer(layer_idx, batch_size, max_keep, q.device)
            packed_positions = (
                self._get_packed_positions_buffer(layer_idx, batch_size, max_keep, q.device)
                if need_positions
                else None
            )
            # 构造每个seq的 packed view：先是 lease 内稳定的部分（GPU 上已有），再拼上 recent 这部分
            for b, (
                static_len,
                static_slots_gpu,
                static_positions_gpu,
                recent_slots,
                recent_start,
                recent_len,
            ) in enumerate(row_views):
                k = static_len + recent_len
                packed_slots[b, :static_len].copy_(static_slots_gpu)
                packed_slots[b, static_len:k] = torch.as_tensor(recent_slots, dtype=torch.int32, device=q.device)
                if packed_positions is not None:
                    packed_positions[b, :static_len].copy_(static_positions_gpu)
                    packed_positions[b, static_len:k] = torch.arange(
                        recent_start, recent_start + recent_len, dtype=torch.int32, device=q.device
                    )

            view_lens = torch.tensor(keep_counts, dtype=torch.int32, device=q.device)
            local_req_indices = self._get_local_req_indices(batch_size, q.device)

            # 保存本次 decode view，attention kernel 会把 logits 写到 attn_score；
            # attention 结束后 predict_next_mask() 用这些元数据把 sparse logits 还原到 full positions。
            self._last_decode_view[layer_idx] = (
                {
                    "positions": packed_positions.detach(),
                    "req_indices_cpu": rows_cpu,
                    "view_lens_cpu": keep_counts,
                    "full_context_lens_cpu": full_lens,
                }
                if packed_positions is not None
                else None
            )
            return packed_slots, local_req_indices, view_lens

    def predict_next_mask(self, layer_idx: int, attn_logits: torch.Tensor) -> None:
        """根据当前层刚算完的 attention logits，预测下一步 active set。

        这个函数由 SparseController.on_attention_end() 调用。
        它只记录当前 CUDA stream 上的 event，然后把真正的 softmax/CNN/CPU gather/H2D
        放到后台线程和本层 prefetch stream 里执行。
        """
        view = self._last_decode_view[layer_idx]
        # 记录 attention 所在主 stream 的完成事件。后台 prefetch stream 必须 wait 它，
        # 才能安全读取 attn_logits。
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        view_copy = dict(view)
        logits_ref = attn_logits.detach()
        self._prefetch_futures[layer_idx] = self._prefetch_executor.submit(
            self._predict_and_prefetch_worker,
            layer_idx,
            logits_ref,
            view_copy,
            event,
        )

    def on_prefill_layer_end(self, layer_idx: int) -> None:
        """最后一个 prefill chunk 的 attention 完成后，准备首个 decode 的 active set。

        prefill 本身保持 full attention，不做稀疏。只有最后一个 prefill chunk
        才会在这里根据首个 decode mask 释放非 active GPU slots，并保留/预取
        首个 decode 需要的 token。mask 初始化和 residency 更新都在后台 worker 里完成。
        """
        ctx = get_context()
        if not ctx.prefill_is_last_chunk: 
            return

        pending_view = self._pending_prefill_views[layer_idx]
        if pending_view is None:
            return
        self._pending_prefill_views[layer_idx] = None
        # 同一层已有后台任务时，先消费/完成旧任务，再提交新的 prefill 初始化任务。
        self._wait_prefetch(layer_idx)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream()) # 标记当前主 stream 的完成事件，后台 prefetch stream 必须 wait 它，才能安全读取 prefill tail_score。
        self._prefetch_futures[layer_idx] = self._prefetch_executor.submit(
            self._prefill_predict_and_prefetch_worker,
            layer_idx,
            pending_view,
            event,
        )

    def _predict_and_prefetch_worker(
        self,
        layer_idx: int,
        attn_logits: torch.Tensor,
        view: dict[str, torch.Tensor | None],
        attention_done_event: torch.cuda.Event,
    ) -> dict[str, object]:
        """后台任务：等待 attention 完成，预测下一步 mask，并预取下一步 KV。

        返回一个 CUDA event。下一次同层 get_layer_store_view() 会等待这个 event，
        确认本层 active pool 已经准备好。
        """
        torch.cuda.set_device(attn_logits.device)
        stream = self._prefetch_streams[layer_idx]
        with torch.inference_mode(), torch.cuda.stream(stream):
            # 必须等主 stream 的 attention kernel 写完 attn_logits。
            stream.wait_event(attention_done_event)
            with profiler.record("attnpredict_offload_predict_cnn_stream"):
                hot_positions = self._predict_next_positions(layer_idx, attn_logits, view)
            lease_starts = {
                row: full_len
                for row, full_len in zip(view["req_indices_cpu"], view["full_context_lens_cpu"])
                if row in hot_positions
            }

        # residency 更新和 H2D 拷贝也挂在本层 prefetch stream 上。
        layer_results = self._expand_layer_results(layer_idx, hot_positions, lease_starts)
        self._ensure_layer_results_resident(layer_results, stream=stream, release_old=False)
        with torch.cuda.stream(stream):
            done = torch.cuda.Event()
            done.record(stream)
        return {"event": done, "layer_results": layer_results, "needs_cleanup": True}

    def _prefill_predict_and_prefetch_worker(
        self,
        layer_idx: int,
        view: dict[str, object],
        attention_done_event: torch.cuda.Event, #主 stream 的完成事件。后台 stream 必须先等这个 event，确保 prefill attention 已经写完 tail_score。
    ) -> dict[str, object]:
        """后台任务：prefill 最后一块结束后初始化 predictor，并准备首个 decode active set。"""
        tail_score = view["tail_score"]
        torch.cuda.set_device(tail_score.device)
        stream = self._prefetch_streams[layer_idx]
        with torch.inference_mode(), torch.cuda.stream(stream):
            stream.wait_event(attention_done_event)
            with profiler.record("attnpredict_offload_prefill_predict_stream"):
                hot_positions = self._predict_prefill_positions(layer_idx, view)
            lease_starts = {
                row: full_len
                for row, full_len in zip(
                    view["req_indices"].tolist(),
                    view["context_lens"].tolist(),
                )
                if row in hot_positions
            }

        layer_results = self._expand_layer_results(layer_idx, hot_positions, lease_starts)
        self._ensure_layer_results_resident(layer_results, stream=stream) # prefill阶段根据首个 decode 要用的 lease，把 GPU active pool 从“完整 prompt KV”收缩成“sink + hot + recent”。
        with torch.cuda.stream(stream):
            done = torch.cuda.Event()# 在 stream 当前队列的末尾放一个 event。
            done.record(stream) # 这个 event 完成，就表示它前面已经排到这个 stream 上的任务都完成了。
        return {"event": done, "layer_results": layer_results, "needs_cleanup": False}

    def _predict_next_positions(
        self,
        layer_idx: int,
        attn_logits: torch.Tensor,
        view: dict[str, torch.Tensor | None],
    ) -> dict[int, np.ndarray]:
        """执行 AttentionPredictor 的“logits -> 下一步 positions”逻辑。

        输入 attn_logits 是当前 decode attention kernel 写出的 sparse view logits。
        如果当前层只看了部分 token，logits 的最后一维只覆盖 packed view。
        这里直接按 packed positions 做 block pooling，避免构造 [heads, full_len]。
        """
        with profiler.record("attnpredict_offload_predict_next_positions"):
            positions = view["positions"]
            req_indices_cpu = view["req_indices_cpu"]
            view_lens_cpu = view["view_lens_cpu"]
            full_context_lens_cpu = view["full_context_lens_cpu"]

            row_positions: dict[int, np.ndarray] = {}
            for b, row_idx in enumerate(req_indices_cpu):
                # req_indices_cpu 是 cache manager 的全局 row；view_lens_cpu 是本次实际可见 token 数；
                # full_len 是当前序列完整逻辑长度。
                view_len = int(view_lens_cpu[b])
                full_len = int(full_context_lens_cpu[b])

                # attn_logits 是未 scale 的 raw logits；这里和原 attnpredict 一样乘 attn_scale 后 softmax。
                with profiler.record("attnpredict_offload_softmax"):
                    logits = attn_logits[b, :, :view_len].to(torch.float32)
                    attn = torch.softmax(logits * self.attn_scale, dim=-1)
                pos = positions[b, :view_len].to(device=attn.device, dtype=torch.long)
                pooled_len = (full_len + self.pooling_block_size - 1) // self.pooling_block_size
                block_idx = pos // self.pooling_block_size
                with profiler.record("attnpredict_offload_scatter_reduce"):
                    attn_pooling = torch.zeros(
                        (attn.shape[0], pooled_len),
                        dtype=attn.dtype,
                        device=attn.device,
                    )
                    attn_pooling.scatter_reduce_(
                        1,
                        block_idx.unsqueeze(0).expand(attn.shape[0], -1),
                        attn,
                        reduce="amax",
                        include_self=True,
                    )

                with profiler.record("attnpredict_offload_update_history"):
                    hist = self._update_pooled_attn_history(
                        self.attn_history[layer_idx].get(row_idx),
                        attn_pooling.unsqueeze(1).to(self.hf_config.torch_dtype),
                    )
                    self.attn_history[layer_idx][row_idx] = hist
                with profiler.record("attnpredict_offload_cnn"):
                    tsp_attn, start_block = self._time_sequence_predict(hist)
                with profiler.record("attnpredict_offload_topk"):
                    row_positions[row_idx] = self._hot_positions_from_scores(
                        tsp_attn,
                        seq_len=full_len,
                        start_block=start_block,
                    )
            return row_positions

    def _time_sequence_predict(self, attn_history: torch.Tensor) -> tuple[torch.Tensor, int]:
        """串行化 CNN predictor forward。

        这个方法不是直接在 offload 文件里显式调用，而是通过父类
        _update_row_prediction() 多态调用。当前调用链包括：
        prepare_prefill_predictor_inputs/_predict_prefill_positions/_predict_next_positions ->
        _update_row_prediction() -> self._time_sequence_predict()。

        多个层的后台线程可能同时调用同一个 self.cnn。为避免共享模块并发 forward
        带来的状态/stream 风险，这里用一个轻量锁把 CNN 调用串行化；具体计算仍然
        发生在调用方设置好的 CUDA stream 上。
        """
        with self._cnn_lock:
            return super()._time_sequence_predict(attn_history)

    def _prepare_prefill(self, seqs: list[Sequence]):
        """准备 prefill 输入，并为本 chunk 所有 token 分配 GPU/CPU slots。

        offload 的 prefill 策略是 full attention：
        - 当前 chunk 的所有 token 都分配 GPU active slots。
        - 同一批 token 也分配 CPU full backing slots。
        - context_lens 是每个序列 prefill 到当前 chunk 后的完整长度。

        这样多 chunk prefill 时，每个 chunk 都能看到“已有历史 + 当前 chunk”的完整 KV。
        真正的 GPU active 裁剪只发生在最后一个 prefill chunk 的 attention 结束后。
        """
        with profiler.record("cache_prepare_prefill"):
            total_chunk_tokens = sum(seq.current_chunk_size for seq in seqs)

            # 模型 forward 的扁平输入。多个序列的 chunk 会拼成一个 1D token 列表。
            input_ids_np = np.empty(total_chunk_tokens, dtype=np.int64)
            positions_np = np.empty(total_chunk_tokens, dtype=np.int64)
            cu_seqlens_q = [0]

            # 每层各自有 GPU active pool，所以 prefill slot_mapping 也是 per-layer 的。
            layers_slot_mapping_cuda = torch.empty(
                (self.num_layers, total_chunk_tokens), dtype=torch.int32, device="cuda"
            )
            context_lens_list = [[] for _ in range(self.num_layers)]

            # CPU slot_mapping 对所有层相同：同一个 token 的 CPU slot 在每层 CPU cache 中
            # 存该层对应 K/V。
            # CPU index_copy_ 使用 int64/long 索引。
            cpu_slot_mapping = np.empty((total_chunk_tokens,), dtype=np.int64)
            # 本轮 prefill 中可连续写入 CPU backing 的 token/slot 区间。
            cpu_store_segments: list[tuple[int, int, int, int]] = []

            token_offset = 0
            for seq in seqs:
                chunk_size = seq.current_chunk_size
                start_idx = seq.num_prefilled_tokens
                end_idx = start_idx + chunk_size
                row_idx = self._get_free_row(seq.seq_id)

                # 如果该序列是 chunked prefill 的后续 chunk，row_seq_lens 必须刚好等于
                # 已经 prefill 的 token 数，否则说明调度/释放账本出错。
                if self.row_seq_lens[row_idx] != start_idx:
                    raise ValueError(
                        "AttnPredict offload row length mismatch in prefill: "
                        f"seq_id={seq.seq_id} row_seq_len={self.row_seq_lens[row_idx]} start_idx={start_idx}"
                    )

                # 先给完整历史 backing 分配 CPU slots。row_seq_lens 会在这里增长。
                cpu_slots = self._allocate_cpu_slots(seq.seq_id, chunk_size)
                cpu_slot_mapping[token_offset:token_offset + chunk_size] = cpu_slots
                if (
                    chunk_size > 0
                    and int(cpu_slots[-1]) - int(cpu_slots[0]) + 1 == chunk_size
                    and np.all(cpu_slots == np.arange(int(cpu_slots[0]), int(cpu_slots[-1]) + 1))
                ):
                    # 连续 CPU slots 走 slice copy；不连续时回退 index_copy_。
                    cpu_store_segments.append(
                        (token_offset, token_offset + chunk_size, int(cpu_slots[0]), int(cpu_slots[-1]) + 1)
                    )

                for layer_id in range(self.num_layers):
                    # 每一层都要给当前 chunk 分配 GPU slots，保证 prefill full attention。
                    with self._layer_locks[layer_id]:
                        gpu_slots = self._allocate_gpu_slots_locked(layer_id, chunk_size)
                        self.gpu_req_to_token_slots_cpu[layer_id][row_idx, start_idx:end_idx] = gpu_slots

                    # 同步更新 GPU 映射表，并记录本层 store_kvcache 要用的 slot_mapping。
                    slots_cuda = torch.tensor(gpu_slots, dtype=torch.int32, device="cuda")
                    self.buffer_req_to_token_slots[layer_id][row_idx, start_idx:end_idx] = slots_cuda
                    layers_slot_mapping_cuda[layer_id, token_offset:token_offset + chunk_size] = slots_cuda
                    context_lens_list[layer_id].append(end_idx)

                chunk_tokens = seq.token_ids
                if len(chunk_tokens) > chunk_size:
                    chunk_tokens = chunk_tokens[start_idx:end_idx]
                input_ids_np[token_offset:token_offset + chunk_size] = chunk_tokens
                positions_np[token_offset:token_offset + chunk_size] = np.arange(start_idx, end_idx)

                # cu_seqlens_q 用于 prefill attention kernel 区分每个序列 chunk 的边界。
                cu_seqlens_q.append(cu_seqlens_q[-1] + chunk_size)
                token_offset += chunk_size

            # 写入每层 LayerBatchStates。attention.py 会按 now_layer_idx 取当前层状态。
            layers_context_lens_cuda = torch.tensor(context_lens_list, dtype=torch.int32, device="cuda")
            req_ids = [self.seq_id_to_row[seq.seq_id] for seq in seqs]
            req_ids_cuda = torch.tensor(req_ids, dtype=torch.int32, device="cuda")
            for layer_id in range(self.num_layers):
                state = self.layer_batch_states[layer_id]
                state.slot_mapping = layers_slot_mapping_cuda[layer_id]
                state.context_lens = layers_context_lens_cuda[layer_id]
                state.req_indices = req_ids_cuda
                # on_kv_stored() 按当前 layer 取映射，内容跨层相同。
                self._layer_cpu_slot_mapping[layer_id] = cpu_slot_mapping
            self._cpu_store_segments = cpu_store_segments if len(cpu_store_segments) == len(seqs) else None

            input_ids = torch.from_numpy(input_ids_np).to("cuda")
            positions = torch.from_numpy(positions_np).to("cuda")
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cuda")
            return input_ids, positions, cu_seqlens_q

    def _prepare_decode(self, seqs: list[Sequence]):
        """准备 decode 输入，并只分配 CPU full backing slot。

        decode 新 token 的 GPU slot 不在这里一次性为所有层分配，而是在
        get_layer_store_view(layer) 中逐层分配。这样每层可以先消费自己的
        prefetch 结果，再写入当前层的新 KV。
        """
        with profiler.record("cache_prepare_decode"):
            batch_size = len(seqs)
            input_ids_list = [seq.last_token for seq in seqs]
            positions_list = [seq.num_tokens - 1 for seq in seqs]
            cpu_slots_batch = np.empty((batch_size,), dtype=np.int64)
            rows = np.empty((batch_size,), dtype=np.int64)
            current_positions = np.empty((batch_size,), dtype=np.int64)

            for b, seq in enumerate(seqs):
                row_idx = self._get_free_row(seq.seq_id)

                # 当前 token 的逻辑位置就是 CPU backing 分配前的 row_seq_lens。
                current_positions[b] = self.row_seq_lens[row_idx]
                cpu_slots_batch[b] = self._allocate_cpu_slots(seq.seq_id, 1)[0]
                rows[b] = row_idx

            # _allocate_cpu_slots 后 row_seq_lens 已经 +1，所以 context_lens 包含当前 token。
            context_lens_np = self.row_seq_lens[rows]
            context_lens_cuda = torch.tensor(context_lens_np, dtype=torch.int32, device="cuda")
            req_indices_cuda = torch.tensor(rows, dtype=torch.int32, device="cuda")

            # decode 当前 token 的 GPU slot 逐层分配，所以 prepare 阶段先放 -1 占位。
            empty_slot_mapping = torch.full((batch_size,), -1, dtype=torch.int32, device="cuda")

            for layer_id in range(self.num_layers):
                state = self.layer_batch_states[layer_id]
                state.slot_mapping = empty_slot_mapping
                state.context_lens = context_lens_cuda
                state.req_indices = req_indices_cuda
                self._layer_cpu_slot_mapping[layer_id] = cpu_slots_batch

            # 后续每层 get_layer_store_view() 需要这两个数组来构建 base/current positions。
            self._decode_rows = rows
            self._decode_current_positions = current_positions

            input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device="cuda")
            positions = torch.tensor(positions_list, dtype=torch.int64, device="cuda")
            return input_ids, positions, None

    def free_seq(self, seq_id: int):
        """释放某个序列占用的 GPU active slots、CPU full slots 和 predictor 状态。"""
        row_idx = self.seq_id_to_row.pop(seq_id)
        for layer_idx in range(self.num_layers):
            # 释放前先等后台任务结束，避免后台还在访问这个 row 的映射或 slot。
            self._wait_prefetch(layer_idx)
            lease_hot_positions = self._lease_hot_positions[layer_idx]
            if lease_hot_positions is not None:
                lease_hot_positions.pop(row_idx, None)
            self._lease_start_positions[layer_idx].pop(row_idx, None)
            with self._layer_locks[layer_idx]:
                # 回收该 row 在该层当前仍驻留 GPU 的 active slots。
                resident = self._resident_positions[layer_idx].pop(row_idx, set())
                if not resident:
                    full_len = self.row_seq_lens[row_idx]
                    resident.update(
                        np.flatnonzero(
                            self.gpu_req_to_token_slots_cpu[layer_idx][row_idx, :full_len] >= 0
                        ).tolist()
                    )
                if resident:
                    pos_arr = np.fromiter(resident, dtype=np.int64)
                    slots = self.gpu_req_to_token_slots_cpu[layer_idx][row_idx, pos_arr]
                    slots = slots[slots >= 0].astype(np.int32, copy=False)
                    if slots.size:
                        self.free_slots_stack[layer_idx].extend(slots.tolist())
                        self._num_free_slots[layer_idx] += slots.size
                self.gpu_req_to_token_slots_cpu[layer_idx][row_idx, :] = -1
                self.buffer_req_to_token_slots[layer_idx][row_idx, :] = -1
                dirty = self._dirty_gpu_positions[layer_idx]
                dirty.difference_update({key for key in dirty if key[0] == row_idx})
                self._static_view_cache[layer_idx].pop(row_idx, None)

            # 清理该层该 row 的 predictor history/mask。
            self.attn_history[layer_idx].pop(row_idx, None)
            self.tsp_mask[layer_idx].pop(row_idx, None)

        # 回收 CPU full backing slots。CPU slot 是跨层共享 token slot，
        # 所以只需要按 row/pos 回收一次。
        full_len = self.row_seq_lens[row_idx]
        cpu_slots = self.cpu_req_to_token_slots[row_idx, :full_len]
        cpu_slots = cpu_slots[cpu_slots >= 0]
        if cpu_slots.size:
            self.cpu_free_slots_stack.extend(cpu_slots.tolist())
            self._num_free_cpu_slots += cpu_slots.size
        self.cpu_req_to_token_slots[row_idx, :] = -1
        self.row_seq_lens[row_idx] = 0
        self.free_rows.append(row_idx)
