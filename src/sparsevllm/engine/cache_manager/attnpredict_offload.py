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
        assert world_size == 1, "attnpredict-offload currently supports tensor_parallel_size=1." # TODO 为什么无法支持world_size >1 的例子
        # 分配 GPU active KV pool 和 CPU full KV backing
        self.allocate_kv_cache()

        # ------------------------------------------------------------------
        # GPU active pool 元数据
        # ------------------------------------------------------------------
        # 每层都有一套独立的 GPU slot 池。slot 里只放当前驻留 GPU 的 token KV，
        # 不再表示“这个序列的完整历史都在 GPU 上”。
        num_slots = int(config.num_kvcache_slots)
        self.layer_num_slots = [num_slots for _ in range(self.num_layers)]
        self.free_slots_stack: list[list[int]] = [list(range(num_slots)) for _ in range(self.num_layers)]
        self._num_free_slots = [num_slots for _ in range(self.num_layers)]

        # GPU 侧 row -> slot 映射。decode kernel 会用它把逻辑 token 位置映射到
        # 实际 GPU KV slot。值为 -1 表示这个 token 当前不在 GPU 上。
        self.buffer_req_to_token_slots = [
            torch.full((self.max_buffer_rows, self.max_model_len), -1, dtype=torch.int32, device="cuda")
            for _ in range(self.num_layers)
        ]

        # CPU 侧镜像，方便 Python 后台线程快速判断某个 token 是否已驻留 GPU。
        # 这个镜像和 buffer_req_to_token_slots 保持同样语义，但存在 CPU numpy 数组里。
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
        self._num_free_cpu_slots = int(self.cpu_num_slots)
        self._cpu_backing_dirty: list[dict[int, set[int]]] = [
            dict() for _ in range(self.num_layers)
        ]

        # 本轮 forward 中每个 token 对应的 CPU slot。attention.py 写完 GPU KV 后，
        # on_kv_stored() 会用它把同一批 K/V 复制到 CPU full backing。
        self._layer_cpu_slot_mapping: list[np.ndarray | None] = [None for _ in range(self.num_layers)]

        # decode 当前 step 的 row 和当前位置。get_layer_store_view() 逐层消费时会用它
        # 构造“本层需要先确保驻留 GPU 的历史位置”。
        self._decode_current_positions: np.ndarray | None = None
        self._decode_rows: np.ndarray | None = None

        # 每层本轮 decode 实际要读的逻辑 token positions。get_layer_store_view() 先准备，
        # build_decode_view() 再把它转换成 packed GPU slots 交给 decode kernel。
        self._decode_view_positions: list[list[np.ndarray] | None] = [None for _ in range(self.num_layers)]
        # 每层每 row 当前 resident positions。值为 None 表示该 row 从 0 到
        # row_seq_lens[row] 全量连续驻留，这是 full-resident fast path 的常见状态。
        self._resident_positions: list[dict[int, np.ndarray | None]] = [
            dict() for _ in range(self.num_layers)
        ]
        # 复用 decode packed view 的 GPU buffers，减少每层每步重复分配。
        self._decode_packed_slots: list[torch.Tensor | None] = [None for _ in range(self.num_layers)]
        self._decode_packed_positions: list[torch.Tensor | None] = [None for _ in range(self.num_layers)]
        self._decode_view_lens_buf: list[torch.Tensor | None] = [None for _ in range(self.num_layers)]
        self._decode_local_req_indices: list[torch.Tensor | None] = [None for _ in range(self.num_layers)]
        self._decode_current_slot_mapping_buf: list[torch.Tensor | None] = [None for _ in range(self.num_layers)]
        self._decode_fast_view_plans: list[list[dict[str, object]] | None] = [None for _ in range(self.num_layers)]
        self._decode_fast_prefix_meta: list[list[dict[str, int]] | None] = [
            None for _ in range(self.num_layers)
        ]
        self._decode_full_resident_layers: list[bool] = [False for _ in range(self.num_layers)]
        self._decode_current_slots_np: np.ndarray | None = None
        self._decode_current_slots_cuda: torch.Tensor | None = None
        self._decode_recent_positions_np: dict[int, np.ndarray] = {}
        self._decode_recent_positions_cuda: dict[int, torch.Tensor] = {}
        self._lease_versions: list[dict[int, int]] = [dict() for _ in range(self.num_layers)]
        self._full_resident_prefix_cache: list[dict[int, dict[str, object]]] = [
            dict() for _ in range(self.num_layers)
        ]
        # lease 只缓存 predictor 选出的中间 hot positions。sink/recent/current 每步
        # 动态拼接，避免复用旧完整 view 导致 local window 漂移。
        self._leased_hot_positions: list[dict[int, np.ndarray]] = [dict() for _ in range(self.num_layers)]
        self._lease_start_positions: list[dict[int, int]] = [dict() for _ in range(self.num_layers)]
        self._last_refresh_positions: list[dict[int, int]] = [dict() for _ in range(self.num_layers)]
        self._reuse_steps = int(config.attnpredict_reuse_steps)
        self._max_stale_steps = int(config.attnpredict_max_stale_steps)
        self._cnn_batch_max_heads = 128

        # AttentionPredictor 算法状态与普通 attnpredict 完全一致，复用父类 helper。
        # StandardCacheManager 的全局 GPU slot 结构，和 offload 的 per-layer active pool 冲突。
        self._init_attnpredictor_state(config)

        # ------------------------------------------------------------------
        # 异步预取资源
        # ------------------------------------------------------------------
        # 后台 CPU 线程负责提交 predictor/prefetch 任务；每层一个 CUDA stream，
        # 用于让预测和 H2D 拷贝尽量与主计算流 overlap。
        cpu_threads = int(config.attnpredict_offload_cpu_threads or 1)
        torch.set_num_threads(cpu_threads)
        self._prefetch_enabled = bool(config.attnpredict_offload_prefetch)
        self._pin_staging = bool(config.attnpredict_offload_pin_staging)
        self._prefetch_executor = ThreadPoolExecutor(max_workers=cpu_threads)
        self._prefetch_streams = [torch.cuda.Stream() for _ in range(self.num_layers)] # 每层一个 CUDA stream，让后台预取不占用主计算流
        self._prefetch_futures: list[Future | None] = [None for _ in range(self.num_layers)] # 后台任务句柄
        self._pending_prefill_views: list[dict[str, object] | None] = [None for _ in range(self.num_layers)]
        self._pending_decode_predict_inputs: list[dict[str, object]] = []
        self._layer_locks = [threading.RLock() for _ in range(self.num_layers)] #  Python 锁，保护这一层的 GPU active pool 元数据，防止主线程和后台线程同时改
        self._cnn_lock = threading.RLock() # 所有层共享 CNN predictor 的并发保护锁

        logger.info(
            "AttnPredict offload allocation: gpu_active_slots={} cpu_full_slots={} "
            "layers={} prefetch={} cpu_threads={} reuse_steps={} max_stale_steps={}".format(
                num_slots,
                self.cpu_num_slots,
                self.num_layers,
                self._prefetch_enabled,
                cpu_threads,
                self._reuse_steps,
                self._max_stale_steps,
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
        full_resident_target = int(self.max_model_len) * int(
            max(1, self.config.max_num_seqs_in_batch)
        )
        self.config.num_kvcache_slots = max(
            1,
            min(capacity_slots, max(1, full_resident_target)),
        )

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

        # 理想情况下至少能容纳一个最大 batch 中所有序列的完整上下文。
        desired = int(self.max_model_len) * int(max(1, self.config.max_num_seqs_in_batch))# TODO 这里是不是应该受限于gpu利用率，即一次前向传播最大能允许多少个token，而不是max_model_len
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
        return min(int(gpu_free), int(self._num_free_cpu_slots))

    def prompt_admission_free_slots(self) -> int:
        """prompt 准入时使用的可用容量。

        prefill token 必须同时写入 GPU active pool 和 CPU full backing，
        num_free_slots 已经取了 GPU/CPU 两侧的较小值。
        """
        return int(self.num_free_slots)

    def free_slot_stats(self) -> dict[str, int]:
        """返回调试日志用的 GPU/CPU 剩余 slot 统计。"""
        gpu_free = min(self._num_free_slots)
        return {
            "free_slots": int(self.num_free_slots),
            "free_gpu_active_slots": int(gpu_free),
            "free_cpu_full_slots": int(self._num_free_cpu_slots),
        }

    def prefill_batched_tokens_margin(self) -> int:
        """给 chunk prefill 预留的 token 预算余量。

        AttentionPredictor 需要最后 history_step 个 query 来初始化历史，
        因此调度时留一点 batch token 余量，避免最后一段被切得太小。
        """
        return int(self.config.attnpredict_history_steps)

    def remaining_prefill_tokens(self, seq: Sequence) -> int:
        """返回调度器本轮还应该继续 prefill 的 token 数。

        这里沿用 SnapKV/StreamingLLM 的“保留尾部窗口”写法：
        如果剩余 token 多于 history_step，就先只推进到尾部 history_step 之前，
        让最后一块有足够 query 用于初始化 predictor history。
        """
        remaining = int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
        history = int(self.config.attnpredict_history_steps)
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
        cur_len = int(self.row_seq_lens[row_idx])

        # row_seq_lens 是完整逻辑长度，不能超过 max_model_len 对应的映射表宽度。
        if cur_len + int(size) > int(self.max_model_len):
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
        del self.free_slots_stack[layer_idx][-size:] #TODO 标准的cachemanager就没有del。这一步是否多余、浪费时间？
        self._num_free_slots[layer_idx] -= size
        return slots

    @staticmethod
    def _normalize_positions(positions: np.ndarray, full_len: int) -> np.ndarray:
        """过滤越界位置并返回升序唯一 int64 positions。"""
        if full_len <= 0:
            return np.empty((0,), dtype=np.int64)
        positions = np.asarray(positions, dtype=np.int64)
        positions = positions[(positions >= 0) & (positions < int(full_len))]
        if positions.size == 0:
            return np.empty((0,), dtype=np.int64)
        return np.unique(positions)

    @staticmethod
    def _concat_int_arrays(parts: list[np.ndarray], dtype) -> np.ndarray:
        if not parts:
            return np.empty((0,), dtype=dtype)
        return np.concatenate(parts).astype(dtype, copy=False)

    def _resident_array_locked(self, layer_idx: int, row_idx: int, full_len: int) -> np.ndarray:
        """返回 resident positions 数组；None marker 表示全量连续驻留。"""
        resident = self._resident_positions[layer_idx]
        row = int(row_idx)
        if row in resident and resident[row] is None:
            return np.arange(int(full_len), dtype=np.int64)
        positions = resident.get(row)
        if positions is None:
            return np.empty((0,), dtype=np.int64)
        return positions

    def _set_resident_positions_locked(
        self,
        layer_idx: int,
        row_idx: int,
        positions: np.ndarray,
        full_len: int,
    ) -> None:
        row = int(row_idx)
        positions = self._normalize_positions(positions, int(full_len))
        if positions.size == int(full_len) and (
            int(full_len) == 0 or (int(positions[0]) == 0 and int(positions[-1]) == int(full_len) - 1)
        ):
            self._resident_positions[layer_idx][row] = None
        else:
            self._resident_positions[layer_idx][row] = positions

    def _add_resident_position_locked(self, layer_idx: int, row_idx: int, pos: int) -> None:
        """记录单个新 resident position，保持 None marker 的 full-resident 语义。"""
        row = int(row_idx)
        pos = int(pos)
        resident = self._resident_positions[layer_idx]
        if row in resident and resident[row] is None:
            return
        old = resident.get(row)
        if old is None or old.size == 0:
            resident[row] = np.asarray([pos], dtype=np.int64)
            return
        insert_at = int(np.searchsorted(old, pos))
        if insert_at < old.size and int(old[insert_at]) == pos:
            return
        resident[row] = np.insert(old, insert_at, pos).astype(np.int64, copy=False)

    def _can_keep_full_resident_for_rows(self, layer_idx: int, rows) -> bool:
        """判断这些 row 是否已经完整驻留，可跳过 offload 驻留管理。"""
        if rows is None:
            return False
        rows_arr = np.asarray(list(rows), dtype=np.int64)
        if rows_arr.size == 0:
            return False
        with self._layer_locks[layer_idx]:
            total_len = 0
            resident = self._resident_positions[layer_idx]
            for row_idx in rows_arr.tolist():
                row = int(row_idx)
                total_len += int(self.row_seq_lens[row])
                if total_len > int(self.layer_num_slots[layer_idx]):
                    return False
                if row not in resident or resident[row] is not None:
                    return False
        return True

    def _has_full_resident_rows(self, layer_idx: int, rows) -> bool:
        if rows is None:
            return False
        rows_arr = np.asarray(list(rows), dtype=np.int64)
        if rows_arr.size == 0:
            return False
        with self._layer_locks[layer_idx]:
            resident = self._resident_positions[layer_idx]
            return any(int(row) in resident and resident[int(row)] is None for row in rows_arr.tolist())

    def _stage_dirty_cpu_backing_locked(
        self,
        layer_idx: int,
        row_idx: int,
        pos: int,
        slot: int,
    ) -> None:
        row = int(row_idx)
        pos = int(pos)
        if pos < 0:
            return
        cpu_slot = int(self.cpu_req_to_token_slots[row, pos])
        if cpu_slot < 0:
            raise RuntimeError(
                f"Missing CPU backing slot for dirty KV: layer={layer_idx} row={row} pos={pos}"
            )
        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        self.cpu_kv_cache[0, layer_idx, cpu_slot].copy_(k_cache[int(slot)], non_blocking=False)
        self.cpu_kv_cache[1, layer_idx, cpu_slot].copy_(v_cache[int(slot)], non_blocking=False)
        dirty = self._cpu_backing_dirty[layer_idx].get(row)
        if dirty is not None:
            dirty.discard(pos)
            if not dirty:
                self._cpu_backing_dirty[layer_idx].pop(row, None)

    def _write_gpu_map(
        self,
        layer_idx: int,
        rows,
        positions,
        slots,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """把 CPU 侧 residency 变更同步写到 GPU row/pos -> slot 映射表。

        decode kernel 实际读取的是 GPU 上的 buffer_req_to_token_slots，
        因此后台线程完成释放/加载后也必须把映射写回 GPU。
        如果传入 stream，就在指定 CUDA stream 上执行这些写入。
        """
        if len(positions) == 0:
            return
        ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with ctx, profiler.record("attnpredict_offload_write_gpu_map"):
            rows_t = torch.as_tensor(rows, dtype=torch.long, device="cuda")
            pos_t = torch.as_tensor(positions, dtype=torch.long, device="cuda")
            slots_t = torch.as_tensor(slots, dtype=torch.int32, device="cuda")
            self.buffer_req_to_token_slots[layer_idx][rows_t, pos_t] = slots_t

    def _copy_cpu_to_gpu(
        self,
        layer_idx: int,
        cpu_slots,
        gpu_slots,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """把一批 CPU full backing slots 对应的 K/V 拷贝到 GPU active slots。

        两阶段：
        1. CPU index_select：从 CPU full KV backing 中按 cpu_slots gather 出 host_k/host_v。
        2. H2D + index_copy_：把 host_k/host_v 搬到 GPU，并写入对应 gpu_slots。

        如果启用 pin staging，会先复制到 pinned memory，使后续 non_blocking H2D
        更容易和主计算流重叠。
        """
        if len(cpu_slots) == 0:
            return
        with profiler.record("attnpredict_offload_cpu_gather_background"):
            # CPU gather：这里仍在 CPU 上做 index_select，避免先把完整 CPU cache 搬上 GPU。
            cpu_idx = torch.as_tensor(cpu_slots, dtype=torch.long, device="cpu")
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
            gpu_idx = torch.as_tensor(gpu_slots, dtype=torch.long, device="cuda")
            gpu_k = host_k.to(device="cuda", non_blocking=True)
            gpu_v = host_v.to(device="cuda", non_blocking=True)
            k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
            k_cache.index_copy_(0, gpu_idx, gpu_k)
            v_cache.index_copy_(0, gpu_idx, gpu_v)

    def _ensure_positions_resident(
        self,
        layer_idx: int,
        row_positions: dict[int, np.ndarray],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """确保给定 row的seq 的 positions 都驻留在指定层的 GPU active pool 中。

        输入 row_positions 的含义：
            row_idx -> 下一次 attention 需要可见的 token 逻辑位置集合

        本方法做三件事：
        1. 释放“不在目标集合中”的旧 GPU resident token。
        2. 为“目标集合中但当前不在 GPU”的 token 分配 GPU slot。
        3. 从 CPU full backing 把缺失 token 的 K/V 拷贝回 GPU active pool。

        注意：是否保留 current/recent/sink/topk 已经体现在 row_positions 里；
        这里不再关心 token 为什么被选中，只负责 residency。
        """
        # CPU 侧 GPU residency 镜像：mirror[row, pos] = gpu_slot，-1 表示当前不在 GPU。
        mirror = self.gpu_req_to_token_slots_cpu[layer_idx]

        # 待释放/加载清单，使用 numpy 分段拼接，避免 128K 级 Python for 扫描。
        free_rows_parts: list[np.ndarray] = []
        free_pos_parts: list[np.ndarray] = []
        free_slot_parts: list[np.ndarray] = []
        load_rows_parts: list[np.ndarray] = []
        load_pos_parts: list[np.ndarray] = []
        load_cpu_slot_parts: list[np.ndarray] = []
        load_gpu_slot_parts: list[np.ndarray] = []

        with self._layer_locks[layer_idx], profiler.record("attnpredict_offload_residency_diff"):
            for row_idx, positions in row_positions.items():
                row = int(row_idx)
                full_len = int(self.row_seq_lens[row])
                target = self._normalize_positions(positions, full_len)
                old = self._resident_array_locked(layer_idx, row, full_len)

                # 第一步：释放已经驻留 GPU、但下一步不再需要的 token。
                to_free = np.setdiff1d(old, target, assume_unique=True)
                if to_free.size:
                    slots = mirror[row, to_free].astype(np.int32, copy=False)
                    valid = slots >= 0
                    if np.any(valid):
                        to_free_valid = to_free[valid]
                        slots_valid = slots[valid].astype(np.int32, copy=True)
                        dirty = self._cpu_backing_dirty[layer_idx].get(row)
                        if dirty:
                            with profiler.record("attnpredict_offload_lazy_d2h_before_evict"):
                                dirty_pos = np.fromiter(dirty, dtype=np.int64)
                                idx = np.searchsorted(to_free_valid, dirty_pos)
                                in_free = (idx < to_free_valid.size) & (to_free_valid[idx.clip(max=max(to_free_valid.size - 1, 0))] == dirty_pos)
                                for pos in dirty_pos[in_free].tolist():
                                    slot = int(mirror[row, int(pos)])
                                    if slot >= 0:
                                        self._stage_dirty_cpu_backing_locked(layer_idx, row, int(pos), slot)
                        mirror[row, to_free_valid] = -1
                        self.free_slots_stack[layer_idx].extend(int(x) for x in slots_valid.tolist())
                        self._num_free_slots[layer_idx] += int(slots_valid.size)
                        free_rows_parts.append(np.full(to_free_valid.shape, row, dtype=np.int64))
                        free_pos_parts.append(to_free_valid.astype(np.int64, copy=False))
                        free_slot_parts.append(np.full(to_free_valid.shape, -1, dtype=np.int32))

                # 第二步：找出目标集合里还没有 GPU slot 的 positions。
                missing = np.setdiff1d(target, old, assume_unique=True)
                if missing.size:
                    gpu_slots = self._allocate_gpu_slots_locked(layer_idx, int(missing.size))
                    cpu_slots = self.cpu_req_to_token_slots[row, missing]
                    if np.any(cpu_slots < 0):
                        bad_pos = int(missing[np.flatnonzero(cpu_slots < 0)[0]])
                        raise RuntimeError(
                            f"Missing CPU backing slot: layer={layer_idx} row={row} pos={bad_pos}"
                        )
                    mirror[row, missing] = gpu_slots
                    load_rows_parts.append(np.full(missing.shape, row, dtype=np.int64))
                    load_pos_parts.append(missing.astype(np.int64, copy=False))
                    load_cpu_slot_parts.append(cpu_slots.astype(np.int64, copy=True))
                    load_gpu_slot_parts.append(gpu_slots.astype(np.int32, copy=True))

                self._set_resident_positions_locked(layer_idx, row, target, full_len)

        free_rows = self._concat_int_arrays(free_rows_parts, np.int64)
        free_pos = self._concat_int_arrays(free_pos_parts, np.int64)
        free_slots = self._concat_int_arrays(free_slot_parts, np.int32)
        load_rows = self._concat_int_arrays(load_rows_parts, np.int64)
        load_pos = self._concat_int_arrays(load_pos_parts, np.int64)
        load_cpu_slots = self._concat_int_arrays(load_cpu_slot_parts, np.int64)
        load_gpu_slots = self._concat_int_arrays(load_gpu_slot_parts, np.int32)

        # 第三步：把释放/加载后的 row/pos -> gpu_slot 变更写到 GPU 映射表，
        # 再把缺失 KV 从 CPU 搬到刚分配的 GPU slots。
        self._write_gpu_map(layer_idx, free_rows, free_pos, free_slots, stream=stream)
        self._write_gpu_map(layer_idx, load_rows, load_pos, load_gpu_slots, stream=stream)
        self._copy_cpu_to_gpu(layer_idx, load_cpu_slots, load_gpu_slots, stream=stream)

    def _ensure_positions_loaded(
        self,
        layer_idx: int,
        row_positions: dict[int, np.ndarray],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """只加载缺失 positions，不释放旧 resident token。

        refresh future 还没被主线程消费时，后台预取的新 hot set 可能已经写入
        residency map。此时主线程如果用旧 lease 做释放，可能把后台刚加载的
        新 hot token 释放掉。因此 pending refresh 期间使用 no-release 加载。
        """
        mirror = self.gpu_req_to_token_slots_cpu[layer_idx]
        load_rows_parts: list[np.ndarray] = []
        load_pos_parts: list[np.ndarray] = []
        load_cpu_slot_parts: list[np.ndarray] = []
        load_gpu_slot_parts: list[np.ndarray] = []

        with self._layer_locks[layer_idx], profiler.record("attnpredict_offload_load_diff"):
            for row_idx, positions in row_positions.items():
                row = int(row_idx)
                full_len = int(self.row_seq_lens[row])
                target = self._normalize_positions(positions, full_len)
                if row in self._resident_positions[layer_idx] and self._resident_positions[layer_idx][row] is None:
                    continue
                old = self._resident_positions[layer_idx].get(row)
                if old is None:
                    old = np.empty((0,), dtype=np.int64)
                missing = np.setdiff1d(target, old, assume_unique=True)
                if not missing.size:
                    continue

                gpu_slots = self._allocate_gpu_slots_locked(layer_idx, int(missing.size))
                cpu_slots = self.cpu_req_to_token_slots[row, missing]
                if np.any(cpu_slots < 0):
                    bad_pos = int(missing[np.flatnonzero(cpu_slots < 0)[0]])
                    raise RuntimeError(
                        f"Missing CPU backing slot: layer={layer_idx} row={row} pos={bad_pos}"
                    )
                mirror[row, missing] = gpu_slots
                load_rows_parts.append(np.full(missing.shape, row, dtype=np.int64))
                load_pos_parts.append(missing.astype(np.int64, copy=False))
                load_cpu_slot_parts.append(cpu_slots.astype(np.int64, copy=True))
                load_gpu_slot_parts.append(gpu_slots.astype(np.int32, copy=True))
                self._resident_positions[layer_idx][row] = np.union1d(old, target).astype(np.int64, copy=False)

        load_rows = self._concat_int_arrays(load_rows_parts, np.int64)
        load_pos = self._concat_int_arrays(load_pos_parts, np.int64)
        load_cpu_slots = self._concat_int_arrays(load_cpu_slot_parts, np.int64)
        load_gpu_slots = self._concat_int_arrays(load_gpu_slot_parts, np.int32)

        self._write_gpu_map(layer_idx, load_rows, load_pos, load_gpu_slots, stream=stream)
        self._copy_cpu_to_gpu(layer_idx, load_cpu_slots, load_gpu_slots, stream=stream)

    def _hot_positions_from_mask(self, layer_idx: int, row_idx: int, full_len: int) -> np.ndarray:
        """只取 predictor 选择的中间 hot positions，排除动态 sink/recent。"""
        pred_mask = self.tsp_mask[layer_idx][int(row_idx)]
        valid_len = min(int(pred_mask.numel()), int(full_len))
        if valid_len <= 0:
            return np.empty((0,), dtype=np.int64)

        middle_mask = pred_mask[:valid_len].clone()
        sink_end = min(int(self.sink_token), valid_len)
        recent_start = max(sink_end, valid_len - int(self.local_token))
        if sink_end > 0:
            middle_mask[:sink_end] = False
        if recent_start < valid_len:
            middle_mask[recent_start:] = False

        pos = middle_mask.nonzero(as_tuple=False).squeeze(-1)
        return pos.to(device="cpu", dtype=torch.long).numpy().astype(np.int64, copy=False)

    def _hot_positions_from_scores(
        self,
        tsp_attn: torch.Tensor,
        *,
        seq_len: int,
        start_block: int,
    ) -> np.ndarray:
        """直接从 block scores 生成中间 hot positions。

        语义等价于父类 _create_tsp_mask() 后再调用 _hot_positions_from_mask()：
        sink/recent 仍计入 num_top_tokens 预算，但这里不物化 128K bool mask。
        """
        seq_len = int(seq_len)
        if seq_len <= 0:
            return np.empty((0,), dtype=np.int64)

        pred_len = int(tsp_attn.shape[-1])
        if pred_len < 1:
            return np.empty((0,), dtype=np.int64)

        middle_budget = int(self.topk) - int(self.sink_token) - int(self.local_token)
        block_budget = middle_budget // max(1, int(self.pooling_block_size))
        block_budget = max(0, min(block_budget, pred_len))
        if block_budget <= 0:
            return np.empty((0,), dtype=np.int64)

        block_scores = tsp_attn.max(dim=0).values if tsp_attn.dim() == 2 else tsp_attn
        _, topk_indices = torch.topk(block_scores, block_budget, dim=-1)
        token_indices = (
            (topk_indices + int(start_block)).unsqueeze(-1) * int(self.pooling_block_size)
            + torch.arange(int(self.pooling_block_size), device=tsp_attn.device)
        ).reshape(-1)

        sink_end = min(int(self.sink_token), seq_len)
        recent_start = max(sink_end, seq_len - int(self.local_token))
        valid = (token_indices >= sink_end) & (token_indices < recent_start)
        if not bool(valid.any()):
            return np.empty((0,), dtype=np.int64)
        token_indices = token_indices[valid]
        token_indices = torch.unique(token_indices, sorted=True)
        return token_indices.to(device="cpu", dtype=torch.long).numpy().astype(np.int64, copy=False)

    def _compose_positions_from_hot(
        self,
        hot_positions: np.ndarray | None,
        full_len: int,
        *,
        current_pos: int | None = None,
    ) -> np.ndarray:
        """动态拼接 sink + leased hot + recent + current。"""
        full_len = int(full_len)
        if full_len <= 0:
            return np.empty((0,), dtype=np.int64)

        parts: list[np.ndarray] = []
        sink_end = min(int(self.sink_token), full_len)
        if sink_end > 0:
            parts.append(np.arange(sink_end, dtype=np.int64))

        if hot_positions is not None and len(hot_positions) > 0:
            hot = np.asarray(hot_positions, dtype=np.int64)
            hot = hot[(hot >= 0) & (hot < full_len)]
            if hot.size:
                parts.append(hot)

        local = int(self.local_token)
        if local > 0:
            recent_start = max(sink_end, full_len - local)
            if recent_start < full_len:
                parts.append(np.arange(recent_start, full_len, dtype=np.int64))

        if current_pos is not None and 0 <= int(current_pos) < full_len:
            parts.append(np.asarray([int(current_pos)], dtype=np.int64))

        if not parts:
            return np.empty((0,), dtype=np.int64)
        return np.unique(np.concatenate(parts).astype(np.int64, copy=False))

    def _recent_positions_for_decode(
        self,
        row_idx: int,
        full_len: int,
        current_pos: int | None,
    ) -> np.ndarray:
        """返回 decode 动态尾部窗口，包含 current token。"""
        full_len = int(full_len)
        if full_len <= 0:
            return np.empty((0,), dtype=np.int64)

        sink_end = min(int(self.sink_token), full_len)
        local = int(self.local_token)
        parts: list[np.ndarray] = []
        if local > 0:
            recent_start = max(sink_end, full_len - local)
            if recent_start < full_len:
                parts.append(np.arange(recent_start, full_len, dtype=np.int64))

        if current_pos is not None and 0 <= int(current_pos) < full_len:
            cur = int(current_pos)
            if not parts or parts[-1].size == 0 or int(parts[-1][-1]) != cur:
                parts.append(np.asarray([cur], dtype=np.int64))

        if not parts:
            return np.empty((0,), dtype=np.int64)
        if len(parts) == 1:
            return parts[0]
        return np.unique(np.concatenate(parts).astype(np.int64, copy=False))

    def _get_full_resident_prefix_cache(
        self,
        layer_idx: int,
        row_idx: int,
        full_len: int,
        device: torch.device,
    ) -> dict[str, object]:
        """缓存 full-resident sparse view 中不随 decode step 滑动的部分。

        prefix = sink + leased middle hot positions。recent/current 每步变化，
        因此不放进这个缓存。
        """
        row = int(row_idx)
        version = int(self._lease_versions[layer_idx].get(row, 0))
        cache = self._full_resident_prefix_cache[layer_idx].get(row)
        if (
            cache is not None
            and int(cache.get("version", -1)) == version
            and cache.get("device") == device
        ):
            return cache

        sink_end = min(int(self.sink_token), int(full_len))
        parts: list[np.ndarray] = []
        if sink_end > 0:
            parts.append(np.arange(sink_end, dtype=np.int64))

        hot = self._leased_hot_positions[layer_idx].get(row)
        if hot is not None and len(hot) > 0:
            hot_arr = np.asarray(hot, dtype=np.int64)
            hot_arr = hot_arr[(hot_arr >= sink_end) & (hot_arr < int(full_len))]
            if hot_arr.size:
                parts.append(hot_arr)

        if parts:
            prefix_positions = np.concatenate(parts).astype(np.int64, copy=False)
            if prefix_positions.size > 1 and np.any(prefix_positions[1:] <= prefix_positions[:-1]):
                prefix_positions = np.unique(prefix_positions)
        else:
            prefix_positions = np.empty((0,), dtype=np.int64)

        mirror = self.gpu_req_to_token_slots_cpu[layer_idx]
        if prefix_positions.size:
            prefix_slots = mirror[row, prefix_positions].astype(np.int32, copy=True)
            if np.any(prefix_slots < 0):
                raise RuntimeError(
                    f"Full-resident prefix has nonresident slots at layer={layer_idx} row={row}"
                )
            prefix_positions_t = torch.as_tensor(prefix_positions, dtype=torch.int32, device=device)
            prefix_slots_t = torch.as_tensor(prefix_slots, dtype=torch.int32, device=device)
        else:
            prefix_positions_t = torch.empty((0,), dtype=torch.int32, device=device)
            prefix_slots_t = torch.empty((0,), dtype=torch.int32, device=device)

        cache = {
            "version": version,
            "device": device,
            "positions_np": prefix_positions,
            "positions": prefix_positions_t,
            "slots": prefix_slots_t,
        }
        self._full_resident_prefix_cache[layer_idx][row] = cache
        return cache

    def _make_full_resident_decode_plan(
        self,
        layer_idx: int,
        rows: np.ndarray,
    ) -> list[dict[str, object]] | None:
        """为 full-resident decode 构造轻量打包计划。

        返回 None 表示出现 prefix/recent 重叠等边界情况，需要回退到通用路径。
        """
        plans: list[dict[str, object]] = []
        device = torch.device("cuda")
        for row_idx in rows.tolist():
            row = int(row_idx)
            full_len = int(self.row_seq_lens[row])
            prefix = self._get_full_resident_prefix_cache(layer_idx, row, full_len, device)
            version = int(self._lease_versions[layer_idx].get(row, 0))
            recent_np = self._decode_recent_positions_np.get(row)
            recent_t = self._decode_recent_positions_cuda.get(row)
            if recent_np is None or recent_t is None:
                return None

            prefix_np = prefix["positions_np"]
            if (
                isinstance(prefix_np, np.ndarray)
                and prefix_np.size > 0
                and recent_np.size > 0
                and int(recent_np[0]) <= int(prefix_np[-1])
            ):
                return None

            plans.append(
                {
                    "row": row,
                    "version": version,
                    "full_len": full_len,
                    "prefix": prefix,
                    "recent_np": recent_np,
                    "recent": recent_t,
                    "keep": int(prefix["positions"].numel()) + int(recent_t.numel()),
                }
            )
        return plans

    def _commit_hot_leases(
        self,
        layer_idx: int,
        hot_positions: dict[int, np.ndarray],
        lease_start_positions: dict[int, int],
    ) -> None:
        for row_idx, hot in hot_positions.items():
            row = int(row_idx)
            self._leased_hot_positions[layer_idx][row] = np.asarray(hot, dtype=np.int64)
            start_pos = int(lease_start_positions[row])
            self._lease_start_positions[layer_idx][row] = start_pos
            self._last_refresh_positions[layer_idx][row] = start_pos
            self._lease_versions[layer_idx][row] = int(self._lease_versions[layer_idx].get(row, 0)) + 1
            self._full_resident_prefix_cache[layer_idx].pop(row, None)

    def should_collect_decode_attn_score(self, layer_idx: int) -> bool:
        """每 reuse_steps 步触发一次 refresh，其余 step 复用当前 lease。"""
        if self._decode_rows is None or self._decode_current_positions is None:
            return True
        if self._prefetch_futures[layer_idx] is not None:
            return False
        if self._reuse_steps <= 1:
            return True

        for row_idx, cur_pos in zip(self._decode_rows.tolist(), self._decode_current_positions.tolist()):
            start = self._lease_start_positions[layer_idx].get(int(row_idx))
            if start is None or int(cur_pos) - int(start) + 1 >= self._reuse_steps:
                return True
        return False

    def decode_attn_score_max_len(self, layer_idx: int, context_lens: torch.Tensor) -> int:
        """offload decode 的 attn_score 只需要覆盖 packed sparse view 宽度。"""
        full_max_len = int(context_lens.max().item())
        middle_budget = max(0, int(self.topk) - int(self.sink_token) - int(self.local_token))
        block_budget = middle_budget // max(1, int(self.pooling_block_size))
        keep_bound = int(self.sink_token) + int(self.local_token) + block_budget * int(self.pooling_block_size) + 1
        return max(1, min(full_max_len, keep_bound))

    def _consume_prefetch(self, layer_idx: int, *, wait: bool) -> bool:
        future = self._prefetch_futures[layer_idx]
        if future is None:
            return False
        if not wait and not future.done():
            return False

        record_name = "attnpredict_offload_prefetch_wait" if wait else "attnpredict_offload_prefetch_poll"
        with profiler.record(record_name):
            result = future.result()

        if isinstance(result, dict):
            event = result.get("event")
            if event is not None:
                torch.cuda.current_stream().wait_event(event)
            layer_results = result.get("layer_results")
            if layer_results is not None:
                for commit_layer_idx, layer_result in layer_results.items():
                    hot_positions = layer_result.get("hot_positions")
                    lease_start_positions = layer_result.get("lease_start_positions")
                    if hot_positions is not None and lease_start_positions is not None:
                        self._commit_hot_leases(
                            int(commit_layer_idx),
                            hot_positions,
                            lease_start_positions,
                        )
                for i, known_future in enumerate(self._prefetch_futures):
                    if known_future is future:
                        self._prefetch_futures[i] = None
                return True

            self._prefetch_futures[layer_idx] = None
            hot_positions = result.get("hot_positions")
            lease_start_positions = result.get("lease_start_positions")
            if hot_positions is not None and lease_start_positions is not None:
                self._commit_hot_leases(layer_idx, hot_positions, lease_start_positions)
        elif result is not None:
            self._prefetch_futures[layer_idx] = None
            torch.cuda.current_stream().wait_event(result)
        else:
            self._prefetch_futures[layer_idx] = None
        return True

    def _wait_prefetch(self, layer_idx: int) -> None:
        """等待指定层上一轮异步预取完成。

        只在“下一次同层真正要消费 KV”时等待，而不是在提交预取后立刻等待。
        这就是 offload 版本 overlap 的关键：主线程可以先继续算后续层。
        """
        self._consume_prefetch(layer_idx, wait=True)

    def get_layer_store_view(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回当前层写入 K/V 的目标 cache 和 slot_mapping。

        这是 offload decode 的“按层消费点”：
        - prefill：所有 token 已经在 _prepare_prefill 中分配好 GPU slots，直接返回。
        - decode：到达某一层时，非阻塞消费已完成的 refresh；未完成则继续
          使用旧 hot lease；最后动态拼接 sink/recent/current 并分配当前 token slot。

        这样不会在 decode step 一开始就等待所有层的预测/预取，而是让前面层
        的计算和后面层的后台 refresh 自然 overlap。
        """
        ctx = get_context()
        if not ctx.is_prefill:
            with profiler.record("attnpredict_offload_get_layer_store_view_decode"):
                if (
                    self._decode_full_resident_layers[layer_idx]
                    and self._prefetch_futures[layer_idx] is None
                    and self._decode_rows is not None
                    and self._decode_current_slots_cuda is not None
                ):
                    with profiler.record("attnpredict_offload_full_resident_fast_path"):
                        state = self.layer_batch_states[layer_idx]
                        state.slot_mapping = self._decode_current_slots_cuda[layer_idx]
                        self._decode_fast_view_plans[layer_idx] = self._make_full_resident_decode_plan(
                            layer_idx,
                            self._decode_rows,
                        )
                    k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
                    return k_cache, v_cache, state.slot_mapping

                # refresh 完成则切换到新 lease；未完成时继续用旧 lease。只有超过
                # stale 上限或没有任何可用 lease 时，才阻塞等待。
                must_wait = False
                if self._prefetch_futures[layer_idx] is not None:
                    for row_idx, cur_pos in zip(self._decode_rows.tolist(), self._decode_current_positions.tolist()):
                        start = self._lease_start_positions[layer_idx].get(int(row_idx))
                        if start is None or int(cur_pos) - int(start) + 1 > self._max_stale_steps:
                            must_wait = True
                            break
                self._consume_prefetch(layer_idx, wait=must_wait)
                refresh_pending = self._prefetch_futures[layer_idx] is not None

                state = self.layer_batch_states[layer_idx]

                rows = self._decode_rows
                current_positions = self._decode_current_positions

                with profiler.record("attnpredict_offload_decode_residency"):
                    full_resident = bool(self._decode_full_resident_layers[layer_idx])

                if full_resident:
                    # 当前 benchmark 的典型情况：active pool 足够放完整 KV。
                    # 这里不做 eviction/reload，也不重写 GPU row->pos map；decode
                    # kernel 实际消费的是 build_decode_view() 返回的 packed slots。
                    with profiler.record("attnpredict_offload_full_resident_fast_path"):
                        if (
                            self._decode_current_slots_np is not None
                            and self._decode_current_slots_cuda is not None
                            and int(self._decode_current_slots_np.shape[0]) > layer_idx
                        ):
                            state.slot_mapping = self._decode_current_slots_cuda[layer_idx]
                        else:
                            current_slots = self._ensure_current_decode_slots(
                                layer_idx,
                                rows,
                                current_positions,
                                write_gpu_map=False,
                            )
                            state.slot_mapping = self._decode_slot_mapping_tensor(
                                layer_idx,
                                current_slots,
                                device=torch.device("cuda"),
                            )
                        self._decode_fast_view_plans[layer_idx] = self._make_full_resident_decode_plan(
                            layer_idx,
                            rows,
                        )
                    if self._decode_fast_view_plans[layer_idx] is not None:
                        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
                        return k_cache, v_cache, state.slot_mapping

                # 当前 token 必须加入本层可见 view。上一轮 predictor 无法提前预测当前 token，
                # 但原始 AttentionPredictor decode 也会把 newest KV 拼进去。
                view_positions: list[np.ndarray] = []
                row_positions: dict[int, np.ndarray] = {}
                self._decode_fast_view_plans[layer_idx] = None
                with profiler.record("attnpredict_offload_compose_positions"):
                    for row_idx, cur_pos in zip(rows.tolist(), current_positions.tolist()):
                        row = int(row_idx)
                        positions = self._compose_positions_from_hot(
                            self._leased_hot_positions[layer_idx].get(row),
                            int(self.row_seq_lens[row]),
                            current_pos=int(cur_pos),
                        )
                        view_positions.append(positions)
                        # 当前 token 的 CPU backing 还没写入，不能通过
                        # _ensure_positions_* 从 CPU 加载；它稍后由
                        # _ensure_current_decode_slots 分配 GPU slot 并由
                        # store_kvcache 写入真实 K/V。
                        row_positions[row] = positions[positions != int(cur_pos)]

                with profiler.record("attnpredict_offload_decode_residency"):
                    if refresh_pending and not self._has_full_resident_rows(layer_idx, rows):
                        self._ensure_positions_loaded(layer_idx, row_positions, stream=None)
                    else:
                        self._ensure_positions_resident(layer_idx, row_positions, stream=None)
                self._decode_view_positions[layer_idx] = view_positions

                # 当前 decode token 刚在 _prepare_decode() 中分配了 CPU slot，
                # 但还没有本层 GPU slot。非 full-resident 分支上面会先释放旧
                # resident 再分配缺失 token，避免 active pool 边界处先分配再 OOM。
                with profiler.record("attnpredict_offload_current_decode_slots"):
                    current_slots = self._ensure_current_decode_slots(layer_idx, rows, current_positions) # 为当前 decode token 分配 GPU slot，并返回本层所有decode token 的 GPU slot 列表
                    state.slot_mapping = self._decode_slot_mapping_tensor(
                        layer_idx,
                        current_slots,
                        device=torch.device("cuda"),
                    )

        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        return k_cache, v_cache, self.layer_batch_states[layer_idx].slot_mapping

    def _decode_slot_mapping_tensor(
        self,
        layer_idx: int,
        slots: list[int],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = len(slots)
        buf = self._decode_current_slot_mapping_buf[layer_idx]
        if buf is None or buf.device != device or buf.numel() < batch_size:
            buf = torch.empty((batch_size,), dtype=torch.int32, device=device)
            self._decode_current_slot_mapping_buf[layer_idx] = buf
        out = buf[:batch_size]
        if batch_size == 1:
            out[0] = int(slots[0])
        else:
            out.copy_(torch.as_tensor(slots, dtype=torch.int32, device=device))
        return out

    def _ensure_current_decode_slots(
        self,
        layer_idx: int,
        rows: np.ndarray,
        positions: np.ndarray,
        *,
        write_gpu_map: bool = True,
    ) -> list[int]:
        """确保当前 decode token 在本层有 GPU active slot。

        decode 新 token 是逐层写入的：同一个 token 在每一层的 K/V 不同，
        因此每层走到 attention 前，才为该层分配 GPU slot 并写入当前 K/V。
        """
        mirror = self.gpu_req_to_token_slots_cpu[layer_idx]
        update_rows: list[int] = []
        update_pos: list[int] = []
        update_slots: list[int] = []
        result: list[int] = []
        with self._layer_locks[layer_idx]:
            for row_idx, pos in zip(rows.tolist(), positions.tolist()):
                slot = int(mirror[int(row_idx), int(pos)])
                if slot < 0:
                    slot = int(self._allocate_gpu_slots_locked(layer_idx, 1)[0])
                    mirror[int(row_idx), int(pos)] = slot
                    if write_gpu_map:
                        update_rows.append(int(row_idx))
                        update_pos.append(int(pos))
                        update_slots.append(slot)
                self._add_resident_position_locked(layer_idx, int(row_idx), int(pos))
                result.append(slot)
        if write_gpu_map:
            self._write_gpu_map(layer_idx, update_rows, update_pos, update_slots, stream=None)
        return result

    def on_kv_stored(
        self,
        layer_idx: int,
        k: torch.Tensor,
        slot_mapping: torch.Tensor,
        v: torch.Tensor | None = None,
    ):
        """GPU KV 写入后，把同一批 K/V 同步保存到 CPU full backing。

        attention.py 调用 store_kvcache() 后会立刻调用这个 hook。对于 offload，
        CPU backing 是恢复被驱逐 token 的唯一来源，所以 K 和 V 都必须保存。
        """
        cpu_slots_np = self._layer_cpu_slot_mapping[layer_idx]
        ctx = get_context()
        if not ctx.is_prefill:
            with profiler.record("attnpredict_offload_mark_lazy_cpu_backing"):
                rows = self._decode_rows
                positions = self._decode_current_positions
                dirty = self._cpu_backing_dirty[layer_idx]
                for row_idx, pos in zip(rows.tolist(), positions.tolist()):
                    dirty.setdefault(int(row_idx), set()).add(int(pos))
            return

        with profiler.record("attnpredict_offload_store_cpu_full_kv"):
            # cpu_slots_np 与本轮写入的 k/v 顺序对齐：TODO:这里会不会很耗时：代码在 prefill 阶段确实会很费时间，是 TTFT 变差的重要嫌疑
            # prefill 是 chunk 内所有 token；decode 是 batch 内每个 seq 的当前 token。
            with profiler.record("attnpredict_offload_store_cpu_slots_tensor"):
                cpu_slots = torch.as_tensor(cpu_slots_np, dtype=torch.long, device="cpu")
            with profiler.record("attnpredict_offload_store_d2h_kv"):
                host_k = k.detach().to(device="cpu", dtype=self.hf_config.torch_dtype)
                host_v = v.detach().to(device="cpu", dtype=self.hf_config.torch_dtype)
            with profiler.record("attnpredict_offload_store_cpu_index_copy"):
                self.cpu_kv_cache[0, layer_idx].index_copy_(0, cpu_slots, host_k) # 沿着第0维（slot维度），将host_k的内容根据slot位置cpu_slots复制到cpu_kv_cache的对应位置
                self.cpu_kv_cache[1, layer_idx].index_copy_(0, cpu_slots, host_v)

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
        view = self._make_prefill_tail_score_view(
            q=q,
            req_indices=req_indices,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens_q,
            num_heads=num_heads,
            prefill_is_last_chunk=prefill_is_last_chunk,
        )
        if view is None:
            self._pending_prefill_views[layer_idx] = None
            return None

        self._pending_prefill_views[layer_idx] = view
        return view["tail_score"]

    def prefill_attn_score_block_size(self, layer_idx: int) -> int | None:
        """4D prefill tail-score buffers are block-level for offload."""
        return int(self.pooling_block_size)

    def _make_prefill_tail_score_view(
        self,
        *,
        q: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        cu_seqlens_q: torch.Tensor | None,
        num_heads: int,
        prefill_is_last_chunk: list[bool] | None,
    ) -> dict[str, object] | None:
        if cu_seqlens_q is None or cu_seqlens_q.numel() <= 1:
            return None
        if prefill_is_last_chunk is None:
            return None
        last_flags = [bool(x) for x in prefill_is_last_chunk]
        if not any(last_flags):
            return None

        batch_size = int(req_indices.numel())
        history = int(self.history_step)
        max_len = int(context_lens.max().item())
        block_size = int(self.pooling_block_size)
        if block_size <= 0:
            raise ValueError("attnpredict_pooling_block_size must be > 0")
        pooled_len = (max_len + block_size - 1) // block_size
        tail_score = torch.full(
            (batch_size, int(num_heads), history, pooled_len),
            0.0,
            dtype=torch.float32,
            device=q.device,
        )
        return {
            "tail_score": tail_score,
            "req_indices": req_indices.detach().to(device="cpu", dtype=torch.long).numpy().astype(np.int64, copy=True),
            "context_lens": context_lens.detach().to(device="cpu", dtype=torch.long).numpy().astype(np.int64, copy=True),
            "cu_seqlens_q": cu_seqlens_q.detach().to(device="cpu", dtype=torch.long).numpy().astype(np.int64, copy=True),
            "prefill_is_last_chunk": last_flags,
            "num_heads": int(num_heads),
            "block_size": block_size,
            "pooled_len": pooled_len,
        }

    def _predict_prefill_positions_sync(
        self,
        layer_idx: int,
        view: dict[str, object],
    ) -> dict[str, dict[int, np.ndarray] | dict[int, int]]:
        tail_score = view["tail_score"]
        req_indices = view["req_indices"]
        context_lens = view["context_lens"]
        cu_seqlens_q = view["cu_seqlens_q"]
        prefill_is_last_chunk = view["prefill_is_last_chunk"]
        block_size = int(view["block_size"])
        pooled_len = int(view["pooled_len"])

        hot_positions: dict[int, np.ndarray] = {}
        lease_start_positions: dict[int, int] = {}
        for b, row_idx in enumerate(req_indices.tolist()):
            if not prefill_is_last_chunk[b]:
                continue

            q_end = int(cu_seqlens_q[b + 1])
            q_start = int(cu_seqlens_q[b])
            q_len = q_end - q_start
            full_len = int(context_lens[b])
            if q_len <= 0 or full_len <= 0:
                continue

            take = min(self.history_step, q_len)
            active_pooled_len = min(pooled_len, (full_len + block_size - 1) // block_size)
            attn_pooling = tail_score[b, :, :take, :active_pooled_len].to(self.hf_config.torch_dtype)

            self._update_row_prediction_from_pooled(
                layer_idx,
                int(row_idx),
                attn_pooling,
                seq_len=full_len,
            )
            hot_positions[int(row_idx)] = self._hot_positions_from_mask(layer_idx, int(row_idx), full_len)
            lease_start_positions[int(row_idx)] = full_len
        return {
            "hot_positions": hot_positions,
            "lease_start_positions": lease_start_positions,
        }

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
            fast_plan = self._decode_fast_view_plans[layer_idx]
            if fast_plan is not None:
                return self._build_decode_view_from_fast_plan(
                    layer_idx,
                    q,
                    active_slots,
                    req_indices,
                    context_lens,
                    fast_plan,
                )

            view_positions = self._decode_view_positions[layer_idx]

            batch_size = int(req_indices.numel())
            keep_counts = [int(len(p)) for p in view_positions]
            max_keep = max(keep_counts)
            if max_keep <= 0:
                self._last_decode_view[layer_idx] = None
                return active_slots, req_indices, context_lens

            slot_buf = self._decode_packed_slots[layer_idx]
            if (
                slot_buf is None
                or slot_buf.device != q.device
                or slot_buf.shape[0] < batch_size
                or slot_buf.shape[1] < max_keep
            ):
                slot_buf = torch.empty((batch_size, max_keep), dtype=torch.int32, device=q.device)
                self._decode_packed_slots[layer_idx] = slot_buf
            pos_buf = self._decode_packed_positions[layer_idx]
            if (
                pos_buf is None
                or pos_buf.device != q.device
                or pos_buf.shape[0] < batch_size
                or pos_buf.shape[1] < max_keep
            ):
                pos_buf = torch.empty((batch_size, max_keep), dtype=torch.int32, device=q.device)
                self._decode_packed_positions[layer_idx] = pos_buf

            packed_slots = slot_buf[:batch_size, :max_keep]
            packed_positions = pos_buf[:batch_size, :max_keep]
            rows = self._decode_rows
            mirror = self.gpu_req_to_token_slots_cpu[layer_idx]
            with profiler.record("attnpredict_offload_pack_decode_slots"):
                for b, (row_idx, positions) in enumerate(zip(rows.tolist(), view_positions)):
                    # positions 在 get_layer_store_view() 中已经确保 resident，这里只做查表和打包。
                    row = int(row_idx)
                    k = int(len(positions))
                    slots_np = mirror[row, positions].astype(np.int32, copy=True)
                    if np.any(slots_np < 0):
                        raise RuntimeError(f"AttnPredict offload decode view has nonresident slots at layer {layer_idx}.")
                    packed_positions[b, :k] = torch.as_tensor(positions, dtype=torch.int32, device=q.device)
                    packed_slots[b, :k] = torch.as_tensor(slots_np, dtype=torch.int32, device=q.device)

            keep_counts_np = np.asarray(keep_counts, dtype=np.int32)
            view_lens_buf = self._decode_view_lens_buf[layer_idx]
            if view_lens_buf is None or view_lens_buf.device != q.device or view_lens_buf.numel() < batch_size:
                view_lens_buf = torch.empty((batch_size,), dtype=torch.int32, device=q.device)
                self._decode_view_lens_buf[layer_idx] = view_lens_buf
            view_lens = view_lens_buf[:batch_size]
            view_lens.copy_(torch.as_tensor(keep_counts_np, dtype=torch.int32, device=q.device))

            local_req_indices = self._decode_local_req_indices[layer_idx]
            if local_req_indices is None or local_req_indices.device != q.device or local_req_indices.numel() < batch_size:
                local_req_indices = torch.arange(batch_size, dtype=torch.int32, device=q.device)
                self._decode_local_req_indices[layer_idx] = local_req_indices
            local_req_indices = local_req_indices[:batch_size]

            # 保存本次 decode view，attention kernel 会把 logits 写到 attn_score；
            # positions 必须 clone，因为 packed buffer 会在后续 decode step 复用。
            if self._current_layer_collects_decode_score(layer_idx):
                self._last_decode_view[layer_idx] = {
                    "req_indices": rows.astype(np.int64, copy=True),
                    "positions": packed_positions.detach().clone(),
                    "view_lens": keep_counts_np.copy(),
                    "full_context_lens": self.row_seq_lens[rows].astype(np.int32, copy=True),
                }
            else:
                self._last_decode_view[layer_idx] = None
            return packed_slots, local_req_indices, view_lens

    def _current_layer_collects_decode_score(self, layer_idx: int) -> bool:
        ctx = get_context()
        if ctx.is_prefill:
            return False
        sparse_controller = getattr(ctx, "sparse_controller", None)
        if sparse_controller is None:
            return False
        state = sparse_controller.layer_batch_sparse_states.get(layer_idx)
        return bool(state is not None and state.attn_score is not None)

    def _decode_buffers(
        self,
        layer_idx: int,
        *,
        batch_size: int,
        max_keep: int,
        device: torch.device,
        need_positions: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        slot_buf = self._decode_packed_slots[layer_idx]
        if (
            slot_buf is None
            or slot_buf.device != device
            or slot_buf.shape[0] < batch_size
            or slot_buf.shape[1] < max_keep
        ):
            slot_buf = torch.empty((batch_size, max_keep), dtype=torch.int32, device=device)
            self._decode_packed_slots[layer_idx] = slot_buf
        packed_slots = slot_buf[:batch_size, :max_keep]

        packed_positions = None
        if need_positions:
            pos_buf = self._decode_packed_positions[layer_idx]
            if (
                pos_buf is None
                or pos_buf.device != device
                or pos_buf.shape[0] < batch_size
                or pos_buf.shape[1] < max_keep
            ):
                pos_buf = torch.empty((batch_size, max_keep), dtype=torch.int32, device=device)
                self._decode_packed_positions[layer_idx] = pos_buf
            packed_positions = pos_buf[:batch_size, :max_keep]

        view_lens_buf = self._decode_view_lens_buf[layer_idx]
        if view_lens_buf is None or view_lens_buf.device != device or view_lens_buf.numel() < batch_size:
            view_lens_buf = torch.empty((batch_size,), dtype=torch.int32, device=device)
            self._decode_view_lens_buf[layer_idx] = view_lens_buf
        view_lens = view_lens_buf[:batch_size]

        local_req_indices = self._decode_local_req_indices[layer_idx]
        if local_req_indices is None or local_req_indices.device != device or local_req_indices.numel() < batch_size:
            local_req_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
            self._decode_local_req_indices[layer_idx] = local_req_indices
        local_req_indices = local_req_indices[:batch_size]
        return packed_slots, packed_positions, view_lens, local_req_indices

    def _build_decode_view_from_fast_plan(
        self,
        layer_idx: int,
        q: torch.Tensor,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        fast_plan: list[dict[str, object]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(req_indices.numel())
        keep_counts = [int(plan["keep"]) for plan in fast_plan]
        max_keep = max(keep_counts) if keep_counts else 0
        if max_keep <= 0:
            self._last_decode_view[layer_idx] = None
            return active_slots, req_indices, context_lens

        need_positions = self._current_layer_collects_decode_score(layer_idx)
        packed_slots, packed_positions, view_lens, local_req_indices = self._decode_buffers(
            layer_idx,
            batch_size=batch_size,
            max_keep=max_keep,
            device=q.device,
            need_positions=need_positions,
        )

        mirror = self.gpu_req_to_token_slots_cpu[layer_idx]
        prefix_meta = self._decode_fast_prefix_meta[layer_idx]
        if prefix_meta is None or len(prefix_meta) < batch_size:
            prefix_meta = [{} for _ in range(batch_size)]
            self._decode_fast_prefix_meta[layer_idx] = prefix_meta
        packed_ptr = int(packed_slots.data_ptr())
        for b, plan in enumerate(fast_plan):
            prefix = plan["prefix"]
            prefix_slots = prefix["slots"]
            prefix_len = int(prefix_slots.numel())
            recent_np = plan["recent_np"]
            recent_t = plan["recent"]
            recent_len = int(recent_t.numel())
            keep = prefix_len + recent_len

            meta = prefix_meta[b]
            prefix_is_cached = (
                int(meta.get("ptr", -1)) == packed_ptr
                and int(meta.get("row", -1)) == int(plan["row"])
                and int(meta.get("version", -1)) == int(plan["version"])
                and int(meta.get("prefix_len", -1)) == prefix_len
            )
            if prefix_len and not prefix_is_cached:
                packed_slots[b, :prefix_len].copy_(prefix_slots)
                meta.update(
                    {
                        "ptr": packed_ptr,
                        "row": int(plan["row"]),
                        "version": int(plan["version"]),
                        "prefix_len": prefix_len,
                    }
                )
            if prefix_len and need_positions:
                # Slot prefix can be cached across non-refresh decode steps, but
                # positions are only materialized on refresh steps. Always refill
                # them here so predictor history never sees stale buffer values.
                packed_positions[b, :prefix_len].copy_(prefix["positions"])

            if recent_len:
                row = int(plan["row"])
                recent_start = int(recent_np[0]) if recent_np.size else -1
                can_slide_recent = (
                    not need_positions
                    and prefix_is_cached
                    and recent_len > 1
                    and int(meta.get("recent_len", -1)) == recent_len
                    and int(meta.get("recent_start", -10**18)) + 1 == recent_start
                    and int(meta.get("prefix_len", -1)) == prefix_len
                )
                if can_slide_recent:
                    # 非 refresh step 不需要 positions；decode attention 对 KV 顺序
                    # 置换不变，因此 recent 段可用 ring-buffer 替换最老 slot。
                    ring_pos = int(meta.get("recent_ring_pos", 0)) % recent_len
                    if self._decode_current_slots_cuda is not None:
                        packed_slots[b, prefix_len + ring_pos].copy_(
                            self._decode_current_slots_cuda[layer_idx, b]
                        )
                    else:
                        current_slot = int(mirror[row, int(recent_np[-1])])
                        if current_slot < 0:
                            raise RuntimeError(
                                f"Full-resident fast decode view has nonresident current slot at layer {layer_idx}."
                            )
                        packed_slots[b, prefix_len + ring_pos] = current_slot
                    meta["recent_ring_pos"] = (ring_pos + 1) % recent_len
                else:
                    recent_slots_np = mirror[row, recent_np].astype(np.int32, copy=True)
                    if np.any(recent_slots_np < 0):
                        raise RuntimeError(
                            f"Full-resident fast decode view has nonresident recent slots at layer {layer_idx}."
                        )
                    packed_slots[b, prefix_len:keep] = torch.as_tensor(
                        recent_slots_np,
                        dtype=torch.int32,
                        device=q.device,
                    )
                    if need_positions:
                        packed_positions[b, prefix_len:keep].copy_(recent_t)
                    else:
                        meta["recent_ring_pos"] = 0
                meta["recent_start"] = recent_start
                meta["recent_len"] = recent_len

        keep_counts_np = np.asarray(keep_counts, dtype=np.int32)
        if batch_size == 1:
            view_lens[0] = int(keep_counts_np[0])
        else:
            view_lens.copy_(torch.as_tensor(keep_counts_np, dtype=torch.int32, device=q.device))

        if need_positions:
            self._last_decode_view[layer_idx] = {
                "req_indices": np.asarray([int(plan["row"]) for plan in fast_plan], dtype=np.int64),
                "positions": packed_positions.detach().clone(),
                "view_lens": keep_counts_np.copy(),
                "full_context_lens": np.asarray(
                    [int(plan["full_len"]) for plan in fast_plan],
                    dtype=np.int32,
                ),
            }
        else:
            self._last_decode_view[layer_idx] = None
        return packed_slots, local_req_indices, view_lens

    def predict_next_mask(self, layer_idx: int, attn_logits: torch.Tensor) -> None:
        """根据当前层刚算完的 attention logits，预测下一步 active set。

        这个函数由 SparseController.on_attention_end() 调用。开启异步预取时，
        它只记录当前 CUDA stream 上的 event，然后把真正的 softmax/CNN/CPU gather/H2D
        放到后台线程和本层 prefetch stream 里执行。
        """
        view = self._last_decode_view[layer_idx]
        if view is None:
            return
        if not self._prefetch_enabled:
            # 调试/保守路径：主线程同步完成预测和 residency 更新。
            with torch.inference_mode():
                result = self._predict_next_positions_sync(layer_idx, attn_logits, view)
                hot_positions = result["hot_positions"]
                lease_start_positions = result["lease_start_positions"]
                self._commit_hot_leases(layer_idx, hot_positions, lease_start_positions)
                row_positions = {
                    int(row_idx): self._compose_positions_from_hot(hot_positions.get(int(row_idx)), int(full_len))
                    for row_idx, full_len in lease_start_positions.items()
                }
            if self._can_keep_full_resident_for_rows(layer_idx, lease_start_positions.keys()):
                with profiler.record("attnpredict_offload_full_resident_fast_path"):
                    return
            self._ensure_positions_resident(layer_idx, row_positions, stream=None)
            return

        # 如果上一轮 refresh 还没消费完，继续用旧 lease；不要提交重叠任务。
        if self._prefetch_futures[layer_idx] is not None:
            return

        # 记录 attention 所在主 stream 的完成事件。decode refresh 在本 step
        # 结束时批量提交，把多层 CNN 小 kernel 合并成一次更饱满的 batch。
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self._pending_decode_predict_inputs.append(
            {
                "layer_idx": int(layer_idx),
                "attn_logits": attn_logits.detach(),
                "view": dict(view),
                "event": event,
            }
        )

    def on_decode_step_end(self) -> None:
        """在一个 decode step 结束时批量提交本轮需要 refresh 的层。"""
        if not self._pending_decode_predict_inputs:
            return
        pending = self._pending_decode_predict_inputs
        self._pending_decode_predict_inputs = []
        if not self._prefetch_enabled:
            return

        # 只提交当前没有未消费 future 的层；理论上 should_collect_decode_attn_score()
        # 已经保证这一点，这里保守过滤，避免输出结束/释放等边界造成重叠任务。
        runnable = [
            item
            for item in pending
            if self._prefetch_futures[int(item["layer_idx"])] is None
        ]
        if not runnable:
            return

        future = self._prefetch_executor.submit(
            self._predict_decode_batch_and_prefetch_worker,
            runnable,
        )
        for item in runnable:
            self._prefetch_futures[int(item["layer_idx"])] = future

    def on_prefill_layer_end(self, layer_idx: int) -> None:
        """最后一个 prefill chunk 的 attention 完成后，准备首个 decode 的 active set。

        prefill 本身保持 full attention，不做稀疏。只有最后一个 prefill chunk
        才会在这里根据首个 decode mask 释放非 active GPU slots，并保留/预取
        首个 decode 需要的 token。prefetch=true 时，mask 初始化和 residency
        更新都在后台 worker 里完成。
        """
        ctx = get_context()
        if not ctx.is_prefill or not ctx.prefill_is_last_chunk: 
            return

        pending_view = self._pending_prefill_views[layer_idx]
        if pending_view is None:
            return
        self._pending_prefill_views[layer_idx] = None

        if self._prefetch_enabled:
            self._wait_prefetch(layer_idx)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            self._prefetch_futures[layer_idx] = self._prefetch_executor.submit(
                self._prefill_predict_and_prefetch_worker, # 包含predict_prefill_positions_sync和_ensure_positions_resident
                layer_idx,
                pending_view,
                event,
            )
            return

        with torch.inference_mode(), profiler.record("attnpredict_prepare_prefill_predictor_inputs"):
            result = self._predict_prefill_positions_sync(layer_idx, pending_view)
            hot_positions = result["hot_positions"]
            lease_start_positions = result["lease_start_positions"]
            self._commit_hot_leases(layer_idx, hot_positions, lease_start_positions)
            row_positions = {
                int(row_idx): self._compose_positions_from_hot(hot_positions.get(int(row_idx)), int(full_len))
                for row_idx, full_len in lease_start_positions.items()
            }
        if self._can_keep_full_resident_for_rows(layer_idx, lease_start_positions.keys()):
            with profiler.record("attnpredict_offload_full_resident_fast_path"):
                return
        self._ensure_positions_resident(layer_idx, row_positions, stream=None)

    def on_prefill_step_end(self) -> None:
        """最后一个 prefill step 返回前消费首个 decode 需要的 predictor lease。

        on_prefill_layer_end() 会尽早把每层 predictor 初始化提交到后台 stream，
        以便和后续层计算重叠。但第一个 decode step 必须有可用 lease；如果把
        等待延后到 decode，会污染 decode throughput。这里把等待点前移到
        prefill post-forward，成本归入 TTFT/prefill。
        """
        ctx = get_context()
        if not ctx.is_prefill or not any(bool(x) for x in (ctx.prefill_is_last_chunk or [])):
            return
        for layer_idx in range(self.num_layers):
            self._consume_prefetch(layer_idx, wait=True)

    def _predict_decode_batch_and_prefetch_worker(
        self,
        items: list[dict[str, object]],
    ) -> dict[str, object]:
        """后台任务：把一个 decode step 内多层 refresh 合成批量 CNN。"""
        first_logits = items[0]["attn_logits"]
        assert isinstance(first_logits, torch.Tensor)
        torch.cuda.set_device(first_logits.device)
        stream = self._prefetch_streams[0]
        with torch.inference_mode(), torch.cuda.stream(stream):
            for item in items:
                event = item["event"]
                assert isinstance(event, torch.cuda.Event)
                stream.wait_event(event)
            with profiler.record("attnpredict_offload_predict_cnn_stream"):
                layer_results = self._predict_decode_batch_positions_sync(items)

        # 后台只加载新 lease 需要的 token，不释放旧 view，避免和主线程旧 lease 消费竞态。
        for layer_idx, result in layer_results.items():
            lease_start_positions = result["lease_start_positions"]
            hot_positions = result["hot_positions"]
            row_positions = {
                int(row_idx): self._compose_positions_from_hot(
                    hot_positions.get(int(row_idx)),
                    int(full_len),
                )
                for row_idx, full_len in lease_start_positions.items()
            }
            if self._can_keep_full_resident_for_rows(layer_idx, lease_start_positions.keys()):
                with profiler.record("attnpredict_offload_full_resident_fast_path"):
                    continue
            self._ensure_positions_loaded(layer_idx, row_positions, stream=stream)

        with torch.cuda.stream(stream):
            done = torch.cuda.Event()
            done.record(stream)
        return {
            "event": done,
            "layer_results": layer_results,
        }

    def _predict_decode_batch_positions_sync(
        self,
        items: list[dict[str, object]],
    ) -> dict[int, dict[str, dict[int, np.ndarray] | dict[int, int]]]:
        """批量执行 decode logits -> hot positions。

        每层仍维护独立 history/lease，只把 CNN forward 按 pred_len 分组批处理。
        """
        layer_results: dict[int, dict[str, dict[int, np.ndarray] | dict[int, int]]] = {}
        cnn_groups: dict[int, list[dict[str, object]]] = {}

        for item in items:
            layer_idx = int(item["layer_idx"])
            attn_logits = item["attn_logits"]
            view = item["view"]
            assert isinstance(attn_logits, torch.Tensor)
            assert isinstance(view, dict)

            req_indices = view["req_indices"]
            positions = view["positions"]
            view_lens = view["view_lens"]
            full_context_lens = view["full_context_lens"]
            assert isinstance(positions, torch.Tensor)

            hot_positions: dict[int, np.ndarray] = {}
            lease_start_positions: dict[int, int] = {}
            layer_results[layer_idx] = {
                "hot_positions": hot_positions,
                "lease_start_positions": lease_start_positions,
            }

            batch_size = int(len(req_indices))
            for b in range(batch_size):
                row_idx = int(req_indices[b])
                view_len = int(view_lens[b])
                full_len = int(full_context_lens[b])
                if view_len <= 0 or full_len <= 0:
                    continue

                with profiler.record("attnpredict_offload_predict_softmax"):
                    logits = attn_logits[b, :, :view_len].to(torch.float32)
                    attn = torch.softmax(logits * self.attn_scale, dim=-1)

                with profiler.record("attnpredict_offload_predict_block_pool"):
                    pos = positions[b, :view_len].to(device=attn.device, dtype=torch.long)
                    block_idx = torch.div(pos, int(self.pooling_block_size), rounding_mode="floor")
                    pooled_len = (full_len + int(self.pooling_block_size) - 1) // int(self.pooling_block_size)
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

                with profiler.record("attnpredict_offload_predict_update_history"):
                    hist = self._update_pooled_attn_history(
                        self.attn_history[layer_idx].get(row_idx),
                        attn_pooling.unsqueeze(1).to(self.hf_config.torch_dtype),
                    )
                    self.attn_history[layer_idx][row_idx] = hist

                num_heads, num_rows, attn_len = hist.shape
                start_block = int(self.sink_token) // int(self.pooling_block_size)
                end_block = int(attn_len) - (int(self.local_token) // int(self.pooling_block_size))
                end_block = max(start_block, end_block)
                hist_window = hist[:, :, start_block:end_block]
                pred_len = int(hist_window.shape[-1])
                lease_start_positions[row_idx] = full_len
                if pred_len < 3:
                    tsp_attn = torch.ones(
                        (num_heads, pred_len),
                        dtype=hist.dtype,
                        device=hist.device,
                    )
                    hot_positions[row_idx] = self._hot_positions_from_scores(
                        tsp_attn,
                        seq_len=full_len,
                        start_block=start_block,
                    )
                    continue

                cnn_groups.setdefault(pred_len, []).append(
                    {
                        "layer_idx": layer_idx,
                        "row_idx": row_idx,
                        "full_len": full_len,
                        "start_block": start_block,
                        "num_heads": int(num_heads),
                        "inputs": hist_window.reshape(num_heads, num_rows, pred_len),
                    }
                )

        for pred_len, group in cnn_groups.items():
            chunk: list[dict[str, object]] = []
            chunk_heads = 0

            def run_cnn_chunk(entries: list[dict[str, object]]) -> None:
                with profiler.record("attnpredict_offload_predict_cnn_batch"):
                    inputs = torch.cat(
                        [entry["inputs"] for entry in entries],
                        dim=0,
                    )
                    with self._cnn_lock:
                        tsp_batch = self.cnn(inputs.to(self.cnn_dtype).contiguous()).to(torch.float32)

                offset = 0
                for entry in entries:
                    num_heads = int(entry["num_heads"])
                    tsp_attn = tsp_batch[offset:offset + num_heads]
                    offset += num_heads
                    layer_idx = int(entry["layer_idx"])
                    row_idx = int(entry["row_idx"])
                    layer_results[layer_idx]["hot_positions"][row_idx] = self._hot_positions_from_scores(
                        tsp_attn,
                        seq_len=int(entry["full_len"]),
                        start_block=int(entry["start_block"]),
                    )

            for entry in group:
                entry_heads = int(entry["num_heads"])
                if chunk and chunk_heads + entry_heads > int(self._cnn_batch_max_heads):
                    run_cnn_chunk(chunk)
                    chunk = []
                    chunk_heads = 0
                chunk.append(entry)
                chunk_heads += entry_heads
            if chunk:
                run_cnn_chunk(chunk)

        return layer_results

    def _predict_and_prefetch_worker(
        self,
        layer_idx: int,
        attn_logits: torch.Tensor,
        view: dict[str, torch.Tensor | None],
        attention_done_event: torch.cuda.Event,
    ) -> dict[str, object]:
        """后台任务：等待 attention 完成，预测下一份 hot lease，并预取其 KV。

        返回 hot lease 和 CUDA event。下一次同层 get_layer_store_view() 只在
        future 已完成或 stale 超限时消费它；否则继续使用旧 lease。
        """
        torch.cuda.set_device(attn_logits.device)
        stream = self._prefetch_streams[layer_idx]
        with torch.inference_mode(), torch.cuda.stream(stream):
            # 必须等主 stream 的 attention kernel 写完 attn_logits。
            stream.wait_event(attention_done_event)
            with profiler.record("attnpredict_offload_predict_cnn_stream"):
                result = self._predict_next_positions_sync(layer_idx, attn_logits, view)
                hot_positions = result["hot_positions"]
                row_positions = {
                    int(row_idx): self._compose_positions_from_hot(hot_positions.get(int(row_idx)), int(full_len))
                    for row_idx, full_len in result["lease_start_positions"].items()
                }

        # 后台只加载新 lease 需要的 token，不释放旧 view，避免和主线程旧 lease 消费竞态。
        self._ensure_positions_loaded(layer_idx, row_positions, stream=stream)
        with torch.cuda.stream(stream):
            done = torch.cuda.Event()
            done.record(stream)
        return {
            "event": done,
            "hot_positions": result["hot_positions"],
            "lease_start_positions": result["lease_start_positions"],
        }

    def _prefill_predict_and_prefetch_worker(
        self,
        layer_idx: int,
        view: dict[str, object],
        attention_done_event: torch.cuda.Event,
    ) -> dict[str, object]:
        """后台任务：prefill 最后一块结束后初始化 predictor，并准备首个 decode active set。"""
        tail_score = view["tail_score"]
        torch.cuda.set_device(tail_score.device)
        stream = self._prefetch_streams[layer_idx]
        with torch.inference_mode(), torch.cuda.stream(stream):
            stream.wait_event(attention_done_event)
            with profiler.record("attnpredict_offload_prefill_predict_stream"):
                result = self._predict_prefill_positions_sync(layer_idx, view)
                hot_positions = result["hot_positions"]
                row_positions = {
                    int(row_idx): self._compose_positions_from_hot(hot_positions.get(int(row_idx)), int(full_len))
                    for row_idx, full_len in result["lease_start_positions"].items()
                }

        self._ensure_positions_loaded(layer_idx, row_positions, stream=stream)
        with torch.cuda.stream(stream):
            done = torch.cuda.Event()
            done.record(stream)
        return {
            "event": done,
            "hot_positions": result["hot_positions"],
            "lease_start_positions": result["lease_start_positions"],
        }

    def _predict_next_positions_sync(
        self,
        layer_idx: int,
        attn_logits: torch.Tensor,
        view: dict[str, torch.Tensor | None],
    ) -> dict[str, dict[int, np.ndarray] | dict[int, int]]:
        """同步执行 AttentionPredictor 的“logits -> 下一步 positions”逻辑。

        输入 attn_logits 是当前 decode attention kernel 写出的 sparse view logits。
        如果当前层只看了部分 token，logits 的最后一维只覆盖 packed view。
        因此这里需要：
        1. 对 sparse logits 做 softmax。
        2. 根据 view["positions"] 直接聚合到 block-level attention history。
        3. 复用父类 _update_row_prediction_from_pooled() 更新历史并生成 tsp_mask。
        4. 把 tsp_mask 转成下一步需要驻留 GPU 的 positions。
        """
        req_indices = view["req_indices"]
        positions = view["positions"]
        view_lens = view["view_lens"]
        full_context_lens = view["full_context_lens"]

        hot_positions: dict[int, np.ndarray] = {}
        lease_start_positions: dict[int, int] = {}
        batch_size = int(len(req_indices))
        for b in range(batch_size):
            # req_indices 是 cache manager 的全局 row；view_lens 是本次实际可见 token 数；
            # full_len 是当前序列完整逻辑长度。
            row_idx = int(req_indices[b])
            view_len = int(view_lens[b])
            full_len = int(full_context_lens[b])
            if view_len <= 0 or full_len <= 0:
                continue

            # attn_logits 是未 scale 的 raw logits；这里和原 attnpredict 一样乘 attn_scale 后 softmax。
            with profiler.record("attnpredict_offload_predict_softmax"):
                logits = attn_logits[b, :, :view_len].to(torch.float32)
                attn = torch.softmax(logits * self.attn_scale, dim=-1)

            # sparse view：直接聚合到 block 级 history。等价于先 scatter 回
            # full_len、未可见 token 置 0、再按 block 做 max-pooling。
            with profiler.record("attnpredict_offload_predict_block_pool"):
                pos = positions[b, :view_len].to(device=attn.device, dtype=torch.long)
                block_idx = torch.div(pos, int(self.pooling_block_size), rounding_mode="floor")
                pooled_len = (full_len + int(self.pooling_block_size) - 1) // int(self.pooling_block_size)
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

            # 父类方法内部会更新滚动 history、CNN 预测、topk/sink/recent mask 生成。
            with profiler.record("attnpredict_offload_predict_update_history"):
                self._update_row_prediction_from_pooled(
                    layer_idx,
                    row_idx,
                    attn_pooling.unsqueeze(1).to(self.hf_config.torch_dtype),
                    seq_len=full_len,
                )
            hot_positions[row_idx] = self._hot_positions_from_mask(layer_idx, row_idx, full_len)
            lease_start_positions[row_idx] = full_len
        return {
            "hot_positions": hot_positions,
            "lease_start_positions": lease_start_positions,
        }

    def _time_sequence_predict(self, attn_history: torch.Tensor) -> tuple[torch.Tensor, int]:
        """串行化 CNN predictor forward。

        这个方法不是直接在 offload 文件里显式调用，而是通过父类
        _update_row_prediction() 多态调用。当前调用链包括：
        _predict_prefill_positions_sync/_predict_next_positions_sync ->
        _update_row_prediction*() -> self._time_sequence_predict()。

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
            cpu_slot_mapping = np.empty((total_chunk_tokens,), dtype=np.int64) #TODO 为什么这里是 int64 而不是 int32？

            token_offset = 0
            for seq in seqs:
                chunk_size = seq.current_chunk_size
                start_idx = seq.num_prefilled_tokens
                end_idx = start_idx + chunk_size
                row_idx = self._get_free_row(seq.seq_id)

                # 如果该序列是 chunked prefill 的后续 chunk，row_seq_lens 必须刚好等于
                # 已经 prefill 的 token 数，否则说明调度/释放账本出错。
                if int(self.row_seq_lens[row_idx]) != int(start_idx):
                    raise ValueError(
                        "AttnPredict offload row length mismatch in prefill: "
                        f"seq_id={seq.seq_id} row_seq_len={self.row_seq_lens[row_idx]} start_idx={start_idx}"
                    )

                # 先给完整历史 backing 分配 CPU slots。row_seq_lens 会在这里增长。
                cpu_slots = self._allocate_cpu_slots(seq.seq_id, chunk_size)
                cpu_slot_mapping[token_offset:token_offset + chunk_size] = cpu_slots

                for layer_id in range(self.num_layers):
                    # 每一层都要给当前 chunk 分配 GPU slots，保证 prefill full attention。
                    with self._layer_locks[layer_id]:
                        gpu_slots = self._allocate_gpu_slots_locked(layer_id, chunk_size)
                        self.gpu_req_to_token_slots_cpu[layer_id][row_idx, start_idx:end_idx] = gpu_slots
                        # prefill 阶段本 row 的 0:end_idx 始终完整驻留 GPU。
                        self._resident_positions[layer_id][int(row_idx)] = None

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
                self._layer_cpu_slot_mapping[layer_id] = cpu_slot_mapping # TODO 既然每一层都共享了，为何还单独设置？

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
                current_positions[b] = int(self.row_seq_lens[row_idx])
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

            # 后续每层 get_layer_store_view() 需要这两个数组来构建 dynamic view。
            self._decode_rows = rows
            self._decode_current_positions = current_positions
            total_context_len = int(context_lens_np.sum())
            full_resident_layers: list[int] = []
            for layer_id in range(self.num_layers):
                resident = self._resident_positions[layer_id]
                self._decode_full_resident_layers[layer_id] = (
                    total_context_len <= int(self.layer_num_slots[layer_id])
                    and all(
                        int(row) in resident and resident[int(row)] is None
                        for row in rows.tolist()
                    )
                )
                if self._decode_full_resident_layers[layer_id]:
                    full_resident_layers.append(layer_id)

            self._decode_current_slots_np = None
            self._decode_current_slots_cuda = None
            if full_resident_layers:
                current_slots_all = np.full(
                    (self.num_layers, batch_size),
                    -1,
                    dtype=np.int32,
                )
                for layer_id in full_resident_layers:
                    mirror = self.gpu_req_to_token_slots_cpu[layer_id]
                    with self._layer_locks[layer_id]:
                        for b, (row_idx, pos) in enumerate(zip(rows.tolist(), current_positions.tolist())):
                            row = int(row_idx)
                            cur_pos = int(pos)
                            slot = int(mirror[row, cur_pos])
                            if slot < 0:
                                slot = int(self._allocate_gpu_slots_locked(layer_id, 1)[0])
                                mirror[row, cur_pos] = slot
                            self._add_resident_position_locked(layer_id, row, cur_pos)
                            current_slots_all[layer_id, b] = slot
                    self.layer_batch_states[layer_id].slot_mapping = None
                self._decode_current_slots_np = current_slots_all
                self._decode_current_slots_cuda = torch.as_tensor(
                    current_slots_all,
                    dtype=torch.int32,
                    device="cuda",
                )
                for layer_id in full_resident_layers:
                    self.layer_batch_states[layer_id].slot_mapping = self._decode_current_slots_cuda[layer_id]

            self._decode_recent_positions_np = {}
            self._decode_recent_positions_cuda = {}
            for row_idx, cur_pos in zip(rows.tolist(), current_positions.tolist()):
                row = int(row_idx)
                recent = self._recent_positions_for_decode(
                    row,
                    int(self.row_seq_lens[row]),
                    int(cur_pos),
                )
                self._decode_recent_positions_np[row] = recent
                self._decode_recent_positions_cuda[row] = torch.as_tensor(
                    recent,
                    dtype=torch.int32,
                    device="cuda",
                )
            for layer_id in range(self.num_layers):
                self._decode_fast_view_plans[layer_id] = None

            input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device="cuda")
            positions = torch.tensor(positions_list, dtype=torch.int64, device="cuda")
            return input_ids, positions, None

    def free_seq(self, seq_id: int):
        """释放某个序列占用的 GPU active slots、CPU full slots 和 predictor 状态。"""
        row_idx = self.seq_id_to_row.pop(seq_id)
        for layer_idx in range(self.num_layers):
            # 释放前先等后台任务结束，避免后台还在访问这个 row 的映射或 slot。
            self._wait_prefetch(layer_idx)
            self._leased_hot_positions[layer_idx].pop(int(row_idx), None)
            self._lease_start_positions[layer_idx].pop(int(row_idx), None)
            self._last_refresh_positions[layer_idx].pop(int(row_idx), None)
            self._cpu_backing_dirty[layer_idx].pop(int(row_idx), None)
            self._lease_versions[layer_idx].pop(int(row_idx), None)
            self._full_resident_prefix_cache[layer_idx].pop(int(row_idx), None)
            self._decode_fast_view_plans[layer_idx] = None
            self._decode_fast_prefix_meta[layer_idx] = None
            self._decode_full_resident_layers[layer_idx] = False
            with self._layer_locks[layer_idx]:
                full_len = int(self.row_seq_lens[row_idx])

                # 回收该 row 在该层当前仍驻留 GPU 的 active slots。
                marker = self._resident_positions[layer_idx].pop(int(row_idx), None)
                if marker is None:
                    resident = self.gpu_req_to_token_slots_cpu[layer_idx][row_idx, :full_len]
                else:
                    resident = self.gpu_req_to_token_slots_cpu[layer_idx][row_idx, marker]
                slots = resident[resident >= 0].astype(np.int32, copy=False)
                if slots.size:
                    self.free_slots_stack[layer_idx].extend(int(x) for x in slots.tolist())
                    self._num_free_slots[layer_idx] += int(slots.size)
                self.gpu_req_to_token_slots_cpu[layer_idx][row_idx, :] = -1
                self.buffer_req_to_token_slots[layer_idx][row_idx, :] = -1

            # 清理该层该 row 的 predictor history/mask。
            self.attn_history[layer_idx].pop(int(row_idx), None)
            self.tsp_mask[layer_idx].pop(int(row_idx), None)

        # 回收 CPU full backing slots。CPU slot 是跨层共享 token slot，
        # 所以只需要按 row/pos 回收一次。
        full_len = int(self.row_seq_lens[row_idx])
        self._decode_recent_positions_np.pop(int(row_idx), None)
        self._decode_recent_positions_cuda.pop(int(row_idx), None)
        cpu_slots = self.cpu_req_to_token_slots[row_idx, :full_len]
        cpu_slots = cpu_slots[cpu_slots >= 0]
        if cpu_slots.size:
            self.cpu_free_slots_stack.extend(int(x) for x in cpu_slots.tolist())
            self._num_free_cpu_slots += int(cpu_slots.size)
        self.cpu_req_to_token_slots[row_idx, :] = -1
        self.row_seq_lens[row_idx] = 0
        self.free_rows.append(row_idx)
