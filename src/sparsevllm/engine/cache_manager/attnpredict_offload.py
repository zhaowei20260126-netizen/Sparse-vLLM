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
        # lease 只缓存 predictor 选出的中间 hot positions。sink/recent/current 每步
        # 动态拼接，避免复用旧完整 view 导致 local window 漂移。
        self._leased_hot_positions: list[dict[int, np.ndarray]] = [dict() for _ in range(self.num_layers)]
        self._lease_start_positions: list[dict[int, int]] = [dict() for _ in range(self.num_layers)]
        self._last_refresh_positions: list[dict[int, int]] = [dict() for _ in range(self.num_layers)]
        self._reuse_steps = int(config.attnpredict_reuse_steps)
        self._max_stale_steps = int(config.attnpredict_max_stale_steps)

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
        self.config.num_kvcache_slots = max(1, int(available_memory // slot_bytes))

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

    def _write_gpu_map(
        self,
        layer_idx: int,
        rows: list[int],
        positions: list[int],
        slots: list[int],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """把 CPU 侧 residency 变更同步写到 GPU row/pos -> slot 映射表。

        decode kernel 实际读取的是 GPU 上的 buffer_req_to_token_slots，
        因此后台线程完成释放/加载后也必须把映射写回 GPU。
        如果传入 stream，就在指定 CUDA stream 上执行这些写入。
        """
        if not rows:
            return
        ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with ctx:
            rows_t = torch.tensor(rows, dtype=torch.long, device="cuda")
            pos_t = torch.tensor(positions, dtype=torch.long, device="cuda")
            slots_t = torch.tensor(slots, dtype=torch.int32, device="cuda")
            self.buffer_req_to_token_slots[layer_idx][rows_t, pos_t] = slots_t

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

        # 待释放清单：这些 row/pos 下一步不再需要，需要把 GPU 映射表写成 -1。
        free_rows: list[int] = []   # 待释放 token 所在的 cache row。
        free_pos: list[int] = []    # 待释放 token 在序列中的逻辑位置。
        free_slots: list[int] = []  # 写回 GPU 映射表的值；释放时恒为 -1。

        # 待加载清单：这些 row/pos 下一步需要，但当前不在 GPU，需要从 CPU 拉回 GPU。
        load_rows: list[int] = []       # 待加载 token 所在的 cache row。
        load_pos: list[int] = []        # 待加载 token 在序列中的逻辑位置。
        load_cpu_slots: list[int] = []  # 待加载 token 在 CPU full backing 中的 slot。
        load_gpu_slots: list[int] = []  # 为待加载 token 新分配的 GPU active slot。

        with self._layer_locks[layer_idx]: #TODO 这是有cpu端做的，会不会拉满速度
            for row_idx, positions in row_positions.items():
                full_len = int(self.row_seq_lens[row_idx])

                # 清理非法位置并去重，避免重复分配或访问越界。
                positions = np.asarray(positions, dtype=np.int64)
                positions = positions[(positions >= 0) & (positions < full_len)]
                target = set(int(x) for x in np.unique(positions).tolist())

                # 第一步：释放已经驻留 GPU、但下一步不再需要的 token。
                # mirror[row_idx, pos] >= 0 表示 pos 当前占着某个 GPU slot。
                resident_positions = np.flatnonzero(mirror[row_idx, :full_len] >= 0)
                for pos in resident_positions.tolist():
                    if int(pos) in target:
                        continue
                    slot = int(mirror[row_idx, pos])
                    mirror[row_idx, pos] = -1
                    self.free_slots_stack[layer_idx].append(slot)
                    self._num_free_slots[layer_idx] += 1
                    free_rows.append(int(row_idx))
                    free_pos.append(int(pos))
                    free_slots.append(-1)

                # 第二步：找出目标集合里还没有 GPU slot 的 positions。
                missing = [pos for pos in sorted(target) if int(mirror[row_idx, pos]) < 0]
                if missing:
                    gpu_slots = self._allocate_gpu_slots_locked(layer_idx, len(missing))
                    for pos, gpu_slot in zip(missing, gpu_slots.tolist()):
                        # CPU backing 是最终真实来源；如果这里没有 cpu_slot，说明生命周期账本坏了。
                        cpu_slot = int(self.cpu_req_to_token_slots[row_idx, pos])
                        if cpu_slot < 0:
                            raise RuntimeError(
                                f"Missing CPU backing slot: layer={layer_idx} row={row_idx} pos={pos}"
                            )
                        mirror[row_idx, pos] = int(gpu_slot)
                        load_rows.append(int(row_idx))
                        load_pos.append(int(pos))
                        load_cpu_slots.append(cpu_slot) # 从 CPU 哪些 slot 读
                        load_gpu_slots.append(int(gpu_slot)) # 写到 GPU 哪些 active slot

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
        load_rows: list[int] = []
        load_pos: list[int] = []
        load_cpu_slots: list[int] = []
        load_gpu_slots: list[int] = []

        with self._layer_locks[layer_idx]:
            for row_idx, positions in row_positions.items():
                full_len = int(self.row_seq_lens[row_idx])
                positions = np.asarray(positions, dtype=np.int64)
                positions = positions[(positions >= 0) & (positions < full_len)]
                target = sorted(set(int(x) for x in positions.tolist()))
                missing = [pos for pos in target if int(mirror[row_idx, pos]) < 0]
                if not missing:
                    continue

                gpu_slots = self._allocate_gpu_slots_locked(layer_idx, len(missing))
                for pos, gpu_slot in zip(missing, gpu_slots.tolist()):
                    cpu_slot = int(self.cpu_req_to_token_slots[row_idx, pos])
                    if cpu_slot < 0:
                        raise RuntimeError(
                            f"Missing CPU backing slot: layer={layer_idx} row={row_idx} pos={pos}"
                        )
                    mirror[row_idx, pos] = int(gpu_slot)
                    load_rows.append(int(row_idx))
                    load_pos.append(int(pos))
                    load_cpu_slots.append(cpu_slot)
                    load_gpu_slots.append(int(gpu_slot))

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

    def _consume_prefetch(self, layer_idx: int, *, wait: bool) -> bool:
        future = self._prefetch_futures[layer_idx]
        if future is None:
            return False
        if not wait and not future.done():
            return False

        record_name = "attnpredict_offload_prefetch_wait" if wait else "attnpredict_offload_prefetch_poll"
        with profiler.record(record_name):
            try:
                result = future.result()
            finally:
                self._prefetch_futures[layer_idx] = None

        if isinstance(result, dict):
            event = result.get("event")
            if event is not None:
                torch.cuda.current_stream().wait_event(event)
            hot_positions = result.get("hot_positions")
            lease_start_positions = result.get("lease_start_positions")
            if hot_positions is not None and lease_start_positions is not None:
                self._commit_hot_leases(layer_idx, hot_positions, lease_start_positions)
        elif result is not None:
            torch.cuda.current_stream().wait_event(result)
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

            # 当前 decode token 刚在 _prepare_decode() 中分配了 CPU slot，
            # 但还没有本层 GPU slot；这里逐层分配。
            rows = self._decode_rows
            current_positions = self._decode_current_positions
            current_slots = self._ensure_current_decode_slots(layer_idx, rows, current_positions) # 为当前 decode token 分配 GPU slot，并返回本层所有decode token 的 GPU slot 列表
            state.slot_mapping = torch.tensor(current_slots, dtype=torch.int32, device="cuda")

            # 当前 token 必须加入本层可见 view。上一轮 predictor 无法提前预测当前 token，
            # 但原始 AttentionPredictor decode 也会把 newest KV 拼进去。
            view_positions: list[np.ndarray] = []
            row_positions: dict[int, np.ndarray] = {}
            for row_idx, cur_pos in zip(rows.tolist(), current_positions.tolist()):
                row = int(row_idx)
                positions = self._compose_positions_from_hot(
                    self._leased_hot_positions[layer_idx].get(row),
                    int(self.row_seq_lens[row]),
                    current_pos=int(cur_pos),
                )
                view_positions.append(positions)
                row_positions[row] = positions

            if refresh_pending:
                self._ensure_positions_loaded(layer_idx, row_positions, stream=None)
            else:
                self._ensure_positions_resident(layer_idx, row_positions, stream=None)
            self._decode_view_positions[layer_idx] = view_positions

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
                    update_rows.append(int(row_idx))
                    update_pos.append(int(pos))
                    update_slots.append(slot)
                result.append(slot)
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
        with profiler.record("attnpredict_offload_store_cpu_full_kv"):
            # cpu_slots_np 与本轮写入的 k/v 顺序对齐：TODO:这里会不会很耗时：代码在 prefill 阶段确实会很费时间，是 TTFT 变差的重要嫌疑
            # prefill 是 chunk 内所有 token；decode 是 batch 内每个 seq 的当前 token。
            cpu_slots = torch.tensor(cpu_slots_np, dtype=torch.long, device="cpu")
            host_k = k.detach().to(device="cpu", dtype=self.hf_config.torch_dtype)
            host_v = v.detach().to(device="cpu", dtype=self.hf_config.torch_dtype)
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
        view_positions = self._decode_view_positions[layer_idx]

        batch_size = int(req_indices.numel())
        keep_counts = [int(len(p)) for p in view_positions]
        max_keep = max(keep_counts)

        packed_slots = torch.full((batch_size, max_keep), -1, dtype=torch.int32, device=q.device)
        packed_positions = torch.full((batch_size, max_keep), -1, dtype=torch.int32, device=q.device)
        rows = self._decode_rows
        mirror = self.gpu_req_to_token_slots_cpu[layer_idx]
        for b, (row_idx, positions) in enumerate(zip(rows.tolist(), view_positions)):
            # positions 在 get_layer_store_view() 中已经确保 resident，这里只做查表和打包。
            slots = [int(mirror[int(row_idx), int(pos)]) for pos in positions.tolist()]
            if any(slot < 0 for slot in slots): #TODO:这一步是不是冗余的？
                raise RuntimeError(f"AttnPredict offload decode view has nonresident slots at layer {layer_idx}.")
            k = len(slots)
            packed_positions[b, :k] = torch.tensor(positions, dtype=torch.int32, device=q.device)
            packed_slots[b, :k] = torch.tensor(slots, dtype=torch.int32, device=q.device)

        view_lens = torch.tensor(keep_counts, dtype=torch.int32, device=q.device)
        local_req_indices = torch.arange(batch_size, dtype=torch.int32, device=q.device)

        # 保存本次 decode view，attention kernel 会把 logits 写到 attn_score；
        # attention 结束后 predict_next_mask() 用这些元数据把 sparse logits 还原到 full positions。
        self._last_decode_view[layer_idx] = {
            "req_indices": req_indices.detach().clone(),
            "positions": packed_positions.detach(),
            "view_lens": view_lens.detach().clone(),
            "full_context_lens": context_lens.detach().clone(),
        }
        return packed_slots, local_req_indices, view_lens

    def predict_next_mask(self, layer_idx: int, attn_logits: torch.Tensor) -> None:
        """根据当前层刚算完的 attention logits，预测下一步 active set。

        这个函数由 SparseController.on_attention_end() 调用。开启异步预取时，
        它只记录当前 CUDA stream 上的 event，然后把真正的 softmax/CNN/CPU gather/H2D
        放到后台线程和本层 prefetch stream 里执行。
        """
        view = self._last_decode_view[layer_idx]
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
            self._ensure_positions_resident(layer_idx, row_positions, stream=None)
            return

        # 如果上一轮 refresh 还没消费完，继续用旧 lease；不要提交重叠任务。
        if self._prefetch_futures[layer_idx] is not None:
            return

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
        self._ensure_positions_resident(layer_idx, row_positions, stream=None)

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
        2. 根据 view["positions"] scatter 回完整逻辑长度 full_len。
        3. 复用父类 _update_row_prediction() 更新历史并生成 tsp_mask。
        4. 把 tsp_mask 转成下一步需要驻留 GPU 的 positions。
        """
        req_indices = view["req_indices"]
        positions = view["positions"]
        view_lens = view["view_lens"]
        full_context_lens = view["full_context_lens"]

        hot_positions: dict[int, np.ndarray] = {}
        lease_start_positions: dict[int, int] = {}
        batch_size = int(req_indices.numel())
        for b in range(batch_size):
            # req_indices 是 cache manager 的全局 row；view_lens 是本次实际可见 token 数；
            # full_len 是当前序列完整逻辑长度。
            row_idx = int(req_indices[b].item())
            view_len = int(view_lens[b].item())
            full_len = int(full_context_lens[b].item())

            # attn_logits 是未 scale 的 raw logits；这里和原 attnpredict 一样乘 attn_scale 后 softmax。
            logits = attn_logits[b, :, :view_len].to(torch.float32)
            attn = torch.softmax(logits * self.attn_scale, dim=-1)
            # sparse view：把 packed view 上的 attention 权重 scatter 回完整位置。
            # 没被看的 token 权重为 0，这样 predictor history 仍保持完整长度语义。
            pos = positions[b, :view_len].to(device=attn.device, dtype=torch.long)
            full_attn = torch.zeros((attn.shape[0], full_len), dtype=attn.dtype, device=attn.device)
            full_attn.scatter_(1, pos.unsqueeze(0).expand(attn.shape[0], -1), attn) # 沿着第 1 维，也就是token维度，把 attn 里的值，写到 full_attn 的指定pos位置里。

            # 父类方法内部会做 pooling、CNN 预测、topk/sink/recent mask 生成。
            self._update_row_prediction(
                layer_idx,
                row_idx,
                full_attn.unsqueeze(1).to(self.hf_config.torch_dtype),
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
            with self._layer_locks[layer_idx]:
                full_len = int(self.row_seq_lens[row_idx])

                # 回收该 row 在该层当前仍驻留 GPU 的 active slots。
                resident = self.gpu_req_to_token_slots_cpu[layer_idx][row_idx, :full_len]
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
        cpu_slots = self.cpu_req_to_token_slots[row_idx, :full_len]
        cpu_slots = cpu_slots[cpu_slots >= 0]
        if cpu_slots.size:
            self.cpu_free_slots_stack.extend(int(x) for x in cpu_slots.tolist())
            self._num_free_cpu_slots += int(cpu_slots.size)
        self.cpu_req_to_token_slots[row_idx, :] = -1
        self.row_seq_lens[row_idx] = 0
        self.free_rows.append(row_idx)
