from __future__ import annotations

import queue
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch

from sparsevllm.config import Config
from sparsevllm.utils.profiler import profiler


class _PinnedKVStagingSlot:
    """传输环中的一组可复用 K/V 缓冲和 CUDA 事件。"""

    def __init__(
        self,
        capacity_tokens: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        shape = (capacity_tokens, num_kv_heads, head_dim)
        self.k = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        self.v = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        self.ready_event = torch.cuda.Event()
        self.done_event = torch.cuda.Event()


class PredictiveTransferMixin:
    """预测式卸载 CacheManager 的内部异步传输实现。"""

    def _init_transfer_pipeline(self, config: Config) -> None:
        self._staging_slot_count = int(config.attnpredict_offload_staging_slots)
        self._staging_chunk_tokens = int(
            config.attnpredict_offload_staging_chunk_tokens
        )
        self._d2h_stream = torch.cuda.Stream(priority=0)

        self._prefill_writeback_executor = ThreadPoolExecutor(max_workers=1)
        self._prefill_writeback_futures: deque[Future] = deque()
        self._dirty_writeback_futures: deque[Future] = deque()
        self._prefill_writeback_lock = threading.Lock()
        self._prefill_writeback_bytes = 0
        self._dirty_writeback_bytes = 0
        self._prefill_writeback_backpressure_s = 0.0
        self._prefill_writeback_barrier_s = 0.0

        self._prefill_d2h_slots = self._allocate_pinned_staging_ring()
        self._prefill_d2h_free: queue.Queue[int] = queue.Queue()
        for slot_idx in range(len(self._prefill_d2h_slots)):
            self._prefill_d2h_free.put(slot_idx)

        self._h2d_release_executor = ThreadPoolExecutor(max_workers=1)
        self._h2d_release_futures: deque[Future] = deque()
        self._h2d_release_lock = threading.Lock()
        self._h2d_staging_slots = self._allocate_pinned_staging_ring()
        self._h2d_staging_free: queue.Queue[int] = queue.Queue()
        for slot_idx in range(len(self._h2d_staging_slots)):
            self._h2d_staging_free.put(slot_idx)
        self._h2d_load_tokens = 0

    def _allocate_pinned_staging_ring(self) -> list[_PinnedKVStagingSlot]:
        return [
            _PinnedKVStagingSlot(
                self._staging_chunk_tokens,
                self.num_kv_heads,
                self.head_dim,
                self.hf_config.torch_dtype,
            )
            for _ in range(self._staging_slot_count)
        ]

    @staticmethod
    def _raise_completed_futures(
        pending: deque[Future], lock: threading.Lock
    ) -> None:
        completed: list[Future] = []
        with lock:
            while pending and pending[0].done():
                completed.append(pending.popleft())
        for future in completed:
            future.result()

    def _drain_prefill_staging_slot(
        self,
        slot_idx: int,
        layer_idx: int,
        size: int,
        cpu_slots: np.ndarray,
    ) -> None:
        slot = self._prefill_d2h_slots[slot_idx]
        try:
            torch.cuda.set_device(self.rank)
            slot.done_event.synchronize()
            boundaries = np.flatnonzero(np.diff(cpu_slots) != 1) + 1
            starts = np.concatenate((np.asarray([0]), boundaries))
            ends = np.concatenate((boundaries, np.asarray([size])))
            with profiler.record("attnpredict_offload_prefill_cpu_backing_copy"):
                for source_start, source_end in zip(
                    starts.tolist(), ends.tolist()
                ):
                    target_start = int(cpu_slots[source_start])
                    target_end = target_start + source_end - source_start
                    self.cpu_kv_cache[
                        0, layer_idx, target_start:target_end
                    ].copy_(slot.k[source_start:source_end])
                    self.cpu_kv_cache[
                        1, layer_idx, target_start:target_end
                    ].copy_(slot.v[source_start:source_end])
            with self._layer_locks[layer_idx]:
                self._cpu_backed_valid[layer_idx, cpu_slots] = True
        finally:
            self._prefill_d2h_free.put(slot_idx)

    def _enqueue_prefill_writeback(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        cpu_slots: np.ndarray,
        *,
        writeback_kind: str = "prefill",
    ) -> None:
        total_tokens = int(k.shape[0])
        current_stream = torch.cuda.current_stream()
        for start in range(0, total_tokens, self._staging_chunk_tokens):
            end = min(total_tokens, start + self._staging_chunk_tokens)
            size = end - start
            self._raise_completed_futures(
                self._prefill_writeback_futures,
                self._prefill_writeback_lock,
            )
            self._raise_completed_futures(
                self._dirty_writeback_futures,
                self._prefill_writeback_lock,
            )
            wait_started = time.perf_counter()
            with profiler.record(
                "attnpredict_offload_prefill_d2h_backpressure_wait"
            ):
                slot_idx = self._prefill_d2h_free.get()
            self._prefill_writeback_backpressure_s += (
                time.perf_counter() - wait_started
            )
            slot = self._prefill_d2h_slots[slot_idx]

            slot.ready_event.record(current_stream)
            with torch.cuda.stream(self._d2h_stream), profiler.record(
                "attnpredict_offload_prefill_d2h_enqueue"
            ):
                self._d2h_stream.wait_event(slot.ready_event)
                slot.k[:size].copy_(k[start:end], non_blocking=True)
                slot.v[:size].copy_(v[start:end], non_blocking=True)
                slot.done_event.record(self._d2h_stream)

            future = self._prefill_writeback_executor.submit(
                self._drain_prefill_staging_slot,
                slot_idx,
                layer_idx,
                size,
                np.asarray(cpu_slots[start:end], dtype=np.int64).copy(),
            )
            writeback_bytes = (
                2
                * size
                * self.num_kv_heads
                * self.head_dim
                * k.element_size()
            )
            with self._prefill_writeback_lock:
                if writeback_kind == "dirty":
                    self._dirty_writeback_futures.append(future)
                    self._dirty_writeback_bytes += writeback_bytes
                else:
                    self._prefill_writeback_futures.append(future)
                    self._prefill_writeback_bytes += writeback_bytes
        k.record_stream(self._d2h_stream)
        v.record_stream(self._d2h_stream)

    def _wait_prefill_writebacks(self) -> None:
        started = time.perf_counter()
        while True:
            with self._prefill_writeback_lock:
                if not self._prefill_writeback_futures:
                    break
                futures = list(self._prefill_writeback_futures)
                self._prefill_writeback_futures.clear()
            for future in futures:
                future.result()
        self._prefill_writeback_barrier_s += time.perf_counter() - started

    def _wait_dirty_writebacks(self) -> None:
        while True:
            with self._prefill_writeback_lock:
                if not self._dirty_writeback_futures:
                    break
                futures = list(self._dirty_writeback_futures)
                self._dirty_writeback_futures.clear()
            for future in futures:
                future.result()

    def _release_h2d_staging_slot(self, slot_idx: int) -> None:
        slot = self._h2d_staging_slots[slot_idx]
        try:
            torch.cuda.set_device(self.rank)
            slot.done_event.synchronize()
        finally:
            self._h2d_staging_free.put(slot_idx)

    def _copy_cpu_to_gpu(
        self,
        layer_idx: int,
        cpu_slots: list[int],
        gpu_slots: list[int],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        if not cpu_slots:
            return
        self._h2d_load_tokens += len(cpu_slots)
        cpu_slots_np = np.asarray(cpu_slots, dtype=np.int64)
        if not np.all(self._cpu_backed_valid[layer_idx, cpu_slots_np]):
            self._wait_prefill_writebacks()
            self._wait_dirty_writebacks()
        if not np.all(self._cpu_backed_valid[layer_idx, cpu_slots_np]):
            raise RuntimeError("H2D 读取了尚未写完的 CPU KV")

        with profiler.record("attnpredict_offload_cpu_gather_background"):
            cpu_idx = torch.tensor(cpu_slots, dtype=torch.long, device="cpu")
            host_k = self.cpu_kv_cache[0, layer_idx].index_select(0, cpu_idx)
            host_v = self.cpu_kv_cache[1, layer_idx].index_select(0, cpu_idx)

        copy_stream = stream or torch.cuda.current_stream()
        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        for start in range(0, len(cpu_slots), self._staging_chunk_tokens):
            end = min(len(cpu_slots), start + self._staging_chunk_tokens)
            size = end - start
            self._raise_completed_futures(
                self._h2d_release_futures, self._h2d_release_lock
            )
            with profiler.record("attnpredict_offload_h2d_staging_wait"):
                slot_idx = self._h2d_staging_free.get()
            slot = self._h2d_staging_slots[slot_idx]
            slot.k[:size].copy_(host_k[start:end])
            slot.v[:size].copy_(host_v[start:end])

            with torch.cuda.stream(copy_stream), profiler.record(
                "attnpredict_offload_h2d_prefetch_stream"
            ):
                gpu_idx = torch.tensor(
                    gpu_slots[start:end], dtype=torch.long, device="cuda"
                )
                gpu_k = slot.k[:size].to(device="cuda", non_blocking=True)
                gpu_v = slot.v[:size].to(device="cuda", non_blocking=True)
                k_cache.index_copy_(0, gpu_idx, gpu_k)
                v_cache.index_copy_(0, gpu_idx, gpu_v)
                slot.done_event.record(copy_stream)
            future = self._h2d_release_executor.submit(
                self._release_h2d_staging_slot, slot_idx
            )
            with self._h2d_release_lock:
                self._h2d_release_futures.append(future)

    def _copy_dirty_gpu_to_cpu(
        self,
        layer_idx: int,
        cpu_slots: list[int],
        gpu_slots: list[int],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        if not cpu_slots:
            return
        cpu_slots_np = np.asarray(cpu_slots, dtype=np.int64)
        self._cpu_backed_valid[layer_idx, cpu_slots_np] = False
        source_stream = stream or torch.cuda.current_stream()
        with torch.cuda.stream(source_stream), profiler.record(
            "attnpredict_offload_dirty_d2h_gather_enqueue"
        ):
            gpu_idx = torch.tensor(gpu_slots, dtype=torch.long, device="cuda")
            k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
            self._enqueue_prefill_writeback(
                layer_idx,
                k_cache.index_select(0, gpu_idx),
                v_cache.index_select(0, gpu_idx),
                cpu_slots_np,
                writeback_kind="dirty",
            )
