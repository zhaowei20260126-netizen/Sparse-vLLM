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
from .attnpredict_cnn import AttnPredictCNN
from .base import CacheManager, LayerBatchStates


class AttnPredictOffloadCacheManager(AttnPredictCacheManager):
    """AttentionPredictor with GPU active slots and CPU full-KV backing.

    This intentionally does not call ``StandardCacheManager.__init__``.  The GPU
    cache is an active pool, while CPU tensors keep recoverable full KV for every
    logical token.
    """

    def __init__(self, config: Config, rank: int, world_size: int):
        CacheManager.__init__(self, config, rank, world_size)
        assert world_size == 1, "attnpredict-offload currently supports tensor_parallel_size=1."

        self.allocate_kv_cache()

        num_slots = int(config.num_kvcache_slots)
        self.layer_num_slots = [num_slots for _ in range(self.num_layers)]
        self.free_slots_stack: list[list[int]] = [list(range(num_slots)) for _ in range(self.num_layers)]
        self._num_free_slots = [num_slots for _ in range(self.num_layers)]
        self.buffer_req_to_token_slots = [
            torch.full((self.max_buffer_rows, self.max_model_len), -1, dtype=torch.int32, device="cuda")
            for _ in range(self.num_layers)
        ]
        self.gpu_req_to_token_slots_cpu = [
            np.full((self.max_buffer_rows, self.max_model_len), -1, dtype=np.int32)
            for _ in range(self.num_layers)
        ]
        self.layer_batch_states = [LayerBatchStates() for _ in range(self.num_layers)]

        self.seq_id_to_row: dict[int, int] = {}
        self.free_rows = deque(range(self.max_buffer_rows))
        self.row_seq_lens = np.zeros((self.max_buffer_rows,), dtype=np.int32)

        self.cpu_req_to_token_slots = np.full(
            (self.max_buffer_rows, self.max_model_len),
            -1,
            dtype=np.int64,
        )
        self.cpu_free_slots_stack: list[int] = list(range(self.cpu_num_slots))
        self._num_free_cpu_slots = int(self.cpu_num_slots)

        self._layer_cpu_slot_mapping: list[np.ndarray | None] = [None for _ in range(self.num_layers)]
        self._decode_current_positions: np.ndarray | None = None
        self._decode_rows: np.ndarray | None = None
        self._decode_view_positions: list[list[np.ndarray] | None] = [None for _ in range(self.num_layers)]

        self.topk = int(config.num_top_tokens)
        self.history_step = int(config.attnpredict_history_steps)
        self.pooling_block_size = int(config.attnpredict_pooling_block_size)
        self.sink_token = int(config.num_sink_tokens)
        self.local_token = int(config.num_recent_tokens)
        self.attn_scale = self.head_dim ** -0.5
        self.attn_history: list[dict[int, torch.Tensor]] = [{} for _ in range(self.num_layers)]
        self.tsp_mask: list[dict[int, torch.Tensor]] = [{} for _ in range(self.num_layers)]
        self._last_decode_view: list[dict[str, torch.Tensor | None] | None] = [
            None for _ in range(self.num_layers)
        ]

        self.cnn = AttnPredictCNN()
        state_dict = torch.load(str(config.attnpredict_model_path), map_location="cuda", weights_only=False)
        self.cnn.load_state_dict(state_dict)
        self.cnn.to(dtype=torch.float16, device="cuda")
        self.cnn.eval()
        self.cnn_dtype = next(self.cnn.parameters()).dtype

        cpu_threads = int(config.attnpredict_offload_cpu_threads or 1)
        torch.set_num_threads(cpu_threads)
        self._prefetch_enabled = bool(config.attnpredict_offload_prefetch)
        self._pin_staging = bool(config.attnpredict_offload_pin_staging)
        self._prefetch_executor = ThreadPoolExecutor(max_workers=cpu_threads)
        self._prefetch_streams = [torch.cuda.Stream() for _ in range(self.num_layers)]
        self._prefetch_futures: list[Future | None] = [None for _ in range(self.num_layers)]
        self._prefetch_errors: list[BaseException | None] = [None for _ in range(self.num_layers)]
        self._layer_locks = [threading.RLock() for _ in range(self.num_layers)]
        self._cnn_lock = threading.RLock()

        logger.info(
            "AttnPredict offload allocation: gpu_active_slots={} cpu_full_slots={} "
            "layers={} prefetch={} cpu_threads={}".format(
                num_slots,
                self.cpu_num_slots,
                self.num_layers,
                self._prefetch_enabled,
                cpu_threads,
            )
        )

    def allocate_kv_cache(self):
        available_memory, slot_bytes_per_layer = self._get_available_slots_info()
        slot_bytes = self.num_layers * slot_bytes_per_layer
        self.config.num_kvcache_slots = max(1, int(available_memory // slot_bytes))

        self.kv_cache = torch.empty(
            2,
            self.num_layers,
            self.config.num_kvcache_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.torch_dtype,
            device="cuda",
        )

        self.cpu_num_slots = self._compute_cpu_num_slots()
        self.cpu_k_cache = [
            torch.empty(
                self.cpu_num_slots,
                self.num_kv_heads,
                self.head_dim,
                dtype=self.hf_config.torch_dtype,
                device="cpu",
            )
            for _ in range(self.num_layers)
        ]
        self.cpu_v_cache = [
            torch.empty(
                self.cpu_num_slots,
                self.num_kv_heads,
                self.head_dim,
                dtype=self.hf_config.torch_dtype,
                device="cpu",
            )
            for _ in range(self.num_layers)
        ]

    def _compute_cpu_num_slots(self) -> int:
        explicit = int(getattr(self.config, "attnpredict_offload_cpu_slots", -1) or -1)
        if explicit > 0:
            return explicit

        dtype_size = torch.tensor([], dtype=self.hf_config.torch_dtype).element_size()
        bytes_per_slot_all_layers = (
            self.num_layers * 2 * self.num_kv_heads * self.head_dim * dtype_size
        )
        desired = int(self.max_model_len) * int(max(1, self.config.max_num_seqs_in_batch))
        mem_available = self._cpu_mem_available_bytes()
        if mem_available > 0 and bytes_per_slot_all_layers > 0:
            by_mem = int((mem_available * 0.70) // bytes_per_slot_all_layers)
            desired = min(desired, max(1, by_mem))
        return max(1, desired)

    @staticmethod
    def _cpu_mem_available_bytes() -> int:
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
        return self.layer_batch_states[layer_idx]

    def get_layer_kv_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kv_cache[0, layer_idx], self.kv_cache[1, layer_idx]

    def get_layer_compute_tensors(self, layer_idx: int, sparse_controller):
        raise NotImplementedError

    def get_layer_buffer_req_to_token_slots(self, layer_idx: int) -> torch.Tensor:
        return self.buffer_req_to_token_slots[layer_idx]

    @property
    def num_free_slots(self) -> int:
        gpu_free = min(self._num_free_slots) if self._num_free_slots else 0
        return min(int(gpu_free), int(self._num_free_cpu_slots))

    @property
    def num_free_cpu_slots(self) -> int:
        return int(self._num_free_cpu_slots)

    def prompt_admission_free_slots(self) -> int:
        return min(int(self.num_free_slots), int(self.num_free_cpu_slots))

    def free_slot_stats(self) -> dict[str, int]:
        gpu_free = min(self._num_free_slots) if self._num_free_slots else 0
        return {
            "free_slots": int(self.num_free_slots),
            "free_gpu_active_slots": int(gpu_free),
            "free_cpu_full_slots": int(self.num_free_cpu_slots),
        }

    def prefill_batched_tokens_margin(self) -> int:
        return int(self.config.attnpredict_history_steps)

    def remaining_prefill_tokens(self, seq: Sequence) -> int:
        remaining = int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
        history = int(self.config.attnpredict_history_steps)
        if history > 0 and remaining > history:
            return remaining - history
        return remaining

    def _get_free_row(self, seq_id: int) -> int:
        if seq_id in self.seq_id_to_row:
            return self.seq_id_to_row[seq_id]
        if not self.free_rows:
            raise RuntimeError("No free rows in AttnPredict offload cache manager.")
        row_idx = self.free_rows.popleft()
        self.seq_id_to_row[seq_id] = row_idx
        return row_idx

    def _allocate_cpu_slots(self, seq_id: int, size: int) -> np.ndarray:
        if self._num_free_cpu_slots < size:
            raise RuntimeError(
                f"Out of AttnPredict CPU full-KV slots: need={size}, free={self._num_free_cpu_slots}"
            )
        row_idx = self._get_free_row(seq_id)
        cur_len = int(self.row_seq_lens[row_idx])
        if cur_len + int(size) > int(self.max_model_len):
            raise RuntimeError(
                f"AttnPredict offload sequence exceeds max_model_len: "
                f"seq_id={seq_id} cur_len={cur_len} add={size} max={self.max_model_len}"
            )
        slots = np.asarray(self.cpu_free_slots_stack[-size:], dtype=np.int64)
        del self.cpu_free_slots_stack[-size:]
        self._num_free_cpu_slots -= size
        self.cpu_req_to_token_slots[row_idx, cur_len:cur_len + size] = slots
        self.row_seq_lens[row_idx] += size
        return slots

    def _allocate_gpu_slots_locked(self, layer_idx: int, size: int) -> np.ndarray:
        if self._num_free_slots[layer_idx] < size:
            raise RuntimeError(
                f"Out of AttnPredict GPU active slots: layer={layer_idx} need={size} "
                f"free={self._num_free_slots[layer_idx]}"
            )
        slots = np.asarray(self.free_slots_stack[layer_idx][-size:], dtype=np.int32)
        del self.free_slots_stack[layer_idx][-size:]
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
        if not cpu_slots:
            return
        with profiler.record("attnpredict_offload_cpu_gather_background"):
            cpu_idx = torch.tensor(cpu_slots, dtype=torch.long, device="cpu")
            host_k = self.cpu_k_cache[layer_idx].index_select(0, cpu_idx)
            host_v = self.cpu_v_cache[layer_idx].index_select(0, cpu_idx)
            if self._pin_staging:
                pinned_k = torch.empty(host_k.shape, dtype=host_k.dtype, device="cpu", pin_memory=True)
                pinned_v = torch.empty(host_v.shape, dtype=host_v.dtype, device="cpu", pin_memory=True)
                pinned_k.copy_(host_k, non_blocking=False)
                pinned_v.copy_(host_v, non_blocking=False)
                host_k, host_v = pinned_k, pinned_v

        ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with ctx, profiler.record("attnpredict_offload_h2d_prefetch_stream"):
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
        mirror = self.gpu_req_to_token_slots_cpu[layer_idx]
        free_rows: list[int] = []
        free_pos: list[int] = []
        free_slots: list[int] = []
        load_rows: list[int] = []
        load_pos: list[int] = []
        load_cpu_slots: list[int] = []
        load_gpu_slots: list[int] = []

        with self._layer_locks[layer_idx]:
            for row_idx, positions in row_positions.items():
                full_len = int(self.row_seq_lens[row_idx])
                positions = np.asarray(positions, dtype=np.int64)
                positions = positions[(positions >= 0) & (positions < full_len)]
                target = set(int(x) for x in np.unique(positions).tolist())

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

                missing = [pos for pos in sorted(target) if int(mirror[row_idx, pos]) < 0]
                if missing:
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

        self._write_gpu_map(layer_idx, free_rows, free_pos, free_slots, stream=stream)
        self._write_gpu_map(layer_idx, load_rows, load_pos, load_gpu_slots, stream=stream)
        self._copy_cpu_to_gpu(layer_idx, load_cpu_slots, load_gpu_slots, stream=stream)

    def _positions_from_mask(self, layer_idx: int, row_idx: int, full_len: int) -> np.ndarray:
        pred_mask = self.tsp_mask[layer_idx].get(int(row_idx))
        if pred_mask is None:
            return np.arange(full_len, dtype=np.int64)
        valid_len = min(int(pred_mask.numel()), int(full_len))
        if valid_len <= 0:
            return np.empty((0,), dtype=np.int64)
        pos = pred_mask[:valid_len].nonzero(as_tuple=False).squeeze(-1)
        return pos.to(device="cpu", dtype=torch.long).numpy().astype(np.int64, copy=False)

    def _decode_base_positions(self, layer_idx: int) -> dict[int, np.ndarray]:
        if self._decode_rows is None or self._decode_current_positions is None:
            return {}
        row_positions: dict[int, np.ndarray] = {}
        for row_idx, cur_pos in zip(self._decode_rows.tolist(), self._decode_current_positions.tolist()):
            row_positions[int(row_idx)] = self._positions_from_mask(layer_idx, int(row_idx), int(cur_pos))
        return row_positions

    def _wait_prefetch(self, layer_idx: int) -> bool:
        future = self._prefetch_futures[layer_idx]
        if future is None:
            return True
        with profiler.record("attnpredict_offload_prefetch_wait"):
            try:
                event = future.result()
            except Exception as e:
                self._prefetch_futures[layer_idx] = None
                self._prefetch_errors[layer_idx] = e
                logger.exception(
                    "AttnPredict offload prefetch failed at layer {}; falling back to synchronous reload.",
                    layer_idx,
                )
                return False
            self._prefetch_futures[layer_idx] = None
            self._prefetch_errors[layer_idx] = None
            if event is not None:
                torch.cuda.current_stream().wait_event(event)
        return True

    def get_layer_store_view(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx = get_context()
        if not ctx.is_prefill:
            self._wait_prefetch(layer_idx)
            base_positions = self._decode_base_positions(layer_idx)
            if base_positions:
                self._ensure_positions_resident(layer_idx, base_positions, stream=None)

            state = self.layer_batch_states[layer_idx]
            assert state.req_indices is not None
            assert state.context_lens is not None
            rows = state.req_indices.to(device="cpu", dtype=torch.long).numpy().astype(np.int64, copy=False)
            current_positions = state.context_lens.to(device="cpu", dtype=torch.long).numpy().astype(np.int64, copy=False) - 1
            current_slots = self._ensure_current_decode_slots(layer_idx, rows, current_positions)
            state.slot_mapping = torch.tensor(current_slots, dtype=torch.int32, device="cuda")

            view_positions: list[np.ndarray] = []
            for row_idx, cur_pos in zip(rows.tolist(), current_positions.tolist()):
                base = base_positions.get(int(row_idx), np.empty((0,), dtype=np.int64))
                view_positions.append(np.unique(np.concatenate([base, np.asarray([cur_pos], dtype=np.int64)])))
            self._decode_view_positions[layer_idx] = view_positions

        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        return k_cache, v_cache, self.layer_batch_states[layer_idx].slot_mapping

    def _ensure_current_decode_slots(
        self,
        layer_idx: int,
        rows: np.ndarray,
        positions: np.ndarray,
    ) -> list[int]:
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
        if v is None:
            raise RuntimeError("attnpredict-offload requires V in on_kv_stored().")
        cpu_slots_np = self._layer_cpu_slot_mapping[layer_idx]
        if cpu_slots_np is None or len(cpu_slots_np) == 0:
            return
        with profiler.record("attnpredict_offload_store_cpu_full_kv"):
            cpu_slots = torch.tensor(cpu_slots_np, dtype=torch.long, device="cpu")
            host_k = k.detach().to(device="cpu", dtype=self.hf_config.torch_dtype)
            host_v = v.detach().to(device="cpu", dtype=self.hf_config.torch_dtype)
            self.cpu_k_cache[layer_idx].index_copy_(0, cpu_slots, host_k)
            self.cpu_v_cache[layer_idx].index_copy_(0, cpu_slots, host_v)

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
        view_positions = self._decode_view_positions[layer_idx]
        if view_positions is None:
            return active_slots, req_indices, context_lens

        batch_size = int(req_indices.numel())
        keep_counts = [int(len(p)) for p in view_positions]
        max_keep = max(keep_counts) if keep_counts else 0
        if max_keep <= 0:
            self._last_decode_view[layer_idx] = None
            return active_slots, req_indices, context_lens

        packed_slots = torch.full((batch_size, max_keep), -1, dtype=torch.int32, device=q.device)
        packed_positions = torch.full((batch_size, max_keep), -1, dtype=torch.int32, device=q.device)
        rows = req_indices.to(device="cpu", dtype=torch.long).numpy().astype(np.int64, copy=False)
        mirror = self.gpu_req_to_token_slots_cpu[layer_idx]
        for b, (row_idx, positions) in enumerate(zip(rows.tolist(), view_positions)):
            slots = [int(mirror[int(row_idx), int(pos)]) for pos in positions.tolist()]
            if any(slot < 0 for slot in slots):
                raise RuntimeError(f"AttnPredict offload decode view has nonresident slots at layer {layer_idx}.")
            k = len(slots)
            packed_positions[b, :k] = torch.tensor(positions, dtype=torch.int32, device=q.device)
            packed_slots[b, :k] = torch.tensor(slots, dtype=torch.int32, device=q.device)

        view_lens = torch.tensor(keep_counts, dtype=torch.int32, device=q.device)
        local_req_indices = torch.arange(batch_size, dtype=torch.int32, device=q.device)
        self._last_decode_view[layer_idx] = {
            "req_indices": req_indices.detach().clone(),
            "positions": packed_positions.detach(),
            "view_lens": view_lens.detach().clone(),
            "full_context_lens": context_lens.detach().clone(),
        }
        return packed_slots, local_req_indices, view_lens

    def predict_next_mask(self, layer_idx: int, attn_logits: torch.Tensor) -> None:
        view = self._last_decode_view[layer_idx]
        if view is None:
            return
        if not self._prefetch_enabled:
            with torch.inference_mode():
                row_positions = self._predict_next_positions_sync(layer_idx, attn_logits, view)
            self._ensure_positions_resident(layer_idx, row_positions, stream=None)
            return

        self._wait_prefetch(layer_idx)
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
        ctx = get_context()
        if not ctx.is_prefill or not ctx.prefill_is_last_chunk:
            return
        state = self.layer_batch_states[layer_idx]
        if state.req_indices is None:
            return
        rows = state.req_indices.to(device="cpu", dtype=torch.long).numpy().astype(np.int64, copy=False)
        row_positions = {}
        for b, row_idx in enumerate(rows.tolist()):
            if not ctx.prefill_is_last_chunk[b]:
                continue
            full_len = int(self.row_seq_lens[int(row_idx)])
            row_positions[int(row_idx)] = self._positions_from_mask(layer_idx, int(row_idx), full_len)
        if not row_positions:
            return

        if not self._prefetch_enabled:
            self._ensure_positions_resident(layer_idx, row_positions, stream=None)
            return

        self._wait_prefetch(layer_idx)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self._prefetch_futures[layer_idx] = self._prefetch_executor.submit(
            self._positions_prefetch_worker,
            layer_idx,
            row_positions,
            event,
        )

    def _predict_and_prefetch_worker(
        self,
        layer_idx: int,
        attn_logits: torch.Tensor,
        view: dict[str, torch.Tensor | None],
        attention_done_event: torch.cuda.Event,
    ) -> torch.cuda.Event:
        torch.cuda.set_device(attn_logits.device)
        stream = self._prefetch_streams[layer_idx]
        with torch.inference_mode(), torch.cuda.stream(stream):
            stream.wait_event(attention_done_event)
            with profiler.record("attnpredict_offload_predict_cnn_stream"):
                row_positions = self._predict_next_positions_sync(layer_idx, attn_logits, view)
        self._ensure_positions_resident(layer_idx, row_positions, stream=stream)
        with torch.cuda.stream(stream):
            done = torch.cuda.Event()
            done.record(stream)
        return done

    def _positions_prefetch_worker(
        self,
        layer_idx: int,
        row_positions: dict[int, np.ndarray],
        attention_done_event: torch.cuda.Event,
    ) -> torch.cuda.Event:
        torch.cuda.set_device(self.kv_cache.device)
        stream = self._prefetch_streams[layer_idx]
        with torch.cuda.stream(stream):
            stream.wait_event(attention_done_event)
        self._ensure_positions_resident(layer_idx, row_positions, stream=stream)
        with torch.cuda.stream(stream):
            done = torch.cuda.Event()
            done.record(stream)
        return done

    def _predict_next_positions_sync(
        self,
        layer_idx: int,
        attn_logits: torch.Tensor,
        view: dict[str, torch.Tensor | None],
    ) -> dict[int, np.ndarray]:
        if attn_logits.dim() == 2:
            attn_logits = attn_logits.unsqueeze(1)

        req_indices = view["req_indices"]
        positions = view["positions"]
        view_lens = view["view_lens"]
        full_context_lens = view["full_context_lens"]
        assert req_indices is not None
        assert view_lens is not None
        assert full_context_lens is not None

        row_positions: dict[int, np.ndarray] = {}
        batch_size = int(req_indices.numel())
        for b in range(batch_size):
            row_idx = int(req_indices[b].item())
            view_len = int(view_lens[b].item())
            full_len = int(full_context_lens[b].item())
            if view_len <= 0 or full_len <= 0:
                continue

            logits = attn_logits[b, :, :view_len].to(torch.float32)
            attn = torch.softmax(logits * self.attn_scale, dim=-1)
            if positions is None:
                full_attn = attn[:, :full_len]
            else:
                pos = positions[b, :view_len].to(device=attn.device, dtype=torch.long)
                full_attn = torch.zeros((attn.shape[0], full_len), dtype=attn.dtype, device=attn.device)
                full_attn.scatter_(1, pos.unsqueeze(0).expand(attn.shape[0], -1), attn)

            self._update_row_prediction(
                layer_idx,
                row_idx,
                full_attn.unsqueeze(1).to(self.hf_config.torch_dtype),
            )
            row_positions[row_idx] = self._positions_from_mask(layer_idx, row_idx, full_len)
        return row_positions

    def _time_sequence_predict(self, attn_history: torch.Tensor) -> tuple[torch.Tensor, int]:
        # The CNN module is shared by all layers; serialize forward calls while
        # keeping the work on each layer's prefetch stream.
        with self._cnn_lock:
            return super()._time_sequence_predict(attn_history)

    def _prepare_prefill(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_prefill"):
            total_chunk_tokens = sum(seq.current_chunk_size for seq in seqs)
            input_ids_np = np.empty(total_chunk_tokens, dtype=np.int64)
            positions_np = np.empty(total_chunk_tokens, dtype=np.int64)
            cu_seqlens_q = [0]

            layers_slot_mapping_cuda = torch.empty(
                (self.num_layers, total_chunk_tokens), dtype=torch.int32, device="cuda"
            )
            context_lens_list = [[] for _ in range(self.num_layers)]
            cpu_slot_mapping = np.empty((total_chunk_tokens,), dtype=np.int64)

            token_offset = 0
            for seq in seqs:
                chunk_size = seq.current_chunk_size
                start_idx = seq.num_prefilled_tokens
                end_idx = start_idx + chunk_size
                row_idx = self._get_free_row(seq.seq_id)
                if int(self.row_seq_lens[row_idx]) != int(start_idx):
                    raise ValueError(
                        "AttnPredict offload row length mismatch in prefill: "
                        f"seq_id={seq.seq_id} row_seq_len={self.row_seq_lens[row_idx]} start_idx={start_idx}"
                    )
                cpu_slots = self._allocate_cpu_slots(seq.seq_id, chunk_size)
                cpu_slot_mapping[token_offset:token_offset + chunk_size] = cpu_slots

                for layer_id in range(self.num_layers):
                    with self._layer_locks[layer_id]:
                        gpu_slots = self._allocate_gpu_slots_locked(layer_id, chunk_size)
                        self.gpu_req_to_token_slots_cpu[layer_id][row_idx, start_idx:end_idx] = gpu_slots
                    slots_cuda = torch.tensor(gpu_slots, dtype=torch.int32, device="cuda")
                    self.buffer_req_to_token_slots[layer_id][row_idx, start_idx:end_idx] = slots_cuda
                    layers_slot_mapping_cuda[layer_id, token_offset:token_offset + chunk_size] = slots_cuda
                    context_lens_list[layer_id].append(end_idx)

                chunk_tokens = seq.token_ids
                if len(chunk_tokens) > chunk_size:
                    chunk_tokens = chunk_tokens[start_idx:end_idx]
                input_ids_np[token_offset:token_offset + chunk_size] = chunk_tokens
                positions_np[token_offset:token_offset + chunk_size] = np.arange(start_idx, end_idx)
                cu_seqlens_q.append(cu_seqlens_q[-1] + chunk_size)
                token_offset += chunk_size

            layers_context_lens_cuda = torch.tensor(context_lens_list, dtype=torch.int32, device="cuda")
            req_ids = [self.seq_id_to_row[seq.seq_id] for seq in seqs]
            req_ids_cuda = torch.tensor(req_ids, dtype=torch.int32, device="cuda")
            for layer_id in range(self.num_layers):
                state = self.layer_batch_states[layer_id]
                state.slot_mapping = layers_slot_mapping_cuda[layer_id]
                state.context_lens = layers_context_lens_cuda[layer_id]
                state.req_indices = req_ids_cuda
                self._layer_cpu_slot_mapping[layer_id] = cpu_slot_mapping

            input_ids = torch.from_numpy(input_ids_np).to("cuda")
            positions = torch.from_numpy(positions_np).to("cuda")
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cuda")
            return input_ids, positions, cu_seqlens_q

    def _prepare_decode(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_decode"):
            batch_size = len(seqs)
            input_ids_list = [seq.last_token for seq in seqs]
            positions_list = [seq.num_tokens - 1 for seq in seqs]
            cpu_slots_batch = np.empty((batch_size,), dtype=np.int64)
            rows = np.empty((batch_size,), dtype=np.int64)
            current_positions = np.empty((batch_size,), dtype=np.int64)

            for b, seq in enumerate(seqs):
                row_idx = self._get_free_row(seq.seq_id)
                current_positions[b] = int(self.row_seq_lens[row_idx])
                cpu_slots_batch[b] = self._allocate_cpu_slots(seq.seq_id, 1)[0]
                rows[b] = row_idx

            context_lens_np = self.row_seq_lens[rows]
            context_lens_cuda = torch.tensor(context_lens_np, dtype=torch.int32, device="cuda")
            req_indices_cuda = torch.tensor(rows, dtype=torch.int32, device="cuda")
            empty_slot_mapping = torch.full((batch_size,), -1, dtype=torch.int32, device="cuda")

            for layer_id in range(self.num_layers):
                state = self.layer_batch_states[layer_id]
                state.slot_mapping = empty_slot_mapping
                state.context_lens = context_lens_cuda
                state.req_indices = req_indices_cuda
                self._layer_cpu_slot_mapping[layer_id] = cpu_slots_batch

            self._decode_rows = rows
            self._decode_current_positions = current_positions

            input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device="cuda")
            positions = torch.tensor(positions_list, dtype=torch.int64, device="cuda")
            return input_ids, positions, None

    def free_seq(self, seq_id: int):
        row_idx = self.seq_id_to_row.pop(seq_id, None)
        if row_idx is None:
            return
        for layer_idx in range(self.num_layers):
            self._wait_prefetch(layer_idx)
            with self._layer_locks[layer_idx]:
                full_len = int(self.row_seq_lens[row_idx])
                resident = self.gpu_req_to_token_slots_cpu[layer_idx][row_idx, :full_len]
                slots = resident[resident >= 0].astype(np.int32, copy=False)
                if slots.size:
                    self.free_slots_stack[layer_idx].extend(int(x) for x in slots.tolist())
                    self._num_free_slots[layer_idx] += int(slots.size)
                self.gpu_req_to_token_slots_cpu[layer_idx][row_idx, :] = -1
                self.buffer_req_to_token_slots[layer_idx][row_idx, :] = -1
            self.attn_history[layer_idx].pop(int(row_idx), None)
            self.tsp_mask[layer_idx].pop(int(row_idx), None)

        full_len = int(self.row_seq_lens[row_idx])
        cpu_slots = self.cpu_req_to_token_slots[row_idx, :full_len]
        cpu_slots = cpu_slots[cpu_slots >= 0]
        if cpu_slots.size:
            self.cpu_free_slots_stack.extend(int(x) for x in cpu_slots.tolist())
            self._num_free_cpu_slots += int(cpu_slots.size)
        self.cpu_req_to_token_slots[row_idx, :] = -1
        self.row_seq_lens[row_idx] = 0
        self.free_rows.append(row_idx)

    def free_part_slots(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        raise ValueError("attnpredict-offload manages active GPU slots through predictor prefetch.")
