from __future__ import annotations

import numpy as np
import torch

from sparsevllm.utils.profiler import profiler


class PredictiveResidencyMixin:
    """预测式卸载 CacheManager 的内部驻留和 Lease 代际实现。"""

    def _reclaim_clean_inactive_locked(self, layer_idx: int, target_free: int) -> int:
        """显存接近危险水位时，提前回收上一代 Lease 的干净 block。"""
        if self._residency_policy != "lease-aware":
            return 0
        target_free = min(target_free, self.layer_num_slots[layer_idx])
        if self._num_free_slots[layer_idx] >= target_free:
            return 0

        mirror = self.gpu_req_to_token_slots_cpu[layer_idx]
        resident_rows = self._resident_positions[layer_idx]
        dirty = self._dirty_gpu_positions[layer_idx]
        reclaimed = 0
        for row_idx in sorted(self._lease_grace_blocks[layer_idx]):
            inactive = (
                self._lease_grace_blocks[layer_idx].get(row_idx, set())
                - self._lease_active_blocks[layer_idx].get(row_idx, set())
            )
            positions = self._positions_from_blocks(
                inactive, int(self.row_seq_lens[row_idx])
            )
            if positions.size == 0:
                continue
            positions = positions[mirror[row_idx, positions] >= 0]
            dirty_positions = [pos for row, pos in dirty if row == row_idx]
            if dirty_positions:
                positions = positions[
                    ~np.isin(positions, np.asarray(dirty_positions, dtype=np.int64))
                ]
            if positions.size == 0:
                continue
            cpu_slots = self.cpu_req_to_token_slots[row_idx, positions]
            valid = cpu_slots >= 0
            if np.any(valid):
                valid_indices = np.flatnonzero(valid)
                valid[valid_indices] = self._cpu_backed_valid[
                    layer_idx, cpu_slots[valid_indices]
                ]
            need = target_free - self._num_free_slots[layer_idx]
            selected = positions[valid][:need]
            if selected.size == 0:
                continue
            gpu_slots = mirror[row_idx, selected].astype(np.int32, copy=True)
            mirror[row_idx, selected] = -1
            resident_rows.setdefault(row_idx, set()).difference_update(
                selected.tolist()
            )
            self.free_slots_stack[layer_idx].extend(gpu_slots.tolist())
            count = int(selected.size)
            self._num_free_slots[layer_idx] += count
            self._lease_grace_evictions += count
            reclaimed += count
            if self._num_free_slots[layer_idx] >= target_free:
                break
        return reclaimed

    def _allocate_gpu_slots_locked(self, layer_idx: int, size: int) -> np.ndarray:
        """在指定层的 GPU active pool 中分配 slots。

        调用者必须已经持有该层 lock。原因是后台 prefetch、同步路径、
        free_seq 都可能同时修改同一层的 free stack 和 residency map。
        """
        if (
            self._residency_policy == "lease-aware"
            and self._num_free_slots[layer_idx]
            < size + self._free_critical_watermark
        ):
            self._reclaim_clean_inactive_locked(
                layer_idx, size + self._free_critical_watermark
            )
        if self._num_free_slots[layer_idx] < size:
            raise RuntimeError(
                f"Out of AttnPredict GPU active slots: layer={layer_idx} need={size} "
                f"free={self._num_free_slots[layer_idx]} policy={self._residency_policy}"
            )
        slots = np.asarray(self.free_slots_stack[layer_idx][-size:], dtype=np.int32)
        del self.free_slots_stack[layer_idx][-size:]
        self._num_free_slots[layer_idx] -= size
        return slots

    def _ensure_positions_resident(
        self,
        layer_idx: int,
        row_positions: dict[int, np.ndarray],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """按 eager 策略提交新工作集，并释放不再使用的 KV。"""
        mirror = self.gpu_req_to_token_slots_cpu[layer_idx]
        load_cpu_slots: list[int] = []
        load_gpu_slots: list[int] = []
        dirty_cpu_slots: list[int] = []
        dirty_gpu_slots: list[int] = []

        with profiler.record("attnpredict_offload_residency_plan_cpu"):
            with self._layer_locks[layer_idx]:
                dirty = self._dirty_gpu_positions[layer_idx]
                active_rows = self._active_positions[layer_idx]
                resident_rows = self._resident_positions[layer_idx]
                for row_idx, positions in row_positions.items():
                    full_len = int(self.row_seq_lens[row_idx])
                    positions = positions[(positions >= 0) & (positions < full_len)]
                    positions_list = positions.tolist()
                    target = set(positions_list)
                    resident = resident_rows.setdefault(row_idx, set())

                    if not resident and full_len > 0:
                        dense_positions = np.flatnonzero(
                            mirror[row_idx, :full_len] >= 0
                        )
                        evict = dense_positions[
                            ~np.isin(dense_positions, positions)
                        ]
                        if evict.size:
                            slots = mirror[row_idx, evict].astype(np.int32, copy=True)
                            mirror[row_idx, evict] = -1
                            self.free_slots_stack[layer_idx].extend(slots.tolist())
                            self._num_free_slots[layer_idx] += int(slots.size)
                        resident.update(
                            pos for pos in positions_list if mirror[row_idx, pos] >= 0
                        )

                    active_rows[row_idx] = target
                    for pos in list(resident - target):
                        slot = int(mirror[row_idx, pos])
                        key = (row_idx, pos)
                        if key in dirty:
                            dirty.remove(key)
                            dirty_cpu_slots.append(
                                int(self.cpu_req_to_token_slots[row_idx, pos])
                            )
                            dirty_gpu_slots.append(slot)
                        else:
                            self.free_slots_stack[layer_idx].append(slot)
                            self._num_free_slots[layer_idx] += 1
                        mirror[row_idx, pos] = -1
                        resident.remove(pos)

                    missing = [pos for pos in positions_list if pos not in resident]
                    if not missing:
                        continue
                    gpu_slots = self._allocate_gpu_slots_locked(
                        layer_idx, len(missing)
                    )
                    for pos, gpu_slot in zip(missing, gpu_slots.tolist()):
                        mirror[row_idx, pos] = gpu_slot
                        resident.add(pos)
                        load_cpu_slots.append(
                            int(self.cpu_req_to_token_slots[row_idx, pos])
                        )
                        load_gpu_slots.append(gpu_slot)

        self._copy_dirty_gpu_to_cpu(
            layer_idx, dirty_cpu_slots, dirty_gpu_slots, stream=stream
        )
        if dirty_gpu_slots:
            with self._layer_locks[layer_idx]:
                self.free_slots_stack[layer_idx].extend(dirty_gpu_slots)
                self._num_free_slots[layer_idx] += len(dirty_gpu_slots)
        self._copy_cpu_to_gpu(
            layer_idx, load_cpu_slots, load_gpu_slots, stream=stream
        )

    def _ensure_lease_blocks_resident(
        self,
        layer_idx: int,
        row_hot_blocks: dict[int, np.ndarray],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """按 block 代际增量提交 lease，并只处理新增或过期 block。"""
        mirror = self.gpu_req_to_token_slots_cpu[layer_idx]
        dirty_cpu_slots: list[int] = []
        dirty_gpu_slots: list[int] = []
        load_candidates: dict[int, np.ndarray] = {}

        with profiler.record("attnpredict_offload_residency_plan_cpu"):
            with self._layer_locks[layer_idx]:
                resident_rows = self._resident_positions[layer_idx]
                active_rows = self._lease_active_blocks[layer_idx]
                grace_rows = self._lease_grace_blocks[layer_idx]
                dirty = self._dirty_gpu_positions[layer_idx]

                for row_idx, hot_blocks in row_hot_blocks.items():
                    full_len = int(self.row_seq_lens[row_idx])
                    history_end = max(0, full_len - 1)
                    target = self._lease_target_blocks(
                        hot_blocks, full_len, history_end
                    )
                    target_positions = self._positions_from_blocks(target, history_end)
                    resident = resident_rows.setdefault(row_idx, set())

                    if row_idx not in active_rows:
                        # 首次从 dense prefill 收缩；这是唯一一次扫描完整逻辑行。
                        resident_positions = np.flatnonzero(
                            mirror[row_idx, :history_end] >= 0
                        )
                        target_array = np.fromiter(target, dtype=np.int64)
                        keep = (
                            np.isin(
                                resident_positions // int(self.pooling_block_size),
                                target_array,
                            )
                            if target_array.size
                            else np.zeros(resident_positions.size, dtype=np.bool_)
                        )
                        evict_positions = resident_positions[~keep]
                        if evict_positions.size:
                            evict_slots = mirror[row_idx, evict_positions].astype(
                                np.int32, copy=True
                            )
                            mirror[row_idx, evict_positions] = -1
                            self.free_slots_stack[layer_idx].extend(evict_slots.tolist())
                            self._num_free_slots[layer_idx] += int(evict_slots.size)
                        resident.update(
                            target_positions[
                                mirror[row_idx, target_positions] >= 0
                            ].tolist()
                        )
                        active_rows[row_idx] = target
                        grace_rows[row_idx] = set()
                        load_candidates[row_idx] = target_positions
                        self._active_positions[layer_idx].pop(row_idx, None)
                        continue

                    old_active = active_rows[row_idx]
                    old_grace = grace_rows.get(row_idx, set())
                    expired = old_grace - old_active - target
                    evict_positions = self._positions_from_blocks(expired, history_end)
                    for pos in evict_positions.tolist():
                        slot = int(mirror[row_idx, pos])
                        if slot < 0:
                            continue
                        key = (row_idx, pos)
                        is_dirty = key in dirty
                        if is_dirty:
                            cpu_slot = int(self.cpu_req_to_token_slots[row_idx, pos])
                            if cpu_slot < 0:
                                raise RuntimeError(
                                    f"Missing CPU backing slot for dirty KV: "
                                    f"layer={layer_idx} row={row_idx} pos={pos}"
                                )
                            dirty.remove(key)
                            dirty_cpu_slots.append(cpu_slot)
                            dirty_gpu_slots.append(slot)
                        else:
                            self.free_slots_stack[layer_idx].append(slot)
                            self._num_free_slots[layer_idx] += 1
                        mirror[row_idx, pos] = -1
                        resident.discard(pos)
                        self._lease_grace_evictions += 1

                    # 当前代降为 grace；两代之外的 block 已在上面定点回收。
                    active_rows[row_idx] = target
                    grace_rows[row_idx] = old_active - target
                    load_candidates[row_idx] = self._positions_from_blocks(
                        target - old_active, history_end
                    )

        self._copy_dirty_gpu_to_cpu(
            layer_idx, dirty_cpu_slots, dirty_gpu_slots, stream=stream
        )

        load_cpu_slots: list[int] = []
        load_gpu_slots: list[int] = []
        with self._layer_locks[layer_idx]:
            if dirty_gpu_slots:
                self.free_slots_stack[layer_idx].extend(dirty_gpu_slots)
                self._num_free_slots[layer_idx] += len(dirty_gpu_slots)

            resident_rows = self._resident_positions[layer_idx]
            for row_idx, candidates in load_candidates.items():
                if candidates.size == 0:
                    continue
                missing = candidates[mirror[row_idx, candidates] < 0]
                if missing.size == 0:
                    continue
                gpu_slots = self._allocate_gpu_slots_locked(
                    layer_idx, int(missing.size)
                )
                resident = resident_rows.setdefault(row_idx, set())
                for pos, gpu_slot in zip(missing.tolist(), gpu_slots.tolist()):
                    cpu_slot = int(self.cpu_req_to_token_slots[row_idx, pos])
                    if cpu_slot < 0:
                        raise RuntimeError(
                            f"Missing CPU backing slot: layer={layer_idx} "
                            f"row={row_idx} pos={pos}"
                        )
                    mirror[row_idx, pos] = gpu_slot
                    resident.add(pos)
                    load_cpu_slots.append(cpu_slot)
                    load_gpu_slots.append(gpu_slot)

        self._copy_cpu_to_gpu(
            layer_idx, load_cpu_slots, load_gpu_slots, stream=stream
        )

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
                    positions_list = positions.tolist()
                    missing = [
                        pos
                        for pos in positions_list
                        if int(mirror[row_idx, pos]) < 0
                    ]
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

    def _lease_target_blocks(
        self,
        hot_blocks: np.ndarray,
        full_len: int,
        history_end: int,
    ) -> set[int]:
        """返回本次 attention 历史视图覆盖的逻辑 block。"""
        block_size = int(self.pooling_block_size)
        history_end = min(max(0, int(history_end)), int(full_len))
        if history_end <= 0:
            return set()

        sink_end = min(int(self.sink_token), history_end)
        recent_start = max(min(int(self.sink_token), int(full_len)), int(full_len) - int(self.local_token))
        blocks = set(range((sink_end + block_size - 1) // block_size))
        if hot_blocks.size:
            blocks.update(
                int(block)
                for block in hot_blocks.tolist()
                if int(block) >= 0 and int(block) * block_size < history_end
            )
        if recent_start < history_end:
            blocks.update(range(recent_start // block_size, (history_end - 1) // block_size + 1))
        return blocks

    def _positions_from_blocks(
        self,
        blocks: set[int],
        logical_end: int,
    ) -> np.ndarray:
        """把少量 block id 一次性展开为有效逻辑 token 位置。"""
        if not blocks or logical_end <= 0:
            return np.empty((0,), dtype=np.int64)
        block_size = int(self.pooling_block_size)
        block_ids = np.fromiter(sorted(blocks), dtype=np.int64)
        positions = (
            block_ids[:, None] * block_size
            + np.arange(block_size, dtype=np.int64)[None, :]
        ).reshape(-1)
        return positions[positions < int(logical_end)]

    def _lease_prefetch_positions(
        self,
        layer_idx: int,
        hot_blocks: dict[int, np.ndarray],
        lease_start_positions: dict[int, int],
    ) -> dict[int, np.ndarray]:
        """只展开新 lease 相对当前 lease 新增的 block。"""
        result: dict[int, np.ndarray] = {}
        with self._layer_locks[layer_idx]:
            active_rows = self._lease_active_blocks[layer_idx]
            for row_idx, full_len in lease_start_positions.items():
                # 首份 lease 产生于 prefill 尾部，此时完整 prompt 已驻留，无需预取。
                if row_idx not in active_rows:
                    result[row_idx] = np.empty((0,), dtype=np.int64)
                    continue
                history_end = max(0, int(full_len) - 1)
                target = self._lease_target_blocks(
                    hot_blocks[row_idx], int(full_len), history_end
                )
                added = target - active_rows[row_idx]
                result[row_idx] = self._positions_from_blocks(added, history_end)
        return result
