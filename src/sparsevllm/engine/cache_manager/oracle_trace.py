from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from sparsevllm.config import Config
from sparsevllm.utils.context import get_context

from .standard import StandardCacheManager


class OracleTraceCacheManager(StandardCacheManager):
    """执行完整注意力，并保存预测器离线评估所需的精确 block 轨迹。"""

    def __init__(self, config: Config, rank: int, world_size: int):
        super().__init__(config, rank, world_size)
        self.trace_dir = Path(config.attnpredict_oracle_trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.block_size = int(config.attnpredict_pooling_block_size)
        self.history_steps = int(config.attnpredict_history_steps)
        self.layer_stride = int(config.attnpredict_oracle_layer_stride)
        self.max_steps = int(config.attnpredict_oracle_max_steps)
        self.attn_scale = self.head_dim**-0.5
        self.source_layers = set(range(0, self.num_layers, self.layer_stride))

        self._pending_prefill: list[dict[str, object] | None] = [
            None for _ in range(self.num_layers)
        ]
        self._prefill_rows: dict[tuple[int, int], list[tuple[int, torch.Tensor]]] = {}
        self._initial_history: dict[tuple[int, int], torch.Tensor] = {}
        self._initial_context: dict[tuple[int, int], int] = {}
        self._decode_rows: dict[tuple[int, int], dict[str, list]] = {}

    def _is_source(self, layer_idx: int) -> bool:
        return layer_idx in self.source_layers

    def should_collect_decode_attn_score(self, layer_idx: int) -> bool:
        return self._is_source(layer_idx)

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
        del k_cache, active_slots, num_kv_heads
        if not self._is_source(layer_idx) or cu_seqlens_q is None:
            return None
        if cu_seqlens_q.numel() <= 1:
            return None

        max_blocks = (int(context_lens.max().item()) + self.block_size - 1) // self.block_size
        score = torch.zeros(
            (req_indices.numel(), num_heads, self.history_steps, max_blocks),
            dtype=torch.float32,
            device=q.device,
        )
        self._pending_prefill[layer_idx] = {
            "score": score,
            "rows": req_indices.detach().cpu().tolist(),
            "context_lens": context_lens.detach().cpu().tolist(),
            "chunk_lens": (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).detach().cpu().tolist(),
            "is_last": list(prefill_is_last_chunk or [False] * req_indices.numel()),
        }
        return score

    def prefill_attn_score_block_size(self, layer_idx: int) -> int | None:
        return self.block_size if self._is_source(layer_idx) else None

    def on_prefill_layer_end(self, layer_idx: int) -> None:
        pending = self._pending_prefill[layer_idx]
        if pending is None:
            return
        self._pending_prefill[layer_idx] = None
        score = pending["score"].detach().to(device="cpu", dtype=torch.float16)
        for batch_idx, (row_idx, full_len, chunk_len, is_last) in enumerate(
            zip(
                pending["rows"],
                pending["context_lens"],
                pending["chunk_lens"],
                pending["is_last"],
            )
        ):
            take = min(self.history_steps, int(chunk_len))
            if take <= 0:
                continue
            blocks = (int(full_len) + self.block_size - 1) // self.block_size
            rows = score[batch_idx, :, :take, :blocks]
            start_position = int(full_len) - take
            key = (layer_idx, int(row_idx))
            history = self._prefill_rows.setdefault(key, [])
            history.extend(
                (start_position + offset, rows[:, offset].clone())
                for offset in range(take)
            )
            if len(history) > self.history_steps:
                del history[:-self.history_steps]
            if is_last:
                final_blocks = blocks
                assembled = torch.zeros(
                    (score.shape[1], len(history), final_blocks), dtype=torch.float16
                )
                for offset, (_, row_score) in enumerate(history):
                    assembled[:, offset, : row_score.shape[-1]] = row_score
                self._initial_history[key] = assembled
                self._initial_context[key] = int(full_len)

    @staticmethod
    def _pool(values: torch.Tensor, block_size: int, reduce: str) -> torch.Tensor:
        pad = (-values.shape[-1]) % block_size
        if pad:
            fill = 0.0 if reduce == "sum" else float("-inf")
            values = F.pad(values, (0, pad), value=fill)
        values = values.reshape(values.shape[0], -1, block_size)
        if reduce == "sum":
            return values.sum(dim=-1)
        return values.amax(dim=-1).clamp_min(0.0)

    def record_decode_attention(self, layer_idx: int, logits: torch.Tensor) -> None:
        if not self._is_source(layer_idx):
            return
        state = self.get_layer_batch_states(layer_idx)
        rows = state.req_indices.detach().cpu().tolist()
        lengths = state.context_lens.detach().cpu().tolist()
        for batch_idx, (row_idx, full_len) in enumerate(zip(rows, lengths)):
            key = (layer_idx, int(row_idx))
            record = self._decode_rows.setdefault(
                key,
                {
                    "scores": [],
                    "block_mass": [],
                    "middle_block_mass": [],
                    "fixed_block_scores": [],
                    "fixed_mass": [],
                    "context_positions": [],
                },
            )
            if len(record["scores"]) >= self.max_steps:
                continue
            full_len = int(full_len)
            prob = torch.softmax(
                logits[batch_idx, :, :full_len].to(torch.float32) * self.attn_scale,
                dim=-1,
            )
            sink_end = min(int(self.config.num_sink_tokens), full_len)
            recent_start = max(sink_end, full_len - int(self.config.num_recent_tokens))
            fixed = torch.zeros(full_len, dtype=torch.bool, device=prob.device)
            fixed[:sink_end] = True
            fixed[recent_start:] = True
            middle = ~fixed

            record["scores"].append(
                self._pool(prob, self.block_size, "max").to("cpu", torch.float16)
            )
            record["block_mass"].append(
                self._pool(prob, self.block_size, "sum").to("cpu", torch.float16)
            )
            record["middle_block_mass"].append(
                self._pool(prob * middle, self.block_size, "sum").to("cpu", torch.float16)
            )
            record["fixed_block_scores"].append(
                self._pool(prob.masked_fill(~fixed, 0.0), self.block_size, "max").to(
                    "cpu", torch.float16
                )
            )
            record["fixed_mass"].append(
                prob[:, fixed].sum(dim=-1).to("cpu", torch.float32)
            )
            record["context_positions"].append(full_len)

    def _flush_trace(self, seq_id: int, row_idx: int) -> None:
        for layer_idx in sorted(self.source_layers):
            key = (layer_idx, row_idx)
            record = self._decode_rows.pop(key, None)
            history = self._initial_history.pop(key, None)
            initial_context = self._initial_context.pop(key, None)
            self._prefill_rows.pop(key, None)
            if record is None or not record["scores"] or history is None:
                continue
            max_blocks = max(item.shape[-1] for item in record["scores"])
            payload = {}
            for name in (
                "scores",
                "block_mass",
                "middle_block_mass",
                "fixed_block_scores",
            ):
                payload[name] = torch.stack(
                    [F.pad(item, (0, max_blocks - item.shape[-1])) for item in record[name]]
                )
            payload["fixed_mass"] = torch.stack(record["fixed_mass"])
            payload["initial_history"] = F.pad(
                history, (0, max_blocks - history.shape[-1])
            )
            payload["metadata"] = {
                "trace_format_version": 3,
                "seq_id": int(seq_id),
                "layer_idx": int(layer_idx),
                "context_positions": record["context_positions"],
                "initial_context_position": int(initial_context),
                "block_size": self.block_size,
                "sink_tokens": int(self.config.num_sink_tokens),
                "recent_tokens": int(self.config.num_recent_tokens),
                "topk_tokens": int(self.config.num_top_tokens),
                "model": str(self.config.model),
            }
            torch.save(payload, self.trace_dir / f"seq_{seq_id}_layer_{layer_idx}.pt")

    def free_seq(self, seq_id: int):
        row_idx = self.seq_id_to_row.get(seq_id)
        if row_idx is not None:
            self._flush_trace(int(seq_id), int(row_idx))
        return super().free_seq(seq_id)
