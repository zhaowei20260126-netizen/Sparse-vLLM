from __future__ import annotations

import torch
import torch.nn.functional as F

from sparsevllm.config import Config
from sparsevllm.utils.profiler import profiler

from .standard import StandardCacheManager
from .attnpredict_cnn import AttnPredictCNN


class AttnPredictCacheManager(StandardCacheManager):
    """KV cache manager with AttentionPredictor-guided token selection.

    The original AttentionPredictor cache keeps a rolling attention history per
    layer and predicts the next step's sparse KV set. Sparse-vLLM keeps full KV
    on GPU in this v1 integration, so the predicted set is applied as a logical
    decode view rather than as a CPU-to-GPU KV prefetch buffer.
    """

    def __init__(self, config: Config, rank: int, world_size: int):
        super().__init__(config, rank, world_size)

        self.topk = int(config.attnpredict_topk)
        self.history_step = int(config.attnpredict_history_steps)
        self.pooling_block_size = int(config.attnpredict_pooling_block_size)
        self.sink_token = int(config.attnpredict_sink_tokens)
        self.local_token = int(config.attnpredict_local_tokens)
        self.attn_scale = self.head_dim ** -0.5

        # Per-layer, per-cache-row state. Cache rows survive across decode steps;
        # batch positions do not, so row keys are the stable sequence identity.
        self.attn_history: list[dict[int, torch.Tensor]] = [
            {} for _ in range(self.num_layers)
        ]
        self.tsp_mask: list[dict[int, torch.Tensor]] = [
            {} for _ in range(self.num_layers)
        ]
        self._last_decode_view: list[dict[str, torch.Tensor | None] | None] = [
            None for _ in range(self.num_layers)
        ]

        self.cnn = AttnPredictCNN()
        model_path = str(config.attnpredict_model_path or "")
        state_dict = torch.load(model_path, map_location="cuda", weights_only=False)
        self.cnn.load_state_dict(state_dict)
        self.cnn.to(dtype=torch.float16, device="cuda")
        self.cnn.eval()
        self.cnn_dtype = next(self.cnn.parameters()).dtype

    def free_seq(self, seq_id: int):
        row_idx = self.seq_id_to_row.get(seq_id)
        super().free_seq(seq_id)
        if row_idx is None:
            return
        for layer_idx in range(self.num_layers):
            self.attn_history[layer_idx].pop(int(row_idx), None)
            self.tsp_mask[layer_idx].pop(int(row_idx), None)

    @torch.no_grad()
    def observe_prefill_attention(
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
    ) -> None:
        """Initialize/update history from the last history_step prefill queries.

        This mirrors the original implementation's prefill branch, which computes
        attention for the last 64 query rows and feeds those rows into the CNN so
        the first decode step can already use a predicted sparse view.
        """
        if cu_seqlens_q is None or cu_seqlens_q.numel() <= 1:
            return

        with profiler.record("attnpredict_observe_prefill_attention"):
            group_size = max(1, num_heads // max(1, num_kv_heads))
            batch_size = int(req_indices.numel())
            for b in range(batch_size):
                q_end = int(cu_seqlens_q[b + 1].item())
                q_start = int(cu_seqlens_q[b].item())
                q_len = q_end - q_start
                if q_len <= 0:
                    continue

                full_len = int(context_lens[b].item())
                if full_len <= 0:
                    continue

                row_idx = int(req_indices[b].item())
                take = min(self.history_step, q_len)
                q_tail = q[q_end - take:q_end].to(torch.float32)

                slots = active_slots[row_idx, :full_len].to(torch.long)
                k_full = k_cache.index_select(0, slots).to(torch.float32)
                if group_size > 1:
                    k_full = k_full.repeat_interleave(group_size, dim=1)
                k_full = k_full[:, :num_heads, :]

                logits = torch.einsum("thd,lhd->htl", q_tail, k_full)
                logits *= self.attn_scale

                q_positions = torch.arange(
                    full_len - take, full_len, device=q.device
                )
                kv_positions = torch.arange(full_len, device=q.device)
                causal_mask = kv_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
                logits = logits.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
                attn = torch.softmax(logits, dim=-1).to(self.hf_config.torch_dtype)

                self._update_row_prediction(layer_idx, row_idx, attn)

    @torch.no_grad()
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
        """Apply the previous step's predicted mask to the current decode view."""
        with profiler.record("attnpredict_build_decode_view"):
            batch_size = int(req_indices.numel())
            row_positions: list[torch.Tensor] = []
            keep_counts: list[int] = []
            any_sparse = False

            for b in range(batch_size):
                row_idx = int(req_indices[b].item())
                full_len = int(context_lens[b].item())
                if full_len <= 0:
                    positions = torch.empty(0, dtype=torch.long, device=q.device)
                    row_positions.append(positions)
                    keep_counts.append(0)
                    continue

                pred_mask = self.tsp_mask[layer_idx].get(row_idx)
                if pred_mask is None:
                    positions = torch.arange(full_len, dtype=torch.long, device=q.device)
                else:
                    any_sparse = True
                    valid_len = min(int(pred_mask.numel()), full_len)
                    if valid_len > 0:
                        positions = pred_mask[:valid_len].nonzero(as_tuple=False).squeeze(-1)
                        positions = positions.to(device=q.device, dtype=torch.long)
                    else:
                        positions = torch.empty(0, dtype=torch.long, device=q.device)

                    # The current decode token was just appended to the physical
                    # cache. The previous prediction cannot know it, but the
                    # original implementation always concatenates this newest KV.
                    current_pos = torch.tensor([full_len - 1], dtype=torch.long, device=q.device)
                    positions = torch.unique(torch.cat([positions, current_pos]), sorted=True)

                row_positions.append(positions)
                keep_counts.append(int(positions.numel()))

            if not any_sparse:
                self._last_decode_view[layer_idx] = {
                    "req_indices": req_indices.detach().clone(),
                    "positions": None,
                    "view_lens": context_lens.detach().clone(),
                    "full_context_lens": context_lens.detach().clone(),
                }
                return active_slots, req_indices, context_lens

            max_keep = max(keep_counts) if keep_counts else 0
            if max_keep <= 0:
                self._last_decode_view[layer_idx] = None
                return active_slots, req_indices, context_lens

            packed_slots = torch.full(
                (batch_size, max_keep), -1, dtype=torch.int32, device=q.device
            )
            packed_positions = torch.full(
                (batch_size, max_keep), -1, dtype=torch.int32, device=q.device
            )
            for b, positions in enumerate(row_positions):
                k = int(positions.numel())
                if k == 0:
                    continue
                row_idx = int(req_indices[b].item())
                packed_positions[b, :k] = positions.to(torch.int32)
                packed_slots[b, :k] = active_slots[row_idx, positions].to(torch.int32)

            view_lens = torch.tensor(keep_counts, dtype=torch.int32, device=q.device)
            local_req_indices = torch.arange(batch_size, dtype=torch.int32, device=q.device)
            self._last_decode_view[layer_idx] = {
                "req_indices": req_indices.detach().clone(),
                "positions": packed_positions.detach(),
                "view_lens": view_lens.detach().clone(),
                "full_context_lens": context_lens.detach().clone(),
            }
            return packed_slots, local_req_indices, view_lens

    @torch.no_grad()
    def predict_next_mask(self, layer_idx: int, attn_logits: torch.Tensor) -> None:
        """Update history from decode attention logits and predict next mask."""
        view = self._last_decode_view[layer_idx]
        if view is None:
            return

        with profiler.record("attnpredict_predict_next_mask"):
            if attn_logits.dim() == 2:
                attn_logits = attn_logits.unsqueeze(1)

            req_indices = view["req_indices"]
            positions = view["positions"]
            view_lens = view["view_lens"]
            full_context_lens = view["full_context_lens"]
            assert req_indices is not None
            assert view_lens is not None
            assert full_context_lens is not None

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
                    full_attn = torch.zeros(
                        (attn.shape[0], full_len),
                        dtype=attn.dtype,
                        device=attn.device,
                    )
                    full_attn.scatter_(1, pos.unsqueeze(0).expand(attn.shape[0], -1), attn)

                self._update_row_prediction(
                    layer_idx,
                    row_idx,
                    full_attn.unsqueeze(1).to(self.hf_config.torch_dtype),
                )

    # ---------- AttentionPredictor helpers ----------

    def _update_row_prediction(
        self,
        layer_idx: int,
        row_idx: int,
        attn_weights_full: torch.Tensor,
    ) -> None:
        """Update one row's history and store its predicted keep mask.

        Args:
            attn_weights_full: (num_heads, q_rows, full_seq_len) softmax weights.
        """
        hist = self._update_attn_history(
            self.attn_history[layer_idx].get(row_idx),
            attn_weights_full,
        )
        self.attn_history[layer_idx][row_idx] = hist

        tsp_attn, start_block = self._time_sequence_predict(hist)
        seq_len = int(attn_weights_full.shape[-1])
        self.tsp_mask[layer_idx][row_idx] = self._create_tsp_mask(
            tsp_attn,
            seq_len=seq_len,
            start_block=start_block,
            device=attn_weights_full.device,
        )

    def _max_pooling(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pool the attention sequence dimension exactly like the original code."""
        padding_size = (self.pooling_block_size - tensor.shape[-1] % self.pooling_block_size) % self.pooling_block_size
        if padding_size:
            tensor = F.pad(tensor, (0, padding_size))
        pooled = tensor.view(*tensor.shape[:-1], -1, self.pooling_block_size).max(dim=-1).values
        return pooled

    def _update_attn_history(
        self,
        attn_history: torch.Tensor | None,
        attn_weights_full: torch.Tensor,
    ) -> torch.Tensor:
        attn_pooling = self._max_pooling(attn_weights_full)
        if attn_pooling.shape[-2] > self.history_step:
            attn_pooling = attn_pooling[:, -self.history_step:, :]

        if attn_history is None:
            if attn_pooling.shape[-2] < self.history_step:
                pad_rows = self.history_step - attn_pooling.shape[-2]
                attn_pooling = F.pad(attn_pooling, (0, 0, pad_rows, 0))
            return attn_pooling

        old_len = int(attn_history.shape[-1])
        new_len = int(attn_pooling.shape[-1])
        if new_len > old_len:
            attn_history = F.pad(attn_history, (0, new_len - old_len))
        elif new_len < old_len:
            attn_history = attn_history[..., :new_len]

        hist = torch.cat([attn_history, attn_pooling], dim=-2)
        return hist[:, -self.history_step:, :]

    def _time_sequence_predict(self, attn_history: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Run the CNN over the non-sink/non-local pooled block range."""
        num_heads, num_rows, attn_len = attn_history.shape
        start = self.sink_token // self.pooling_block_size
        end = attn_len - (self.local_token // self.pooling_block_size)
        end = max(start, end)
        attn_history = attn_history[:, :, start:end]
        pred_len = int(attn_history.shape[-1])

        if pred_len < 3:
            return torch.ones(
                (num_heads, pred_len),
                dtype=attn_history.dtype,
                device=attn_history.device,
            ), start

        inputs = attn_history.reshape(num_heads, num_rows, pred_len)
        tsp_attn = self.cnn(inputs.to(self.cnn_dtype).contiguous())
        return tsp_attn.to(torch.float32), start

    def _create_tsp_mask(
        self,
        tsp_attn: torch.Tensor,
        *,
        seq_len: int,
        start_block: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create a shared token-level keep mask from block predictions."""
        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        if seq_len <= 0:
            return keep_mask

        sink_end = min(self.sink_token, seq_len)
        keep_mask[:sink_end] = True

        local_start = max(sink_end, seq_len - self.local_token)
        keep_mask[local_start:] = True

        pred_len = int(tsp_attn.shape[-1])
        if pred_len < 1:
            return keep_mask

        block_budget = (self.topk - self.sink_token - self.local_token) // self.pooling_block_size
        block_budget = max(0, min(block_budget, pred_len))
        if block_budget <= 0:
            return keep_mask

        block_scores = tsp_attn.max(dim=0).values if tsp_attn.dim() == 2 else tsp_attn
        _, topk_indices = torch.topk(block_scores, block_budget, dim=-1)
        token_indices = (
            (topk_indices + start_block).unsqueeze(-1) * self.pooling_block_size
            + torch.arange(self.pooling_block_size, device=device)
        ).reshape(-1)
        token_indices = token_indices[token_indices < seq_len]
        keep_mask[token_indices] = True
        return keep_mask
