from __future__ import annotations

import torch

from sparsevllm.config import Config
from sparsevllm.utils.profiler import profiler

from .base import CacheManager
from .standard import StandardCacheManager
from .attnpredict_cnn import AttnPredictCNN


class AttnPredictCacheManager(StandardCacheManager):
    """KV cache manager with AttentionPredictor-guided token selection.

    Extends StandardCacheManager (full KV on GPU). During decode, a small CNN
    predicts which historical token blocks will be important for the NEXT step.
    The predicted mask is applied via build_decode_view() to filter active_slots.

    Prediction pipeline:
      decode step t:
        1. attention produces attn_weights (softmax scores)
        2. on_layer_end → predict_next_mask():
           max_pool(x16) → update rolling history → CNN forward → create mask
           → store in self.tsp_mask[layer_idx]
      decode step t+1:
        3. build_decode_view() applies tsp_mask[layer_idx] to filter slots
    """

    def __init__(self, config: Config, rank: int, world_size: int):
        super().__init__(config, rank, world_size)

        self.topk = int(config.attnpredict_topk)
        self.history_step = int(config.attnpredict_history_steps)
        self.pooling_block_size = int(config.attnpredict_pooling_block_size)
        self.sink_token = int(config.attnpredict_sink_tokens)
        self.local_token = int(config.attnpredict_local_tokens)

        # Per-layer state
        self.attn_history: list[torch.Tensor | None] = [None] * self.num_layers
        self.tsp_mask: list[torch.Tensor | None] = [None] * self.num_layers

        # Shared CNN predictor (float16 for efficiency, same as original)
        self.cnn = AttnPredictCNN()
        model_path = config.attnpredict_model_path
        if model_path:
            state_dict = torch.load(model_path, map_location="cuda", weights_only=False)
            self.cnn.load_state_dict(state_dict)
        self.cnn.to(dtype=self.hf_config.torch_dtype, device="cuda")
        self.cnn.eval()

        # JIT-scripted MaxPool1d for block-wise pooling
        self.pooling = torch.nn.MaxPool1d(
            kernel_size=self.pooling_block_size,
            stride=self.pooling_block_size,
            padding=0,
            ceil_mode=True,
        )
        self.pooling = torch.jit.script(self.pooling).eval()

    @torch.no_grad()
    def predict_next_mask(self, layer_idx: int, attn_weights: torch.Tensor):
        """Update attention history and predict mask for the next decode step.

        Called from SparseController.on_layer_end() after attention produces
        softmax scores.

        Args:
            layer_idx: current layer index
            attn_weights: (B, seq_len) — already max-pooled over heads
        """
        # 1. Pool attention weights into blocks of size pooling_block_size
        pooled = self._max_pool_1d(attn_weights)  # (B, seq_len) → (B, seq_len//block)

        # 2. Update rolling history window
        self._update_attn_history(layer_idx, pooled)

        # 3. CNN predict future attention from history
        hist = self.attn_history[layer_idx]
        if hist is None:
            return  # Not enough history yet; first 64 steps use full attention

        tsp_attn = self._cnn_predict(hist)  # (B, pooled_seq_len)

        # 4. Create token-level mask from block predictions
        self.tsp_mask[layer_idx] = self._create_tsp_mask(layer_idx, tsp_attn)

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
        """Override to filter active_slots based on predicted attention mask.

        Follows the QuEST pattern: if mask is available, pack only selected
        KV slots into a dense tensor and return adjusted context lengths.
        """
        mask = self.tsp_mask[layer_idx]
        if mask is None:
            return active_slots, req_indices, context_lens

        with profiler.record("attnpredict_build_decode_view"):
            batch_size = mask.shape[0]
            # mask: (B, seq_len); 0=keep, -10000=drop
            keep_mask = mask == 0
            keep_counts = keep_mask.sum(dim=-1)   # (B,)
            max_keep = int(keep_counts.max().item())

            if max_keep == 0:
                return active_slots, req_indices, context_lens

            # Pack selected slots into a dense [B, max_keep] tensor
            packed_slots = torch.full(
                (batch_size, max_keep), -1, dtype=torch.int32, device=mask.device
            )
            # We need to map [req_idx, keep_position] → physical slot.
            # active_slots is (batch_size, max_model_len) from get_read_view.
            # Each row b needs: active_slots[b][keep_mask[b]] → packed_slots[b]
            for b in range(batch_size):
                k = int(keep_counts[b].item())
                if k > 0:
                    # keep_mask[b] has True at positions to keep; gather those slots
                    row_keep = keep_mask[b].nonzero(as_tuple=False).squeeze(-1)[:max_keep]
                    packed_slots[b, :k] = active_slots[b, row_keep]

            local_req_indices = torch.arange(batch_size, dtype=torch.int32, device=mask.device)
            return packed_slots, local_req_indices, keep_counts.to(torch.int32)

    # ---------- Internal helpers ----------

    def _max_pool_1d(self, attn: torch.Tensor) -> torch.Tensor:
        """Max-pool attention weights along seq dim with pooling_block_size.

        Args:
            attn: (B, seq_len) — already head-pooled

        Returns:
            pooled: (B, 1, seq_len//block_size)
        """
        orig_shape = attn.shape
        # Reshape for MaxPool1d: [B, 1, L]
        x = attn.view(attn.shape[0], 1, -1)
        if x.shape[-1] < self.pooling_block_size:
            # Pad to at least one block
            pad = self.pooling_block_size - x.shape[-1]
            x = torch.nn.functional.pad(x, (0, pad))
        pooled = self.pooling(x.to(torch.float32))
        return pooled.to(attn.dtype)  # (B, 1, L//block)

    def _update_attn_history(self, layer_idx: int, pooled: torch.Tensor):
        """Maintain a rolling window of 64 pooled attention snapshots.

        pooled: (B, 1, pooled_len) — single-step attention pooled into blocks.
        """
        hist = self.attn_history[layer_idx]
        if hist is None:
            # Initialize: take last history_step columns (padding if not enough)
            if pooled.shape[-1] >= self.history_step:
                hist = pooled[:, :, -self.history_step:]
            else:
                pad = self.history_step - pooled.shape[-1]
                hist = torch.nn.functional.pad(pooled, (pad, 0))
            self.attn_history[layer_idx] = hist
            return

        if pooled.shape[-1] == hist.shape[-1]:
            # seq_len unchanged: roll up and replace last row
            hist = torch.roll(hist, shifts=-1, dims=-2)
            hist[:, :, -1:] = pooled
        else:
            # seq_len grew: pad history, then replace last row
            new_hist = torch.zeros(
                hist.shape[0], hist.shape[1], hist.shape[2], pooled.shape[-1],
                device=hist.device, dtype=hist.dtype,
            )
            new_hist[:, :, :, :hist.shape[-1]] = hist
            new_hist = torch.roll(new_hist, shifts=-1, dims=-2)
            new_hist[:, :, -1:] = pooled
            hist = new_hist

        self.attn_history[layer_idx] = hist

    def _cnn_predict(self, hist: torch.Tensor) -> torch.Tensor:
        """Run CNN to predict future attention from history.

        Args:
            hist: (B, 1, history_steps=64, pooled_seq_len)

        Returns:
            tsp_attn: (B, pooled_seq_len)
        """
        # CNN expects (B, history_steps, pooled_seq_len)
        x = hist.squeeze(1)  # (B, 64, pooled_len)
        return self.cnn(x.contiguous())

    def _create_tsp_mask(self, layer_idx: int, tsp_attn: torch.Tensor) -> torch.Tensor:
        """Create token-level mask from block-level predictions.

        Always keeps: sink_token (prefix) + local_token (suffix) + topk blocks.

        Args:
            layer_idx: current layer index
            tsp_attn: (B, pooled_seq_len) — CNN-predicted block importance

        Returns:
            mask: (B, seq_len) — 0=keep, -10000=drop
        """
        batch_size = tsp_attn.shape[0]
        seq_len = self.row_seq_lens[self.seq_id_to_row.get(
            list(self.seq_id_to_row.keys())[0], 0
        )] if self.seq_id_to_row else 0

        # Use max context length from current batch state
        ctx_lens = self.layer_batch_state.context_lens
        if ctx_lens is not None:
            max_seq_len = int(ctx_lens.max().item())
        else:
            max_seq_len = seq_len

        mask = torch.full(
            (batch_size, max_seq_len), -10000.0,
            device=tsp_attn.device, dtype=torch.float32,
        )

        # Always keep sink tokens
        sink_end = min(self.sink_token, max_seq_len)
        mask[:, :sink_end] = 0

        # Always keep local tokens (last N tokens)
        local_start = max(sink_end, max_seq_len - self.local_token)
        mask[:, local_start:] = 0

        # Select top-k blocks from CNN prediction
        pooled_len = tsp_attn.shape[-1]
        if pooled_len > 0:
            # How many blocks to select (excluding sink and local blocks)
            sink_blocks = self.sink_token // self.pooling_block_size
            local_blocks = self.local_token // self.pooling_block_size
            num_blocks_to_select = (self.topk - self.sink_token - self.local_token) // self.pooling_block_size
            num_blocks_to_select = max(0, min(num_blocks_to_select, pooled_len - sink_blocks - local_blocks))

            if num_blocks_to_select > 0:
                _, topk_block_indices = torch.topk(tsp_attn[:, sink_blocks:pooled_len - local_blocks],
                                                   num_blocks_to_select, dim=-1)
                # Convert block indices to token indices
                token_indices = (topk_block_indices + sink_blocks).unsqueeze(-1) * self.pooling_block_size \
                    + torch.arange(self.pooling_block_size, device=tsp_attn.device)
                token_indices = token_indices.view(batch_size, -1)
                # Clamp to valid token range
                token_indices = token_indices.clamp(0, max_seq_len - 1)

                for b in range(batch_size):
                    valid_idx = token_indices[b][token_indices[b] < max_seq_len]
                    mask[b, valid_idx] = 0

        return mask