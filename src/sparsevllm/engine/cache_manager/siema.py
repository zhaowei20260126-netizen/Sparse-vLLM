from __future__ import annotations

import torch
import torch.nn.functional as F

from sparsevllm.config import Config

from .predictive_offload import (
    PredictiveOffloadCacheManager,
    prediction_block_window,
)


class SIEMACacheManager(PredictiveOffloadCacheManager):
    """使用 SI-EMA 或普通 EMA 估计关键 block。"""

    def _init_predictor(self, config: Config) -> None:
        self.ema_alpha = float(config.siema_alpha)
        self.scale_invariant = bool(config.siema_scale_invariant)
        self.predictor_name = "siema" if self.scale_invariant else "ema"
        # SI-EMA 对公共尺度不敏感，可以直接使用 block-logit 快速路径。
        self.prefill_uses_block_logits = self.scale_invariant
        self.ema_scores: list[dict[int, torch.Tensor]] = [
            {} for _ in range(self.num_layers)
        ]
        # 记录每个 block 是否已有真实观测，避免新 block 从零开始被系统性压低。
        self.ema_seen_blocks: list[dict[int, torch.Tensor]] = [
            {} for _ in range(self.num_layers)
        ]

    def _free_predictor_row(self, row_idx: int) -> None:
        for layer_state in self.ema_scores:
            layer_state.pop(row_idx, None)
        for layer_state in self.ema_seen_blocks:
            layer_state.pop(row_idx, None)

    def _update_predictor_scores_from_pooled(
        self,
        layer_idx: int,
        row_idx: int,
        block_scores: torch.Tensor,
        *,
        observed_block_mask: torch.Tensor | None = None,
        step_position: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        """只用真实观测更新 EMA，并返回中间历史区的预测分数。"""
        del step_position
        scores = block_scores.to(torch.float32)
        if scores.dim() == 2:
            scores = scores.unsqueeze(1)
        input_blocks = int(scores.shape[-1])

        state = self.ema_scores[layer_idx].get(row_idx)
        target_blocks = int(scores.shape[-1])
        if state is None:
            state = torch.zeros(
                (scores.shape[0], target_blocks),
                dtype=torch.float32,
                device=scores.device,
            )
            seen = torch.zeros(
                target_blocks, dtype=torch.bool, device=scores.device
            )
        elif state.shape[-1] < target_blocks:
            state = F.pad(state, (0, target_blocks - state.shape[-1]))
            seen = F.pad(
                self.ema_seen_blocks[layer_idx][row_idx],
                (0, target_blocks - self.ema_seen_blocks[layer_idx][row_idx].shape[-1]),
                value=False,
            )
        elif scores.shape[-1] < state.shape[-1]:
            scores = F.pad(scores, (0, state.shape[-1] - scores.shape[-1]))
            seen = self.ema_seen_blocks[layer_idx][row_idx]
        else:
            seen = self.ema_seen_blocks[layer_idx][row_idx]

        history_rows = int(scores.shape[1])
        if observed_block_mask is None:
            observed_rows = torch.ones(
                (history_rows, input_blocks),
                dtype=torch.bool,
                device=state.device,
            )
        else:
            observed_rows = observed_block_mask.to(
                device=state.device, dtype=torch.bool
            )
            if observed_rows.dim() == 1:
                if history_rows != 1:
                    raise ValueError("多行 block_scores 必须提供二维观测掩码")
                observed_rows = observed_rows.unsqueeze(0)
            elif observed_rows.dim() != 2:
                raise ValueError("observed_block_mask 必须是一维或二维 block 掩码")
            if observed_rows.shape != (history_rows, input_blocks):
                raise ValueError(
                    "observed_block_mask 形状必须与 [history, blocks] 一致"
                )
        if input_blocks < state.shape[-1]:
            observed_rows = F.pad(
                observed_rows,
                (0, state.shape[-1] - input_blocks),
                value=False,
            )

        for history_idx, row in enumerate(scores.unbind(dim=1)):
            observed = observed_rows[history_idx]
            if self.scale_invariant:
                scale = (row.abs() * observed).sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
                row = row / scale

            first_observation = observed & ~seen
            repeated_observation = observed & seen
            if first_observation.any():
                state[:, first_observation] = row[:, first_observation]
            if repeated_observation.any():
                state[:, repeated_observation] = (
                    state[:, repeated_observation] * (1.0 - self.ema_alpha)
                    + row[:, repeated_observation] * self.ema_alpha
                )
            seen |= observed
        self.ema_scores[layer_idx][row_idx] = state
        self.ema_seen_blocks[layer_idx][row_idx] = seen

        start, end = prediction_block_window(
            state.shape[-1],
            sink_tokens=self.sink_token,
            recent_tokens=self.local_token,
            block_size=self.pooling_block_size,
        )
        return state[:, start:end], start
