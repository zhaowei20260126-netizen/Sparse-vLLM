from __future__ import annotations

import threading

import torch
import torch.nn.functional as F
from torch import nn

from sparsevllm.config import Config

from .predictive_offload import PredictiveOffloadCacheManager, prediction_block_window


class AttnPredictCNN(nn.Module):
    """根据历史 block 分数预测下一步重要性的原始 CNN。"""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1).contiguous(memory_format=torch.channels_last)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.mean(dim=2).contiguous()
        return self.conv3(x).squeeze(1)


class AttnPredictOffloadCacheManager(PredictiveOffloadCacheManager):
    """使用 CNN 预测器的 KV 卸载缓存管理器。"""

    def _init_predictor(self, config: Config) -> None:
        self.predictor_name = "cnn"
        self.prefill_uses_block_logits = False
        self.attn_history: list[dict[int, torch.Tensor]] = [
            {} for _ in range(self.num_layers)
        ]
        self._cnn_lock = threading.RLock()

        self.cnn = AttnPredictCNN()
        model_path = str(config.attnpredict_model_path)
        state_dict = torch.load(model_path, map_location="cuda", weights_only=False)
        self.cnn.load_state_dict(state_dict)
        self.cnn.to(
            dtype=torch.float16,
            device="cuda",
            memory_format=torch.channels_last,
        )
        self.cnn_dtype = torch.float16
        self.cnn.eval()

        self.cnn = torch.compile(
            self.cnn,
            dynamic=True,
            options={"triton.cudagraphs": False},
        )
        pooled_len = (
            self.max_model_len + self.pooling_block_size - 1
        ) // self.pooling_block_size
        pred_len = max(
            3,
            pooled_len
            - self.sink_token // self.pooling_block_size
            - self.local_token // self.pooling_block_size,
        )
        dummy = torch.zeros(
            (
                self.hf_config.num_attention_heads // self.world_size,
                self.history_step,
                pred_len,
            ),
            dtype=self.cnn_dtype,
            device="cuda",
        )
        with torch.inference_mode():
            self.cnn(dummy)
        torch.cuda.synchronize()

    def _free_predictor_row(self, row_idx: int) -> None:
        for layer_state in self.attn_history:
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
        del observed_block_mask, step_position
        scores = block_scores.to(self.cnn_dtype)
        if scores.shape[-2] > self.history_step:
            scores = scores[:, -self.history_step :, :]

        history = self.attn_history[layer_idx].get(row_idx)
        if history is None:
            if scores.shape[-2] < self.history_step:
                scores = F.pad(scores, (0, 0, self.history_step - scores.shape[-2], 0))
            history = scores
        else:
            if scores.shape[-1] > history.shape[-1]:
                history = F.pad(history, (0, scores.shape[-1] - history.shape[-1]))
            history = torch.cat((history, scores), dim=-2)[:, -self.history_step :, :]
        self.attn_history[layer_idx][row_idx] = history

        start, end = prediction_block_window(
            history.shape[-1],
            sink_tokens=self.sink_token,
            recent_tokens=self.local_token,
            block_size=self.pooling_block_size,
        )
        inputs = history[:, :, start:end]
        if inputs.shape[-1] < 3:
            return torch.ones_like(inputs[:, 0, :], dtype=torch.float32), start
        with self._cnn_lock:
            prediction = self.cnn(inputs.contiguous())
        return prediction.to(torch.float32), start
