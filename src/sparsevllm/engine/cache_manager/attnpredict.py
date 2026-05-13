from __future__ import annotations

import torch
import torch.nn.functional as F

from sparsevllm.config import Config
from sparsevllm.utils.profiler import profiler

from .standard import StandardCacheManager
from .attnpredict_cnn import AttnPredictCNN


class AttnPredictCacheManager(StandardCacheManager):
    """基于 AttentionPredictor 的 KV Cache 管理器。

    通过维护每层的滚动 attention 历史，用 CNN 预测下一步解码时需要保留的 KV token 子集。
    v1 版本中全量 KV 常驻 GPU，预测结果以逻辑视图（decode view）的方式生效，
    不做 CPU-GPU 异步预取。
    """

    def __init__(self, config: Config, rank: int, world_size: int):
        super().__init__(config, rank, world_size)

        # ---- AttentionPredictor 超参 ----
        self.topk = int(config.num_top_tokens)
        self.history_step = int(config.attnpredict_history_steps)
        self.pooling_block_size = int(config.attnpredict_pooling_block_size)
        self.sink_token = int(config.num_sink_tokens)
        self.local_token = int(config.num_recent_tokens)
        self.attn_scale = self.head_dim ** -0.5  # 1/sqrt(d_k)，与标准 attention 一致

        # 每层、每个 cache row 独立维护状态。
        # 用 cache row（而非 batch 下标）作为 key，因为连续 batching 中 batch 位置会变，
        # 而 cache row 是序列的稳定标识。
        #   attn_history[layer][row] → (num_heads, history_steps, pooled_len)
        #   tsp_mask[layer][row]     → (seq_len,)  bool tensor
        self.attn_history: list[dict[int, torch.Tensor]] = [
            {} for _ in range(self.num_layers)
        ]
        self.tsp_mask: list[dict[int, torch.Tensor]] = [
            {} for _ in range(self.num_layers)
        ]
        # 记录每层上一次 decode 的 view 信息，用于 predict_next_mask 中
        # 将稀疏 attention 恢复（scatter）回完整序列
        self._last_decode_view: list[dict[str, torch.Tensor | None] | None] = [
            None for _ in range(self.num_layers)
        ]

        # ---- 加载 CNN 预测器 ----
        self.cnn = AttnPredictCNN()
        model_path = str(config.attnpredict_model_path or "")
        state_dict = torch.load(model_path, map_location="cuda", weights_only=False)
        self.cnn.load_state_dict(state_dict)
        self.cnn.to(dtype=torch.float16, device="cuda")
        self.cnn.eval()
        self.cnn_dtype = next(self.cnn.parameters()).dtype

    def free_seq(self, seq_id: int):
        """释放序列时同步清理该 cache row 的 attention 历史与预测 mask。"""
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
        """Prefill 阶段：用最后 history_step 个 query 计算 attention，初始化 CNN 历史。

        这样首个 decode step 就能使用预测 mask，不需要等到积累够 history_step 步。
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
                # 取当前序列最后 history_step 个 query token
                take = min(self.history_step, q_len)
                q_tail = q[q_end - take:q_end].to(torch.float32)

                # 从 GPU KV cache 中按 slot 取出完整 K 序列
                slots = active_slots[row_idx, :full_len].to(torch.long)
                k_full = k_cache.index_select(0, slots).to(torch.float32)
                if group_size > 1:
                    k_full = k_full.repeat_interleave(group_size, dim=1)
                k_full = k_full[:, :num_heads, :]

                # 计算尾部 query 对完整 KV 的 attention logits
                logits = torch.einsum("thd,lhd->htl", q_tail, k_full)
                logits *= self.attn_scale

                # 构造因果 mask，确保每个 query 只能看到当前位置及之前的 KV
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
        """构建当前 decode step 的稀疏 KV 视图。

        使用上一轮预测出的 tsp_mask 来筛选 slot，同时强制保留当前 decode 新 token。
        返回 (packed_slots, local_req_indices, view_lens)，直接供给 attention kernel。
        """
        with profiler.record("attnpredict_build_decode_view"):
            batch_size = int(req_indices.numel())
            row_positions: list[torch.Tensor] = []  # 每行保留的 token 位置
            keep_counts: list[int] = []              # 每行保留的 token 数
            any_sparse = False                        # 是否有任何行使用了稀疏 mask

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
                    # 尚无预测 mask（可能是首步），使用完整序列
                    positions = torch.arange(full_len, dtype=torch.long, device=q.device)
                else:
                    any_sparse = True
                    valid_len = min(int(pred_mask.numel()), full_len)
                    if valid_len > 0:
                        # 将 bool mask 转为位置索引
                        positions = pred_mask[:valid_len].nonzero(as_tuple=False).squeeze(-1)
                        positions = positions.to(device=q.device, dtype=torch.long)
                    else:
                        positions = torch.empty(0, dtype=torch.long, device=q.device)

                    # 当前 decode 新 token 刚写入物理 cache，上一轮的预测不可能知道它。
                    # 原始 AttentionPredictor 始终将最新 KV 拼回稀疏 KV，这里等效强制保留。
                    current_pos = torch.tensor([full_len - 1], dtype=torch.long, device=q.device)
                    positions = torch.unique(torch.cat([positions, current_pos]), sorted=True)

                row_positions.append(positions)
                keep_counts.append(int(positions.numel()))

            # 没有任何行使用稀疏 mask → 返回完整 view
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

            # 将不同行、不同长度的位置/slot 打包到二维 tensor（pad -1）
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

            # 保存本层 decode view 元信息，供 predict_next_mask 恢复完整序列
            self._last_decode_view[layer_idx] = {
                "req_indices": req_indices.detach().clone(),
                "positions": packed_positions.detach(),
                "view_lens": view_lens.detach().clone(),
                "full_context_lens": context_lens.detach().clone(),
            }
            return packed_slots, local_req_indices, view_lens

    @torch.no_grad()
    def predict_next_mask(self, layer_idx: int, attn_logits: torch.Tensor) -> None:
        """Decode 阶段：将 kernel 写出的 raw logits 转为 softmax 权重，更新历史并预测下一步 mask。

        attn_logits 是 flash decode with_score kernel 在 att_value *= sm_scale 前写出的，
        因此这里的 softmax(logits * attn_scale) 是必要的转换，不是二次 softmax。
        """
        view = self._last_decode_view[layer_idx]
        if view is None:
            return

        with profiler.record("attnpredict_predict_next_mask"):
            # 确保 head 维度存在：(batch, seq) → (batch, 1, seq)
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

                # raw logits → softmax 权重
                logits = attn_logits[b, :, :view_len].to(torch.float32)
                attn = torch.softmax(logits * self.attn_scale, dim=-1)

                if positions is None:
                    # 上一步未使用稀疏 view，attention 已经是完整序列
                    full_attn = attn[:, :full_len]
                else:
                    # 稀疏 view 上的 attention 按逻辑位置 scatter 回完整 token 空间
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

    # ================================================================
    # AttentionPredictor 核心辅助方法
    # ================================================================

    def _update_row_prediction(
        self,
        layer_idx: int,
        row_idx: int,
        attn_weights_full: torch.Tensor,
    ) -> None:
        """更新指定 cache row 的 attention 历史，并预测新的 keep mask。

        完整流程：max-pooling → 滚动更新历史 → CNN 预测 block 重要性 → 生成 token mask
        """
        hist = self._update_attn_history(
            self.attn_history[layer_idx].get(row_idx),
            attn_weights_full,
        )
        self.attn_history[layer_idx][row_idx] = hist

        # CNN 预测 block 级重要性得分
        tsp_attn, start_block = self._time_sequence_predict(hist)
        seq_len = int(attn_weights_full.shape[-1])
        self.tsp_mask[layer_idx][row_idx] = self._create_tsp_mask(
            tsp_attn,
            seq_len=seq_len,
            start_block=start_block,
            device=attn_weights_full.device,
        )

    def _max_pooling(self, tensor: torch.Tensor) -> torch.Tensor:
        """对序列维度按 block_size 做 max-pooling，与原始 AttentionPredictor 代码一致。

        输入 shape (..., seq_len) → 输出 shape (..., seq_len // block_size)
        """
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
        """滚动更新 attention 历史窗口。

        每步将新 attention（max-pooling 后）追加到历史末尾，截断至 history_step 行。
        序列长度变化时自动 padding/截断以对齐列数。
        """
        # 先做 block 级 max-pooling，再截取尾部
        attn_pooling = self._max_pooling(attn_weights_full)
        if attn_pooling.shape[-2] > self.history_step:
            attn_pooling = attn_pooling[:, -self.history_step:, :]

        if attn_history is None:
            # 首次记录：不足 history_step 则在前面 pad 0
            if attn_pooling.shape[-2] < self.history_step:
                pad_rows = self.history_step - attn_pooling.shape[-2]
                attn_pooling = F.pad(attn_pooling, (0, 0, pad_rows, 0))
            return attn_pooling

        # 对齐列数（序列长度可能随 decode 增长）
        old_len = int(attn_history.shape[-1])
        new_len = int(attn_pooling.shape[-1])
        if new_len > old_len:
            attn_history = F.pad(attn_history, (0, new_len - old_len))
        elif new_len < old_len:
            attn_history = attn_history[..., :new_len]

        # 拼接后保留最近 history_step 行
        hist = torch.cat([attn_history, attn_pooling], dim=-2)
        return hist[:, -self.history_step:, :]

    def _time_sequence_predict(self, attn_history: torch.Tensor) -> tuple[torch.Tensor, int]:
        """CNN 前向预测：只对 sink 和 local 之外的「中间」block 做预测。

        返回：
            tsp_attn:     (num_heads, pred_len) block 重要性分数
            start_block:  预测起始 block 索引
        """
        num_heads, num_rows, attn_len = attn_history.shape
        # sink 和 local token 是强制保留的，不需要 CNN 预测
        start = self.sink_token // self.pooling_block_size
        end = attn_len - (self.local_token // self.pooling_block_size)
        end = max(start, end)
        attn_history = attn_history[:, :, start:end]
        pred_len = int(attn_history.shape[-1])

        # 预测区间太小则直接返回全 1（全部保留）
        if pred_len < 3:
            return torch.ones(
                (num_heads, pred_len),
                dtype=attn_history.dtype,
                device=attn_history.device,
            ), start

        # CNN 输入: (num_heads, num_rows, pred_len) → 输出: (num_heads, pred_len)
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
        """将 CNN 输出的 block 级别预测转换为 token 级别的 bool keep mask。

        mask 构造逻辑：
        1. 强制保留前 sink_token 个（sink tokens）
        2. 强制保留后 local_token 个（local tokens）
        3. 中间部分按 CNN 预测的 block 重要性取 top-k block
        4. v1 版本用 head 维 max-pooling 合并为整层共享 mask

        返回 shape (seq_len,) 的 bool tensor，True 表示保留。
        """
        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        if seq_len <= 0:
            return keep_mask

        # 强制保留 sink tokens
        sink_end = min(self.sink_token, seq_len)
        keep_mask[:sink_end] = True

        # 强制保留 local tokens
        local_start = max(sink_end, seq_len - self.local_token)
        keep_mask[local_start:] = True

        pred_len = int(tsp_attn.shape[-1])
        if pred_len < 1:
            return keep_mask

        # 计算可选的 block 预算
        block_budget = self.topk // self.pooling_block_size
        block_budget = max(0, min(block_budget, pred_len))
        if block_budget <= 0:
            return keep_mask

        # per-head 得分 → head 维 max → shared block 得分
        block_scores = tsp_attn.max(dim=0).values if tsp_attn.dim() == 2 else tsp_attn
        _, topk_indices = torch.topk(block_scores, block_budget, dim=-1)

        # block 索引展开为 token 索引
        token_indices = (
            (topk_indices + start_block).unsqueeze(-1) * self.pooling_block_size
            + torch.arange(self.pooling_block_size, device=device)
        ).reshape(-1)
        token_indices = token_indices[token_indices < seq_len]
        keep_mask[token_indices] = True
        return keep_mask
