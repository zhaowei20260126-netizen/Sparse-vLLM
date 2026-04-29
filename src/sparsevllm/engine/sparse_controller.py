from dataclasses import dataclass
import torch
from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence
from sparsevllm.engine.cache_manager import CacheManager
from sparsevllm.utils.profiler import profiler
from sparsevllm.utils.context import get_context
from sparsevllm.utils.log import logger, log_level
from sparsevllm.triton_kernel.omnikv_fused import build_omnikv_keep_and_slots


@dataclass
class LayerBatchSparseState:
    """每一层的逻辑稀疏状态"""
    attn_score: torch.Tensor | None = None # 暂存当前层的 Attention Score（注意力分数）
    active_indices: torch.Tensor | None = None # 逻辑索引 [B, K]，经过 Sink + Top-K + Recent 挑选后，决定被保留下来的 Token 在原序列中的逻辑位置（比如第0, 1, 5, 88... 个token）
    active_slots: torch.Tensor | None = None   # 物理槽位 [B, K]，被挑中的 token 实际存放在 GPU 显存中的、全局平铺的物理槽位（Slot ID）
    req_indices: torch.Tensor | None = None # 记录当前 Batch 里的并发序列在 CacheManager 内存池缓冲区里的行号（Row Indices）。
    context_lens: torch.Tensor | None = None # 在当前这种稀疏策略下，本层实际应该看见的有效 KV 长度

    # for DeltaKV
    active_compressed_indices: torch.Tensor | None = None # 类似于普通的 active_indices，但是专门针对 DeltaKV 传给底层做重构计算的经过压缩区域筛选的 Token 逻辑下标。
    # Global row indices (into CacheManager slot maps). For some sparse views we may
    # also return local req indices to kernels.
    global_req_indices: torch.Tensor | None = None
    # DeltaKV uses scratch (temp) slots during reconstruction; only the last layer in a
    # segment should free them so other layers can reuse the same view/slots.
    deltakv_free_temp_slots: bool = False # 控制“临时重构显存池”的释放时机，等最后一层用完了，开关设为 True，通知底层：这部分临时占用的显存

class SparseController:
    """
    稀疏策略控制器，管理 KV Cache 的逻辑视图 (Reading View) 和 压缩策略。
    """
    def __init__(self, config: Config, cache_manager: CacheManager):
        self.sparse_method = config.vllm_sparse_method
        self.is_deltakv_family = isinstance(self.sparse_method, str) and self.sparse_method.startswith('deltakv')
        self.is_deltakv_standalone = self.sparse_method == 'deltakv-standalone'
        self.is_deltakv_snapkv = self.sparse_method == 'deltakv-snapkv'
        self.is_deltakv_standalone_like = self.sparse_method in ('deltakv-standalone', 'deltakv-snapkv')
        
        self.config = config
        self.cache_manager = cache_manager

        # 观察层 (Observation Layers)：在Ominikv中，系统不需要在每一层都重新计算“谁是最重要的 token”。而是选定某些层作为观察层，
        # 用它们的 Attention Score 来决定后面相连的几层该保住哪些 token。
        self.obs_layer_ids = self.config.obs_layer_ids
        
        # 全注意力层 (Full Attention Layers)：强制不进行任何压缩或稀疏化的层。
        self.full_attn_layers = self.config.full_attn_layers
        self.num_layers = self.config.hf_config.num_hidden_layers

        #  General Sparse Config
        # 1. Sink tokens: 序列最开头的几个 token ，被强制保留。
        self.num_sink = self.config.num_sink_tokens
        # 2. Recent tokens: 紧挨着当前正在生成位置的前面 N 个最近的 token，局部连续性最强，被强制保留。
        self.num_recent = self.config.num_recent_tokens
        # 3. Top-K tokens: 在排除了 Sink 和 Recent 的中间广阔历史中，挑出 Attention Score 最高的那部分 token 保留。
        self.num_top = self.config.num_top_tokens
        self.num_top_in_prefill = self.config.num_top_tokens_in_prefill # ominikv config
        
        # 稀疏层私有状态: dict[layer_idx, LayerSparseState]
        # 为每一层独立维护一个逻辑上的稀疏状态（包含了该层的注意力分数、当前生效的物理插槽索引、逻辑长度等）
        self.layer_batch_sparse_states: dict[int, LayerBatchSparseState] = {}
        for i in range(self.num_layers):
            self.layer_batch_sparse_states[i] = LayerBatchSparseState()

        # 静态配置
        self.sparse_config = {
            "vllm_sparse_method": self.sparse_method,
            "num_sink_tokens": self.config.num_sink_tokens,
            "num_recent_tokens": self.config.num_recent_tokens,
            "num_top_tokens": self.config.num_top_tokens,
            "obs_layer_ids": self.config.obs_layer_ids,
            "full_attn_layers": self.config.full_attn_layers,
        }

        self.layers = None

    @torch.no_grad()
    def prepare_forward(self, seqs: list[Sequence], is_prefill: bool):
        """前向计算前，重置并准备各层的稀疏视图。

        在 ModelRunner.run() 的第2步被调用，每次 prefill/decode step 前执行一次。

        职责:
          1. 将所有层的 LayerBatchSparseState 重置为"干净"的初始状态（默认 = 全量注意力）
          2. 为需要收集注意力分数的层（obs layers / SnapKV 所有层等）分配 attn_score 张量

        之后模型逐层前向时:
          - attention kernel 将 softmax 后的权重写入 attn_score
          - on_layer_end() 读取 attn_score 做 top-k 选择，动态更新下游 sparse layer 的视图

        设计要点:
          - active_* = None 表示"不做稀疏选择，使用全部 token"（默认安全策略）
          - 仅 _needs_attn_score() 返回 True 的层才分配 attn_score 张量（vanilla 零开销）
          - attn_score 用 float32 而非 bf16：注意力分数对精度敏感，低精度可能导致 top-k 选择不稳定
        """
        ctx = get_context()
        # 将稀疏方法名和参数（sink/recent/top-k数量等）注入全局上下文，
        # 这样 attention kernel 需要时可从 ctx 直接读取配置
        ctx.sparse_config = self.sparse_config if self.sparse_method else None

        for i in range(self.num_layers):
            state = self.layer_batch_sparse_states[i]
            # 从 CacheManager 获取该层当前 batch 的物理元数据:
            #   slot_mapping: (total_tokens,) → 每个token的物理slot编号
            #   context_lens: (batch_size,) → 每个序列的KV可见长度
            #   req_indices:  (batch_size,) → 每个序列在CacheManager元数据表中的行号
            batch_state = self.cache_manager.get_layer_batch_states(i)

            # --- 从 CacheManager 拷贝基础元数据到稀疏状态 ---
            # context_lens: 表示当前层每个序列应该看到多少个历史 KV token
            #   例: [4096, 4096] 表示 batch=2，各自看到 4096 个历史 token
            #   用 clone() 而非直接赋值——CacheManager 可能在后续更新原张量
            state.context_lens = batch_state.context_lens.clone()
            # req_indices: 每个序列→CacheManager元数据行号的映射，用于查 slot_map 等表
            state.req_indices = batch_state.req_indices
            # global_req_indices: DeltaKV 中需要全局行号（跨层一致），多数方法和 req_indices 相同
            state.global_req_indices = batch_state.req_indices

            # --- 重置稀疏选择状态为"默认安全"值 ---
            # 以下字段都是 None/False = "不做稀疏选择，使用全部 KV token"
            # 只有被 on_layer_end() 显式更新的 sparse layer 才会看到非 None 的稀疏视图
            state.attn_score = None                   # 注意力分数张量，None = 本轮不收集
            state.active_indices = None                # None = 不筛选，返回所有 token 的逻辑位置
            state.active_slots = None                  # None = 不筛选，返回所有 token 的物理 slot
            state.active_compressed_indices = None     # DeltaKV 专用: 需要重建的被压缩 token 逻辑索引
            state.deltakv_free_temp_slots = False      # 本轮是否需要释放 DeltaKV 临时重建 slot

            # --- 为需要收集注意力分数的层分配 attn_score 张量 ---
            if self._needs_attn_score(i, is_prefill, seqs):
                batch_size = len(seqs)
                # Q 的头数除以 world_size: TP 分片后每 rank 只持有 N/tp 个头
                num_heads = self.config.hf_config.num_attention_heads // self.config.tensor_parallel_size
                # 最长序列的 KV 长度，也是 attn_score 张量的最后一维（padding 到一致）
                max_len = int(state.context_lens.max())

                # attn_score 初始填充值:
                #   prefill: 0.0 — 还没计算，初始为0，token加入后被真实分数覆盖
                #   decode: -1e20 — 巨大负数≈softmax后的0，避开padding位置对 top-k 的干扰
                _val = 0.0 if is_prefill else -1e20
                with profiler.record("sparse_prepare_attn_score"):
                    # 分配 (B, num_heads, max_len) 的 float32 张量
                    #   为什么用 float32 而非 bf16/fp16?
                    #     注意力分数的 softmax 对微小差异敏感，低精度可能导致多个 token
                    #     分数"平齐"使得 top-k 选择不稳定，用 float32 保证选择质量
                    state.attn_score = torch.full(
                        (batch_size, num_heads, max_len),
                        _val,
                        dtype=torch.float32,
                        device="cuda",
                    )

    def set_modules(self, modules):
        self.layers = modules

    @torch.no_grad()
    def post_forward(self, seqs: list[Sequence], is_prefill: bool):
        """持久化压缩 (如 SnapKV / DeltaKV)"""
        if get_context().is_long_text is False and not self.is_deltakv_family:
            return

        if is_prefill:
            self.on_every_chunk_prefill_end(seqs)

        # Decode 阶段如果 Recent Buffer 溢出也需要压缩 (对于 DeltaKV)
        if not is_prefill and self.is_deltakv_family:
             self._deltakv_eviction(seqs)
        if not is_prefill and self.sparse_method in ('snapkv', 'pyramidkv'):
            self._snapkv_decode_eviction(seqs)
        if not is_prefill and self.sparse_method in ("streamingllm", "attention-sink", "attention_sink"):
            self._streamingllm_decode_eviction(seqs)

    @torch.no_grad()
    def on_every_chunk_prefill_end(self, seqs: list[Sequence]):
        if get_context().is_long_text is False and not self.is_deltakv_family:
            return

        # DeltaKV: Always try to compress incrementally (to save memory during long prefill)
        if self.is_deltakv_family:
            if not self.is_deltakv_standalone_like:
                assert self.config.chunk_prefill_accel_omnikv
            self._deltakv_eviction(seqs)
            if self.is_deltakv_snapkv and any(seq.is_last_chunk_prefill for seq in seqs):
                self._deltakv_snapkv_finalize(seqs)
            return

        # SnapKV / PyramidKV: Only evict at the end of prefill
        is_last_chunk = any(seq.is_last_chunk_prefill for seq in seqs)
        if not is_last_chunk:
            return

        if self.sparse_method == 'snapkv' or self.sparse_method == 'pyramidkv':
            self._snapkv_prefill_eviction(seqs)
        if self.sparse_method in ("streamingllm", "attention-sink", "attention_sink"):
            self._streamingllm_prefill_eviction(seqs)

    def get_read_view(self, layer_idx: int):
        """
        供 Attention.forward 调用：获取当前层应该读取的逻辑视图。
        返回 (active_slots, active_indices, req_indices, context_lens, attn_score, temp_slots)
        """
        sparse_state = self.layer_batch_sparse_states[layer_idx]
        if (self.sparse_method in ("omnikv", "deltakv") and layer_idx in self.full_attn_layers) or \
            self.sparse_method in ('snapkv', 'pyramidkv', 'quest', 'streamingllm', 'attention-sink', 'attention_sink', ''):

            return (
                self.cache_manager.get_layer_buffer_req_to_token_slots(layer_idx),  # 全部 token slots
                None,
                sparse_state.req_indices,
                sparse_state.context_lens,
                sparse_state.attn_score,
                None,
            )

        assert layer_idx not in self.full_attn_layers
        if self.sparse_method == 'deltakv':
            ctx = get_context()
            # active_compressed_indices: (B, Kmax), padded with -1; may be None (treated as K=0)
            active = sparse_state.active_compressed_indices
            # For DeltaKV we always use a batch-major Req->slots table, so kernels use local req indices.
            chunk_lens = None
            if ctx.is_prefill:
                if ctx.cu_seqlens_q is None or ctx.cu_seqlens_q.numel() <= 1:
                    chunk_lens = None
                else:
                    chunk_lens = (ctx.cu_seqlens_q[1:] - ctx.cu_seqlens_q[:-1]).to(torch.int32)

            active_slots, local_req_indices, new_context_lens, temp_slots = self.cache_manager.deltakv_reconstruct(
                layer_idx=layer_idx,
                active_compressed_indices=active,
                context_lens=sparse_state.context_lens,
                req_indices=sparse_state.global_req_indices,
                chunk_lens=chunk_lens,
            )
            return (
                active_slots,
                None,
                local_req_indices,
                new_context_lens,
                sparse_state.attn_score,
                temp_slots if sparse_state.deltakv_free_temp_slots else None,
            )

        if self.is_deltakv_standalone_like:
            active_slots, local_req_indices, new_context_lens, temp_slots = self.cache_manager.deltakv_reconstruct(
                layer_idx=layer_idx,
                active_compressed_indices=None,
                context_lens=sparse_state.context_lens,
                req_indices=sparse_state.global_req_indices,
                chunk_lens=None,
            )
            attn_score = sparse_state.attn_score if self.is_deltakv_snapkv else None
            return (
                active_slots,
                None,
                local_req_indices,
                new_context_lens,
                attn_score,
                temp_slots,
            )

        if self.sparse_method == 'omnikv':
            if sparse_state.active_slots is not None:
                active_slots = sparse_state.active_slots
                logger.debug('active_slots 是被 omnikv 选到的 slots')
            else:
                active_slots = self.cache_manager.get_layer_buffer_req_to_token_slots(layer_idx)
                logger.debug('active_slots is None')

            return (
                active_slots,
                sparse_state.active_indices,
                sparse_state.req_indices,
                sparse_state.context_lens,
                sparse_state.attn_score,
                None,
            )
        else:
            raise ValueError

    def on_layer_end(self, layer_idx: int, context):
        """每一层结束后的动态策略 (如 OmniKV / DeltaKV)"""
        if get_context().is_long_text is False and not self.is_deltakv_family:
            return

        if self.sparse_method not in ('omnikv', 'deltakv'):
            return

        if context.is_prefill and not self.config.chunk_prefill_accel_omnikv:
            return

        if layer_idx not in self.obs_layer_ids:
            return

        with profiler.record("sparse_on_layer_end"):
            state = self.layer_batch_sparse_states[layer_idx]
            if state.attn_score is None:
                raise ValueError("Attn Score hasn't been initialized")

            if state.attn_score.dim() == 3:
                if context.is_prefill:
                    chunk_lens = context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]
                    state.attn_score /= chunk_lens.view(-1, 1, 1)  # 不除其实也无所谓
                # 对head做max pooling
                state.attn_score = state.attn_score.max(dim=1).values

            target_layers = []
            for j in range(layer_idx + 1, self.num_layers):
                if j in self.full_attn_layers: break
                target_layers.append(j)
            assert len(target_layers) > 0

            self._update_dynamic_omnikv_indices(layer_idx, target_layers)

    @torch.no_grad()
    def _deltakv_eviction(self, seqs: list[Sequence]):
        assert get_context().is_long_text or self.is_deltakv_family
        self.cache_manager.deltakv_evict(seqs)

    @torch.no_grad()
    def reconstruct_deltakv_kv_fused(
        self,
        layer_idx: int,
        module: torch.nn.Module,
        physical_slots: torch.Tensor,
        active_indices: torch.Tensor | None,
        context_lens: torch.Tensor,
        req_indices: torch.Tensor
    ):
        assert get_context().is_long_text or self.is_deltakv_family
        return self.cache_manager.deltakv_reconstruct(
            layer_idx=layer_idx,
            active_compressed_indices=active_indices,
            context_lens=context_lens,
            req_indices=req_indices,
            chunk_lens=None,
        )

    @torch.no_grad()
    def _snapkv_prefill_eviction(self, seqs: list[Sequence]):
        for layer_idx in range(self.num_layers):
            state = self.layer_batch_sparse_states[layer_idx]
            attn_scores = state.attn_score
            if attn_scores is None:
                continue
            if attn_scores.dim() == 3:
                attn_scores = attn_scores.max(dim=1).values
            budget = self._get_layer_budget(layer_idx, is_prefill=True)
            if budget is None:
                continue
            for b_idx, seq in enumerate(seqs):
                if not seq.is_last_chunk_prefill:
                    continue
                kv_len = int(state.context_lens[b_idx])
                if kv_len <= budget:
                    continue
                if log_level == 'DEBUG':
                    logger.debug(
                        "[SnapKV] prefill eviction: "
                        f"layer={layer_idx} seq_id={seq.seq_id} kv_len={kv_len} budget={budget}"
                    )
                keep_indices = self._snapkv_select_indices(
                    attn_scores[b_idx, :kv_len], kv_len, budget
                )
                self.cache_manager.free_part_slots(layer_idx, seq, keep_indices)

    @torch.no_grad()
    def _snapkv_decode_eviction(self, seqs: list[Sequence]):
        for layer_idx in range(self.num_layers):
            state = self.layer_batch_sparse_states[layer_idx]
            attn_scores = state.attn_score
            if attn_scores is None:
                continue
            if attn_scores.dim() == 3:
                attn_scores = attn_scores.max(dim=1).values

            budget = self._get_layer_budget(layer_idx, is_prefill=False)
            if budget is None:
                continue

            top_budget = budget - self.num_sink - self.num_recent
            trigger_len = int(2.0 * top_budget)
            for b_idx, seq in enumerate(seqs):
                kv_len = int(state.context_lens[b_idx])
                if kv_len <= budget or kv_len < trigger_len:
                    continue
                if log_level == 'DEBUG':
                    logger.debug(
                        "[SnapKV] decode eviction: "
                        f"layer={layer_idx} seq_id={seq.seq_id} kv_len={kv_len} budget={budget} trigger_len={trigger_len}"
                    )
                keep_indices = self._snapkv_select_indices(
                    attn_scores[b_idx, :kv_len], kv_len, budget
                )
                self.cache_manager.free_part_slots(layer_idx, seq, keep_indices)

    @torch.no_grad()
    def _streamingllm_prefill_eviction(self, seqs: list[Sequence]):
        budget = self._get_streamingllm_budget()
        if budget is None:
            return

        for layer_idx in range(self.num_layers):
            state = self.layer_batch_sparse_states[layer_idx]
            for b_idx, seq in enumerate(seqs):
                if not seq.is_last_chunk_prefill:
                    continue
                kv_len = int(state.context_lens[b_idx])
                if kv_len <= budget:
                    continue
                keep_indices = self._streamingllm_select_indices(kv_len)
                self.cache_manager.free_part_slots(layer_idx, seq, keep_indices)

    @torch.no_grad()
    def _streamingllm_decode_eviction(self, seqs: list[Sequence]):
        budget = self._get_streamingllm_budget()
        if budget is None:
            return

        for layer_idx in range(self.num_layers):
            state = self.layer_batch_sparse_states[layer_idx]
            for b_idx, seq in enumerate(seqs):
                kv_len = int(state.context_lens[b_idx])
                if kv_len <= budget:
                    continue
                keep_indices = self._streamingllm_select_indices(kv_len)
                self.cache_manager.free_part_slots(layer_idx, seq, keep_indices)

    def _get_streamingllm_budget(self) -> int | None:
        budget = self.num_sink + self.num_recent
        if budget <= 0:
            return None
        return budget

    def _streamingllm_select_indices(self, kv_len: int) -> torch.Tensor:
        assert kv_len > 0
        device = "cuda"
        sink_end = min(self.num_sink, kv_len)
        recent_start = max(sink_end, kv_len - self.num_recent)
        sink_indices = torch.arange(sink_end, device=device, dtype=torch.long)
        recent_indices = torch.arange(recent_start, kv_len, device=device, dtype=torch.long)
        return torch.cat([sink_indices, recent_indices], dim=0)

    def _snapkv_select_indices(self, scores: torch.Tensor, kv_len: int, budget: int) -> torch.Tensor:
        assert kv_len > budget
        device = scores.device
        
        # 1. Sink indices
        sink_indices = torch.arange(self.num_sink, device=device)
        
        # 2. Recent indices
        recent_start = kv_len - self.num_recent
        recent_indices = torch.arange(recent_start, kv_len, device=device)
        
        # 3. Top-K indices
        num_topk = budget - self.num_sink - self.num_recent
        if num_topk > 0 and recent_start > self.num_sink:
            middle_scores = scores[self.num_sink:recent_start]
            topk_indices_relative = middle_scores.topk(min(num_topk, middle_scores.shape[0]), dim=-1).indices
            topk_indices = topk_indices_relative + self.num_sink
            keep_indices = torch.cat([sink_indices, topk_indices, recent_indices])
        else:
            keep_indices = torch.cat([sink_indices, recent_indices])
            
        return keep_indices

    def _update_dynamic_omnikv_indices(self, obs_layer_idx, target_layers):
        assert get_context().is_long_text or self.is_deltakv_family

        with profiler.record("sparse_update_dynamic_indices"):
            ctx = get_context()
            # full attn layer 的 req indices 是未处理的
            obs_sparse_state = self.layer_batch_sparse_states[obs_layer_idx]
            token_scores = obs_sparse_state.attn_score # (B, L)
            batch_size, max_len = token_scores.shape

            # 计算实际可检索的历史长度
            if ctx.is_prefill:
                chunk_lens = ctx.cu_seqlens_q[1:] - ctx.cu_seqlens_q[:-1]
                # num_recent 是在chunk之外额外再留 recent 个token
                hist_lens = obs_sparse_state.context_lens - chunk_lens - self.num_recent
            else:
                # num_recent 覆盖当前token
                hist_lens = obs_sparse_state.context_lens - self.num_recent
            if self.sparse_method == 'omnikv':
                hist_lens = hist_lens.clamp_min(self.num_sink)
            
            # 直接切除 Sink 之前的分数
            search_scores = token_scores[:, self.num_sink:]
            if self.sparse_method == 'omnikv':
                rel_hist_lens = hist_lens - self.num_sink
            elif self.sparse_method == 'deltakv':
                rel_hist_lens = self.cache_manager.get_compressed_lens(obs_sparse_state.req_indices)
            else:
                raise ValueError

            # 2. 掩码处理 (处理不等长 + 防止 topk 选到 buffer/chunk 区域)
            mask = torch.arange(search_scores.size(1), device="cuda") >= rel_hist_lens.unsqueeze(1)
            search_scores.masked_fill_(mask, -1e10)

            # 3. 提取 Top-K (per-seq, padded with -1)
            num_top = self.num_top_in_prefill if ctx.is_prefill else self.num_top
            topk_list = []
            k_list = []
            for b in range(batch_size):
                avail = int(rel_hist_lens[b].item())
                k_b = min(int(num_top), int(search_scores.size(1)), max(0, avail))
                k_list.append(k_b)
                if k_b <= 0:
                    topk_list.append(torch.empty((0,), device="cuda", dtype=torch.int32))
                else:
                    idx = search_scores[b].topk(k_b, dim=0).indices.to(torch.int32) + self.num_sink
                    topk_list.append(idx)
            k_max = max(k_list) if k_list else 0
            if k_max > 0:
                topk_indices = torch.full((batch_size, k_max), -1, device="cuda", dtype=torch.int32)
                for b in range(batch_size):
                    k_b = k_list[b]
                    if k_b > 0:
                        topk_indices[b, :k_b] = topk_list[b]
            else:
                topk_indices = torch.empty((batch_size, 0), device="cuda", dtype=torch.int32)

            # 4. 根据方法更新目标层状态
            if self.sparse_method == 'omnikv':
                local_req_indices = torch.arange(batch_size, dtype=torch.int32, device="cuda")
                topk_lens = torch.tensor(k_list, dtype=torch.int32, device="cuda")
                keep_indices, active_slots, new_context_lens = build_omnikv_keep_and_slots(
                    topk_indices,
                    topk_lens,
                    hist_lens,
                    obs_sparse_state.context_lens - hist_lens,  # lens of recent and chunk
                    self.cache_manager.get_layer_buffer_req_to_token_slots(obs_layer_idx + 1),
                    obs_sparse_state.req_indices,
                    self.num_sink,
                )

                for l_idx in target_layers:
                    target_sparse_state = self.layer_batch_sparse_states[l_idx]
                    target_sparse_state.active_indices = keep_indices
                    target_sparse_state.active_slots = active_slots
                    target_sparse_state.context_lens = new_context_lens
                    target_sparse_state.req_indices = local_req_indices
            
            elif self.sparse_method == 'deltakv':
                for l_idx in target_layers:
                    target_sparse_state = self.layer_batch_sparse_states[l_idx]
                    target_sparse_state.active_compressed_indices = topk_indices
                    # context_lens is finalized in cache_manager.deltakv_reconstruct(); keep a placeholder here.
                    target_sparse_state.context_lens = obs_sparse_state.context_lens
                    target_sparse_state.req_indices = obs_sparse_state.req_indices
                    target_sparse_state.global_req_indices = obs_sparse_state.req_indices
                    target_sparse_state.deltakv_free_temp_slots = (l_idx == target_layers[-1])
            else:
                raise ValueError

    @torch.no_grad()
    def _deltakv_snapkv_finalize(self, seqs: list[Sequence]):
        finalize_row_idx = [i for i, seq in enumerate(seqs) if seq.is_last_chunk_prefill]
        if not finalize_row_idx:
            return
        finalize_rows = torch.tensor(finalize_row_idx, device="cuda", dtype=torch.long)
        finalize_seqs = [seqs[i] for i in finalize_row_idx]

        layer_scores = []
        for layer_idx in range(self.num_layers):
            state = self.layer_batch_sparse_states[layer_idx]
            attn_scores = state.attn_score
            if attn_scores is None:
                continue
            if attn_scores.dim() == 3:
                attn_scores = attn_scores.max(dim=1).values
            layer_scores.append(attn_scores.index_select(0, finalize_rows))

        if not layer_scores:
            return

        combined_scores = torch.stack(layer_scores, dim=0).max(dim=0).values
        self.cache_manager.deltakv_snapkv_finalize_static_prune(finalize_seqs, combined_scores)
    
    def _needs_attn_score(self, layer_idx: int, is_prefill: bool, seqs: list[Sequence]) -> bool:
        """判断某一层在本轮前向中是否需要分配 attn_score 张量来收集注意力分数。

        attn_score 是稀疏方法做 token 选择的依据——attention kernel 将 softmax(QK^T/√d)
        的权重写入此张量，然后 on_layer_end() 读取它来选出 top-k 重要的 token。

        不同稀疏方法对"何时收集、在哪些层收集"有完全不同的策略：
        """

        # =====================================================================
        # 分支 1: DeltaKV-SnapKV（DeltaKV 压缩 + SnapKV 静态剪枝）
        # =====================================================================
        # 只在 prefill 最后一个 chunk 时收集，且必须是长文本。
        # 策略: 综合所有层的注意力分数，选出最重要的 token 做静态剪枝（永久删除），
        #       剩余 token 再由 DeltaKV 压缩为 latent。这里的 attn_score 只用于剪枝阶段。
        if self.sparse_method == 'deltakv-snapkv':
            if not is_prefill or get_context().is_long_text is False:
                return False
            # 仅最后一个 prefill chunk 收集——剪枝是一次性决策，不用每个 chunk 都做
            return any(seq.is_last_chunk_prefill for seq in seqs)

        # =====================================================================
        # 分支 2: OmniKV / DeltaKV —— 观察层模式
        # =====================================================================
        # 只在 obs_layer_ids 中的层收集分数（这些"观察层"使用全量注意力，能获取
        # 精确的 token 重要性信息）。非观察层不收集分数，它们从观察层"继承"选择结果。
        #
        # Prefill 时需要 chunk_prefill_accel_omnikv=True 才启用（否则 prefill 阶段
        # 不做动态选择，每个 chunk 内部独立做 full attention）。
        if self.sparse_method in ('omnikv', 'deltakv') and layer_idx in self.obs_layer_ids:
            if is_prefill and not self.config.chunk_prefill_accel_omnikv:
                return False
            return True

        # =====================================================================
        # 分支 3: SnapKV / PyramidKV —— 全层独立驱逐模式
        # =====================================================================
        # 与 OmniKV/DeltaKV 不同，SnapKV/PyramidKV 在**每一层**都独立收集分数，
        # 因为每层的注意力分布不同，都独立决定"该层要保留哪些 token"。
        if self.sparse_method in ('snapkv', 'pyramidkv'):
            if is_prefill:
                # === Prefill 阶段 ===
                # 短文本不触发稀疏
                if get_context().is_long_text is False:
                    return False
                # 只在最后一个 prefill chunk 时收集——驱逐是 prefill 完成后一次性执行
                # 的，中间 chunk 不触发驱逐（因为中间 chunk 的 attn_score 不完整）
                is_last_chunk = any(seq.is_last_chunk_prefill for seq in seqs)
                return is_last_chunk

            # === Decode 阶段 ===
            # Decode 时不需要每个 step 都收集——而是等序列长度超过一定阈值后才触发。
            # 这样避免频繁的驱逐开销：序列生成过程中，每生成 trigger_len 个 token 才
            # 驱逐一次。
            budget = self._get_layer_budget(layer_idx, is_prefill=False)
            if budget is None:
                # 该层不参与稀疏（例如 snapkv_num_full_layers 以内的层）
                return False
            state = self.layer_batch_sparse_states[layer_idx]
            if state.context_lens is None:
                return False
            # 可供选择的中间区域预算: budget - sink - recent（sink 和 recent 是强制保留的）
            top_budget = budget - self.num_sink - self.num_recent
            # 触发阈值 = 2×预算（一个启发式值: 当序列长度超过 2倍 预算时才值得驱逐）
            # 例: budget=4096(sink=64,recent=512,top=3520), trigger_len=7040
            #     当序列长度 > 7040 时开始驱逐，驱逐后保留 ≤4096 个 KV
            trigger_len = int(2.0 * top_budget)
            # 检查 batch 中是否有任何序列同时满足: 长度 ≥ trigger_len 且 超过 budget
            return bool(((state.context_lens >= trigger_len) & (state.context_lens > budget)).any())

        # 其他方法（vanilla, streamingllm, quest）不需要 attn_score
        return False
    
    def _get_layer_budget(self, layer_idx: int, is_prefill: bool) -> int | None:
        """获取指定层在驱逐后最多保留多少个 KV token（budget）。

        返回值: budget = sink + top_k + recent 三项之和。
          - None 表示该层不参与驱逐（保留全部 KV，即"预算无限"）

        三种情况:

        情况 1: 该层在 snapkv_num_full_layers 以内
          → 返回 None，表示前几层强制做 full attention，不驱逐任何 KV。
          → 设计原因: 浅层的注意力分布通常更均匀，聚合度低，
            硬选 top-k 会丢失全局上下文信息。

        情况 2: PyramidKV 模式（self.config.pyramid_layer_ratios 非空）
          → 不同层有不同预算，越深的层预算越小（金字塔形）。
          → 比例 = ratios[layer_idx] / ratios[0]，第 0 层是全量基础。
          → 例: ratios=[1.0, 0.8, 0.6, 0.4, ...]
              layer_0: top_k = 4096 * 1.0/1.0 = 4096, budget = 64+4096+512 = 4672
              layer_10: top_k = 4096 * 0.6/1.0 = 2458, budget = 64+2458+512 = 3034
          → 设计原因: 深层特征更抽象，对长距离依赖需求更低，可以用更小的缓存。
            这比 SnapKV 的"所有层相同预算"更省显存。

        情况 3: SnapKV 模式（默认）
          → 返回固定值: sink + top_k + recent（所有层相同）。
          → 例: 64 + 4096 + 512 = 4672
          → 设计原因: 简单、通用，每层用相同的选择标准独立做 KV 驱逐。
        """
        # 情况 1: 前 snapkv_num_full_layers 层不参与稀疏，保留全部 KV
        if layer_idx < self.config.snapkv_num_full_layers:
            return None

        # Prefill 和 Decode 可以用不同的 top-k 数量（prefill 通常更大，因为信息更密集）
        num_top = self.num_top_in_prefill if is_prefill else self.num_top

        if self.config.pyramid_layer_ratios is not None:
            # 情况 2: PyramidKV — 按层分配不同预算
            # ratio/baseratio = 该层相对第 0 层的缩放比例
            ratio = self.config.pyramid_layer_ratios[layer_idx]
            base_ratio = self.config.pyramid_layer_ratios[0]
            # top-k 按比例缩放（越深越少）
            scaled_top_tokens = int(num_top * ratio / base_ratio)
            # budget = sink(强制保留开头) + top_k(分数最高的) + recent(强制保留末尾)
            return self.num_sink + scaled_top_tokens + self.num_recent
        elif self.sparse_method == 'snapkv':
            # 情况 3: SnapKV — 所有层相同预算
            return self.num_sink + num_top + self.num_recent

        # 不是 snapkv/pyramidkv 的方法（如 streamingllm, quest，它们的 budget 不是这样算的）
        return None
