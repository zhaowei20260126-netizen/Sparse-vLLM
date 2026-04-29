from __future__ import annotations

import os
from collections import deque

import numpy as np
import torch

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence
from sparsevllm.utils.log import logger, log_level
from sparsevllm.utils.profiler import profiler

from .base import CacheManager, LayerBatchStates


class StandardCacheManager(CacheManager):
    """标准全历史 KV 缓存管理器：为每个序列维护完整的 K/V 存储，无压缩或窗口化。
    
    核心设计：
    - 插槽模型：每个token占用一个 GPU 显存插槽，使用栈式 LIFO 分配
    - 行/序列映射：每个序列映射到缓冲区中的一行，记录该序列已存储的令牌数
    - 请求到插槽映射：buffer_req_to_token_slots[行][位置] = 插槽ID，用于快速定位令牌在 KV 缓存中的位置
    """

    def __init__(self, config: Config, rank: int, world_size: int):
        """初始化标准 KV 缓存管理器。
        
        Args:
            config: 模型配置，包含 num_kvcache_slots（动态计算）
            rank: 分布式进程号
            world_size: 分布式世界大小
        
        初始化步骤：
        1. 调用父类初始化，设置模型参数和基础状态
        2. 分配 GPU KV 缓存张量 (shape: [2, num_layers, num_slots, num_kv_heads, head_dim])
        3. 初始化插槽栈：用于 LIFO 式的自由插槽管理
        4. 初始化序列映射表：seq_id -> 缓冲区行索引
        5. 初始化请求到插槽映射：记录每个序列令牌的插槽位置
        """
        super().__init__(config, rank, world_size)
        self.allocate_kv_cache() # 设置self.config.num_kvcache_slots和self.kv_cache

        num_slots = config.num_kvcache_slots
        # 插槽栈：栈顶指针是 _num_free_slots，自由插槽范围 [_num_free_slots, num_slots)
        # 分配时从栈顶取，释放时压回栈顶
        self.free_slots_stack = torch.arange(num_slots, dtype=torch.int32, device="cuda")
        self._num_free_slots = num_slots

        # buffer_req_to_token_slots[行][令牌位置] = 插槽ID
        # 用于快速查询每个序列的某个位置令牌存储在哪个 KV 缓存插槽中
        self.buffer_req_to_token_slots = torch.zeros(
            (self.max_buffer_rows, self.max_model_len), dtype=torch.int32, device="cuda"
        )

        # 序列 ID 到缓冲区行的映射（一个序列对应一行）
        self.seq_id_to_row: dict[int, int] = {}
        # 空闲行队列（FIFO）：记录还未分配的缓冲区行
        self.free_rows = deque(range(self.max_buffer_rows))
        # 每行已使用的令牌数，用于追踪序列长度
        self.row_seq_lens = np.zeros((self.max_buffer_rows,), dtype=np.int32)
        # 单层批次状态（所有层共享同一状态）
        self.layer_batch_state = LayerBatchStates()

    def allocate_kv_cache(self):
        """根据可用 GPU 显存动态分配 KV 缓存张量。
        
        策略：使用全部剩余显存来最大化缓存容量，计算能容纳多少令牌。
        公式：num_slots = available_memory / (num_layers * bytes_per_slot)
        """
        available_memory, slot_bytes_per_layer = self._get_available_slots_info() # 同时修改了config.max_num_batched_tokens
        num_layers = self.num_layers

        # 单个 token 需要在所有层都分配一份 KV Cache
        # slot_bytes = 1 个 token 跨所有层占用的总字节数
        slot_bytes = num_layers * slot_bytes_per_layer
        # 计算出整个模型最多能同时容纳的 token 总数
        '''
                    slot_0   slot_1   slot_2  ...  slot_N    ← "位置编号"(也就是每一层可以存多少 token)
                    ┌───────┬───────┬───────┬─────┬───────┐
            layer_0 │ K+V   │ K+V   │ 空    │ ... │ K+V   │
            layer_1 │ K+V   │ K+V   │ 空    │ ... │ K+V   │
            layer_2 │ K+V   │ K+V   │ 空    │ ... │ K+V   │
                ... │ ...   │ ...   │ ...   │ ... │ ...   │
            layer_L │ K+V   │ K+V   │ 空    │ ... │ K+V   │
                    └───────┴───────┴───────┴─────┴───────┘
                    ↑ 这整列就是一个 "slot": 跨所有层的一竖条
        '''
        self.config.num_kvcache_slots = available_memory // slot_bytes
        assert self.config.num_kvcache_slots > 0, "可用显存不足以分配 KV Cache"

        logger.info(
            f"Standard Mode: Each layer can accommodate {self.config.num_kvcache_slots} tokens."
        )
        # KV 缓存张量结构：
        # - 维度 0: [K, V]（2 个 cache）
        # - 维度 1: num_layers（Transformer 层数）
        # - 维度 2: num_slots（最大令牌数）
        # - 维度 3: num_kv_heads（KV 头数）
        # - 维度 4: head_dim（头维度）
        self.kv_cache = torch.empty(
            2,
            num_layers,
            self.config.num_kvcache_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.torch_dtype,
            device="cuda",
        )

    def get_layer_batch_states(self, layer_idx: int) -> LayerBatchStates:
        """获取某层的批次状态（所有层共享同一状态）。
        
        Returns: LayerBatchStates 包含 slot_mapping、context_lens、req_indices
        """
        return self.layer_batch_state

    def get_layer_kv_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """获取某层的 K 和 V 缓存张量。
        
        Returns: (K_cache, V_cache)，shape: (num_slots, num_kv_heads, head_dim)
        """
        return self.kv_cache[0, layer_idx], self.kv_cache[1, layer_idx]

    def get_layer_store_view(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取层级存储视图：K、V 缓存和当前批次的插槽映射。
        
        Returns: (K_cache, V_cache, slot_mapping)
        用于 attention 层的 KV 存储操作：存储新的 K/V 到指定插槽
        """
        return self.kv_cache[0, layer_idx], self.kv_cache[1, layer_idx], self.layer_batch_state.slot_mapping

    def get_layer_compute_tensors(self, layer_idx: int, sparse_controller):
        """标准 cache 管理器不需要额外的计算张量（无压缩/稀疏操作）。"""
        raise NotImplementedError

    def get_layer_buffer_req_to_token_slots(self, layer_idx: int) -> torch.Tensor:
        """获取请求到插槽的映射（所有层共享）。
        
        Returns: buffer_req_to_token_slots[seq_row][令牌位置] = 插槽ID
        """
        return self.buffer_req_to_token_slots

    @property
    def num_free_slots(self) -> int:
        """当前剩余自由插槽数。"""
        return self._num_free_slots

    def _get_free_row(self, seq_id: int) -> int:
        """获取或分配序列的缓冲区行。
        
        Args:
            seq_id: 序列 ID
        
        Returns:
            缓冲区行索引（该序列的行）
        
        逻辑：
        - 若序列已分配行，直接返回
        - 否则从空闲行队列取一个新行，并记录映射关系
        """
        if seq_id in self.seq_id_to_row:
            return self.seq_id_to_row[seq_id]
        if not self.free_rows:
            raise RuntimeError("No free rows in cache manager buffer!")
        row_idx = self.free_rows.popleft()
        self.seq_id_to_row[seq_id] = row_idx
        return row_idx

    @torch.no_grad()
    def _allocate(self, seq_id: int, size: int) -> torch.Tensor:
        """为单个序列分配 KV 缓存插槽（通常在 prefill 阶段）。
        
        Args:
            seq_id: 序列 ID
            size: 需要分配的插槽数（通常是 chunk_size）
        
        Returns:
            select_index：分配到的插槽 ID 张量，shape: (size,)
        
        分配流程：
        1. 检查是否有足够插槽
        2. 获取/创建序列对应的缓冲区行
        3. 从插槽栈顶取 size 个插槽
        4. 记录到请求-插槽映射表
        5. 更新该行已使用的令牌数
        """
        with profiler.record("cache_allocate"):
            assert self._num_free_slots >= size, (
                f"Out of KV cache slots: need {size}, free {self._num_free_slots}"
            )

            row_idx = self._get_free_row(seq_id)
            # 该行当前已存储的令牌数
            cur_len = self.row_seq_lens[row_idx]

            # 从插槽栈的栈顶取 size 个插槽
            ptr = self._num_free_slots
            select_index = self.free_slots_stack[ptr - size: ptr]
            self._num_free_slots -= size

            # 记录这些新令牌应该存储到哪些插槽
            # buffer_req_to_token_slots[行][cur_len:cur_len+size] = 插槽ID 序列
            self.buffer_req_to_token_slots[row_idx, cur_len: cur_len + size] = select_index
            self.row_seq_lens[row_idx] += size

            return select_index

    @torch.no_grad()
    def _allocate_batch(self, seq_ids: list[int], size: int) -> torch.Tensor:
        """为一批序列分配 KV 缓存插槽（通常在 decode 阶段，size=1）。
        
        Args:
            seq_ids: 序列 ID 列表（batch_size 个序列）
            size: 每个序列分配的插槽数（通常为 1）
        
        Returns:
            select_indices：分配到的插槽 ID 张量，shape: (batch_size,)
        
        注意：当前只支持 size=1（decode 每步一个令牌）
        """
        assert size == 1, "Batch allocation currently only supports size=1 (Decode)"
        batch_size = len(seq_ids)
        assert self._num_free_slots >= batch_size, (
            f"Out of KV cache slots: need {batch_size}, free {self._num_free_slots}"
        )

        # 为每个序列获取行索引
        row_indices = [self._get_free_row(sid) for sid in seq_ids]
        # 获取各行当前的令牌数
        cur_lens = self.row_seq_lens[row_indices]

        # 从插槽栈顶取 batch_size 个插槽（每个序列一个）
        ptr = self._num_free_slots
        select_indices = self.free_slots_stack[ptr - batch_size: ptr]
        self._num_free_slots -= batch_size

        # 使用张量索引更新映射关系
        # buffer_req_to_token_slots[row][令牌位置] = 插槽ID
        rows_gpu = torch.tensor(row_indices, dtype=torch.long, device="cuda")
        cols_gpu = torch.tensor(cur_lens, dtype=torch.long, device="cuda")
        self.buffer_req_to_token_slots[rows_gpu, cols_gpu] = select_indices
        # 更新每行的令牌计数（+1）
        self.row_seq_lens[row_indices] += 1

        return select_indices

    def free_seq(self, seq_id: int):
        """释放序列占用的所有 KV 缓存插槽和缓冲区行。
        
        Args:
            seq_id: 序列 ID
        
        释放流程：
        1. 找到该序列对应的行
        2. 收集该行占用的所有插槽
        3. 将插槽压回插槽栈
        4. 清理映射关系和计数
        5. 将行归还到空闲行队列
        """
        with profiler.record("cache_free_seq"):
            debug_slots = os.getenv("SPARSEVLLM_DEBUG_SLOTS", "0") == "1"
            # 获取该序列的行并从映射表移除
            row_idx = self.seq_id_to_row.pop(seq_id, None)
            if row_idx is None:
                raise ValueError

            # 该行已使用的令牌数 = 该行占用的插槽数
            cur_len = self.row_seq_lens[row_idx]
            # 获取该行占用的所有插槽 ID
            slots = self.buffer_req_to_token_slots[row_idx, :cur_len]

            assert cur_len > 0
            before_free = self._num_free_slots
            # 将这些插槽压回插槽栈
            ptr = self._num_free_slots
            self.free_slots_stack[ptr: ptr + cur_len] = slots
            self._num_free_slots += cur_len
            after_free = self._num_free_slots

            # 清理该行的数据
            self.buffer_req_to_token_slots[row_idx, :] = 0
            self.row_seq_lens[row_idx] = 0
            # 将行归还到空闲行队列
            self.free_rows.append(row_idx)

            if debug_slots:
                logger.info(
                    "free_seq seq_id={} row_idx={} freed_tokens={} free_slots_before={} free_slots_after={}",
                    seq_id,
                    row_idx,
                    int(cur_len),
                    int(before_free),
                    int(after_free),
                )
            if log_level == 'DEBUG': logger.debug(f'free seq {row_idx} with {cur_len} tokens')

    def debug_live_seq_slots(self) -> dict[int, int]:
        """调试方法：返回所有活跃序列的插槽占用情况。
        
        Returns: {seq_id: 该序列占用的令牌/插槽数}
        """
        return {
            int(seq_id): int(self.row_seq_lens[row_idx])
            for seq_id, row_idx in self.seq_id_to_row.items()
            if int(self.row_seq_lens[row_idx]) > 0
        }

    def free_part_slots(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        """标准 cache 不支持部分释放（无压缩/选择机制）。"""
        raise ValueError('不需要实现该方法')

    def _prepare_prefill(self, seqs: list[Sequence]):
        """准备 prefill 阶段的批次数据和 KV 缓存分配。
        
        Args:
            seqs: 需要 prefill 的序列列表
        
        Returns:
            (input_ids, positions, cu_seqlens_q)
            - input_ids: 所有 prefill 令牌的 ID（连接为 1D 张量）
            - positions: 对应的位置（0-based 序列内位置）
            - cu_seqlens_q: 累积序列长度，用于 flash-attn
        
        同时更新 layer_batch_state 中的插槽映射和上下文长度信息。
        """
        with profiler.record("cache_prepare_prefill"):
            # 计算本批次总令牌数
            total_chunk_tokens = sum(seq.current_chunk_size for seq in seqs)

            # 准备输入 ID 和位置的 numpy 数组（后续转 GPU 张量）
            input_ids_np = np.empty(total_chunk_tokens, dtype=np.int64)
            positions_np = np.empty(total_chunk_tokens, dtype=np.int64)
            # 累积序列长度：用于 flash-attn 的块状计算,区分每个序列的边界
            # cu_seqlens_q[i] = sum of seq_lens[0:i]
            cu_seqlens_q = [0]

            # 本批次中每个令牌应存储到的插槽 ID
            slot_mapping = torch.empty(total_chunk_tokens, dtype=torch.int32, device="cuda")
            # 各序列的 prefill 后长度（包括已有令牌和新令牌）
            context_lens_list = []
            # 各序列对应的缓冲区行索引
            req_indices = []

            # 遍历每个序列，分配 KV 缓存插槽并收集令牌信息
            token_offset = 0
            for seq in seqs:
                chunk_size = seq.current_chunk_size
                # 该序列之前已 prefill 的令牌数
                start_idx = seq.num_prefilled_tokens
                # prefill 后的令牌位置范围
                end_idx = start_idx + chunk_size

                # 安全检查：若序列已有行分配，验证行内令牌计数一致
                if seq.seq_id in self.seq_id_to_row:
                    row_idx = self.seq_id_to_row[seq.seq_id]
                    if self.row_seq_lens[row_idx] != start_idx:
                        raise ValueError(
                            "KV cache row length mismatch in prefill: "
                            f"seq_id={seq.seq_id} row_seq_len={self.row_seq_lens[row_idx]} "
                            f"start_idx={start_idx}"
                        )

                # 为新令牌分配 KV 插槽 ，self.buffer_req_to_token_slots[row_idx, cur_len: cur_len + size] = select_index,self.row_seq_lens[row_idx] += size
                self._allocate(seq.seq_id, chunk_size)
                row_idx = self.seq_id_to_row[seq.seq_id]
                # 提取这些新令牌的插槽映射
                slot_mapping[token_offset: token_offset + chunk_size] = self.buffer_req_to_token_slots[row_idx, start_idx:end_idx]
                # prefill 后序列的总长度
                context_lens_list.append(end_idx)
                # 记录序列所在的行
                req_indices.append(row_idx)

                # 提取该 chunk 的令牌 ID
                chunk_tokens = seq.token_ids
                if len(chunk_tokens) > chunk_size:
                    chunk_tokens = chunk_tokens[start_idx:end_idx]

                # 填充输入 ID 和位置
                input_ids_np[token_offset: token_offset + chunk_size] = chunk_tokens
                positions_np[token_offset: token_offset + chunk_size] = np.arange(start_idx, end_idx)

                # 更新累积序列长度
                cu_seqlens_q.append(cu_seqlens_q[-1] + chunk_size)
                token_offset += chunk_size

            # 转换为 GPU 张量
            context_lens = torch.tensor(context_lens_list, dtype=torch.int32, device="cuda")
            req_indices_tensor = torch.tensor(req_indices, dtype=torch.int32, device="cuda")

            # 更新批次状态供 attention 层使用
            self.layer_batch_state.slot_mapping = slot_mapping
            self.layer_batch_state.context_lens = context_lens  # 各序列的上下文长度
            self.layer_batch_state.req_indices = req_indices_tensor  # 各序列的缓冲区行

            if log_level == 'DEBUG':
                logger.debug(f'{context_lens_list=}   {req_indices=}  {slot_mapping[:10].tolist()=}  {slot_mapping[-10:].tolist()=}')

            # 转换输入 ID 和位置到 GPU
            input_ids = torch.from_numpy(input_ids_np).to("cuda")
            positions = torch.from_numpy(positions_np).to("cuda")
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cuda")
            return input_ids, positions, cu_seqlens_q

    def _prepare_decode(self, seqs: list[Sequence]):
        """准备 decode 阶段的批次数据和 KV 缓存分配。
        
        Args:
            seqs: 需要 decode 的序列列表
        
        Returns:
            (input_ids, positions, None)
            - input_ids: 待生成令牌的 ID（当前最后令牌，shape: (batch_size,)）
            - positions: 对应的位置
            - None: decode 无需 cu_seqlens
        
        在 decode 中，每个序列每步生成 1 个新令牌，需要 1 个新插槽。
        同时更新 layer_batch_state 中的插槽映射和上下文长度。
        """
        with profiler.record("cache_prepare_decode"):
            batch_size = len(seqs)
            # 收集批次中各序列的最后令牌（待输入 attention）
            input_ids_list = [seq.last_token for seq in seqs]
            # 各序列的当前长度 - 1（0-based 位置）
            positions_list = [seq.num_tokens - 1 for seq in seqs]
            seq_ids = [seq.seq_id for seq in seqs]

            # 为整个批次分配 batch_size 个新插槽（每个序列 1 个）
            new_slots_batch = self._allocate_batch(seq_ids, 1)
            # 获取各序列的缓冲区行
            row_indices = [self.seq_id_to_row[sid] for sid in seq_ids]
            # 各序列当前的上下文长度（prefill + 已 decode 的令牌数）
            context_lens = torch.tensor(
                self.row_seq_lens[row_indices],
                dtype=torch.int32,
                device="cuda",
            )
            # 各序列的缓冲区行索引
            req_indices = torch.tensor(row_indices, dtype=torch.int32, device="cuda")

            # 本轮 decode 分配的插槽映射
            slot_mapping = torch.empty((batch_size,), dtype=torch.int32, device="cuda")
            slot_mapping[:] = new_slots_batch

            # 更新批次状态
            self.layer_batch_state.slot_mapping = slot_mapping
            self.layer_batch_state.context_lens = context_lens
            self.layer_batch_state.req_indices = req_indices

            logger.debug(f'{slot_mapping=}   {context_lens.tolist()=}  {slot_mapping[:10]=}  {slot_mapping[-10:]=}')

            # 构造 decode 输入张量
            input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device="cuda")
            positions = torch.tensor(positions_list, dtype=torch.int64, device="cuda")
            return input_ids, positions, None
