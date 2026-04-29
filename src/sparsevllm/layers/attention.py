import torch
from torch import nn
import triton
import triton.language as tl

from sparsevllm.triton_kernel.context_flashattention_nopad import context_attention_fwd
from sparsevllm.triton_kernel.flash_decoding_stage1 import flash_decode_stage1 as mha_flash_decode_stage1
from sparsevllm.triton_kernel.flash_decoding_stage1 import flash_decode_stage1_with_score as mha_flash_decode_stage1_with_score
from sparsevllm.triton_kernel.gqa_flash_decoding_stage1 import flash_decode_stage1 as gqa_flash_decode_stage1
from sparsevllm.triton_kernel.gqa_flash_decoding_stage1 import flash_decode_stage1_with_score as gqa_flash_decode_stage1_with_score
from sparsevllm.triton_kernel.flash_decoding_stage2 import flash_decode_stage2
from sparsevllm.utils.context import get_context
from sparsevllm.utils.log import logger

from sparsevllm.engine.sparse_controller import SparseController


# =============================================================================
# Triton Kernel: store_kvcache_kernel
# =============================================================================
# 作用: 将一批 token 的 K 和 V 张量写入 KV Cache 的物理存储中。
# 这是一个 GPU kernel，每个 GPU thread 独立处理一个 token。
#
# =============================================================================
# Triton 基础概念（供未学过 Triton 的读者）:
# =============================================================================
#   @triton.jit          — 装饰器，标记这是一个 Triton kernel 函数。
#                          被编译为 GPU 机器码，在 GPU 上并行执行。
#
#   tl.program_id(axis)  — 获取当前 thread 在 axis 维度上的编号。
#                          类似 CUDA 中的 blockIdx / threadIdx。
#                          program_id(0) = "我是第几个并行工作单元"。
#
#   tl.load(ptr)         — 从 GPU 内存地址 ptr 加载数据。
#                          ptr 可以是单个地址，也可以是一个地址数组（批量加载）。
#
#   tl.store(ptr, val)   — 将 val 写入 GPU 内存地址 ptr。
#
#   tl.arange(a, b)      — 创建一个从 a 到 b-1 的整数序列。
#                          tl.arange(0, 128) = [0, 1, 2, ..., 127]。
#                          用于构造批量访问的偏移数组。
#
#   tl.constexpr         — 标记一个参数是编译时常量（在 kernel 编译时就确定，
#                          不会在每次调用时改变）。
#
#   ptr + offsets        — GPU 内存地址 + 偏移数组 = 一组连续地址。
#                          例如 ptr + [0,1,2] 表示 ptr, ptr+1, ptr+2 三个地址。
#
# 调用方式: store_kvcache_kernel[(N,)](key, key.stride(0), ...)
#          [(N,)] = "启动 N 个并行 thread，每个 thread 的 program_id(0) 从 0 到 N-1"
#
# =============================================================================
# 内存布局说明:
# =============================================================================
#   key 形状:  (N, num_kv_heads, head_dim)    例如 (4096, 4, 128)
#   k_cache 形状: (num_slots, num_kv_heads, head_dim)  例如 (30000, 4, 128)
#
#   key 展平为一维内存后的布局:
#     token_0: [h0_d0, h0_d1, ..., h0_d127, h1_d0, h1_d1, ..., h1_d127, h2_d0, ..., h3_d127]
#              ├────── head_0 (128个) ──────┤├────── head_1 ──────┤
#     token_1: [h0_d0, ...]
#     ...
#     key_stride (= stride(0)) = 4 × 128 = 512，即每个 token 占 512 个元素
#
#   D = num_kv_heads × head_dim = 512，即每个 token 的 K（或 V）的全部元素数
#
# =============================================================================
# 具体例子:
# =============================================================================
#   假设 N=3 个 token，head_dim=128, num_kv_heads=4, D=512
#   slot_mapping = [100, 200, -1]
#     含义: token_0 写入 slot 100, token_1 写入 slot 200, token_2 跳过（无效token）
#
#   Thread 0 (idx=0): slot=100. 读 key[0] 全部 512 元素 → 写 k_cache[100]
#   Thread 1 (idx=1): slot=200. 读 key[1] 全部 512 元素 → 写 k_cache[200]
#   Thread 2 (idx=2): slot=-1.  直接 return，什么都不做
# =============================================================================

@triton.jit
def store_kvcache_kernel(
    key_ptr,                # key 张量在 GPU 内存中的起始地址（指向 token_0 的第 0 个元素）
    key_stride,             # 从 token_i 跳到 token_{i+1} 需要跨越的元素个数 (= stride(0))
    value_ptr,              # value 张量的起始地址
    value_stride,           # value 的跨 token 步长
    k_cache_ptr,            # KV Cache 中 K 部分的起始地址
    v_cache_ptr,            # KV Cache 中 V 部分的起始地址
    slot_mapping_ptr,       # slot_mapping 数组的起始地址（每个元素是一个 slot 编号）
    D: tl.constexpr,        # 编译时常量：每个 token 的 K 包含的元素总数 (= num_kv_heads × head_dim)
):
    # --- 第1步: 获取身份和任务 ---
    # program_id(0): "我是第几个 thread"。N 个 token → N 个 thread，idx ∈ [0, N-1]
    idx = tl.program_id(0)

    # --- 第2步: 读取该 token 对应写入哪个 slot ---
    # slot_mapping 是一个整数数组，slot_mapping[idx] = 该 token 应写入的 slot 编号
    # tl.load(slot_mapping_ptr + idx): 从 GPU 内存读 slot_mapping[idx]
    slot = tl.load(slot_mapping_ptr + idx)

    # --- 第3步: 检查是否是无效 token ---
    # slot == -1 表示该 token 不需要写入 KV Cache（例如 batch 中 pad 的 token）
    # 直接 return，这个 thread 的工作完成了
    if slot == -1: return

    # --- 第4步: 构造读取偏移数组并加载 Key ---
    # 要加载 key[idx] 的全部 D 个元素。key[idx] 在内存中的起始位置 = idx * key_stride
    # 需要读 D 个连续元素: [起始, 起始+1, 起始+2, ..., 起始+D-1]
    # tl.arange(0, D) = [0, 1, 2, ..., D-1]
    # key_offsets = [idx*key_stride+0, idx*key_stride+1, ..., idx*key_stride+D-1]
    #
    # 例: idx=1, key_stride=512, D=512
    #     key_offsets = [512, 513, ..., 1023]  ← token_1 的 512 个元素在内存中的地址偏移
    key_offsets = idx * key_stride + tl.arange(0, D)
    # tl.load(key_ptr + key_offsets): 一次性加载这 D 个元素（GPU 支持向量化内存访问）
    key = tl.load(key_ptr + key_offsets)     # key 现在是长度为 D 的向量

    # --- 第5步: 加载 Value (与 Key 完全相同的逻辑) ---
    value_offsets = idx * value_stride + tl.arange(0, D)
    value = tl.load(value_ptr + value_offsets)

    # --- 第6步: 构造写入偏移数组并写入 KV Cache ---
    # 将数据写入 KV Cache 的第 slot 号位置（不是按 token 顺序存储！）
    # 因为不同 batch 的 token 可能被分配到不连续的 slot 编号上
    #
    # k_cache 第一个维度的跨度 = D 个元素（每个 slot 占 D 个连续元素）
    # slot 号的起始偏移 = slot * D
    # cache_offsets = [slot*D+0, slot*D+1, ..., slot*D+D-1]
    #
    # 例: slot=100, D=512
    #     cache_offsets = [51200, 51201, ..., 51711]  ← slot 100 在 kv cache 中的地址偏移
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)    # K 写入 k_cache[slot] 位置
    tl.store(v_cache_ptr + cache_offsets, value)  # V 写入 v_cache[slot] 位置

    # 至此，一个 token 的 KV 写入完成。
    # N 个 thread 并行执行，所有 N 个 token 同时写入各自被分配的 slot 中。


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    '''
    stride 是 PyTorch 张量的步长——在某一维度上，移动到下一个元素需要在内存中跨越多少个元素。
    举个例子，一个形状为 (3, 4, 128) 的 contiguous 张量，在内存中是一个连续的 3×4×128=1536 个元素的数组：

        key.shape = (3, 4, 128)   # (N, num_heads, head_dim)

        key.stride(0) = 4 × 128 = 512   # 跳到下一个 token，跨 512 个元素
        key.stride(1) = 128              # 跳到下一个 head，跨 128 个元素
        key.stride(2) = 1                # 跳到下一个 dim 元素，跨 1 个元素（相邻）
        key.stride(-1) = 1               # -1 即最后一维，同上
    '''
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(-1) == 1
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        cache_manager = context.cache_manager  # 管理 KV 的物理存储
        sparse_controller: SparseController = context.sparse_controller # 决定读取哪些 token 的 KV
        store_k_cache, store_v_cache, slot_mapping = cache_manager.get_layer_store_view(context.now_layer_idx) # kv_cache[0, layer_idx], kv_cache[1, layer_idx], layer_batch_state.slot_mapping

        # 1. 写入 KV Cache (物理行为)
        # 无论是 DeltaKV 还是全量/SnapKV，均先将当前 KV 写入物理槽位 (对于 DeltaKV，是写入 Base Pool 作为 Recent)
        store_kvcache(k, v, store_k_cache, store_v_cache, slot_mapping)
        cache_manager.on_kv_stored(context.now_layer_idx, k, slot_mapping) # 默认空操作。只有 QuEST 覆写了这个方法，用于在 KV 写入后更新 page 的 max/min 元数据缓存。

        # 2. 获取逻辑视图
        layer_active_slots, layer_active_indices, layer_req_indices, layer_context_lens, layer_attn_score, deltakv_temp_slots = \
            sparse_controller.get_read_view(context.now_layer_idx) # 这是稀疏方法**差异化**的核心入口。不同方法返回不同的 `active_slots`

        assert layer_active_slots is not None
        b_req_idx = layer_req_indices

        try:
            k_cache, v_cache = cache_manager.get_layer_compute_tensors(context.now_layer_idx, sparse_controller)
        except NotImplementedError:
            k_cache, v_cache = cache_manager.get_layer_kv_cache(context.now_layer_idx)

        # --- 通用稀疏/全量路径 (使用 Triton) ---
        try:
            if context.is_prefill:
                if context.cu_seqlens_q is None or context.cu_seqlens_q.numel() <= 1:
                    return torch.empty_like(q)

                b_start_loc = context.cu_seqlens_q[:-1]           # 每个序列在 flatten 输入中的起始位置
                chunk_lens = context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]  # 每个序列的 chunk 长度
                b_seq_len = layer_context_lens                     # 每个序列实际可见的 KV 长度
                b_prompt_cache_len = b_seq_len - chunk_lens        # 每个序列的历史 KV 长度
                max_input_len = b_seq_len.max().item()

                # Triton 路径需要物理槽位 layer_active_slots 用于 Req_to_tokens 寻址
                # 它内部通过 prompt_cache_len 实现因果掩码，目前不需要显式的 pos_ids
                o = torch.empty_like(q)
                context_attention_fwd(
                    q, k_cache, v_cache, o,
                    b_req_idx, b_start_loc, b_seq_len, b_prompt_cache_len, max_input_len,
                    layer_active_slots,      # ★ 决定实际读哪些物理 slot
                    attn_score=layer_attn_score,  # ★ 收集注意力分数到 attn_score 张量
                )
            else:    # decode
                batch_size = q.shape[0]
                layer_active_slots, b_req_idx, layer_context_lens = cache_manager.build_decode_view(
                    context.now_layer_idx,
                    q,
                    layer_active_slots,
                    b_req_idx,
                    layer_context_lens,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                )

                max_len_in_batch = layer_context_lens.max().item()
                BLOCK_SEQ = 256

                mid_o = torch.empty(
                    (batch_size, self.num_heads, (max_len_in_batch + BLOCK_SEQ - 1) // BLOCK_SEQ, self.head_dim),
                    dtype=torch.float32,
                    device=q.device,
                )
                mid_o_logexpsum = torch.empty(
                    (batch_size, self.num_heads, (max_len_in_batch + BLOCK_SEQ - 1) // BLOCK_SEQ),
                    dtype=torch.float32,
                    device=q.device,
                )

                is_gqa = self.num_heads > self.num_kv_heads
                if layer_attn_score is not None:
                    if is_gqa:
                        gqa_flash_decode_stage1_with_score(
                            q, k_cache, v_cache, layer_active_slots, b_req_idx, layer_context_lens,
                            max_len_in_batch, mid_o, mid_o_logexpsum, layer_attn_score, BLOCK_SEQ,
                        )
                    else:
                        mha_flash_decode_stage1_with_score(
                            q, k_cache, v_cache, layer_active_slots, b_req_idx, layer_context_lens,
                            max_len_in_batch, mid_o, mid_o_logexpsum, layer_attn_score, BLOCK_SEQ,
                        )
                else:
                    if is_gqa:
                        gqa_flash_decode_stage1(
                            q, k_cache, v_cache, layer_active_slots, b_req_idx, layer_context_lens,
                            max_len_in_batch, mid_o, mid_o_logexpsum, BLOCK_SEQ,
                        )
                    else:
                        mha_flash_decode_stage1(
                            q, k_cache, v_cache, layer_active_slots, b_req_idx, layer_context_lens,
                            max_len_in_batch, mid_o, mid_o_logexpsum, BLOCK_SEQ,
                        )

                o = torch.empty_like(q)
                flash_decode_stage2(mid_o, mid_o_logexpsum, layer_context_lens, o, BLOCK_SEQ)

            return o
        finally:
            # DeltaKV reconstructs some KV into scratch slots; recycle them immediately after use.
            if deltakv_temp_slots is not None and deltakv_temp_slots.numel() > 0:
                cache_manager.free_temp_deltakv_full(deltakv_temp_slots)
