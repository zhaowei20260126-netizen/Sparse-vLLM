import torch
import triton
import triton.language as tl
import math
import torch.nn.functional as F

TESLA = "Tesla" in torch.cuda.get_device_name(0)


# =============================================================================
# _fwd_kernel — Sparse-vLLM 的 Prefill Flash Attention Triton Kernel
# =============================================================================
# 基于 Flash Attention 论文算法 (Tiling + Online Softmax) 的 Triton 实现。
#
# ┌─────────────────────────────────────────────────────────────────┐
# │ 相比标准 Flash Attention Triton 实现，Sparse-vLLM 做了 4 处修改: │
# │                                                                 │
# │  ★修改1: NestedTensor / 变长序列布局                             │
# │   标准FA: Q形状 (B, S, H, D) 固定长度，batch内所有序列等长       │
# │   本实现: Q flatten 为 (total_tokens, H, D), 用 B_Start_Loc +    │
# │          B_Seqlen 描述各序列的起止位置                            │
# │                                                                 │
# │  ★修改2: 间接 KV 寻址 (稀疏注意力的核心)                          │
# │   标准FA: K[batch, head, seq_pos, :] 直接按位置索引               │
# │   本实现: 先查 Req_to_tokens[row][pos] → slot编号                │
# │          再从 K[slot, head, :] 读 KV                              │
# │   这使得 KV cache 可以散列存储(支持SnapKV驱逐/DeltaKV多池)        │
# │                                                                 │
# │  ★修改3: Chunked Prefill 的因果掩码                               │
# │   标准FA: 每个 token 只能看到自己之前的 token                     │
# │   本实现: chunk 内的 token 可以看到历史 chunk 的所有 KV，         │
# │          但不能看到当前 chunk 中位置比自己靠后的 token            │
# │          通过 prompt_cache_len 区分"历史"和"当前chunk"            │
# │                                                                 │
# │  ★修改4: GQA (Grouped Query Attention) 支持                       │
# │   通过 kv_group_num 将多个 Q head 映射到同一个 KV head            │
# └─────────────────────────────────────────────────────────────────┘
# =============================================================================

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, Out, B_Start_Loc, B_Seqlen, Req_to_tokens, B_req_idx,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    kv_group_num, b_prompt_cache_len,
    H: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # =========================================================================
    # 阶段 A: 获取当前 program 的身份 — "我是谁，我负责什么"
    # =========================================================================

    # program_id(0): Q 的 tile 索引 → "我处理 Q 的第几块"
    start_m = tl.program_id(0)

    # program_id(1): batch × head 的组合索引 → "我负责哪个序列的哪个头"
    cur_bh = tl.program_id(1)

    # 解码 batch 和 head 编号
    cur_batch = cur_bh // H      # "第几个序列"     (0, 0, ..., 0, 1, 1, ...)
    cur_head = cur_bh % H        # "第几个 Q head"   (0, 1, ..., 27, 0, 1, ...)

    # ★修改4: GQA — Q head → KV head 的映射
    # kv_group_num=7 时: Q_head 0-6→KV_head 0, Q_head 7-13→KV_head 1
    cur_kv_head = cur_head // kv_group_num

    # =========================================================================
    # 阶段 B: ★修改1 — NestedTensor 布局: 加载该序列的元数据
    # =========================================================================
    # 区别于标准 FA 的 Q[cur_batch, :, :] 直接索引:
    #   这里 Q 是变长序列 flatten 的，需要 B_Start_Loc 定位起始位置

    # 该序列在 flatten Q 中的起始索引
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    # ★修改3: 历史 KV 长度 (前几个 chunk 已 prefill 并存储的 token 数)
    # 用于区分"对所有 Q token 可见的历史 KV"和"需要因果掩码的当前 chunk KV"
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)

    # 当前 chunk 的 token 数 = 总可见长度 - 历史长度
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len

    # ★修改2: 该序列在 req_to_token 映射表中的行号
    # 之后通过 Req_to_tokens[row][pos] 查表获取 slot 编号
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    # =========================================================================
    # 阶段 C: 加载 Q tile 到 SRAM
    # =========================================================================

    block_start_loc = BLOCK_M * start_m       # 本 tile 在 chunk 内的起始位置

    offs_n = tl.arange(0, BLOCK_N)             # KV 侧分块内的索引 [0..127]
    offs_d = tl.arange(0, BLOCK_DMODEL)        # head_dim 维度的索引 [0..127]

    # Q token 在 chunk 内的局部索引 [block_start_loc .. block_start_loc+127]
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)

    # ★修改1: Q 的全局内存偏移计算
    # cur_batch_in_all_start_index + offs_m: 先定位到序列起始 + chunk内偏移 → Q 的 token 索引
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs  # token维度
        + cur_head * stride_qh                                          # head维度
        + offs_d[None, :] * stride_qd                                   # dim维度
    )
    # 加载 Q tile: (BLOCK_M, BLOCK_DMODEL), mask 滤掉超出 chunk 长度的位置
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    # =========================================================================
    # 阶段 D: 初始化 Online Softmax 状态
    # =========================================================================
    # Flash Attention 核心思想: 不一次性算完所有 softmax，而是逐 KV block 更新

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # running max (每个 Q token)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # running sum (softmax分母)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)   # 累积的 attention output

    # ★修改1: block_mask — 如果当前 Q tile 的起始位置已经超出 chunk 长度，
    # 则整个 tile 无效，block_end_loc=0，跳过主循环
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    # 总共需要看的 KV token 数: 历史缓存 128 + 当前 chunk 内不超过当前位置的
    block_end_loc = tl.minimum(
        block_start_loc + BLOCK_M + prompt_cache_len,   # Q token 可见的最大 KV 位置
        cur_batch_seq_len + prompt_cache_len             # 所有可见 KV 的上限
    )

    # =========================================================================
    # 阶段 E: 主循环 — 逐 KV block 计算 Attention (Flash Attention 核心算法)
    # =========================================================================
    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)  # 编译器优化提示

        # =====================================================================
        # ★修改2 (核心): 间接 KV 寻址
        # =====================================================================
        # 标准 FA 这里直接: k = K[cur_batch, start_n:start_n+BLOCK_N, cur_kv_head, :]
        #
        # Sparse-vLLM:
        #   步骤1: 先查 Req_to_tokens 表，把这 128 个 KV 位置的 slot 编号读出来
        #   Req_to_tokens[row][pos] = slot_ID 或 -1(该位置没有KV/已被驱逐)
        kv_loc = tl.load(
            Req_to_tokens                                                      # 映射表基地址
            + stride_req_to_tokens_b * cur_batch_req_idx                        # 定位到该序列的行
            + stride_req_to_tokens_s * (start_n + offs_n),                     # 定位到 pos start_n..start_n+127
            mask=(start_n + offs_n) < block_end_loc, other=0,                  # 超出范围填0
        )
        # kv_loc: (BLOCK_N,) — 128 个 slot 编号，可能是离散的(稀疏)或连续的(密集)

        # =====================================================================
        # 步骤2: 加载 K — 用 slot 编号构造 2D 偏移
        # =====================================================================
        # off_k 形状: (BLOCK_DMODEL, BLOCK_N) = (128, 128)
        #   off_k[:, j] = slot_j * stride_kbs + cur_kv_head * stride_kh + [0..127]
        #   即从 slot_j 位置读取 head_dim=128 维的 K 向量
        off_k = kv_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=(start_n + offs_n[None, :]) < block_end_loc, other=0.0)
        # k: (BLOCK_DMODEL, BLOCK_N) — 转置布局，适合 tl.dot

        # =====================================================================
        # 步骤3: 计算 QK^T
        # =====================================================================
        # q: (BLOCK_M, BLOCK_DMODEL) × k: (BLOCK_DMODEL, BLOCK_N) = qk: (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, k)

        # =====================================================================
        # ★修改3: Chunked Prefill 因果掩码
        # =====================================================================
        # 问题: 当前 chunk 的 token 只能看到"历史 KV"+ "当前 chunk 内位置≤自己的 KV"
        #       不能看到当前 chunk 中位置比自己靠后的 token
        #
        # mask[i][j] = (Q token的绝对位置 >= KV token的绝对位置)
        #   Q token 的绝对位置 = chunk内偏移i + 历史长度prompt_cache_len
        #   KV token 的绝对位置 = start_n + j
        mask = (offs_m[:, None] + prompt_cache_len) >= (start_n + offs_n[None, :])

        # 可见 → qk × sm_scale; 不可见 → -1e8 (exp2(-1e8)≈0)
        qk = tl.where(mask, qk * sm_scale, -1.0e8)

        # =====================================================================
        # 步骤4: Online Softmax — Flash Attention 的精髓
        # =====================================================================
        # 不一次性算完 softmax，而是逐块更新 running max/sum/output
        m_ij = tl.maximum(m_i, tl.max(qk, 1))   # 更新 running max
        qk -= m_ij[:, None]                       # 减去 max (防 exp 溢出)
        p = tl.math.exp2(qk)                     # exp2 (配合 sm_scale 中的 1.4427)
        l_ij = tl.sum(p, 1)                      # 当前块的 softmax 分母

        # 修正已累积的结果: 因为 max 可能变大，之前 exp 用的 max 不够大
        alpha = tl.math.exp2(m_i - m_ij)         # 修正系数 (exp2(旧max-新max))
        l_i = l_i * alpha + l_ij                 # 更新分母
        acc = acc * alpha[:, None]               # 修正已累积的输出

        # =====================================================================
        # 步骤5: 加载 V 并累积 attention output
        # =====================================================================
        # 同样通过 slot 编号间接寻址 V
        off_v = kv_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=(start_n + offs_n[:, None]) < block_end_loc, other=0.0)
        # v: (BLOCK_DMODEL, BLOCK_N) — 与 K 相同布局

        # acc += p @ v^T, 即 attention 权重 × value 向量
        acc = tl.dot(p.to(v.dtype), v, acc)

        m_i = m_ij  # 更新 running max

    # =========================================================================
    # 阶段 F: 最终归一化并写回
    # =========================================================================
    # acc / l_i = 最终 attention output (除以 softmax 分母完成归一化)
    acc = acc / l_i[:, None]

    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh + offs_d[None, :] * stride_od
    )
    tl.store(Out + off_o, acc, mask=offs_m[:, None] < cur_batch_seq_len)


@triton.jit
def _fwd_kernel_with_score(
    Q, K, V, sm_scale, Out, B_Start_Loc, B_Seqlen, Req_to_tokens, B_req_idx, Attn_Score,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    stride_asb, stride_ash, stride_asl,
    kv_group_num, b_prompt_cache_len,
    H: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H
    cur_kv_head = cur_head // kv_group_num

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)

    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * (start_n + offs_n),
            mask=(start_n + offs_n) < block_end_loc, other=0,
        )
        off_k = kv_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=(start_n + offs_n[None, :]) < block_end_loc, other=0.0)
        qk = tl.dot(q, k)
        
        # 收集评分：使用原始点积 (Raw Logits)，且掩码位置设为 0 以便后续计算 Mean
        mask = (offs_m[:, None] + prompt_cache_len) >= (start_n + offs_n[None, :])
        score_to_collect = tl.where(mask, qk, 0.0)
        block_sum = tl.sum(score_to_collect, 0)
        tl.atomic_add(Attn_Score + cur_batch * stride_asb + cur_head * stride_ash + (start_n + offs_n) * stride_asl, 
                      block_sum, mask=(start_n + offs_n) < block_end_loc)

        qk = tl.where(mask, qk * sm_scale, -1.0e8)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        off_v = kv_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=(start_n + offs_n[:, None]) < block_end_loc, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh + offs_d[None, :] * stride_od
    )
    tl.store(Out + off_o, acc, mask=offs_m[:, None] < cur_batch_seq_len)


@triton.jit
def _fwd_kernel_with_tail_score(
    Q, K, V, sm_scale, Out, B_Start_Loc, B_Seqlen, Req_to_tokens, B_req_idx, Attn_Score,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    stride_asb, stride_ash, stride_ast, stride_asl,
    kv_group_num, b_prompt_cache_len,
    HISTORY_STEP: tl.constexpr, TAIL_BLOCK_SIZE: tl.constexpr, TAIL_BLOCKS_PER_N: tl.constexpr,
    H: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H
    cur_kv_head = cur_head // kv_group_num

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)
    tail_start = tl.maximum(cur_batch_seq_len - HISTORY_STEP, 0)
    tail_idx = offs_m - tail_start
    tail_q_mask = (offs_m >= tail_start) & (offs_m < cur_batch_seq_len) & (tail_idx >= 0) & (tail_idx < HISTORY_STEP)

    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_pos = start_n + offs_n
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * kv_pos,
            mask=kv_pos < block_end_loc, other=0,
        )
        off_k = kv_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=kv_pos[None, :] < block_end_loc, other=0.0)
        qk = tl.dot(q, k)

        mask = (offs_m[:, None] + prompt_cache_len) >= kv_pos[None, :]

        qk = tl.where(mask, qk * sm_scale, -1.0e8)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        off_v = kv_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=kv_pos[:, None] < block_end_loc, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    # AttentionPredictor 只需要 block 级历史。这里第二遍扫描 K，
    # 用第一遍得到的全局 softmax 归一化量，把每个 KV block 内的
    # token probability 做 max 聚合，等价于 Python 端
    # softmax(token logits) -> max_pooling(block)。
    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_pos = start_n + offs_n
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * kv_pos,
            mask=kv_pos < block_end_loc, other=0,
        )
        off_k = kv_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=kv_pos[None, :] < block_end_loc, other=0.0)
        qk = tl.dot(q, k)

        mask = (offs_m[:, None] + prompt_cache_len) >= kv_pos[None, :]
        valid = mask & (kv_pos[None, :] < block_end_loc)
        qk = tl.where(valid, qk * sm_scale, -1.0e20)
        prob = tl.math.exp2(qk - m_i[:, None]) / l_i[:, None]
        prob = tl.where(valid, prob, 0.0)

        for block_group in tl.static_range(0, TAIL_BLOCKS_PER_N):
            block_start = start_n + block_group * TAIL_BLOCK_SIZE
            block_idx = block_start // TAIL_BLOCK_SIZE
            in_block = (kv_pos >= block_start) & (kv_pos < block_start + TAIL_BLOCK_SIZE)
            block_score = tl.max(tl.where(in_block[None, :], prob, 0.0), 1)
            score_offsets = (
                cur_batch * stride_asb
                + cur_head * stride_ash
                + tail_idx * stride_ast
                + block_idx * stride_asl
            )
            tl.store(
                Attn_Score + score_offsets,
                block_score,
                mask=tail_q_mask & (block_start < block_end_loc),
            )

    acc = acc / l_i[:, None]
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh + offs_d[None, :] * stride_od
    )
    tl.store(Out + off_o, acc, mask=offs_m[:, None] < cur_batch_seq_len)


@triton.jit
def _fwd_kernel_with_score_2d(
    Q, K, V, sm_scale, Out, B_Start_Loc, B_Seqlen, Req_to_tokens, B_req_idx, Attn_Score,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    stride_asb, stride_ash, stride_asl,
    kv_group_num, b_prompt_cache_len,
    H: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H
    cur_kv_head = cur_head // kv_group_num

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)

    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * (start_n + offs_n),
            mask=(start_n + offs_n) < block_end_loc, other=0,
        )
        off_k = kv_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=(start_n + offs_n[None, :]) < block_end_loc, other=0.0)
        qk = tl.dot(q, k)
        mask = (offs_m[:, None] + prompt_cache_len) >= (start_n + offs_n[None, :])
        
        # Mean across Q-tokens in this chunk (using raw logits)
        score_to_collect = tl.where(mask, qk, 0.0)
        block_mean = tl.sum(score_to_collect, 0) / (cur_batch_seq_len * 1.0) 
        
        # Max across Heads into 2D Attn_Score [batch, max_kv_len]
        tl.atomic_max(Attn_Score + cur_batch * stride_asb + (start_n + offs_n) * stride_asl, 
                      block_mean, mask=(start_n + offs_n) < block_end_loc)

        qk = tl.where(mask, qk * sm_scale, -1.0e8)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        off_v = kv_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=(start_n + offs_n[:, None]) < block_end_loc, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh + offs_d[None, :] * stride_od
    )
    tl.store(Out + off_o, acc, mask=offs_m[:, None] < cur_batch_seq_len)


# =============================================================================
# context_attention_fwd — Sparse-vLLM Prefill 阶段的 Flash Attention 入口
# =============================================================================
# 这是用 Triton 实现的"Flash Attention 算法"的 Sparse-vLLM 变体。
# 核心改动：通过 req_to_token_indexs 映射表间接寻址 KV Cache，支持稀疏注意力。
#
# 参数说明:
#   q: (total_tokens, num_heads, head_dim)   — 变长序列 flatten 后的 Q
#   k, v: (num_slots, num_kv_heads, head_dim) — 按 slot 散列存储的 KV Cache
#   o: (total_tokens, num_heads, head_dim)    — 输出张量（与 q 同形状）
#   b_req_idx: (batch,)     — 每个序列在 req_to_token_indexs 表中的行号
#   b_start_loc: (batch,)   — 每个序列在 flatten Q 中的起始位置 (= cu_seqlens[:-1])
#   b_seq_len: (batch,)     — 每个序列总共可见的 KV 长度（历史 + 当前 chunk）
#   b_prompt_cache_len: (batch,) — 每个序列的历史 KV 长度 (= b_seq_len - chunk_len)
#   max_input_len: int      — 最长序列的 KV 长度（用于 grid 大小）
#   req_to_token_indexs: (max_rows, max_model_len) — ★ 间接寻址表 [row][pos] → slot编号
#   attn_score: 可选, 形状见下方 — 收集注意力分数用于稀疏 token 选择
#
# Grid 结构 (二维):
#   program_id(0): Q 的块索引     → 共 ceil(max_input_len / BLOCK_M) 个
#   program_id(1): batch × head   → 共 batch * num_heads 个
#   每个 program 处理: Q 的一个 chunk × 某个序列的某个 head
# =============================================================================

@torch.no_grad()
def context_attention_fwd(
    q, k, v, o, b_req_idx, b_start_loc, b_seq_len, b_prompt_cache_len, max_input_len, req_to_token_indexs,
    attn_score=None,
    attn_score_block_size=None,
):
    # --- 第1步: 确定分块大小 ---
    # Tesla 老卡 (T4/V100) SRAM 较小，用更小的 BLOCK_M 避免溢出
    BLOCK_M = 128 if not TESLA else 64

    # head_dim: 每个注意力头的维度 (例如 Qwen2.5-0.5B 为 64, Qwen2.5-7B 为 128)
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]

    # --- 第2步: 安全断言 —— 确保张量布局与 kernel 内部假设一致 ---
    # Q/K/V 维度必须相等（标准 MHA 设计）
    assert Lq == Lk and Lk == Lv
    # head_dim 必须是 2 的幂且在 16-256 之间（Triton 的 BLOCK_DMODEL 要求）
    assert Lk in {16, 32, 64, 128, 256}
    # Q/K/V 的数据类型必须一致（混合精度会导致矩阵乘出错）
    assert q.dtype == k.dtype and k.dtype == v.dtype
    # 最后一维必须连续：kernel 内用 tl.arange(0,D) 一次加载 D 个连续元素，不连续会读到错误数据
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1 and o.stride(-1) == 1

    # --- 第3步: 计算 softmax 缩放因子 ---
    # 为什么乘 1.4427?
    #   kernel 内用 tl.math.exp2() (2^x) 而不是 exp() (e^x)，因为 GPU 上 exp2 更快
    #   e^x = 2^(x * log2(e)) = 2^(x * 1.4427)
    #   所以 sm_scale = 1/√d × 1.4427 = 预先把 1/√d 转为 exp2 需要的缩放
    sm_scale = 1.0 / (Lq ** 0.5) * 1.4426950408889634

    # batch: 当前 batch 中的序列数
    # head: 每个序列的 Q 头数 (已除以 TP size)
    batch, head = b_seq_len.shape[0], q.shape[1]

    # kv_group_num: GQA (Grouped Query Attention) 组数
    #   例: Q=28 heads, KV=4 heads → kv_group_num=7 (每7个Q头共享1个KV头)
    #   MHA 时 kv_group_num=1
    kv_group_num = q.shape[1] // k.shape[1]

    # --- 第4步: 定义 Grid —— 决定启动多少个并行 program ---
    # grid[0] = ceil(max_input_len / BLOCK_M): Q 维度切成多少块
    #   max_input_len = 最长序列的 b_seq_len，短序列靠 mask 跳过
    # grid[1] = batch * head: 每个序列的每个 head 独立并行
    #   例: batch=2, head=28 → 56 个 program 同时运行
    grid = lambda meta: (triton.cdiv(max_input_len, meta["BLOCK_M"]), batch * head, 1)

    # BLOCK_N: KV 侧的分块大小，与 BLOCK_M 相同（正方形 tile，SRAM 利用率高）
    BLOCK_N = BLOCK_M

    # num_warps: 每个 program 的 GPU 线程数
    #   warp = 32 线程一组。head_dim≤64 → 4 warps (128线程) 够用
    #   head_dim>64 (如128) → 8 warps (256线程)
    num_warps = 4 if Lk <= 64 else 8

    # num_stages: Triton pipeline 阶段数，=1 减少寄存器溢出
    num_stages = 1

    # --- 第5步: 根据 attn_score 形状分派到不同 kernel 变体 ---

    if attn_score is None:
        # =================================================================
        # 分支 A: 不收集注意力分数 → 纯 Flash Attention
        # =================================================================
        # 场景: Vanilla 所有层、SnapKV 非 eviction 步、非观察层的 full_attn_layer
        # 行为: 标准 Flash Attention，只管算输出不管 token 选择
        _fwd_kernel[grid](
            q, k, v, sm_scale, o, b_start_loc, b_seq_len, req_to_token_indexs, b_req_idx,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            req_to_token_indexs.stride(0), req_to_token_indexs.stride(1),
            kv_group_num=kv_group_num, b_prompt_cache_len=b_prompt_cache_len,
            H=head, BLOCK_DMODEL=Lk, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=num_warps, num_stages=num_stages,
        )
    elif attn_score.dim() == 4:
        # =================================================================
        # 分支 B0: 收集 AttentionPredictor block-level tail 分数
        # =================================================================
        # 形状: (B, num_heads, history_step, ceil(kv_len / block_size))
        # 行为: 模型 attention 仍完整看 KV；旁路分数只保存每个 block 内
        #       max softmax probability，供 AttentionPredictor 初始化历史。
        if attn_score_block_size is None:
            raise ValueError("4D prefill attn_score requires attn_score_block_size")
        attn_score_block_size = int(attn_score_block_size)
        if attn_score_block_size <= 0:
            raise ValueError("attn_score_block_size must be > 0")
        if BLOCK_N % attn_score_block_size != 0:
            raise ValueError(
                f"attn_score_block_size={attn_score_block_size} must divide BLOCK_N={BLOCK_N}"
            )
        _fwd_kernel_with_tail_score[grid](
            q, k, v, sm_scale, o, b_start_loc, b_seq_len, req_to_token_indexs, b_req_idx,
            attn_score,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            req_to_token_indexs.stride(0), req_to_token_indexs.stride(1),
            attn_score.stride(0), attn_score.stride(1), attn_score.stride(2), attn_score.stride(3),
            kv_group_num=kv_group_num, b_prompt_cache_len=b_prompt_cache_len,
            HISTORY_STEP=attn_score.shape[2],
            TAIL_BLOCK_SIZE=attn_score_block_size,
            TAIL_BLOCKS_PER_N=BLOCK_N // attn_score_block_size,
            H=head, BLOCK_DMODEL=Lk, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=num_warps, num_stages=num_stages,
        )
    elif attn_score.dim() == 3:
        # =================================================================
        # 分支 B: 收集 3D 分数 (B, num_heads, kv_len)
        # =================================================================
        # 场景: OmniKV/DeltaKV 的 obs_layer, SnapKV/PyramidKV 所有层
        # 行为: 算 attention 的同时，将原始 QK^T 分数累加到 attn_score
        #   写入方式: 对每个 KV block，对 Q 维度求和 → atomic_add 到 attn_score
        #   之后 on_layer_end() 再对 head 维度做 max pooling，得到每个 KV 位置的总分
        _fwd_kernel_with_score[grid](
            q, k, v, sm_scale, o, b_start_loc, b_seq_len, req_to_token_indexs, b_req_idx,
            attn_score,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            req_to_token_indexs.stride(0), req_to_token_indexs.stride(1),
            attn_score.stride(0), attn_score.stride(1), attn_score.stride(2),
            kv_group_num=kv_group_num, b_prompt_cache_len=b_prompt_cache_len,
            H=head, BLOCK_DMODEL=Lk, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=num_warps, num_stages=num_stages,
        )
    else:
        # =================================================================
        # 分支 C: 收集 2D 分数 (B, kv_len) — attn_score.dim() == 2
        # =================================================================
        # 场景: 不需要 head 维度细分的 token 重要性评分
        # 行为: kernel 内部直接完成跨 head 的聚合:
        #   1. 对 Q tokens 求 mean (除以序列长度归一化)
        #   2. 对 head 维度求 max (atomic_max)
        #   最终 attn_score[b, kv_pos] = max_h mean_q QK[b,h,q,kv_pos]
        # ash=0: 2D 张量没有 head 维度的步长，传0避免 kernel 用错误 stride 寻址
        _fwd_kernel_with_score_2d[grid](
            q, k, v, sm_scale, o, b_start_loc, b_seq_len, req_to_token_indexs, b_req_idx,
            attn_score,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            req_to_token_indexs.stride(0), req_to_token_indexs.stride(1),
            attn_score.stride(0), 0, attn_score.stride(1), # ash=0: 无 head 维度步长
            kv_group_num=kv_group_num, b_prompt_cache_len=b_prompt_cache_len,
            H=head, BLOCK_DMODEL=Lk, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=num_warps, num_stages=num_stages,
        )
