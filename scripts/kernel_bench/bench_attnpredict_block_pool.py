import argparse
import statistics

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_stats_kernel(
    logits,
    max_out,
    denom_out,
    view_len: tl.constexpr,
    attn_scale: tl.constexpr,
    stride_h: tl.constexpr,
    stride_v: tl.constexpr,
    BLOCK_VIEW: tl.constexpr,
):
    head = tl.program_id(0)
    offs = tl.arange(0, BLOCK_VIEW)
    mask = offs < view_len
    x = tl.load(logits + head * stride_h + offs * stride_v, mask=mask, other=-float("inf"))
    x = x.to(tl.float32) * attn_scale
    m = tl.max(x, axis=0)
    denom = tl.sum(tl.where(mask, tl.exp(x - m), 0.0), axis=0)
    tl.store(max_out + head, m)
    tl.store(denom_out + head, denom)


@triton.jit
def _non_atomic_pool_kernel(
    logits,
    positions,
    max_vals,
    denom_vals,
    out,
    view_len: tl.constexpr,
    pooled_len: tl.constexpr,
    attn_scale: tl.constexpr,
    pooling_block_size: tl.constexpr,
    stride_h: tl.constexpr,
    stride_v: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    SEARCH_ITERS: tl.constexpr,
):
    block_id = tl.program_id(0)
    head = tl.program_id(1)
    block_start_pos = block_id * pooling_block_size
    block_end_pos = block_start_pos + pooling_block_size

    lo = tl.full((), 0, tl.int32)
    hi = tl.full((), view_len, tl.int32)
    for _ in tl.static_range(SEARCH_ITERS):
        mid = (lo + hi) // 2
        val = tl.load(positions + mid, mask=mid < view_len, other=2147483647)
        is_left = val < block_start_pos
        lo = tl.where(is_left, mid + 1, lo)
        hi = tl.where(is_left, hi, mid)
    start = lo

    lo2 = tl.full((), 0, tl.int32)
    hi2 = tl.full((), view_len, tl.int32)
    for _ in tl.static_range(SEARCH_ITERS):
        mid = (lo2 + hi2) // 2
        val = tl.load(positions + mid, mask=mid < view_len, other=2147483647)
        is_left = val < block_end_pos
        lo2 = tl.where(is_left, mid + 1, lo2)
        hi2 = tl.where(is_left, hi2, mid)
    end = lo2

    offs = start + tl.arange(0, BLOCK_TOKENS)
    mask = offs < end
    x = tl.load(logits + head * stride_h + offs * stride_v, mask=mask, other=-float("inf"))
    m = tl.load(max_vals + head)
    denom = tl.load(denom_vals + head)
    attn = tl.exp(x.to(tl.float32) * attn_scale - m) / denom
    score = tl.max(tl.where(mask, attn, 0.0), axis=0)
    tl.store(out + head * pooled_len + block_id, score)


@triton.jit
def _compact_pool_kernel(
    logits,
    positions,
    block_ids,
    max_vals,
    denom_vals,
    out,
    view_len: tl.constexpr,
    num_blocks: tl.constexpr,
    pooled_len: tl.constexpr,
    attn_scale: tl.constexpr,
    pooling_block_size: tl.constexpr,
    stride_h: tl.constexpr,
    stride_v: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    SEARCH_ITERS: tl.constexpr,
):
    block_ord = tl.program_id(0)
    head = tl.program_id(1)
    block_id = tl.load(block_ids + block_ord).to(tl.int32)
    block_start_pos = block_id * pooling_block_size
    block_end_pos = block_start_pos + pooling_block_size

    lo = tl.full((), 0, tl.int32)
    hi = tl.full((), view_len, tl.int32)
    for _ in tl.static_range(SEARCH_ITERS):
        mid = (lo + hi) // 2
        val = tl.load(positions + mid, mask=mid < view_len, other=2147483647)
        is_left = val < block_start_pos
        lo = tl.where(is_left, mid + 1, lo)
        hi = tl.where(is_left, hi, mid)
    start = lo

    lo2 = tl.full((), 0, tl.int32)
    hi2 = tl.full((), view_len, tl.int32)
    for _ in tl.static_range(SEARCH_ITERS):
        mid = (lo2 + hi2) // 2
        val = tl.load(positions + mid, mask=mid < view_len, other=2147483647)
        is_left = val < block_end_pos
        lo2 = tl.where(is_left, mid + 1, lo2)
        hi2 = tl.where(is_left, hi2, mid)
    end = lo2

    offs = start + tl.arange(0, BLOCK_TOKENS)
    mask = offs < end
    x = tl.load(logits + head * stride_h + offs * stride_v, mask=mask, other=-float("inf"))
    m = tl.load(max_vals + head)
    denom = tl.load(denom_vals + head)
    attn = tl.exp(x.to(tl.float32) * attn_scale - m) / denom
    score = tl.max(tl.where(mask, attn, 0.0), axis=0)
    tl.store(out + head * pooled_len + block_id, score)


def make_positions(
    full_len: int,
    view_len: int,
    pooling_block_size: int,
    sink_tokens: int,
    recent_tokens: int,
) -> torch.Tensor:
    middle = max(0, view_len - sink_tokens - recent_tokens)
    sink = torch.arange(0, sink_tokens, dtype=torch.int32, device="cuda")
    recent_start = full_len - recent_tokens
    recent = torch.arange(recent_start, full_len, dtype=torch.int32, device="cuda")
    max_mid_blocks = max(0, (recent_start - sink_tokens) // pooling_block_size)
    num_mid_blocks = middle // pooling_block_size
    step = max(1, max_mid_blocks // max(1, num_mid_blocks))
    blocks = torch.arange(0, num_mid_blocks, dtype=torch.int32, device="cuda") * step
    blocks = torch.clamp(blocks, max=max(0, max_mid_blocks - 1))
    offsets = torch.arange(0, pooling_block_size, dtype=torch.int32, device="cuda")
    mid = sink_tokens + blocks[:, None] * pooling_block_size + offsets[None, :]
    positions = torch.cat([sink, mid.reshape(-1), recent]).sort().values
    return positions[:view_len].contiguous()


def torch_pool(logits: torch.Tensor, positions: torch.Tensor, pooled_len: int, scale: float, pool: int):
    attn = torch.softmax(logits * scale, dim=-1)
    block_idx = positions.to(torch.long) // pool
    out = torch.zeros((logits.shape[0], pooled_len), dtype=torch.float32, device=logits.device)
    out.scatter_reduce_(1, block_idx.unsqueeze(0).expand(logits.shape[0], -1), attn, reduce="amax", include_self=True)
    return out


def triton_pool(logits: torch.Tensor, positions: torch.Tensor, pooled_len: int, scale: float, pool: int):
    heads, view_len = logits.shape
    max_vals = torch.empty((heads,), dtype=torch.float32, device=logits.device)
    denom_vals = torch.empty((heads,), dtype=torch.float32, device=logits.device)
    out = torch.empty((heads, pooled_len), dtype=torch.float32, device=logits.device)
    block_view = triton.next_power_of_2(view_len)
    search_iters = (view_len - 1).bit_length()
    _softmax_stats_kernel[(heads,)](
        logits,
        max_vals,
        denom_vals,
        view_len,
        float(scale),
        logits.stride(0),
        logits.stride(1),
        BLOCK_VIEW=block_view,
        num_warps=8,
    )
    _non_atomic_pool_kernel[(pooled_len, heads)](
        logits,
        positions,
        max_vals,
        denom_vals,
        out,
        view_len,
        pooled_len,
        float(scale),
        int(pool),
        logits.stride(0),
        logits.stride(1),
        BLOCK_TOKENS=triton.next_power_of_2(pool),
        SEARCH_ITERS=search_iters,
        num_warps=1,
    )
    return out


def triton_compact_pool(
    logits: torch.Tensor,
    positions: torch.Tensor,
    block_ids: torch.Tensor,
    pooled_len: int,
    scale: float,
    pool: int,
):
    heads, view_len = logits.shape
    max_vals = torch.empty((heads,), dtype=torch.float32, device=logits.device)
    denom_vals = torch.empty((heads,), dtype=torch.float32, device=logits.device)
    out = torch.empty((heads, pooled_len), dtype=torch.float32, device=logits.device)
    out.zero_()
    block_view = triton.next_power_of_2(view_len)
    search_iters = (view_len - 1).bit_length()
    _softmax_stats_kernel[(heads,)](
        logits,
        max_vals,
        denom_vals,
        view_len,
        float(scale),
        logits.stride(0),
        logits.stride(1),
        BLOCK_VIEW=block_view,
        num_warps=8,
    )
    _compact_pool_kernel[(block_ids.numel(), heads)](
        logits,
        positions,
        block_ids,
        max_vals,
        denom_vals,
        out,
        view_len,
        block_ids.numel(),
        pooled_len,
        float(scale),
        int(pool),
        logits.stride(0),
        logits.stride(1),
        BLOCK_TOKENS=triton.next_power_of_2(pool),
        SEARCH_ITERS=search_iters,
        num_warps=1,
    )
    return out


def time_fn(fn, repeats: int, warmup: int):
    times = []
    for i in range(warmup + repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(end))
    return out, times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--view-len", type=int, default=4096)
    parser.add_argument("--full-len", type=int, default=128000)
    parser.add_argument("--pool", type=int, default=16)
    parser.add_argument("--sink", type=int, default=64)
    parser.add_argument("--recent", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    pooled_len = triton.cdiv(args.full_len, args.pool)
    scale = 1.0 / (128 ** 0.5)
    logits = torch.randn(args.heads, args.view_len, dtype=torch.float32, device="cuda")
    positions = make_positions(args.full_len, args.view_len, args.pool, args.sink, args.recent)
    block_ids = torch.unique_consecutive(positions // args.pool).to(torch.int32)

    ref = torch_pool(logits, positions, pooled_len, scale, args.pool)
    got = triton_pool(logits, positions, pooled_len, scale, args.pool)
    got_compact = triton_compact_pool(logits, positions, block_ids, pooled_len, scale, args.pool)
    torch.cuda.synchronize()
    max_abs = (ref - got).abs().max().item()
    max_abs_compact = (ref - got_compact).abs().max().item()
    if not torch.allclose(ref, got, rtol=1e-5, atol=1e-6):
        raise SystemExit(f"Full block mismatch: max_abs={max_abs}")
    if not torch.allclose(ref, got_compact, rtol=1e-5, atol=1e-6):
        raise SystemExit(f"Compact block mismatch: max_abs={max_abs_compact}")

    _, torch_times = time_fn(lambda: torch_pool(logits, positions, pooled_len, scale, args.pool), args.repeats, args.warmup)
    _, triton_times = time_fn(lambda: triton_pool(logits, positions, pooled_len, scale, args.pool), args.repeats, args.warmup)
    _, compact_times = time_fn(
        lambda: triton_compact_pool(logits, positions, block_ids, pooled_len, scale, args.pool),
        args.repeats,
        args.warmup,
    )
    _, compact_with_ids_times = time_fn(
        lambda: triton_compact_pool(
            logits,
            positions,
            torch.unique_consecutive(positions // args.pool).to(torch.int32),
            pooled_len,
            scale,
            args.pool,
        ),
        args.repeats,
        args.warmup,
    )

    def fmt(times):
        return f"median={statistics.median(times):.4f}ms mean={statistics.mean(times):.4f}ms"

    print(
        f"shape heads={args.heads} view_len={args.view_len} pooled_len={pooled_len} "
        f"active_blocks={block_ids.numel()} pool={args.pool}"
    )
    print(f"correctness full_max_abs={max_abs:.3e} compact_max_abs={max_abs_compact:.3e}")
    print(f"torch_softmax_scatter {fmt(torch_times)}")
    print(f"triton_non_atomic    {fmt(triton_times)}")
    print(f"triton_compact       {fmt(compact_times)}")
    print(f"compact_with_ids     {fmt(compact_with_ids_times)}")
    print(f"speedup_full={statistics.median(torch_times) / statistics.median(triton_times):.3f}x")
    print(f"speedup_compact={statistics.median(torch_times) / statistics.median(compact_times):.3f}x")
    print(f"speedup_compact_with_ids={statistics.median(torch_times) / statistics.median(compact_with_ids_times):.3f}x")


if __name__ == "__main__":
    main()
