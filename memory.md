# AttnPredict-Offload Session Memory

Date: 2026-05-23
Repo: `/home/zhaowei/Sparse-vLLM`
Main method: `vllm_sparse_method="attnpredict-offload"`
Main files discussed/modified:

- `src/sparsevllm/engine/cache_manager/attnpredict.py`
- `src/sparsevllm/engine/cache_manager/attnpredict_offload.py`
- `src/sparsevllm/engine/cache_manager/base.py`
- `src/sparsevllm/engine/sparse_controller.py`
- `src/sparsevllm/layers/attention.py`
- `src/sparsevllm/triton_kernel/context_flashattention_nopad.py`
- `src/sparsevllm/config.py`
- `scripts/bench_attnpredict_vs_vanilla_128k.sh`
- Reference original implementation: `attentionpredictor原始核心python实现.py`

## User Preferences And Ground Rules

- User wants objective answers, not agreement.
- Reply in Chinese.
- For this repo, keep `attention.py` method-agnostic and put sparse-method runtime state in cache managers.
- `attnpredict-offload` should remain a cache-manager-first implementation.
- Be careful with budget semantics: `num_top_tokens` is the final keep budget. `sink` and `recent` are included in it, not added on top.

## Baseline AttnPredict / Offload Strategy

The intended `attnpredict-offload` design is:

1. Maintain a per-layer GPU active KV pool.
2. Maintain a CPU full KV backing store.
3. During prefill, use full attention so the model still sees the complete prompt.
4. After prefill, predictor chooses a sparse hot set for decode.
5. During decode, only the selected active set should be resident/read on GPU.
6. If a needed token was evicted from GPU, reload it from CPU backing.
7. Prediction and prefetch should run asynchronously where possible.

Important distinction:

- Model attention visibility in prefill should remain full-length.
- Predictor-side scores can be compressed or pooled; this should not change model attention computation.

## Original / Pre-Optimization Flow Before Tail-Score Blockization

Before tail-score optimization, prefill predictor initialization used token-level 4D `tail_score`:

```text
[B, H, history, max_len]
```

At 128K, BS=1, H=32, history=64, fp32:

```text
1 * 32 * 64 * 128000 * 4 bytes ~= 1 GB per layer
```

This tensor was only for predictor input collection, not for model output itself.

Problem:

- `_get_available_slots_info()` allocates remaining GPU memory to KV cache.
- Token-level `tail_score` was allocated later and was not considered in that KV allocation.
- With `GPU_MEMORY_UTILIZATION=0.8`, server hit OOM trying to allocate about 1000 MiB for `tail_score`.

User's first OOM result:

```text
gpu_memory_utilization=0.8
AttnPredict offload allocation: gpu_active_slots=493198 cpu_full_slots=128164 layers=32
OOM in _make_prefill_tail_score_view(), torch.full(...), tried to allocate 1000.00 MiB
Vanilla 128K BS=1:
  TTFT 22.84s, PreTP 5604 tok/s, Decode 29.1 tok/s, Mem 75.61 GB
attnpredict-offload:
  FAILED
```

## Tail-Score Optimization Implemented

Changed prefill predictor `tail_score` from token-level to block-level:

```text
old: [B, H, history, max_len]
new: [B, H, history, ceil(max_len / attnpredict_pooling_block_size)]
```

For 128K, BS=1, H=32, history=64, block=16, fp32:

```text
old ~= 1 GB/layer
new ~= 64 MB/layer
```

Key implementation decisions:

- Delete old token-level 4D fallback. In offload path, 4D prefill score means block-level tail score.
- Add generic cache-manager hook:

```python
prefill_attn_score_block_size(layer_idx) -> int | None
```

- `attention.py` passes the block size generically into `context_attention_fwd()`.
- Triton prefill 4D branch writes block-level max softmax probability.
- Model prefill attention still computes against the full KV sequence.
- `_predict_prefill_positions_sync()` consumes pooled block scores directly and no longer constructs token-level score then max-pools.
- `AttnPredictCacheManager` got helper logic for updating prediction from pooled history.

Validation done locally:

- `python -m py_compile` passed for touched Python files.
- Synthetic CUDA reference passed:
  - attention output diff about `4.9e-4`
  - block tail-score diff about `1e-7`
- Small 2D/3D prefill score paths were checked.

## Cross-Step Reuse Implemented

Goal:

- Avoid running predictor / with-score decode / offload refresh every decode step.
- Use previous prediction for several steps.

New config:

```python
attnpredict_reuse_steps = 4
attnpredict_max_stale_steps = 8
```

Semantics:

- Lease stores only predicted middle hot positions.
- Sink/recent/current are dynamically composed each decode step.
- Final active positions are:

```text
sink positions + leased middle hot positions + dynamic recent positions + current token
```

Budget correction:

- Although composed conceptually from sink/hot/recent/current, the final keep set should be about `num_top_tokens`.
- `num_top_tokens=4096` is the final budget.
- `sink=64` and `recent=512` are included in this budget.
- Middle predictor budget is:

```python
middle_budget = num_top_tokens - num_sink_tokens - num_recent_tokens
```

In current code this is implemented in `AttnPredictCacheManager._create_tsp_mask()`.

Decode refresh strategy:

- `SparseController._needs_attn_score()` calls cache-manager hook:

```python
should_collect_decode_attn_score(layer_idx)
```

- If no lease exists or lease age reaches `reuse_steps`, collect decode `attn_score`.
- If a refresh future is pending, do not collect another score; keep using old lease.
- If stale age exceeds `max_stale_steps`, force wait/consume refresh.

Important helpers after simplification:

- `_ensure_positions_loaded`
- `_hot_positions_from_mask`
- `_compose_positions_from_hot`
- `_commit_hot_leases`
- `should_collect_decode_attn_score`
- `_consume_prefetch`
- `get_layer_store_view`
- `predict_next_mask`
- `_predict_next_positions_sync`

Removed/inlined during simplification because user felt helper count was too high:

- `_positions_from_mask`
- `_compose_decode_positions`
- `_compose_targets_for_lease`
- `_lease_age`
- `_must_wait_prefetch`

## Current Offload Decode Flow After Optimization

At each decode step:

1. `_prepare_decode()` allocates CPU full backing slot for current token.
2. GPU slot for current token is not allocated globally in prepare.
3. For each layer, `get_layer_store_view(layer_idx)`:
   - polls or waits for prefetch future if necessary,
   - allocates current token GPU slot for this layer,
   - composes visible positions from current hot lease plus dynamic sink/recent/current,
   - ensures positions are resident or loaded,
   - stores layer-specific `slot_mapping`.
4. `attention.py` writes current K/V into cache via `store_kvcache()`.
5. `cache_manager.on_kv_stored()` copies K/V into CPU full backing.
6. `SparseController.get_read_view()` returns the mapping and optional `attn_score`.
7. `cache_manager.build_decode_view()` packs actual visible GPU slots for decode kernel.
8. Decode attention runs with either with-score or no-score kernel.
9. `SparseController.on_attention_end()` may call `predict_next_mask()` on refresh steps.
10. `predict_next_mask()` launches async predict+prefetch worker if prefetch is enabled.

## Benchmark Results User Reported

### After lowering GPU utilization before reuse was effective

User ran with `GPU_MEMORY_UTILIZATION=0.6`:

```text
Vanilla 128K BS=1:
  TTFT 22.84s
  PreTP 5604.2 tok/s
  Decode 29.1 tok/s
  ITL 34.37 ms
  Mem 56.62 GB

attnpredict-offload 128K BS=1:
  TTFT 30.81s
  PreTP 4154.6 tok/s
  Decode 1.23 tok/s
  ITL 811.90 ms
  Mem 72.70 GB
```

Analysis at that stage:

- No OOM because GPU utilization was lower.
- Decode was extremely slow.
- Offload overhead dominated.

### After block tail-score and cross-step reuse

Server script used:

```text
LENGTHS=128000
BATCH_SIZES=1
OUTPUT_LEN=64
GPU_MEMORY_UTILIZATION=0.8
CHUNK_PREFILL_SIZE=4096
NUM_TOP_TOKENS=4096
NUM_SINK_TOKENS=64
NUM_RECENT_TOKENS=512
ATTNPREDICT_HISTORY_STEPS=64
ATTNPREDICT_POOLING_BLOCK_SIZE=16
ATTNPREDICT_OFFLOAD_PREFETCH=true
ATTNPREDICT_OFFLOAD_CPU_THREADS=8
ATTNPREDICT_OFFLOAD_CPU_SLOTS=-1
ATTNPREDICT_OFFLOAD_CPU_MEMORY_UTILIZATION=0.8
ATTNPREDICT_OFFLOAD_PIN_STAGING=true
```

Note: script did not explicitly pass `ATTNPREDICT_REUSE_STEPS` and `ATTNPREDICT_MAX_STALE_STEPS`, but config log showed defaults were active:

```text
attnpredict_reuse_steps=4
attnpredict_max_stale_steps=8
```

Result:

```text
Vanilla 128K BS=1:
  TTFT 22.80s
  PreTP 5613.0 tok/s
  Decode 29.17 tok/s
  ITL 34.28 ms
  Mem 75.61 GB

attnpredict-offload 128K BS=1:
  TTFT 30.69s
  PreTP 4170.6 tok/s
  Decode 2.91 tok/s
  ITL 343.46 ms
  Mem 80.22 GB
```

Interpretation:

- Reuse improved decode from about `1.2 tok/s` to `2.91 tok/s`.
- But performance is still unacceptable.
- With only about 4096 visible KV tokens, decode should not be 10x slower than 128K full attention.
- Therefore the bottleneck is likely not attention computation, but CPU/GPU/cache-manager overhead.

## Critical Correctness / Semantics Notes

### Top-k Budget

Do not double count sink/recent.

Wrong:

```text
visible ~= topk + sink + recent
```

Correct:

```text
visible ~= topk
middle_hot_budget = topk - sink - recent
```

With config:

```text
topk=4096
sink=64
recent=512
middle_hot_budget=3520
```

Current token is normally included in recent because context length already includes current token during decode.

### Block-Level Tail Score Does Not Shorten Model Attention

Block-level `tail_score` is only predictor-side compression.

It does not mean the model prefill attention only attends to 8K blocks/tokens.
The prefill attention kernel still computes against full KV.

### Decode `attn_score`

Decode `attn_score` is produced by the attention kernel on refresh steps.
It is not a separate already-known score.
The with-score decode kernel writes logits/scores so predictor can update history.

### CUDA Tensor CPU Access

Avoid accidental sync:

- `.item()`, `.tolist()`, `.cpu()`, `.numpy()`, and GPU->CPU `.to("cpu")` can synchronize.
- A CUDA tensor is not silently moved to CPU for arbitrary CPU operation; explicit CPU extraction/copy is needed, and it can be expensive.

### `_allocate_gpu_slots_locked`

The `locked` suffix means caller must already hold the relevant layer lock.
This is because free slot stack and residency maps may be touched by main thread, background prefetch, and free path.

## Current Main Suspected Bottlenecks

### 1. `gpu_active_slots` Is Larger Than Prompt Length In Current Benchmark

The server log showed:

```text
gpu_active_slots=493198
context_len=128000
```

This means for BS=1 128K, GPU active pool is enough to keep full KV.

Current offload path therefore does:

```text
GPU full-enough active pool
+ CPU full backing
+ offload residency bookkeeping
+ sparse view packing
```

This is the worst case for offload. It does not test the intended constrained-GPU advantage.

Potential fix:

- Add a fast path when active GPU pool can hold the full current batch context.
- In that mode, skip CPU offload residency/reload management and only build sparse view.

### 2. `on_kv_stored()` Sync GPU->CPU Copy

Current code copies every stored K/V to CPU backing:

```python
host_k = k.detach().to(device="cpu", dtype=self.hf_config.torch_dtype)
host_v = v.detach().to(device="cpu", dtype=self.hf_config.torch_dtype)
```

This happens inside every layer during decode.

Even though current-token KV is small, GPU->CPU copy can synchronize and add large per-layer latency.

Potential fix:

- Lazy CPU backing: only copy a token to CPU before it is evicted from GPU.
- Or async D2H to pinned CPU memory with event tracking.
- Do not block each layer's decode path on CPU backing if the token will remain resident.

### 3. `_ensure_positions_resident()` Full 128K Scan

Current logic scans:

```python
np.flatnonzero(mirror[row_idx, :full_len] >= 0)
```

For 128K, 32 layers, 64 decode tokens this is a large CPU-side repeated scan.

Potential fix:

- Maintain per-layer/per-row resident position sets or sorted arrays.
- Compute incremental diff:

```text
to_load = new_view - resident
to_free = old_view - new_view
```

- Complexity should be based on changed tokens, not full sequence length.

### 4. `build_decode_view()` Python Packing

Current build path repeatedly:

- loops over positions in Python,
- reads CPU mirror slot values,
- creates CUDA tensors for packed positions/slots.

For topk=4096, this is thousands of Python operations per layer per step.

Potential fix:

- Cache hot/sink/recent packed slots.
- Reuse per-layer decode buffers.
- On reuse steps, only update rolling recent/current parts.
- Prefer GPU gather or vectorized tensor writes over Python list construction.

### 5. First Decode May Pay Large Release Cost

Prefill keeps full GPU KV until final active pruning.
If pruning/release is delayed until first decode, the first decode step can pay a large cost.

Need verify with profiling.

### 6. Refresh Still Builds Full Attention Vector

`_predict_next_positions_sync()` on refresh step scatters sparse attention into:

```text
[heads, full_len]
```

At 128K this is still expensive, although only every `reuse_steps`.

Potential fix:

- Move decode history update to block-level, analogous to block-level prefill tail_score.
- Avoid constructing token-level `full_attn`.

## Recommended Next Plan

Priority order:

1. Add detailed profiler scopes without changing behavior.
2. Run short output benchmark with profiler:

```bash
PROFILER_SVLLM=1 CUDA_SYNC_SVLLM=1 \
LENGTHS=128000 BATCH_SIZES=1 OUTPUT_LEN=8 \
bash scripts/bench_attnpredict_vs_vanilla_128k.sh
```

3. Run non-sync profiler benchmark:

```bash
PROFILER_SVLLM=1 \
LENGTHS=128000 BATCH_SIZES=1 OUTPUT_LEN=64 \
bash scripts/bench_attnpredict_vs_vanilla_128k.sh
```

4. Add full-resident fast path:

```text
if max context in current batch <= gpu_active_slots:
    skip offload eviction/reload path
    keep GPU full mapping
    only use sparse decode view
```

5. Convert residency management to incremental diff.
6. Cache decode view buffers/slots.
7. Make CPU backing lazy or async.
8. Optimize decode predictor history to block-level.

## Useful Diagnostic Experiments

To check whether predictor is still the bottleneck:

```bash
EXTRA_HYPER_PARAMS_JSON='{"attnpredict_reuse_steps":100000,"attnpredict_max_stale_steps":100000}' \
LENGTHS=128000 BATCH_SIZES=1 OUTPUT_LEN=64 \
bash scripts/bench_attnpredict_vs_vanilla_128k.sh
```

If decode remains very slow, predictor is not the main problem.

To test smaller sparse budget:

```bash
NUM_TOP_TOKENS=1024 \
EXTRA_HYPER_PARAMS_JSON='{"attnpredict_reuse_steps":100000,"attnpredict_max_stale_steps":100000}' \
LENGTHS=128000 BATCH_SIZES=1 OUTPUT_LEN=64 \
bash scripts/bench_attnpredict_vs_vanilla_128k.sh
```

If decode barely improves, fixed CPU/Python management overhead dominates.

Important caveat:

- Current offload allocation ignores explicit `num_kvcache_slots` in its allocation path by recomputing from available memory.
- To test truly constrained active slots, may need to add explicit active-slot cap support or reduce GPU memory utilization much more.

## Current High-Level Conclusion

Tail-score OOM was a real bug/design issue and block-level tail score fixed the large temporary allocation.

Cross-step reuse is working in the sense that decode improved from about `1.2 tok/s` to `2.91 tok/s`.

However, the method is still far too slow. With `topk=4096`, decode should not be 10x slower than 128K vanilla full attention.

The next bottleneck is almost certainly system overhead:

- synchronous CPU backing writes,
- full-length residency scans,
- Python decode-view packing,
- missing fast path when GPU active pool can already hold full context,
- possible delayed release cost after prefill,
- token-level full-attn scatter during refresh.

Future sessions should not keep tuning `reuse_steps` first. Start with profiling and then optimize residency/view/CPU-copy paths.
