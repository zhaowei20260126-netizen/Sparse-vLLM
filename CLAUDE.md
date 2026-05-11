# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Install

```bash
conda create -n svllm python=3.10 -y && conda activate svllm
pip install torch==2.8.0 transformers[torch]==4.53.3 accelerate deepspeed==0.15.4 torchvision datasets==4.1.0
pip install fire matplotlib seaborn wandb loguru ansible
MAX_JOBS=8 pip install flash-attn --no-build-isolation
pip install -e .
```

Optional CUDA gather extension (only for `deltakv-triton-v3-cuda-offload`):
```bash
cd src/sparsevllm/cuda_kernel && pip install -e .
```

## Tests and verification

```bash
# Correctness smoke test (compares sparse methods against vanilla)
python scripts/test_sparse_vllm_correctness.py

# DeltaKV checkpoint config sync
python tests/test_deltakv_checkpoint_config_sync.py

# Throughput benchmark
python scripts/bench_sparse_vllm.py \
  --model_path <MODEL_PATH> --lengths 512000 --batch_sizes 2 \
  --methods vanilla --hyper_params '{"gpu_memory_utilization": 0.9}'
```

## Architecture

Two separate packages under `src/`:

### 1. `src/sparsevllm/` — Sparse-first inference engine

Public API: `sparsevllm.LLM` (aliases `LLMEngine` at [llm.py](src/sparsevllm/llm.py)).

**Initialization flow** (`LLMEngine.__init__`):
1. `Config` parses and validates all knobs (method, budgets, compressor paths)
2. TP subprocesses spawn for ranks > 0, each running `ModelRunner.__init__`
3. Rank 0 creates `ModelRunner` → loads model weights → `CacheManager.create()` → `SparseController`
4. `Scheduler` receives the CacheManager reference for memory-aware scheduling
5. Warmup runs one dummy prefill+decode step

**CacheManager factory** (`cache_manager/__init__.py` + `base.py:CacheManager.create()`):
Dispatches by `config.vllm_sparse_method`:
- `""` → `StandardCacheManager`
- `"streamingllm"` → `StreamingLLMCacheManager`
- `"snapkv"/"pyramidkv"` → `SnapKVCacheManager`
- `"quest"` → `QuestCacheManager`
- `"omnikv"` → `OmniKVCacheManager`
- `"deltakv*"` → one of the `DeltaKVCacheTritonManager*` variants in [deltakv.py](src/sparsevllm/engine/cache_manager/deltakv.py)
- `"deltakv-standalone"` → `DeltaKVStandaloneCacheManager`
- `"deltakv-snapkv"` → `DeltaKVSnapKVCacheManager`

`CacheManager` is the **physical storage layer** — owns `kv_cache` tensors, allocates/frees slots, tracks `req_to_token_slots` mappings. Method-specific managers override eviction, compression, and reconstruction behavior.

**Step loop** (`LLMEngine.step()`):
1. `scheduler.schedule()` — picks either all-prefill or all-decode batch (never mixed)
2. `model_runner.call("run", seqs, is_prefill)` — all TP ranks execute in parallel
3. `scheduler.postprocess()` — updates sequence state, migrates between waiting/decoding queues
4. Finished sequences have their KV slots freed

**Scheduler**: Two queues — `waiting` (new/chunked-prefill/preempted) and `decoding`. Prefill is chunked by `chunk_prefill_size`. Long-text and short-text batches are separated by threshold.

**ModelRunner.run()** ([model_runner.py:358](src/sparsevllm/engine/model_runner.py#L358)):
1. `cache_manager.prepare_step()` — gets input_ids, positions, assigns physical slots
2. `sparse_controller.prepare_forward()` — resets per-layer sparse states, allocates attn_score tensors
3. `model(input_ids, positions)` — full forward pass
4. `sampler(logits, temperatures)` — token sampling (rank 0 only)
5. `sparse_controller.post_forward()` — eviction/compression (SnapKV prefinal evict, DeltaKV compress, etc.)

**Attention layer** ([layers/attention.py](src/sparsevllm/layers/attention.py#L174)):
Method-agnostic by design. Forward order:
1. `cache_manager.get_layer_store_view()` → write KV to physical slots via Triton kernel
2. `cache_manager.on_kv_stored()` (QuEST uses this to update page metadata)
3. `sparse_controller.get_read_view()` → gets `active_slots` for this layer (the key sparse decision point)
4. Run prefill/decode attention kernel with those active slots
5. If DeltaKV, free temp reconstruction slots after use

**SparseController** ([engine/sparse_controller.py](src/sparsevllm/engine/sparse_controller.py)):
Per-layer `LayerBatchSparseState` dataclass holds `attn_score`, `active_indices`, `active_slots`, `context_lens`, `req_indices`. Key methods:
- `prepare_forward()` — reset state, allocate attn_score for layers that need it
- `get_read_view(layer_idx)` — the central dispatch: full-attention layers return all slots; sparse layers return filtered slots (OmniKV) or trigger DeltaKV reconstruction
- `on_layer_end(layer_idx)` — for OmniKV/DeltaKV: read attn_score from an "observation layer," select top-k tokens, propagate selection to target sparse layers below
- `post_forward()` — trigger physical eviction/compression after the full forward pass

**Model support**: Qwen2 ([models/qwen2.py](src/sparsevllm/models/qwen2.py)), Qwen3, DeepSeek-V2. DeepSeek-V3.2 sparsevllm path is disabled.

### 2. `src/deltakv/` — DeltaKV compressor training + HF evaluation wrappers

- `train_compressor.py` — trains the compressor (freeze LLM, train compress/decompress modules with MSE+NTP loss)
- `modeling/` — HF-compatible model wrappers: `qwen2/qwen2_e2e.py`, `qwen2/qwen2_e2e_cluster.py`, `qwen2/qwen2_full_deltakv_compress_inference.py`, etc.
- `get_chat_api.py` — factory for `backend="hf"` vs `backend="sparsevllm"` in benchmarks
- `data_prepare/` — tokenization, packing, data collation for training
- `analysis/` — QKV similarity analysis scripts that motivated the method

### Benchmarks

- `benchmark/long_bench/pred.py` — LongBench evaluation (supports both `backend="hf"` and `backend="sparsevllm"`)
- `benchmark/math_bench/pred.py` — Math benchmarks (GSM8K, AIME)
- `benchmark/scbench/` — SCBench evaluation
- `benchmark/niah/` — Needle-in-a-Haystack

`--hyper_param` accepts a JSON string (or file path) of Config fields. When `--backend sparsevllm`, method selection is entirely through `--hyper_param` including `vllm_sparse_method` and `deltakv_path`.

### Baselines

Vendored under `baselines/`: AdaKV, KIVI, KVZip, PALU, QuEST.

## Important constraints

- Sparse-vLLM enforces **no mixed prefill/decode batches** and **no mixed long/short batches** (scheduler separates them). The error "Mixed long/short batch detected" means you're bypassing scheduler logic.
- DeltaKV in sparsevllm: `assert world_size == 1` (no TP support yet)
- `SamplingParams.temperature` must be `> 0.0` (use `1e-5` for near-greedy)
- `num_tokens` convention: prefill returns positive (prompt chunk tokens), decode returns negative (-batch_size)
- Obs layers are auto-derived: each non-terminal entry in `full_attn_layers` becomes an observation layer whose attn_score guides token selection for all sparse layers below it (until the next full-attn layer)