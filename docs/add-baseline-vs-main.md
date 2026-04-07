# Explanation of Code Changes in `add-baseline` Relative to `main`

## 1. Comparison Scope and Overall Conclusion

This document compares the current branch `add-baseline` against `main`, specifically:

* Comparison range: `main...add-baseline`
* Number of commits: 41
* Code stats: 89 files changed, `+12413 / -1708`

Overall, this branch is not a single-point fix, but a fairly comprehensive expansion covering “baseline support + runtime support + evaluation pipeline completion”:

1. `src/deltakv/` has expanded from a DeltaKV implementation mainly focused on Qwen2/Llama into a broader family that includes more runtime variants, Qwen3 support, quantized loading, more baseline adapters, and training/data generation pipelines.
2. `src/sparsevllm/` adds `deltakv-standalone` and `deltakv-snapkv` as first-class sparse methods, and further solidifies DeltaKV configuration syncing, scheduling, reconstruction, capacity budgeting, and kernel organization.
3. `benchmark/`, `scripts/`, and `baselines/kvzip/` have been systematically strengthened, making it much easier to batch-run, reproduce, filter samples, switch baselines, and cover new method combinations across LongBench, SCBench, MathBench, and NIAH.

In other words, the core objective of this branch is not merely “to get one method working,” but to move the Sparse-vLLM repository from “having a DeltaKV demo” to “supporting comparison across multiple baselines, large-scale batch evaluation, training/analysis, and switching between HF and sparse-vLLM runtimes.”

---

## 2. High-Level Change Trajectory

### 2.1 The DeltaKV model family has clearly expanded

This set of changes grows DeltaKV into a more complete method family rather than a single implementation path:

* Added `full_deltakv`
* Added `origin_residual_quant`
* Added `all_origin_residual_quant`
* Added `deltasnapkv`
* Added Qwen3 support for `e2e / cluster_e2e / cluster_e2e_big / inference`
* Added OmniKV / KIVI loading adapters
* Added helper logic for loading low-bit base models

In other words, this branch has now separated multiple dimensions that were previously entangled: base model loading strategy, compressed cache form, cluster/reference token selection strategy, static pruning behavior, and residual quantization.

### 2.2 sparse-vLLM evolved from “supporting DeltaKV” to “supporting more DeltaKV variants”

The main focus of the `src/sparsevllm/` changes is twofold:

* Under a cache-manager-first design, `deltakv-standalone` and `deltakv-snapkv` are split out into independent cache managers.
* The scheduler, attention layers, runner, kernels, and config synchronization are unified to the point where these new methods can run stably.

The biggest structural change here is that `attention.py` becomes more generalized. It is now mainly responsible for “writing the current layer’s KV” and “reading the compute view provided by the cache manager,” while the question of whether data is read directly from persistent storage or first reconstructed into a temporary pool is handled by the cache manager.

### 2.3 The evaluation and scripting layer has basically become an experiment platform

LongBench, SCBench, MathBench, and NIAH were all enhanced, and in a very consistent way:

* Paths are now configurable through environment variables, reducing hard-coded values.
* Compatibility was added for Qwen3, KVzip, and multiple DeltaKV variants.
* Multi-GPU data-parallel workers, result merging, and logging were filled in.
* Subset/context-length filtering, thinking stripping, and preprocessed parquet reading were added.
* A set of batch queueing and sweep scripts was introduced.

This means the branch does not only add method implementations, but also solves the engineering problem of “how to run all these methods as a coherent large-scale experimental suite.”

---

## 3. Key Code Changes

### 3.1 `src/deltakv/`: comprehensive expansion of configs, loading, model families, training, and analysis

#### Configuration layer

`src/deltakv/configs/model_config_cls.py` introduces several important extensions:

* Adds `KVQwen3Config`, formally bringing Qwen3 into the DeltaKV training/inference config system.
* Adds `parse_full_attn_layers()`, which normalizes string/list layer specifications into integer lists.
* Adds `k_neighbors`, and via `finalize_cluster_args()` makes it compatible with the old `seq_chunk_size`, conceptually separating “number of clustering neighbors” from “chunk granularity.”
* Adds `stride_alpha`, used for dynamic-stride cluster center scheduling.
* Adds `deltakv_use_omnikv_selection`, which controls whether DeltaKV uses OmniKV-style token selection.
* Adds `deltasnapkv_total_budget` and `deltasnapkv_ref_budget`, providing total and reference-token budgets for DeltaSnapKV.
* Changes several defaults noticeably: `use_cluster=True`, `chunk_prefill_accel_omnikv=False`, and `num_top_tokens_in_prefill=8192`.

The meaning of these changes is that the config layer is no longer just a direct container for original DeltaKV parameters. It is now a unified abstraction for multiple compression, selection, and static-pruning paths.

#### Model loading and inference entrypoint

`src/deltakv/get_chat_api.py` is one of the main convergence points in this branch. It adds:

* Official Qwen3 support under `model_cls='deltakv'`
* New loading branches for `full_deltakv`, `origin_residual_quant`, `all_origin_residual_quant`, and `deltasnapkv`
* Adapter entrypoints for `omnikv` and `kivi` baselines
* Low-bit base model loading helper logic on the HF path
* Dtype restoration for modules skipped during quantization after loading compressor weights
* A strict requirement that `deltasnapkv` must use empty `full_attn_layers`, making it explicit that mixed full-attention layers are unsupported

This allows a single unified loading function to cover many more experiment configurations, instead of forcing external scripts to assemble custom logic themselves.

#### Quantization and baseline adaptation

Two important helper modules were added:

* `src/deltakv/quantization.py`

  * Parses inference settings such as `load_in_4bit / load_in_8bit / torch_dtype / quant_skip_modules`
  * Builds `BitsAndBytesConfig`
  * Defines the default set of modules to skip during quantization, such as `compress_down`, `compress_up`, `cluster`, and `transform`
  * Provides `restore_modules_to_dtype()` to restore compressor and clustering modules back to the target precision

* `src/deltakv/baseline_adapters.py`

  * Adds OmniKV loading adapters by reusing the DeltaKV inference family while disabling compression and clustering
  * Adds KIVI loading adapters by injecting `k_bits / v_bits / group_size / residual_length`, and switching to Llama or Mistral variants depending on model type

Together, these greatly reduce the amount of glue code needed to switch baselines within the same benchmark scripts.

#### New and expanded cache/model implementations

New cache implementations include:

* `src/deltakv/modeling/full_deltakv_compress_cache.py`
* `src/deltakv/modeling/origin_residual_quant_cache.py`
* `src/deltakv/modeling/all_origin_residual_quant_cache.py`

Specifically:

* `full_deltakv_compress_cache.py` corresponds to the full DeltaKV compressed-cache path
* `origin_residual_quant_cache.py` corresponds to origin residual quant
* `all_origin_residual_quant_cache.py` corresponds to all-origin residual quant

`src/deltakv/modeling/kv_cache.py` itself is also significantly enhanced. Key updates include:

* Adding dynamic center construction logic based on `stride_alpha`
* Switching cluster-related logic to the new `get_cluster_k_neighbors()`
* Providing the underlying capability needed for dynamic stride center progression, parent-node selection, and cluster maintenance

That is, the new runtime variants are not all reinventing the wheel independently. Shared functionality has been pushed down into `kv_cache.py` and dedicated cache classes.

#### Llama / Qwen2 / Qwen3 model families

On the Llama side:

* Added `llama_deltasnapkv.py`
* Added `llama_full_deltakv_compress_inference.py`
* Added `llama_origin_residual_quant_inference.py`
* Added `llama_all_origin_residual_quant_inference.py`
* Modified `llama_with_compress_inference.py`, `llama_e2e.py`, `llama_e2e_cluster.py`, and `llama_pyramidkv.py`

The most important substantive changes on the Llama side are:

* DeltaSnapKV is implemented as a “static prune + reference-token retention + protected suffix” path built on top of DeltaKV cluster cache
* `llama_with_compress_inference.py` now supports `deltakv_use_omnikv_selection`
* `llama_e2e_cluster.py` now uses the new `k_neighbors` semantics
* Other inference variants are aligned with the new cache/config/quantized-loading logic

On the Qwen2 side:

* Added `qwen2_deltasnapkv.py`
* Added `qwen2_full_deltakv_compress_inference.py`
* Added `qwen2_origin_residual_quant_inference.py`
* Added `qwen2_all_origin_residual_quant_inference.py`
* Modified `qwen2_with_compress_inference.py`, `qwen2_e2e.py`, `qwen2_e2e_cluster.py`, `qwen2_e2e_cluster_for_big_model.py`, `qwen2_snapkv.py`, and `qwen2_pyramidkv.py`

The Qwen2 side plays the same role as the Llama side: wiring in the new cache variants, cluster parameters, and DeltaSnapKV support.

Qwen3 is one of the largest newly added model families in this branch:

* Added `src/deltakv/modeling/qwen3/__init__.py`
* Added `qwen3_e2e.py`
* Added `qwen3_e2e_cluster.py`
* Added `qwen3_e2e_cluster_for_big_model.py`
* Added `qwen3_with_compress_inference.py`
* Added `qwen3_full_deltakv_compress_inference.py`
* Added `qwen3_origin_residual_quant_inference.py`
* Added `qwen3_all_origin_residual_quant_inference.py`

This means the HF-side DeltaKV family now includes Qwen3 for training, cluster training, large-model cluster training, standard inference, and multiple inference variants.

One important detail: Qwen3 is only expanded on the HF path; the sparse-vLLM side still explicitly disables `qwen3 + deltakv`.

#### Training, saving, and data preparation

The key training-path changes are concentrated in the following files:

* `src/deltakv/train_compressor.py`

  * Adds a `deepspeed` parameter
  * Adds `k_neighbors` and finalizes it after config construction
  * Adds a Qwen3 training entrypoint
  * Binds `LOCAL_RANK` to the CUDA device
  * Broadcasts a unified timestamp across ranks in multi-GPU settings so different ranks do not write to different directories
  * Changes the default `device_map` from `auto` to fixed local-rank mapping
  * Supports Qwen3 implementations for `cluster_e2e / cluster_e2e_big`

* `src/deltakv/save_trainable_trainer.py`

  * Adds `_collect_trainable_state_dict()`
  * Uses `accelerator.unwrap_model()` to retrieve the real model
  * Saves only `requires_grad=True` parameters more robustly, and handles `module.` prefixes

* `src/deltakv/configs/ds_zero2_bf16.json`

  * Adds a DeepSpeed ZeRO-2 BF16 config file

* `src/deltakv/data_prepare/generate_train_data.py`

  * Changes `vr1.0` processing to support custom `dataset_path` / `output_root`
  * Adds a `vi1.0` mixed training data generation pipeline
  * `vi1.0` mixes FineWeb, reasoning chat, and synthetic UUID-KV into a training set
  * Supports loading data from local parquet directories
  * Adds reasoning-sample transforms, chat template rendering, and UUID-KV multi-turn conversation synthesis
  * Adds control knobs such as `fineweb_skip_factor`, `reasoning_ratio`, and `uuid_kv_ratio`
  * Adds recognition of Qwen3 tokenizer paths

These changes move the training side beyond a single-purpose data-processing script toward support for mixed data recipes and Qwen3 training.

#### Analysis and visualization

* `src/deltakv/analysis/verify_insight_pdf.py`

  * Expands significantly into a multi-experiment analysis script
  * Adds parsing for `stride_alpha` sequences
  * Adds `stride_alpha`-based center construction and cluster similarity statistics
  * Distinguishes `history_only` and `runtime_visible` similarity
  * Adds summaries and PDF outputs over multiple alpha values, such as `exp2_cluster_similarity_vs_alpha.pdf`
  * Retains existing experiments such as distance distribution, SVD variance, and norm distribution

* `src/deltakv/analysis/visualize_ablation_loss.py`

  * Unifies cluster chunk parameters under `k_neighbors` semantics, reducing old/new parameter mixing

These changes serve paper/report-oriented analysis rather than runtime behavior.

### 3.2 `src/sparsevllm/`: new standalone/snapkv variants plus stronger scheduling and kernel support

#### Configuration and method registration

* `src/sparsevllm/config.py`

  * Adds `deltakv-standalone` and `deltakv-snapkv` to `vllm_sparse_method`
  * Changes default `chunk_prefill_accel_omnikv` to `False`
  * Changes default `num_top_tokens_in_prefill` to `8192`
  * More robustly parses empty-string `full_attn_layers`
  * Requires `deltakv_path` for all `deltakv*` methods
  * Automatically clears `full_attn_layers` and `obs_layer_ids` for the new standalone/snapkv methods

* `src/sparsevllm/engine/cache_manager/__init__.py`

  * Registers `DeltaKVStandaloneCacheManager` and `DeltaKVSnapKVCacheManager`

* `src/sparsevllm/engine/cache_manager/base.py`

  * Adds factory branches for the two new cache managers
  * Explicitly disables `qwen3 + deltakv` on the sparse-vLLM path
  * Adds a `debug_live_seq_slots()` debugging interface

#### New cache manager: `deltakv-standalone`

`src/sparsevllm/engine/cache_manager/deltakv_standalone.py` is one of the largest new files on the sparse-vLLM side. Its core semantics are:

* All layers use DeltaKV compression
* Mixed full-attention layers are no longer retained
* Each layer maintains a persistent DeltaKV cache
* A global temporary reconstruction pool is shared
* During decode/prefill attention, the currently visible context is first reconstructed into the temp pool, then used for attention computation

It also implements:

* An independent capacity allocation strategy across persistent/full/latent/temp resources
* Per-sequence row management and slot maps
* Per-layer `compress_down / compress_up` after loading compressors
* Admission-related interfaces such as `prompt_admission_costs()`, `prompt_admission_budgets()`, and `reserved_prefill_slots()`
* Debug and statistics utilities such as `free_slot_stats()` and `debug_live_seq_slots()`

It can be understood as the sparse-vLLM version in which full-attention layers are removed and DeltaKV fully takes over cache layout for all layers.

#### New cache manager: `deltakv-snapkv`

`src/sparsevllm/engine/cache_manager/deltakv_snapkv.py` is built on top of standalone, but adds one more step of static pruning:

* Retains a longer protected suffix at the end of prefill
* Applies SnapKV-style static pruning to the compressed middle region
* Uses `row_deltakv_comp_abs_pos` to maintain the mapping from “compressed logical position” to “original absolute position”
* Materializes retained latent tokens back into persistent raw slots
* Reserves more persistent slots for the SnapKV window and keep budget

This gives the sparse-vLLM side its first “DeltaKV compression + SnapKV static retention” hybrid runtime.

#### Strengthening the existing DeltaKV cache manager

`src/sparsevllm/engine/cache_manager/deltakv.py` is not a new file, but it is significantly strengthened:

* It uses explicit config fields more systematically instead of `getattr(..., default)`
* Admission and free-slot statistics interfaces are further improved
* Parameters such as `deltakv_k_neighbors`, `num_top_tokens_in_prefill`, and `full_pool_reserve_ratio` are used more directly
* It is aligned with the new loader/config synchronization logic

This is one of the key stability improvements on the sparse-vLLM side, because cache allocation must happen only after the true latent dimension and compressor structure are known.

#### Scheduler, runner, and attention path

* `src/sparsevllm/engine/model_runner.py`

  * Calls `sync_deltakv_config_from_checkpoint(config)` before creating the cache manager
  * Dynamically computes long-text thresholds based on current batch length
  * Uses different long-text definitions for `deltakv-standalone` / `deltakv-snapkv` versus old DeltaKV
  * Keeps inference under `torch.inference_mode()` to reduce unnecessary autograd graphs

* `src/sparsevllm/engine/scheduler.py`

  * Computes long-text thresholds separately by sparse method
  * Adds budget checks, defer logic, and more detailed deadlock/starvation diagnostics during prompt admission
  * Avoids repeated thrashing between prefill/decode/preempt
  * Distinguishes more clearly between normal idle states and real hung states that should error out

* `src/sparsevllm/engine/llm_engine.py`

  * Handles warmup length separately for standalone/snapkv
  * Adds protection against busy loops and clearer error messages when the scheduler returns no runnable sequences

* `src/sparsevllm/layers/attention.py`

  * Uses `get_layer_store_view()` when writing KV
  * Allows attention computation to read from `get_layer_compute_tensors()`
  * This lets the cache manager decide where data is stored and where it is read from for compute, enabling the temp-pool reconstruction design used by standalone/snapkv

* `src/sparsevllm/engine/sparse_controller.py`

  * Explicitly recognizes `deltakv-standalone` and `deltakv-snapkv`
  * Uses a unified reconstruction path for standalone-like methods
  * Executes finalize logic for `deltakv-snapkv` at the end of chunk prefill
  * Adjusts attention-score collection and read-view logic accordingly

#### Small changes to standard cache managers and helper logic

* `src/sparsevllm/engine/cache_manager/standard.py`

  * Adds release logs under `SPARSEVLLM_DEBUG_SLOTS`
  * Implements `debug_live_seq_slots()`

* `src/sparsevllm/engine/cache_manager/snapkv.py`

* `src/sparsevllm/engine/cache_manager/streamingllm.py`

  * Switch to explicit config fields
  * Simplify logic for `prefill_batched_tokens_margin()` and `remaining_prefill_tokens()`

#### Triton kernel reorganization

* `src/sparsevllm/triton_kernel/deltakv_kernels.py`

  * Adds or merges `deltakv_reconstruct_writeback_grouped_heads_srcdst()`
  * Merges reconstruct writeback and blockwise L2 top-k logic from the former `rekv_kernels.py`
  * Continues strengthening grouped-head kernel variants

* `src/sparsevllm/triton_kernel/rekv_kernels.py`

  * Removed

This suggests a kernel-layer cleanup in which functionality was consolidated from scattered files into the main DeltaKV kernel file.

#### Loader and test support

* `src/sparsevllm/utils/compressor.py`

  * Gives priority to `hf_config.head_dim`, improving compatibility with more model configs

* `src/sparsevllm/utils/loader.py`

  * Adds `sync_deltakv_config_from_checkpoint()`
  * Supports parsing compressor configs from either checkpoint directories or single files
  * Prefers reading `kv_compressed_size`, compressor type, intermediate size, bias, etc. from `config.json`
  * Falls back to inferring from weight shapes if needed
  * Explicitly errors on split-KV checkpoints
  * Reuses unified file parsing logic in `load_deltakv_compressors_to_cache_manager()`

This is one of the key stability upgrades for sparse-vLLM, because cache allocation must wait until the real compressor structure is known.

### 3.3 `benchmark/`: LongBench, SCBench, MathBench, and NIAH were all strengthened

#### LongBench

* `benchmark/long_bench/eval.py`

  * Makes output directories configurable through `DELTAKV_OUTPUT_DIR`
  * Adds `TASK_HIERARCHY`
  * Adds category-level aggregated scores and an overall category average

* `benchmark/long_bench/pred.py`

  * Makes the data root configurable through `DELTAKV_LONGBENCH_DATA_DIR / DELTAKV_DATA_DIR`
  * Validates the data directory and required JSONL files before startup
  * Adds `NO_CHAT_TEMPLATE_DATASETS`
  * Adds `thinking_mode`, including stripping Qwen3 `<think>` outputs
  * Adds KVzip LongBench prompt adaptation, splitting into `prefill_text + query_text`
  * Adds `temperature / top_p / top_k / max_new_tokens_override`
  * For KVzip with `ws > 1`, launches one single-GPU worker per GPU
  * Lets the parent process aggregate logs, run `eval.py`, and write evaluation logs automatically

The most valuable improvement here is that LongBench can now more naturally support regular HF paths, KVzip, Qwen3 thinking templates, and multi-GPU data parallelism at the same time.

#### MathBench

* `benchmark/math_bench/pred.py`

  * Adds KVzip prompt adaptation
  * Uses `apply_chat_template(add_generation_prompt=False/True)` to split prefill/query segments
  * Explicitly disables `aime2024` for KVzip

#### NIAH

* `benchmark/niah/test_niah.py`

  * Extends parameters to cover new DeltaKV config options
  * Makes the output base directory environment-variable driven
  * Adds `num_top_tokens_in_prefill`, `stride_alpha`, `deltakv_use_omnikv_selection`, `chunk_prefill_accel_omnikv`, and `omnikv_score_method`
  * Lets `context_lengths` accept list/tuple forms

#### SCBench CLI and main flow

* `benchmark/scbench/args.py`

  * Adds `load_in_4bit`, `load_in_8bit`, and `model_torch_dtype`
  * Adds `tensor_parallel_size` and `copy_on_gpu`
  * Adds `context_min_tokens`, `context_max_tokens`, and `subset_indices_file`
  * Adds `num_data_shards` and `data_shard_id`
  * Includes `full_deltakv`, `origin_residual_quant`, and `all_origin_residual_quant` as `attn_type` candidates

* `benchmark/scbench/run_scbench.py`

  * Changes `BASE_PATH` to be driven by an output-directory environment variable
  * Unifies result-directory naming through `_build_result_dir()`
  * Adds subset index loading and context-length filtering
  * Adds one-worker-per-GPU data-parallel execution with result merging
  * Model loading now supports `tensor_parallel_size`, `copy_on_gpu`, and quantization parameters
  * Supports `use_chat_template`
  * Uses a unified `real_model_name_tag` for result logging
  * Both single-GPU and multi-GPU paths now produce more standardized `result.json` files and logs

* `benchmark/scbench/eval_utils.py`

  * Adds KV-state snapshot and restore logic to `GreedySearch`
  * In multi-turn / same-context multi-query evaluation, later queries can continue from saved cache state instead of recomputing everything
  * Handles cases where `prepare_inputs_for_generation()` does not return `past_key_values`
  * Checks `hasattr` before calling `clear_temp_kv_cache()`
  * Makes EOS handling more robust

#### New SCBench runners for preprocessed data

* `benchmark/scbench/run_scbench_preprocessed.py`

  * Directly runs DeltaKV/HF paths on `SCBench-preprocessed` parquet data
  * Includes built-in KVzip-style prompt template construction
  * Supports data-parallel workers and result merging
  * Can compute scores directly from prediction files

* `benchmark/scbench/run_kvzip_preprocessed.py`

  * Directly runs KVzip on `SCBench-preprocessed` parquet data
  * Supports `ratio / level / kv_type / prefill_chunk_size`
  * Also supports data-parallel workers, merging, and score reporting

* `benchmark/scbench/scripts/run_scbench_three_eval.sh`

  * Adds a batch script to run three SCBench tasks at once for both KVzip and DeltaKV

Together, these form the first complete version of a “preprocessed-data evaluation pipeline.”

### 3.4 `baselines/kvzip/`: stronger local data support, Qwen compatibility, debugging, and result parsing

Although `baselines/kvzip/` did not add many new files, nearly every change is practical:

* `attention/kvcache.py`

  * Adds debug and VRAM monitoring output
  * Prints more contextual information on cache-update OOM
  * Extends eviction logs with CUDA alloc/reserved/max alloc info

* `csrc/build.py`

  * Dynamically collects CUDA architectures
  * In addition to `sm_80/sm_90`, also compiles for the current machine’s GPU capability

* `data/load.py`

  * Adds `SCBENCH_PREPROCESSED_ROOT`
  * Prefers loading `SCBench-preprocessed` from local parquet files, otherwise falls back to HuggingFace datasets

* `model/monkeypatch.py`

  * Broadens Qwen matching to support `qwen2` and `distill-qwen`

* `model/wrapper.py`

  * Prints detailed debug information when KV score lengths do not match
  * Output trimming no longer blindly drops the last token, but trims against an EOS/PAD/EOT set
  * Adds `@torch.inference_mode()` to `test_scdq()`

* `results/parse.py`

  * Like `data/load.py`, prefers local `SCBench-preprocessed` data

These changes are all essentially aimed at making KVzip easier to include in the unified benchmark framework and easier to debug.

### 3.5 `scripts/`: batch experimentation and rerun scripts now form a coherent toolkit

* `scripts/bench_sparse_vllm.py`

  * Allows `hyper_params` to be passed through JSON
  * Uses a stable set of defaults for benchmarking
  * Breaks down stats for prefill/decode/TTFT/ITL/AvgBS in more detail
  * Adds staged-admission / wave-decode bench logic
  * Saves clearer state when failures occur

* `scripts/queue_scbench_llama_jobs.py`

  * Adds a unified queueing script covering Qwen LongBench alpha sweeps, Qwen SCBench alpha sweeps, Llama DeltaKV and KVzip SCBench, and more
  * Encodes model paths, compressor paths, task combinations, `stride_alpha`, `token_budget`, `ratio`, and similar parameters into a job queue

* `scripts/run_llama31_alpha_sweep_longbench_scbench_b0p17.sh`

  * Adds a batch script to sweep `stride_alpha` while running both LongBench and SCBench

* `scripts/run_llama_deltasnapkv_missing.sh`

* `scripts/run_llama_deltasnapkv_missing_ws2.sh`

* `scripts/run_llama_deltasnapkv_remaining.sh`

  * Used to rerun missing LongBench tasks for Llama DeltaSnapKV, covering different `ws` setups

* `scripts/simulate_linear_stride_ref_tokens.py`

  * A pure simulation script for estimating the number of reference tokens and total retention under different `stride_alpha` values

These scripts ensure that the new methods and parameters introduced by the branch are not merely present in code, but are already packaged into directly executable large-scale experiment entrypoints.

### 3.6 README, ignore rules, tests, and documentation

* `README.md`

  * Changes `flash-attn` installation from a fixed version to an unlocked version
  * Adds baseline usage examples for OmniKV and KIVI

* `.gitignore`

  * Adds `wandb/`, `tmp/`, `outputs/`, and `benchmark/scbench/results/`

* New tests:

  * `tests/test_deltakv_checkpoint_config_sync.py`

    * Tests whether `sync_deltakv_config_from_checkpoint()` can infer compressor config from `config.json` or weight shapes
  * `tests/test_quantization_helpers.py`

    * Tests low-bit loading argument construction, preservation of `chunk_prefill_size`, and dtype restoration for quantization-skipped modules

* Documentation and directory cleanup:

  * `blog/1.md` renamed to `docs/1.md`
  * Added `docs/todo.md`

* Packaging metadata:

  * `src/deltakv.egg-info/PKG-INFO`
  * `src/deltakv.egg-info/SOURCES.txt`
  * Mainly to synchronize package metadata with the new files and package contents

---

## 4. File-by-File List by Directory

The following section lists every changed file in the current branch relative to `main`, organized by directory, to make file-level backtracking easier.

### 4.1 Repository root

| File         | Change description                                                                                          |
| ------------ | ----------------------------------------------------------------------------------------------------------- |
| `.gitignore` | Adds ignore rules for common training/evaluation output directories.                                        |
| `README.md`  | Updates dependency installation instructions and adds usage examples for OmniKV, KIVI, and other baselines. |

### 4.2 `baselines/kvzip/`

| File                                   | Change description                                                                                     |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `baselines/kvzip/attention/kvcache.py` | Adds KVzip cache update/debug/OOM output and more detailed eviction VRAM logs.                         |
| `baselines/kvzip/csrc/build.py`        | Dynamically adds the current machine’s GPU architecture to improve CUDA extension build compatibility. |
| `baselines/kvzip/data/load.py`         | Adds local `SCBench-preprocessed` parquet data loading.                                                |
| `baselines/kvzip/model/monkeypatch.py` | Expands Qwen model matching to support `qwen2` and `distill-qwen`.                                     |
| `baselines/kvzip/model/wrapper.py`     | Strengthens score-length checks, fixes output trimming, and adds inference mode to the SCDQ path.      |
| `baselines/kvzip/results/parse.py`     | Prefers local `SCBench-preprocessed` data during result parsing.                                       |

### 4.3 `benchmark/long_bench/`

| File                           | Change description                                                                                                                                               |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `benchmark/long_bench/eval.py` | Adds environment-variable output directories, category aggregated scores, and overall category average.                                                          |
| `benchmark/long_bench/pred.py` | Adds data-path validation, Qwen3 thinking handling, KVzip prompt adaptation, sampling parameters, single-GPU worker launching, and automatic logging/evaluation. |

### 4.4 `benchmark/math_bench/` and `benchmark/niah/`

| File                           | Change description                                                                                                |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `benchmark/math_bench/pred.py` | Adds KVzip prompt splitting and disables the `aime2024` task.                                                     |
| `benchmark/niah/test_niah.py`  | Extends parameters for new DeltaKV config options and switches to environment-variable-driven output directories. |

### 4.5 `benchmark/scbench/`

| File                                                  | Change description                                                                                                                                               |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `benchmark/scbench/args.py`                           | Expands CLI arguments: quantization, dtype, context filtering, subset files, sharding, TP, copy_on_gpu, etc.                                                     |
| `benchmark/scbench/eval_utils.py`                     | Adds KV-state snapshot/restore for multi-turn evaluation and improves EOS/temp-cache compatibility.                                                              |
| `benchmark/scbench/run_kvzip_preprocessed.py`         | Adds a data-parallel evaluation script for running KVzip directly on `SCBench-preprocessed`.                                                                     |
| `benchmark/scbench/run_scbench.py`                    | Refactors the main SCBench flow to support data-parallel workers, subset filtering, context-length filtering, quantized loading, and unified result directories. |
| `benchmark/scbench/run_scbench_preprocessed.py`       | Adds an evaluation script for running DeltaKV/HF directly on `SCBench-preprocessed`.                                                                             |
| `benchmark/scbench/scripts/run_scbench_three_eval.sh` | Adds a batch script that runs three SCBench tasks for both KVzip and DeltaKV.                                                                                    |

### 4.6 `docs/`

| File           | Change description                                       |
| -------------- | -------------------------------------------------------- |
| `docs/1.md`    | Renamed from `blog/1.md` into `docs/`.                   |
| `docs/todo.md` | Adds follow-up TODOs related to cache manager/rope work. |

### 4.7 `scripts/`

| File                                                         | Change description                                                                             |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `scripts/bench_sparse_vllm.py`                               | Upgrades benchmark logic, supporting JSON hyperparameters and finer-grained staged statistics. |
| `scripts/queue_scbench_llama_jobs.py`                        | Adds a queueing script to organize alpha sweeps, SCBench, KVzip, and related experiments.      |
| `scripts/run_llama31_alpha_sweep_longbench_scbench_b0p17.sh` | Adds a Llama `stride_alpha` joint sweep script for LongBench and SCBench.                      |
| `scripts/run_llama_deltasnapkv_missing.sh`                   | Adds a rerun script for missing DeltaSnapKV LongBench tasks.                                   |
| `scripts/run_llama_deltasnapkv_missing_ws2.sh`               | Adds a two-GPU version of the DeltaSnapKV rerun script.                                        |
| `scripts/run_llama_deltasnapkv_remaining.sh`                 | Adds a rerun script for the remaining DeltaSnapKV tasks.                                       |
| `scripts/simulate_linear_stride_ref_tokens.py`               | Adds a simulation script for reference-token counts under `stride_alpha`.                      |

### 4.8 `src/deltakv/analysis/`

| File                                              | Change description                                                                                     |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `src/deltakv/analysis/verify_insight_pdf.py`      | Expands support for `stride_alpha`, cluster similarity summaries, and additional PDF experiment plots. |
| `src/deltakv/analysis/visualize_ablation_loss.py` | Migrates cluster chunk parameters toward `k_neighbors` semantics.                                      |

### 4.9 `src/deltakv/configs/`

| File                                      | Change description                                                                                            |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `src/deltakv/configs/ds_zero2_bf16.json`  | Adds a DeepSpeed ZeRO-2 BF16 config.                                                                          |
| `src/deltakv/configs/model_config_cls.py` | Adds Qwen3 config, `k_neighbors`, `stride_alpha`, DeltaSnapKV budgets, and more unified config parsing logic. |

### 4.10 Top-level `src/deltakv/` and training/loading helpers

| File                                              | Change description                                                                                                                   |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `src/deltakv/baseline_adapters.py`                | Adds OmniKV and KIVI loading adapters.                                                                                               |
| `src/deltakv/data_prepare/generate_train_data.py` | Expands `vr1.0`, adds `vi1.0` mixed training data generation, and supports local parquet loading.                                    |
| `src/deltakv/get_chat_api.py`                     | Extends the unified HF inference entrypoint to include Qwen3, quantization, OmniKV/KIVI, DeltaSnapKV, and multiple DeltaKV variants. |
| `src/deltakv/quantization.py`                     | Adds helpers for low-bit model loading and dtype restoration for skipped modules.                                                    |
| `src/deltakv/save_trainable_trainer.py`           | Saves only trainable parameters and supports accelerator unwrap.                                                                     |
| `src/deltakv/train_compressor.py`                 | Extends training to Qwen3, DeepSpeed, multi-GPU timestamp sync, and `k_neighbors`.                                                   |

### 4.11 Common `src/deltakv/modeling/` layer

| File                                                      | Change description                                                                          |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `src/deltakv/modeling/all_origin_residual_quant_cache.py` | Adds the all-origin residual quant cache implementation.                                    |
| `src/deltakv/modeling/full_deltakv_compress_cache.py`     | Adds the full DeltaKV compressed-cache implementation.                                      |
| `src/deltakv/modeling/kv_cache.py`                        | Expands dynamic-stride center construction, cluster selection, and new parameter semantics. |
| `src/deltakv/modeling/origin_residual_quant_cache.py`     | Adds the origin residual quant cache implementation.                                        |

### 4.12 `src/deltakv/modeling/llama/`

| File                                                                      | Change description                                                       |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `src/deltakv/modeling/llama/llama_all_origin_residual_quant_inference.py` | Adds Llama all-origin residual quant inference.                          |
| `src/deltakv/modeling/llama/llama_deltasnapkv.py`                         | Adds Llama DeltaSnapKV inference.                                        |
| `src/deltakv/modeling/llama/llama_e2e.py`                                 | Syncs new configs and shared capabilities for Llama end-to-end training. |
| `src/deltakv/modeling/llama/llama_e2e_cluster.py`                         | Migrates cluster training logic to the new `k_neighbors` semantics.      |
| `src/deltakv/modeling/llama/llama_full_deltakv_compress_inference.py`     | Adds Llama full DeltaKV inference.                                       |
| `src/deltakv/modeling/llama/llama_origin_residual_quant_inference.py`     | Adds Llama origin residual quant inference.                              |
| `src/deltakv/modeling/llama/llama_pyramidkv.py`                           | Syncs new config/loading interfaces into Llama PyramidKV.                |
| `src/deltakv/modeling/llama/llama_with_compress_inference.py`             | Adds new DeltaKV selection logic such as `deltakv_use_omnikv_selection`. |

### 4.13 `src/deltakv/modeling/qwen2/`

| File                                                                      | Change description                                                            |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `src/deltakv/modeling/qwen2/qwen2_all_origin_residual_quant_inference.py` | Adds Qwen2 all-origin residual quant inference.                               |
| `src/deltakv/modeling/qwen2/qwen2_deltasnapkv.py`                         | Adds Qwen2 DeltaSnapKV inference.                                             |
| `src/deltakv/modeling/qwen2/qwen2_e2e.py`                                 | Syncs new configs and shared capabilities for Qwen2 end-to-end training.      |
| `src/deltakv/modeling/qwen2/qwen2_e2e_cluster.py`                         | Migrates cluster training logic to the new `k_neighbors` semantics.           |
| `src/deltakv/modeling/qwen2/qwen2_e2e_cluster_for_big_model.py`           | Syncs the new cluster parameter semantics into the large-model training path. |
| `src/deltakv/modeling/qwen2/qwen2_full_deltakv_compress_inference.py`     | Adds Qwen2 full DeltaKV inference.                                            |
| `src/deltakv/modeling/qwen2/qwen2_origin_residual_quant_inference.py`     | Adds Qwen2 origin residual quant inference.                                   |
| `src/deltakv/modeling/qwen2/qwen2_pyramidkv.py`                           | Syncs new config/loading interfaces into Qwen2 PyramidKV.                     |
| `src/deltakv/modeling/qwen2/qwen2_snapkv.py`                              | Syncs new config/loading interfaces into Qwen2 SnapKV.                        |
| `src/deltakv/modeling/qwen2/qwen2_with_compress_inference.py`             | Adds new DeltaKV selection and config logic.                                  |

### 4.14 `src/deltakv/modeling/qwen3/`

| File                                                                      | Change description                              |
| ------------------------------------------------------------------------- | ----------------------------------------------- |
| `src/deltakv/modeling/qwen3/__init__.py`                                  | Adds the Qwen3 modeling package marker file.    |
| `src/deltakv/modeling/qwen3/qwen3_all_origin_residual_quant_inference.py` | Adds Qwen3 all-origin residual quant inference. |
| `src/deltakv/modeling/qwen3/qwen3_e2e.py`                                 | Adds Qwen3 end-to-end DeltaKV training.         |
| `src/deltakv/modeling/qwen3/qwen3_e2e_cluster.py`                         | Adds Qwen3 cluster training.                    |
| `src/deltakv/modeling/qwen3/qwen3_e2e_cluster_for_big_model.py`           | Adds Qwen3 large-model cluster training.        |
| `src/deltakv/modeling/qwen3/qwen3_full_deltakv_compress_inference.py`     | Adds Qwen3 full DeltaKV inference.              |
| `src/deltakv/modeling/qwen3/qwen3_origin_residual_quant_inference.py`     | Adds Qwen3 origin residual quant inference.     |
| `src/deltakv/modeling/qwen3/qwen3_with_compress_inference.py`             | Adds standard Qwen3 DeltaKV inference.          |

### 4.15 `src/sparsevllm/`

| File                                                        | Change description                                                                                |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `src/sparsevllm/config.py`                                  | Adds config entrypoints for `deltakv-standalone` / `deltakv-snapkv` and adjusts several defaults. |
| `src/sparsevllm/engine/cache_manager/__init__.py`           | Registers the new standalone/snapkv cache managers.                                               |
| `src/sparsevllm/engine/cache_manager/base.py`               | Extends the cache manager factory logic and disables qwen3+sparsevllm deltakv.                    |
| `src/sparsevllm/engine/cache_manager/deltakv.py`            | Further strengthens DeltaKV cache manager capacity budgeting and parameter-access logic.          |
| `src/sparsevllm/engine/cache_manager/deltakv_snapkv.py`     | Adds the sparse-vLLM DeltaKV+SnapKV hybrid cache manager.                                         |
| `src/sparsevllm/engine/cache_manager/deltakv_standalone.py` | Adds the sparse-vLLM full-layer DeltaKV standalone cache manager.                                 |
| `src/sparsevllm/engine/cache_manager/snapkv.py`             | Adjusts SnapKV prefill margin and remaining-token calculations.                                   |
| `src/sparsevllm/engine/cache_manager/standard.py`           | Adds slot debug logs and a live-slot debugging interface.                                         |
| `src/sparsevllm/engine/cache_manager/streamingllm.py`       | Adjusts StreamingLLM prefill margin and remaining-token calculations.                             |
| `src/sparsevllm/engine/llm_engine.py`                       | Improves warmup and no-runnable-sequence protection for the new sparse methods.                   |
| `src/sparsevllm/engine/model_runner.py`                     | Syncs DeltaKV checkpoint config before cache allocation and adapts to new long-text semantics.    |
| `src/sparsevllm/engine/scheduler.py`                        | Strengthens admission/defer/preempt logic and adapts to standalone/snapkv long-text thresholds.   |
| `src/sparsevllm/engine/sparse_controller.py`                | Recognizes new DeltaKV variants and manages the matching reconstruct/finalize flow.               |
| `src/sparsevllm/layers/attention.py`                        | Abstracts “store view” and “compute view” to support temp-pool reconstruction.                    |
| `src/sparsevllm/triton_kernel/deltakv_kernels.py`           | Merges more reconstruct/top-k kernels and adds a src-dst writeback variant.                       |
| `src/sparsevllm/triton_kernel/rekv_kernels.py`              | Removed; related kernels were merged into `deltakv_kernels.py`.                                   |
| `src/sparsevllm/utils/compressor.py`                        | Gives priority to explicit `head_dim` from HF config.                                             |
| `src/sparsevllm/utils/loader.py`                            | Adds automatic DeltaKV checkpoint config sync and unified compressor weight parsing.              |

### 4.16 Tests and packaging metadata

| File                                           | Change description                                        |
| ---------------------------------------------- | --------------------------------------------------------- |
| `src/deltakv.egg-info/PKG-INFO`                | Packaging metadata updated.                               |
| `src/deltakv.egg-info/SOURCES.txt`             | Packaging source-file list updated.                       |
| `tests/test_deltakv_checkpoint_config_sync.py` | Adds tests for DeltaKV checkpoint config synchronization. |
| `tests/test_quantization_helpers.py`           | Adds tests for quantization helper functions.             |

---

## 5. What This Branch Ultimately Delivers

If summarized in one sentence, this branch moves the repository from “having a Sparse-vLLM/DeltaKV prototype” to “an engineering state where multiple baselines, multiple DeltaKV variants, two runtime paths, multiple benchmarks, and continued training/analysis/rerun workflows can all be compared and operated stably.”

More concretely, the main additions are:

* More complete method coverage: DeltaKV, full DeltaKV, origin residual quant, all-origin residual quant, DeltaSnapKV, OmniKV, KIVI, KVzip, and others now enter a unified experiment framework.
* More complete model coverage: Qwen3 is formally integrated into the HF training/inference system.
* More complete runtime coverage: sparse-vLLM adds standalone and snapkv hybrid variants.
* More complete experiment coverage: LongBench, SCBench, MathBench, and NIAH all receive the engineering features needed for large-scale experimentation.

So the essence of the `add-baseline` branch is not simply “adding a few baseline files,” but a systematic expansion centered on “baseline comparison and large-scale experimentation.”
