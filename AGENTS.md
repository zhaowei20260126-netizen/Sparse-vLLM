# Sparse-vLLM Agent Notes
## Repo Skills

This repository includes a repo-local Codex skill.

## Available skills

- `add-sparse-method`: Add or refactor a first-class Sparse-vLLM sparse method following this repo's architecture. Use when Codex needs to introduce a new `vllm_sparse_method`, move method logic out of `attention.py` or `utils/`, add method-specific cache metadata or decode-time view building, and preserve the cache-manager-first design. File: `skills/add-sparse-method/SKILL.md`

## How to use

- In this repo, invoke the skill as `$add-sparse-method`.
- Keep method-specific runtime state in `src/sparsevllm/engine/cache_manager/`.
- Keep `src/sparsevllm/layers/attention.py` generic and hook new methods through shared cache-manager interfaces when possible.


## Repo shape (what matters for edits)
- Two first-class codepaths live in one package: `src/sparsevllm/` (inference engine) and `src/deltakv/` (HF-style compressor/train/eval tooling).
- Main runtime entrypoint is `sparsevllm.LLM` (`src/sparsevllm/llm.py` -> `engine/llm_engine.py`).
- Sparse method dispatch is centralized in `CacheManager.create(...)` (`src/sparsevllm/engine/cache_manager/base.py`); add new methods there, not in random model files.
- `benchmark/long_bench/pred.py` and `benchmark/math_bench/pred.py` both call `deltakv.get_chat_api.get_generate_api(...)`; backend behavior is wired there.

## Setup and dependencies
- Follow `README.md` install order: install PyTorch/Transformers stack first, then `flash-attn`, then `pip install -e .`.
- No root `Makefile`, `pytest.ini`, `tox.ini`, or pre-commit config exists; do not assume standard `make test` / `pytest` workflows.
- Optional CUDA extension under `src/sparsevllm/cuda_kernel/` is only needed for `deltakv-triton-v3-cuda-offload` (`pip install -e src/sparsevllm/cuda_kernel`).

## High-signal run commands
- Throughput benchmark: `python scripts/bench_sparse_vllm.py --model_path <model_dir> --methods <method> --lengths <n> --batch_sizes <n> --hyper_params '{...}'`.
- LongBench prediction+eval: `python benchmark/long_bench/pred.py ...` (this script auto-runs `benchmark/long_bench/eval.py` at the end).
- MathBench prediction+eval: `python benchmark/math_bench/pred.py ...` (this script auto-runs `benchmark/math_bench/eval.py` at the end).
- DeltaKV training CLI: `deltakv-train ...` (mapped from `src/deltakv/train_compressor.py`).

## Validation shortcuts
- Prefer `python -m py_compile` on touched Python files before broader runs.
- For focused behavior checks, use the smallest benchmark or unit test that exercises the touched path; do not assume a root-level test runner exists.
- When editing sparse-method runtime code, validate the engine path and then one benchmark path; when editing benchmark glue, validate the exact script you touched.

## Benchmark/data env quirks
- `scripts/bench_sparse_vllm.py` treats `--lengths` as prompt length and internally sets `max_model_len = length + output_len + 100`.
- LongBench data root must contain `data/*.jsonl`; configure with `DELTAKV_LONGBENCH_DATA_DIR` (or fallback `DELTAKV_DATA_DIR`).
- Benchmark outputs/logs default to `DELTAKV_OUTPUT_DIR` (fallback `/root/autodl-fs/deltakv_outputs`).
- `benchmark/long_bench/pred.py` and `benchmark/math_bench/pred.py` both route generation through `deltakv.get_chat_api.get_generate_api(...)`; backend selection and method knobs live in `--backend` and `--hyper_param`.
- `benchmark/math_bench/pred.py` hard-enforces `--temperature` in `[0.5, 0.7]`.
- LongBench auto-runs its eval step after prediction; MathBench does the same.
- `benchmark/math_bench/pred.py` uses `ENABLE_THINKING` and `DEBUG` environment variables for chat-template and prompt-debug behavior.

## Architecture guardrails for sparse-method work
- Use repo skill `$add-sparse-method` before implementing/refactoring a method (`skills/add-sparse-method/SKILL.md`).
- Keep method-specific persistent state in `src/sparsevllm/engine/cache_manager/`.
- Keep `src/sparsevllm/layers/attention.py` method-agnostic; prefer cache-manager hooks like `build_decode_view(...)` / `on_kv_stored(...)`.
- Put cross-layer orchestration in `src/sparsevllm/engine/sparse_controller.py`, not in ad-hoc utils.

## Reference docs to link, not duplicate
- `docs/project-architecture-map_zh.md` for module boundaries and file map.
- `docs/hf-vs-sparsevllm-parameter-guide.md` for backend and parameter differences.
- `docs/add-baseline-vs-main.md` when comparing baseline wrappers against the engine path.
- `skills/add-sparse-method/SKILL.md` for any first-class sparse-method work.

## Verified runtime constraints (prefer code over README prose)
- `SamplingParams.temperature` allows `0.0` (`src/sparsevllm/sampling_params.py`), so greedy decode is valid.
- `sparsevllm` currently disables `qwen3 + deltakv` and DeepSeek-V3.2 paths in cache manager routing (`src/sparsevllm/engine/cache_manager/base.py`).

## 交互要求
- Thinking思考过程用中文表述
- Reply回答也要用中文回复