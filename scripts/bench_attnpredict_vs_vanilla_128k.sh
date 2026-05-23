#!/usr/bin/env bash
set -euo pipefail

# Benchmark vanilla full attention vs AttentionPredictor on one long context.
#
# Required:
#   MODEL_PATH=/path/to/base/model
#   ATTNPREDICT_MODEL_PATH=/path/to/CNN/best_model.pth
#
# Optional overrides:
#   LENGTHS=128000
#   BATCH_SIZES=1
#   OUTPUT_LEN=64
#   GPU_MEMORY_UTILIZATION=0.9
#   CHUNK_PREFILL_SIZE=4096
#   TENSOR_PARALLEL_SIZE=1
#   NUM_TOP_TOKENS=4096
#   NUM_SINK_TOKENS=64
#   NUM_RECENT_TOKENS=512
#   ATTNPREDICT_HISTORY_STEPS=64
#   ATTNPREDICT_POOLING_BLOCK_SIZE=16
#   SPARSEVLLM_MASTER_PORT=2345
#   EXTRA_HYPER_PARAMS_JSON='{"max_num_batched_tokens":8192}'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${MODEL_PATH:?Set MODEL_PATH=/path/to/base/model}"
: "${ATTNPREDICT_MODEL_PATH:?Set ATTNPREDICT_MODEL_PATH=/path/to/CNN/best_model.pth}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH does not exist or is not a directory: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${ATTNPREDICT_MODEL_PATH}" ]]; then
  echo "ATTNPREDICT_MODEL_PATH does not exist or is not a file: ${ATTNPREDICT_MODEL_PATH}" >&2
  exit 1
fi

export SPARSEVLLM_MASTER_PORT="${SPARSEVLLM_MASTER_PORT:-2345}"

LENGTHS="${LENGTHS:-128000}"
BATCH_SIZES="${BATCH_SIZES:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
CHUNK_PREFILL_SIZE="${CHUNK_PREFILL_SIZE:-4096}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
NUM_TOP_TOKENS="${NUM_TOP_TOKENS:-4096}"
NUM_SINK_TOKENS="${NUM_SINK_TOKENS:-64}"
NUM_RECENT_TOKENS="${NUM_RECENT_TOKENS:-512}"
ATTNPREDICT_HISTORY_STEPS="${ATTNPREDICT_HISTORY_STEPS:-64}"
ATTNPREDICT_POOLING_BLOCK_SIZE="${ATTNPREDICT_POOLING_BLOCK_SIZE:-16}"
EXTRA_HYPER_PARAMS_JSON="${EXTRA_HYPER_PARAMS_JSON:-{}}"
export ATTNPREDICT_MODEL_PATH
export GPU_MEMORY_UTILIZATION
export CHUNK_PREFILL_SIZE
export TENSOR_PARALLEL_SIZE
export NUM_TOP_TOKENS
export NUM_SINK_TOKENS
export NUM_RECENT_TOKENS
export ATTNPREDICT_HISTORY_STEPS
export ATTNPREDICT_POOLING_BLOCK_SIZE
export EXTRA_HYPER_PARAMS_JSON

HYPER_PARAMS="$(
  python - <<'PY'
import json
import os

params = {
    "gpu_memory_utilization": float(os.environ["GPU_MEMORY_UTILIZATION"]),
    "chunk_prefill_size": int(os.environ["CHUNK_PREFILL_SIZE"]),
    "tensor_parallel_size": int(os.environ["TENSOR_PARALLEL_SIZE"]),
    "attnpredict_model_path": os.environ["ATTNPREDICT_MODEL_PATH"],
    "attnpredict_history_steps": int(os.environ["ATTNPREDICT_HISTORY_STEPS"]),
    "attnpredict_pooling_block_size": int(os.environ["ATTNPREDICT_POOLING_BLOCK_SIZE"]),
    "num_top_tokens": int(os.environ["NUM_TOP_TOKENS"]),
    "num_sink_tokens": int(os.environ["NUM_SINK_TOKENS"]),
    "num_recent_tokens": int(os.environ["NUM_RECENT_TOKENS"]),
}
extra = json.loads(os.environ["EXTRA_HYPER_PARAMS_JSON"])
if not isinstance(extra, dict):
    raise SystemExit("EXTRA_HYPER_PARAMS_JSON must be a JSON object")
params.update(extra)
print(json.dumps(params, separators=(",", ":")))
PY
)"

echo "Benchmarking vanilla vs attnpredict"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  ATTNPREDICT_MODEL_PATH=${ATTNPREDICT_MODEL_PATH}"
echo "  LENGTHS=${LENGTHS}"
echo "  BATCH_SIZES=${BATCH_SIZES}"
echo "  OUTPUT_LEN=${OUTPUT_LEN}"
echo "  SPARSEVLLM_MASTER_PORT=${SPARSEVLLM_MASTER_PORT}"
echo "  HYPER_PARAMS=${HYPER_PARAMS}"

python scripts/bench_sparse_vllm.py \
  --model_path "${MODEL_PATH}" \
  --methods vanilla,attnpredict \
  --lengths "${LENGTHS}" \
  --batch_sizes "${BATCH_SIZES}" \
  --output_len "${OUTPUT_LEN}" \
  --temperature 0.0 \
  --hyper_params "${HYPER_PARAMS}"
