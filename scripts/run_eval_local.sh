#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a local checkpoint directory or Hugging Face model ID}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ANNOTATIONS="${ANNOTATIONS:-${REPOSITORY_ROOT}/instruction/VenusBench-CAPTCHA.json}"
PREDICTIONS="${PREDICTIONS:-${REPOSITORY_ROOT}/results/predictions-local.jsonl}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-50000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
DTYPE="${DTYPE:-auto}"
MODE_ARGS=(--resume)
for argument in "$@"; do
  if [[ "${argument}" == "--resume" || "${argument}" == "--overwrite" ]]; then
    MODE_ARGS=()
    break
  fi
done

python "${REPOSITORY_ROOT}/eval.py" run \
  --backend vllm \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --annotations "${ANNOTATIONS}" \
  --predictions "${PREDICTIONS}" \
  --tensor-parallel "${TENSOR_PARALLEL}" \
  --batch-size "${BATCH_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --dtype "${DTYPE}" \
  --concurrency 1 \
  "${MODE_ARGS[@]}" \
  "$@"
