#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a local checkpoint directory or Hugging Face model ID}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_NAME_OR_PATH}}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-50000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
DTYPE="${DTYPE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
SERVER_ARGS=()

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  SERVER_ARGS+=(--trust-remote-code)
fi
if [[ -n "${API_KEY:-}" ]]; then
  SERVER_ARGS+=(--api-key "${API_KEY}")
fi

exec vllm serve "${MODEL_NAME_OR_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TENSOR_PARALLEL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --dtype "${DTYPE}" \
  "${SERVER_ARGS[@]}" \
  "$@"
