#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to the name exposed by the inference server}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
ANNOTATIONS="${ANNOTATIONS:-${REPOSITORY_ROOT}/instruction/VenusBench-CAPTCHA.json}"
PREDICTIONS="${PREDICTIONS:-${REPOSITORY_ROOT}/results/predictions.jsonl}"
CONCURRENCY="${CONCURRENCY:-32}"
MAX_TOKENS="${MAX_TOKENS:-25600}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0}"
SEED="${SEED:-996}"
MODE_ARGS=(--resume)
for argument in "$@"; do
  if [[ "${argument}" == "--resume" || "${argument}" == "--overwrite" ]]; then
    MODE_ARGS=()
    break
  fi
done

python "${REPOSITORY_ROOT}/eval.py" run \
  --backend openai-compatible \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --annotations "${ANNOTATIONS}" \
  --predictions "${PREDICTIONS}" \
  --concurrency "${CONCURRENCY}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --presence-penalty "${PRESENCE_PENALTY}" \
  --seed "${SEED}" \
  "${MODE_ARGS[@]}" \
  "$@"
