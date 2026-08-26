#!/usr/bin/env bash

# Usage:
# 1. Follow models/browser/README.md to start Chrome with a CDP port.
# 2. Set MODEL_URL, MODEL_NAME, and API_KEY below or override them with environment variables. MODEL_API_KEY is also accepted.
# 3. From the repository root, run: bash scripts/browser.sh "Open https://example.com and report the page title"

set -euo pipefail

MODEL_URL="${MODEL_URL:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-UI-Venus-2}"
API_KEY="${API_KEY:-${MODEL_API_KEY:-EMPTY}}"
OUTPUT_DIR="${OUTPUT_DIR:-results/browser}"

CDP_URL="${CDP_URL:-http://127.0.0.1:9222}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-4096}"
LLM_THINKING="${LLM_THINKING:-false}"
MAX_STEPS="${MAX_STEPS:-30}"

if (( $# > 0 )); then
    TASK="$*"
else
    TASK="${TASK:-Open https://example.com and report the page title}"
fi

export CDP_URL
export LLM_API_URL="${MODEL_URL}"
export LLM_API_KEY="${API_KEY}"
export LLM_MODEL="${MODEL_NAME}"
export LLM_MAX_TOKENS LLM_THINKING

python models/browser/venus_browser.py \
    --max-steps "${MAX_STEPS}" \
    --output "${OUTPUT_DIR}" \
    "${TASK}"
