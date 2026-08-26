#!/usr/bin/env bash

# Usage:
# 1. Set MODEL_URL, MODEL_NAME, and API_KEY below or override them with environment variables. MODEL_API_KEY is also accepted.
# 2. INPUT_FILE defaults to models/computer/examples/example_input.json; it may also be the first argument.
# 3. From the repository root, run: bash scripts/computer.sh

set -euo pipefail

MODEL_URL="${MODEL_URL:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-UI-Venus-2}"
API_KEY="${API_KEY:-${MODEL_API_KEY:-EMPTY}}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.7}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
N_IMG="${N_IMG:-2}"
SUDO_PASSWORD="${SUDO_PASSWORD:-password}"
PARSE_RETRIES="${PARSE_RETRIES:-1}"
TIMEOUT="${TIMEOUT:-120}"

INPUT_FILE="${INPUT_FILE:-${1:-models/computer/examples/example_input.json}}"
OUTPUT_DIR="${OUTPUT_DIR:-results/computer}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/output.json}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"

extra_args=()
if [[ "${ENABLE_THINKING}" == "true" ]]; then
    extra_args+=(--enable-thinking)
fi

python models/computer/computer_example.py \
    --model-url "${MODEL_URL}" \
    --model-name "${MODEL_NAME}" \
    --api-key "${API_KEY}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --max-tokens "${MAX_TOKENS}" \
    --n-img "${N_IMG}" \
    --sudo-password "${SUDO_PASSWORD}" \
    --parse-retries "${PARSE_RETRIES}" \
    --timeout "${TIMEOUT}" \
    --input-file "${INPUT_FILE}" \
    --output-file "${OUTPUT_FILE}" \
    "${extra_args[@]}"
