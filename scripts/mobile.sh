#!/usr/bin/env bash

# Usage:
# 1. Set MODEL_URL, MODEL_NAME, and API_KEY below or override them with environment variables. MODEL_API_KEY is also accepted.
# 2. INPUT_FILE defaults to models/mobile/examples/example_input.json; output defaults to results/mobile/output.json.
# 3. N_IMG=0 keeps only assistant text in history; N_IMG>0 also includes the N most recent historical screenshots.
# 4. From the repository root, run: bash scripts/mobile.sh

set -euo pipefail

MODEL_URL="${MODEL_URL:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-UI-Venus-2}"
API_KEY="${API_KEY:-${MODEL_API_KEY:-EMPTY}}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
N_IMG="${N_IMG:-2}"

INPUT_FILE="${INPUT_FILE:-models/mobile/examples/example_input.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/mobile}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/output.json}"

python models/mobile/mobile_example.py \
    --model-url "${MODEL_URL}" \
    --model-name "${MODEL_NAME}" \
    --api-key "${API_KEY}" \
    --temperature "${TEMPERATURE}" \
    --max-tokens "${MAX_TOKENS}" \
    --n-img "${N_IMG}" \
    --input-file "${INPUT_FILE}" \
    --output-file "${OUTPUT_FILE}"
