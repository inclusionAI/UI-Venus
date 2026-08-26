#!/usr/bin/env bash

# Usage:
# 1. Set MODEL_URL, MODEL_NAME, and API_KEY below or override them with environment variables. MODEL_API_KEY is also accepted.
# 2. By default, infer on the bundled CAPTCHA image and save JSON and HTML under results/captcha/.
# 3. From the repository root, run: bash scripts/captcha.sh

set -euo pipefail

MODEL_URL="${MODEL_URL:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-UI-Venus-2}"
API_KEY="${API_KEY:-${MODEL_API_KEY:-EMPTY}}"
OUTPUT_DIR="${OUTPUT_DIR:-results/captcha}"
IMAGE="${IMAGE:-models/captcha/examples/assets/jiusuoge_5238.png}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/result.json}"
HTML_OUTPUT="${HTML_OUTPUT:-${OUTPUT_DIR}/result.html}"
MAX_TOKENS="${MAX_TOKENS:-40960}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"
VISUALIZE="${VISUALIZE:-true}"

mkdir -p "$(dirname "${OUTPUT_FILE}")" "$(dirname "${HTML_OUTPUT}")"

thinking_arg="--enable-thinking"
if [[ "${ENABLE_THINKING}" != "true" ]]; then
    thinking_arg="--no-enable-thinking"
fi

python models/captcha/infer_captcha.py \
    --base-url "${MODEL_URL}" \
    --api-key "${API_KEY}" \
    --model "${MODEL_NAME}" \
    --image "${IMAGE}" \
    --max-tokens "${MAX_TOKENS}" \
    "${thinking_arg}" \
    --output "${OUTPUT_FILE}"

if [[ "${VISUALIZE}" == "true" ]]; then
    python models/captcha/visualize_captcha.py \
        --result "${OUTPUT_FILE}" \
        --output "${HTML_OUTPUT}"
fi
