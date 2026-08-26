#!/usr/bin/env bash

# Usage:
# 1. Set MODEL_URL, MODEL_NAME, and API_KEY below or override them with environment variables. MODEL_API_KEY is also accepted.
# 2. MODE=single evaluates three bundled samples, MODE=smoke checks the API, and MODE=multi runs multiple benchmarks.
# 3. From the repository root, run: bash scripts/grounding.sh

set -euo pipefail

MODE="${MODE:-single}"
MODEL_URL="${MODEL_URL:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-UI-Venus-2}"
API_KEY="${API_KEY:-${MODEL_API_KEY:-EMPTY}}"
OUTPUT_DIR="${OUTPUT_DIR:-results/grounding}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/checkpoints"

case "${MODE}" in
    single)
        python models/grounding/eval_single_benchmark.py \
            --base_url "${MODEL_URL}" \
            --api_key "${API_KEY}" \
            --model_name "${MODEL_NAME}" \
            --task test_samples \
            --language en \
            --gt_type positive \
            --inst_style instruction \
            --num_processes "${NUM_PROCESSES}" \
            --num_workers "${NUM_WORKERS}" \
            --norm_type 0-1000 \
            --checkpoint_interval 20 \
            --log_path "${OUTPUT_DIR}/test_samples.json" \
            --checkpoint_path "${OUTPUT_DIR}/checkpoints/test_samples.json"
        ;;
    smoke)
        export GROUNDING_BASE_URL="${MODEL_URL}"
        export GROUNDING_API_KEY="${API_KEY}"
        export GROUNDING_MODEL="${MODEL_NAME}"
        python models/grounding/test_grounding_api.py
        ;;
    multi)
        python models/grounding/eval_multi_benchmark.py \
            --base_url "${MODEL_URL}" \
            --api_key "${API_KEY}" \
            --model_name "${MODEL_NAME}" \
            --benchmarks all \
            --num_processes "${NUM_PROCESSES}" \
            --num_workers "${NUM_WORKERS}" \
            --log_dir "${OUTPUT_DIR}/multi" \
            --checkpoint_dir "${OUTPUT_DIR}/checkpoints/multi"
        ;;
    *)
        echo "Unsupported MODE: ${MODE}; expected single, smoke, or multi" >&2
        exit 2
        ;;
esac
