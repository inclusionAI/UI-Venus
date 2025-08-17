#!/bin/bash
set -euo pipefail

export PYTHONPATH=.

model_path='inclusionAI/UI-Venus-Navi-72B'
input_file='examples/trace/trace.json'
output_file='./saved_trace.json'

python models/navigation/runner.py \
    --max_pixels=12845056 \
    --min_pixels=3136 \
    --model_path="${model_path}" \
    --input_file="${input_file}" \
    --output_file="${output_file}"
