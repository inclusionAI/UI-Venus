#!/bin/bash
set -e


models=("ui_venus_ground_7b")
for model in "${models[@]}"
do
    python eval.py  \
        --model_type ${model}  \
        --images_dir "VenusBench-GD/images"  \
        --anns_dir "VenusBench-GD/instruction"  \
        --model_name_or_path "inclusionAI/UI-Venus-Ground-7B" \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "results/venus_7b_result.json" \
        --inst_style "instruction"
done
