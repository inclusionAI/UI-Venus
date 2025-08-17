#!/bin/bash
set -e


models=("ui_venus_ground_72b")
for model in "${models[@]}"
do
    python models/grounding/eval_screenspot_pro.py  \
        --model_type ${model}  \
        --screenspot_imgs "ScreenSpot-v2-variants/screenspotv2_image"  \
        --screenspot_test "ScreenSpot-v2-variants"  \
        --model_name_or_path "inclusionAI/UI-Venus-Ground-72B" \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "venus_72b_ss2.json" \
        --inst_style "instruction"

done



models=("ui_venus_ground_72b") 
for model in "${models[@]}"
do
    python models/grounding/eval_screenspot_pro.py  \
        --model_type ${model}  \
        --screenspot_imgs "Screenspot-pro/images"  \
        --screenspot_test "Screenspot-pro/annotations"  \
        --model_name_or_path "inclusionAI/UI-Venus-Ground-72B" \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "venus_72b_pro.json" \
        --inst_style "instruction"

done


models=("ui_venus_ground_72b") 
for model in "${models[@]}"
do
    python models/grounding/eval_screenspot_pro.py  \
        --model_type ${model}  \
        --screenspot_imgs "data/osworld"  \
        --screenspot_test "data/osworld_meta"  \
        --model_name_or_path "inclusionAI/UI-Venus-Ground-72B" \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "osworld_g_72b.json" \
        --inst_style "instruction"

done

models=("ui_venus_ground_72b") 
for model in "${models[@]}"
do
    python models/grounding/eval_screenspot_pro.py  \
        --model_type ${model}  \
        --screenspot_imgs "data/ui_vision/ui-vision/images"  \
        --screenspot_test "data/ui_vision/ui-vision/annotations/element_grounding"  \
        --model_name_or_path "inclusionAI/UI-Venus-Ground-72B" \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "vison_72b.json" \
        --inst_style "instruction"

done


models=("ui_venus_ground_72b") 
for model in "${models[@]}"
do
    python models/grounding/eval_screenspot_pro.py  \
        --model_type ${model}  \
        --screenspot_imgs "CAGUI/CAGUI_grounding/images/"  \
        --screenspot_test "CAGUI/CAGUI_grounding/json_files/"  \
        --model_name_or_path "inclusionAI/UI-Venus-Ground-72B" \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "cpm_72b.json" \
        --inst_style "instruction"

done
