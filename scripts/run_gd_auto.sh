#!/bin/bash
# UI-Venus 1.5 Grounding Evaluation Script - Auto Mode (device_map="auto")
# Suitable for: UI-Venus-1.5-Pro and other large models
set -e

# ==================== Configuration ====================
# GPU Configuration (auto mode will distribute automatically)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Model Configuration
MODEL_PATH="your_model_path"  # Modify to your model path, e.g. /path/to/UI-Venus-1.5-Pro

# Dataset Configuration (uncomment to select dataset)
# ScreenSpot-Pro
IMGS_PATH="/path/to/Screenspot-pro/images"
TEST_PATH="/path/to/Screenspot-pro/annotations"

# ScreenSpot-v2
# IMGS_PATH="/path/to/screenspotv2_image/"
# TEST_PATH="/path/to/screenspotv2_json/"

# MMBench-GUI
# IMGS_PATH="/path/to/MMBench-GUI-OfflineImages/"
# TEST_PATH="/path/to/MMBench-GUI-OfflineImages/json/"

# VenusBench-GD
# IMGS_PATH="/path/to/VenusBench_GD/images/"
# TEST_PATH="/path/to/VenusBench_GD/basic_instruction/"

# Evaluation Configuration
TASK="all"
LANGUAGE="en"
GT_TYPE="positive"
INST_STYLE="instruction"

# Output Configuration
OUTPUT_DIR="./results/grounding"
LOG_FILE="${OUTPUT_DIR}/$(date +%Y%m%d_%H%M%S)_auto_result.json"

# ==================== Execution ====================
mkdir -p ${OUTPUT_DIR}

echo "Running evaluation with Auto mode (device_map=auto)"
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${TEST_PATH}"
echo "Output: ${LOG_FILE}"

python models/grounding/eval_screenspot_pro.py \
    --model_name_or_path ${MODEL_PATH} \
    --screenspot_imgs ${IMGS_PATH} \
    --screenspot_test ${TEST_PATH} \
    --task ${TASK} \
    --language ${LANGUAGE} \
    --gt_type ${GT_TYPE} \
    --inst_style ${INST_STYLE} \
    --log_path ${LOG_FILE} \
    --device auto

echo "Evaluation complete! Results saved to: ${LOG_FILE}"
