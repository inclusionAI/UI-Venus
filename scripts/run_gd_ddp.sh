#!/bin/bash
# UI-Venus 1.5 Grounding Evaluation Script - DDP Multi-GPU Parallel Mode
# Suitable for: UI-Venus-1.5 and other models that support DDP
set -e

# ==================== Configuration ====================
# GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Model Configuration
MODEL_PATH="your_model_path"  # Modify to your model path, e.g. /path/to/UI-Venus-1.5
MODEL_TYPE="ui_venus_v15"     # UI-Venus 1.5

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
LOG_FILE="${OUTPUT_DIR}/$(date +%Y%m%d_%H%M%S)_ddp_result.json"

# ==================== Execution ====================
mkdir -p ${OUTPUT_DIR}

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Running DDP evaluation with ${NUM_GPUS} GPUs"
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${TEST_PATH}"
echo "Output: ${LOG_FILE}"

torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} --master-port=39600 \
    models/grounding/eval_screenspot_pro_ddp.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_PATH} \
    --screenspot_imgs ${IMGS_PATH} \
    --screenspot_test ${TEST_PATH} \
    --task ${TASK} \
    --language ${LANGUAGE} \
    --gt_type ${GT_TYPE} \
    --inst_style ${INST_STYLE} \
    --log_path ${LOG_FILE}

echo "Evaluation complete! Results saved to: ${LOG_FILE}"
