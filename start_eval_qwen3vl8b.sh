#!/bin/bash

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate android_world

current_time=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_NORMAL="log_qwen3vl8b_normal_"$current_time".log"
LOG_DARK="log_qwen3vl8b_darkmode_"$current_time".log"
LOG_PAD="log_qwen3vl8b_padmode_"$current_time".log"

MODEL_NAME="qwen3vl"
MODEL="model"
API_KEY="your api key"
BASE_URL="your base url"
TRAJ_OUTPUT_PATH="traj_"$current_time
RESULT_ROOT_PATH=""
GRPC_PORT=8554
CONSOLE_PORT=5554
DEVICE_NAME="device_name"

mkdir -p "${RESULT_ROOT_PATH}/results_qwen3vl8b_normal"
mkdir -p "${RESULT_ROOT_PATH}/results_qwen3vl8b_darkmode"
mkdir -p "${RESULT_ROOT_PATH}/results_qwen3vl8b_padmode"

export PYTHONUNBUFFERED=1


# Normal mode
echo " Normal mode..."s
nohup python run_venusbenchnavi.py \
  --agent_name=$MODEL_NAME \
  --model=$MODEL \
  --api_key=$API_KEY \
  --base_url=$BASE_URL \
  --traj_output_path=$TRAJ_OUTPUT_PATH \
  --grpc_port=$GRPC_PORT \
  --console_port=$CONSOLE_PORT \
  --device_name=$DEVICE_NAME \
  > "$LOG_NORMAL" 2>&1 &

NORMAL_PID=$!
echo "Normal mode PID: $NORMAL_PID: $LOG_NORMAL"


wait $NORMAL_PID
NORMAL_EXIT=$?
echo ""

# Dark mode
echo " Dark mode..."
nohup python run_venusbenchnavi.py \
  --agent_name=$MODEL_NAME \
  --model=$MODEL \
  --api_key=$API_KEY \
  --tasks='stability' \
  --base_url=$BASE_URL \
  --traj_output_path=$TRAJ_OUTPUT_PATH \
  --grpc_port=$GRPC_PORT \
  --console_port=$CONSOLE_PORT \
  --device_name=$DEVICE_NAME \
  --dark_mode='on' \
  > "$LOG_DARK" 2>&1 &

DARK_PID=$!
echo "Dark mode PID: $DARK_PID : $LOG_DARK"





# Pad mode
echo "Pad mode..."
nohup python run_venusbenchnavi.py \
  --agent_name=$MODEL_NAME \
  --model=$MODEL \
  --api_key=$API_KEY \
  --tasks='stability' \
  --base_url=$BASE_URL \
  --traj_output_path=$TRAJ_OUTPUT_PATH \
  --grpc_port=$GRPC_PORT \
  --console_port=$CONSOLE_PORT \
  --device_name=$DEVICE_NAME \
  --pad_mode='on' \
  > "$LOG_PAD" 2>&1 &

PAD_PID=$!
echo "Pad mode PID: $PAD_PID : $LOG_PAD"


echo "=========================================="
echo ""
echo "Normal mode : $LOG_NORMAL"
echo "Dark mode : $LOG_DARK"
echo "Pad mode : $LOG_PAD"
echo "=========================================="
echo ""

python ./utils/stability_statistic.py \
    "${RESULT_ROOT_PATH}/results_qwen3vl8b_normal" \
    "${RESULT_ROOT_PATH}/results_qwen3vl8b_darkmode" \
    "${RESULT_ROOT_PATH}/results_qwen3vl8b_padmode"