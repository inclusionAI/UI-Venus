#!/bin/bash

# This script runs all tasks registered in the VENUS_FAMILY with Gemini 3 agent

export GEMINI_PROXY_API_KEY="your_api_key_here"

current_time=$(date +"%Y-%m-%d_%H-%M-%S")

RESULT_ROOT_PATH=""
# Agent
AGENT_NAME="QwenGD_V"
TRAJ_OUTPUT_PATH="traj_"$current_time

# Emulator
GRPC_PORT=8554
CONSOLE_PORT=5554
DEVICE_NAME="name"
# Log Path
LOG_NORMAL="log_${AGENT_NAME}_normal_"$current_time".log"
LOG_DARK="log_${AGENT_NAME}_darkmode_"$current_time".log"
LOG_PAD="log_${AGENT_NAME}_padmode_"$current_time".log"


# Result Path
mkdir -p "${RESULT_ROOT_PATH}/results_${AGENT_NAME}_normal"
mkdir -p "${RESULT_ROOT_PATH}/results_${AGENT_NAME}_darkmode"
mkdir -p "${RESULT_ROOT_PATH}/results_${AGENT_NAME}_padmode"
NORMAL_RESULT_PATH="${RESULT_ROOT_PATH}/results_${AGENT_NAME}_normal"
DARK_RESULT_PATH="${RESULT_ROOT_PATH}/results_${AGENT_NAME}_darkmode"
PAD_RESULT_PATH="${RESULT_ROOT_PATH}/results_${AGENT_NAME}_padmode"

export PYTHONUNBUFFERED=1

# Normal mode
echo " Normal mode..."
nohup python -u run_venusbenchnavi.py \
    --agent_name=$AGENT_NAME  \
    --tv_or_vt="tv" \
    --traj_output_path=$TRAJ_OUTPUT_PATH \
    --suite_family="venus" \
    --grpc_port=$GRPC_PORT \
    --console_port=$CONSOLE_PORT \
    --device_name=$DEVICE_NAME \
    > "$LOG_NORMAL" 2>&1 &

NORMAL_PID=$!
echo "Normal mode PID: $NORMAL_PID : $LOG_NORMAL"

wait $NORMAL_PID
echo "Normal mode "
echo ""

# Dark mode
echo " Dark mode..."
nohup python run_venusbenchnavi.py \
    --agent_name=$AGENT_NAME  \
    --tv_or_vt="tv" \
    --traj_output_path=$TRAJ_OUTPUT_PATH \
    --suite_family="venus" \
    --grpc_port=$GRPC_PORT \
    --console_port=$CONSOLE_PORT \
    --device_name=$DEVICE_NAME \
    --tasks='stability' \
    --dark_mode='on' \
    > "$LOG_DARK" 2>&1 &

DARK_PID=$!
echo "Dark mode PID: $DARK_PID : $LOG_DARK"

wait $DARK_PID
echo "Dark mode "
echo ""

bash ./restart_emulator.sh "$DEVICE_NAME" "$GRPC_PORT" "$CONSOLE_PORT"
sleep 10
# ------------------


# Pad Emulator
GRPC_PORT=8572
CONSOLE_PORT=5572
DEVICE_NAME="venus_emulator1"

# Pad mode
echo " Pad mode..."
nohup python run_venusbenchnavi.py \
    --agent_name=$AGENT_NAME  \
    --tv_or_vt="tv" \
    --traj_output_path=$TRAJ_OUTPUT_PATH \
    --suite_family="venus" \
    --grpc_port=$GRPC_PORT \
    --console_port=$CONSOLE_PORT \
    --device_name=$DEVICE_NAME \
    --tasks='stability' \
    --pad_mode='on' \
    > "$LOG_PAD" 2>&1 &


PAD_PID=$!
echo "Pad mode PID: $PAD_PID : $LOG_PAD"

wait $PAD_PID
echo "Pad mode "
echo ""



echo "=========================================="
echo "Normal mode : $LOG_NORMAL"
echo "Dark mode : $LOG_DARK"
echo "Pad mode : $LOG_PAD"
echo "=========================================="
echo ""

python ./utils/stability_statistic.py \
    "${NORMAL_RESULT_PATH}" \
    "${DARK_RESULT_PATH}" \
    "${PAD_RESULT_PATH}"

