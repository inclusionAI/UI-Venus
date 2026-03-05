#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_main_with_ssh_tunnel.sh "你的任务指令" [trace_dir]
#
# Optional env overrides:
#   SSH_HOST_ALIAS=A100-Ubuntu
#   REMOTE_VLLM_PORT=8000
#   LOCAL_VLLM_PORT=8000
#   CONFIG_PATH=config/ui_venus_single.yaml
#   ADB_BIN=/path/to/adb
#   ADB_DEVICE=<device-serial-or-ip:port>
#   VLLM_MODEL=model
#   STEP_LIMIT=30
#   LOG_FILE=logs/main.log
#   SAVE_DIR=record/screenshots/

PURPOSE="${1:-打开铁路12306，查一下明天上海到北京的高铁还有哪些车次}"
TRACE_DIR="${2:-record/traces/$(date +%Y%m%d_%H%M%S)}"
if [[ -z "${PURPOSE}" ]]; then
  echo "Usage: bash run_main_with_ssh_tunnel.sh \"purpose\" [trace_dir]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SSH_HOST_ALIAS="${SSH_HOST_ALIAS:-A100-Ubuntu}"
REMOTE_VLLM_PORT="${REMOTE_VLLM_PORT:-22022}"
LOCAL_VLLM_PORT="${LOCAL_VLLM_PORT:-22022}"
CONFIG_PATH="${CONFIG_PATH:-config/ui_venus_single.yaml}"
ADB_BIN="${ADB_BIN:-/Users/yql-mac/Library/Android/sdk/platform-tools/adb}"
ADB_DEVICE="${ADB_DEVICE:-}"
VLLM_MODEL="${VLLM_MODEL:-/diskpool/models/UI-Venus-1.5-30B-A3B}"
STEP_LIMIT="${STEP_LIMIT:-}"
LOG_FILE="${LOG_FILE:-}"
SAVE_DIR="${SAVE_DIR:-}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_PATH}"
  exit 1
fi

if [[ -n "${ADB_BIN}" && -x "${ADB_BIN}" ]]; then
  export PATH="$(dirname "${ADB_BIN}"):${PATH}"
fi

if ! command -v adb >/dev/null 2>&1; then
  echo "[ERROR] adb not found. Set ADB_BIN or add adb to PATH."
  exit 1
fi

echo "[INFO] SSH host alias: ${SSH_HOST_ALIAS}"
echo "[INFO] Tunnel: 127.0.0.1:${LOCAL_VLLM_PORT} -> ${SSH_HOST_ALIAS}:127.0.0.1:${REMOTE_VLLM_PORT}"
echo "[INFO] Config: ${CONFIG_PATH}"
echo "[INFO] Trace dir: ${TRACE_DIR}"

echo "[INFO] adb version:"
adb version | head -n 1 || true
echo "[INFO] adb devices:"
adb devices || true

if nc -z 127.0.0.1 "${LOCAL_VLLM_PORT}" >/dev/null 2>&1; then
  echo "[INFO] Local tunnel port ${LOCAL_VLLM_PORT} already listening, reuse it."
else
  ssh -fN \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=20 \
    -o ServerAliveCountMax=3 \
    -L "${LOCAL_VLLM_PORT}:127.0.0.1:${REMOTE_VLLM_PORT}" \
    "${SSH_HOST_ALIAS}"
  echo "[INFO] Tunnel established."
fi

TMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/ui_venus_config.XXXXXX.yaml")"
cleanup() {
  rm -f "${TMP_CONFIG}"
}
trap cleanup EXIT

if ! awk -v p="${LOCAL_VLLM_PORT}" '
BEGIN { replaced=0 }
{
  if (!replaced && $0 ~ /^[[:space:]]*model_port:[[:space:]]*[0-9]+([[:space:]]*#.*)?$/) {
    sub(/model_port:[[:space:]]*[0-9]+/, "model_port: " p)
    replaced=1
  }
  print
}
END {
  if (!replaced) {
    exit 42
  }
}
' "${CONFIG_PATH}" > "${TMP_CONFIG}"; then
  cp "${CONFIG_PATH}" "${TMP_CONFIG}"
  echo "[WARN] model_port not found in config, keep original config port."
fi

CMD=(
  python main.py
  --config "${TMP_CONFIG}"
  --purpose "${PURPOSE}"
  --trace-dir "${TRACE_DIR}"
  --model-host "http://127.0.0.1"
  --model-name "${VLLM_MODEL}"
)

if [[ -n "${ADB_DEVICE}" ]]; then
  CMD+=(--device-id "${ADB_DEVICE}")
fi
if [[ -n "${STEP_LIMIT}" ]]; then
  CMD+=(--step-limit "${STEP_LIMIT}")
fi
if [[ -n "${LOG_FILE}" ]]; then
  CMD+=(--log-file "${LOG_FILE}")
fi
if [[ -n "${SAVE_DIR}" ]]; then
  CMD+=(--save-dir "${SAVE_DIR}")
fi

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"
