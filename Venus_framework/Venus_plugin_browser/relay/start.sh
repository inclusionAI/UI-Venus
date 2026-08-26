#!/usr/bin/env bash

# Usage: VENUS_UPSTREAM_BASE="https://example.com/v1" ./start.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${VENUS_UPSTREAM_BASE:-}" ]]; then
  echo "请先设置 VENUS_UPSTREAM_BASE，例如 https://example.com/v1。" >&2
  exit 1
fi

if [[ -z "${VENUS_EXTENSION_ID:-}" ]]; then
  read -r -p "请输入 chrome://extensions 中显示的 Venus 插件 ID: " VENUS_EXTENSION_ID
fi

if [[ ! "$VENUS_EXTENSION_ID" =~ ^[a-p]{32}$ ]]; then
  echo "插件 ID 应为 32 位、由 a-p 组成的字符串。" >&2
  exit 1
fi

echo "正在启动 http://127.0.0.1:8765/v1 ……"
exec python3 "$SCRIPT_DIR/venus_relay.py" \
  --host 127.0.0.1 \
  --port 8765 \
  --allow-origin "chrome-extension://$VENUS_EXTENSION_ID" \
  --upstream-base "$VENUS_UPSTREAM_BASE"
