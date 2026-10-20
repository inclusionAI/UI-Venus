#!/usr/bin/env bash
# Start, stop, or inspect a local OpenAI-compatible vLLM endpoint.
# Usage: serve_vllm.sh start MODEL [OPTIONS] | stop [--name NAME] | status [--name NAME]
# Defaults: --tp 2 --dp 8 (16 devices), --port 8000, --max-model-len 65536,
# --max-num-seqs 64 per replica, --health-timeout 1800 seconds.
# Set --tp and --dp explicitly for the available hardware.
# --name selects the served model ID; --api-key enables authentication.
# --extra passes additional vLLM arguments; --daemon enables background serving.
# --env-file selects the generated environment file; VLLM_BIN selects the executable.
# Logs and PID files are stored in logs/. Use --name to target one instance.
# LBS checkpoint: Qwen3.8-27b after SFT and Agentic RL; path below is illustrative.
#   scripts/serve_vllm.sh start /models/lbs-qwen3.8-27b --name evaluated-model --tp 4 --dp 1 --port 8001 --daemon
#   scripts/serve_vllm.sh status --name evaluated-model
#   scripts/serve_vllm.sh stop --name evaluated-model
#   scripts/run_sspro.sh .env.vllm.<name> results/smoke_sspro --max_samples 10
set -euo pipefail
KIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$KIT_ROOT/logs"
mkdir -p "$LOG_DIR"

VLLM_BIN="${VLLM_BIN:-vllm}"

info(){ echo "[vllm-serve] $*"; }
die(){ echo "[vllm-serve] ERROR: $*" >&2; exit 1; }

usage_die(){ cat >&2 <<'EOF'
Usage:
  scripts/serve_vllm.sh start <model_path> [options]
  scripts/serve_vllm.sh stop  [--name NAME]
  scripts/serve_vllm.sh status [--name NAME]
start options: --name/--port/--tp/--dp/--max-model-len/--max-num-seqs/--api-key/
            --extra "ARGS"/--daemon/--health-timeout/--env-file
See the script header or evaluation/README.md.
EOF
  exit 1; }

health_check(){
  local p="$1" k="${2:-}"
  if [ -n "$k" ]; then
    curl -sf -o /dev/null -H "Authorization: Bearer $k" "http://127.0.0.1:$p/v1/models"
  else
    curl -sf -o /dev/null "http://127.0.0.1:$p/v1/models"
  fi
}

device_count(){
  local n
  n="$(nvidia-smi -L 2>/dev/null | wc -l)"
  [ "$n" -ge 1 ] || n=1
  echo "$n"
}

pid_file_of(){ echo "$LOG_DIR/vllm_$1.pid"; }

do_stop(){
  local want="${1:-}" stopped=0 pidf pid i
  for pidf in "$LOG_DIR"/vllm_*.pid; do
    [ -f "$pidf" ] || continue
    local name; name="$(basename "$pidf" .pid)"; name="${name#vllm_}"
    if [ -n "$want" ] && [ "$name" != "$want" ]; then continue; fi
    pid="$(sed -n 1p "$pidf")"
    if kill -0 "$pid" 2>/dev/null; then
      info "Stopping $name (pid $pid) ..."
      kill -TERM "$pid" 2>/dev/null || true
      for i in $(seq 1 60); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
      if kill -0 "$pid" 2>/dev/null; then
        info "$name did not exit after TERM; sending KILL"
        kill -KILL "$pid" 2>/dev/null || true
      fi
    else
      info "$name: pid $pid is absent; removing its PID file"
    fi
    rm -f "$pidf"
    stopped=$((stopped + 1))
  done
  if [ "$stopped" -eq 0 ]; then
    info "No registered instance${want:+ (name=$want)}; inspect any remaining worker processes manually"
  fi
}

do_status(){
  local want="${1:-}" any=0 pidf pid port key name
  for pidf in "$LOG_DIR"/vllm_*.pid; do
    [ -f "$pidf" ] || continue
    name="$(basename "$pidf" .pid)"; name="${name#vllm_}"
    if [ -n "$want" ] && [ "$name" != "$want" ]; then continue; fi
    any=1
    pid="$(sed -n 1p "$pidf")"; port="$(sed -n 2p "$pidf")"; key="$(sed -n 3p "$pidf")"
    if kill -0 "$pid" 2>/dev/null; then
      if health_check "$port" "$key"; then
        info "$name: pid=$pid port=$port ready (/v1/models 200); env=<kit>/.env.vllm.$name"
      else
        info "$name: pid=$pid port=$port starting/not ready → tail -f $LOG_DIR/vllm_$name.log"
      fi
    else
      info "$name: pid=$pid exited (use stop to remove the stale PID file)"
    fi
  done
  if [ "$any" -eq 0 ]; then
    info "No registered instances (only --daemon writes PID files)"
  fi
}

CMD="${1:-}"; case "$CMD" in start|stop|status) ;; *) usage_die ;; esac
shift

MODEL_PATH=""
if [ $# -ge 1 ] && [[ "$1" != --* ]]; then MODEL_PATH="$1"; shift; fi

NAME="" PORT=8000 TP="" DP="" MAX_MODEL_LEN=65536 MAX_NUM_SEQS=64
API_KEY="" EXTRA="" DAEMON=0 HEALTH_TIMEOUT=1800 ENV_FILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --name)           NAME="${2:?[serve_vllm] --name requires a value}"; shift 2 ;;
    --port)           PORT="${2:?[serve_vllm] --port requires a value}"; shift 2 ;;
    --tp)             TP="${2:?[serve_vllm] --tp requires a value}"; shift 2 ;;
    --dp)             DP="${2:?[serve_vllm] --dp requires a value}"; shift 2 ;;
    --max-model-len)  MAX_MODEL_LEN="${2:?[serve_vllm] --max-model-len requires a value}"; shift 2 ;;
    --max-num-seqs)   MAX_NUM_SEQS="${2:?[serve_vllm] --max-num-seqs requires a value}"; shift 2 ;;
    --api-key)        API_KEY="${2:?[serve_vllm] --api-key requires a value}"; shift 2 ;;
    --extra)          EXTRA="${2:?[serve_vllm] --extra requires a value}"; shift 2 ;;
    --daemon)         DAEMON=1; shift ;;
    --health-timeout) HEALTH_TIMEOUT="${2:?[serve_vllm] --health-timeout requires a value}"; shift 2 ;;
    --env-file)       ENV_FILE="${2:?[serve_vllm] --env-file requires a value}"; shift 2 ;;
    *) echo "[vllm-serve] Unknown argument: $1" >&2; usage_die ;;
  esac
done

if [ "$CMD" = "stop" ]; then do_stop "$NAME"; exit 0; fi
if [ "$CMD" = "status" ]; then do_status "$NAME"; exit 0; fi

# ---- start ----
[ -n "$MODEL_PATH" ] || die "start requires <model_path>"
[ -n "$NAME" ] || NAME="$(basename "$MODEL_PATH")"

if [ -d "$MODEL_PATH" ]; then
  [ -f "$MODEL_PATH/config.json" ] || die "$MODEL_PATH has no config.json (expected a Hugging Face model directory)"
  if ! grep -qiE 'vision|visual|image' "$MODEL_PATH/config.json" 2>/dev/null; then
    info "WARN: no vision keyword in config.json; these benchmarks require a multimodal model"
  fi
else
  info "$MODEL_PATH is not a local directory; treating it as a Hugging Face repository ID"
fi

if [ "$MAX_MODEL_LEN" -lt 32768 ]; then
  info "WARN: --max-model-len=$MAX_MODEL_LEN < 32768, multi-turn image context may exceed the limit (see README.md)"
fi

if health_check "$PORT" "$API_KEY"; then
  die "Port $PORT already serves an OpenAI-compatible endpoint (stop it or use another --port)"
fi

PID_FILE="$(pid_file_of "$NAME")"
LOG_FILE="$LOG_DIR/vllm_$NAME.log"
ENV_FILE="${ENV_FILE:-$KIT_ROOT/.env.vllm.$NAME}"

if [ -f "$PID_FILE" ] && kill -0 "$(sed -n 1p "$PID_FILE")" 2>/dev/null; then
  die "Instance $NAME is running (pid $(sed -n 1p "$PID_FILE")); stop it first or use another --name"
fi
rm -f "$PID_FILE"

DP="${DP:-8}"
[[ "$DP" =~ ^[1-9][0-9]*$ ]] || die "--dp must be a positive integer (got: '$DP')"
n_dev="$(device_count)"
TP="${TP:-2}"
[[ "$TP" =~ ^[1-9][0-9]*$ ]] || die "--tp must be a positive integer (got: '$TP')"
if [ "$((TP * DP))" -gt "$n_dev" ]; then
  die "tp×dp = $TP×$DP = $((TP * DP)) exceeds $n_dev devices; choose smaller --tp/--dp (e.g. --tp 4 --dp 2 for eight GPUs)"
fi
info "model=$MODEL_PATH  name=$NAME  port=$PORT  tp=$TP  dp=$DP (total devices $((TP * DP))/$n_dev)"
info "max-model-len=$MAX_MODEL_LEN  max-num-seqs=$MAX_NUM_SEQS(per replica; endpoint concurrency approximately $((DP * MAX_NUM_SEQS)))  api_key_configured=${API_KEY:+yes}"

cat > "$ENV_FILE" <<EOF
OPENAI_API_KEY=${API_KEY:-dummy-not-checked}
OPENAI_BASE_URL=http://127.0.0.1:$PORT/v1
MODEL=$NAME
EOF
info "Wrote evaluation environment: $ENV_FILE  (vLLM log: $LOG_FILE)"

run_args=(serve "$MODEL_PATH" --served-model-name "$NAME" --host 0.0.0.0 --port "$PORT"
          --tensor-parallel-size "$TP" --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS")
if [ "$DP" -gt 1 ]; then run_args+=(--data-parallel-size "$DP"); fi
if [ -n "$API_KEY" ]; then run_args+=(--api-key "$API_KEY"); fi
extra_arr=()
if [ -n "$EXTRA" ]; then read -r -a extra_arr <<< "$EXTRA"; fi
if [ "${#extra_arr[@]}" -gt 0 ]; then run_args+=("${extra_arr[@]}"); fi

if [ "$DAEMON" -eq 0 ]; then
  info "Starting in foreground (Ctrl-C stops; use --daemon for background serving), log tee → $LOG_FILE"
  "${VLLM_BIN}" "${run_args[@]}" 2>&1 | tee "$LOG_FILE"
  exit 0
fi

info "Starting in background; readiness timeout ${HEALTH_TIMEOUT}s"
nohup "${VLLM_BIN}" "${run_args[@]}" > "$LOG_FILE" 2>&1 &
pid=$!
printf '%s\n%s\n%s\n' "$pid" "$PORT" "$API_KEY" > "$PID_FILE"
info "pid=$pid  (PID file: $PID_FILE)"

elapsed=0
while :; do
  if health_check "$PORT" "$API_KEY"; then break; fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "--- $LOG_FILE tail ---" >&2
    tail -n 40 "$LOG_FILE" >&2
    rm -f "$PID_FILE"
    die "vLLM process exited (see log tail above)"
  fi
  if [ "$elapsed" -ge "$HEALTH_TIMEOUT" ]; then
    rm -f "$PID_FILE"
    die "Readiness timed out after ${HEALTH_TIMEOUT}s; inspect: tail -n 100 $LOG_FILE"
  fi
  sleep 10
  elapsed=$((elapsed + 10))
done

info "Service ready: curl http://127.0.0.1:$PORT/v1/models  (should contain \"$NAME\")"
info "Smoke test: scripts/run_sspro.sh $ENV_FILE results/smoke_sspro --max_samples 10"
info "Full run: scripts/run_all.sh    $ENV_FILE results/run-name"
