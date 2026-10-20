#!/usr/bin/env bash
# ============================================================================
# Agentic GUI Grounding RL: zoom-in crop-image GRPO training script
#                                                         train_zoomin_grounding_grpo.sh
# ----------------------------------------------------------------------------
# Example script:
#   This script is a reference example. Our actual training runs used multiple
#   nodes; GPU models, device memory, interconnects, and software environments
#   may differ from your setup. Adapt the topology and configuration to your
#   environment, and perform the necessary testing and debugging before a full
#   training run. Start with a small smoke test to check rollout generation,
#   communication between nodes, weight synchronization, and memory usage.
#
# Model:
#   LBS: Qwen3.8-27b -> cold-start SFT -> Agentic RL (GRPO).
#   MODEL_PATH points to the SFT checkpoint used to initialize Agentic RL.
#
# Task:
#   Multi-turn crop-image conversational RL. The assistant emits textual
#   actions `<action>crop_image(...)</action>`; the environment (a scheduler
#   running inside the rollout server) executes the crops and returns each
#   result (image + region metadata) as the next user message. The model
#   progressively zooms in, then returns final [0,1000]-normalized
#   coordinates `[x.xx,y.yy]` on the ORIGINAL image.
#
#     rewards           : zoomin_accuracy + zoomin_crop_process_gated +
#                         zoomin_format + zoomin_zoom_bonus (+ optional
#                         zoomin_cot_rm LLM judge; disabled unless
#                         ZOOMIN_COT_RM_API_BASE is set)
#     multi_turn_scheduler: zoomin_grounding_scheduler
#     plugin             : plugins/grounding_zoom_in/ (deployed automatically)
#     algorithm          : GRPO (ms-swift `--rlhf_type grpo`, DeepSpeed ZeRO-3)
#     rollout            : vLLM server mode, one `swift rollout` server per
#                          rollout node, training nodes connect via
#                          --vllm_server_host/--vllm_server_port lists
#
# Required software:
#   pip install ms-swift vllm deepspeed pillow
#   (this plugin needs the `grounding_zoom_in` plugin files shipped with this
#    repo; the script deploys them into your ms-swift repo/plugin dir)
#
# Usage:
#   # single-node smoke test (first N devices run the rollout server,
#   # remaining devices run GRPO training)
#   MODEL_PATH=/models/qwen3.8-27b-sft NNODES=1 bash train_zoomin_grounding_grpo.sh
#
#   # multi-node: the first ROLLOUT_NNODES nodes each run a vLLM rollout
#   # server; the remaining nodes run the GRPO trainer.
#   # Every node must be started with the same environment variables:
#   #   NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT, VLLM_SERVER_HOSTS,
#   #   VLLM_SERVER_PORTS (see "Topology" below)
#   NNODES=4 ROLLOUT_NNODES=1 NODE_RANK=0 MASTER_ADDR=<host0> ... \
#       bash train_zoomin_grounding_grpo.sh
#
# Data:
#   A GRPO dataset is a jsonl where each row looks like
#     {"problem": "<image>... following instruction: <instruction> ...",
#      "solution": "[x1, y1, x2, y2]",       # GT bbox, original-image pixels
#      "images": ["relative/or/absolute/path.png"]}
#   See data_samples/ for an 8-row example (prompt-only) plus complete
#   multi-turn rollout trajectories.
#
# Compatibility note (multi-server):
#   Passing MULTIPLE rollout servers via space-separated
#   --vllm_server_host/--vllm_server_port lists requires an ms-swift build
#   whose VLLMClient shards inference requests across servers and performs
#   parallel weight synchronization. Community ms-swift generally supports
#   a single server; keep ROLLOUT_NNODES=1 unless your build supports lists.
# ============================================================================
set -e

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
_HOSTNAME=$(hostname 2>/dev/null || echo unknown-host)
log() { local level="$1"; shift; echo "[zoomin-rl][$(date '+%Y-%m-%d %H:%M:%S')][${_HOSTNAME}][rank=${NODE_RANK:-NA}][${level}] $*" >&2; }
dumpvar() {
    local name
    for name in "$@"; do
        echo "[zoomin-rl][$(date '+%Y-%m-%d %H:%M:%S')][${_HOSTNAME}][rank=${NODE_RANK:-NA}][VAR] ${name}=${!name-<unset>}" >&2
    done
}
log INFO "script start: $0 (pid=$$)"

if [ "${DEBUG:-0}" = "1" ]; then
    export PS4='+[zoomin-rl][$(date "+%H:%M:%S")][${_HOSTNAME}][${BASH_SOURCE##*/}:${LINENO}] '
    set -x
fi

# ---------------------------------------------------------------------------
# 0) Configuration (every knob is overridable via environment variables)
# ---------------------------------------------------------------------------
_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && pwd)

# Model & plugin
MODEL_PATH=${MODEL_PATH:-""}                       # required SFT checkpoint, e.g. /models/qwen3.8-27b-sft
PLUGIN_DIR_REL=${PLUGIN_DIR_REL:-"plugins/grounding_zoom_in"}
PLUGIN_REL="${PLUGIN_DIR_REL}/zoomin_grounding_plugin.py"
PLUGIN_SRC_DIR="${_SCRIPT_DIR}/${PLUGIN_DIR_REL}"
MAX_TURNS=${MAX_TURNS:-6}                          # max interaction turns per rollout
ENABLE_THINKING=${ENABLE_THINKING:-true}           # text-action protocol relies on think blocks

# Where the plugin gets deployed (any writable dir; swift imports plugins by path)
LOCAL_PLUGIN_ROOT=${LOCAL_PLUGIN_ROOT:-"${_SCRIPT_DIR}/deployed_plugins"}

# Datasets: space-separated jsonl paths.
# Default = the 8-row smoke sample shipped in this repo; replace for real training.
# The small sample does not reduce model memory requirements; size hardware accordingly.
DATASETS=${DATASETS:-"${_SCRIPT_DIR}/data_samples/grpo_train_prompts/train_prompts_head8.jsonl"}

# Context / image budgets (rollout server and trainer MUST agree)
MAX_LENGTH=${MAX_LENGTH:-32768}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-12288}
MAX_COMPLETION_TOKENS=${MAX_COMPLETION_TOKENS:-2560}
MAX_PIXELS=${MAX_PIXELS:-16777216}                 # original-image vision budget
MIN_PIXELS=${MIN_PIXELS:-65536}

# Zoom-in tool protocol knobs (must match plugin limits)
MAX_TOOL_CALLS=${MAX_TOOL_CALLS:-3}                # plugin hard cap MAX_CROP_IMAGE_CALLS=3
MAX_ACTION_CORRECTIONS=${MAX_ACTION_CORRECTIONS:-2}
CROP_UPSCALE_FACTOR=${CROP_UPSCALE_FACTOR:-4.0}    # crops are upscaled for inspection
CROP_MAX_PIXELS=${CROP_MAX_PIXELS:-4000000}        # per-crop pixel budget
CROP_ROOT=${CROP_ROOT:-"${HOME}/.cache/zoomin_crops"}

# Reward components: four deterministic geometry rewards + optional LLM judge.
# The judge (zoomin_cot_rm) calls any OpenAI-compatible endpoint through the
# official OpenAI Python SDK and is DISABLED while ZOOMIN_COT_RM_API_BASE is
# empty (neutral fallback) — training works without it.
export ZOOMIN_COT_RM_API_BASE="${ZOOMIN_COT_RM_API_BASE:-}"   # e.g. https://api.deepseek.com/v1 or http://host:8000/v1
export ZOOMIN_COT_RM_API_KEY="${ZOOMIN_COT_RM_API_KEY:-}"     # leave empty for unauthenticated endpoints
export ZOOMIN_COT_RM_MODEL="${ZOOMIN_COT_RM_MODEL:-deepseek-chat}"
REWARD_FUNCS=${REWARD_FUNCS:-"zoomin_accuracy zoomin_crop_process_gated zoomin_format zoomin_zoom_bonus zoomin_cot_rm"}
REWARD_WEIGHTS=${REWARD_WEIGHTS:-"1.0 0.3 0.5 1.0 0.3"}

# vLLM server-side inference parameters
VLLM_SERVER_PORT=${VLLM_SERVER_PORT:-8000}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-65536}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.9}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-32}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}
# Expert parallel only helps MoE models; keep off for dense models.
VLLM_ENABLE_EXPERT_PARALLEL=${VLLM_ENABLE_EXPERT_PARALLEL:-false}
# Extra vLLM EngineArgs as JSON, e.g. GPU async scheduling or NPU compile modes.
# Ascend NPU example:
#   VLLM_ENGINE_KWARGS='{"max_num_batched_tokens":8192,"async_scheduling":true,
#     "compilation_config":{"cudagraph_mode":"FULL_DECODE_ONLY"},
#     "additional_config":{"enable_cpu_binding":true}}'
VLLM_ENGINE_KWARGS=${VLLM_ENGINE_KWARGS:-''}

# Training hyperparameters
NUM_GENERATIONS=${NUM_GENERATIONS:-8}              # GRPO group size
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
LR=${LR:-1e-6}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_P=${TOP_P:-0.95}
SAVE_STEPS=${SAVE_STEPS:-50}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
TARGET_EFF_BATCH=${TARGET_EFF_BATCH:-512}          # global effective batch aim
MOVE_MODEL_BATCHES=${MOVE_MODEL_BATCHES:-3}        # model-move batching for weight sync

# Output & logs
EXP_NAME=${EXP_NAME:-"zoomin_grounding_grpo"}
SAVE_PATH=${SAVE_PATH:-"${_SCRIPT_DIR}/output/${EXP_NAME}"}
LOG_DIR=${LOG_DIR:-"${SAVE_PATH}/logs"}
mkdir -p "$LOG_DIR" "$CROP_ROOT"

# Export the protocol environment consumed by the scheduler
export ZOOMIN_MAX_TOOL_CALLS="$MAX_TOOL_CALLS"
export ZOOMIN_MAX_ACTION_CORRECTIONS="$MAX_ACTION_CORRECTIONS"
export ZOOMIN_MAX_COMPLETION_TOKENS="$MAX_COMPLETION_TOKENS"
export ZOOMIN_CROP_UPSCALE_FACTOR="$CROP_UPSCALE_FACTOR"
export ZOOMIN_CROP_MAX_PIXELS="$CROP_MAX_PIXELS"
export ZOOMIN_CROP_ROOT="$CROP_ROOT"
export MAX_PIXELS MIN_PIXELS

dumpvar MODEL_PATH DATASETS MAX_LENGTH MAX_TURNS MAX_TOOL_CALLS MAX_ACTION_CORRECTIONS \
         MAX_PIXELS MIN_PIXELS CROP_UPSCALE_FACTOR CROP_MAX_PIXELS CROP_ROOT \
         VLLM_SERVER_PORT VLLM_ENABLE_EXPERT_PARALLEL NUM_GENERATIONS LR SAVE_PATH

if [ -z "$MODEL_PATH" ]; then
    log ERROR "MODEL_PATH is empty. Point it at a HuggingFace-style SFT checkpoint dir, e.g. MODEL_PATH=/models/qwen3.8-27b-sft."
    exit 1
fi

# ---------------------------------------------------------------------------
# 1) Deploy the plugin (the plugin imports zoomin_grounding_protocol.py, so
#    both files must sit in the same directory)
# ---------------------------------------------------------------------------
if [ -f "$PLUGIN_SRC_DIR/zoomin_grounding_plugin.py" ] && [ -f "$PLUGIN_SRC_DIR/zoomin_grounding_protocol.py" ]; then
    mkdir -p "$LOCAL_PLUGIN_ROOT/${PLUGIN_DIR_REL}/"
    cp -f "$PLUGIN_SRC_DIR"/*.py "$LOCAL_PLUGIN_ROOT/${PLUGIN_DIR_REL}/"
    rm -rf "${LOCAL_PLUGIN_ROOT:?}/${PLUGIN_DIR_REL}/__pycache__"
    PLUGIN_PATH="$LOCAL_PLUGIN_ROOT/${PLUGIN_REL}"
    log INFO "[plugin] deployed: $PLUGIN_PATH"
else
    log ERROR "plugin sources not found under $PLUGIN_SRC_DIR"; exit 1
fi

# ---------------------------------------------------------------------------
# 2) Distributed topology
#    - first ROLLOUT_NNODES nodes: dedicated vLLM rollout servers
#    - remaining nodes: GRPO training
#    - NNODES=1 degenerates to single-node colocate mode (first N devices run
#      the server, the rest train)
# ---------------------------------------------------------------------------
export NNODES=${NNODES:-${WORLD_SIZE:-1}}
export NODE_RANK=${NODE_RANK:-${RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-1}

# Device count: GPUs by default, Ascend NPUs when USE_ASCEND_NPU=1
if [ "${USE_ASCEND_NPU:-0}" = "1" ]; then
    DEV_VAR=ASCEND_RT_VISIBLE_DEVICES
    [ -n "${NPUS_PER_NODE:-}" ] && TOTAL_DEV="${NPUS_PER_NODE}" \
        || TOTAL_DEV=$(npu-smi info -l 2>/dev/null | grep -oE '[0-9]+' | head -n1) || TOTAL_DEV=8
else
    DEV_VAR=CUDA_VISIBLE_DEVICES
    if [ -n "${GPUS_PER_NODE:-}" ]; then
        TOTAL_DEV="$GPUS_PER_NODE"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        TOTAL_DEV=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
    else
        TOTAL_DEV=8
    fi
fi
[ -z "${TOTAL_DEV:-}" ] && TOTAL_DEV=8
[ "$TOTAL_DEV" -lt 1 ] && TOTAL_DEV=8
dumpvar NNODES NODE_RANK MASTER_ADDR MASTER_PORT ROLLOUT_NNODES "$DEV_VAR" TOTAL_DEV

VLLM_TP=${VLLM_TP:-1}
VLLM_DP=${VLLM_DP:-1}
ROLLOUT_NUM_DEV=$((VLLM_TP * VLLM_DP))
if [ "$ROLLOUT_NUM_DEV" -gt "$TOTAL_DEV" ] && [ "$NNODES" -le 1 ]; then
    ROLLOUT_NUM_DEV=$(( TOTAL_DEV / 2 )); [ "$ROLLOUT_NUM_DEV" -lt 1 ] && ROLLOUT_NUM_DEV=1
    VLLM_TP=$ROLLOUT_NUM_DEV; VLLM_DP=1
fi

if [ "$NNODES" -le 1 ]; then
    # ---- single-node colocate mode ----
    log INFO "[stage 1] single-node colocate topology."
    TRAIN_NNODES=1
    export NPROC_PER_NODE=$(( TOTAL_DEV - ROLLOUT_NUM_DEV ))
    if [ "${NPROC_PER_NODE:-0}" -lt 1 ]; then
        log ERROR "no devices left for training (total=$TOTAL_DEV rollout=$ROLLOUT_NUM_DEV)."; exit 1
    fi
    ROLLOUT_DEVS=$(seq -s, 0 $(( ROLLOUT_NUM_DEV - 1 )))
    TRAIN_DEVS=$(seq -s, "$ROLLOUT_NUM_DEV" $(( TOTAL_DEV - 1 )))
    VLLM_SERVER_HOSTS="localhost"
    VLLM_SERVER_PORTS="$VLLM_SERVER_PORT"
    IS_ROLLOUT_NODE=1
    IS_TRAIN_NODE=1
else
    # ---- multi-node mode: rank < ROLLOUT_NNODES runs a server ----
    export TRAIN_NNODES=$(( NNODES - ROLLOUT_NNODES ))
    [ "$TRAIN_NNODES" -lt 1 ] && { log ERROR "no training nodes."; exit 1; }
    if [ "$NODE_RANK" -lt "$ROLLOUT_NNODES" ]; then
        IS_ROLLOUT_NODE=1; IS_TRAIN_NODE=0; TRAIN_DEVS=""
        ROLLOUT_DEVS=$(seq -s, 0 $(( ROLLOUT_NUM_DEV - 1 )))
    else
        IS_ROLLOUT_NODE=0; IS_TRAIN_NODE=1
        export NODE_RANK=$(( NODE_RANK - ROLLOUT_NNODES ))
        export NNODES=$TRAIN_NNODES
        export NPROC_PER_NODE=$TOTAL_DEV
        TRAIN_DEVS=$(seq -s, 0 $(( TOTAL_DEV - 1 )))
        # train nodes form their own rendezvous on the first training node.
        # Set TRAIN_MASTER_ADDR explicitly for multi-server multi-node runs.
        if [ -z "${TRAIN_MASTER_ADDR:-}" ]; then
            log WARN "TRAIN_MASTER_ADDR not set; falling back to MASTER_ADDR=$MASTER_ADDR \
(this is only correct when the first training node IS the launch master."
        fi
        export MASTER_ADDR=${TRAIN_MASTER_ADDR:-$MASTER_ADDR}
        export MASTER_PORT=${TRAIN_MASTER_PORT:-$(( MASTER_PORT + 100 ))}
    fi
    # rollout server addresses must be provided explicitly for multi-node runs
    if [ -z "${VLLM_SERVER_HOSTS:-}" ]; then
        log ERROR "multi-node mode requires VLLM_SERVER_HOSTS (space-separated addresses of the rollout nodes)."
        exit 1
    fi
    VLLM_SERVER_PORTS=$(for _ in $(seq 1 "$ROLLOUT_NNODES"); do echo -n "$VLLM_SERVER_PORT "; done)
fi
dumpvar IS_ROLLOUT_NODE IS_TRAIN_NODE VLLM_TP VLLM_DP ROLLOUT_DEVS TRAIN_DEVS VLLM_SERVER_HOSTS VLLM_SERVER_PORTS

# Weight-sync group ports (server #i listens on BASE+i; all nodes must agree)
VLLM_SERVER_GROUP_PORT_BASE=${VLLM_SERVER_GROUP_PORT_BASE:-$(( 40000 + $(echo -n "$EXP_NAME" | cksum | awk '{print $1}') % 20000 ))}
VLLM_SERVER_GROUP_PORTS_LIST=$(seq -s' ' "$VLLM_SERVER_GROUP_PORT_BASE" $(( VLLM_SERVER_GROUP_PORT_BASE + ROLLOUT_NNODES - 1 )))

# ---------------------------------------------------------------------------
# 3) DeepSpeed config: ZeRO-3 with optimizer offload only.
#    (offload_param can slow multi-node startup badly; omit it unless OOM.)
# ---------------------------------------------------------------------------
GLOBAL_TRAIN_NPUS=$(( NPROC_PER_NODE * TRAIN_NNODES ))
DEFAULT_GA=$(( TARGET_EFF_BATCH / GLOBAL_TRAIN_NPUS )); [ "$DEFAULT_GA" -lt 1 ] && DEFAULT_GA=1
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-$DEFAULT_GA}

_DS_CONFIG_FILE="$SAVE_PATH/zero3_optim_offload.json"
mkdir -p "$SAVE_PATH"
cat > "$_DS_CONFIG_FILE" <<'DSJSON'
{
    "fp16": {"enabled": "auto", "loss_scale": 0, "loss_scale_window": 1000,
             "initial_scale_power": 16, "hysteresis": 2, "min_loss_scale": 1},
    "bf16": {"enabled": "auto"},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": true},
        "offload_param": {"device": "none", "pin_memory": true},
        "overlap_comm": false,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
DSJSON
log INFO "[stage 2] DeepSpeed config written: $_DS_CONFIG_FILE"

# ---------------------------------------------------------------------------
# 4) NPU (Ascend) runtime tuning - optional, only applied when USE_ASCEND_NPU=1.
#    GPU users can ignore this block entirely.
# ---------------------------------------------------------------------------
if [ "${USE_ASCEND_NPU:-0}" = "1" ]; then
    if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
        # shellcheck disable=SC1091
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
    fi
    export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
    export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}
    export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
    export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-512}
    export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
    export ASCEND_LAUNCH_BLOCKING=${ASCEND_LAUNCH_BLOCKING:-0}
    export VLLM_ASCEND_BALANCE_SCHEDULING=${VLLM_ASCEND_BALANCE_SCHEDULING:-0}
    export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
    log INFO "[stage 3] Ascend NPU tuning applied."
fi
export VLLM_SKIP_MEMORY_CHECK=${VLLM_SKIP_MEMORY_CHECK:-1}

# ---------------------------------------------------------------------------
# 5) Rollout server nodes: launch one `swift rollout` server (TP x DP)
# ---------------------------------------------------------------------------
_run_rollout_server() {
    local label="server#${NODE_RANK}"
    log INFO "===== [stage 4] starting rollout ${label} @ 0.0.0.0:${VLLM_SERVER_PORT} (TP=$VLLM_TP, DP=$VLLM_DP) ====="
    local engine_args=()
    if [ -n "$VLLM_ENGINE_KWARGS" ]; then
        engine_args=(--vllm_engine_kwargs "$VLLM_ENGINE_KWARGS")
    fi
    env "$DEV_VAR=$ROLLOUT_DEVS" \
    swift rollout \
        --model "$MODEL_PATH/" \
        --external_plugins "$PLUGIN_PATH" \
        --multi_turn_scheduler zoomin_grounding_scheduler \
        --vllm_use_async_engine true \
        --vllm_tensor_parallel_size "$VLLM_TP" \
        --vllm_data_parallel_size "$VLLM_DP" \
        --vllm_enable_expert_parallel "$VLLM_ENABLE_EXPERT_PARALLEL" \
        --max_turns "$MAX_TURNS" \
        --enable_thinking "$ENABLE_THINKING" \
        --preserve_thinking true \
        --max_pixels "$MAX_PIXELS" \
        --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
        --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
        --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
        --vllm_enable_prefix_caching false \
        --vllm_mm_processor_cache_gb 0 \
        "${engine_args[@]}" \
        --host 0.0.0.0 \
        --port "$VLLM_SERVER_PORT" \
        > "$LOG_DIR/rollout_server_${NODE_RANK}.log" 2>&1 &
    ROLLOUT_PID=$!
    log INFO "[stage 4] ${label} pid=$ROLLOUT_PID; waiting for ready..."
    _t_server=$(date +%s)
    for i in $(seq 1 720); do
        if curl -sf "http://127.0.0.1:${VLLM_SERVER_PORT}/health" >/dev/null 2>&1 \
           || curl -sf "http://127.0.0.1:${VLLM_SERVER_PORT}/v1/models" >/dev/null 2>&1; then
            log INFO "[stage 4] ${label} ready in $(( $(date +%s) - _t_server ))s."
            return 0
        fi
        if ! kill -0 "$ROLLOUT_PID" 2>/dev/null; then
            log ERROR "[stage 4] ${label} exited early; log tail:"
            tail -n 80 "$LOG_DIR/rollout_server_${NODE_RANK}.log" >&2 || true
            exit 1
        fi
        [ $(( i % 12 )) -eq 0 ] && log INFO "[stage 4] still waiting for ${label} ... ($(( $(date +%s) - _t_server ))s)"
        sleep 5
    done
    log ERROR "[stage 4] ${label} not ready in time."; exit 1
}

if [ "$IS_ROLLOUT_NODE" = "1" ]; then
    _run_rollout_server
    if [ "$IS_TRAIN_NODE" = "0" ]; then
        log INFO "[stage 4] dedicated rollout node; waiting for server process to exit."
        wait "$ROLLOUT_PID"; exit $?
    fi
fi

# ---------------------------------------------------------------------------
# 6) Training nodes: wait for every rollout server to answer, then start GRPO
# ---------------------------------------------------------------------------
log INFO "===== [stage 5] training node waiting for $ROLLOUT_NNODES rollout server(s) ====="
WAIT_TIMEOUT=${WAIT_TIMEOUT:-3600}
_srv_idx=0
for _srv_host in $VLLM_SERVER_HOSTS; do
    _srv_port=$(echo "$VLLM_SERVER_PORTS" | cut -d' ' -f$((_srv_idx + 1)))
    for i in $(seq 1 $(( WAIT_TIMEOUT / 5 ))); do
        if curl -sf "http://${_srv_host}:${_srv_port}/health" >/dev/null 2>&1 \
           || curl -sf "http://${_srv_host}:${_srv_port}/v1/models" >/dev/null 2>&1; then
            log INFO "[stage 5] server #${_srv_idx} reachable."; break
        fi
        [ "$i" -eq $(( WAIT_TIMEOUT / 5 )) ] && { log ERROR "server #${_srv_idx} unreachable: ${_srv_host}:${_srv_port}"; exit 1; }
        sleep 5
    done
    _srv_idx=$((_srv_idx + 1))
done

log INFO "===== [stage 6] starting GRPO ($DEV_VAR=[${TRAIN_DEVS}], NPROC=$NPROC_PER_NODE, NNODES=$NNODES, MASTER=$MASTER_ADDR:$MASTER_PORT) ====="
_t_train=$(date +%s)
set +e
env "$DEV_VAR=${TRAIN_DEVS}" \
swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL_PATH/" \
    --external_plugins "$PLUGIN_PATH" \
    --reward_funcs $REWARD_FUNCS \
    --reward_weights $REWARD_WEIGHTS \
    --columns '{}' \
    --remove_unused_columns false \
    --enable_thinking "$ENABLE_THINKING" \
    --preserve_thinking true \
    --max_pixels "$MAX_PIXELS" \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host $VLLM_SERVER_HOSTS \
    --vllm_server_port $VLLM_SERVER_PORTS \
    --vllm_server_group_port $VLLM_SERVER_GROUP_PORTS_LIST \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --dataset $DATASETS \
    --max_length "$MAX_LENGTH" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --truncation_strategy left \
    --num_train_epochs 1 \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LR" \
    --lr_scheduler_type cosine \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --logging_steps 1 \
    --warmup_ratio 0.0 \
    --dataloader_num_workers 4 \
    --num_generations "$NUM_GENERATIONS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --deepspeed "$_DS_CONFIG_FILE" \
    --move_model_batches "$MOVE_MODEL_BATCHES" \
    --log_completions true \
    --report_to tensorboard \
    --max_grad_norm 1.0 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --scale_rewards none \
    --save_only_model true \
    --output_dir "$SAVE_PATH" \
    2>&1 | tee "$LOG_DIR/train_${EXP_NAME}.log"
TRAIN_EXIT_CODE=${PIPESTATUS[0]}
set -e
if [ "$TRAIN_EXIT_CODE" -eq 0 ]; then
    log INFO "===== [stage 6] training finished OK in $(( $(date +%s) - _t_train ))s (output: $SAVE_PATH) ====="
else
    log ERROR "===== [stage 6] training FAILED (exit=$TRAIN_EXIT_CODE) ====="
fi

# ---------------------------------------------------------------------------
# 7) Cleanup local rollout server (single-node colocate path)
# ---------------------------------------------------------------------------
if [ -n "${ROLLOUT_PID:-}" ]; then
    log INFO "[stage 7] stopping rollout server pid=$ROLLOUT_PID ..."
    kill "$ROLLOUT_PID" 2>/dev/null || true
fi
log INFO "script end: exit=${TRAIN_EXIT_CODE:-0}"
exit "${TRAIN_EXIT_CODE:-0}"