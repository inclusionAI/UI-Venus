#!/usr/bin/env bash
# Evaluate 3000 PerceptionBench tasks with optional deferred judging.
# Usage: run_perceptionbench.sh ENV_FILE RESULTS_DIR [NUM_SHARDS]
# Each worker handles every N-th record and resumes its own JSONL file.
# FINALIZE=0 keeps shards without running aggregation.
# FINALIZE=0 retains shards; CLEAN_INTERMEDIATES=0 retains wrapper logs.
# Manual finalization needs --keep-shards to preserve its input shards.
#       <results_dir>/perceptionbench.jsonl + perceptionbench_scores.json (finalize)
set -euo pipefail
KIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=config.sh
source "$KIT_ROOT/scripts/config.sh"

ENV_FILE="${1:?Usage: $0 <env_file> <results_dir> [num_shards]}"
shift
RES="${1:?Usage: $0 <env_file> <results_dir> [num_shards]}"
shift
SHARD_OVERRIDE="${1:-}"

source_env_file "$ENV_FILE"
NUM_SHARDS="${SHARD_OVERRIDE:-$PB_NUM_SHARDS}"
if ! [[ "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[pb] ERROR: num_shards must be a positive integer" >&2
  exit 1
fi
export THINKING_BUDGET
export PB_OUTPUT_SUFFIX="${PB_OUTPUT_SUFFIX:-}"
require_env MODEL OPENAI_API_KEY OPENAI_BASE_URL
prepare_results_dir "$RES"

if [ "${FORCE_RERUN:-0}" != "1" ] && [ "${FINALIZE:-1}" = "1" ] && [ "${PB_SKIP_JUDGE:-0}" != "1" ] \
   && stage_output_ready "$RES/perceptionbench_scores.json" "$MODEL" "config.num_shards=$NUM_SHARDS" "config.output_suffix=$PB_OUTPUT_SUFFIX"; then
  echo "[pb] Complete (perceptionbench_scores.json, model=$MODEL, shards=$NUM_SHARDS), skipping (set FORCE_RERUN=1 to rerun)"
  exit 0
fi

export DATASET="$PB_DATASET"
export PB_RESULTS_DIR="$RES"

echo "[pb] MODEL=$MODEL  effort=$REASONING_EFFORT  judge=${JUDGE_MODEL:-<unset>}"
echo "[pb] DATASET=$DATASET"
echo "[pb] shards=$NUM_SHARDS  concurrency/shard=$PB_SHARD_CONCURRENCY  skip_judge=$PB_SKIP_JUDGE"

cd "$PB_DIR"
pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  NUM_SHARDS="$NUM_SHARDS" SHARD_INDEX="$i" CONCURRENCY="$PB_SHARD_CONCURRENCY" \
  PB_SKIP_JUDGE="$PB_SKIP_JUDGE" PYTHONUNBUFFERED=1 \
    python3 eval_perceptionbench_agentic.py \
    > "$RES/pb_shard${i}of${NUM_SHARDS}.log" 2>&1 &
  pids+=($!)
done
echo "[pb] Shard PIDs: ${pids[*]}"
for p in "${pids[@]}"; do wait "$p"; done
echo "[pb] All shards finished"

if [ "${FINALIZE:-1}" = "1" ]; then
  echo "[pb] Finalizing results"
  python3 "$KIT_ROOT/scripts/finalize_perceptionbench.py" \
    --results "$RES" --model "$MODEL" --num-shards "$NUM_SHARDS" --output-suffix="$PB_OUTPUT_SUFFIX" \
    | tee -a "$RES/pb_finalize.log"
  if [ "${CLEAN_INTERMEDIATES:-1}" = "1" ]; then
    rm -f "$RES"/pb_shard*of*.log
    echo "[pb] Removed shard logs pb_shard*of*.log (set CLEAN_INTERMEDIATES=0 to retain)"
  fi
else
  echo "[pb] FINALIZE=0, skipping aggregation (run scripts/finalize_perceptionbench.py later)"
fi
