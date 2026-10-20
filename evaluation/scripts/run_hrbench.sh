#!/usr/bin/env bash
# Evaluate HR-Bench 4K/8K: 800 option-rotation records per split.
# Usage: run_hrbench.sh ENV_FILE RESULTS_DIR [EVALUATOR_ARGS...]
# Splits run concurrently; HRBENCH_SPLITS=4k selects one split.
# Smoke-test size must be divisible by four, e.g. --max_samples 8.
# Report strict matching for publication; recorded judge metrics remain available.
# Output: hrbench_<split>.json and, for multiple splits, hrbench.json.
# Hidden inference/judge progress JSONL files support resume.
# The prepared image cache is sufficient; a source TSV is optional.
#   ./run_hrbench.sh .env.vllm.xxx results/myexp
#   HRBENCH_SPLITS=4k ./run_hrbench.sh .env.vllm.xxx results/hrbench_4k
set -euo pipefail
KIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=config.sh
source "$KIT_ROOT/scripts/config.sh"

ENV_FILE="${1:?Usage: $0 <env_file> <results_dir> [EVALUATOR_ARGS...]}"
shift
RES="${1:?Usage: $0 <env_file> <results_dir> [EVALUATOR_ARGS...]}"
shift

source_env_file "$ENV_FILE"
require_env MODEL OPENAI_API_KEY OPENAI_BASE_URL
derive_chat_url
prepare_results_dir "$RES"

SPLITS="${HRBENCH_SPLITS:-4k,8k}"
CROP_ROOT="${HRBENCH_CROP_ROOT:-$RES/crops_hrbench}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-$((THINKING_BUDGET + 1024))}"

echo "[hrbench] MODEL=$MODEL"
echo "[hrbench] API=$CHAT_COMPLETIONS_URL  workers=$HRBENCH_WORKERS/process  effort=$REASONING_EFFORT  splits=$SPLITS(parallel)"
echo "[hrbench] thinking_budget=$THINKING_BUDGET  max_completion_tokens=$MAX_COMPLETION_TOKENS"

report_paths=()
IFS=',' read -ra split_list <<< "$SPLITS"

if [ "${FORCE_RERUN:-0}" != "1" ] && ! has_smoke_arg "$@"; then
  hr_ready=1
  for split in "${split_list[@]}"; do
    stage_output_ready "$RES/hrbench_${split}.json" "$MODEL" || hr_ready=0
  done
  if [ "$hr_ready" = "1" ]; then
    if [ "${CLEAN_INTERMEDIATES:-1}" = "1" ]; then
      rm -f "$RES"/.hrbench_*.inference.progress.jsonl "$RES"/.hrbench_*.judge.progress.jsonl
    fi
    echo "[hrbench] Complete (splits=$SPLITS, model=$MODEL), skipping (set FORCE_RERUN=1 to rerun)"
    exit 0
  fi
fi

declare -a split_pids=()
for split in "${split_list[@]}"; do
  report="$RES/hrbench_${split}.json"
  report_paths+=("$report")
  echo "[hrbench] === split=$split started (log: $RES/hrbench_${split}_run.log) ==="
  (cd "$HRBENCH_DIR" && python3 eval_hrbench_agentic.py \
    --split "$split" \
    --image_cache "$HRBENCH_CACHE" \
    --log_path "$report" \
    --api_url "$CHAT_COMPLETIONS_URL" \
    --model "$MODEL" \
    --prompt_mode agentic \
    --reasoning_effort "$REASONING_EFFORT" \
    --thinking_budget "$THINKING_BUDGET" \
    --max_completion_tokens "$MAX_COMPLETION_TOKENS" \
    --crop_root "$CROP_ROOT/$split" \
    --workers "$HRBENCH_WORKERS" \
    --judge_workers "$HRBENCH_WORKERS" \
    "$@") > "$RES/hrbench_${split}_run.log" 2>&1 &
  split_pids+=("$split:$!")
done

hr_fail=0
for item in "${split_pids[@]}"; do
  split="${item%%:*}"; pid="${item##*:}"
  if wait "$pid"; then
    echo "[hrbench] split=$split OK"
  else
    echo "[hrbench] split=$split FAILED (rc=$?; see $RES/hrbench_${split}_run.log)"
    hr_fail=1
  fi
done
[ "$hr_fail" -eq 0 ] || exit 1

if [ "${CLEAN_INTERMEDIATES:-1}" = "1" ]; then
  rm -f "$RES"/.hrbench_*.inference.progress.jsonl "$RES"/.hrbench_*.judge.progress.jsonl
fi

if [ "${#report_paths[@]}" -gt 1 ]; then
  (cd "$HRBENCH_DIR" && python3 eval_hrbench_agentic.py \
    --combine_reports "${report_paths[@]}" \
    --log_path "$RES/hrbench.json")
  echo "[hrbench] Completed: $RES/hrbench.json"
else
  echo "[hrbench] Completed: ${report_paths[0]}"
fi
python3 - "${report_paths[@]}" <<'PY'
import json, sys
for path in sys.argv[1:]:
    r = json.load(open(path))
    m = r["metrics"]
    print("[hrbench] %s: strict_accuracy=%.5f" % (
        r["benchmark"],
        m["strict"]["average"]["all"]["accuracy"]))
PY
