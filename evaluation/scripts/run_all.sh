#!/usr/bin/env bash
# Run SSPro -> OS-World-G -> PerceptionBench -> HR-Bench, then report scores.
# Usage: run_all.sh ENV_FILE [RESULTS_DIR]
# SSPro/OSWorld use SSPRO_NUM_PROCESSES * SSPRO_WORKERS workers.
# PB uses PB_NUM_SHARDS * PB_SHARD_CONCURRENCY concurrent requests.
# HR-Bench runs 4K and 8K concurrently, each with HRBENCH_WORKERS.
# Completed stages are skipped unless FORCE_RERUN=1; partial runs resume.
# Use CLEAN_INTERMEDIATES=0 to retain checkpoints and shard artifacts.
#   ./run_all.sh <env_file> [results_dir]
set -euo pipefail
KIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=config.sh
source "$KIT_ROOT/scripts/config.sh"

ENV_FILE="${1:?Usage: $0 <env_file> [results_dir]}"
shift
if [ $# -ge 1 ]; then
  RES="$1"; shift
else
  RES="$RESULTS_ROOT/eval__$(date '+%Y%m%d-%H%M%S')"
fi
say(){ echo "[chain][$(date '+%F %T')] $*" | tee -a "$LOG"; }

source_env_file "$ENV_FILE"
require_env MODEL OPENAI_API_KEY OPENAI_BASE_URL
derive_chat_url
mkdir -p "$RES"
RES="$(cd "$RES" && pwd)"
LOG="$RES/chain.log"
say "Starting chain | RES=$RES | ENV=$ENV_FILE"
say "MODEL=$MODEL  API=$CHAT_COMPLETIONS_URL"
say "SSPro/OSW processes=$SSPRO_NUM_PROCESSES × threads/process=$SSPRO_WORKERS (total concurrency=$((SSPRO_NUM_PROCESSES * SSPRO_WORKERS))) | PB shards=$PB_NUM_SHARDS"

run_one(){  # run_one <name> <cmd...>
  local name="$1"; shift
  say ">>> $name started"
  if "$@" >> "$LOG" 2>&1; then
    say "<<< $name finished OK"
  else
    say "<<< $name FAILED (rc=$?; see $LOG and *_run.log in the output directory)"
    return 1
  fi
}

run_one "SSPro (1581 tasks)" \
  bash "$KIT_ROOT/scripts/run_sspro.sh" "$ENV_FILE" "$RES" --num_processes "$SSPRO_NUM_PROCESSES"
run_one "OS-World-G(pos+refusal+merge)" \
  bash "$KIT_ROOT/scripts/run_osworld.sh" "$ENV_FILE" "$RES" --num_processes "$SSPRO_NUM_PROCESSES"
run_one "PerceptionBench(${PB_NUM_SHARDS} shards + judge)" \
  bash "$KIT_ROOT/scripts/run_perceptionbench.sh" "$ENV_FILE" "$RES" "$PB_NUM_SHARDS"
run_one "HR-Bench(4k+8k)" \
  bash "$KIT_ROOT/scripts/run_hrbench.sh" "$ENV_FILE" "$RES"

say "Chain complete: $RES"
python3 - "$RES" <<'PY' | tee -a "$LOG"
import json, sys
res = sys.argv[1]
print("==== Evaluation summary ====")
try:
    ss = json.load(open(f"{res}/sspro.json"))
    print("SSPro action_acc = %.4f" % ss["metrics"]["overall"]["action_acc"])
except Exception as e:
    print(f"SSPro: missing output or read failure ({e})")
try:
    ow = json.load(open(f"{res}/osworld.json"))
    print("OSWorld-G action_acc = %.4f" % ow["metrics"]["overall"]["action_acc"])
except Exception as e:
    print(f"OSWorld-G: missing output or read failure ({e})")
try:
    pb = json.load(open(f"{res}/perceptionbench_scores.json"))
    print(f"PB overall(excl judge fail) = {pb['overall']}  "
          f"(paper={pb['overall_judge_failures_as_wrong']}, counts={pb['counts']['total']})")
except Exception as e:
    print(f"PerceptionBench: missing output or read failure ({e})")
try:
    hb = json.load(open(f"{res}/hrbench.json"))
    for split in ("HRBench4K", "HRBench8K"):
        print(f"{split} strict_accuracy = {hb['splits'][split]['strict_accuracy']:.5f}")
except Exception as e:
    print(f"HR-Bench: missing output or read failure ({e})")
PY
