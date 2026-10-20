#!/usr/bin/env bash
# Evaluate 1581 ScreenSpot-Pro tasks using geometric correctness.
# Usage: run_sspro.sh ENV_FILE RESULTS_DIR [EVALUATOR_ARGS...]
# Smoke test: --max_samples 10. Output: sspro.json; resume: sspro.ckpt.json.
#   ./run_sspro.sh .env.vllm.myexp results/myexp
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

if [ "${FORCE_RERUN:-0}" != "1" ] && ! has_selection_arg "$@" \
   && stage_output_ready "$RES/sspro.json" "$MODEL" \
      "completion.scope=full" "run_config.task=all" "run_config.gt_type=positive" \
      "run_config.language=en" "run_config.inst_style=instruction" \
      "run_config.screenspot_imgs=$SSPRO_IMGS" "run_config.screenspot_test=$SSPRO_ANN"; then
  clean_agentic_intermediates "$RES" sspro
  echo "[sspro] Complete (sspro.json, model=$MODEL), skipping (set FORCE_RERUN=1 to rerun)"
  exit 0
fi

echo "[sspro] MODEL=$MODEL"
echo "[sspro] API=$CHAT_COMPLETIONS_URL  workers=$SSPRO_WORKERS  effort=$REASONING_EFFORT"

cd "$SSPRO_DIR"
python3 eval_screenspot_pro_agentic.py \
  --api_url "$CHAT_COMPLETIONS_URL" \
  --api_key "$OPENAI_API_KEY" \
  --model "$MODEL" \
  --screenspot_imgs "$SSPRO_IMGS" \
  --screenspot_test "$SSPRO_ANN" \
  --reasoning_effort "$REASONING_EFFORT" \
  --thinking_budget "$THINKING_BUDGET" \
  --num_workers "$SSPRO_WORKERS" \
  --log_path "$RES/sspro.json" \
  --checkpoint_path "$RES/sspro.ckpt.json" \
  "$@" 2>&1 | tee "$RES/sspro_run.log"

echo "[sspro] Completed: $RES/sspro.json"
python3 - "$RES/sspro.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("[sspro] action_acc = %.4f" % r["metrics"]["overall"]["action_acc"])
PY
clean_agentic_intermediates "$RES" sspro
