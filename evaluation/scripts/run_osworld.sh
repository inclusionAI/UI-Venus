#!/usr/bin/env bash
# Evaluate OS-World-G: 510 positive tasks + 54 refusal tasks = 564 total.
# Usage: run_osworld.sh ENV_FILE RESULTS_DIR [EVALUATOR_ARGS...]
# Positive and refusal passes use the SSPro entry, followed by per-type aggregation.
# Output: osworld.json; partial segments are resumable until successful cleanup.
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
   && stage_output_ready "$RES/osworld.json" "$MODEL" \
      "completion.scope=full" "run_config.task=osworld_g" "run_config.gt_type=positive" \
      "run_config.language=en" "run_config.inst_style=instruction" \
      "run_config.screenspot_imgs=$OSW_IMGS" "run_config.screenspot_test=$OSW_ANN"; then
  clean_agentic_intermediates "$RES" osworld_pos osworld_neg
  echo "[osworld] Complete (osworld.json, model=$MODEL), skipping (set FORCE_RERUN=1 to rerun)"
  exit 0
fi

common_args=(
  --api_url "$CHAT_COMPLETIONS_URL"
  --api_key "$OPENAI_API_KEY"
  --model "$MODEL"
  --screenspot_imgs "$OSW_IMGS"
  --screenspot_test "$OSW_ANN"
  --reasoning_effort "$REASONING_EFFORT"
  --thinking_budget "$THINKING_BUDGET"
  --num_workers "$SSPRO_WORKERS"
)

cd "$SSPRO_DIR"

echo "[osworld] starting positive pass"
python3 eval_screenspot_pro_agentic.py \
  "${common_args[@]}" --task osworld_g --gt_type positive \
  --log_path "$RES/osworld_pos.partial.json" \
  --checkpoint_path "$RES/osworld_pos.ckpt.json" \
  "$@" 2>&1 | tee "$RES/osworld_pos_run.log"

echo "[osworld] starting refusal pass"
python3 eval_screenspot_pro_agentic.py \
  "${common_args[@]}" --task osworld_g_refusal --gt_type negative \
  --log_path "$RES/osworld_neg.partial.json" \
  --checkpoint_path "$RES/osworld_neg.ckpt.json" \
  "$@" 2>&1 | tee "$RES/osworld_neg_run.log"

echo "[osworld] merging positive and refusal passes"
python3 "$KIT_ROOT/scripts/merge_osworld_results.py" \
  --pos "$RES/osworld_pos.partial.json" \
  --neg "$RES/osworld_neg.partial.json" \
  --out "$RES/osworld.json" | tee -a "$RES/osworld_merge.log"

clean_agentic_intermediates "$RES" osworld_pos osworld_neg
echo "[osworld] Completed: $RES/osworld.json"
