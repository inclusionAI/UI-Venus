#!/usr/bin/env bash
# Shared directory discovery, environment loading, and runtime defaults.
# Each runner sources this file, then the user environment file.
# stage_output_ready checks readable output, model identity, and supplied settings.
# CLEAN_INTERMEDIATES controls removal after successful completion.
KIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SSPRO_DIR="$KIT_ROOT/sspro"
PB_DIR="$KIT_ROOT/perceptionbench"
HRBENCH_DIR="$KIT_ROOT/hrbench"

DATA_ROOT="$KIT_ROOT/data"
SSPRO_IMGS="${SSPRO_IMGS:-$DATA_ROOT/sspro/Screenspot-pro/images}"
SSPRO_ANN="${SSPRO_ANN:-$DATA_ROOT/sspro/Screenspot-pro/annotations}"
OSW_IMGS="${OSW_IMGS:-$DATA_ROOT/osworld_g/os-world-g-images}"
OSW_ANN="${OSW_ANN:-$DATA_ROOT/osworld_g/sspro_annotations}"
PB_DATASET="${PB_DATASET:-$DATA_ROOT/perceptionbench/PerceptionBench.jsonl}"
HRBENCH_CACHE="${HRBENCH_CACHE:-$DATA_ROOT/hrbench}"

RESULTS_ROOT="${RESULTS_ROOT:-$KIT_ROOT/results}"

SSPRO_WORKERS="${SSPRO_WORKERS:-8}"
SSPRO_NUM_PROCESSES="${SSPRO_NUM_PROCESSES:-8}"
HRBENCH_WORKERS="${HRBENCH_WORKERS:-8}"
REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"
THINKING_BUDGET="${THINKING_BUDGET:-4096}"
PB_NUM_SHARDS="${PB_NUM_SHARDS:-8}"
PB_SHARD_CONCURRENCY="${PB_SHARD_CONCURRENCY:-4}"
PB_SKIP_JUDGE="${PB_SKIP_JUDGE:-0}"
CLEAN_INTERMEDIATES="${CLEAN_INTERMEDIATES:-1}"

derive_chat_url() {
  CHAT_COMPLETIONS_URL="${OPENAI_BASE_URL%/}/chat/completions"
}

require_env() {  # require_env VAR1 VAR2 ...
  local missing=()
  for v in "$@"; do
    if [ -z "${!v:-}" ]; then missing+=("$v"); fi
  done
  if [ ${#missing[@]} -gt 0 ]; then
    echo "[config] ERROR: Missing environment variables: ${missing[*]} (see .env.example)" >&2
    return 1
  fi
}

source_env_file() {  # source_env_file <env_file>
  local envf="${1:?Usage: source_env_file <env_file>}"
  if [ ! -f "$envf" ]; then
    echo "[config] ERROR: Environment file does not exist: $envf" >&2
    return 1
  fi
  set -a; source "$envf"; set +a
}

prepare_results_dir() {
  RES="${1:?Usage: prepare_results_dir <results_dir>}"
  mkdir -p "$RES"
  RES="$(cd "$RES" && pwd)"
  echo "[config] results dir = $RES"
}


has_smoke_arg() {
  local a
  for a in "$@"; do
    case "$a" in --max_samples|--max_samples=*) return 0 ;; esac
  done
  return 1
}

has_selection_arg() {
  local a
  for a in "$@"; do
    case "$a" in
      --max_samples|--max_samples=*|--task|--task=*|--gt_type|--gt_type=*|\
      --language|--language=*|--inst_style|--inst_style=*|\
      --screenspot_imgs|--screenspot_imgs=*|--screenspot_test|--screenspot_test=*) return 0 ;;
    esac
  done
  return 1
}

stage_output_ready() {
  local j="${1:?Usage: stage_output_ready <json> <model> [key=value...]}"
  local m="${2:?Usage: stage_output_ready <json> <model>}"
  shift 2
  [ -f "$j" ] || return 1
  python3 - "$j" "$m" "$KIT_ROOT" "$@" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(1)
if not isinstance(d, dict):
    sys.exit(1)
if not isinstance(d.get("metrics"), dict) and not isinstance(d.get("counts"), dict):
    sys.exit(1)
model = None
for path in (("run_config", "model"), ("config", "model"), ("model",)):
    v = d
    for k in path:
        v = v.get(k) if isinstance(v, dict) else None
    if isinstance(v, str) and v:
        model = v
        break
if model != sys.argv[2]:
    sys.exit(1)
for key in ("run_config", "config"):
    rc = d.get(key) if isinstance(d.get(key), dict) else {}
    if rc.get("max_samples") is not None:
        sys.exit(1)
if "run_config" in d or any(kv.startswith("completion.") for kv in sys.argv[4:]):
    sys.path.insert(0, sys.argv[3])
    from common.grounding_completion import validate_completion
    try:
        validate_completion(d, require_full=True)
    except (ValueError, KeyError, TypeError):
        sys.exit(1)
for kv in sys.argv[4:]:
    key, _, val = kv.partition("=")
    v = d
    for k in key.split("."):
        v = v.get(k) if isinstance(v, dict) else None
    # Finalizers predating suffix support only consumed the default filename.
    if (key == "config.output_suffix" and isinstance(d.get("config"), dict)
            and "output_suffix" not in d["config"]):
        v = ""
    if str(v) != val:
        sys.exit(1)
sys.exit(0)
PY
}

clean_agentic_intermediates() {
  local dir="${1:?Usage: clean_agentic_intermediates <dir> <prefix>...}"
  shift
  if [ "${CLEAN_INTERMEDIATES:-1}" != "1" ]; then
    echo "[config] CLEAN_INTERMEDIATES=${CLEAN_INTERMEDIATES:-1}, Retaining intermediates: $*"
    return 0
  fi
  local p pat f n=0
  for p in "$@"; do
    for pat in "$dir/$p.ckpt.json" "$dir/$p.ckpt_p"*.json "$dir/$p.ckpt"*.tmp \
               "$dir/$p.partial.json" "$dir/$p.partial_p"*.json "$dir/${p}_p"*.json; do
      for f in $pat; do
        [ -f "$f" ] || continue
        rm -f "$f" && n=$((n+1))
      done
    done
  done
  echo "[config] Removed $n intermediate files (prefixes: $*)"
}
