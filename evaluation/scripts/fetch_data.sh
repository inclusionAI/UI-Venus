#!/usr/bin/env bash
# Download official benchmark data, prepare the local layout, and verify it.
# Usage: fetch_data.sh all|sspro|osworld_g|perceptionbench|hrbench|verify
# Requires git, Python, and huggingface-cli for Hugging Face downloads.
# HF_ENDPOINT selects the download endpoint; SRC_DIR selects the raw cache.
# Raw sources may require about 11 GB; prepared data is independent of that cache.
# PerceptionBench images are embedded in JSONL; HR-Bench TSVs are decoded locally.
# Byte checks cover 2211 files; two HR-Bench manifests use task-content checks.
# Existing source downloads are reused. All/verify require all datasets to pass.
# ============================================================================
#                                                         scripts/fetch_data.sh
# ----------------------------------------------------------------------------
#   ScreenSpot-Pro   HF dataset  likaixin/ScreenSpot-Pro (images + annotations)
#   OS-World-G       GitHub      xlang-ai/OSWorld-G      (benchmark/)
#   HR-Bench         HF dataset  DreamMr/HR-Bench       (hr_bench_{4k,8k}.tsv)
#   curl git python3 (kit requirements.txt)
# ============================================================================
set -euo pipefail

KIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${SRC_DIR:-$KIT_ROOT/data_sources}"
DATA_ROOT="$KIT_ROOT/data"

log() { echo "[fetch_data][$(date '+%H:%M:%S')] $*" >&2; }
die() { echo "[fetch_data][ERROR] $*" >&2; exit 1; }

ensure_hf_cli() {
    if ! command -v huggingface-cli >/dev/null 2>&1; then
        die "huggingface-cli is required for dataset downloads: pip install -U \"huggingface_hub[cli]\""
    fi
}

fetch_sspro() {
    if [ ! -f "$SRC_DIR/screenspot-pro/annotations/blender_windows.json" ]; then
        ensure_hf_cli
        log "Downloading official ScreenSpot-Pro data (HF likaixin/ScreenSpot-Pro, ~3.2G)..."
        HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" \
            huggingface-cli download likaixin/ScreenSpot-Pro \
                --repo-type dataset --local-dir "$SRC_DIR/screenspot-pro"
    else
        log "Using cached ScreenSpot-Pro data"
    fi
    python3 "$KIT_ROOT/scripts/prepare_data.py" sspro \
        --official_sspro "$SRC_DIR/screenspot-pro" --data_root "$DATA_ROOT"
}

fetch_osworld_g() {
    if [ ! -f "$SRC_DIR/OSWorld-G/benchmark/OSWorld-G_refined.json" ]; then
        log "Cloning official OSWorld-G repository (GitHub xlang-ai/OSWorld-G)..."
        mkdir -p "$SRC_DIR"
        git clone --depth 1 https://github.com/xlang-ai/OSWorld-G "$SRC_DIR/OSWorld-G"
    else
        log "Using cached OSWorld-G repository"
    fi
    python3 "$KIT_ROOT/scripts/prepare_data.py" osworld \
        --official_benchmark "$SRC_DIR/OSWorld-G/benchmark" --data_root "$DATA_ROOT"
}

fetch_perceptionbench() {
    local dst="$DATA_ROOT/perceptionbench/PerceptionBench.jsonl"
    if [ -f "$dst" ] && [ "$(stat -Lf %z "$dst" 2>/dev/null || stat -Lc %s "$dst")" = "1626804092" ]; then
        log "PerceptionBench.jsonl is present (1626804092 bytes); skipping download"
        return 0
    fi
    if [ -f "$SRC_DIR/perceptionbench/PerceptionBench.jsonl" ] \
       && [ "$(stat -Lf %z "$SRC_DIR/perceptionbench/PerceptionBench.jsonl" 2>/dev/null || stat -Lc %s "$SRC_DIR/perceptionbench/PerceptionBench.jsonl")" = "1626804092" ]; then
        mkdir -p "$DATA_ROOT/perceptionbench"
        cp "$SRC_DIR/perceptionbench/PerceptionBench.jsonl" "$dst"
        log "Copied PerceptionBench.jsonl from SRC_DIR"
        return 0
    fi
    ensure_hf_cli
    log "Downloading official PerceptionBench data (HF moonshotai/PerceptionBench, ~1.6G)..."
    HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" \
        huggingface-cli download moonshotai/PerceptionBench \
            --repo-type dataset --include "PerceptionBench.jsonl" \
            --local-dir "$SRC_DIR/perceptionbench"
    mkdir -p "$DATA_ROOT/perceptionbench"
    cp "$SRC_DIR/perceptionbench/PerceptionBench.jsonl" "$dst"
}

fetch_hrbench() {
    local ok=1
    for f in hr_bench_4k.tsv hr_bench_8k.tsv; do
        [ -f "$SRC_DIR/hrbench/$f" ] || ok=0
    done
    local s4 s8
    s4=0; s8=0
    [ -f "$SRC_DIR/hrbench/hr_bench_4k.tsv" ] && s4=$(stat -Lf %z "$SRC_DIR/hrbench/hr_bench_4k.tsv" 2>/dev/null || stat -Lc %s "$SRC_DIR/hrbench/hr_bench_4k.tsv")
    [ -f "$SRC_DIR/hrbench/hr_bench_8k.tsv" ] && s8=$(stat -Lf %z "$SRC_DIR/hrbench/hr_bench_8k.tsv" 2>/dev/null || stat -Lc %s "$SRC_DIR/hrbench/hr_bench_8k.tsv")
    if [ "$ok" = "1" ] && [ "$s4" = "1932055330" ] && [ "$s8" = "4630731290" ]; then
        log "Using cached HR-Bench TSV files"
    else
        ensure_hf_cli
        log "Downloading official HR-Bench TSV files (HF DreamMr/HR-Bench, ~6.5 GB; rerun to resume)..."
        HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" \
            huggingface-cli download DreamMr/HR-Bench \
                --repo-type dataset --include "hr_bench_4k.tsv" "hr_bench_8k.tsv" \
                --local-dir "$SRC_DIR/hrbench"
    fi
    python3 "$KIT_ROOT/scripts/prepare_data.py" hrbench \
        --official_tsv_dir "$SRC_DIR/hrbench" --data_root "$DATA_ROOT"
}

run_verify() {
    if python3 "$KIT_ROOT/scripts/verify_data.py" --data-root "$DATA_ROOT"; then
        return 0
    fi
    if [ "$CMD" = "all" ] || [ "$CMD" = "verify" ]; then
        die "Data verification failed (all/verify require every benchmark to match)"
    fi
    log "WARN: verification failed in single-benchmark mode ($CMD); fetch missing datasets and rerun verify"
    return 0
}

CMD="${1:-all}"
case "$CMD" in
    sspro)            fetch_sspro ;;
    osworld_g)        fetch_osworld_g ;;
    perceptionbench)  fetch_perceptionbench ;;
    hrbench)          fetch_hrbench ;;
    all)              fetch_sspro
                      fetch_osworld_g
                      fetch_perceptionbench
                      fetch_hrbench ;;
    verify)           ;;
    *) die "Unknown subcommand: $CMD (all|sspro|osworld_g|perceptionbench|hrbench|verify)" ;;
esac
run_verify
log "Completed: $CMD"
