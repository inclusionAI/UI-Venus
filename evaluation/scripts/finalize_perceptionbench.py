#!/usr/bin/env python3
"""Merge PerceptionBench shards, compute scores, and archive predictions.

Usage: python3 scripts/finalize_perceptionbench.py --results DIR --model MODEL
       [--num-shards 8] [--output-suffix SUFFIX] [--judge-model NAME] [--keep-shards]

Reads PerceptionBench_agentic_<MODEL><SUFFIX>[_shard{i}of{N}].jsonl from the run
directory (single shard has no shard tag; _shard0of1 is also accepted). Fallbacks
are evaluation/results then perceptionbench/results; partial/ambiguous sets fail. Writes perceptionbench.jsonl and
perceptionbench_scores.json, plus a model_card.yaml template if absent.
The headline excludes failed/skipped judgments; the all-sample score includes
them as wrong. Shards are deleted unless --keep-shards is supplied.
Rejudging merged predictions does not recreate shards: retain and update the
shards before running this finalizer again."""
import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

_EVALUATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EVALUATION_ROOT))
from perceptionbench.artifacts import discover_predictions, prediction_prefix

DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "<judge-model>")
MODEL_CARD_TEMPLATE = """\
# Run metadata (fill in setting and eval_protocol)
model: "{model}"
setting: ""       # Describe reward configuration, training data, and checkpoint step
eval_protocol: "sspro+osworld+perceptionbench+hrbench agentic evaluation"
judge_note: "judge={judge}"
results:
  sspro: sspro.json            # metrics.overall.action_acc
  osworld_g: osworld.json       # metrics.overall.action_acc (+by_data_type)
  perceptionbench: perceptionbench_scores.json   # overall (excluding failed judgments)
  hrbench_4k: hrbench.json     # splits.HRBench4K.strict_accuracy
  hrbench_8k: hrbench.json     # splits.HRBench8K.strict_accuracy
"""


def discover_shard_dirs(results_dir, prefix, num_shards, pb_results_dir):
    """Find a complete shard set, preferring the supplied run directory."""
    try:
        paths = discover_predictions((results_dir, pb_results_dir), prefix, num_shards)
        return str(paths[0].parent)
    except FileNotFoundError:
        return None



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--output-suffix", default=os.getenv("PB_OUTPUT_SUFFIX", ""),
                    help="Prediction suffix; defaults to PB_OUTPUT_SUFFIX")
    ap.add_argument("--keep-shards", action="store_true",
                    help="Keep intermediate shard predictions and scores")
    args = ap.parse_args()
    RES, MODEL = args.results, args.model
    try:
        prefix = prediction_prefix(MODEL, args.output_suffix)
        shard_paths = discover_predictions(
            (RES, _EVALUATION_ROOT / "results", _EVALUATION_ROOT / "perceptionbench/results"),
            prefix, args.num_shards)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"[finalize] {exc}") from exc
    shard_dir = str(shard_paths[0].parent)

    # Merge shards.
    records = {}
    for path in shard_paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    records[r["index"]] = r  # Deduplicate by index; resumed shards can contain repeated entries.

    rows = sorted(records.values(), key=lambda r: r["index"])
    total = len(rows)
    correct = sum(1 for r in rows if r.get("judge_result") == 1)
    wrong = sum(1 for r in rows if r.get("judge_result") == 0)
    skipped = sum(1 for r in rows if r.get("judge_status") == "skipped")
    judge_fail = total - correct - wrong - skipped
    no_answer = sum(1 for r in rows if r.get("judge_status") == "model_no_answer"
                    or r.get("final_answer") in (None, ""))
    judged = total - judge_fail - skipped
    overall_exc = correct / judged if judged > 0 else 0.0
    overall_paper = correct / total if total else 0.0

    per_cat = defaultdict(lambda: [0, 0])
    for r in rows:
        cat = r.get("error_category", "unknown")
        per_cat[cat][1] += 1
        if r.get("judge_result") == 1:
            per_cat[cat][0] += 1
    per_cat_score = {c: round(n[0] / n[1], 4) for c, n in per_cat.items() if n[1]}

    # Read headline metrics from other benchmarks in the same run directory.
    summary, counts_extra = {}, {}
    sspro_json = os.path.join(RES, "sspro.json")
    if os.path.exists(sspro_json):
        summary["sspro_action_acc"] = json.load(
            open(sspro_json))["metrics"]["overall"]["action_acc"]
    osw_json = os.path.join(RES, "osworld.json")
    if os.path.exists(osw_json):
        summary["osworld_action_acc"] = json.load(
            open(osw_json))["metrics"]["overall"]["action_acc"]

    scores = {
        "overall": round(overall_exc, 4),
        "overall_judge_failures_as_wrong": round(overall_paper, 4),
        "per_category": dict(sorted(per_cat_score.items(), key=lambda x: -x[1])),
        "counts": {"total": total, "correct": correct, "wrong": wrong,
                   "judge_failures": judge_fail, "judge_skipped": skipped,
                   "model_no_or_empty_answer": no_answer},
        "config": {"model": MODEL, "prompt_mode": "agentic",
                   "num_shards": args.num_shards, "output_suffix": args.output_suffix,
                   "judge_model": args.judge_model},
        "other_benchmarks_in_dir": summary,
    }

    # Write merged predictions and scores.
    with open(os.path.join(RES, "perceptionbench.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(RES, "perceptionbench_scores.json"), "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    if not os.path.exists(os.path.join(RES, "model_card.yaml")):
        with open(os.path.join(RES, "model_card.yaml"), "w", encoding="utf-8") as f:
            f.write(MODEL_CARD_TEMPLATE.format(model=MODEL, judge=args.judge_model))

    # Clean intermediate shards unless explicitly retained.
    if not args.keep_shards:
        removed = []
        for path in shard_paths:
            for artifact in (path, path.with_name(path.stem + "_scores.json")):
                if artifact.exists():
                    artifact.unlink()
                    removed.append(str(artifact))
        print(f"Removed {len(removed)} intermediate shard artifacts")

    # Display completion summary.
    print(f"Shard directory: {shard_dir} ({total} unique records)")
    print(f"PB overall(excl judge fail)={overall_exc:.4f}")
    print(f"PB overall(paper-style)={overall_paper:.4f}  ({correct}/{total})")
    print(f"judge={args.judge_model}  judge_fail={judge_fail}  "
          f"judge_skipped={skipped}  no_answer={no_answer}")
    for k, v in summary.items():
        print(f"{k}={v}")
    if total and judge_fail / total > 0.02:
        print("judge_fail > 2%: use perceptionbench/debug/rejudge_agentic.py "
              "on retained shards, then finalize again (--statuses skipped,judge_error for deferred judging)")
    print("Saved: perceptionbench.jsonl / perceptionbench_scores.json / model_card.yaml")


if __name__ == "__main__":
    main()
