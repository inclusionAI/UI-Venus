#!/usr/bin/env python3
"""Merge OS-World-G positive and refusal results using the shared scorer.

Usage: python3 merge_osworld_results.py --pos POS --neg NEG --out OUTPUT

Joins data_type using (image basename, instruction), adds per-type metrics and
refusal behavior, and retains the positive run configuration. The annotation
file can be overridden with --annotation."""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
SSPRO_BENCH = os.path.join(_HERE, "..", "sspro")
sys.path.insert(0, os.path.abspath(SSPRO_BENCH))
from sspro_eval_common import evaluate  # noqa: E402
sys.path.insert(0, os.path.dirname(_HERE))
from common.grounding_completion import completion_metadata, result_key, validate_completion

ANN = os.path.join(_HERE, "..", "data", "osworld_g",
                   "os-world-g-refine", "annotation.json")


def load_details(path):
    with open(path) as f:
        d = json.load(f)
    return d["details"], d.get("run_config")


def merge_completion(pos, neg, annotation):
    """Require completed segments, or exact annotation coverage for legacy inputs."""
    rows = pos["details"] + neg["details"]
    if "completion" in pos or "completion" in neg:
        a, b = validate_completion(pos), validate_completion(neg)
        if a["scope"] != b["scope"]:
            raise ValueError("Cannot merge a smoke segment with a full segment")
        for key in ("model", "max_samples", "screenspot_imgs", "screenspot_test",
                    "language", "inst_style"):
            if pos.get("run_config", {}).get(key) != neg.get("run_config", {}).get(key):
                raise ValueError(f"Segment configurations disagree: {key}")
        scope = a["scope"]
    else:
        # Old files contain no completion evidence. Only a complete, unique match
        # to the supplied benchmark annotation is sufficient to migrate them.
        scope = "full"
    if scope == "full":
        expected = {(r["img_filename"], r["instruction"],
                     "negative" if r["data_type"] == "refusal" else "positive")
                    for r in annotation}
        actual = [(os.path.basename(r["img_path"]), r["prompt_to_evaluate"], r["gt_type"])
                  for r in rows]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Full OS-World-G segments do not cover the benchmark annotation")
    return completion_metadata(rows, [result_key(r) for r in rows],
                               1 if scope == "smoke" else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True)
    ap.add_argument("--neg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--annotation", default=ANN,
                    help="Source data_type annotation file (default: data/osworld_g/os-world-g-refine/annotation.json)")
    args = ap.parse_args()

    with open(args.pos) as f:
        pos = json.load(f)
    with open(args.neg) as f:
        neg = json.load(f)
    pos_details, run_config = pos["details"], pos.get("run_config")
    neg_details = neg["details"]
    details = pos_details + neg_details

    # Join source data_type metadata.
    with open(args.annotation) as f:
        ann = json.load(f)
    completion = merge_completion(pos, neg, ann)
    dt_map = {(x["img_filename"], x["instruction"]): x["data_type"] for x in ann}
    for r in details:
        key = (os.path.basename(r["img_path"]), r["prompt_to_evaluate"])
        r["data_type"] = dt_map.get(key, "unknown")

    report = evaluate(details)

    # Aggregate each data_type.
    by_dt = {}
    for dt in sorted({r["data_type"] for r in details}):
        sub = [r for r in details if r["data_type"] == dt]
        n = len(sub)
        nc = sum(1 for r in sub if r["correctness"] == "correct")
        nwf = sum(1 for r in sub if r["correctness"] == "wrong_format")
        by_dt[dt] = {
            "num_total": n,
            "num_correct_action": nc,
            "wrong_format_num": nwf,
            "action_acc": nc / n if n else 0,
        }
    report["metrics"]["by_data_type"] = by_dt

    # Refusal behavior: negative is correct refusal; positive is an attempted click.
    # Read samples[*].result; the top-level record has no result field.
    # Equivalent correctness labels: correct / wrong / wrong_format.
    neg_beh = {"negative": 0, "positive": 0, "no_answer": 0}
    for r in neg_details:
        res = None
        for s in r.get("samples", []):
            if s.get("result"):
                res = s["result"]
                break
        neg_beh[res if res in neg_beh else "no_answer"] += 1
    # Cross-check refusal counts against correctness.
    n_correct = sum(1 for r in neg_details if r["correctness"] == "correct")
    assert neg_beh["negative"] == n_correct, (
        f"negative count {neg_beh['negative']} != correct count {n_correct}")
    report["metrics"]["refusal_behavior"] = neg_beh

    run_config = dict(run_config or {})
    run_config["dataset"] = (
        "OS-World-G GD_evaluation (os-world-g-refine/annotation.json, 564 records; "
        "positive=bbox470+polygon40 enclosing-box containment, negative=refusal54 refusal checks; "
        "positive/negative segments evaluated separately and merged)"
    )
    run_config["merged_from"] = [os.path.basename(args.pos),
                                 os.path.basename(args.neg)]
    report["run_config"] = run_config
    report["completion"] = completion

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".tmp", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.replace(args.out + ".tmp", args.out)

    ov = report["metrics"]["overall"]
    print(f"overall: {ov['num_correct_action']}/{ov['num_total']} "
          f"action_acc={ov['action_acc']:.4f} wrong_format={ov['wrong_format_num']}")
    for dt, m in by_dt.items():
        print(f"  {dt:8s}: {m['num_correct_action']}/{m['num_total']} "
              f"acc={m['action_acc']:.4f} wf={m['wrong_format_num']}")
    print(f"refusal behavior: {neg_beh}")


if __name__ == "__main__":
    main()
