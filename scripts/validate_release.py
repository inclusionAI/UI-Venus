#!/usr/bin/env python3
"""Validate release artifacts, syntax, local links, and publication metadata.

Run from any directory. --original-results additionally compares saved metrics
and replays scoring with the current evaluators, without model or judge requests.
"""
from __future__ import annotations

import argparse
import collections
import copy
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "evaluation"), str(ROOT / "evaluation/sspro")]
from export_release import BENCHMARKS, COUNTS, trajectory


def load(path):
    return json.loads(Path(path).read_text())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def metric_equal(left, right):
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(metric_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(metric_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=0, abs_tol=1e-12)
    return left == right


def score_checks(source):
    from sspro_eval_common import evaluate
    hr = module(ROOT / "evaluation/hrbench/eval_hrbench_agentic.py", "release_hrbench")
    results = {}
    for b in BENCHMARKS:
        published = load(ROOT / f"benchmark_results/metrics/{b}.json")
        if b == "perceptionbench":
            original = load(source / "perceptionbench_scores.json")
            for k in ("overall", "overall_judge_failures_as_wrong", "per_category", "counts", "other_benchmarks_in_dir"):
                assert published[k] == original[k], (b, k)
            continue
        original = load(source / f"{b}.json")
        assert published["metrics"] == original["metrics"], b
        rows = original["details"]
        if b in ("sspro", "osworld"):
            metrics = evaluate(rows)["metrics"]
            assert all(original["metrics"][k] == v for k, v in metrics.items()), b
        else:
            args = argparse.Namespace(**original["config"])
            replay = hr.build_report(args, copy.deepcopy(rows), original["run_fingerprint"])
            assert metric_equal(replay["metrics"], original["metrics"]), b
        results[b] = "full metric tree preserved; scoring replay passed"
    assert load(ROOT / "benchmark_results/metrics/hrbench.json") == load(source / "hrbench.json")
    # Run actual finalizers and the OS-World-G merge in an isolated temporary run.
    with tempfile.TemporaryDirectory(prefix="lbs-score-replay-") as directory:
        tmp = Path(directory)
        rows = [json.loads(s) for s in (ROOT / "benchmark_results/per_sample/perceptionbench.jsonl").open()]
        for shard in range(8):
            with (tmp / f"PerceptionBench_agentic_release-validation_shard{shard}of8.jsonl").open("w") as handle:
                for row in rows[shard::8]:
                    handle.write(json.dumps(row) + "\n")
        for b in ("sspro", "osworld"):
            (tmp / f"{b}.json").write_text(json.dumps(load(ROOT / f"benchmark_results/metrics/{b}.json")))
        subprocess.run([sys.executable, str(ROOT / "evaluation/scripts/finalize_perceptionbench.py"),
                        "--results", str(tmp), "--model", "release-validation", "--judge-model",
                        "Deepseekv4-flash-0731", "--keep-shards"], check=True, capture_output=True)
        scored = load(tmp / "perceptionbench_scores.json")
        saved = load(source / "perceptionbench_scores.json")
        assert {k:v for k,v in scored.items() if k != "config"} == {k:v for k,v in saved.items() if k != "config"}
        results["perceptionbench"] = "all score fields preserved; 3000-record finalizer replay passed"
        osw = load(source / "osworld.json")
        for part, gt in (("pos", "positive"), ("neg", "negative")):
            (tmp / f"{part}.json").write_text(json.dumps({"details": [r for r in osw["details"] if r["gt_type"] == gt],
                                                       "run_config": osw["run_config"]}))
        subprocess.run([sys.executable, str(ROOT / "evaluation/scripts/merge_osworld_results.py"),
                        "--pos", str(tmp / "pos.json"), "--neg", str(tmp / "neg.json"),
                        "--out", str(tmp / "merged.json")], check=True, capture_output=True)
        assert load(tmp / "merged.json")["metrics"] == osw["metrics"]
        # HR-Bench resume tolerates truncated lines, ignores another fingerprint,
        # and uses the latest record for duplicate keys.
        progress = tmp / "progress.jsonl"
        progress.write_text('\n'.join([json.dumps({"run_fingerprint": f, "eval_key": k, "score": s})
                                      for f,k,s in [("a","one",0),("b","two",1),("a","one",1)]]) + '\n{"incomplete":')
        resumed = hr.load_progress(progress, "a")
        assert list(resumed) == ["one"] and resumed["one"]["score"] == 1
    results["hrbench_resume"] = "fingerprint filtering, duplicate replacement, truncated-line recovery passed"
    results["floating_point_note"] = "Stored metrics are exact copies. HR-Bench replay allows 1e-12 absolute rounding tolerance."
    return results


def public_files():
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if not path.is_file() or any(x in rel.parts for x in ("_release_archive", "__pycache__", ".git", "_work", ".venv")):
            continue
        if str(rel).startswith(("evaluation/data/", "evaluation/data_sources/", "evaluation/results/")):
            continue
        yield path


def artifact_checks():
    manifest = load(ROOT / "benchmark_results/manifest.json")
    total = 0
    for b in BENCHMARKS:
        entry = manifest["benchmarks"][b]
        files = list((ROOT / f"benchmark_results/trajectories/{b}").glob("*.json"))
        assert len(files) == len(entry["samples"]) == entry["selected_count"] == COUNTS[b]
        rows = [json.loads(s) for s in (ROOT / "benchmark_results" / entry["per_sample"]).open()]
        assert len(rows) == entry["evaluated_count"]
        total += len(rows)
        if b == "perceptionbench":
            counts = collections.Counter((x["category"], x["passed"]) for x in entry["samples"])
            assert len(counts) == 20 and set(counts.values()) == {2}
        if b.startswith("hrbench"):
            groups = [load(p)["record"]["group_index"] for p in files]
            assert len(set(groups)) == len(groups)
        for p in files:
            row = load(p)["record"]
            trace = trajectory(row)
            produced = set()
            for event in trace.get("tool_events", []):
                if not event.get("result", {}).get("ok"):
                    continue
                parent = event["arguments"].get("image_path")
                if isinstance(parent, str) and parent.startswith("crops/"):
                    assert parent in produced, (p.name, parent)
                produced.add(event["result"]["image_path"])
            image = row.get("img_path", row.get("image_path"))
            if image and image.startswith("evaluation/data/"):
                assert (ROOT / image).is_file(), image
    assert total == 6745
    samples = ROOT / "agentic_gui_grounding_rl/data_samples"
    prompts = [json.loads(line) for line in
               (samples / "grpo_train_prompts/train_prompts_head8.jsonl").read_text().splitlines()]
    rollouts = [json.loads(line) for line in
                (samples / "rollout_trajectories/rollout_trajectories_head32.jsonl").read_text().splitlines()]
    assert len(prompts) == 8 and len(rollouts) == 32
    used = set()
    for folder, rows in (("grpo_train_prompts", prompts), ("rollout_trajectories", rollouts)):
        for row in rows:
            for image in row["images"]:
                path = samples / folder / image
                assert path.is_file(), f"Missing training sample image: {path.relative_to(ROOT)}"
                if folder == "grpo_train_prompts":
                    used.add(path)
    assert used == set((samples / "grpo_train_prompts/images").glob("*.png"))
    groups = collections.defaultdict(list)
    for row in rollouts:
        groups[row["prompt_group"]].append(row["generation"])
    assert len(groups) == 4 and all(sorted(generations) == list(range(8)) for generations in groups.values())
    return {"per_item_records": total, "trajectories": sum(COUNTS.values()),
            "training_prompt_rows": len(prompts), "training_prompt_images": len(used),
            "training_rollout_trajectories": len(rollouts),
            "category_balance": "PerceptionBench: 10 categories, 2 successes + 2 failures each",
            "crop_references": "selected successful nested crop references resolve"}


def syntax_and_links():
    counts = collections.Counter()
    for path in public_files():
        if path.suffix == ".py":
            compile(path.read_text(), str(path), "exec")
            counts["python"] += 1
        elif path.suffix == ".sh":
            subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)
            counts["shell"] += 1
        elif path.suffix == ".md":
            text = path.read_text()
            for target in re.findall(r'\]\(([^)]+)\)', text):
                if target.startswith(("http:", "https:", "#", "mailto:")) or "<" in target or " " in target:
                    continue
                local = target.split("#")[0]
                if local:
                    assert (path.parent / local).exists(), (path.relative_to(ROOT), target)
            counts["markdown"] += 1
    return dict(counts)


def privacy_checks(original_results=None):
    # Match literal machine-specific paths, private deployment names, and stale labels.
    patterns = [r"/ossfs/", r"/root/", r"/Users/", r"/home/",
                r"TODO\(release\)", r"TODO-paper-url", r"TODO-huggingface-url"]
    if original_results:
        for name in ("sspro", "osworld", "hrbench_4k", "hrbench_8k", "perceptionbench_scores"):
            original = load(original_results / f"{name}.json")
            config = original.get("run_config", original.get("config", {}))
            for key in ("model", "judge_model"):
                if config.get(key) and config[key] not in {"released-checkpoint", "Deepseekv4-flash-0731"}:
                    patterns.append(re.escape(config[key]))
    # This validator contains its own denylist; it is not private run data.
    for path in public_files():
        if path.resolve() == Path(__file__).resolve() or path.suffix not in (".py", ".sh", ".md", ".json", ".jsonl", ".txt", ".yaml"):
            continue
        value = path.read_text()
        assert not any(re.search(pattern, value) for pattern in patterns), path.relative_to(ROOT)
    return "no matching private paths, checkpoint/deployment aliases, or stale release placeholders"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--original-results", type=Path,
                    help="Directory containing complete evaluator reports for metric comparison and scoring replay")
    ap.add_argument("--report", type=Path, default=ROOT / "_release_archive/validation/release_validation.json")
    args = ap.parse_args()
    report = {"artifacts": artifact_checks(), "syntax_and_links": syntax_and_links(),
              "privacy": privacy_checks(args.original_results), "live_training_or_model_evaluation": "not run"}
    if args.original_results:
        report["score_replay"] = score_checks(args.original_results)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
