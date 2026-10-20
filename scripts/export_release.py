#!/usr/bin/env python3
"""Export publication metrics, per-item scores, and a deterministic trajectory subset.

This offline utility never alters evaluator outputs or re-runs a model/judge.
Example: python3 scripts/export_release.py --source /path/to/full/run
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ("sspro", "osworld", "perceptionbench", "hrbench_4k", "hrbench_8k")
FEATURED = {"sspro": [811], "osworld": [103, 344], "perceptionbench": [31, 7],
            "hrbench_4k": [45], "hrbench_8k": [132]}
COUNTS = {"sspro": 20, "osworld": 20, "perceptionbench": 40,
          "hrbench_4k": 10, "hrbench_8k": 10}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def trajectory(row):
    return row.get("agentic", row.get("trajectory", (row.get("trajectories") or [{}])[0]))


def passed(benchmark, row):
    if benchmark.startswith("hrbench"):
        return row["strict_score"] == 1
    if benchmark == "perceptionbench":
        return row["judge_result"] == 1
    return row["correctness"] == "correct"


def tags(row):
    trace = trajectory(row)
    events = trace.get("tool_events", [])
    result = set()
    count = trace.get("crop_action_count", 0)
    result.add(f"crops_{count}")
    seen = set()
    turns = []
    for event in events:
        if event.get("arguments", {}).get("image_path") in seen:
            result.add("nested_zoom")
        output = event.get("result", {})
        if output.get("ok"):
            seen.add(output.get("image_path"))
            turns.append(event.get("model_turn"))
    if len(turns) > len(set(turns)):
        result.add("multiple_regions_same_turn")
    for turn in trace.get("rounds", [])[1:]:
        text = turn.get("content", turn.get("response", {}).get("content", ""))
        if re.search(r"previous crop.{0,100}(?:miss|too far)|cropped the wrong|crop missed", text, re.I):
            result.add("self_correction_candidate")
    return sorted(result)


def select(benchmark, rows):
    """Balance outcomes/categories and greedily diversify recorded behaviors.

    Featured records are curated. Remaining ties use a stable seeded digest;
    the digest is only a sampling tie-breaker, not a source-file manifest.
    """
    chosen = list(FEATURED[benchmark])
    def category(row):
        if benchmark == "perceptionbench":
            return row["error_category"]
        if benchmark == "osworld":
            return row["data_type"]
        if benchmark == "sspro":
            return row["application"]
        return row["category"]
    def rank(i):
        return hashlib.sha256(f"LBS-release-v1:{benchmark}:{i}".encode()).hexdigest()
    def add_pool(pool, amount, success=None):
        while amount:
            candidates = [i for i in pool if i not in chosen and
                          (success is None or passed(benchmark, rows[i]) == success)]
            if benchmark.startswith("hrbench"):
                groups = {rows[i]["group_index"] for i in chosen}
                candidates = [i for i in candidates if rows[i]["group_index"] not in groups]
            if not candidates:
                raise ValueError(f"Insufficient distinct candidates for {benchmark}")
            used = collections.Counter(t for i in chosen for t in tags(rows[i]))
            cats = collections.Counter(category(rows[i]) for i in chosen)
            i = min(candidates, key=lambda i: (
                cats[category(rows[i])],
                sum(used[t] for t in tags(rows[i])) / max(1, len(tags(rows[i]))), rank(i)))
            chosen.append(i)
            amount -= 1
    if benchmark == "perceptionbench":
        for cat in sorted({category(r) for r in rows}):
            pool = [i for i, r in enumerate(rows) if category(r) == cat]
            for success in (True, False):
                existing = sum(category(rows[i]) == cat and passed(benchmark, rows[i]) == success
                               for i in chosen)
                add_pool(pool, 2 - existing, success)
    elif benchmark == "osworld":
        for cat, number in (("bbox", 12), ("polygon", 4), ("refusal", 4)):
            pool = [i for i, r in enumerate(rows) if category(r) == cat]
            for success, quota in ((True, number - max(1, number // 4)),
                                   (False, max(1, number // 4))):
                existing = sum(category(rows[i]) == cat and passed(benchmark, rows[i]) == success
                               for i in chosen)
                add_pool(pool, quota - existing, success)
    else:
        total = COUNTS[benchmark]
        for success, quota in ((True, total - total // 4), (False, total // 4)):
            add_pool(range(len(rows)), quota - sum(passed(benchmark, rows[i]) == success
                                                  for i in chosen), success)
    assert len(chosen) == COUNTS[benchmark] == len(set(chosen))
    return sorted(chosen)


class Sanitizer:
    """Consistent path/model pseudonyms; numeric results are never transformed."""
    def __init__(self, private_models, judge_models, public_model, public_judge):
        self.models = {x: public_model for x in private_models if x}
        self.models.update({x: public_judge for x in judge_models if x})
        self.paths = {}

    def register(self, benchmark, index, row):
        self.paths = {}
        trace = trajectory(row)
        rid = f"{benchmark}-{index:05d}"
        events = trace.get("tool_events", [])
        for event in events:
            path = event.get("result", {}).get("image_path")
            if isinstance(path, str) and path:
                self.paths[path] = f"crops/{rid}/{Path(path).name}"
        directory = trace.get("crop_directory")
        if directory:
            self.paths[directory] = f"crops/{rid}"
        if benchmark == "perceptionbench":
            originals = []
            for event in events:
                path = event.get("arguments", {}).get("image_path")
                if isinstance(path, str) and path and path not in self.paths and path not in originals:
                    originals.append(path)
            for number, path in enumerate(originals):
                self.paths[path] = f"dataset:perceptionbench/{row['index']}/image_{number}"

    def text(self, value):
        for old, new in sorted(self.paths.items(), key=lambda x: -len(x[0])):
            value = value.replace(old, new)
        for old, new in sorted(self.models.items(), key=lambda x: -len(x[0])):
            value = value.replace(old, new)
        # Replace the machine-specific prefix, retaining official dataset identity.
        value = re.sub(r"/(?:ossfs|root|Users|home)/[^\s\"'<>]*?/data/", "evaluation/data/", value)
        # Crop paths are logical references. Physical assets are reconstructed offline.
        value = re.sub(r"/(?:ossfs|root|Users|home|tmp|private)/[^\s\"'<>]*?/(?:crops_agentic|crops_hrbench|crops)/",
                       "crops/", value)
        value = re.sub(r"/(?:ossfs|root|Users|home|tmp|private)/[^\s\"'<>),;]+", "[redacted-path]", value)
        return value

    def __call__(self, value, key=""):
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                if k in {"api_key", "authorization", "exception_traceback"}:
                    result[k] = None
                elif k in {"api_url", "judge_api_url", "base_url"} and v:
                    result[k] = "http://localhost:8000/v1/chat/completions"
                elif k in {"response_metadata"} and isinstance(v, dict):
                    result[k] = self({a: b for a, b in v.items() if a not in {"id", "created"}})
                else:
                    result[k] = self(v, k)
            return result
        if isinstance(value, list):
            return [self(x, key) for x in value]
        return self.text(value) if isinstance(value, str) else value


def compact(benchmark, row):
    if benchmark == "perceptionbench":
        out = {k: v for k, v in row.items() if k not in {"agentic", "prediction"}}
    else:
        drop = {"trajectories", "trajectory", "additional_trajectories", "raw_response",
                "final_content", "official_judge_prompt", "exception_traceback"}
        out = {k: v for k, v in row.items() if k not in drop}
    out["trajectory_summary"] = {k: trajectory(row).get(k)
                                 for k in ("crop_action_count", "model_turn_count", "error")}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=ROOT / "benchmark_results")
    ap.add_argument("--model-label", default="released-checkpoint")
    ap.add_argument("--judge-label", default="Deepseekv4-flash-0731")
    args = ap.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output == source or output in source.parents or source in output.parents:
        ap.error("Source and output must be separate, non-nested directories")
    if output.exists() and any(output.iterdir()):
        ap.error("Output must be empty; use a new directory to avoid stale selected records")
    docs = {b: read_json(source / f"{b}.json") for b in BENCHMARKS if b != "perceptionbench"}
    pb = [json.loads(line) for line in (source / "perceptionbench.jsonl").open() if line.strip()]
    pb_scores = read_json(source / "perceptionbench_scores.json")
    private_models = {d.get("run_config", d.get("config", {})).get("model") for d in docs.values()}
    private_models.add(pb_scores["config"]["model"])
    clean = Sanitizer(private_models, {pb_scores["config"]["judge_model"]},
                      args.model_label, args.judge_label)
    manifest = {"schema_version": 1, "artifact_type": "publication_export",
                "model_label": args.model_label, "model_label_is_alias": True,
                "perceptionbench_judge": args.judge_label,
                "judge_note": "Self-hosted; stochastic judging can vary between runs.",
                "selection": "LBS-release-v1; curated examples plus deterministic category/outcome/behavior balancing",
                "redactions": ["machine paths", "internal checkpoint alias", "endpoint addresses",
                               "response request IDs and timestamps", "exception tracebacks"],
                "path_semantics": "Image references are repository-relative dataset paths or logical crop references; see docs/TRAJECTORIES.md.",
                "benchmarks": {}}
    for benchmark in BENCHMARKS:
        rows = pb if benchmark == "perceptionbench" else docs[benchmark]["details"]
        metric_doc = pb_scores if benchmark == "perceptionbench" else {
            k: v for k, v in docs[benchmark].items() if k != "details"}
        write_json(output / "metrics" / f"{benchmark}.json", clean(metric_doc))
        path = output / "per_sample" / f"{benchmark}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for i, row in enumerate(rows):
                clean.register(benchmark, i, row)
                value = clean(compact(benchmark, row))
                value["release_id"] = f"{benchmark}-{i:05d}"
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        selected = select(benchmark, rows)
        entries = []
        for i in selected:
            clean.register(benchmark, i, rows[i])
            rid = f"{benchmark}-{i:05d}"
            filename = f"trajectories/{benchmark}/{rid}.json"
            write_json(output / filename, {"release_id": rid, "source_row_zero_based": i,
                       "passed": passed(benchmark, rows[i]), "behavior_tags": tags(rows[i]),
                       "record": clean(rows[i])})
            entries.append({"release_id": rid, "path": filename,
                            "category": rows[i].get("error_category", rows[i].get("data_type", rows[i].get("category", rows[i].get("application")))),
                            "passed": passed(benchmark, rows[i]), "behavior_tags": tags(rows[i]),
                            "featured_candidate": i in FEATURED[benchmark]})
        manifest["benchmarks"][benchmark] = {"evaluated_count": len(rows), "selected_count": len(selected),
                "metrics": f"metrics/{benchmark}.json", "per_sample": f"per_sample/{benchmark}.jsonl", "samples": entries}
    write_json(output / "metrics/hrbench.json", clean(read_json(source / "hrbench.json")))
    hr = read_json(source / "hrbench.json")
    write_json(output / "headline.json", {
        "model": args.model_label,
        "sspro_action_accuracy": docs["sspro"]["metrics"]["overall"]["action_acc"],
        "osworld_action_accuracy": docs["osworld"]["metrics"]["overall"]["action_acc"],
        "perceptionbench_accuracy": pb_scores["overall"],
        "hrbench_4k_strict_accuracy": hr["splits"]["HRBench4K"]["strict_accuracy"],
        "hrbench_8k_strict_accuracy": hr["splits"]["HRBench8K"]["strict_accuracy"]})
    write_json(output / "manifest.json", manifest)
    print(json.dumps({b: manifest["benchmarks"][b]["selected_count"] for b in BENCHMARKS}))


if __name__ == "__main__":
    main()
