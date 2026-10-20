#!/usr/bin/env python3
"""Export the six reviewed trajectories for the buildless HTML explorer.

Uses sanitized publication records, never runs inference, and never writes to
benchmark data. --reconstruct-images reuses the existing checked crop executor.
Without that option the shipped image excerpts are reused. --check is read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets/trajectory_explorer"
CASES = [
    ("osworld", 344, "Recover from a misplaced crop", "Observation correction",
     "A wrong first view becomes useful feedback: the model returns to the original screenshot and relocates the target.",
     ["The first crop contains only the blue slide background.",
      "The second crop returns to the original screenshot and isolates the yellow NOTE style.",
      "The recorded click falls inside the reference target box."],
     "A correct final click does not make the initial crop useful; the recovery is the behavior illustrated here."),
    ("sspro", 811, "Look closer, then closer again", "Nested observation",
     "The model first isolates a toolbar, then crops that returned image to distinguish a small Programmer icon.",
     ["A wide toolbar crop retains neighboring icons as context.",
      "Crop 2 uses the resized Crop 1 as its parent, rather than the original screenshot.",
      "The recorded point maps back into the original-image target box."],
     "Coordinates in a nested action belong to its parent crop. The original-image mapping must be preserved."),
    ("perceptionbench", 31, "Combine evidence across regions", "Parallel observation",
     "Two regions are requested in one model turn. The next turn combines their evidence to count the bold-text modules.",
     ["The upper and lower regions branch from the original image in the same turn.",
      "The final reasoning counts 14 bordered modules and excludes one with regular-weight text, leaving 13."],
     "The stored Deepseekv4-flash-0731 judgment is correct. This is a curated example; repeated judgments can vary."),
    ("hrbench_8k", 132, "Reframe to read the inscription", "High-resolution transfer",
     "After recognizing that the initial view missed the sundial, the model repositions its crop and reads the year 1762.",
     ["The first crop shows sky and a column edge, missing the sundial.",
      "The replacement crop uses the original image and contains the sundial inscription.",
      "The recorded option A matches the reference under strict option scoring."],
     "The first localization estimate is inaccurate. The evidence is the corrected crop and strict answer, not every claim in the model text."),
    ("osworld", 103, "Restore the missing context", "Context recovery",
     "A wider second observation restores the spreadsheet headers needed to locate cell Q35.",
     ["The first crop lacks the column headers needed to establish context.",
      "The wider crop restores O, P, and Q headers while retaining the target row.",
      "The recorded point falls inside Q35's reference box."],
     "The model says its first crop missed Q. Visual review supports the narrower claim that column-header context was missing."),
    ("hrbench_4k", 45, "Recognize the limit of enlargement", "Observation limitation",
     "A nested crop isolates a light-colored glove, but further enlargement does not recover detail absent from the source.",
     ["The first crop locates the person near the recycling bin.",
      "The nested crop isolates the glove; the image remains blurred.",
      "Option B matches the stored White gloves reference under strict scoring."],
     "The model describes the crop as clear. Visual review finds it blurred: a correct answer does not establish that zooming created new detail."),
]
LABELS = {"osworld": "OS-World-G (Refine)", "sspro": "ScreenSpot-Pro",
          "perceptionbench": "PerceptionBench", "hrbench_8k": "HR-Bench 8K",
          "hrbench_4k": "HR-Bench 4K"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_case(case):
    benchmark, index, title, behavior, analysis, notes, limitation = case
    rid = f"{benchmark}-{index:05d}"
    source = ROOT / f"benchmark_results/trajectories/{benchmark}/{rid}.json"
    record = json.loads(source.read_text())["record"]
    trace = record.get("agentic", record.get("trajectory", (record.get("trajectories") or [{}])[0]))
    events = trace["tool_events"]
    paths = {trace.get("original_image_path", events[0]["arguments"]["image_path"]): "Original"}
    crops = []
    for i, event in enumerate(events, 1):
        result = event["result"]
        if not result.get("ok"):
            raise ValueError(f"Unsuccessful crop in reviewed case {rid}")
        crops.append({"number": i, "turn": event["model_turn"],
                      "parent": paths[event["arguments"]["image_path"]],
                      "action": event["raw_action"],
                      "parent_box": result["crop_in_parent"],
                      "original_box": result["original_image_region_pixels"],
                      "dimensions": [result["width"], result["height"]],
                      "image": f"assets/trajectory_explorer/media/{rid}/crop_{i:02d}.png"})
        paths[result["image_path"]] = f"Crop {i}"
    turns = []
    for i, turn in enumerate(trace["rounds"]):
        response = turn.get("response", {})
        turns.append({"number": turn["model_turn"],
                      "content": response.get("content", turn.get("content", "")),
                      "reasoning_content": response.get("reasoning_content", ""),
                      "analysis": notes[i]})
    if benchmark == "perceptionbench":
        dimensions = [549, 654]  # Verified official image; checked on reconstruction.
        answer = record["final_answer"]
        reference = "13"  # Verified official reference documented in TRAJECTORIES.md.
        metric = "Stored judge result"
        outcome = "Correct" if record["judge_result"] == 1 else "Incorrect"
        score = {k: record[k] for k in ("judge_result", "judge_status", "judge_reason")}
        score["judge_model"] = "Self-hosted Deepseekv4-flash-0731"
    elif benchmark.startswith("hrbench"):
        dimensions = record["img_size"]
        answer = record["prediction_option"]
        reference = f'{record["answer"]} · {record["answer_text"]}'
        metric = "Strict option match"
        outcome = "Correct" if record["strict_score"] == 1 else "Incorrect"
        score = {"strict_score": record["strict_score"]}
    else:
        dimensions = [trace["original_image_width"], trace["original_image_height"]]
        answer = "[" + ", ".join(f"{n:.2f}" for n in trace["prediction_relative_0_1000"]) + "]"
        reference = "Point inside reference box"
        metric = "Point-in-box"
        outcome = record["correctness"].capitalize()
        score = {"correctness": record["correctness"], "bbox": record["bbox"], "pred": record["pred"]}
    return {"id": rid, "benchmark": LABELS[benchmark], "title": title,
            "behavior": behavior, "analysis": analysis, "limitation": limitation,
            "question": trace["instruction"], "dimensions": dimensions,
            "original": f"assets/trajectory_explorer/media/{rid}/original.jpg",
            "figure": f"assets/trajectories/{rid}.png", "turns": turns, "crops": crops,
            "answer": answer, "reference": reference, "metric": metric,
            "outcome": outcome, "score": score,
            "source": str(source.relative_to(ROOT)), "source_sha256": sha256(source),
            "source_record": f"assets/trajectory_explorer/records/{rid}.json"}


def reconstruct_images(cases):
    # Reuse the existing executor and its checks; no changes to crop geometry.
    from render_trajectories import load_pb, reconstruct
    from PIL import Image
    pb = load_pb({31})
    for case, spec in zip(cases, CASES):
        original, crops, checks, _ = reconstruct(spec[0], spec[1], pb)
        destination = ASSETS / "media" / case["id"]
        destination.mkdir(parents=True, exist_ok=True)
        with Image.open(original) as image:
            assert list(image.size) == case["dimensions"]
            preview = image.convert("RGB")
            preview.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            preview.save(destination / "original.jpg", quality=90, optimize=True)
        for crop, check, exported in zip(crops, checks, case["crops"]):
            assert check["original_image_region_pixels"] == exported["original_box"]
            with Image.open(crop) as image:
                assert list(image.size) == exported["dimensions"]
                image.save(ROOT / "docs" / exported["image"], optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruct-images", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check and args.reconstruct_images:
        parser.error("--check cannot reconstruct images")
    cases = [export_case(case) for case in CASES]
    if args.reconstruct_images:
        reconstruct_images(cases)
    image_files = [ROOT / "docs" / c[key] for c in cases for key in ("original",)]
    image_files += [ROOT / "docs" / crop["image"] for c in cases for crop in c["crops"]]
    for path in image_files:
        if not path.is_file():
            raise SystemExit(f"Missing {path}; run with --reconstruct-images after fetching benchmark data.")
    for case in cases:
        snapshot = ROOT / "docs" / case["source_record"]
        original = ROOT / case["source"]
        if args.check:
            if not snapshot.exists() or snapshot.read_bytes() != original.read_bytes():
                raise SystemExit(f"Source snapshot differs: {snapshot}")
        else:
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(original.read_bytes())
    payload = {"schema_version": 1, "cases": cases,
               "images_sha256": {str(p.relative_to(ROOT / "docs")): sha256(p) for p in image_files}}
    content = "// Generated by scripts/export_trajectory_html.py; do not edit.\nwindow.LBS_TRAJECTORIES = " + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n"
    target = ASSETS / "data.js"
    if args.check:
        if not target.exists() or target.read_text() != content:
            raise SystemExit("HTML trajectory data or image hashes differ; regenerate the export.")
        print(f"Verified {len(cases)} source records, verbatim round text, crop mappings, and {len(image_files)} image hashes.")
    else:
        ASSETS.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        print(f"Exported {len(cases)} reviewed cases to {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
