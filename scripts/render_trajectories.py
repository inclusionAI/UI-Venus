#!/usr/bin/env python3
"""Render evidence figures from official images and recorded crop events.

No model is called. Crops are reconstructed with the shared evaluation executor;
dimensions, clamped rectangles, and original-image mappings are checked against
the recorded events. These are reconstructions, not archived API image payloads.
Requires Pillow and matplotlib; run from any directory after fetching the data.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lbs-matplotlib")
sys.path.insert(0, str(ROOT / "evaluation"))
from common.agentic_pipeline import AgenticGroundingPipeline
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from figure_style import (INK, MUTED, RULE, CLAY, CLAY_DARK, SAGE_DARK,
                          BLUE_DARK, SAGE_BG, SERIF, apply_style)

ASSETS = ROOT / "docs/assets/trajectories"
CASES = [
    ("osworld", 344, "Recovering from a misplaced crop", "Click the light-yellow NOTE style in the Fontwork Gallery.",
     "The first crop contains the slide background.", "Return to the original image and inspect the gallery.", "Correct target: [510.42, 663.19]"),
    ("sspro", 811, "Looking closer, then closer again", "Locate the Programmer tool in the Quartus toolbar.",
     "Inspect the toolbar to identify a candidate icon.", "Crop the first crop to resolve the small icon.", "Correct target: [317.70, 40.83]"),
    ("perceptionbench", 31, "Combining observations across regions", "Count rounded-rectangle process modules containing bold text.",
     "Inspect the upper half of the flowchart.", "Inspect the lower half in the same turn; combine the counts.", "Final answer: 13  |  Dataset reference: 13"),
    ("hrbench_8k", 132, "Correcting the view to read fine text", "Read the year inscribed on the sundial.",
     "The first crop misses the sundial.", "Reposition the crop to read the inscription.", "Final answer: A (1762)  |  Strict match: correct"),
    ("osworld", 103, "Expanding the view to restore context", "Click cell Q35 in LibreOffice Calc.",
     "The first view lacks column headers.", "A wider view restores the column-header context.", "Correct target: [940.63, 912.04]"),
    ("hrbench_4k", 45, "Resolving a small visual attribute", "Identify the gloves worn by the person in the background.",
     "Inspect the person near the recycling bin.", "Zoom into the first crop to inspect the glove.", "Final answer: B (White gloves)  |  Strict match: correct"),
]


def load_pb(indices):
    found = {}
    with (ROOT / "evaluation/data/perceptionbench/PerceptionBench.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["index"] in indices:
                found[row["index"]] = row
            if len(found) == len(indices):
                break
    return found


def reconstruct(benchmark, index, pb):
    rid = f"{benchmark}-{index:05d}"
    record = json.loads((ROOT / f"benchmark_results/trajectories/{benchmark}/{rid}.json").read_text())["record"]
    trace = record.get("agentic", record.get("trajectory", (record.get("trajectories") or [{}])[0]))
    work = ASSETS / "_work" / rid
    work.mkdir(parents=True, exist_ok=True)
    if benchmark == "perceptionbench":
        encoded = pb[index]["image"][0].split(",", 1)[-1]
        image_path = work / "original.png"
        # Decode the official image bytes without changing the source dataset.
        image_path.write_bytes(base64.b64decode(encoded))
        original_ref = f"dataset:perceptionbench/{index}/image_0"
    else:
        original_ref = record.get("img_path", record.get("image_path"))
        image_path = ROOT / original_ref
    image_path = image_path.resolve()
    with Image.open(image_path) as image:
        width, height = image.size
    registry = {str(image_path): {"path": str(image_path), "width": width, "height": height,
                "original_region": [0, 0, width, height], "original_width": width, "original_height": height}}
    mapping = {original_ref: str(image_path)}
    executor = AgenticGroundingPipeline.__new__(AgenticGroundingPipeline)
    executor.answer_mode = "vqa" if benchmark == "perceptionbench" else ("mcq" if benchmark.startswith("hrbench") else "point")
    executor.crop_upscale_factor = 4.0
    executor.crop_max_pixels = 4_000_000
    executor.crop_max_dimension = 4096
    crops, checks = [], []
    for number, event in enumerate(trace["tool_events"], 1):
        expected = event["result"]
        if not expected.get("ok"):
            continue
        args = dict(event["arguments"])
        args["image_path"] = mapping[args["image_path"]]
        actual = executor._execute_crop(args, registry, work, number)
        assert actual.get("ok"), actual
        for field in ("width", "height", "crop_in_parent", "original_image_region_pixels"):
            assert actual[field] == expected[field], (rid, number, field, actual[field], expected[field])
        mapping[expected["image_path"]] = actual["image_path"]
        crops.append(Path(actual["image_path"]))
        checks.append({"event": number, "dimensions": [actual["width"], actual["height"]],
                       "crop_in_parent": actual["crop_in_parent"],
                       "original_image_region_pixels": actual["original_image_region_pixels"]})
    return image_path, crops, checks, record


BENCHMARK_LABELS = {"sspro": "ScreenSpot-Pro", "osworld": "OS-World-G (Refine)",
                    "perceptionbench": "PerceptionBench", "hrbench_4k": "HR-Bench 4K",
                    "hrbench_8k": "HR-Bench 8K"}


def caption(fig, x, y, text, width, color=MUTED, size=14):
    """Keep captions at a readable size and wrap within their assigned panel."""
    fig.text(x, y, textwrap.fill(text, width), color=color, fontsize=size,
             va="top", linespacing=1.3)


def heading(fig, x, y, label, color=INK):
    fig.text(x, y, label, color=color, fontsize=13.5, weight="bold", va="bottom")


def arrow(fig, start, end, color=MUTED, connectionstyle="arc3"):
    fig.add_artist(FancyArrowPatch(start, end, transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=19, color=color, linewidth=1.9,
        connectionstyle=connectionstyle))


def image_panel(fig, path, bounds, anchor="C"):
    ax = fig.add_axes(bounds, facecolor="white")
    with Image.open(path) as image:
        ax.imshow(image.convert("RGB"), interpolation="antialiased")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_anchor(anchor)
    for spine in ax.spines.values():
        spine.set_color(RULE)
        spine.set_linewidth(1.3)
    return ax


def region_box(ax, box, color, label=None):
    x1, y1, x2, y2 = box
    ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                           edgecolor=color, linewidth=2.3))
    if label:
        ax.annotate(label, (x1, y1), xytext=(2, -3), textcoords="offset points",
                    color="white", fontsize=11, weight="bold", va="top",
                    bbox={"facecolor": color, "edgecolor": "none", "pad": 1.5})


def render(case, pb):
    benchmark, index, title, question, note1, note2, answer = case
    original, crops, checks, record = reconstruct(benchmark, index, pb)
    rid = f"{benchmark}-{index:05d}"
    assert len(crops) >= 2
    parallel = benchmark == "perceptionbench"
    nested = benchmark in ("sspro", "hrbench_4k")
    portrait = (benchmark, index) in (("osworld", 103), ("hrbench_4k", 45))
    apply_style()
    height = 8.5 if parallel else 7.8 if portrait else 6.8
    fig = plt.figure(figsize=(12, height), facecolor="white")
    fig.text(.035, .957, "LBS   /   OBSERVATION TRAJECTORIES", color=CLAY_DARK, fontsize=12, weight="bold")
    fig.text(.965, .957, BENCHMARK_LABELS[benchmark], color=MUTED, fontsize=12, ha="right")
    fig.text(.035, .898, title, color=INK, fontsize=23, weight="bold", fontfamily=SERIF)
    fig.text(.035, .845, question, color=MUTED, fontsize=14)
    fig.add_artist(plt.Line2D([.035, .965], [.815, .815], transform=fig.transFigure,
                             color=RULE, linewidth=1.3))

    if parallel:
        # Two branches are observations from one turn, not successive zooms.
        assert record["final_answer"] == pb[index]["answer"]
        heading(fig, .035, .784, "INPUT  /  Complete flowchart", BLUE_DARK)
        heading(fig, .565, .784, "CROP 1  /  Upper region", CLAY_DARK)
        heading(fig, .565, .465, "CROP 2  /  Lower region", SAGE_DARK)
        axes = [image_panel(fig, original, [.035, .235, .375, .525]),
                image_panel(fig, crops[0], [.565, .495, .40, .265]),
                image_panel(fig, crops[1], [.565, .175, .40, .265])]
        arrow(fig, (.425, .50), (.55, .63), CLAY, "angle,angleA=90,angleB=180,rad=10")
        arrow(fig, (.425, .50), (.55, .31), SAGE_DARK, "angle,angleA=-90,angleB=180,rad=10")
        caption(fig, .035, .205, "Two regions, requested in the same turn. Combine the observations before answering.", 43)
        # Shared branch caption sits away from image content.
        fig.text(.472, .52, "same\nturn", ha="center", color=MUTED, fontsize=13, linespacing=1.3)
    elif benchmark == "sspro":
        # A wide toolbar and its detail crop need different aspect-ratio slots.
        heading(fig, .035, .772, "INPUT  /  Original screenshot", BLUE_DARK)
        heading(fig, .535, .772, "CROP 1  /  Toolbar", CLAY_DARK)
        heading(fig, .535, .460, "CROP 2  /  Detail of Crop 1", SAGE_DARK)
        axes = [image_panel(fig, original, [.035, .315, .425, .425]),
                image_panel(fig, crops[0], [.535, .630, .430, .085]),
                image_panel(fig, crops[1], [.535, .235, .430, .205])]
        arrow(fig, (.477, .675), (.518, .675), CLAY)
        caption(fig, .535, .595, "Inspect the toolbar; select a smaller region inside this crop.", 43)
        arrow(fig, (.75, .505), (.75, .482), SAGE_DARK)
        caption(fig, .035, .255, "Terracotta: Crop 1. Sage: Crop 2. The second observation zooms into the first.", 43)
        # Crop 2's rectangle is expressed in Crop 1's resized pixel coordinates.
        region_box(axes[1], checks[1]["crop_in_parent"], SAGE_DARK)
    elif portrait:
        heading(fig, .035, .784, "INPUT  /  Original image", BLUE_DARK)
        heading(fig, .475, .784, "CROP 1", CLAY_DARK)
        heading(fig, .770, .784, "CROP 2", SAGE_DARK)
        axes = [image_panel(fig, original, [.035, .375, .38, .385]),
                image_panel(fig, crops[0], [.475, .375, .235, .385]),
                image_panel(fig, crops[1], [.770, .205, .195, .555])]
        arrow(fig, (.425, .57), (.46, .57), CLAY)
        arrow(fig, (.722, .57), (.755, .57), SAGE_DARK)
        caption(fig, .035, .315, "Terracotta: Crop 1. Sage: Crop 2. " +
                ("Crop 2 uses Crop 1 as its source." if nested else "Both crops use the original image."), 39)
        caption(fig, .475, .315, note1, 25)
        if nested:
            region_box(axes[1], checks[1]["crop_in_parent"], SAGE_DARK)
            caption(fig, .475, .205, "The enlarged glove remains blurred.", 25)
        else:
            caption(fig, .475, .205, "Crop 2 restores the column headers.", 25)
    else:
        heading(fig, .035, .772, "INPUT  /  Original image", BLUE_DARK)
        heading(fig, .415, .772, "CROP 1  /  Initial view", CLAY_DARK)
        heading(fig, .685, .772, "CROP 2  /  Reframed", SAGE_DARK)
        axes = [image_panel(fig, original, [.035, .325, .335, .410]),
                image_panel(fig, crops[0], [.415, .325, .225, .410]),
                image_panel(fig, crops[1], [.685, .325, .280, .410])]
        arrow(fig, (.38, .53), (.404, .53), CLAY)
        arrow(fig, (.65, .53), (.674, .53), SAGE_DARK)
        caption(fig, .035, .280, "Terracotta: Crop 1. Sage: Crop 2. Both crops use the original image.", 35)
        caption(fig, .415, .280, note1, 24)
        caption(fig, .685, .280, note2, 30)

    for i, check in enumerate(checks[:2]):
        region_box(axes[0], check["original_image_region_pixels"], CLAY if i == 0 else SAGE_DARK, str(i+1))
    if benchmark in ("osworld", "sspro"):
        # Ground-truth box and recorded prediction are evidence overlays only.
        box, point = record["bbox"], record["pred"]
        event = checks[1]
        x1, y1, x2, y2 = event["original_image_region_pixels"]
        w, h = event["dimensions"]
        sx, sy = w/(x2-x1), h/(y2-y1)
        mapped = [(box[0]-x1)*sx, (box[1]-y1)*sy, (box[2]-x1)*sx, (box[3]-y1)*sy]
        region_box(axes[2], mapped, SAGE_DARK)
        axes[2].plot((point[0]-x1)*sx, (point[1]-y1)*sy, marker="+", color=CLAY_DARK, ms=14, mew=2.5)

    fig.add_artist(FancyBboxPatch((.035, .071), .93, .070, transform=fig.transFigure,
        boxstyle="round,pad=0.008,rounding_size=0.008", facecolor=SAGE_BG, edgecolor=RULE, linewidth=1, zorder=-1))
    if benchmark in ("osworld", "sspro"):
        answer = answer.replace("Correct target:", "Correct click:")
        fig.text(.055, .106, answer, color=SAGE_DARK, fontsize=15, weight="bold", va="center")
        fig.text(.945, .106, "Sage box: reference   +: prediction", color=MUTED, fontsize=11.5,
                 ha="right", va="center")
    else:
        fig.text(.055, .106, answer, color=SAGE_DARK, fontsize=15, weight="bold", va="center")
    fig.text(.035, .026, f"{rid}  ·  Official image; recorded crop geometry. Captions summarize the trace.",
             color=MUTED, fontsize=10.5)
    out = ASSETS / f"{rid}.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return {"release_id": rid, "figure": str(out.relative_to(ROOT)), "crop_checks": checks,
            "recorded_outcome": "correct", "visual_review_notes": "docs/TRAJECTORIES.md",
            "layout": "parallel_regions" if parallel else "nested_toolbar" if benchmark == "sspro"
                      else "portrait_detail" if portrait else "sequential_reframing"}


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    pb = load_pb({case[1] for case in CASES if case[0] == "perceptionbench"})
    checks = [render(case, pb) for case in CASES]
    (ASSETS / "reconstruction_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(f"Rendered {len(checks)} figures; all recorded crop geometry checks passed.")


if __name__ == "__main__":
    main()
