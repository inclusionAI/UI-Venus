#!/usr/bin/env python3
"""Render README tables from full saved scores, checking every displayed metric.

No evaluator, judge, or dataset is changed. Use --check to verify the tables without
writing the README. Run from any directory with the Python standard library.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- BEGIN RECORDED CATEGORY METRICS -->"
END = "<!-- END RECORDED CATEGORY METRICS -->"

# Display order and terminology follow the linked benchmark pages. Metric keys
# remain those of the saved evaluation reports.
SS_GROUPS = [
    ("Dev", "Development and Programming", [
        ("vscode", "Visual Studio Code"), ("pycharm", "PyCharm"),
        ("android_studio", "Android Studio"), ("quartus", "Quartus"), ("vmware", "VMware")]),
    ("Creative", "Creative", [
        ("photoshop", "Photoshop"), ("premiere", "Premiere"),
        ("illustrator", "Illustrator"), ("blender", "Blender"),
        ("fruitloops", "FL Studio"), ("unreal_engine", "Unreal Engine"), ("davinci", "DaVinci Resolve")]),
    ("CAD", "CAD and Engineering", [
        ("autocad", "AutoCAD"), ("solidworks", "SolidWorks"),
        ("inventor", "Inventor"), ("vivado", "Vivado")]),
    ("Scientific", "Scientific and Analytical", [
        ("matlab", "MATLAB"), ("origin", "Origin"), ("stata", "Stata"), ("eviews", "EViews")]),
    ("Office", "Office Suite", [("word", "Word"), ("powerpoint", "PowerPoint"), ("excel", "Excel")]),
    ("OS", "Operating System Commons", [
        ("windows_common", "Windows"), ("macos_common", "macOS"), ("linux_common", "Linux")]),
]
PB_CATEGORIES = [
    ("visual_relation_error", "Visual relation", "VRel"),
    ("visual_counting_error", "Counting", "Count"),
    ("visual_attribute_error", "Attribute", "Attr"),
    ("depth_3d_perception_error", "Depth and 3D perception", "Depth"),
    ("visual_localization_error", "Localization", "Loc"),
    ("visual_comparison_error", "Comparison", "Comp"),
    ("fine_grained_recognition_error", "Fine-grained recognition", "FGR"),
    ("context_integration_error", "Contextual integration", "Ctx"),
    ("ocr_error", "Optical character recognition", "OCR"),
    ("hallucination", "Perception-related hallucination", "Hallu"),
]


def load_records(benchmark):
    path = ROOT / f"benchmark_results/per_sample/{benchmark}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def ss_values(rows, saved):
    correct = sum(r["correctness"] == "correct" for r in rows)
    assert (correct, len(rows)) == (saved["num_correct_action"], saved["num_total"])
    assert math.isclose(correct / len(rows), saved["action_acc"], abs_tol=1e-12)
    counts = Counter(r["ui_type"] for r in rows)
    for ui in ("text", "icon"):
        passed = sum(r["correctness"] == "correct" for r in rows if r["ui_type"] == ui)
        assert math.isclose(passed / counts[ui], saved[f"{ui}_acc"], abs_tol=1e-12)
    return (f"{len(rows)} | {counts['text']} / {counts['icon']} | "
            f"{100 * saved['text_acc']:.2f} | {100 * saved['icon_acc']:.2f} | "
            f"{100 * saved['action_acc']:.2f}")


def build_tables():
    ss = json.loads((ROOT / "benchmark_results/metrics/sspro.json").read_text())["metrics"]
    pb = json.loads((ROOT / "benchmark_results/metrics/perceptionbench.json").read_text())
    ss_rows, pb_rows = load_records("sspro"), load_records("perceptionbench")
    assert {r["group"] for r in ss_rows} == {g for g, _, _ in SS_GROUPS}
    assert {r["application"] for r in ss_rows} == {a for _, _, apps in SS_GROUPS for a, _ in apps}
    assert {r["error_category"] for r in pb_rows} == {c for c, _, _ in PB_CATEGORIES}
    lines = [START, "", "### ScreenSpot-Pro: domains and target types", "",
        "The six domains and 26 applications follow the [ScreenSpot-Pro grounding",
        "leaderboard](https://gui-agent.github.io/grounding-leaderboard/). All values are",
        "**action accuracy (%)**, computed over the complete 1581-record run. Text and",
        "Icon use their respective target-type denominators; Overall is a **micro-average**",
        "over records, rather than an average of domain or application scores. The",
        "`Text / Icon` column gives the number of records of each type.", "",
        "| Domain | Records | Text / Icon | Text (%) | Icon (%) | Overall (%) |",
        "|---|---:|---:|---:|---:|---:|"]
    app_lines = ["", "<details>", "<summary>All 26 applications: sample counts and Text / Icon accuracy</summary>", "",
        "| Domain | Application | Records | Text / Icon | Text (%) | Icon (%) | Overall (%) |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for group, label, apps in SS_GROUPS:
        rows = [r for r in ss_rows if r["group"] == group]
        assert {r["application"] for r in rows} == {a for a, _ in apps}
        lines.append(f"| {label} | {ss_values(rows, ss['leaderboard_simple_style']['group:' + group])} |")
        for app, app_label in apps:
            rows = [r for r in ss_rows if r["application"] == app]
            app_lines.append(f"| {label} | {app_label} | {ss_values(rows, ss['leaderboard_detailed_style']['app:' + app])} |")
    lines.append(f"| **All domains** | {ss_values(ss_rows, ss['overall'])} |")
    lines += app_lines + ["", "</details>", "",
        "Source: [saved SSPro metrics](benchmark_results/metrics/sspro.json) and",
        "[all per-item scores](benchmark_results/per_sample/sspro.jsonl).", "",
        "### PerceptionBench: ten perception categories", "",
        "Category names, abbreviations, and order follow the [official PerceptionBench",
        "leaderboard](https://github.com/MoonshotAI/PerceptionBench#leaderboard).",
        "Scores below use our **self-hosted Deepseekv4-flash-0731** judge; the upstream",
        "leaderboard uses GPT-oss-120B, so the judge configurations differ. LLM judgments",
        "may vary across runs. These are the saved `judge_result` values used by the",
        "published report; optional second judgments do not replace them.", "",
        "| Category | Abbreviation | Records | Correct | Accuracy (%) |",
        "|---|:---:|---:|---:|---:|"]
    for category, label, abbreviation in PB_CATEGORIES:
        rows = [r for r in pb_rows if r["error_category"] == category]
        correct = sum(r["judge_result"] == 1 for r in rows)
        assert round(correct / len(rows), 4) == pb["per_category"][category]
        lines.append(f"| {label} | {abbreviation} | {len(rows)} | {correct} | {100 * pb['per_category'][category]:.2f} |")
    correct = sum(r["judge_result"] == 1 for r in pb_rows)
    assert (len(pb_rows), correct) == (pb["counts"]["total"], pb["counts"]["correct"])
    assert all(r["judge_status"] == "ok" for r in pb_rows)
    assert round(correct / len(pb_rows), 4) == pb["overall"]
    lines += [f"| **Overall** | — | **{len(pb_rows)}** | **{correct}** | **{100 * pb['overall']:.2f}** |", "",
        "Overall is weighted by record count. There are **zero failed or skipped",
        "judgments**, so all 3000 records contribute. Percentages retain the precision",
        "of the saved report (accuracy rounded to four decimal places before conversion",
        "to percent). Source: [category metrics](benchmark_results/metrics/perceptionbench.json)",
        "and [all per-item scores](benchmark_results/per_sample/perceptionbench.jsonl).", "", END]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check existing README tables without writing")
    args = parser.parse_args()
    path = ROOT / "README.md"
    before = path.read_text()
    table = build_tables()
    if START in before:
        start, end = before.index(START), before.index(END) + len(END)
        after = before[:start] + table + before[end:]
    else:
        anchor = "## Observation trajectories"
        assert anchor in before
        after = before.replace(anchor, table + "\n\n" + anchor, 1)
    if args.check:
        assert before == after, "README category tables differ from the saved metrics"
    else:
        path.write_text(after)
    print("Verified 6 SSPro domains, 26 applications, 10 PerceptionBench categories, and both overall scores.")


if __name__ == "__main__":
    main()
