#!/usr/bin/env python3
"""Render the README overview from saved overall scores and frozen references.

Requires matplotlib. Run from any directory; no network, model, judge, or dataset
access is needed. External references and their provenance are kept beside the
figure. The released metrics are read without changing their stored precision.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lbs-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

from figure_style import (INK, MUTED, RULE, REFERENCE_COLORS, CLAY, CLAY_DARK,
                          SERIF, apply_style)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets/benchmark_overview"


def load_scores():
    ss = json.loads((ROOT / "benchmark_results/metrics/sspro.json").read_text())
    pb = json.loads((ROOT / "benchmark_results/metrics/perceptionbench.json").read_text())
    if ss["metrics"]["overall"]["num_total"] != 1581 or pb["counts"]["total"] != 3000:
        raise ValueError("The overview expects the complete released benchmark runs")
    if pb["counts"]["judge_failures"] or pb["counts"]["judge_skipped"]:
        raise ValueError("Report failed/skipped judgments before publishing the overview")
    if ss["run_config"]["prompt_mode"] != "agentic" or pb["config"]["prompt_mode"] != "agentic":
        raise ValueError("The legend expects the recorded agentic evaluation setup")
    if pb["config"]["judge_model"] != "Deepseekv4-flash-0731":
        raise ValueError("Update the figure's judge disclosure for this run")
    return {"sspro": 100 * ss["metrics"]["overall"]["action_acc"],
            "perceptionbench": 100 * pb["overall"]}


def draw_panel(ax, references, score, title, subtitle, upper):
    values = [row["score_percent"] for row in references] + [score]
    labels = [row["label"] for row in references] + ["LBS\n(ours)"]
    last = len(values) - 1
    # Keep zero baselines and a visible gap before our separately measured result.
    positions = list(range(last)) + [last + 0.35]
    bars = ax.bar(positions, values, width=0.65,
                  color=REFERENCE_COLORS[:last] + [CLAY],
                  edgecolor=[MUTED] * last + [CLAY_DARK],
                  linewidth=1.6, zorder=3)
    bars[-1].set_linewidth(2.8)
    for index, (bar, value) in enumerate(zip(bars, values)):
        # Upstream PB references have only one decimal of published precision.
        precision = (references[index].get("precision", 1 if "PerceptionBench" in title else 2)
                     if index != last else 2)
        ax.annotate(f"{value:.{precision}f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=17, weight="bold" if index == last else "normal",
                    color=CLAY_DARK if index == last else INK)
    ax.set_ylim(0, upper)
    ax.set_xlim(-0.65, positions[-1] + 0.7)
    ax.set_yticks(range(0, upper + 1, 20))
    ax.set_ylabel("Overall accuracy (%)", fontsize=16, labelpad=9)
    ax.set_xticks(positions, labels, fontsize=15.5)
    ax.get_xticklabels()[-1].set_color(CLAY_DARK)
    ax.get_xticklabels()[-1].set_weight("bold")
    ax.tick_params(axis="x", length=0, pad=11)
    ax.tick_params(axis="y", length=4.5, width=1.5, labelsize=14, pad=5)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=RULE, linewidth=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(1.8)
    ax.text(0, 1.17, title, transform=ax.transAxes, fontsize=23,
            weight="bold", fontfamily=SERIF, color=INK)
    ax.text(0, 1.065, subtitle, transform=ax.transAxes, fontsize=14, color=MUTED)


def main():
    references = json.loads((ASSETS / "references.json").read_text())
    scores = load_scores()
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.8), facecolor="white")
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.36, top=0.77, wspace=0.18)
    fig.text(0.5, 0.965, "Learning to look beyond screens", ha="center",
             va="top", fontsize=27, fontfamily=SERIF, weight="bold")
    draw_panel(axes[0], references["sspro"]["references"], scores["sspro"],
               "(a)  ScreenSpot-Pro", "GUI grounding · action accuracy · 1,581 tasks", 100)
    draw_panel(axes[1], references["perceptionbench"]["references"], scores["perceptionbench"],
               "(b)  PerceptionBench", "Visual perception · judge accuracy · 3,000 questions", 80)
    fig.legend(handles=[tuple(Patch(facecolor=color, edgecolor=MUTED, linewidth=1)
                               for color in REFERENCE_COLORS[:3]),
                        Patch(facecolor=CLAY, edgecolor=CLAY_DARK, linewidth=2.8)],
               labels=["Reference models", "LBS (ours; agentic zoom-in)"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)},
               loc="lower center", bbox_to_anchor=(0.5, 0.175), ncol=2,
               frameon=False, fontsize=15, handlelength=2.5, columnspacing=3)
    fig.text(0.5, 0.125, "SSPro: * denotes agentic / zoom-in inference. Reference scores: technical reports and the SSPro leaderboard.",
             ha="center", fontsize=12, color=MUTED)
    fig.text(0.5, 0.081, "PerceptionBench judges: LBS, Deepseekv4-flash-0731; published references, GPT-oss-120B.",
             ha="center", fontsize=12, color=MUTED)
    fig.text(0.5, 0.037,
             "LBS learns visual observation that generalizes beyond GUIs using only GUI grounding data\n"
             "in both cold-start and RL training, with no general-domain data or VQA questions.",
             ha="center", va="center", fontsize=12, linespacing=1.4, color=MUTED)
    for extension in ("svg", "png"):
        output = ASSETS / f"benchmark_overview.{extension}"
        fig.savefig(output, dpi=200, facecolor="white", metadata={"Creator": "LBS offline figure renderer"})
        print(output.relative_to(ROOT))
    plt.close(fig)
    print(f"Saved overall scores: SSPro {scores['sspro']:.2f}%; PerceptionBench {scores['perceptionbench']:.2f}%")


if __name__ == "__main__":
    main()
