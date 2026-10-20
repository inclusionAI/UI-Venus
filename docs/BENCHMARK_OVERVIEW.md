# README benchmark overview

The [overview figure](assets/benchmark_overview/benchmark_overview.svg) compares
overall ScreenSpot-Pro and PerceptionBench scores. Each panel uses a zero-based
percentage axis; LBS is highlighted in terracotta. Shared styling is defined in
[figure_style.py](../scripts/figure_style.py).

## Scores and comparison scope

LBS scores come from the complete recorded evaluation:

| Benchmark | Saved field | Overall (%) | Records |
|---|---|---:|---:|
| ScreenSpot-Pro | `benchmark_results/metrics/sspro.json`: `metrics.overall.action_acc` | 81.15 | 1581 |
| PerceptionBench | `benchmark_results/metrics/perceptionbench.json`: `overall` | 59.67 | 3000 |

Scores are weighted by record count. PerceptionBench retains the saved report's
precision and has zero failed or skipped judgments. No predictions or judgments
are rerun to generate the figure.

Reference scores use the [ScreenSpot-Pro leaderboard](https://gui-agent.github.io/grounding-leaderboard/)
and [PerceptionBench leaderboard](https://github.com/MoonshotAI/PerceptionBench#leaderboard),
with a snapshot date of **2026-10-15**. Three additional ScreenSpot-Pro zoom-in
scores were supplied by the repository author:

| Model | Setup | Overall (%) | Source |
|---|---|---:|---|
| Qwen-UI-Agent-27B | Zoom-in | 81.5 | Author-reported |
| Qwen 3.7-Plus | Zoom-in | 80.7 | Author-reported |
| Seed 2.1 Pro | Zoom-in | 80.7 | Author-reported |

UI-Venus2-27B's **74.1%** comes from the
[UI-Venus-2 grounding table](https://github.com/inclusionAI/UI-Venus/blob/UI-Venus-2/README.md).
[references.json](assets/benchmark_overview/references.json) records model labels,
values, precision, source URLs, and metric fields. ScreenSpot-Pro leaderboard
values use `results.overall.avg`; PerceptionBench references use the published
Overall column. Asterisks identify agentic / zoom-in inference.

These are selected reported results with different model sizes and inference
setups. **PerceptionBench judges also differ:** LBS uses self-hosted
Deepseekv4-flash-0731; references use GPT-oss-120B. The chart is not a controlled
comparison under a shared judge, and 59.67% does not establish an improvement over
a reported 59.7%. See [judge protocols](JUDGE_PROTOCOL.md).

## Regeneration

Install Matplotlib and run from the repository root:

```bash
python3 scripts/render_benchmark_overview.py
python3 scripts/render_metric_tables.py --check
```

The renderer uses saved metrics and reference values to write a 3200 × 1560 PNG
and an SVG. It makes no network or model requests. Update reference values,
sources, and snapshot dates together, then regenerate and inspect the figure.
