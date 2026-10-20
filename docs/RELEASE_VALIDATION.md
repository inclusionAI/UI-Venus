# Offline validation

Run from the repository root after installing `evaluation/requirements.txt`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
python3 scripts/validate_release.py
python3 scripts/render_metric_tables.py --check
python3 scripts/export_trajectory_html.py --check
```

These checks validate recorded artifacts and local behavior without training,
model inference, or judge requests. The runtime tests use local model stubs and
require no GPU or benchmark downloads; one test requires local multiprocessing.
The release validator also checks referenced benchmark image paths, so download
benchmark data before running it. Table and HTML checks use the shipped assets.

## Coverage

| Check | Scope |
|---|---|
| Runtime tests | 24 tests covering completion, failed workers, resume, task coverage, OS-World-G merging, PB shard discovery, cleanup, and wrapper configuration. |
| Release artifacts | 6745 per-item scores, 100 selected trajectories, category/outcome balance, distinct HR-Bench groups, crop references, 8 training inputs, and 32 rollouts. |
| Syntax and links | Python compilation, shell syntax, and local Markdown targets. |
| Publication metadata | Targeted checks for machine paths, private deployment aliases, and release placeholders; not an exhaustive secret audit. |
| Metric tables | Saved SSPro and PerceptionBench scores, category totals, and English README tables. |
| HTML export | Six source records, saved response text, crop metadata, and 18 image hashes. |

## Scoring replay

To compare published metrics with complete evaluator outputs and replay scoring:

```bash
python3 scripts/validate_release.py --original-results /path/to/full/run
```

The source directory must contain `sspro.json`, `osworld.json`, `hrbench_4k.json`,
`hrbench_8k.json`, `hrbench.json`, and `perceptionbench_scores.json`, with the full
per-item records in the applicable reports. The check compares saved metrics,
replays current grounding and HR-Bench scoring, and runs the PerceptionBench
finalizer and OS-World-G merger in a temporary directory. Source reports remain
read-only. HR-Bench replay allows an absolute floating-point tolerance of `1e-12`.

Reports are written to the ignored `_release_archive/validation/` directory by
default; use `--report PATH` to choose another location.

Published HR-Bench scores use `splits.HRBench4K.strict_accuracy` and
`splits.HRBench8K.strict_accuracy` separately. PerceptionBench uses the saved
self-hosted Deepseekv4-flash-0731 judgments and retains failure counts; replay
does not perform new judgments. See [judge protocols](JUDGE_PROTOCOL.md),
[trajectory reconstruction](TRAJECTORIES.md), and the evaluation
[known limitations](../evaluation/README.md#known-limitations).
