# Recorded benchmark results

This publication export retains all numeric aggregate/fine-grained metrics and
all **6745** per-item scores from one complete evaluation run. It includes **100**
selected complete text trajectories; it is not a resumable runtime output directory.

| Benchmark | Evaluated records | Selected trajectories | Headline |
|---|---:|---:|---:|
| ScreenSpot-Pro | 1581 | 20 | 81.151170% action accuracy |
| OS-World-G (Refine) | 564 | 20 | 84.574468% action accuracy |
| PerceptionBench | 3000 | 40 | 59.67% judge accuracy (stored precision) |
| HR-Bench 4K | 800 | 10 | 89.125% strict accuracy |
| HR-Bench 8K | 800 | 10 | 87.750% strict accuracy |

The OS-World-G result uses the **Refine** annotations, including 470 bounding-box,
40 polygon, and 54 refusal records. Read the [SSPro domain/application tables](../README.md#screenspot-pro-domains-and-target-types)
and [PerceptionBench category table](../README.md#perceptionbench-ten-perception-categories)
in the root README. Those tables use the full per-item scores and saved metrics,
independently of trajectory selection.

HR-Bench contributes 20 selected trajectories across its two splits.
PerceptionBench has four selected records in each of its ten categories, including
two successful and two unsuccessful judgments per category. Other subsets cover
successful/failed cases and varied observation behavior. HR-Bench selections avoid
repeated `group_index` values within a split. Selection rules and every selected
record are listed in [manifest.json](manifest.json).

## Files and interpretation

- `metrics/`: the original metric trees, configuration with privacy edits, and
  the combined HR-Bench report. All numeric metrics are preserved, including
  non-headline judge metrics, diagnostics, category scores, and rotation scores.
- `per_sample/`: every record's scoring inputs/outputs and compact trajectory
  statistics; long conversations are omitted here.
- `trajectories/`: one JSON file per selected complete text trajectory, with a
  stable release ID, source row, selection tags, and the sanitized original record.
- `headline.json`: the publication metrics, explicitly selecting strict HR-Bench scores.

`released-checkpoint` is an alias, not a public endpoint/model ID. The PerceptionBench
judge is identified as self-hosted **Deepseekv4-flash-0731**; the internal deployment
suffix and endpoint address are omitted. Judgments may vary across repeated runs.
The saved PerceptionBench report has no failed/skipped judgments.

The original run used HR-Bench's optional judge as well as strict scoring. Its
`primary_metric` and associated fields remain unchanged for audit. **Publication
uses strict option matching separately for 4K and 8K**, read from
`metrics/hrbench.json` at `splits.HRBench4K.strict_accuracy` and
`splits.HRBench8K.strict_accuracy`. Never substitute `average.primary_accuracy`.

Dataset image references start with `evaluation/data/` and resolve from the repository
root after downloading the data. `dataset:perceptionbench/<index>/image_<n>` refers
to an image embedded in the official JSONL record. `crops/<release-id>/...` is a
logical reference to a recorded crop; those intermediate files are not all shipped.
The [figure renderer](../scripts/render_trajectories.py) reconstructs displayed
cases from official images. Response IDs, timestamps, private paths, endpoint
addresses, and exception tracebacks are removed or replaced; numeric answers,
coordinates, verdicts, and reasoning content are preserved apart from path/model
redactions. Figures are reconstructions, not exact archived API JPEG payloads.

## Regeneration

```bash
python3 scripts/export_release.py --source /path/to/full/run --output /path/to/new/export
```

Run from the repository root. The exporter is offline and does not invoke models,
change scores, or modify the source. Curated IDs apply to this recorded run; inspect
and update the selection specification before applying it to a different run.

Sampled trajectories illustrate behavior and limitations. Aggregate scores always
refer to the full counts above, not to the curated subset. See
[verified visual cases](../docs/TRAJECTORIES.md) and the [validation report](../docs/RELEASE_VALIDATION.md).

## Benchmark display scope

Benchmark excerpts included here are used solely to display recorded evaluation
results and observation trajectories, with no other use intended. Their original
ownership, licenses, and terms remain unchanged. The Apache-2.0 software license
does not relicense these materials. Source attribution is retained in
[NOTICE](../NOTICE).
