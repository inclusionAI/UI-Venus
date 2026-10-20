# Evaluation architecture

This component isolates dataset preparation, model interaction, scoring, and
orchestration. See [README.md](README.md) for commands and [AGENTS.md](../AGENTS.md)
for contribution boundaries.

## Modules

| Module | Responsibility |
|---|---|
| `common/agentic_pipeline.py` | HTTP/SSE text-action loop, crop registry, image-coordinate mapping, retries, and trace records |
| `sspro/sspro_eval_common.py` | Task expansion, positive/refusal judgments, sample keys, and all grounding metric breakdowns |
| `sspro/eval_screenspot_pro_agentic.py` | Pipeline adaptation, thread/process scheduling, multiple samples, and atomic checkpoints |
| `perceptionbench/models/agentic_helpers.py` | Shared parsing, encoding, and rate-limit helpers for the VQA entry |
| `perceptionbench/eval_perceptionbench_agentic.py` | VQA conversations, indexed shards, agentic judge, and optional judge audits |
| `perceptionbench/eval/eval.py` | Judge decoding and normalization; loads `judge_prompt.txt` at import time |
| `perceptionbench/debug/rejudge_agentic.py` | Backup and retry selected judge statuses using the same agentic template |
| `hrbench/eval_hrbench_agentic.py` | MCQ tasks, four option rotations, strict scores, optional judge, resume, and split summaries |
| `scripts/config.sh` | Root discovery, defaults, environment loading, stage guards, and intermediate cleanup |
| `scripts/prepare_data.py` | Official data conversion without changing task semantics |
| `scripts/verify_data.py` | Byte-level files and content-level HR-Bench manifest checks |
| `scripts/finalize_perceptionbench.py` | Shard deduplication, aggregate/category scores, merged predictions, and model-card metadata |
| `scripts/merge_osworld_results.py` | Shared grounding aggregation plus data-type and refusal diagnostics |

## Data flow

```text
Official benchmark sources
    -> fetch_data.sh -> prepare_data.py -> data/ -> verify_data.py
                                                  |
.env.run -> run_*.sh -> evaluator -> OpenAI-compatible model endpoint
                           |              |
                           |       textual crop_image actions
                           |              |
                           |       local image crop + feedback
                           v
                   checkpoints / progress / shard JSONL
                           |
                   scoring and aggregation
                           v
                      results/run-name/
                           |
             offline scripts/export_release.py (repository root)
                           v
                  benchmark_results/ (publication only)
```

SSPro and OS-World-G share the point-grounding entry. OS-World-G evaluates 510
positive tasks and 54 refusal tasks separately, then joins `data_type` using image
basename plus instruction. Polygon targets use the existing enclosing-box rule.

PerceptionBench shards by record position modulo shard count and resumes by index.
Its agentic judge grades the final extracted answer. The standard and zoom-in prompt
files are both required; attribution is documented outside their runtime contents.

HR-Bench validates four option rotations per question group. Strict scoring and
optional LLM judgments are aggregated separately by cycle and category. The combined
summary stores each split under `splits`; publication uses each split's
`strict_accuracy`. Existing `primary_metric`, judge scores, and averages remain
available for compatibility and audit.

## Behavioral invariants

The text-action protocol, coordinate systems, crop budgets and resizing, prompt
variants, and scoring functions are unchanged by publication export. Nested crops
retain a mapping to the original image. Direct SSPro prompts and the shared
`grounding` variant are distinct and must not be interchanged as an editorial fix.

Model reasoning options are forwarded through `chat_template_kwargs`. Checkpoint
files are written atomically where supported by the entry; wrapper stage guards
avoid repeating completed runs but are not complete configuration fingerprints.
PerceptionBench finalization consumes shards, not an independently rejudged merged
file. Preserve shards when planning post-hoc judging.

## Portability and release separation

Wrappers derive the evaluation root from their own file locations. All shipped
benchmark references use the prepared local data layout. The shared pipeline's
standalone default crop location is `evaluation/results/crops`; explicit caller
paths retain precedence. Training and scoring do not depend on the outer folder name.

Publication artifacts keep all original numeric metrics and all per-item scores,
while selecting only a documented subset of complete text trajectories. They have
separate metadata and must not be fed into a runner's completion/resume checks.
