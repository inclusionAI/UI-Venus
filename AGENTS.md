# Contributor and agent guide

This repository accompanies **Learning to Look Beyond Screens: Transferable Visual
Observation from GUI Grounding**. Read the root README and the relevant component
README before making changes.

## Scope and interfaces

- `agentic_gui_grounding_rl/`: GRPO training, rewards, scheduler, and crop protocol.
- `evaluation/`: standalone benchmark clients, scoring, data preparation, and runners.
- `benchmark_results/`: publication artifacts; never use these as resumable run outputs.
- `scripts/`: offline release export, validation, and figure generation.
- `docs/`: protocol notes and figures reconstructed from recorded trajectories.

Keep Python module names, reward registrations, CLI flags, environment keys, and
runtime result schemas compatible. Use lowercase snake_case for Python names and
directories, UPPER_SNAKE_CASE for environment keys, and English contributor docs.

## Behavior that must be preserved

Do not change training rewards, loss shaping, sampling, random seeds, action parsing,
crop geometry/resizing, coordinate mapping, model prompts, judge prompts, scoring,
or checkpoint/resume behavior as part of documentation or release cleanup.
The Chinese training CoT judge prompt is intentional; its English explanation is
documentation only. Attribution must not be appended to runtime prompt templates.

Keep benchmark data read-only. Preserve data verification manifests. Read-only
validation may reconstruct crops into a separate output directory.

The wrappers use `reasoning_effort=xhigh` and `thinking_budget=4096` by default;
reasoning options are sent through `chat_template_kwargs`. Direct Python entry
defaults can differ; document those differences rather than silently changing them.
Use the actual served model ID from `/v1/models`. Configure endpoints and credentials
through an ignored `.env.run` copied from `evaluation/.env.example`.

## Reporting and verification

Report HR-Bench 4K and 8K separately using `splits.HRBench4K.strict_accuracy` and
`splits.HRBench8K.strict_accuracy` in the combined report. Retain other recorded
metrics, but do not present LLM judge accuracy as the paper's HR-Bench result.
PerceptionBench results use a self-hosted Deepseekv4-flash-0731 judge. Record judge
configuration and failures; repeated judgments can vary.

When editing score aggregation, replay existing records and compare every metric.
When editing runtime code, test the affected behavior without requiring live models
where possible. Never claim a full training/evaluation run from offline checks.
Keep runtime prompts byte-identical for editorial changes; compare executable
syntax independently of comments and docstrings.

## Operations and release hygiene

Resume by repeating the same command and configuration. Preserve SSPro checkpoints,
PerceptionBench shards, and HR-Bench progress files until completion. Existing runner
cleanup is controlled by `CLEAN_INTERMEDIATES`; use `0` to retain those artifacts.
A failed grounding worker or incomplete coverage must not publish a final report.
Keep explicit smoke/full completion metadata; do not treat legacy unmarked reports
as complete. PB producers and finalizers must use the shared artifacts resolver.
PerceptionBench finalization independently deletes shards: use `FINALIZE=0` before
manual finalization with `--keep-shards`. Concurrency must fit endpoint
capacity; `PB_NUM_SHARDS * PB_SHARD_CONCURRENCY` is the aggregate concurrency.

Do not commit credentials, downloaded datasets, raw run directories, or checkpoints.
Preserve shipped training samples and curated publication artifacts. During the
release cleanup, move removed files into `_release_archive/`; do not delete that
archive. Exclude it from Git and distribution archives. Keep third-party attribution
and dataset terms distinct from the repository's software license.
