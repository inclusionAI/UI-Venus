# LBS Evaluation

Standalone evaluation for **Learning to Look Beyond Screens: Transferable Visual
Observation from GUI Grounding**. Models are served through an OpenAI-compatible
endpoint; benchmark data is fetched separately from official sources. The LBS
model is based on **Qwen3.8-27b**, trained with **SFT** followed by
**Agentic RL (GRPO)**. For LBS evaluation, serve the checkpoint after both stages;
`/path/to/checkpoint` in the examples denotes that local checkpoint directory.

## Benchmarks and reported metrics

| Benchmark | Evaluated records | Published metric | Runtime report field |
|---|---:|---|---|
| ScreenSpot-Pro | 1581 | Point-in-box action accuracy | `sspro.json`: `metrics.overall.action_acc` |
| OS-World-G (Refine) | 564 (510 positive + 54 refusal) | Action accuracy, with per-type/refusal breakdowns | `osworld.json`: `metrics.overall.action_acc` |
| PerceptionBench | 3000 | Judge accuracy, excluding failed/skipped judgments | `perceptionbench_scores.json`: `overall` |
| HR-Bench 4K | 800 | **Strict option matching** | `hrbench.json`: `splits.HRBench4K.strict_accuracy` |
| HR-Bench 8K | 800 | **Strict option matching** | `hrbench.json`: `splits.HRBench8K.strict_accuracy` |

The evaluated OS-World-G variant is **Refine**, using
`os-world-g-refine/annotation.json` (470 bounding-box, 40 polygon, 54 refusal
records). Positive and refusal subsets are merged into the 564-record headline.
The root README includes [SSPro domain/application tables](../README.md#screenspot-pro-domains-and-target-types)
and [PerceptionBench category scores](../README.md#perceptionbench-ten-perception-categories)
from the complete recorded run.

HR-Bench has 200 question groups per split, each evaluated under four option
rotations. We report the existing strict metric separately for 4K and 8K, **not
LLM-as-a-judge accuracy**. Strict scoring compares the option parsed by the
unchanged evaluator with the reference option; it is not a raw-response byte
comparison. A split report stores this metric at
`metrics.strict.average.all.accuracy`. The evaluator also retains optional judge
metrics and its existing `primary_metric` field for compatibility; those fields
are not the HR-Bench publication headline. The combined report has `splits` and
`average`, not a top-level `metrics` field.

The released PerceptionBench run used a **self-hosted Deepseekv4-flash-0731** judge.
LLM judging introduces randomness and serving variation, so repeated judgments
may differ. Keep judge configuration fixed when comparing runs and report failures.
`overall_judge_failures_as_wrong` uses all evaluated records as its denominator;
all category scores likewise include failed judgments in their denominators.

## Layout

```text
evaluation/
├── common/                 # Shared SSPro / OS-World-G (Refine) / HR-Bench pipeline
├── sspro/                  # Task construction, grounding, and aggregation
├── perceptionbench/        # VQA pipeline, judge templates, and rejudging
├── hrbench/                # MCQ rotations, strict scores, optional judge
├── scripts/                # Data preparation, serving, wrappers, aggregation
├── data/                   # Downloaded official data (ignored by Git)
├── results/                # Local runs (ignored by Git)
├── .env.example
├── requirements.txt
└── ARCHITECTURE.md
```

The repository-level [AGENTS.md](../AGENTS.md) documents contributor boundaries.
Curated publication artifacts live in [benchmark_results](../benchmark_results/),
separate from resumable runtime results.

## Installation and data

Use Python 3.10 or newer. The evaluation client requires no training framework;
serving a model requires a separately installed vLLM environment and suitable GPUs.

```bash
cd evaluation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -U 'huggingface_hub[cli]'
bash scripts/fetch_data.sh all
python3 scripts/verify_data.py
```

| Dataset | Official source | Prepared location |
|---|---|---|
| ScreenSpot-Pro | [likaixin/ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) | `data/sspro/Screenspot-pro/` |
| OS-World-G (Refine) | [xlang-ai/OSWorld-G](https://github.com/xlang-ai/OSWorld-G) | `data/osworld_g/` |
| PerceptionBench | [moonshotai/PerceptionBench](https://huggingface.co/datasets/moonshotai/PerceptionBench) | `data/perceptionbench/PerceptionBench.jsonl` |
| HR-Bench | [DreamMr/HR-Bench](https://huggingface.co/datasets/DreamMr/HR-Bench) | `data/hrbench/` |

Raw downloads require approximately 11 GB; allow additional space for prepared data
and reconstructed crops.
The script supports individual dataset names and `verify`. `HF_ENDPOINT` selects
the Hugging Face endpoint; `SRC_DIR` selects the raw download cache. Repeating the
command reuses completed downloads. Dataset licenses remain those of their owners.

Verification checks 2211 files byte-for-byte and two HR-Bench manifests by task
content and source size. The manifests use paths relative to the evaluation root,
such as `data/hrbench/images/<image-id>.jpg`. Copying a runtime cache with absolute
paths from another machine requires relocating those image references; do not
change questions, options, annotations, or verification references to hide a mismatch.

## Serve and configure

```bash
# Four-GPU example; set both topology arguments explicitly.
bash scripts/serve_vllm.sh start /path/to/checkpoint --name evaluated-model --tp 4 --dp 1 --daemon
bash scripts/serve_vllm.sh status --name evaluated-model
# Stop a specific instance when finished:
# bash scripts/serve_vllm.sh stop --name evaluated-model
```

The serving script defaults to TP=2 and DP=8 (16 devices), context length 65536,
and 64 sequences per replica. Override these for your hardware. It writes
`.env.vllm.evaluated-model`; alternatively configure an existing endpoint:

```bash
cp .env.example .env.run
# Edit OPENAI_API_KEY, OPENAI_BASE_URL, and MODEL in .env.run.
# Set JUDGE_API_KEY, JUDGE_BASE_URL, and JUDGE_MODEL for PerceptionBench.
```

`MODEL` must match `/v1/models`. A judge deployment may expose a custom served ID;
set `JUDGE_MODEL` to that ID even if its documented model family is
Deepseekv4-flash-0731. Environment files are private and ignored by Git.

Wrappers default to `reasoning_effort=xhigh` and `thinking_budget=4096`, delivered
through `chat_template_kwargs`. Model templates must support the requested options.
Direct Python entries can default to `high`; use the wrappers for the documented
configuration. No prompt or judge template is translated at runtime.

The following defaults differ by benchmark and must not be normalized as an
editorial change:

| Wrapper / entry | Reasoning-token setting | Per-turn completion tokens | Correction rounds | API JPEG quality |
|---|---|---:|---:|---:|
| SSPro / OS-World-G (Refine) | `THINKING_BUDGET` (4096) | Uses thinking budget when unset | 2 | 90 |
| PerceptionBench | `AGENTIC_THINKING_BUDGET` → `THINKING_BUDGET` → 4096 | `MAX_TOKENS` (16384) | 2 | 90 |
| HR-Bench | `THINKING_BUDGET` (4096) | `MAX_COMPLETION_TOKENS` (thinking budget + 1024) | 4 | 95 |

PerceptionBench uses `AGENTIC_THINKING_BUDGET` when explicitly set, otherwise the
shared `THINKING_BUDGET`, then 4096. The standalone wrapper loads the environment
file before resolving `PB_NUM_SHARDS`; an explicit third positional argument
takes precedence. See [known limitations](#known-limitations) for remaining
operational caveats and [offline checks](../docs/RELEASE_VALIDATION.md) for
regression coverage.

## Run and resume

Run these commands from `evaluation/`:

```bash
# Smoke tests
bash scripts/run_sspro.sh .env.run results/smoke_sspro --max_samples 10
bash scripts/run_osworld.sh .env.run results/smoke_osworld --max_samples 10
LIMIT=10 bash scripts/run_perceptionbench.sh .env.run results/smoke_pb 1
bash scripts/run_hrbench.sh .env.run results/smoke_hrbench --max_samples 8

# Full suite, with benchmarks scheduled sequentially
bash scripts/run_all.sh .env.run results/run-name

# Individual benchmarks
bash scripts/run_sspro.sh .env.run results/run-name
bash scripts/run_osworld.sh .env.run results/run-name
bash scripts/run_perceptionbench.sh .env.run results/run-name 8
bash scripts/run_hrbench.sh .env.run results/run-name
```

The single-shard PB smoke test auto-finalizes. The producer and finalizer share
`PerceptionBench_agentic_<model><PB_OUTPUT_SUFFIX>[_shard<i>of<N>]`: one shard
omits the shard tag, and the finalizer also accepts the legacy `_shard0of1` spelling.
It rejects ambiguous or incomplete input sets. Manual finalization reads
`PB_OUTPUT_SUFFIX` from the environment or `--output-suffix`; use the same suffix
as inference. Cleanup removes only artifacts consumed by that finalization.

HR-Bench's 4K and 8K splits run **in parallel** inside its stage. Each split uses
`HRBENCH_WORKERS` workers. SSPro/OSWorld in `run_all.sh` use
`SSPRO_NUM_PROCESSES * SSPRO_WORKERS`; standalone wrappers preserve the Python
entry's process default unless `--num_processes` is passed. PerceptionBench uses
`PB_NUM_SHARDS * PB_SHARD_CONCURRENCY`, by default 8 × 4 = 32. Tune concurrency to
endpoint capacity; there is no universal safe rate limit.

Rerun the same command with the same configuration to resume. Keep the following
until the stage finishes:

- SSPro/OSWorld: `*.ckpt.json` and per-process checkpoint files.
- PerceptionBench: `PerceptionBench_agentic_<model><suffix>[_shard<i>of<N>].jsonl`.
- HR-Bench: hidden `.hrbench_<split>.{inference,judge}.progress.jsonl` files.

`CLEAN_INTERMEDIATES=0` retains wrapper-managed intermediates. PerceptionBench
finalization independently deletes shards unless `--keep-shards` is passed; use
`FINALIZE=0` to retain shards before manual finalization. `FORCE_RERUN=1` bypasses completed-stage
checks. Changing evaluation settings requires a new results directory: the stage
completion checks are not a full configuration-equivalence guarantee.

SSPro/OS-World-G reports include `run_config.max_samples` and a `completion`
object recording full/smoke scope, task counts, and a digest of task identities.
A failed worker, missing shard, or incomplete/duplicate task coverage fails the
stage before publishing the final report and retains checkpoints. An omitted
SSPro multiprocess checkpoint path defaults to `<log_path stem>.ckpt.json`.
The skip guard requires consistent full-run completion evidence, model identity,
and the default task/dataset selection; legacy reports without that evidence
are not automatically skipped. Existing legacy OS-World-G segments can be merged
only when they cover the complete annotation. Smoke and full runs should use
separate output directories, particularly when process partitions change.
Checkpoint files with duplicate or foreign tasks are retained and rejected, rather
than silently merged into a new task selection.

Check `chain.log` and each `*_run.log` for progress. Completed outputs include
`sspro.json`, `osworld.json`, `perceptionbench.jsonl`, `perceptionbench_scores.json`,
`hrbench_4k.json`, `hrbench_8k.json`, and `hrbench.json`. PerceptionBench also creates
a model-card template if one is not already present.

## Deferred judging and rejudging

Keep shards when planning to judge later:

```bash
PB_SKIP_JUDGE=1 FINALIZE=0 CLEAN_INTERMEDIATES=0 bash scripts/run_perceptionbench.sh .env.run results/deferred 8
```

The entry checks `JUDGE_API_KEY` at startup even in inference-only mode. Keep
that variable configured; no judge request is made when judging is skipped.

Load the endpoint configuration and rejudge **each retained shard** using the same
judge/template. For example, replace the filename with the actual model ID:

```bash
set -a
source .env.run
set +a
python3 perceptionbench/debug/rejudge_agentic.py \
  --results results/deferred/PerceptionBench_agentic_evaluated-model_shard0of8.jsonl \
  --dataset data/perceptionbench/PerceptionBench.jsonl \
  --statuses skipped,judge_error --workers 2
# Repeat for shards 1 through 7, then aggregate:
python3 scripts/finalize_perceptionbench.py --results results/deferred \
  --model "$MODEL" --judge-model "$JUDGE_MODEL" --num-shards 8 --keep-shards
```

The rejudge tool creates a `.bak_rejudge` backup before updating its input. Rejudging
only the merged `perceptionbench.jsonl` does not update retained shards; finalization
reads shards and would overwrite that merged file. Do not rerun finalization on
stale shards. Report nonzero failures and retry them before publishing comparisons.

## Known limitations

- PerceptionBench initializes its judge client even with `PB_SKIP_JUDGE=1`;
  the current startup path still requires judge credentials.
- If both HR-Bench split reports already exist but `hrbench.json` is missing,
  the wrapper's skip path does not rebuild the combined report. Combine the
  existing reports directly without model inference (from `evaluation/`):

  ```bash
  python3 hrbench/eval_hrbench_agentic.py \
    --combine_reports results/run-name/hrbench_4k.json results/run-name/hrbench_8k.json \
    --log_path results/run-name/hrbench.json
  ```

- PerceptionBench gold-pass diagnostic filenames are not scoped by run/shard;
  concurrent writes can overwrite failure evidence. Its wrapper can also return
  on the first failed child before other children finish. Inspect all shard logs
  and preserve intermediates before retrying.
- PerceptionBench worker summaries and the finalized report use different
  populations when answers/judgments fail. Use the finalized report for published
  scores; do not compare worker `overall` values as if their denominators matched.

These caveats do not change the published run's saved scores. Offline regression
tests are available in `tests/`.

## Troubleshooting

| Symptom | Check |
|---|---|
| Model 404 | Match `MODEL` to `/v1/models` |
| Connection refused | Confirm readiness with `serve_vllm.sh status` |
| Judge 429 | Reduce shard/concurrency settings and rejudge failed records |
| Context-length error | Check image tokens, completion budget, and server context length |
| Unexpected thinking behavior | Confirm the model template consumes `chat_template_kwargs` |
| Missing crop/image | Verify prepared dataset paths and writable output directories |
| Missing final answer | Inspect the saved trace and protocol compliance; do not silently change prompts |

## License and provenance

Software uses the repository-root [Apache-2.0 license](../LICENSE). PerceptionBench
judge templates and code are adapted from [MoonshotAI/PerceptionBench](https://github.com/MoonshotAI/PerceptionBench).
See [NOTICE](../NOTICE) and [judge protocol notes](../docs/JUDGE_PROTOCOL.md).

## Benchmark display scope

Benchmark excerpts included here are used solely to display recorded evaluation
results and observation trajectories, with no other use intended. Their original
ownership, licenses, and terms remain unchanged. The Apache-2.0 software license
does not relicense these materials. Source attribution is retained in
[NOTICE](../NOTICE).
