# Zoom-In Grounding Plugin (textual crop protocol, five-component reward)

A GUI-grounding GRPO training plugin for ms-swift with these properties:

- A single tool: `crop_image` (crop, upscale, re-inspect)
- Custom **text-action protocol** `<action>crop_image(...)</action>` — no
  native tools / `tool_calls` schema required
- Multi-turn tool loops run through `swift rollout` server mode
- A **five-component reward**: deterministic geometry (accuracy / process /
  zoom bonus / format) plus an **optional LLM-as-judge CoT reward** served by
  any OpenAI-compatible endpoint (standard OpenAI Python SDK client)
- Crop arguments use **local absolute pixels of the referenced image**; the
  final answer always uses the ORIGINAL image's global `[0,1000]`-normalized
  coordinates
- Hard cap of **3** crop calls (`MAX_CROP_IMAGE_CALLS`), at most 2 per turn

## Registered names

```
scheduler   zoomin_grounding_scheduler
rewards     zoomin_accuracy  zoomin_crop_process_gated  zoomin_format
            zoomin_zoom_bonus  zoomin_cot_rm
```

Usage:

```bash
--reward_funcs zoomin_accuracy zoomin_crop_process_gated zoomin_format \
               zoomin_zoom_bonus zoomin_cot_rm \
--reward_weights 1.0 0.3 0.5 1.0 0.3
```

## Files

- [zoomin_grounding_plugin.py](zoomin_grounding_plugin.py)
  - registers `zoomin_grounding_scheduler` (multi-turn rollout scheduler)
  - registers the 5 ORM rewards above
- [zoomin_grounding_protocol.py](zoomin_grounding_protocol.py)
  - system-prompt template, tolerant text-action parser, `execute_crop`
    (crop + LANCZOS upsample + coordinate mapping), crop-result user-message
    construction, reward math, CoT transcript builder

## Reward composition

| Component (default weight) | Fires when | Shapes |
|---|---|---|
| accuracy (1.0) | point inside GT bbox | outcome — the main signal |
| crop_process_gated (0.3) | answer correct; best-IoU − penalties − `call_cost` × #crops | tool-call quality + usage cost |
| format (0.5) | canonical protocol compliance | output stability |
| zoom_bonus (1.0, internal 0.2) | answer correct AND hard sample AND high-quality crop | pays for effective zooming on hard samples |
| cot_rm (0.3) | answer correct (gate configurable) AND trajectory has think blocks | reasoning / tool-decision quality (LLM judge) |

Except format, every component pays only when the final answer is correct:
credit concentrates on correct trajectories, preventing "crop well but answer
wrong" farming and lucky wide-area search credit. A direct correct answer
with no crops keeps its full accuracy credit and pays nothing (and receives
nothing) in process components; each extra crop call must be justified by a
real IoU improvement over the best crop so far (repeat-crop and
no-improve penalties). Loss-side, PRM-style quality + uncertainty turn
filtering (`zoomin_grounding_scheduler`) additionally masks bad decision
turns.

## The LLM-as-judge CoT reward (`zoomin_cot_rm`, AsyncORM)

1. Flattens each trajectory's assistant think-blocks + action decisions +
   text-only crop feedback into a compact transcript (images/base64 dropped,
   truncation budgets `ZOOMIN_COT_RM_MAX_SEGMENT_CHARS` / `_MAX_ACTION_CHARS`
   / `_MAX_TOTAL_CHARS` = 1500 / 600 / 6000 chars).
2. Sends it to an **OpenAI-compatible chat completions endpoint using the
   official OpenAI Python SDK** (`pip install openai`, `AsyncOpenAI` +
   `client.chat.completions.create`). Configure the endpoint for your deployment:

   ```bash
   # example: the public DeepSeek API
   export ZOOMIN_COT_RM_API_BASE=https://api.deepseek.com/v1
   export ZOOMIN_COT_RM_API_KEY=<your-key>          # never hardcode secrets
   export ZOOMIN_COT_RM_MODEL=deepseek-chat

   # example: a local vLLM OpenAI-compatible server (no auth)
   export ZOOMIN_COT_RM_API_BASE=http://localhost:8000/v1
   export ZOOMIN_COT_RM_API_KEY=                    # empty is fine
   export ZOOMIN_COT_RM_MODEL=<served-model-name>
   ```

   If `ZOOMIN_COT_RM_API_BASE` is empty (default), the judge is disabled:
   every trajectory receives the neutral score (`ZOOMIN_COT_RM_NEUTRAL`,
   default 0.5) and training proceeds with the other four rewards.
3. The judge scores 0~1 across five dimensions (goal orientation, feedback
   use, thought-action consistency, incremental progress, conciseness);
   `ZOOMIN_COT_RM_ORACLE=1` adds a sixth "alignment with the GT annotation"
   dimension and gives the judge the GT reference.
4. The judge score is multiplied by binary accuracy
   (`ZOOMIN_COT_RM_GATE_ACCURACY=1` default) — wrong answers earn no CoT
   credit. Trajectories without any think block skip the judge call and
   score 0.
5. Calls run at `ZOOMIN_COT_RM_CONCURRENCY` (=8) per rank with a
   `ZOOMIN_COT_RM_TIMEOUT` (=30s) timeout, `ZOOMIN_COT_RM_RETRIES` (=1)
   bounded retries and cross-endpoint failover (configure a second endpoint
   via `ZOOMIN_COT_RM_API_BASE_2` / `ZOOMIN_COT_RM_API_KEY_2`); on total
   failure a neutral score is returned and a >10% cumulative failure rate
   warns once per rank.

Judge observables: `rewards/ZoomInCotRM/mean` (accuracy-gated — its level
reflects both correctness and reasoning quality), `rewards/
ZoomInCropProcessGated/mean`, `rewards/ZoomInZoomBonus/mean`, `num_turns`,
`rewards/ZoomInAccuracy/mean`, and the occasional
`zoomin_cot_rm: ... failed` warning in the logs.

## Data format

Each sample row requires:

```json
{
  "problem": "<image>... following instruction: <instruction> ...",
  "solution": "[x1, y1, x2, y2]",
  "images": ["/path/to/image.png"]
}
```

- `solution` is the GT bbox in ORIGINAL-image pixel coordinates
- the instruction inside `problem` is extracted by `extract_instruction()`
  (text between `following instruction:` and `the output should just be`)
- infeasible-task samples: `solution = [-1, -1, -1, -1]`; the expected answer
  is `[-1.00,-1.00]`

## Assistant protocol

The exact prompt text lives in `SYSTEM_PROMPT_TEMPLATE`
([zoomin_grounding_protocol.py](zoomin_grounding_protocol.py)); informally:

**Tool-call turn** (one or two independent actions per turn):

```text
Think...
<action>crop_image(image_path="exact path", x1=100.00, y1=200.00, x2=500.00, y2=600.00)</action>
```

**Final-answer turn**:

```text
Think...
[x.xx,y.yy]
```

Coordinate-system rules (critical):

- `crop_image`'s `x1,y1,x2,y2` are LOCAL absolute pixels of the image passed as
  `image_path`; crops are returned resized (upscaled by
  `ZOOMIN_CROP_UPSCALE_FACTOR`, default 4x, capped at
  `ZOOMIN_CROP_MAX_PIXELS`, default 4M px) — nested crops must use the
  returned (post-resize) dimensions
- crops that become too elongated fail with an error
  (`ZOOMIN_MAX_CROP_ASPECT_RATIO`, default 100 — guards template encoders
  against pathological letterbox crops)
- each crop result states the region both in original-image pixels and in
  original-image `[0,1000]` normalized coordinates, plus an explicit
  `mapping_to_original_0_1000` conversion formula
- the final answer must be the ORIGINAL image's `[0,1000]` normalized
  coordinates with exactly two decimals; for infeasible tasks output
  `[-1.00,-1.00]`

## Budgets and turns

- Hard cap: `MAX_CROP_IMAGE_CALLS = 3` (in `zoomin_grounding_protocol.py`).
  `ZOOMIN_MAX_TOOL_CALLS` above 3 is silently clamped to 3, so just leave
  `MAX_TOOL_CALLS=3`.
- Format-correction budget: `ZOOMIN_MAX_ACTION_CORRECTIONS` (default 2).
  Trajectories that exhaust it end in an error state with 0 rewards.
- Recommended `MAX_TURNS = 3 + 2 + 1 = 6`.
- Once the budget is spent, further `crop_image` calls are refused and the
  scheduler asks the model for the final coordinate directly.

## Running

The plugin is deployed automatically by `train_zoomin_grounding_grpo.sh` and
used on both sides (the scheduler runs INSIDE the rollout server):

```bash
MODEL_PATH=/models/your-vlm NNODES=1 bash train_zoomin_grounding_grpo.sh
```

Key environment variables consumed by the scheduler (exported by the training
script): `ZOOMIN_MAX_TOOL_CALLS`, `ZOOMIN_MAX_ACTION_CORRECTIONS`,
`ZOOMIN_MAX_COMPLETION_TOKENS` (per-turn completion cap),
`ZOOMIN_CROP_UPSCALE_FACTOR`, `ZOOMIN_CROP_MAX_PIXELS`, `ZOOMIN_CROP_ROOT`.

Process-reward switches: `ZOOMIN_CALL_COST` (0.05), repeat/no-improve
penalty caps (`ZOOMIN_REPEAT_*`, `ZOOMIN_NO_IMPROVE_PENALTY`), zoom bonus
tuning (`ZOOMIN_ZOOM_BONUS*`), and the judge's `ZOOMIN_COT_RM_*` family
(see above).

To remove the judge reward, omit `zoomin_cot_rm` and its matching weight from
`--reward_funcs` and `--reward_weights`. Leaving the endpoint unset skips judge
requests but still uses the configured neutral reward.
`ZOOMIN_COT_RM_GATE_ACCURACY=0` removes the accuracy gate; it does not disable
the judge and can reward reasoning attached to incorrect answers.
`ZOOMIN_CALL_COST=0` keeps the gating but waives the usage cost.

## Context-length management

Multi-turn cropping stacks "original image + N crops + text" into the context:

- training-side `--max_length` should align with rollout-side
  `--vllm_max_model_len` (mismatched truncation breaks the loss mask);
  the training script keeps both configurable in one place
- `MAX_PIXELS` (base image budget) and `CROP_MAX_PIXELS` (per-crop budget)
  must stay consistent across rollout/training sides
- image-token usage depends on the model processor. Budget for the original
  image, all returned crops, and conversation text within `MAX_LENGTH`; verify
  actual token usage with the target model before a full run

The training launcher defaults to `MAX_PIXELS=16777216` and
`MAX_LENGTH=32768`. These are independent image-pixel and context-token limits;
confirm that the model processor and crop history fit both.

## Training/serving consistency & robustness

- the scheduler restores `response_token_ids` and `rollout_logprobs` for all
  assistant turns (importance-sampling correction); the loss mask covers all
  assistant turns
- `_sanitize_output` scrubs NaN/Inf values (bf16 vision-encoder overflow)
  so one bad token cannot fail the whole rollout batch
- crops reach the training node as base64 data URIs (`rollout_infos['images']`)
  — no shared filesystem needed
- the parser falls back gracefully: coordinate-after-think → trailing
  coordinate → `<action>` block (ast parse + repair for malformed arg lists) →
  JSON → XML `tool_calls` → last coordinate in text; NaN/Inf JSON constants
  are rejected

## Known limitations

- all reward components except format pay only on correct answers, so an
  all-wrong GRPO group yields ~no gradient (known cost of gating)
- every assistant turn shares the trajectory-level advantage, so individual
  crop decisions cannot be attributed
- the judge score is a heuristic proxy for reasoning quality and sits on the
  reward critical path: a slow endpoint slows the training step (weight 0.3
  + accuracy gating cap its influence radius)
- the parser's last-coordinate fallback weakens format enforcement

## Compatibility

Do not load two different zoom-in plugin directories in one process (same
protocol module name and scheduler registration key).
