# Zoom-In Grounding RL Data Samples

This directory contains 8 prompt-only rows and 32 complete rollout trajectories used by the
zoom-in crop-policy GRPO experiment, provided as a reference for building your
own training data. LBS is based on **Qwen3.8-27b**, trained with cold-start
**SFT** followed by **Agentic RL (GRPO)**. The samples were taken from the
Agentic RL stage initialized from the SFT checkpoint (ms-swift framework,
vLLM rollout server, multi-node
topology, `--rlhf_type grpo`, `num_generations=8`, multi-turn
`zoomin_grounding_scheduler`).

| Artifact | Records | Unique original screenshots | What it illustrates |
|---|---:|---:|---|
| Training inputs | 8 | 8 | Prompt-only data consumed by GRPO, with a target box for reward computation. |
| Real training rollouts | 32 | 4 | Four prompt groups, each with eight generated observation trajectories and recorded rewards/advantages. |

## Layout

```
├── grpo_train_prompts/             # form 1: GRPO training input (prompt-only)
│   ├── train_prompts_head8.jsonl       8 rows (first 8 rows of a training set,
│   │                                    loaded in --dataset order)
│   └── images/                          8 unique original screenshots
│       └── img_000.png ... img_007.png
└── rollout_trajectories/          # form 2: complete RL rollout trajectories
    ├── rollout_trajectories_head32.jsonl   32 multi-turn trajectories (4 prompts x 8 generations)
    └── images/
        ├── originals/                4 original screenshots
        └── crops/rXX/crop_NN.png     47 crops rebuilt offline with the exact training protocol
```

## Form 1: GRPO training input (`grpo_train_prompts/`)

GRPO consumes prompt-only data (no reference responses; the answer is used
only by the reward). Rows load directly as `swift rlhf --dataset` input:

```json
{"problem": "<image>Output the center point of ... instruction text ...",
 "solution": "[793, 113, 799, 123]",
 "images": ["images/img_000.png"]}
```

| Field | Meaning |
|---|---|
| `problem` | Task text; `<image>` placeholders map to the `images` array in order |
| `solution` | GT bbox `[x1, y1, x2, y2]` in original-image pixels, used by the point-in-bbox reward |
| `images` | Screenshot paths relative to this jsonl |

- Source: a cleaned RL grounding training set with a mix of hard (marginally
  learnable), rescued (only correct after zooming), and easy strata; this
  sample retains its first 8 rows without editing their text, bounding boxes,
  or image references. Only the eight referenced screenshots are shipped.

## Form 2: rollout trajectories (`rollout_trajectories/`)

Complete multi-turn rollout conversations logged with `--log_completions true`
(first 32 rows of training step 1 = 4 distinct prompts x 8 GRPO generations),
restored from the chat-template log into standard swift/OpenAI `messages`
JSON. The file contains **32 distinct conversations**, with 32 unique
`rollout_index` values. Each `prompt_group` (0–3) has eight generations (0–7),
all sharing that group's original image. Thus **4 original screenshots correspond
to 32 rollout trajectories**. Keeping all eight generations preserves the
within-group reward and advantage comparisons used by GRPO.

Each record stores the model's reasoning and actions, tool feedback, and final
assistant response from the actual training process. Accompanying crop images
are offline reconstructions, as described below. Example record:

```json
{"messages": [{"role": "system", "content": "..."},
              {"role": "user", "content": "image_path: ...<image>"},
              {"role": "assistant", "content": "# contains <action>crop_image(...)</action>"},
              {"role": "user", "content": "crop_image action result:\n{json}\nimage_path: ...<image> ..."},
              ..., {"role": "assistant", "content": "...\n\n[811.02,511.28]"}],
 "images": ["images/originals/org_00.png", "images/crops/r00/crop_01.png", ...],
 "rollout_index": 0, "prompt_group": 0, "generation": 0, "step": 1,
 "num_turns": 3, "solution": "[706, 474, 938, 510]",
 "rewards": {"ZoomInAccuracy": 0.0, "ZoomInCropProcessGated": 0.0, "ZoomInFormat": 1.0,
             "ZoomInZoomBonus": 0.0, "ZoomInCotRM": 0.0},
 "advantages": -0.3642054498195648}
```

### Interaction protocol (multi-turn crop zoom-in)

1. `system` — agentic grounding task description (instruction, original-image
   metadata, action protocol, final-answer format `[x.xx,y.yy]` normalized to
   [0,1000])
2. `user` — `image_path: <path><image>` provides the original screenshot
3. `assistant` — reasons first, then emits `<action>crop_image(image_path="...", x1=..., y1=..., x2=..., y2=...)</action>`
4. `user` — `crop_image action result:` + JSON feedback (new image path /
   size / crop→parent→original mappings), with the cropped `<image>` and a
   prompt to continue zooming or answer
5. repeat 3-4 (at most 2 crops per turn and **3 crop calls in total**, as
   stated in the embedded system prompt; invalid actions receive an error
   message and don't consume the budget; up to 2 corrections)
6. `assistant` — final answer `[x.xx,y.yy]`

Invalid actions (more than 2 crops per turn, malformed coordinates, mixed
crop-action + final answer) receive a
`Your previous answer was not valid...` correction turn.

### Extra metadata (ignored by swift; useful for analysis)

| Field | Meaning |
|---|---|
| `prompt_group` / `generation` | GRPO group: `generation`-th rollout for the same `prompt_group` (0-7) |
| `step` | training step number |
| `num_turns` | number of interaction turns in this rollout |
| `solution` | GT bbox (original-image pixels) |
| `rewards.ZoomInAccuracy` | point-in-bbox hit (1.0 / 0.0) |
| `rewards.ZoomInCropProcessGated` | process reward (gated on final correctness; charged per useful crop) |
| `rewards.ZoomInFormat` | final answer format compliance |
| `rewards.ZoomInZoomBonus` | bonus for answering correctly after genuine zoom-in |
| `rewards.ZoomInCotRM` | LLM-as-judge reasoning score (gated on correctness) |
| `advantages` | GRPO within-group advantage |

The shipped plugin registers exactly these five reward components
(`zoomin_accuracy`, `zoomin_crop_process_gated`, `zoomin_format`,
`zoomin_zoom_bonus`, `zoomin_cot_rm`), renamed here from the logged
`ZoomIn*` labels.

### About the crop images

During training, crops were generated live by the rollout node and never
entered the logs. The crops in `images/crops/rXX/crop_NN.png` were rebuilt
offline with the exact training protocol: `crop_in_parent` region →
LANCZOS 4× upsampling (4M-pixel cap); rebuilt dimensions match the log JSON's
`width`/`height` for every entry.

### Anonymization notes

The trajectories are verbatim from the training log except:

- every `image_path` and `images` reference was rewritten to an anonymized
  relative path and the image files renamed accordingly (original dataset
  paths and generator tags were removed)
- reward names were renamed from their internal experiment names to neutral
  `ZoomIn*` labels (values untouched)
- some turns keep a `<think>` open-tag residue ("model only thought
  and never emitted a well-formed action"), exactly as it occurred in the
  GRPO rollout

## Loading with ms-swift

The paths below are illustrative local directories: `qwen3.8-27b-sft` is the
cold-start SFT checkpoint used to initialize RL, and `qwen3.8-27b` is the base
model. The second command illustrates optional reuse of recorded RL trajectories;
it is not the cold-start SFT dataset or recipe used for the reported model.

```bash
# form 1 as the GRPO dataset (same as the sampled experiment run)
swift rlhf --rlhf_type grpo \
  --model /models/qwen3.8-27b-sft \
  --dataset grpo_train_prompts/train_prompts_head8.jsonl \
  --external_plugins ../plugins/grounding_zoom_in/zoomin_grounding_plugin.py \
  --multi_turn_scheduler zoomin_grounding_scheduler \
  --reward_funcs zoomin_accuracy \
  --enable_thinking true --preserve_thinking true \
  --log_completions true ...   # see train_zoomin_grounding_grpo.sh for full hyperparams

# form 2 for inspection / distillation / SFT-style reuse
# (messages + images are the standard swift format)
swift sft --model /models/qwen3.8-27b \
  --dataset rollout_trajectories/rollout_trajectories_head32.jsonl ...
```

`images` paths are relative to the jsonl; run from this directory or convert
to absolute paths. The `messages` rows are also directly usable with any
OpenAI-compatible chat API (see the main README's serving section).

## Sample provenance

- training data: rows from a cleaned, deduplicated RL grounding
  training set (~33k rows across difficulty strata); this sample = its first
  8 rows without changes to task text, target boxes, or image references
- rollout trajectories: first 32 entries of training step 1 (512 rollouts /
  step); every 8 consecutive rows form one GRPO group (same prompt), so
  within-group `rewards` and `advantages` are directly comparable
- screenshots retain their original ownership and terms; see [NOTICE](../../NOTICE)

## Sample terms

Sample screenshots retain their original ownership and terms. The repository's
Apache-2.0 software license does not relicense screenshot content; source
attribution is retained in [NOTICE](../../NOTICE).
