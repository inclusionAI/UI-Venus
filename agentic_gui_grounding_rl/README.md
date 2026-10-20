# Agentic GUI Grounding RL — Zoom-In Crop-Policy GRPO

A GRPO recipe for training **agentic GUI grounding** models with
reinforcement learning. Instead of answering a grounding question in a single
pass, the model interacts with a simple environment: it may ask to **zoom into
image regions** (`crop_image`) to inspect small UI elements, and only then
outputs its final normalized coordinate. Training uses **GRPO** on top of
[ms-swift](https://github.com/modelscope/ms-swift) + [vLLM](https://github.com/vllm-project/vllm).

```
User:                      image_path: screenshot.png <image>   ← original screenshot
Assistant:  <think>small target, zoom in</think>
           <action>crop_image(image_path="screenshot.png",
                              x1=560.00, y1=150.00, x2=844.00, y2=240.00)</action>
User:                      crop result (resized crop <image> + metadata + mapping)
Assistant:  <think>found it</think>
           [798.23,457.17]                                    ← final [0,1000] normalized point
```

- **Text-action protocol** — `crop_image` is emitted as plain assistant text and
  executed by a rollout scheduler. No native tool-calling schema is required,
  so it works with any instruct VLM.
- **Five-component reward** — deterministic geometric rewards (binary
  point-in-bbox accuracy, outcome-gated crop-process scoring with a per-call
  usage cost, zoom bonus on hard samples, protocol-format compliance) plus an
  **optional LLM-as-judge CoT reward**. The judge connects to any
  OpenAI-compatible endpoint through the official OpenAI Python SDK
  (`pip install openai`); with no endpoint configured, no judge request is made and the configured
  neutral score is used.
- **Multi-turn GRPO** — a custom `MultiTurnScheduler` drives multi-turn
  crop conversations inside a vLLM rollout server; the trainer assigns
  trajectory-level GRPO advantages.
- **Both RL code and real data samples** included: 8 prompt-only training rows
  and 32 complete multi-turn rollout trajectories (4 prompts x 8 GRPO
  generations) from an actual training run. The 32 rollouts share 4 original
  screenshots, one per prompt group.

## Reported training setup

LBS is based on **Qwen3.8-27b**, with cold-start **SFT** followed by
**Agentic RL (GRPO)**. It uses **entirely synthetic GUI grounding training data** in both cold-start
SFT and RL, with **no general-domain data or VQA questions** in either stage.
The reported experiment uses the following training budget:

| Setting | Reported value |
|---|---|
| Base model | Qwen3.8-27b |
| Training stages | Cold-start SFT → Agentic RL (GRPO) |
| Cold-start SFT data | Approximately 10,000 synthetic GUI grounding examples |
| RL training | 80 GRPO steps |
| RL batch size | 64 |
| Rollouts per prompt | 8 |

These values describe the reported experiment. The
[training launcher](train_zoomin_grounding_grpo.sh) supports environment overrides
for different device topologies and training schedules; the examples below cover
smoke tests and multi-node execution. The released
[data and rollout samples](data_samples/README.md) illustrate the training inputs,
multi-turn observations, and recorded rewards.

## Repository layout

```
├── README.md                          # this file
├── train_zoomin_grounding_grpo.sh     # GRPO training orchestration script
├── plugins/grounding_zoom_in/
│   ├── zoomin_grounding_plugin.py      # scheduler + reward registration (RL core)
│   ├── zoomin_grounding_protocol.py    # system prompt / action parser / crop exec / reward
│   └── README.md                       # protocol & plugin details
└── data_samples/                       # RL data samples (see data_samples/README.md)
    ├── grpo_train_prompts/             # form 1: GRPO prompt-only training data
    └── rollout_trajectories/           # form 2: complete multi-turn rollout logs
```

## Quickstart

### Install

```bash
pip install ms-swift vllm deepspeed pillow qwen-vl-utils
# only needed when you enable the LLM-as-judge CoT reward:
pip install openai
```

The training script deploys the plugin automatically; the plugin files only
need to stay together in one directory.

### Smoke test (single node)

Initialize Agentic RL from the Qwen3.8-27b checkpoint after cold-start SFT.
The illustrative local path `/models/qwen3.8-27b-sft` below refers to that
checkpoint. Run the shipped 8-row sample dataset on a single node with sufficient
device memory: the first N devices run a vLLM rollout server, and the remaining
devices run the GRPO trainer. Adjust topology for your hardware.

```bash
MODEL_PATH=/models/qwen3.8-27b-sft NNODES=1 \
VLLM_TP=1 bash train_zoomin_grounding_grpo.sh
```

Logs and checkpoints land in `./output/zoomin_grounding_grpo/`.

### Real training / multi-node

Run the examples in this section from `agentic_gui_grounding_rl/`. Provide your
own dataset as JSONL rows
(`{"problem": "<image>...instruction...", "solution": "[x1,y1,x2,y2]", "images": ["a.png"]}`)
and pass several via the `DATASETS` env var (space-separated).

```bash
# rollout node (rank 0): one vLLM server
MODEL_PATH=/models/qwen3.8-27b-sft NNODES=4 NODE_RANK=0 ROLLOUT_NNODES=1 \
MASTER_ADDR=gw0 MASTER_PORT=29500 VLLM_SERVER_HOSTS="gw0" \
DATASETS="/data/ds1.jsonl /data/ds2.jsonl" bash train_zoomin_grounding_grpo.sh

# training nodes (ranks 1..3): GRPO trainer
MODEL_PATH=/models/qwen3.8-27b-sft NNODES=4 NODE_RANK=1 ROLLOUT_NNODES=1 \
MASTER_ADDR=gw0 VLLM_SERVER_HOSTS="gw0" \
DATASETS="/data/ds1.jsonl /data/ds2.jsonl" bash train_zoomin_grounding_grpo.sh
```

Almost everything is overridable via environment variables (see the script
header). Keep `MAX_LENGTH`/`MAX_PIXELS`/crop budgets identical across rollout
and training sides; the script already exports the shared variables from one
place.

## The zoom-in protocol

Each rollout unfolds as a multi-turn conversation built by the scheduler:

1. **system**: task instruction, original-image metadata (path, resolution),
   the `crop_image` action spec, answer format (`[x.xx,y.yy]`, normalized to
   [0,1000] on the *original* image).
2. **user**: `image_path: ...` + the original screenshot.
3. **assistant**: reasoning, then one or two `<action>crop_image(...)</action>`
   blocks (or a final `[x.xx,y.yy]`).
4. **user**: `crop_image action result:` JSON (new image path/size, region in
   parent and original pixels + [0,1000] mapping) + the resized crop image.
5. Repeat until the final answer. Budgets: at most **3** `crop_image` calls in
   total, at most **2** per turn, and at most **2** format-correction rounds;
   recommended `MAX_TURNS = 3 + 2 + 1 = 6`.

Invalid actions (more than 2 crops per turn, malformed coordinates, mixing
crops with a final answer) receive a correction message and do not consume the
crop budget. When the budget is exhausted the scheduler asks the model to
answer directly.

Crops are executed with PIL: LANCZOS upscaled by `ZOOMIN_CROP_UPSCALE_FACTOR`
(default 4x, capped at `ZOOMIN_CROP_MAX_PIXELS`, default 4M px). Nested crops
are allowed; every result carries an explicit mapping back to original-image
coordinates so the policy can translate crop-local findings into the global
coordinate system.

Full protocol (exact prompts, parser fallback chain, reward math):
[plugins/grounding_zoom_in/README.md](plugins/grounding_zoom_in/README.md).

## RL core: plugin architecture

`plugins/grounding_zoom_in/zoomin_grounding_plugin.py` registers:

| Registered name | What it is |
|---|---|
| `zoomin_grounding_scheduler` | `MultiTurnScheduler` — builds the system/user prompt from each `problem` row, parses assistant actions, executes crops, returns tool results as user messages, applies PRM/uncertainty turn filtering, restores per-turn `response_token_ids`/`rollout_logprobs` for importance-sampling correction |
| `zoomin_accuracy` | ORM reward — parse the final `[x.xx,y.yy]`, map [0,1000] → pixel space, binary point-in-GT-bbox (main outcome signal) |
| `zoomin_crop_process_gated` | ORM reward — pays only on correct answers: best-crop IoU minus repeat/no-improve penalties minus a per-call usage cost (`ZOOMIN_CALL_COST`, default 0.05) |
| `zoomin_format` | ORM reward — canonical protocol compliance of the final assistant turn |
| `zoomin_zoom_bonus` | ORM reward — extra credit for correct answers earned through effective zooming on hard samples |
| `zoomin_cot_rm` | `AsyncORM` reward — LLM-as-judge reasoning score over the text-only trajectory transcript, accuracy-gated, called via the **OpenAI Python SDK** against any OpenAI-compatible endpoint (or disabled) |

Suggested composition (also the script default):

```bash
--reward_funcs zoomin_accuracy zoomin_crop_process_gated zoomin_format zoomin_zoom_bonus zoomin_cot_rm \
--reward_weights 1.0 0.3 0.5 1.0 0.3
```

The judge component is deliberately provider-neutral: point
`ZOOMIN_COT_RM_API_BASE` at any OpenAI-compatible endpoint
(e.g. `https://api.deepseek.com/v1` with `ZOOMIN_COT_RM_MODEL=deepseek-chat`,
or a local vLLM server at `http://host:8000/v1`), set `ZOOMIN_COT_RM_API_KEY`
when the endpoint requires one (empty = placeholder for unauthenticated
endpoints). With no base URL configured the judge is disabled and the neutral
score is used, so training never hard-depends on an external service. Details,
anti-hacking (accuracy gating), concurrency/failover knobs:
[plugins/grounding_zoom_in/README.md](plugins/grounding_zoom_in/README.md).

`zoomin_grounding_protocol.py` holds the pure logic (prompt template,
multi-level tolerant action parser, crop execution with coordinate mapping,
reward math, CoT transcript builder), fully unit-testable without the
training stack.

The implementation also preserves rollout state across multi-turn interactions:

- All assistant turns share the trajectory-level GRPO advantage; the loss mask
  covers every assistant turn (see plugin `run()`).
- NaN/Inf sanitizer on engine outputs (vision towers can overflow in bf16) to
  prevent one non-finite output from failing the entire rollout batch.
- Crops are transferred to the training node as base64 data URIs — no shared
  filesystem required between rollout and training nodes.
- The parser accepts several action dialects (think-then-coordinate, `<action>`
  blocks, JSON, XML `tool_calls`, a last-coordinate fallback) and rejects
  NaN/Inf JSON constants — tuned for robustness against weak policies.

## Data samples

See [data_samples/README.md](data_samples/README.md) for the two shipped forms:

- **`grpo_train_prompts/`** — 8 prompt-only rows (the standard swift-format GRPO
  dataset consumed by `--dataset`), solution bbox kept for the reward.
- **`rollout_trajectories/`** — 32 complete multi-turn conversations logged
  from an actual GRPO run (`--log_completions true`): 4 prompts x 8
  generations, with per-trajectory rewards and GRPO advantages recorded.
  Standard swift `messages` JSON — directly usable for inspection,
  distillation, or SFT-style re-use.

The rollout JSONL contains **32 distinct conversation records**, grouped by
`prompt_group=0–3`, each with `generation=0–7`. The four original images are
shared inputs for these groups; each generation records its own observation
trajectory, reward values, and advantage. The records were taken from training
step 1. The accompanying 47 crop images were reconstructed offline from the
saved crop metadata using the training protocol; see the sample guide for
anonymization and reconstruction details.

## Serving / inference: OpenAI-compatible endpoints

During RL training, multi-turn rollout requests run over ms-swift's own
rollout-server protocol; the zoom-in environment (crop scheduler) executes
inside that server. For **serving after training**, vLLM exposes a standard
OpenAI-compatible HTTP API, so you can drive the model with the plain OpenAI
SDK (or any OpenAI-compatible client) — no proprietary client needed. Serve the
LBS checkpoint after SFT and Agentic RL on Qwen3.8-27b; set `model` to its actual
served ID from `/v1/models` (the example uses the alias `evaluated-model`):

Run the following first-turn example from the repository root after configuring
`evaluation/.env.run` and supplying `screenshot.png`. It constructs the system
prompt with the shipped protocol; subsequent crop actions still require the client
loop described below.

```python
import base64
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

sys.path.insert(0, str(Path("agentic_gui_grounding_rl/plugins/grounding_zoom_in").resolve()))
from zoomin_grounding_protocol import build_system_prompt

load_dotenv("evaluation/.env.run")
client = OpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
)
with Image.open("screenshot.png") as image:
    width, height = image.size
system_prompt = build_system_prompt(
    instruction="Open the File menu",  # replace with the grounding task
    image_path="screenshot.png",
    width=width,
    height=height,
    max_tool_calls=3,
)

with open("screenshot.png", "rb") as f:
    data_url = "data:image/png;base64," + base64.b64encode(f.read()).decode()

resp = client.chat.completions.create(
    model=os.environ["MODEL"],  # actual served ID from /v1/models
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": "image_path: screenshot.png"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ],
    max_tokens=2560,
)
print(resp.choices[0].message.content)   # e.g. "[798.23,457.17]"
```

The `messages` rows in `data_samples/rollout_trajectories/` use exactly this
OpenAI chat structure, which makes the trajectories directly replayable
against any OpenAI-compatible endpoint for evaluation or distillation.

Note that a plain OpenAI serving endpoint executes only the *model*, not the
crop environment: the (optional) tool loop must be driven by your client —
parse `<action>crop_image(...)</action>` from the response, crop with PIL
(`zoomin_grounding_protocol.execute_crop`), and append the result message;
when serving through ms-swift's rollout server instead, the scheduler does
this for you.

## Multi-node topology & compatibility notes

- Topology: first `ROLLOUT_NNODES` nodes each run a dedicated vLLM rollout
  server (TP x DP); the remaining nodes run the GRPO trainer and connect
  through `--vllm_server_host/--vllm_server_port` lists. `NNODES=1` degrades to
  a single-node colocated smoke test.
- Serving multiple rollout servers to one trainer (space-separated host/port
  lists) requires an ms-swift build with a multi-server inference client +
  parallel weight sync. Community ms-swift releases generally accept a single
  server — keep `ROLLOUT_NNODES=1` unless you know your build supports lists.
- Weights sync tuning knobs (`SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE`,
  `--move_model_batches`) are exported/forwarded by the script and safe to
  adjust for slow interconnects.
- The recorded training run used Ascend NPUs with vLLM server-mode rollouts.
  The launcher also supports GPU configurations; validate your hardware and
  framework versions with a smoke test. `USE_ASCEND_NPU=1` enables NPU-specific
  tuning. Some multi-server/NPU configurations require ms-swift features or
  patches that are not included in this release.

## Known limitations

- On shared filesystems, training log names do not include node rank; use a
  separate `LOG_DIR` per training node to retain each node's output.
- `VLLM_MAX_NUM_BATCHED_TOKENS` is declared but not forwarded by the launcher;
  use the existing explicit `VLLM_ENGINE_KWARGS` when that setting is needed.
  In multi-node mode, the launcher regenerates `VLLM_SERVER_PORTS` from
  `VLLM_SERVER_PORT`, overriding a supplied per-server list.
- All shaping rewards except format pay only on correct answers, so an
  all-wrong GRPO group produces almost no gradient (the known cost of
  outcome gating).
- All assistant turns share one trajectory-level advantage, so the quality of
  individual crop decisions cannot be separated from the final outcome.
- The parser's last-resort fallback (take the last coordinate in the text as
  the final answer) weakens the formatting constraint.
- The judge score is a heuristic proxy for reasoning quality and sits on the
  reward critical path — a slow judge endpoint slows the training step; its
  weight (0.3) and accuracy gating cap the influence radius.

## Requirements

- Python >= 3.9, ms-swift, vLLM, DeepSpeed, Pillow; `openai` only when the
  judge reward is enabled
- Size device memory and topology for the Qwen3.8-27b SFT checkpoint, image
  resolution, context length, and rollout concurrency. The launcher generates
  a DeepSpeed ZeRO-3 configuration with optimizer offload; the 8-row smoke
  dataset reduces the data volume, not the model memory requirement.

## License

The repository-root [Apache License 2.0](../LICENSE) covers this component.
Portions of the plugin implementation are adapted from
the open-source verl zoom-in grounding recipe (also Apache-2.0). Sample screenshots retain their original ownership and terms;
see [NOTICE](../NOTICE).

## Judge prompt language

The optional CoT reward retains its original Chinese runtime prompt and criteria.
An English reading aid is provided in [judge protocol notes](../docs/JUDGE_PROTOCOL.md).
It is documentation only; changing the runtime prompt would change the reward input.

## Sample terms

Sample screenshots retain their original ownership and terms. The repository's
Apache-2.0 software license does not relicense screenshot content; source
attribution is retained in [NOTICE](../NOTICE).
