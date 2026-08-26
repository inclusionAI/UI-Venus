# Single-image CAPTCHA vLLM Inference and Visualization

[中文](README_CN.md)

This directory uses a single-image workflow rather than batch evaluation:

- `infer_captcha.py` runs vLLM inference on one image and parses its actions.
- `captcha_prompt.py` stores the default `SYS_PROMPT` and `USER_PROMPT`.
- `visualize_captcha.py` reads one inference JSON result and generates a separate visualization HTML file.

Run the unified entry point from the repository root:

```bash
bash scripts/captcha.sh
```

The model URL, model name, API key, input image, output paths, and thinking toggle are defined at the beginning of `scripts/captcha.sh`. Environment variables with the same names override them. By default, the script uses a test image from `examples/assets/` and writes results under `results/captcha/`.

The inference action DSL is:

```text
<think>Brief analysis</think>
<action>Click(box=(310,456)),Click(box=(520,640))</action>
```

Supported actions are Click, LongPress, Type, and Drag:

```text
Click(box=(x,y))
Click(box=(x,y))Type(content='text')
LongPress(box=(x,y))
Drag(start=(x1,y1),end=(x2,y2))
```

## Environment

The client sends requests through the vLLM OpenAI-compatible API and uses Pillow to read image dimensions. It does not load a vLLM model in the current process and does not depend on the OpenAI SDK.

Relevant official documentation:

- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
- [vLLM multimodal Base64 image input](https://docs.vllm.ai/en/latest/examples/generate/multimodal/)

## Infer on One Image with the vLLM API

Start a server, for example:

```bash
vllm serve path/to/vision-model \
  --served-model-name captcha-model \
  --tensor-parallel-size 2
```

Test one bundled image directly:

```bash
python models/captcha/infer_captcha.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model captcha-model \
  --image models/captcha/examples/assets/jiusuoge_5238.png \
  --enable-thinking \
  --output results/captcha/result.json
```

If the service requires authentication, pass `--api-key` or set `OPENAI_API_KEY`.

Each run sends one image and produces one JSON object. It does not scan directories or construct batches. The script supports only vLLM API calls, and the unified entry point writes to `results/captcha/result.json` by default.

## Coordinates and Prompts

The default prompts are:

- `SYS_PROMPT`: The complete GUI Agent + CAPTCHA-Specific Extension text from the dataset.
- `USER_PROMPT`: `&lt;image&gt;`.

Both constants are defined in `captcha_prompt.py`; `infer_captcha.py` only imports and uses them.

When the OpenAI multimodal message is built, the `&lt;image&gt;` placeholder is replaced with exactly one Base64 `image_url`. No additional Chinese user text or literal duplicate `&lt;image&gt;` is sent.

Thinking is explicitly enabled by default. API requests include:

```json
{"chat_template_kwargs": {"enable_thinking": true}}
```

Pass `--no-enable-thinking` for a comparison with thinking disabled.

The dataset system prompt fixes normalized coordinates to the 0-999 range, so `coord-scale` defaults to 999. When using another coordinate system, provide a matching prompt through `--system-prompt` as well.

You can customize the single-image task and system prompt:

```bash
python models/captcha/infer_captcha.py ... \
  --task 'Identify the CAPTCHA requirements and provide all required actions' \
  --system-prompt 'Custom system prompt' \
  --coord-scale 999
```

## Single-image Result

`models/captcha/examples/test_result.json` is a single output example retained in the repository. Its image path is relative to the JSON file:

```json
{
  "image": "assets/jiusuoge_5238.png",
  "image_size": [480, 847],
  "coord_scale": 999,
  "enable_thinking": true,
  "task": "<image>",
  "model_output": "<action>Click(box=(237,413))</action>",
  "reasoning_content": null,
  "inference": {
    "backend": "vllm-api",
    "model": "captcha-model"
  },
  "parsed_actions": [
    {"type": "Click", "x": 237, "y": 413}
  ]
}
```

`infer_captcha.py` contains no scoring or visualization logic.

## Standalone Visualization

After inference, generate HTML with the separate visualization script:

```bash
python models/captcha/visualize_captcha.py \
  --result results/captcha/result.json \
  --output results/captcha/result.html
```

The HTML embeds the source image and maps action coordinates back to it using `coord-scale`:

- Click: orange numbered point.
- LongPress: blue numbered point.
- Drag: purple arrow.
- Type after Click/LongPress: text label at that point.
- Type without a preceding point: separate entry in the action panel.

Override the source image if it has moved:

```bash
python models/captcha/visualize_captcha.py \
  --result results/captcha/result.json \
  --image models/captcha/examples/assets/jiusuoge_5238.png \
  --output results/captcha/result.html
```

## Use Another Test Image

Each invocation still selects only one image:

```bash
python models/captcha/infer_captcha.py ... \
  --image models/captcha/examples/assets/slide_4469747273.png

python models/captcha/infer_captcha.py ... \
  --image models/captcha/examples/assets/captchaOper_902440b13d984d2ba0f1b13581b31027_no_zhihu.png

python models/captcha/infer_captcha.py ... \
  --image models/captcha/examples/assets/captchaOper_7f438c2492ae4de28c7cf25beab63ea9.png
```
