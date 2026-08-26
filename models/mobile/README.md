# UI-Venus-2 Mobile Multi-turn Example

[中文](README_CN.md)

This directory provides a multi-turn Mobile inference example over prerecorded screenshots. It reads screenshots from `examples/example_input.json` in order, calls a model through an OpenAI-compatible API, and saves each step's `think`, `action`, parsed action, and raw response.

Run it from the repository root:

```bash
bash scripts/mobile.sh
```

The model URL, model name, API key, generation settings, and input/output paths are defined at the beginning of `scripts/mobile.sh`. Environment variables with the same names override those defaults:

```bash
MODEL_URL=http://127.0.0.1:8000/v1 \
MODEL_NAME=UI-Venus-2 \
N_IMG=2 \
bash scripts/mobile.sh
```

`N_IMG` is the number of recent historical screenshots included in addition to the current screenshot, not the total image count in a request. With `N_IMG=0`, historical assistant text is still retained.

Default input and output paths:

```text
models/mobile/examples/example_input.json
models/mobile/examples/example_output.json
models/mobile/examples/screenshots/
results/mobile/output.json
```

This is an offline screenshot-sequence inference example. It does not execute predicted actions or use ADB to capture the next screenshot. For real-device automation, trajectory recording, batch tasks, or reflection, use `Venus_framework/Venus_framework_mobile/`.
