# UI-Venus-2 Computer

[中文](README_CN.md)

`computer_example.py` runs multi-turn UI-Venus-2 inference over a prerecorded
sequence of desktop screenshots. It safely parses Computer actions with Python's
AST, validates their basic schema, and converts normalized coordinates in the
0–999 range to screenshot pixels. The example has no dependency on an external
desktop runtime and does not execute predicted actions.

## CLI

The input JSON uses the same structure as the Mobile example:

```json
{
  "task": "Open Settings and inspect the display resolution",
  "screenshots": ["screenshots/step_001.png", "screenshots/step_002.png"]
}
```

The repository includes a one-step desktop example at
`models/computer/examples/example_input.json`. Screenshot paths may be absolute
or relative to the JSON file.

Run the unified entry point from the repository root:

```bash
MODEL_URL=http://127.0.0.1:8000/v1 \
MODEL_NAME=UI-Venus-2 \
N_IMG=2 \
bash scripts/computer.sh
```

Pass another input file as the first argument or through `INPUT_FILE`.

Alternatively, invoke the Python entry point directly:

```bash
python models/computer/computer_example.py \
  --model-url http://127.0.0.1:8000/v1 \
  --model-name UI-Venus-2 \
  --input-file /path/to/input.json \
  --output-file results/computer/output.json
```

You can also provide a task and one or more screenshots without an input JSON:

```bash
python models/computer/computer_example.py \
  --model-url http://127.0.0.1:8000/v1 \
  --model-name UI-Venus-2 \
  --task "Open Settings" \
  --screenshot /path/to/step_001.png \
  --screenshot /path/to/step_002.png
```

`--n-img` controls how many recent historical screenshots are included in
addition to the current screenshot. The current screenshot and all accepted
assistant text are always retained. If action parsing fails, the request is
retried once by default with the exact same messages. Rejected responses are
never added to the conversation history.

## Python API

`parse_action_call()`, `parse_response()`, `normalized_point()`,
`normalize_action()`, and `build_messages()` are side-effect-free public
functions. `VenusComputerAgent.infer(task, screenshot)` accepts a PNG path or
PNG bytes and returns the current thought, raw action, and JSON-safe
`parsed_action`.

`Finished` maps to `terminal=success`, while `CallUser` maps to
`terminal=needs_user`. Unknown, malformed, or incomplete actions raise
`ComputerActionError` explicitly.
