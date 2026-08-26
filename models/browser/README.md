# Venus Browser

`models/browser/venus_browser.py` is a standalone browser agent. You start a
local Chrome instance with a Chrome DevTools Protocol (CDP) port, then the agent
connects to that browser and completes one natural-language task.

```text
local Chrome + persistent profile <-- CDP --> models/browser/venus_browser.py <-- API --> vision model
```

The script does not launch or close Chrome. This makes browser startup explicit
and allows the same isolated Chrome profile to retain logins between runs.

## 1. Requirements

- Python 3.10 or newer
- Google Chrome, Chromium, or Chrome for Testing
- An OpenAI-compatible multimodal chat-completions endpoint

Install the Python dependencies:

```bash
python -m pip install openai playwright
```

`playwright install chromium` is not required because the script connects to an
already-running browser instead of launching Playwright's bundled Chromium.

## 2. Start a local browser with CDP

Use a dedicated profile directory. Do not point `--user-data-dir` at your normal
Chrome profile. Chrome 136 and newer require remote-debugging switches to use a
non-default data directory.

### Linux

```bash
mkdir -p "$PWD/.venus-mini-profile"

google-chrome \
  --remote-debugging-port=9222 \
  --remote-debugging-address=127.0.0.1 \
  --user-data-dir="$PWD/.venus-mini-profile" \
  --window-size=1024,768 \
  --no-first-run
```

If the executable is named `chromium` or `chromium-browser`, replace
`google-chrome` accordingly.

### macOS

```bash
mkdir -p "$PWD/.venus-mini-profile"

"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9222 \
  --remote-debugging-address=127.0.0.1 \
  --user-data-dir="$PWD/.venus-mini-profile" \
  --window-size=1024,768 \
  --no-first-run
```

### Windows PowerShell

```powershell
$profile = Join-Path $PWD ".venus-mini-profile"
New-Item -ItemType Directory -Force -Path $profile | Out-Null

$chrome = "$env:ProgramFiles\Google\Chrome\Application\chrome.exe"
Start-Process $chrome -ArgumentList @(
  "--remote-debugging-port=9222",
  "--remote-debugging-address=127.0.0.1",
  "--user-data-dir=$profile",
  "--window-size=1024,768",
  "--no-first-run"
)
```

For a headless browser, add `--headless=new` to the Chrome command. The agent
does not need a separate headless option because this is controlled when Chrome
starts.

## 3. Verify the CDP endpoint

On Linux or macOS:

```bash
curl http://127.0.0.1:9222/json/version
```

On Windows PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:9222/json/version
```

The response should contain browser metadata and a `webSocketDebuggerUrl`.
Chrome also exposes the protocol description at
`http://127.0.0.1:9222/json/protocol`.

## 4. Configure the model

Linux or macOS:

```bash
export CDP_URL="http://127.0.0.1:9222"
export LLM_API_URL="https://your-openai-compatible-endpoint/v1"
export LLM_API_KEY="your-api-key"
export LLM_MODEL="your-vision-model"
```

Windows PowerShell:

```powershell
$env:CDP_URL = "http://127.0.0.1:9222"
$env:LLM_API_URL = "https://your-openai-compatible-endpoint/v1"
$env:LLM_API_KEY = "your-api-key"
$env:LLM_MODEL = "your-vision-model"
```

`CDP_URL` defaults to `http://127.0.0.1:9222`. Local endpoints that do not
validate API keys may omit `LLM_API_KEY`. Optional variables:

```bash
export LLM_MAX_TOKENS=4096
export LLM_THINKING=true
```

## 5. Run one task

The unified entry point is executed from the repository root:

```bash
bash scripts/browser.sh \
  "Open https://example.com and report the page title"
```

Configure `MODEL_URL`, `MODEL_NAME`, `API_KEY`, `OUTPUT_DIR`, `CDP_URL`, and other options at the top of `scripts/browser.sh` or override them with environment variables. The script maps the common model variables to the standalone implementation's `LLM_*` variables.

To call the Python entry point directly:

Include the initial URL in the task when possible:

```bash
python models/browser/venus_browser.py \
  "Open https://example.com and report the page title"
```

The task can also be passed without shell quotes:

```bash
python models/browser/venus_browser.py Open https://example.com and report the page title
```

Useful options:

```bash
python models/browser/venus_browser.py --max-steps 50 --output ./runs "your task"
```

If the task contains no URL, the agent starts at Google. If Chrome already has
tabs, the agent uses the most recently opened tab.

## 6. Output

Each run creates a timestamped directory:

```text
results/browser/20260817_120000_000000/
├── screenshots/
│   ├── step_001.png
│   ├── step_002.png
│   └── ...
├── final.png
├── history.jsonl
└── result.json
```

`history.jsonl` contains every model response, parsed action, action arguments,
URL, screenshot path, notes, token counts, and execution error. Failed actions
and their screenshots are retained, and the error is sent to the model in the
next user message.

`result.json` contains the final answer and whether the model emitted
`Finished(...)`. The process exits with code `0` when finished and `2` when the
maximum step count is reached without completion.

Disconnecting the agent does not close Chrome. Close the dedicated Chrome
window manually when it is no longer needed.

## 7. Troubleshooting and security

- **Connection refused:** confirm Chrome is running with port `9222`, then check
  `/json/version` before starting the agent.
- **CDP flag appears ignored:** use a dedicated non-default `--user-data-dir`.
- **The browser opens with no login:** log in once inside the dedicated Chrome
  window; that profile will be reused on later starts.
- **The model cannot act reliably:** confirm the selected model accepts image
  inputs and returns the prompt's `<think>/<action>` format.
- **Protect the CDP port:** keep it bound to `127.0.0.1`. CDP grants extensive
  control over the browser, its pages, and the dedicated profile. Do not expose
  it directly to an untrusted network.
- Do not commit `.venus-mini-profile/`, run artifacts, API keys, or cookies.

References: [Chrome remote-debugging security guidance](https://developer.chrome.com/blog/remote-debugging-port),
[Chrome DevTools Protocol HTTP endpoints](https://chromedevtools.github.io/devtools-protocol/).

## Chrome Extension

Venus also provides a Chrome extension version that runs the Browser Agent
directly inside Chrome. Its main features include:

- A Side Panel for entering tasks, managing conversations, and configuring the model
- A screenshot-driven Agent loop that operates the user-authorized active tab
- Support for OpenAI-compatible multimodal model APIs
- Persistent conversation history with bounded-context summarization
- Configurable step limits, retry handling, manual stopping, and automatic debugger detachment

See the [Chrome extension README](../../Venus_framework/Venus_plugin_browser/README.md)
for installation, configuration, usage, and implementation details.
