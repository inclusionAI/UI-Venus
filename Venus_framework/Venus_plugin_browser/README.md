# Venus Browser Agent Extension

[中文](README_CN.md)

A minimal Browser Agent demo that runs entirely inside a Chrome extension. After configuring an OpenAI Chat Completions-compatible vision model endpoint, users can assign tasks in the Side Panel. With explicit authorization, the extension repeatedly captures the current tab, requests a model action, and executes that GUI action.

## Capabilities

- Side Panel chat and settings UI
- API URL, model, API key, maximum step, and temperature settings
- HTTP(S) host access for the Page Assistant and configured model endpoint
- Control of the active tab through `chrome.debugger`
- A screenshot-driven, extension-only agent loop
- A manually controlled Page Assistant toggle at the top of the Side Panel. The setting is stored locally. When enabled, an input appears at the bottom of web pages; it hides when the Side Panel closes and returns according to the saved setting. During a run, it streams a summary of the current step's thinking and provides a stop button. After `Finished` or `CallUser`, it shows a dismissible result card.
- Shared task validation and lifecycle behavior across Page Assistant and the Side Panel. Missing API, model, workspace, or permission settings produce consistent prompts, and the input is restored after stops or runtime errors.
- Persistent multi-conversation transcripts, history switching, and restoration after reopening the Side Panel
- Bounded context similar to Claude Code: a compressed summary, recent text trajectory, and the current screenshot
- A default limit of 100 steps with stop-at-any-time support. The debugger attaches only while the agent is running and detaches immediately after completion, a stop, or Side Panel closure.
- `Upload` and `Download` actions for file selection and naming. Both use the task directory's `workspace/`.
- Unlimited retry of action-model requests after 408/409/425/429, 5xx, timeout, network, or non-JSON successful responses, until the request succeeds or the user stops it

## Installation

1. Open `chrome://extensions`.
2. Enable **Developer mode** in the upper-right corner.
3. Click **Load unpacked**.
4. Select this `Venus_plugin_browser` directory.
5. Click the Venus toolbar icon to open the Side Panel.

The extension requires the `debugger` permission to perform real mouse, keyboard, and screenshot operations. It does not take control during installation or merely while Page Assistant is visible. It attaches to the selected tab only after a task starts from the Side Panel or Page Assistant and detaches as soon as the task ends.

## Conversations and Context

- Every conversation has a separate ID. User tasks, model analysis, actions, and execution results are continuously stored in IndexedDB.
- Running another task appends to the active conversation by default. Use the conversation selector to restore an older conversation or **New conversation** to start with clean context.
- The complete transcript is retained locally for UI restoration, but old screenshots are not persisted.
- Model requests do not replay the complete transcript. They contain a compressed summary, limited recent text history, and the current screenshot.
- Each action request includes at most three observation screenshots: the preceding two steps and the current screenshot. User-provided reference images are additional; each message accepts up to four images, with a 10 MiB limit per image and a 20 MiB combined limit.
- A task replays at most the latest 30 action rounds. When the previous request's prompt usage exceeds 32,000 tokens, Venus asks the current model to compact all unsummarized conversation records before the next task.
- **Delete** permanently removes the selected conversation after confirmation.

## Model Configuration

Configure:

- **API URL:** Either a `/v1` base URL or a complete `/chat/completions` URL.
- **Model:** The name of a model that supports visual input.
- **API Key:** Stored only in `chrome.storage.session` by default and cleared after a browser restart or extension reload.
- **Maximum steps:** The maximum number of agent steps per task, from 1 to 200; default 100.
- **Temperature:** Sampling temperature for normal action requests, from 0 to 2; default 0. Connection tests and context compression always use 0.

The extension sends OpenAI-compatible Chat Completions requests:

```json
{
  "model": "your-vision-model",
  "messages": [],
  "temperature": 0
}
```

When **Remember Key on this device** is selected, the key is stored in `chrome.storage.local`. Because the extension is a client application, it cannot provide server-grade secret protection; this option is recommended only for personal local demos.

### Local Relay

If a model gateway rejects the `chrome-extension://` origin, use the local loopback relay under [`relay/`](relay/):

```bash
cd relay
VENUS_UPSTREAM_BASE="https://example.com/v1" ./start.sh
```

After it starts, set the API URL to `http://127.0.0.1:8765/v1`. On the first run, enter only the current extension ID; certificates, hosts-file changes, and sudo are unnecessary. See the [`relay` documentation](relay/README.md) for details.

### Upload and Download Workspace

Automatic file handling uses the Chrome File System Access API and does not depend on the model relay:

- Before first use, click **Choose directory** in extension settings, select the task's `workspace/`, and grant read/write access. The directory handle is stored in the extension's IndexedDB. You must authorize it again if Chrome revokes access.
- Before an upload, place files in the authorized directory. After the model clicks a web file control, the next user message contains an inventory with relative paths, sizes, and MIME types. The model must then output `Upload(file='relative/path')` explicitly.
- After the model clicks a download button, the extension intercepts the download URL. The model then outputs `Download(filename='final-name.ext')`, and the Service Worker writes the response stream directly to the authorized directory.
- For documents such as arXiv `/pdf/<id>` links that normally open inline, the click executor intercepts the URL without opening Chrome PDF Viewer. After `Download(filename=...)`, the Service Worker streams the file into the authorized directory, avoiding the operating system's Save As window.
- Uploaded files must appear in the inventory. A download name must be one file name and cannot contain a directory or `..`. Existing files are never overwritten.
- Extension-only uploads are limited to 64 MiB and downloads to 256 MiB.

The extension can read and write only the directory explicitly authorized by the user. The model API may use either a remote endpoint or the optional relay; these choices are independent.

## Limitations

- The extension controls only the `http://` or `https://` tab selected when a task starts, plus tabs created by the agent.
- It does not support `chrome://` pages, the Chrome Web Store, other extension pages, or native operating-system windows.
- Closing the Side Panel, clicking Stop, or reloading the extension stops the active session and detaches the debugger.
- Model output must contain `<action>...</action>` and use `point` coordinate arguments.
- DOM trees, a separate reflection supervisor, and trajectory ZIP export are not currently included. The agent only adds a loop-recovery warning after two identical consecutive responses.

## Development Checks

After the first checkout, install test dependencies and run the checks:

```bash
npm ci
npm test
npm run check
```

## Directory Structure

```text
manifest.json
page-assistant.js            # In-page task composer and running state
service-worker.js           # Browser attachment, screenshots, and action execution
sidepanel.html/css/js       # UI and settings
assets/icons/               # Extension icons
src/
  action-parser.js          # Strict Venus action parser without eval
  agent-session.js          # Agent loop
  browser-bridge.js         # Side Panel to Service Worker RPC
  config-validation.js      # Task configuration validation
  context-manager.js        # Context selection and compaction planning
  conversation-store.js     # IndexedDB transcripts and conversation metadata
  file-transfer.js          # Upload and download path handling
  hotkey.js                 # Cross-platform hotkey normalization
  model-client.js           # OpenAI-compatible model client
  settings.js               # Permissions and credential storage
  workspace-store.js        # Authorized workspace persistence
prompts/venus_system.txt
tests/                      # Unit and contract tests
scripts/check.mjs           # Static extension checks
package.json/package-lock.json
relay/                      # Local model forwarding on 127.0.0.1
workspace/.gitkeep          # Empty workspace placeholder
```
