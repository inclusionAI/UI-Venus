# Venus Local Relay

[中文](README_CN.md)

This relay forwards extension requests from:

```text
http://127.0.0.1:8765/v1/chat/completions
```

to the upstream configured through `VENUS_UPSTREAM_BASE`, for example:

```text
https://example.com/v1/chat/completions
```

It listens only on `127.0.0.1`, does not forward the browser's `Origin`, cookies, or referrer to the upstream model, and accepts requests only from the configured Chrome extension origin. The relay is optional and handles only model API forwarding; file workspaces do not depend on it.

## Run

From this directory, run:

```bash
VENUS_UPSTREAM_BASE="https://example.com/v1" ./start.sh
```

The script asks for the 32-character Venus extension ID shown on `chrome://extensions`, then starts the relay at `127.0.0.1:8765`. It requires no certificate, hosts-file changes, or sudo. Press `Ctrl+C` to stop it.

## Extension Configuration

```yaml
llm:
  api_url: "http://127.0.0.1:8765/v1"
  api_key: "empty"
  model: "your-vision-model"
```

## Change the Upstream

Set `VENUS_UPSTREAM_BASE` to the OpenAI-compatible base URL each time the relay starts.

To give the relay its own API key, run `venus_relay.py` directly with `--upstream-api-key`. By default, it forwards the Authorization header sent by the extension unchanged.
