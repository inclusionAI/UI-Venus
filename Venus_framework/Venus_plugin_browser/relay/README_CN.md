# Venus 本地 relay

[English](README.md)

这个 relay 将插件请求从：

```text
http://127.0.0.1:8765/v1/chat/completions
```

转发到通过 `VENUS_UPSTREAM_BASE` 配置的上游，例如：

```text
https://example.com/v1/chat/completions
```

它只监听 `127.0.0.1`，不会把浏览器的 `Origin`、Cookie 或 Referer 转发到模型上游，并且只允许指定的 Chrome 插件 Origin 调用。它只负责可选的模型 API 转发，文件 workspace 不依赖 relay。

## 运行

进入本目录运行：

```bash
VENUS_UPSTREAM_BASE="https://example.com/v1" ./start.sh
```

脚本会要求输入 `chrome://extensions` 页面中 Venus 插件的 32 位 ID，然后在 `127.0.0.1:8765` 启动 relay。它不需要证书、hosts 配置或 sudo。按 `Ctrl+C` 停止服务。

## 插件配置

```yaml
llm:
  api_url: "http://127.0.0.1:8765/v1"
  api_key: "empty"
  model: "your-vision-model"
```

## 更换上游

每次启动 relay 时，都需要用 `VENUS_UPSTREAM_BASE` 指定 OpenAI-compatible base URL。

如需由 relay 使用独立密钥，可直接运行 `venus_relay.py` 并传入 `--upstream-api-key`。默认情况下，它会原样转发插件发送的 Authorization 请求头。
