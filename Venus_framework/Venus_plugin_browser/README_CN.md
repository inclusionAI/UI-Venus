# Venus Browser Agent Extension

[English](README.md)

一个完全运行在 Chrome 插件中的最小 Browser Agent Demo。用户配置一个兼容 OpenAI Chat Completions 的视觉模型接口后，可以在 Side Panel 中下达任务；插件会在用户明确授权的当前标签页中循环截图、请求模型并执行 GUI action。

## 当前能力

- Side Panel 对话与设置 UI
- API URL、Model、API Key、最大步数和 Temperature 配置
- 页面助手和已配置模型接口所需的 HTTP(S) host 访问权限
- `chrome.debugger` 接管当前标签页
- 截图驱动的纯插件 Agent loop
- Side Panel 顶部可手动开关“页面助手”，选择会保存在本机；开启时在网页底部显示输入框，关闭 Side Panel 时同步隐藏，重新打开后按已保存的开关恢复；运行时流式展示当前 step 的 think 摘要和停止按钮，`Finished` / `CallUser` 后展示可关闭结果卡片
- 页面助手与 Side Panel 共用任务检查和生命周期：API、模型、Workspace 或权限缺失时显示相同提示，停止及运行错误后都会恢复输入框
- 多会话 transcript 持久化、历史切换及 Side Panel 重开恢复
- 类 Claude Code 的有界上下文：压缩摘要 + 近期文本轨迹 + 当前截图
- 默认最多 100 步，支持随时停止；仅在 Agent 执行任务期间 attach，任务结束、停止或关闭 Side Panel 后立即 detach
- `Upload` / `Download` action 处理文件选择与命名；上传、下载统一使用任务目录下的 `workspace/`
- Action 模型请求遇到 408/409/425/429、5xx、超时、网络异常或非 JSON 成功响应时无限重试，直到成功或用户停止

## 安装

1. 打开 `chrome://extensions`。
2. 开启右上角“开发者模式”。
3. 点击“加载已解压的扩展程序”。
4. 选择本目录 `Venus_plugin_browser`。
5. 点击工具栏中的 Venus 图标打开 Side Panel。

插件需要 `debugger` 权限，这是执行真实鼠标、键盘和截图动作所必需的。它不会在安装或显示页面助手时接管页面；只有用户从 Side Panel 或页面助手运行任务后才会 attach 当前标签页，任务结束后立即 detach。

## 会话与上下文管理

- 每个 conversation 都有独立 ID。用户任务、模型分析、action 和执行结果会持续写入 IndexedDB。
- 再次运行任务默认追加到当前 conversation；可以通过会话选择器恢复旧会话，也可以点击“新会话”获得干净上下文。
- 完整 transcript 保留在本机用于恢复 UI，但不持久化旧截图。
- 模型请求不会重放完整 transcript，只携带压缩摘要、有限的近期文本历史以及当前截图。
- 每次 action 请求最多携带 3 张图：前两个 step 的 observation 截图和当前 step 截图；更老 step 只保留文本/action。
- 未压缩记录超过 60 条或约 48,000 字符时，Venus 会让当前模型总结较老记录，并原样保留最近 24 条。
- “删除”会在确认后永久删除选中的 conversation。

## 模型配置

填写：

- API URL：可以是 `/v1` base URL，也可以是完整的 `/chat/completions` URL。
- Model：支持视觉输入的模型名称。
- API Key：默认只保存在 `chrome.storage.session`，浏览器重启或插件重载后清除。
- 最大步数：每次任务最多执行多少个 Agent step，范围 1～200，默认 100。
- Temperature：正常 action 请求使用的采样温度，范围 0～2，默认 0；连接测试和上下文压缩固定为 0。

使用 OpenAI-compatible Chat Completions 请求格式：

```json
{
  "model": "your-vision-model",
  "messages": [],
  "temperature": 0
}
```

如果勾选“在本机记住 Key”，Key 会写入 `chrome.storage.local`。插件是客户端应用，无法提供服务端级别的密钥保密；只建议用于个人本地 Demo。

### Aistudio 本地 relay

如果模型网关拒绝 `chrome-extension://` Origin，可以使用 [`relay/`](relay/) 中的本地 loopback relay：

```bash
cd relay
VENUS_UPSTREAM_BASE="https://example.com/v1" ./start.sh
```

启动后把 API URL 设置为 `http://127.0.0.1:8765/v1`。首次运行只需输入当前插件 ID，不需要证书、hosts 配置或 sudo。具体说明见 [`relay/README_CN.md`](relay/README_CN.md)。

### 上传与下载 workspace

文件自动处理使用 Chrome File System Access API，不依赖模型 relay：

- 首次使用时，在插件设置中点击“选择目录”，选择任务的 `workspace/` 并授予读写权限。目录句柄保存在插件 IndexedDB 中；Chrome 撤销权限后需要重新授权。
- 上传前，把待上传文件放入已授权目录。模型点击网页文件控件后，插件在下一轮 user message 中提供文件相对路径、大小和 MIME 清单。模型随后必须明确输出 `Upload(file='relative/path')`。
- 模型点击下载按钮后，插件拦截下载 URL。模型随后输出 `Download(filename='final-name.ext')`，Service Worker 直接把响应流写入已授权目录。
- 对 arXiv `/pdf/<id>` 这类默认在浏览器内联预览的文档链接，click 执行器会拦截 URL 而不打开 PDF Viewer；模型给出 `Download(filename=...)` 后，由 Service Worker 直接流式写入已授权目录，因此不会出现 OS“另存为”窗口。
- 上传文件必须来自清单；下载文件名只允许单个文件名，不能包含目录或 `..`。已有同名文件不会被覆盖。
- 纯插件上传限制为 64 MiB，下载限制为 256 MiB。

扩展只能读取和写入用户明确授权的目录。模型 API 可以使用远程地址，也可以选择使用 relay，两者互不影响。

## 使用限制

- 仅接管运行任务时选中的 `http://` 或 `https://` 标签页，以及 Agent 自己新建的标签页。
- 不支持 `chrome://`、Chrome Web Store、其他扩展页面和操作系统原生窗口。
- 用户关闭 Side Panel、点击停止或刷新插件时，当前 session 会停止并 detach。
- 模型输出必须包含 `<action>...</action>`，并使用 `point` 坐标参数。
- 当前不包含 DOM tree、reflection supervisor 和轨迹 ZIP 导出。

## 开发检查

首次检出后安装测试依赖，再运行检查：

```bash
npm ci
npm test
npm run check
```

## 目录结构

```text
manifest.json
service-worker.js           # 浏览器 attach、截图和 action 执行
sidepanel.html/css/js       # UI 与设置
src/
  action-parser.js          # 严格解析 Venus action，不使用 eval
  agent-session.js          # Agent loop
  browser-bridge.js         # Side Panel ↔ Service Worker RPC
  context-manager.js        # 上下文选择与压缩规划
  conversation-store.js     # IndexedDB transcript 与会话元数据
  model-client.js           # OpenAI-compatible model client
  settings.js               # 权限和密钥存储
prompts/venus_system.txt
tests/action-parser.test.mjs
relay/                      # 127.0.0.1 本地模型转发
```
