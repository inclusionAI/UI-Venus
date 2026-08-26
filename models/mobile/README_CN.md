# UI-Venus-2 Mobile Multi-turn Example

[English](README.md)

本目录提供基于预录屏幕截图的多轮 Mobile 推理示例。它会按顺序读取 `examples/example_input.json` 中的截图，使用 OpenAI-compatible API 调用模型，保存每一步的 `think`、`action`、解析动作和原始响应。

在仓库根目录运行：

```bash
bash scripts/mobile.sh
```

模型地址、模型名、API key、生成参数和输入输出路径集中在 `scripts/mobile.sh` 开头，也可以使用同名环境变量覆盖：

```bash
MODEL_URL=http://127.0.0.1:8000/v1 \
MODEL_NAME=UI-Venus-2 \
N_IMG=2 \
bash scripts/mobile.sh
```

`N_IMG` 表示额外携带最近多少轮历史截图，不是请求中的图片总数；当前轮截图始终携带。`N_IMG=0` 时仍保留历史 assistant 文本。

默认输入和输出：

```text
models/mobile/examples/example_input.json
models/mobile/examples/example_output.json
models/mobile/examples/screenshots/
results/mobile/output.json
```

这是离线截图序列推理示例，不会执行预测动作或通过 ADB 获取下一张截图。需要真实设备自动操作、轨迹记录、批量任务或 reflection 时，使用 `Venus_framework/Venus_framework_mobile/`。
