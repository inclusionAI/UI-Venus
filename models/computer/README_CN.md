# UI-Venus-2 Computer

[English](README.md)

`computer_example.py` 基于预录的桌面截图序列运行多轮 UI-Venus-2 推理。它使用 Python AST 安全解析 Computer 动作，验证动作的基本结构，并将 0–999 范围内的归一化坐标转换为截图像素坐标。该示例不依赖外部桌面运行环境，也不会执行预测动作。

## 命令行使用

输入 JSON 与 Mobile 示例采用相同结构：

```json
{
  "task": "Open Settings and inspect the display resolution",
  "screenshots": ["screenshots/step_001.png", "screenshots/step_002.png"]
}
```

仓库在 `models/computer/examples/example_input.json` 中提供了一个单步桌面示例。截图路径可以是绝对路径，也可以相对于 JSON 文件。

从仓库根目录运行统一入口：

```bash
MODEL_URL=http://127.0.0.1:8000/v1 \
MODEL_NAME=UI-Venus-2 \
N_IMG=2 \
bash scripts/computer.sh
```

可以通过第一个位置参数或 `INPUT_FILE` 传入其他输入文件。

也可以直接调用 Python 入口：

```bash
python models/computer/computer_example.py \
  --model-url http://127.0.0.1:8000/v1 \
  --model-name UI-Venus-2 \
  --input-file /path/to/input.json \
  --output-file results/computer/output.json
```

还可以不使用输入 JSON，直接提供任务和一张或多张截图：

```bash
python models/computer/computer_example.py \
  --model-url http://127.0.0.1:8000/v1 \
  --model-name UI-Venus-2 \
  --task "Open Settings" \
  --screenshot /path/to/step_001.png \
  --screenshot /path/to/step_002.png
```

`--n-img` 控制除当前截图外额外携带的最近历史截图数量。当前截图和所有已接受的 assistant 文本始终保留。如果动作解析失败，默认使用完全相同的消息重试一次；被拒绝的响应不会加入对话历史。

## Python API

`parse_action_call()`、`parse_response()`、`normalized_point()`、`normalize_action()` 和 `build_messages()` 都是无副作用的公共函数。`VenusComputerAgent.infer(task, screenshot)` 接受 PNG 路径或 PNG 字节，返回当前思考、原始动作和可 JSON 序列化的 `parsed_action`。

`Finished` 映射为 `terminal=success`，`CallUser` 映射为 `terminal=needs_user`。未知、格式错误或不完整的动作会明确抛出 `ComputerActionError`。
