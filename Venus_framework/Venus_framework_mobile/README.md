# UI-Venus-2 Mobile Agent Framework

[中文](./README_CN.md)

Android automation framework for deploying UI-Venus-2 as an autonomous mobile agent.

Run the following commands from the repository root before using the relative commands in this document:

```bash
cd Venus_framework/Venus_framework_mobile
```

## Features

- 🤖 Vision-language model based intelligent decision making
- 📱 Support for 40+ mainstream Chinese applications
- 🔄 Multi-device parallel batch execution
- 📊 Complete trajectory recording and replay
- 🔍 Optional pre-action reflection supervision and correction
- 🔁 Intelligent repeated action detection to avoid infinite loops

> ⚠️ This project is for research and educational purposes only.

---

## Setup

### 1. Python Environment

Python 3.10 or later is required.

### 2. Install ADB

Download ADB Platform Tools (Google official Android SDK Platform Tools) and add to PATH:

```bash
export PATH=${PATH}:~/Downloads/platform-tools
```

### 3. Phone Configuration

**Enable Developer Mode:**
1. Go to `Settings → About Phone → Build Number`, tap 7-10 times
2. "Developer mode enabled" message appears

**Enable USB Debugging:**
1. Go to `Settings → Developer Options`
2. Enable `USB Debugging`
3. Some devices also require `USB Debugging (Security Settings)`

### 4. Install ADB Keyboard

Download and install ADB Keyboard APK (search senzhk/ADBKeyBoard on GitHub).

```bash
adb shell ime enable com.android.adbkeyboard/.AdbIME
```

---

## Installation

```bash
pip install -r requirement.txt
adb devices  # Verify connection
```

---

## Configuration

Edit `config/ui_venus_2_single.yaml`:

```yaml
policy:
  type: "ui_venus_2"
  params:
    model_url: "http://your-model-server/v1"
    model_name: "model"
    temperature: 0.1
    n_img: 0
```

**Deploy with vLLM:**

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --served-model-name model \
  --model /path/to/ui-venus-2 \
  --port 8000
```

---

## Usage

### Single Task

```bash
python main.py \
  --device-id "192.168.1.100:5555" \
  --purpose "Open Xiaohongshu, search for fashion posts" \
  --trace-dir "record/traces/"
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|:--------:|
| `--config` | Config file path | No |
| `--device-id` | Device ID (IP:port or serial) | No |
| `--purpose` | Task description | ✅ |
| `--trace-dir` | Trajectory save directory | ✅ |
| `--step-limit` | Maximum steps | No |
| `--model-host` | Model server URL | No |
| `--model-url` | OpenAI-compatible model API URL | No |
| `--model-name` | Model name | No |
| `--log-file` | Log file path | No |
| `--reflection` | Enable pre-action reflection supervision | No |
| `--reflection-config` | Reflection configuration file; defaults to `config/reflection.yaml` | No |

### Batch Tasks

**1. Edit task list** `data/purpose.txt` (one task per line):

```
Open Weibo, search for Hangzhou weather
Open Meituan, search for nearby hotpot restaurants
```

**2. Configure devices** `config/config_multi.yaml`:

```yaml
devices:
  - "192.168.1.100:5555"
  - "192.168.1.101:5555"

ep_config:
  step_limit: 30

policy:
  type: "ui_venus_2"
  params:
    model_url: "http://your-model-server/v1"
    model_name: "model"

record_config:
  save_dir: "record/batch/"

trace_dir: "record/traces/"
single_task_config: "config/ui_venus_2_single.yaml"
```

**3. Run:**

```bash
python batch_runner.py
```

**Output:**

```
record/batch/
├── task_0/
│   └── task.log          # Execution log (with model thinking)
├── task_1/
│   └── ...
└── batch_report_*.json   # Execution report

record/traces/
└── <episode_id>/
    ├── screenshots/      # Step screenshots
    └── trajectory.pkl.gz # Complete trajectory

logs/
└── batch_runner.log      # Batch execution log
```

### Reflection Supervision

Reflection is disabled by default. When enabled, a supervisor model reviews each candidate action before it is sent to the device. The review uses the task goal, current screenshot, candidate action, agent reasoning, and recent step context.

- `CORRECT` and `EXPLORATORY` actions continue to execution.
- `INCORRECT` and `INEFFECTIVE` actions are returned to the policy as feedback so it can generate a revised action.
- Each step can be reviewed up to `max_retries` times.
- Review details and rejected candidates are saved in each step's `reflection_history` field inside the episode's `trajectory.pkl.gz`.

Configure the supervisor endpoint in `config/reflection.yaml`. It may use the same model service as the policy or a separate OpenAI-compatible vision model:

```yaml
reflection:
  params:
    model_url: "http://your-model-server/v1"
    model_name: "model"
    api_key: ""
    temperature: 0.0
    scale_factor: 1000
    image_window: 3
    max_retries: 3
```

| Field | Description |
|---|---|
| `model_url` | OpenAI-compatible endpoint used by the supervisor |
| `model_name` | Vision model served by the endpoint |
| `api_key` | API key; if empty, falls back to `API_KEY`, `MODEL_API_KEY`, then `OPENAI_API_KEY` |
| `temperature` | Supervisor sampling temperature |
| `scale_factor` | Coordinate scale described to the supervisor, normally `1000` |
| `image_window` | Positive number of recent screenshots retained in the supervisor conversation |
| `max_retries` | Maximum reflection review cycles for each environment step |

Enable reflection for a single task:

```bash
python main.py \
  --purpose "Open Xiaohongshu, search for fashion posts" \
  --trace-dir "record/traces/" \
  --reflection \
  --reflection-config config/reflection.yaml
```

Enable it for every task launched by the batch runner:

```bash
python batch_runner.py \
  --reflection \
  --reflection-config config/reflection.yaml
```

---

## Action Space

### Basic Interaction

| Action | Description | Parameters |
|--------|-------------|------------|
| `Click(point)` | Tap at coordinates | `[x, y]` |
| `DoubleClick(point)` | Double tap | `[x, y]` |
| `LongPress(point)` | Long press | `[x, y]` |
| `Type(content)` | Input text (auto-clear) | text content |

### Scroll & Drag

| Action | Description | Parameters |
|--------|-------------|------------|
| `Swipe(start, end)` | Swipe between two coordinates | `[x, y]` → `[x, y]` |
| `Drag(start, end)` | Drag operation | `[x, y]` → `[x, y]` |

### System Keys

| Action | Description |
|--------|-------------|
| `PressBack()` | Go back |
| `PressHome()` | Go to home |
| `PressEnter()` | Press enter |
| `PressRecent()` | Recent apps |

### App Control

| Action | Description | Parameters |
|--------|-------------|------------|
| `LaunchApp(app)` | Launch app | app name |
| `Wait()` | Wait for loading | - |
| `GetScreenshot()` | Save a screenshot to the device album | - |
| `Answer(content)` | Answer the user | answer text |
| `Finished(content)` | Task completed | result text |
| `CallUser(content)` | Request human takeover | reason |

---

## Supported Applications

41 applications are pre-configured:

| Category | Apps |
|----------|------|
| Social | Weibo, Xiaohongshu, Zhihu, Douban |
| E-commerce | Pinduoduo, Vipshop |
| Food & Delivery | Meituan, Meituan Waimai, Eleme, Dianping |
| Travel | Ctrip, Tongcheng, Railway 12306, Didi, Baidu Maps |
| Video | Bilibili, Kuaishou, Tencent Video, iQiyi, Youku, Mango TV |
| Music | QQ Music, Kuwo, Ximalaya, Qishui Music, Qingting FM |
| News & Reading | Toutiao, Fanqie Novel, Qimao Novel |
| Tools | WPS, Feishu |
| AI Apps | Yuanbao, Doubao, Qianwen |
| Services & Housing | Alipay, 58, Beike, Anjuke |
| Other | Markor, Honkai: Star Rail, Tonghuashun |

See full list: `config/app_mapping.yaml`

---

## Project Structure

```
Venus_framework_mobile/
├── main.py                 # Single task entry
├── batch_runner.py         # Batch task executor
├── requirement.txt         # Dependencies
├── config/
│   ├── ui_venus_2_single.yaml
│   ├── config_multi.yaml
│   ├── reflection.yaml
│   └── app_mapping.yaml
├── app/
│   ├── run_handler.py
│   ├── runtime_context.py
│   └── verify.py
├── device/
│   ├── adb_controller.py
│   └── device_manager.py
├── policy/
│   ├── base_policy.py
│   ├── ui_venus_policy.py
│   └── ui_venus_2_policy.py
├── processor/
│   ├── base_processor.py
│   ├── uivenus_processor.py
│   ├── ui_venus_2_processor.py
│   └── rpa_utils/
│       └── postproc_utils.py
├── verify/
│   └── reflection_supervisor.py
├── utils/
│   └── pickle_utils.py
└── data/
    └── purpose.txt
```

---

## Troubleshooting

### Device Not Found

```bash
adb kill-server && adb start-server
adb devices
```

Check: USB debugging enabled, data cable supports data transfer, "Allow USB debugging" confirmed on phone.

### Text Input Not Working

```bash
adb shell ime set com.android.adbkeyboard/.AdbIME
```

### Task Stuck in Loop

Built-in repeated action detection: auto-terminates after 5 consecutive identical actions (swipe excluded).

---

## Quick Reference

```bash
# ADB connection
adb devices
adb connect 192.168.1.100:5555

# Single task
python main.py --purpose "your task" --trace-dir "record/traces/"

# Batch tasks
python batch_runner.py

# Single task with reflection
python main.py --purpose "your task" --trace-dir "record/traces/" --reflection

# Batch tasks with reflection
python batch_runner.py --reflection

# View logs
tail -f logs/batch_runner.log
```

---

## License

This project is for research and educational purposes only.
