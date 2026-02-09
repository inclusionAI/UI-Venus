# UI-Venus-1.5 Agent Framework

**[中文文档](./README_CN.md)**

Android automation framework for deploying UI-Venus-1.5 as an autonomous mobile agent.

## Features

- 🤖 Vision-language model based intelligent decision making
- 📱 Support for 40+ mainstream Chinese applications
- 🔄 Multi-device parallel batch execution
- 📊 Complete trajectory recording and replay
- 🔁 Intelligent repeated action detection to avoid infinite loops

> ⚠️ This project is for research and educational purposes only.

---

## Setup

### 1. Python Environment

Python 3.10+ recommended.

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

Edit `config/ui_venus_single.yaml`:

```yaml
policy:
  type: "ui_venus"
  params:
    model_host: "http://your-model-server"
    model_port: 8000
    model_name: "model"
    temperature: 0.0
```

**Deploy with vLLM:**

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --served-model-name model \
  --model /path/to/ui-venus-1.5 \
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
| `--log-file` | Log file path | No |
| `--save-dir` | Screenshot save directory | No |

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
  type: "ui_venus"
  params:
    model_host: "http://your-model-server"
    model_name: "model"

record_config:
  save_dir: "record/batch/"

trace_dir: "record/traces/"
single_task_config: "config/ui_venus_single.yaml"
```

**3. Run:**

```bash
python batch_runner.py
```

**Output:**

```
record/batch/
├── task_0/
│   ├── task.log          # Execution log (with model thinking)
│   └── screenshots/
├── task_1/
│   └── ...
└── batch_report_*.json   # Execution report

logs/
└── batch_runner.log      # Batch execution log
```

---

## Action Space

### Basic Interaction

| Action | Description | Parameters |
|--------|-------------|------------|
| `Click(point)` | Tap at coordinates | `[x, y]` |
| `LongPress(point)` | Long press | `[x, y]` |
| `Type(content)` | Input text (auto-clear) | text content |

### Scroll & Drag

| Action | Description | Parameters |
|--------|-------------|------------|
| `Scroll(direction)` | Scroll screen | `up/down/left/right` |
| `Drag(start, end)` | Drag operation | `[x, y]` → `[x, y]` |

### System Keys

| Action | Description |
|--------|-------------|
| `PressBack` | Go back |
| `PressHome` | Go to home |
| `PressEnter` | Press enter |
| `PressRecent` | Recent apps |

### App Control

| Action | Description | Parameters |
|--------|-------------|------------|
| `Launch(app_name)` | Launch app | app name |
| `Wait` | Wait for loading | duration (ms) |
| `Finished` | Task completed | - |
| `CallUser` | Request human takeover | reason |

---

## Supported Applications

50+ mainstream Chinese apps are pre-configured:

| Category | Apps |
|----------|------|
| Social | Weibo, Xiaohongshu, Zhihu, Douban |
| E-commerce | Taobao, Pinduoduo, Vipshop |
| Food & Delivery | Meituan, Eleme, Dianping |
| Travel | Ctrip, Tongcheng, 12306, Didi, Baidu Maps |
| Video | Bilibili, Kuaishou, Tencent Video, iQiyi, Youku |
| Music | QQ Music, Kuwo, Ximalaya, Qishui Music |
| News & Reading | Toutiao, Fanqie Novel, Qimao Novel |
| Tools | WPS, Feishu |
| AI Apps | Yuanbao, Doubao, Qianwen |
| Services | Alipay, 58, Beike, Anjuke |

See full list: `config/app_mapping.yaml`

---

## Project Structure

```
hunter_framework/
├── main.py                 # Single task entry
├── batch_runner.py         # Batch task executor
├── requirement.txt         # Dependencies
├── config/
│   ├── ui_venus_single.yaml
│   ├── config_multi.yaml
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
│   └── qw_utils.py
├── processor/
│   ├── base_processor.py
│   ├── uivenus_processor.py
│   └── rpa_utils/
├── utils/
│   ├── pickle_utils.py
│   └── templates/
├── data/
│   └── purpose.txt
├── logs/
└── record/
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

# View logs
tail -f logs/batch_runner.log
```

---

## License

This project is for research and educational purposes only.
