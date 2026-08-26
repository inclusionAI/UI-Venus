# UI-Venus-2 Mobile 智能体框架

[English](./README.md)

基于 UI-Venus-2 的 Android 自动化智能体框架。

本文后续相对路径命令均在 Mobile Framework 目录执行。先从仓库根目录进入：

```bash
cd Venus_framework/Venus_framework_mobile
```

## 功能特性

- 🤖 基于视觉语言模型的智能决策
- 📱 支持 40+ 款主流中文应用
- 🔄 多设备并行批量执行
- 📊 完整的轨迹记录与回放
- 🔍 可选的动作执行前反思监督与纠错
- 🔁 智能重复动作检测，避免死循环

> ⚠️ 本项目仅供研究和学习使用。

---

## 环境准备

### 1. Python 环境

需要使用 Python 3.10 及以上版本。

### 2. 安装 ADB

下载 ADB 平台工具（Google 官方 Android SDK Platform Tools），添加到环境变量：

```bash
export PATH=${PATH}:~/Downloads/platform-tools
```

### 3. 手机配置

**启用开发者模式：**
1. 进入 `设置 → 关于手机 → 版本号`，连续点击 7-10 次
2. 出现"开发者模式已启用"提示

**启用 USB 调试：**
1. 进入 `设置 → 开发者选项`
2. 开启 `USB 调试`
3. 部分机型需要同时开启 `USB 调试(安全设置)`

### 4. 安装 ADB Keyboard

下载安装 ADB Keyboard APK（GitHub 搜索 senzhk/ADBKeyBoard）。

```bash
adb shell ime enable com.android.adbkeyboard/.AdbIME
```

---

## 安装依赖

```bash
pip install -r requirement.txt
adb devices  # 验证连接
```

---

## 配置说明

编辑 `config/ui_venus_2_single.yaml`：

```yaml
policy:
  type: "ui_venus_2"
  params:
    model_url: "http://your-model-server/v1"
    model_name: "model"
    temperature: 0.1
    n_img: 0
```

**使用 vLLM 部署模型：**

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --served-model-name model \
  --model /path/to/ui-venus-2 \
  --port 8000
```

---

## 使用方法

### 单任务执行

```bash
python main.py \
  --device-id "192.168.1.100:5555" \
  --purpose "打开小红书，搜索穿搭帖子" \
  --trace-dir "record/traces/"
```

**参数说明：**

| 参数 | 说明 | 必填 |
|-----|------|:----:|
| `--config` | 配置文件路径 | 否 |
| `--device-id` | 设备 ID（IP:端口 或 序列号） | 否 |
| `--purpose` | 任务描述 | ✅ |
| `--trace-dir` | 轨迹保存目录 | ✅ |
| `--step-limit` | 最大步数 | 否 |
| `--model-host` | 模型服务地址 | 否 |
| `--model-url` | OpenAI 兼容模型 API 地址 | 否 |
| `--model-name` | 模型名称 | 否 |
| `--log-file` | 日志文件路径 | 否 |
| `--reflection` | 启用动作执行前反思监督 | 否 |
| `--reflection-config` | 反思配置文件，默认为 `config/reflection.yaml` | 否 |

### 批量任务执行

**1. 编辑任务列表** `data/purpose.txt`（每行一个任务）：

```
打开微博，搜索杭州天气
打开美团，搜索附近的火锅店
```

**2. 配置多设备** `config/config_multi.yaml`：

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

**3. 运行：**

```bash
python batch_runner.py
```

**输出结构：**

```
record/batch/
├── task_0/
│   └── task.log          # 执行日志（含模型思考过程）
├── task_1/
│   └── ...
└── batch_report_*.json   # 执行报告

record/traces/
└── <episode_id>/
    ├── screenshots/      # 分步截图
    └── trajectory.pkl.gz # 完整轨迹

logs/
└── batch_runner.log      # 批量执行总日志
```

### Reflection 反思监督

Reflection 默认关闭。启用后，监督模型会在候选动作发送到设备之前进行检查。检查信息包括任务目标、当前截图、候选动作、Agent 思考过程和近期步骤上下文。

- `CORRECT` 和 `EXPLORATORY` 动作会继续执行。
- `INCORRECT` 和 `INEFFECTIVE` 动作会作为反馈返回策略，由策略重新生成动作。
- 每个环境步骤最多进行 `max_retries` 次反思检查。
- 每次判断及被拒绝的候选动作都会记录在 episode 的 `trajectory.pkl.gz` 中，对应每一步的 `reflection_history` 字段。

在 `config/reflection.yaml` 中配置监督模型。监督模型可以与策略模型共用服务，也可以使用单独的 OpenAI 兼容视觉模型：

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

| 字段 | 说明 |
|---|---|
| `model_url` | 监督模型使用的 OpenAI 兼容 API 地址 |
| `model_name` | API 服务提供的视觉模型名称 |
| `api_key` | API key；为空时依次读取 `API_KEY`、`MODEL_API_KEY`、`OPENAI_API_KEY` |
| `temperature` | 监督模型采样温度 |
| `scale_factor` | 传递给监督模型的坐标尺度，通常为 `1000` |
| `image_window` | 监督对话中保留的最近截图数量，应为正数 |
| `max_retries` | 每个环境步骤允许的最大反思检查次数 |

为单任务启用 reflection：

```bash
python main.py \
  --purpose "打开小红书，搜索穿搭帖子" \
  --trace-dir "record/traces/" \
  --reflection \
  --reflection-config config/reflection.yaml
```

为 Batch Runner 启动的所有任务启用 reflection：

```bash
python batch_runner.py \
  --reflection \
  --reflection-config config/reflection.yaml
```

---

## 动作空间

### 基础交互

| 动作 | 说明 | 参数 |
|-----|------|------|
| `Click(point)` | 点击坐标 | `[x, y]` |
| `DoubleClick(point)` | 双击坐标 | `[x, y]` |
| `LongPress(point)` | 长按 | `[x, y]` |
| `Type(content)` | 输入文本（自动清空） | 文本内容 |

### 滑动操作

| 动作 | 说明 | 参数 |
|-----|------|------|
| `Swipe(start, end)` | 在两个坐标之间滑动 | `[x, y]` → `[x, y]` |
| `Drag(start, end)` | 拖拽操作 | `[x, y]` → `[x, y]` |

### 系统按键

| 动作 | 说明 |
|-----|------|
| `PressBack()` | 返回上一页 |
| `PressHome()` | 回到桌面 |
| `PressEnter()` | 按回车键 |
| `PressRecent()` | 最近应用 |

### 应用控制

| 动作 | 说明 | 参数 |
|-----|------|------|
| `LaunchApp(app)` | 启动应用 | 应用名称 |
| `Wait()` | 等待加载 | - |
| `GetScreenshot()` | 保存截图到设备相册 | - |
| `Answer(content)` | 回答用户 | 回答文本 |
| `Finished(content)` | 任务完成 | 结果文本 |
| `CallUser(content)` | 请求人工接管 | 原因说明 |

---

## 支持的应用

预配置了 41 款应用：

| 分类 | 应用 |
|-----|------|
| 社交媒体 | 微博、小红书、知乎、豆瓣 |
| 电商购物 | 拼多多、唯品会 |
| 外卖美食 | 美团、美团外卖、饿了么、大众点评 |
| 出行旅游 | 携程、同程、铁路12306、滴滴、百度地图 |
| 视频娱乐 | 哔哩哔哩、快手、腾讯视频、爱奇艺、优酷、芒果TV |
| 音乐音频 | QQ音乐、酷我音乐、喜马拉雅、汽水音乐、蜻蜓FM |
| 资讯阅读 | 今日头条、番茄小说、七猫小说 |
| 办公工具 | WPS、飞书 |
| AI 应用 | 元宝、豆包、千问 |
| 生活服务 | 支付宝、58同城、贝壳找房、安居客 |
| 其他 | Markor、星穹铁道、同花顺 |

完整列表见：`config/app_mapping.yaml`

---

## 项目结构

```
Venus_framework_mobile/
├── main.py                 # 单任务入口
├── batch_runner.py         # 批量任务执行器
├── requirement.txt         # 依赖列表
├── config/
│   ├── ui_venus_2_single.yaml  # 单任务配置
│   ├── config_multi.yaml       # 批量任务配置
│   ├── reflection.yaml         # 反思监督配置
│   └── app_mapping.yaml        # 应用映射
├── app/
│   ├── run_handler.py      # 运行处理器
│   ├── runtime_context.py  # 运行时上下文
│   └── verify.py           # 验证工具
├── device/
│   ├── adb_controller.py   # ADB 控制器
│   └── device_manager.py   # 设备管理器
├── policy/
│   ├── base_policy.py      # 策略基类
│   ├── ui_venus_policy.py  # UI-Venus 策略
│   └── ui_venus_2_policy.py # UI-Venus-2 策略
├── processor/
│   ├── base_processor.py   # 处理器基类
│   ├── uivenus_processor.py # UI-Venus 处理器
│   ├── ui_venus_2_processor.py # UI-Venus-2 处理器
│   └── rpa_utils/
│       └── postproc_utils.py # 动作描述转换
├── verify/
│   └── reflection_supervisor.py # 动作执行前反思监督
├── utils/
│   └── pickle_utils.py     # 序列化工具
└── data/
    └── purpose.txt         # 批量任务列表
```

---

## 常见问题

### 设备未找到

```bash
adb kill-server && adb start-server
adb devices
```

检查：USB 调试是否开启、数据线是否支持数据传输、是否点击了"允许 USB 调试"

### 文本输入问题

```bash
adb shell ime set com.android.adbkeyboard/.AdbIME
```

### 任务陷入循环

系统内置重复动作检测，连续 5 次相同动作会自动终止（swipe 操作除外）。

---

## 命令速查

```bash
# ADB 连接
adb devices
adb connect 192.168.1.100:5555

# 单任务执行
python main.py --purpose "你的任务" --trace-dir "record/traces/"

# 批量任务执行
python batch_runner.py

# 启用 reflection 的单任务
python main.py --purpose "你的任务" --trace-dir "record/traces/" --reflection

# 启用 reflection 的批量任务
python batch_runner.py --reflection

# 查看日志
tail -f logs/batch_runner.log
```

---

## License

本项目仅供研究和学习使用。
