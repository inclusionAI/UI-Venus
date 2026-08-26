# Venus Framework

[English](README.md)

本目录包含两个面向真实环境执行的框架：

```text
Venus_framework/
├── Venus_framework_mobile/   # Android/ADB 单任务、批任务、轨迹记录与 reflection
└── Venus_plugin_browser/     # Chrome Side Panel Browser Agent 插件
```

## Mobile Framework

- [English](Venus_framework_mobile/README.md) · [中文](Venus_framework_mobile/README_CN.md)

从仓库根目录进入框架后安装依赖和运行：

```bash
cd Venus_framework/Venus_framework_mobile
pip install -r requirement.txt
python main.py --purpose "打开小红书，搜索穿搭帖子" --trace-dir record/traces
```

## Browser Plugin

- [English](Venus_plugin_browser/README.md) · [中文](Venus_plugin_browser/README_CN.md)

在 Chrome 的扩展管理页面选择“加载已解压的扩展程序”，目录指向：

```text
Venus_framework/Venus_plugin_browser
```

不需要插件的 Browser 独立推理示例位于 `models/browser/`，统一入口是 `bash scripts/browser.sh`。
