# Venus Framework

[中文](README_CN.md)

This directory contains two frameworks for execution in real environments:

```text
Venus_framework/
├── Venus_framework_mobile/   # Android/ADB single tasks, batch tasks, trajectories, and reflection
└── Venus_plugin_browser/     # Chrome Side Panel Browser Agent extension
```

## Mobile Framework

- [English](Venus_framework_mobile/README.md) · [中文](Venus_framework_mobile/README_CN.md)

Install dependencies and run from the repository root:

```bash
cd Venus_framework/Venus_framework_mobile
pip install -r requirement.txt
python main.py --purpose "Open Xiaohongshu and search for outfit posts" --trace-dir record/traces
```

## Browser Plugin

- [English](Venus_plugin_browser/README.md) · [中文](Venus_plugin_browser/README_CN.md)

On the Chrome extensions page, select **Load unpacked** and choose:

```text
Venus_framework/Venus_plugin_browser
```

The standalone Browser inference example, which does not require the extension, is under `models/browser/`. Its unified entry point is `bash scripts/browser.sh`.
