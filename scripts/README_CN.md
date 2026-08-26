# Unified Domain Entry Points

[English](README.md)

所有脚本都从仓库根目录执行，使用相同的组织方式：

```text
scripts/<domain>.sh           # 用户入口与环境变量
models/<domain>/              # 推理、评测实现和示例数据
results/<domain>/             # 默认输出
```

| Domain | 入口 | 实现说明 | 状态 |
|---|---|---|---|
| Mobile | `bash scripts/mobile.sh` | `models/mobile/README_CN.md` | 可用 |
| Computer | `bash scripts/computer.sh` | `models/computer/README_CN.md` | 可用 |
| Browser | `bash scripts/browser.sh` | `models/browser/README.md` | 可用 |
| Grounding | `bash scripts/grounding.sh` | `models/grounding/README.md` | 可用 |
| CAPTCHA | `bash scripts/captcha.sh` | `models/captcha/README_CN.md` | 可用 |

每个脚本都把常用配置集中在文件开头，并允许通过同名环境变量覆盖。可用任务统一使用 `MODEL_URL`、`MODEL_NAME`、`API_KEY` 和 `OUTPUT_DIR`；`MODEL_API_KEY` 可作为 `API_KEY` 的别名，其余变量是各领域特有配置。模型服务统一使用 OpenAI-compatible API；Browser 还需要本地 Chrome CDP，Mobile Framework 还需要 ADB，具体见对应 README。
