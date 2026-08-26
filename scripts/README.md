# Unified Domain Entry Points

[中文](README_CN.md)

Run every script from the repository root. All domains use the same organization:

```text
scripts/<domain>.sh           # User entry point and environment variables
models/<domain>/              # Inference/evaluation implementation and examples
results/<domain>/             # Default output
```

| Domain | Entry point | Implementation guide | Status |
|---|---|---|---|
| Mobile | `bash scripts/mobile.sh` | `models/mobile/README.md` | Available |
| Computer | `bash scripts/computer.sh` | `models/computer/README.md` | Available |
| Browser | `bash scripts/browser.sh` | `models/browser/README.md` | Available |
| Grounding | `bash scripts/grounding.sh` | `models/grounding/README.md` | Available |
| CAPTCHA | `bash scripts/captcha.sh` | `models/captcha/README.md` | Available |

Each script keeps common settings near the beginning and lets identically named environment variables override them. Available domains consistently use `MODEL_URL`, `MODEL_NAME`, `API_KEY`, and `OUTPUT_DIR`; `MODEL_API_KEY` is accepted as an alias for `API_KEY`, and other variables are domain-specific. Model services use OpenAI-compatible APIs. Browser additionally requires local Chrome CDP, while the Mobile Framework requires ADB. See each linked README for details.
