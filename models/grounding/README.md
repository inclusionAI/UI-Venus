# UI-Venus-2 Grounding 测试方案 / Grounding Test Plan

本目录提供 UI-Venus-2 的 GUI grounding 评测工具链：通过 OpenAI 兼容 API 端点对待测模型发起 direct 直推（单次调用输出坐标），按 bbox 命中判定正确性。**用户需自备 OpenAI 兼容 API 服务**（如 vLLM / SGLang 部署的模型服务），以下命令中的端点地址、API key、模型名均为占位示例。统一入口为仓库根目录下的 `scripts/grounding.sh`。
This directory provides the UI-Venus-2 GUI grounding evaluation toolchain: it sends direct (single-call) grounding requests to the model under test via an OpenAI-compatible API endpoint, and judges correctness by bounding-box hit. **You need your own OpenAI-compatible API service** (e.g. a model served with vLLM / SGLang); endpoint URLs, API keys, and model names in the commands below are placeholders.

---

## 目录结构 / Directory Layout

```
models/grounding/
├── ui_venus2_gd.py          # OpenAI-compatible grounding 模型适配器
├── eval_single_benchmark.py   # 单 benchmark 评测脚本（direct 直推、并发、断点续跑）
│                              # Single-benchmark evaluator (direct inference, concurrency, checkpoint/resume)
├── eval_multi_benchmark.py    # 多 benchmark 驱动脚本（为每个 benchmark 拉起上面的子进程）
│                              # Multi-benchmark driver (spawns one subprocess per benchmark)
├── test_grounding_api.py      # 测试用例方案：连通性 + 端到端 grounding + 参数合理性核验
│                              # Test plan: connectivity + end-to-end grounding + parameter sanity
└── test_cases/                # 从 ScreenSpot-Pro 随机采样的 3 条测试样本（seed=42）
    ├── images/                #   图片，相对路径与标注 img_filename 对齐
    │                          #   Images; relative paths match the img_filename field
    └── annotations/
        └── test_samples.json  #   标注（bbox / instruction / ui_type 等）
                               #   Annotations (bbox / instruction / ui_type, etc.)
```

## 模型调用方式 / How the Model Is Called

- 模型类 `Qwen35GroundModel`（`models/grounding/ui_venus2_gd.py`），user-only 消息，无 function-call 包装。
  `Qwen35GroundModel` (in `models/grounding/ui_venus2_gd.py`) uses user-only messages without function-call wrapping.
- 模型输出 `[x, y]` 归一化坐标，默认坐标空间为 `[0, 1000]`（`--norm_type 0-1000`）。
  The model outputs `[x, y]` coordinates in `[0, 1000]` space by default (`--norm_type 0-1000`).
- Grounding 固定使用非 thinking 的单次直推。
  Grounding uses fixed non-thinking, single-shot inference.
- 已移除全部 zoom-in / test-time-scaling 逻辑，评测路径为单次直推。
  All zoom-in / test-time-scaling logic has been removed; evaluation is single-shot direct inference.

## 快速开始 / Quick Start

在仓库根目录配置 `scripts/grounding.sh` 开头的模型地址、模型名和 API key，然后运行内置 3 条样本：

```bash
bash scripts/grounding.sh
```

也可以用环境变量覆盖；`MODE=smoke` 运行 API 与内置样例检查，`MODE=multi` 运行多 benchmark：

```bash
MODE=smoke MODEL_URL=http://127.0.0.1:8000/v1 MODEL_NAME=UI-Venus-2 bash scripts/grounding.sh
```

### 0. 冒烟测试（3 条采样样本）/ Smoke test (3 sampled cases)

```bash
export GROUNDING_BASE_URL="http://127.0.0.1:8000/v1"   # 你的端点 / your endpoint
export GROUNDING_API_KEY="your-api-key"                # 你的 key（无鉴权可填 empty）
export GROUNDING_MODEL="your-model-name"               # 你的模型名 / your model name
python3 models/grounding/test_grounding_api.py
```

默认冒烟测试运行三条真实端到端样例。设置 `GROUNDING_TEST_PARAMETERS=true` 可额外运行底层参数组合探测；额外探测会增加请求数量。
The default smoke test runs three real end-to-end cases. Set `GROUNDING_TEST_PARAMETERS=true` to additionally probe low-level parameter combinations; the optional probes add requests.

### 1. 单 benchmark 评测 / Single-benchmark evaluation

```bash
python3 models/grounding/eval_single_benchmark.py \
    --base_url http://127.0.0.1:8000/v1 \
    --api_key your-api-key \
    --model_name your-model-name \
    --task all \
    --language en \
    --gt_type positive \
    --inst_style instruction \
    --num_processes 2 \
    --num_workers 4 \
    --norm_type 0-1000 \
    --checkpoint_interval 20 \
    --log_path results/screenspot_pro.json \
    --checkpoint_path results_mid/screenspot_pro.json \
    --screenspot_imgs /path/to/Screenspot-pro/images \
    --screenspot_test /path/to/Screenspot-pro/annotations
```

### 2. 多 benchmark 评测 / Multi-benchmark evaluation

```bash
python3 models/grounding/eval_multi_benchmark.py \
    --base_url http://127.0.0.1:8000/v1 \
    --api_key your-api-key \
    --model_name your-model-name \
    --benchmarks all \
    --num_processes 4 \
    --num_workers 4 \
    --log_dir results/exp01 \
    --checkpoint_dir results_mid/exp01
```

支持的 benchmark / Supported benchmarks: `screenspot_pro`, `venusbench_gd`, `osworld_g_refine`, `ss_v2_resize`, `mmbench_gui`, `data_uivision`, `ui_vision_full`（数据路径在 `eval_multi_benchmark.py` 的 `BENCHMARK_CONFIGS` 中配置 / dataset paths are configured in `BENCHMARK_CONFIGS`）。

## 评测指标 / Metrics

结果 JSON 中包含 / The result JSON contains:

- `overall`：总体 action_acc / text_acc / icon_acc
- `fine_grained`：按 platform × application × instruction_style × gt_type 细分
- `seeclick_style`：按 platform × instruction_style × gt_type 细分
- `leaderboard_simple_style` / `leaderboard_detailed_style`：按 group / application 细分

判定规则 / Judgement rules:

- **positive 样本**：预测点（归一化）落在 GT bbox（归一化）内记为 correct；GT 与预测均为 `[-1,-1]`（拒答）也记为 correct。
  **Positive**: a normalized prediction inside the normalized GT bbox counts as correct; both GT and prediction being `[-1,-1]` (refusal) also counts as correct.
- **negative 样本**：模型返回 `[-1,-1]`（不可行）记为 correct。
  **Negative**: the model returning `[-1,-1]` (infeasible) counts as correct.

## 测试数据说明 / Test Data Notes

- `test_cases/` 的 3 条样本来自 ScreenSpot-Pro 全量 1641 条，`random.seed(42)` 可复现；图片按 `img_filename` 相对路径存放，与 `--screenspot_imgs` / `--screenspot_test` 参数约定一致。
  The 3 cases in `test_cases/` are drawn from the full ScreenSpot-Pro set (1641 samples) with `random.seed(42)`; images are stored at their `img_filename` relative paths, matching the `--screenspot_imgs` / `--screenspot_test` contract.
- 新增测试样本时，保持同样的目录结构即可被评测脚本直接加载。
  To add more cases, keep the same layout and the eval scripts will load them directly.

## 参数注意事项 / Parameter Notes

- `logprobs=True/False` 两种组合均被端点接受；PPL 统计走 `need_logprobs=True` 路径。
  Both `logprobs=True/False` combinations are accepted; PPL statistics use the `need_logprobs=True` path.
