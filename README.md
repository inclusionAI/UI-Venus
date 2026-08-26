<h1 align="center">
  <img src="assets/ui-venus-logo-3.png" width="60" align="center"> UI-Venus 2
</h1>


<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Report-Coming%20Soon-lightgrey" alt="Report: Coming Soon">
  <a href="https://ui-venus.github.io/UI-Venus-2"><img src="https://img.shields.io/badge/🌐%20Website-UI--Venus%202-blue" alt="Website: UI-Venus 2"></a>
  <a href="https://github.com/inclusionAI/UI-Venus"><img src="https://img.shields.io/badge/GitHub-Repository-green?logo=github" alt="GitHub"></a>
  <a href="https://huggingface.co/inclusionAI/UI-Venus-2-9b"><img src="https://img.shields.io/badge/Hugging%20Face-Model-orange?logo=huggingface" alt="Hugging Face Model"></a>
</p>

<p align="center">
  <em>A <strong>general-purpose foundation GUI agent</strong> for mobile apps, web platforms, and desktop operating systems — scaling <strong>environments, tasks, and feedback</strong> jointly within one closed-loop perception–reasoning–action agent.</em>
</p>

## 🌟 What's New in UI-Venus 2

Compared with UI-Venus 1.5, we introduce:

- 📱 **Scaled multilingual mobile environments:** A substantially expanded executable mobile pool covering Chinese and English app ecosystems, paired with a **deep-research-driven query-generation** strategy that grounds task queries in real application functionality — improving the accuracy, validity, and executability of generated instructions.
- 🖥️ **Computer use, built from the ground up:** Dedicated desktop operating-system capabilities constructed from scratch through computer-use data collection and task-specific training, extending the UI-Venus family to **mobile, web, and OS interaction in one unified end-to-end agent**.
- 🎯 **Keypoint-grounded verification:** Task completion is judged on **task-relevant visual keypoints** rather than a coarse holistic look at the final screen, with **multi-model voting** aggregating heterogeneous judges — reducing single-judge bias and making the reward signal robust to reward hacking.
- 🔄 **Verification-augmented reflection:** Verified feedback is distilled back into training as **reflection supervision**, so the agent can distinguish partial progress from true completion, avoid premature termination caused by observation misinterpretation, and recover during long-horizon interaction.

---

# 📰 News

* [2026/08] We release **UI-Venus 2**, a 9B/27B general-purpose foundation GUI agent that unifies mobile, web, and desktop interaction with scaled multilingual environments, keypoint-grounded verification, and verification-augmented reflection.
* [2026/02] We release **[UI-Venus 1.5](https://ui-venus.github.io/UI-Venus-1.5/)**, an end-to-end GUI Agent designed for robust real-world applications.
* [2026/02] We release **VenusBench-Mobile**, a challenging online benchmark for mobile GUI agents. See branch [VenusBench-Mobile](https://github.com/inclusionAI/UI-Venus/tree/VenusBench-Mobile).
* [2025/12] We release [VenusBench-GD](https://ui-venus.github.io/VenusBench-GD/), a comprehensive multi-platform GUI grounding benchmark. See branch [VenusBench-GD](https://github.com/inclusionAI/UI-Venus/tree/VenusBench-GD).
* [2025/8] We release **[UI-Venus 1.0](https://github.com/inclusionAI/UI-Venus/tree/UI-Venus-1.0)**, the first version of our UI agent model.

---

# 🧭 Overview

* [Demo](#-demo)
* [Venus Framework](#-venus-framework)
* [Quick Start](#-quick-start)
* [Benchmark Results](#-benchmark-results)
* [Contact](#-contact)
* [Citation](#-citation)

---

# ✨ Demo

[See more demos](https://ui-venus.github.io/UI-Venus-2/#demos)

<img src="https://ui-venus.github.io/UI-Venus-2/assets/demo_gifs/mobile_cn_weather_train_booking.gif" alt="UI-Venus-2 mobile demo" width="1200">

---

# 🛠️ Venus Framework

We provide two frameworks for running agents in real environments:

| Framework | Description | Documentation |
|---|---|---|
| Mobile Framework | Android/ADB agent framework for single-task execution, multi-device batch execution, trajectory recording, and reflection. | [English](./Venus_framework/Venus_framework_mobile/README.md) · [中文](./Venus_framework/Venus_framework_mobile/README_CN.md) |
| Browser Plugin | Chrome Side Panel extension that connects UI-Venus to the active browser tab and executes browser tasks interactively. | [English](./Venus_framework/Venus_plugin_browser/README.md) · [中文](./Venus_framework/Venus_plugin_browser/README_CN.md) |

See the [Venus Framework overview](./Venus_framework/README.md) for the directory layout and entry points. The lightweight domain examples below can be used without either framework unless their individual requirements state otherwise.

---

# 🚀 Quick Start

### Installation

```bash
conda create -n ui-venus-2 python=3.11 -y
conda activate ui-venus-2
pip install -r requirements.txt
```

Python 3.10 or newer is required. All commands below are executed from the repository root. Configure the OpenAI-compatible model service through `MODEL_URL`, `MODEL_NAME`, and either `API_KEY` or `MODEL_API_KEY`; the same values can also be edited at the top of each domain script. `API_KEY` takes precedence when both key variables are set. Set `MODEL_NAME` to the served name of your UI-Venus 2 9B or 27B model.

### Mobile

Run multi-turn inference over the included prerecorded screenshot sequence. `N_IMG` controls how many recent historical screenshots are retained:

```bash
MODEL_URL=http://127.0.0.1:8000/v1 \
MODEL_NAME=UI-Venus-2 \
N_IMG=2 \
bash scripts/mobile.sh
```

This example performs model inference only and does not execute actions on a device. Use the Mobile Framework above for real-device ADB automation.

[Mobile multi-turn example and input/output format](./models/mobile/README.md)

### Computer

Run multi-turn Computer inference over a prerecorded desktop screenshot sequence:

```bash
MODEL_URL=http://127.0.0.1:8000/v1 \
MODEL_NAME=UI-Venus-2 \
N_IMG=2 \
bash scripts/computer.sh
```

The default command uses the included desktop screenshot sample. The standalone example validates and normalizes model actions but does not execute them on the host. It has no runtime dependency on OSWorld.

[Computer multi-turn example and action format](./models/computer/README.md)

### Browser

Start Chrome with a CDP port as described in the domain documentation, then run one natural-language browser task:

```bash
MODEL_URL=http://127.0.0.1:8000/v1 \
MODEL_NAME=UI-Venus-2 \
bash scripts/browser.sh "Open https://example.com and report the page title"
```

[Browser usage and CDP setup](./models/browser/README.md)

### Grounding

Run the direct grounding evaluation on the three samples included with the repository:

```bash
MODEL_URL=http://127.0.0.1:8000/v1 \
MODEL_NAME=UI-Venus-2 \
bash scripts/grounding.sh
```

[Grounding evaluation, smoke test, and benchmark configuration](./models/grounding/README.md)

### CAPTCHA

Run inference on the included CAPTCHA image and save the parsed JSON and visualization under `results/captcha/`:

```bash
MODEL_URL=http://127.0.0.1:8000/v1 \
MODEL_NAME=UI-Venus-2 \
bash scripts/captcha.sh
```

[CAPTCHA usage, prompt, action format, and visualization](./models/captcha/README.md)

---

# 📊 Benchmark Results

### Mobile

| Models | MobileGym | VenusBench&#8209;Mobile | AndroidWorld | MobileWorld | KnowUBench | MemGUI |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| *General&nbsp;VLMs* |  |  |  |  |  |  |
| Qwen3.5&#8209;9B | 9.0 | 15.3 | 57.8 | 17.9 | 33.3 | 6.2 |
| Qwen3.6&#8209;27B | 24.6 | 28.0 | 70.3 | 41.9 | — | 25.7 |
| Opus&nbsp;4.6 | — | 36.5 | — | 44.5 | — | — |
| Kimi&nbsp;K2.6 | 38.7 | 31.2 | — | 55.6 | — | 39.1 |
| Seed2.0&nbsp;Pro | 52.0 | 20.1 | — | 63.2 | 51.6 | <u>65.6</u> |
| *GUI&#8209;specific&nbsp;Models* |  |  |  |  |  |  |
| UI&#8209;Venus&#8209;1.5&#8209;8B | 18.4 | 16.1 | 73.7 | 22.2 | 26.0 | 3.9 |
| UI&#8209;Venus&#8209;1.5&#8209;30B&#8209;A3B | 21.5 | 21.5 | 77.6 | 17.1 | — | 10.9 |
| GUI&#8209;Owl&#8209;1.5&#8209;32B&#8209;Instruct | 20.3 | — | 69.8 | 43.9 | — | 10.9 |
| MAI&#8209;UI&#8209;8B | 21.5 | 12.7 | 70.7 | 27.5† | 26.0 | 12.0 |
| Qwen&#8209;UI&#8209;Agent&#8209;27B | — | — | — | **82.1** | — | — |
| *Ours* |  |  |  |  |  |  |
| **UI&#8209;Venus&#8209;2&#8209;9B** | <u>52.7</u> | <u>46.5</u> | <u>80.2</u> | 65.8 (75.2) | <u>56.5</u> | 62.6 |
| **UI&#8209;Venus&#8209;2&#8209;27B** | **60.5** | **48.7** | **84.0** | <u>76.1 (82.9)</u> | **59.7** | **70.3** |

> MobileWorld reports success rate on the GUI-only subset. MemGUI reports Main Results Pass@1.

### Computer

| Models | OSWorld&#8209;Verified | OSWorld&#8209;V2 | DeskCraft | OpenComputer |
|---|:---:|:---:|:---:|:---:|
| *General&nbsp;VLMs* |  |  |  |  |
| Qwen3.5&#8209;9B | 41.8 | 2.53 | 14.6 | — |
| Qwen3.6&#8209;27B | 62.0 | — | 28.7 | — |
| Kimi&nbsp;K2.6 | 73.1 | 7.1 | 41.4 | — |
| Seed2.0&nbsp;Pro | — | 6.3 | 40.0 | — |
| *GUI&#8209;specific&nbsp;Models* |  |  |  |  |
| GUI&#8209;Owl&#8209;1.5&#8209;32B&#8209;Instruct | 56.5 | — | — | — |
| Qwen&#8209;UI&#8209;Agent&#8209;27B | 79.5 | — | — | — |
| *Ours* |  |  |  |  |
| **UI&#8209;Venus&#8209;2&#8209;9B** | 70.8 | — | 48.0 | — |
| **UI&#8209;Venus&#8209;2&#8209;27B** | 80.5 | — | 55.5 | — |

### Browser

| Models | WebVoyager | Online&#8209;Mind2Web | REAL | Odysseys&nbsp;Perfect |
|---|:---:|:---:|:---:|:---:|
| *General&nbsp;VLMs* |  |  |  |  |
| Qwen3.5&#8209;9B | 46.9 | 27.3 | 18.2 | 13.5 |
| Qwen3.6&#8209;27B | 84.3 | 55.3 | 27.3 | 18.5 |
| OpenAI&nbsp;Operator | 87.0 | 61.3 | — | — |
| Seed&#8209;2.0 | 85.1 | 68.5 | 74.4 | 30.1 |
| GLM&#8209;5V&#8209;Turbo | 88.5 | — | — | — |
| Claude&nbsp;Opus&nbsp;4.6 | 88.0 | — | — | 44.5 |
| Kimi&#8209;K2.6 | 76.8 | — | 74.4 | — |
| *GUI&#8209;specific&nbsp;Models* |  |  |  |  |
| UI&#8209;TARS&#8209;1.5 | 84.8 | <u>75.8</u> | — | — |
| UI&#8209;Venus&#8209;1.5&#8209;30B&#8209;A3B | 76.0 | — | 38.0 | — |
| GUI&#8209;Owl&#8209;1.5&#8209;32B&#8209;Thinking | 82.1 | — | 44.6 | — |
| MolmoWeb&#8209;8B | 78.2 | 35.3 | — | — |
| Fara&#8209;1.5&#8209;9B | 86.6 | 63.4 | — | — |
| Fara&#8209;1.5&#8209;27B | 89.3 | 72.3 | — | — |
| *Ours* |  |  |  |  |
| **UI&#8209;Venus&#8209;2&#8209;9B** | <u>90.8</u> | 74.0 | <u>76.9</u> | <u>62.0</u> |
| **UI&#8209;Venus&#8209;2&#8209;27B** | **93.4** | **78.3** | **80.2** | **66.3** |

> For Odysseys, we report the perfect rubric score (Perfect).

### Grounding

| Models | VenusBench&#8209;GD | ScreenSpot&#8209;Pro | OSWorld&#8209;G&#8209;R | UI&#8209;Vision |
|---|:---:|:---:|:---:|:---:|
| *General&nbsp;VLMs* |  |  |  |  |
| Qwen&nbsp;3.7&nbsp;Plus | — | 68.9 | 78.2 | 68.0 |
| Seed&nbsp;2.1&nbsp;Pro | — | 65.3 | 78.0 | 62.0 |
| Claude&nbsp;Opus&nbsp;4.6 | — | — | — | — |
| Kimi&nbsp;K2.6 | — | — | — | — |
| *GUI&#8209;specific&nbsp;Models* |  |  |  |  |
| UI&#8209;Venus&#8209;1.0&#8209;72B | 70.2 | 61.9 | 69.5 | 36.8 |
| Holo2&#8209;30B&#8209;A3B | 59.5* | 66.1 | 76.1 | 40.9* |
| Step&#8209;GUI&#8209;4B | 54.6* | 60.0 | 66.9 | 30.0* |
| MAI&#8209;UI&#8209;32B | — | 67.9 | <u>73.9</u> | <u>47.1</u> |
| UI&#8209;Venus&#8209;1.5&#8209;30B&#8209;A3B | 75.0 | 69.6 | 76.4 | 54.7 |
| Qwen&#8209;UI&#8209;Agent&#8209;27B | — | 76.6 | 78.5 | 70.0 |
| *Ours* |  |  |  |  |
| **UI&#8209;Venus&#8209;2&#8209;9B** | 77.1 | 73.0 | 78.5 | 53.2 |
| **UI&#8209;Venus&#8209;2&#8209;27B** | 80.1 | 74.1 | 79.1 | 66.9 |

> For each benchmark, bold and underlined scores indicate the best and second-best results, respectively. `*` indicates results that may require verification with the original sources.

### CAPTCHA

| Models | VenusBench&#8209;CAPTCHA | Spatial&#8209;CAPTCHA&#8209;Bench | MCA&#8209;Bench | NextGen&#8209;CAPTCHAs | OpenCaptcha |
|---|:---:|:---:|:---:|:---:|:---:|
| *General&nbsp;VLMs* |  |  |  |  |  |
| Qwen3.5&#8209;9B | 29.7 | 4.9 | 30.4 | 2.8 | 36.4 |
| Qwen3.6&#8209;27B | 54.8 | 31.0 | 51.7 | 14.1 | 47.7 |
| Doubao&#8209;Seed&#8209;2.0&#8209;Pro | 48.4 | <u>43.6</u> | 35.5 | 20.4 | <u>55.6</u> |
| Kimi&#8209;K2.6 | 40.6 | 24.8 | 38.7 | 7.2 | 47.8 |
| Claude&#8209;Opus&#8209;4.6 | 15.4 | 9.5 | 25.9 | 2.8 | 23.3 |
| *Ours* |  |  |  |  |  |
| **UI&#8209;Venus&#8209;2.0&#8209;9B** | <u>78.1</u> | 42.8 | <u>75.7</u> | <u>47.6</u> | 50.7 |
| **UI&#8209;Venus&#8209;2.0&#8209;27B** | **79.9** | **48.6** | **79.6** | **54.5** | **56.3** |

> All results are Pass@1 percentages, and higher is better. We evaluate on VenusBench-CAPTCHA and four public benchmarks: Spatial-CAPTCHA-Bench, MCA-Bench, NextGen-CAPTCHAs, and Open CaptchaWorld. We use 1,000 sampled MCA-Bench examples, 15 NextGen-CAPTCHAs task types, and 16 OpenCaptcha task types; see the appendix for selection details. Bold and underlined scores indicate the best and second-best results in each column, respectively.

---

# 📬 Contact

For any questions or collaboration, please contact the maintainers.

---

# 📚 Citation

```bibtex
# UI-Venus 1.5
@misc{venusteam2026uivenus15technicalreport,
      title={UI-Venus-1.5 Technical Report}, 
      author={Venus-Team and Changlong Gao and Zhangxuan Gu and Yulin Liu and Xinyu Qiu and Shuheng Shen and Yue Wen and Tianyu Xia and Zhenyu Xu and Zhengwen Zeng and Beitong Zhou and Xingran Zhou and Weizhi Chen and Sunhao Dai and Jingya Dou and Yichen Gong and Yuan Guo and Zhenlin Guo and Feng Li and Qian Li and Jinzhen Lin and Yuqi Zhou and Linchao Zhu and Liang Chen and Zhenyu Guo and Changhua Meng and Weiqiang Wang},
      year={2026},
      eprint={2602.09082},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.09082}, 
}

# UI-Venus 1.0
@misc{gu2025uivenustechnicalreportbuilding,
      title={UI-Venus Technical Report: Building High-performance UI Agents with RFT}, 
      author={Zhangxuan Gu and Zhengwen Zeng and Zhenyu Xu and others},
      year={2025},
      eprint={2508.10833},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.10833}, 
}
```

---

# ⚖️ License

This project is for research and educational purposes only.
