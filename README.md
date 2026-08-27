<h1 align="center">
  <img src="assets/ui-venus-logo-3.png" height="32"> UI-Venus-2
</h1>

<p align="center">
  <strong>English</strong> | <a href="./README_CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Report-Coming%20Soon-lightgrey" alt="Report: Coming Soon">
  <a href="https://ui-venus.github.io/UI-Venus-2"><img src="https://img.shields.io/badge/🌐%20Website-UI--Venus--2-blue" alt="Website: UI-Venus-2"></a>
  <a href="https://github.com/inclusionAI/UI-Venus"><img src="https://img.shields.io/badge/GitHub-Repository-green?logo=github" alt="GitHub"></a>
  <a href="https://huggingface.co/inclusionAI/UI-Venus-2-9b"><img src="https://img.shields.io/badge/Hugging%20Face-Model-orange?logo=huggingface" alt="Hugging Face Model"></a>
</p>

<p align="center">
  <em>A <strong>general-purpose foundation GUI agent</strong> for mobile apps, web platforms, and desktop operating systems — scaling <strong>environments, tasks, and feedback</strong> jointly within one closed-loop perception–reasoning–action agent.</em>
</p>

## 🌟 What's New in UI-Venus-2

Compared with UI-Venus-1.5, we introduce:

- 📱 **Scaled multilingual mobile environments:** A substantially expanded executable mobile pool covering Chinese and English app ecosystems, paired with a **deep-research-driven query-generation** strategy that grounds task queries in real application functionality — improving the accuracy, validity, and executability of generated instructions.
- 🖥️ **Computer use, built from the ground up:** Dedicated desktop operating-system capabilities constructed from scratch through computer-use data collection and task-specific training, extending the UI-Venus family to **mobile, web, and OS interaction in one unified end-to-end agent**.
- 🎯 **Keypoint-grounded verification:** Task completion is judged on **task-relevant visual keypoints** rather than a coarse holistic look at the final screen, with **multi-model voting** aggregating heterogeneous judges — reducing single-judge bias and making the reward signal robust to reward hacking.
- 🔄 **Verification-augmented reflection:** Verified feedback is distilled back into training as **reflection supervision**, so the agent can distinguish partial progress from true completion, avoid premature termination caused by observation misinterpretation, and recover during long-horizon interaction.

---

<p align="center">
  📈 <strong>UI-Venus-2 Benchmark Performance</strong>
</p>

<p align="center">
  <img src="assets/venus2_page1_v7_minimal_editorial.png" alt="UI-Venus-2 Benchmark Performance" width="1200" />
</p>

> **Figure:** Performance comparison of UI-Venus-2 across representative mobile, browser, computer, CAPTCHA, and grounding benchmarks.

---

# 📰 News

* [2026/08] We release **UI-Venus-2**, a 9B/27B general-purpose foundation GUI agent that unifies mobile, web, and desktop interaction with scaled multilingual environments, keypoint-grounded verification, and verification-augmented reflection.
* [2026/02] We release **[UI-Venus-1.5](https://ui-venus.github.io/UI-Venus-1.5/)**, an end-to-end GUI Agent designed for robust real-world applications.
* [2026/02] We release **VenusBench-Mobile**, a challenging online benchmark for mobile GUI agents. See branch [VenusBench-Mobile](https://github.com/inclusionAI/UI-Venus/tree/VenusBench-Mobile).
* [2025/12] We release [VenusBench-GD](https://ui-venus.github.io/VenusBench-GD/), a comprehensive multi-platform GUI grounding benchmark. See branch [VenusBench-GD](https://github.com/inclusionAI/UI-Venus/tree/VenusBench-GD).
* [2025/8] We release **[UI-Venus](https://github.com/inclusionAI/UI-Venus/tree/UI-Venus-1.0)**, the first version of our UI agent model.

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

<img src="assets/demo_gifs/mobile_cn_weather_train_booking.gif" alt="UI-Venus-2 mobile demo" width="1200">

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

Python 3.10 or newer is required. All commands below are executed from the repository root. Configure the OpenAI-compatible model service through `MODEL_URL`, `MODEL_NAME`, and either `API_KEY` or `MODEL_API_KEY`; the same values can also be edited at the top of each domain script. `API_KEY` takes precedence when both key variables are set. Set `MODEL_NAME` to the served name of your UI-Venus-2 9B or 27B model.

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

<div align="center">

<table>
  <thead>
    <tr>
      <th align="left"><sub>Models</sub></th>
      <th align="center"><sub>MobileGym</sub></th>
      <th align="center"><sub>VenusBench&#8209;Mobile</sub></th>
      <th align="center"><sub>AndroidWorld</sub></th>
      <th align="center"><sub>MobileWorld</sub></th>
      <th align="center"><sub>KnowUBench</sub></th>
      <th align="center"><sub>MemGUI</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="7" align="left"><sub><em>General&nbsp;VLMs</em></sub></td>
    </tr>
    <tr>
      <td align="left"><sub>Qwen3.5&#8209;9B</sub></td>
      <td align="center">9.0*</td>
      <td align="center">15.3*</td>
      <td align="center">57.8</td>
      <td align="center">18.0 (18.0)*</td>
      <td align="center">33.3</td>
      <td align="center">6.2*</td>
    </tr>
    <tr>
      <td align="left"><sub>Qwen3.6&#8209;27B</sub></td>
      <td align="center">24.6*</td>
      <td align="center">28.0*</td>
      <td align="center">70.3</td>
      <td align="center">36.8 (41.9)*</td>
      <td align="center">-</td>
      <td align="center">25.7*</td>
    </tr>
    <tr>
      <td align="left"><sub>Opus&nbsp;4.6</sub></td>
      <td align="center">-</td>
      <td align="center">36.5*</td>
      <td align="center">-</td>
      <td align="center">44.5</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="left"><sub>Kimi&nbsp;K2.6</sub></td>
      <td align="center">38.7*</td>
      <td align="center">31.2*</td>
      <td align="center">-</td>
      <td align="center">55.6</td>
      <td align="center">-</td>
      <td align="center">39.1</td>
    </tr>
    <tr>
      <td align="left"><sub>Seed2.0&nbsp;Pro</sub></td>
      <td align="center">52.0</td>
      <td align="center">20.1*</td>
      <td align="center">-</td>
      <td align="center">63.2</td>
      <td align="center">51.6</td>
      <td align="center"><ins>65.6</ins>*</td>
    </tr>
    <tr>
      <td colspan="7" align="left"><sub><em>GUI&#8209;specific&nbsp;Models</em></sub></td>
    </tr>
    <tr>
      <td align="left"><sub>UI&#8209;Venus&#8209;1.5&#8209;8B</sub></td>
      <td align="center">18.4*</td>
      <td align="center">16.1</td>
      <td align="center">73.7</td>
      <td align="center">22.2*</td>
      <td align="center">26.0</td>
      <td align="center">3.9*</td>
    </tr>
    <tr>
      <td align="left"><sub>UI&#8209;Venus&#8209;1.5&#8209;30B&#8209;A3B</sub></td>
      <td align="center">21.5*</td>
      <td align="center">21.5</td>
      <td align="center">77.6</td>
      <td align="center">17.1</td>
      <td align="center">-</td>
      <td align="center">10.9*</td>
    </tr>
    <tr>
      <td align="left"><sub>GUI&#8209;Owl&#8209;1.5&#8209;32B&#8209;Instruct</sub></td>
      <td align="center">20.3*</td>
      <td align="center">-</td>
      <td align="center">69.8</td>
      <td align="center">43.9</td>
      <td align="center">-</td>
      <td align="center">10.9</td>
    </tr>
    <tr>
      <td align="left"><sub>MAI&#8209;UI&#8209;8B</sub></td>
      <td align="center">21.5*</td>
      <td align="center">12.7</td>
      <td align="center">70.7</td>
      <td align="center">27.5</td>
      <td align="center">26.0</td>
      <td align="center">17.2*</td>
    </tr>
    <tr>
      <td align="left"><sub>Qwen&#8209;UI&#8209;Agent&#8209;27B</sub></td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center"><strong>82.1 (85.5)</strong></td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td colspan="7" align="left"><sub><em>Ours</em></sub></td>
    </tr>
    <tr>
      <td align="left"><sub><strong>UI&#8209;Venus&#8209;2&#8209;9B</strong></sub></td>
      <td align="center"><ins>52.7</ins></td>
      <td align="center"><ins>46.5</ins></td>
      <td align="center"><ins>80.2</ins></td>
      <td align="center">65.8 (75.2)</td>
      <td align="center"><ins>56.5</ins></td>
      <td align="center">62.6</td>
    </tr>
    <tr>
      <td align="left"><sub><strong>UI&#8209;Venus&#8209;2&#8209;27B</strong></sub></td>
      <td align="center"><strong>60.5</strong></td>
      <td align="center"><strong>48.7</strong></td>
      <td align="center"><strong>84.0</strong></td>
      <td align="center"><ins>76.1 (82.9)</ins></td>
      <td align="center"><strong>59.7</strong></td>
      <td align="center"><strong>70.3</strong></td>
    </tr>
  </tbody>
</table>

</div>

> In the MobileWorld column, success rates on the GUI-only subset are reported for maximum step limits of 50 and 100 (with the 100-step results shown in parentheses). MemGUI reports Main Results Pass@1. `*` indicates our reproduced results.

### Computer

<div align="center">

<table>
  <thead>
    <tr>
      <th align="left"><sub>Models</sub></th>
      <th align="center"><sub>OSWorld&#8209;Verified</sub></th>
      <th align="center"><sub>OSWorld&#8209;V2</sub></th>
      <th align="center"><sub>DeskCraft</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="4" align="left"><sub><em>General&nbsp;VLMs</em></sub></td>
    </tr>
    <tr>
      <td align="left"><sub>Qwen3.5&#8209;9B</sub></td>
      <td align="center">41.8</td>
      <td align="center">2.5</td>
      <td align="center">14.6*</td>
    </tr>
    <tr>
      <td align="left"><sub>Qwen3.6&#8209;27B</sub></td>
      <td align="center">62.0</td>
      <td align="center">3.8</td>
      <td align="center">28.7*</td>
    </tr>
    <tr>
      <td align="left"><sub>Kimi&nbsp;K2.6</sub></td>
      <td align="center">73.1</td>
      <td align="center">7.1</td>
      <td align="center">41.4*</td>
    </tr>
    <tr>
      <td align="left"><sub>Seed2.0&nbsp;Pro</sub></td>
      <td align="center">62.3</td>
      <td align="center">6.3</td>
      <td align="center">40.0*</td>
    </tr>
    <tr>
      <td colspan="4" align="left"><sub><em>GUI&#8209;specific&nbsp;Models</em></sub></td>
    </tr>
    <tr>
      <td align="left"><sub>GUI&#8209;Owl&#8209;1.5&#8209;32B&#8209;Instruct</sub></td>
      <td align="center">56.5</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="left"><sub>Qwen&#8209;UI&#8209;Agent&#8209;27B</sub></td>
      <td align="center">79.5</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td colspan="4" align="left"><sub><em>Ours</em></sub></td>
    </tr>
    <tr>
      <td align="left"><sub><strong>UI&#8209;Venus&#8209;2&#8209;9B</strong></sub></td>
      <td align="center">70.8</td>
      <td align="center">7.52</td>
      <td align="center">48.0</td>
    </tr>
    <tr>
      <td align="left"><sub><strong>UI&#8209;Venus&#8209;2&#8209;27B</strong></sub></td>
      <td align="center">80.5</td>
      <td align="center">13.24</td>
      <td align="center">55.5</td>
    </tr>
  </tbody>
</table>

</div>

> Performance comparison on various computer use agent benchmarks. For DeskCraft, we report the overall performance on the full benchmark of 538 tasks.

### Browser

<div align="center">

<table>
  <thead>
    <tr>
      <th align="left"><sub>Models</sub></th>
      <th align="center"><sub>WebVoyager</sub></th>
      <th align="center"><sub>Online&#8209;Mind2Web</sub></th>
      <th align="center"><sub>REAL</sub></th>
      <th align="center"><sub>Odysseys&nbsp;Avg.</sub></th>
      <th align="center"><sub>Odysseys&nbsp;Perfect</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="6" align="left"><sub><em>General&nbsp;VLMs</em></sub></td>
    </tr>
    <tr>
      <td align="left"><sub>Qwen3.5&#8209;9B</sub></td>
      <td align="center">46.9*</td>
      <td align="center">27.3*</td>
      <td align="center">18.2*</td>
      <td align="center">42.6*</td>
      <td align="center">13.5*</td>
    </tr>
    <tr>
      <td align="left"><sub>Qwen3.6&#8209;27B</sub></td>
      <td align="center">84.3*</td>
      <td align="center">55.3*</td>
      <td align="center">27.3*</td>
      <td align="center">39.5*</td>
      <td align="center">18.5*</td>
    </tr>
    <tr>
      <td align="left"><sub>OpenAI&nbsp;Operator</sub></td>
      <td align="center">87.0</td>
      <td align="center">61.3</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="left"><sub>Seed2.0&nbsp;Pro</sub></td>
      <td align="center">85.1*</td>
      <td align="center">68.5*</td>
      <td align="center">74.4*</td>
      <td align="center">60.2*</td>
      <td align="center">30.1*</td>
    </tr>
    <tr>
      <td align="left"><sub>GLM&#8209;5V&#8209;Turbo</sub></td>
      <td align="center">88.5</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="left"><sub>Claude&nbsp;Opus&nbsp;4.6</sub></td>
      <td align="center">88.0</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">68.9</td>
      <td align="center">44.5</td>
    </tr>
    <tr>
      <td align="left"><sub>Kimi&#8209;K2.6</sub></td>
      <td align="center">76.8*</td>
      <td align="center">-</td>
      <td align="center">74.4*</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td colspan="6" align="left"><sub><em>GUI&#8209;specific&nbsp;Models</em></sub></td>
    </tr>
    <tr>
      <td align="left"><sub>UI&#8209;TARS&#8209;1.5</sub></td>
      <td align="center">84.8</td>
      <td align="center"><ins>75.8</ins></td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="left"><sub>UI&#8209;Venus&#8209;1.5&#8209;30B&#8209;A3B</sub></td>
      <td align="center">76.0</td>
      <td align="center">-</td>
      <td align="center">38.0*</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="left"><sub>GUI&#8209;Owl&#8209;1.5&#8209;32B&#8209;Thinking</sub></td>
      <td align="center">82.1</td>
      <td align="center">-</td>
      <td align="center">44.6*</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="left"><sub>MolmoWeb&#8209;8B</sub></td>
      <td align="center">78.2</td>
      <td align="center">35.3</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="left"><sub>Fara&#8209;1.5&#8209;9B</sub></td>
      <td align="center">86.6</td>
      <td align="center">63.4</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="left"><sub>Fara&#8209;1.5&#8209;27B</sub></td>
      <td align="center">89.3</td>
      <td align="center">72.3</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td colspan="6" align="left"><sub><em>Ours</em></sub></td>
    </tr>
    <tr>
      <td align="left"><sub><strong>UI&#8209;Venus&#8209;2&#8209;9B</strong></sub></td>
      <td align="center"><ins>90.8</ins></td>
      <td align="center">74.0</td>
      <td align="center"><ins>76.9</ins></td>
      <td align="center"><ins>77.3</ins></td>
      <td align="center"><ins>62.0</ins></td>
    </tr>
    <tr>
      <td align="left"><sub><strong>UI&#8209;Venus&#8209;2&#8209;27B</strong></sub></td>
      <td align="center"><strong>93.4</strong></td>
      <td align="center"><strong>78.3</strong></td>
      <td align="center"><strong>80.2</strong></td>
      <td align="center"><strong>80.4</strong></td>
      <td align="center"><strong>66.3</strong></td>
    </tr>
  </tbody>
</table>

</div>

> Performance comparison on four live-web benchmarks: WebVoyager, Online-Mind2Web, REAL, and Odysseys. For Odysseys, we report both the averaged rubric score (Avg.) and the perfect rubric score (Perfect). Bold and underlined scores indicate the best and second-best reported results in each column, respectively. `*` indicates our reproduced results, which are not available from official reports.

### Grounding

<div align="center">

<table>
  <thead>
    <tr>
      <th align="left"><sub>Models</sub></th>
      <th align="center"><sub>VenusBench&#8209;GD</sub></th>
      <th align="center"><sub>ScreenSpot&#8209;Pro</sub></th>
      <th align="center"><sub>OSWorld&#8209;G&#8209;R</sub></th>
      <th align="center"><sub>UI&#8209;Vision</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="5" align="left"><sub><em>General&nbsp;VLMs</em></sub></td>
    </tr>
    <tr>
      <td align="left"><sub>Qwen&nbsp;3.7&nbsp;Plus</sub></td>
      <td align="center">-</td>
      <td align="center">68.9</td>
      <td align="center">78.2</td>
      <td align="center"><ins>68.0</ins></td>
    </tr>
    <tr>
      <td align="left"><sub>Seed&nbsp;2.1&nbsp;Pro</sub></td>
      <td align="center">-</td>
      <td align="center">65.3</td>
      <td align="center">78.0</td>
      <td align="center">62.0</td>
    </tr>
    <tr>
      <td align="left"><sub>Kimi&nbsp;K2.6</sub></td>
      <td align="center">73.1*</td>
      <td align="center">52.0*</td>
      <td align="center">-*</td>
      <td align="center">-*</td>
    </tr>
    <tr>
      <td align="left"><sub>Qwen3.6&#8209;27B</sub></td>
      <td align="center">67.7*</td>
      <td align="center">65.2*</td>
      <td align="center">76.9*</td>
      <td align="center">58.3*</td>
    </tr>
    <tr>
      <td colspan="5" align="left"><sub><em>GUI&#8209;specific&nbsp;Models</em></sub></td>
    </tr>
    <tr>
      <td align="left"><sub>UI&#8209;Venus&#8209;Ground&#8209;72B</sub></td>
      <td align="center">70.2</td>
      <td align="center">61.9</td>
      <td align="center">69.5</td>
      <td align="center">36.8</td>
    </tr>
    <tr>
      <td align="left"><sub>Holo2&#8209;30B&#8209;A3B</sub></td>
      <td align="center">59.5*</td>
      <td align="center">66.1</td>
      <td align="center">76.1</td>
      <td align="center">40.9*</td>
    </tr>
    <tr>
      <td align="left"><sub>Step&#8209;GUI&#8209;4B</sub></td>
      <td align="center">54.6*</td>
      <td align="center">60.0</td>
      <td align="center">66.9</td>
      <td align="center">30.0*</td>
    </tr>
    <tr>
      <td align="left"><sub>MAI&#8209;UI&#8209;32B</sub></td>
      <td align="center">-</td>
      <td align="center">67.9</td>
      <td align="center">73.9</td>
      <td align="center">47.1</td>
    </tr>
    <tr>
      <td align="left"><sub>UI&#8209;Venus&#8209;1.5&#8209;30B&#8209;A3B</sub></td>
      <td align="center">75.0</td>
      <td align="center">69.6</td>
      <td align="center">76.4</td>
      <td align="center">54.7</td>
    </tr>
    <tr>
      <td align="left"><sub>Qwen&#8209;UI&#8209;Agent&#8209;27B</sub></td>
      <td align="center">-</td>
      <td align="center"><strong>76.6</strong></td>
      <td align="center">78.5</td>
      <td align="center"><strong>70.0</strong></td>
    </tr>
    <tr>
      <td colspan="5" align="left"><sub><em>Ours</em></sub></td>
    </tr>
    <tr>
      <td align="left"><sub><strong>UI&#8209;Venus&#8209;2&#8209;9B</strong></sub></td>
      <td align="center">77.1</td>
      <td align="center">73.0</td>
      <td align="center"><ins>78.5</ins></td>
      <td align="center">53.2</td>
    </tr>
    <tr>
      <td align="left"><sub><strong>UI&#8209;Venus&#8209;2&#8209;27B</strong></sub></td>
      <td align="center"><strong>80.1</strong></td>
      <td align="center"><ins>74.1</ins></td>
      <td align="center"><strong>79.1</strong></td>
      <td align="center">66.9</td>
    </tr>
  </tbody>
</table>

</div>

> For each benchmark, bold and underlined scores indicate the best and second-best results, respectively. `*` indicates the results we reproduced.

### CAPTCHA

<div align="center">

<table>
  <thead>
    <tr>
      <th align="left"><sub>Models</sub></th>
      <th align="center"><sub>VenusBench&#8209;CAPTCHA</sub></th>
      <th align="center"><sub>MCA&#8209;Bench</sub></th>
      <th align="center"><sub>Spatial&#8209;CAPTCHA&#8209;Bench</sub></th>
      <th align="center"><sub>NextGen&#8209;CAPTCHAs</sub></th>
      <th align="center"><sub>Open&nbsp;CaptchaWorld</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="6" align="left"><sub><em>General&nbsp;VLMs</em></sub></td>
    </tr>
    <tr>
      <td align="left"><sub>Qwen3.5&#8209;9B</sub></td>
      <td align="center">28.3</td>
      <td align="center">30.4</td>
      <td align="center">4.9</td>
      <td align="center">2.8</td>
      <td align="center">36.4</td>
    </tr>
    <tr>
      <td align="left"><sub>Qwen3.6&#8209;27B</sub></td>
      <td align="center">53.0</td>
      <td align="center">51.7</td>
      <td align="center">31.0</td>
      <td align="center">14.1</td>
      <td align="center">47.7</td>
    </tr>
    <tr>
      <td align="left"><sub>Doubao&#8209;Seed&#8209;2.0&#8209;Pro</sub></td>
      <td align="center">47.9</td>
      <td align="center">35.5</td>
      <td align="center"><ins>43.6</ins></td>
      <td align="center">20.4</td>
      <td align="center"><ins>55.6</ins></td>
    </tr>
    <tr>
      <td align="left"><sub>Kimi&#8209;K2.6</sub></td>
      <td align="center">39.7</td>
      <td align="center">38.7</td>
      <td align="center">24.8</td>
      <td align="center">7.2</td>
      <td align="center">47.8</td>
    </tr>
    <tr>
      <td align="left"><sub>Claude&#8209;Opus&#8209;4.6</sub></td>
      <td align="center">16.0</td>
      <td align="center">25.9</td>
      <td align="center">9.5</td>
      <td align="center">2.8</td>
      <td align="center">23.3</td>
    </tr>
    <tr>
      <td colspan="6" align="left"><sub><em>Ours</em></sub></td>
    </tr>
    <tr>
      <td align="left"><sub><strong>UI&#8209;Venus&#8209;2&#8209;9B</strong></sub></td>
      <td align="center"><ins>78.1</ins></td>
      <td align="center"><ins>75.7</ins></td>
      <td align="center">42.8</td>
      <td align="center"><ins>47.6</ins></td>
      <td align="center">50.7</td>
    </tr>
    <tr>
      <td align="left"><sub><strong>UI&#8209;Venus&#8209;2&#8209;27B</strong></sub></td>
      <td align="center"><strong>79.9</strong></td>
      <td align="center"><strong>79.6</strong></td>
      <td align="center"><strong>48.6</strong></td>
      <td align="center"><strong>54.5</strong></td>
      <td align="center"><strong>56.3</strong></td>
    </tr>
  </tbody>
</table>

</div>

> All results are Pass@1 percentages, and higher is better. We evaluate on VenusBench-CAPTCHA and four public benchmarks: MCA-Bench, Spatial-CAPTCHA-Bench, NextGen-CAPTCHAs, and Open CaptchaWorld. We use 1,000 sampled MCA-Bench examples, 15 NextGen-CAPTCHAs task types, and 16 Open CaptchaWorld task types; see the appendix for selection details. Bold and underlined scores indicate the best and second-best reported results in each column, respectively.

---

# 📬 Contact

For any questions or collaboration, please contact the maintainers.

---

# 📚 Citation

```bibtex
# UI-Venus-2
@misc{venusteam2026uivenus2technicalreport,
      title={UI-Venus-2 Technical Report}, 
      author={Venus-Team and xxx},
      year={2026},
      eprint={xxx},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={comming soon}, 
}

# UI-Venus-1.5
@misc{venusteam2026uivenus15technicalreport,
      title={UI-Venus-1.5 Technical Report}, 
      author={Venus-Team and Changlong Gao and Zhangxuan Gu and Yulin Liu and Xinyu Qiu and Shuheng Shen and Yue Wen and Tianyu Xia and Zhenyu Xu and Zhengwen Zeng and Beitong Zhou and Xingran Zhou and Weizhi Chen and Sunhao Dai and Jingya Dou and Yichen Gong and Yuan Guo and Zhenlin Guo and Feng Li and Qian Li and Jinzhen Lin and Yuqi Zhou and Linchao Zhu and Liang Chen and Zhenyu Guo and Changhua Meng and Weiqiang Wang},
      year={2026},
      eprint={2602.09082},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.09082}, 
}

# UI-Venus
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
