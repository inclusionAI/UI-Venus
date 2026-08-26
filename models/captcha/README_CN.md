# 单图 CAPTCHA vLLM 推理与可视化

[English](README.md)

本目录采用单图工作流，不做 batch 评测：

- infer_captcha.py：只负责一张图片的 vLLM 推理和动作解析。
- captcha_prompt.py：保存默认的 `SYS_PROMPT` 和 `USER_PROMPT`。
- visualize_captcha.py：读取单图推理 JSON，单独生成可视化 HTML。

统一入口从仓库根目录执行：

~~~bash
bash scripts/captcha.sh
~~~

模型地址、模型名、API key、输入图片、输出路径和 thinking 开关集中在 `scripts/captcha.sh` 开头，也可以通过同名环境变量覆盖。默认使用 `examples/assets/` 中的测试图片，并将结果写入 `results/captcha/`。

推理输出的动作 DSL：

~~~text
<think>简短分析</think>
<action>Click(box=(310,456)),Click(box=(520,640))</action>
~~~

支持 Click、LongPress、Type 和 Drag：

~~~text
Click(box=(x,y))
Click(box=(x,y))Type(content='text')
LongPress(box=(x,y))
Drag(start=(x1,y1),end=(x2,y2))
~~~

## 环境

客户端通过 vLLM OpenAI-compatible API 发起推理，使用 Pillow 读取
图片尺寸，不在当前进程加载 vLLM 模型，也不依赖 OpenAI SDK。

相关官方接口：

- [vLLM OpenAI 兼容服务](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
- [vLLM 多模态 base64 图片输入](https://docs.vllm.ai/en/latest/examples/generate/multimodal/)

## 调用 vLLM API 推理一张图片

先启动服务，例如：

~~~bash
vllm serve path/to/vision-model \
  --served-model-name captcha-model \
  --tensor-parallel-size 2
~~~

直接测试 assets 中的一张图片：

~~~bash
python models/captcha/infer_captcha.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model captcha-model \
  --image models/captcha/examples/assets/jiusuoge_5238.png \
  --enable-thinking \
  --output results/captcha/result.json
~~~

如果服务需要鉴权，使用 --api-key，或者设置 OPENAI_API_KEY。

每次运行只发送一张图片、产生一个 JSON 对象，不会扫描目录或构造 batch。
脚本只支持 vLLM API 调用；统一入口默认输出到 `results/captcha/result.json`。

## 坐标和 Prompt

默认提示输入为：

- `SYS_PROMPT`：数据集中的完整 GUI Agent + CAPTCHA-Specific Extension 原文。
- `USER_PROMPT`：&lt;image&gt;。

两个常量均定义在 `captcha_prompt.py`，`infer_captcha.py` 只负责导入和使用。

发送 OpenAI 多模态消息时，&lt;image&gt; 占位符转换为唯一一个 base64 image_url，
不会再附加之前的中文 user 文本，也不会重复发送字面量 &lt;image&gt;。

脚本默认显式开启 thinking。API 请求会携带：

~~~json
{"chat_template_kwargs": {"enable_thinking": true}}
~~~

如需关闭进行对照，传入 --no-enable-thinking。

该数据集 system 固定声明 0 到 999 归一化坐标，因此默认 coord-scale 为 999。若使用
其他坐标体系，必须同时通过 --system-prompt 提供匹配的新说明。

可以调整单图任务和系统提示词：

~~~bash
python models/captcha/infer_captcha.py ... \
  --task '识别图中的验证码要求并给出完整操作' \
  --system-prompt '自定义系统提示词' \
  --coord-scale 999
~~~

## 单图结果

`models/captcha/examples/test_result.json` 是随仓库保留的单个 JSON 输出样例，图片路径相对于该 JSON 文件：

~~~json
{
  "image": "assets/jiusuoge_5238.png",
  "image_size": [480, 847],
  "coord_scale": 999,
  "enable_thinking": true,
  "task": "<image>",
  "model_output": "<action>Click(box=(237,413))</action>",
  "reasoning_content": null,
  "inference": {
    "backend": "vllm-api",
    "model": "captcha-model"
  },
  "parsed_actions": [
    {"type": "Click", "x": 237, "y": 413}
  ]
}
~~~

infer_captcha.py 不包含评分或可视化逻辑。

## 独立可视化

推理完成后，用另一个文件生成 HTML：

~~~bash
python models/captcha/visualize_captcha.py \
  --result results/captcha/result.json \
  --output results/captcha/result.html
~~~

HTML 自包含原图，按 coord-scale 将动作坐标映射回图片：

- Click：橙色编号点。
- LongPress：蓝色编号点。
- Drag：紫色箭头。
- Click/LongPress 后的 Type：在对应点位显示文本标签。
- 没有前置点位的 Type：在动作面板单独显示。

原图位置发生变化时可以覆盖：

~~~bash
python models/captcha/visualize_captcha.py \
  --result results/captcha/result.json \
  --image models/captcha/examples/assets/jiusuoge_5238.png \
  --output results/captcha/result.html
~~~

## 使用其他测试单图

每次仍然只选择一张：

~~~bash
python models/captcha/infer_captcha.py ... \
  --image models/captcha/examples/assets/slide_4469747273.png

python models/captcha/infer_captcha.py ... \
  --image models/captcha/examples/assets/captchaOper_902440b13d984d2ba0f1b13581b31027_no_zhihu.png

python models/captcha/infer_captcha.py ... \
  --image models/captcha/examples/assets/captchaOper_7f438c2492ae4de28c7cf25beab63ea9.png
~~~
