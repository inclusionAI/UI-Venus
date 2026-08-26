import base64
import json
import logging
import math
import os
import re
import time
from io import BytesIO

from openai import OpenAI
from PIL import Image, ImageDraw


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个 GUI 操作先验验证专家，负责在操作执行前判断智能体操作的正确性。

## 验证原则
你的判断必须仅基于以下先验信息：
1. 当前截图（操作前的页面状态）
截图中可能包含系统叠加的操作位置标记，不是原始界面的一部分：
- 红色圆点：单击位置
- 橙色圆点：双击位置
- 紫色圆点：长按位置
- 绿色线段、绿色起点和蓝色终点：滑动轨迹
2. 操作类型和详情（坐标范围为 0 到 __SCALE_FACTOR__）
3. Agent 的思考过程
4. 任务目标
5. 前序步骤上下文

你不需要也不应该依赖操作执行后的结果来判断。

## 判定类型
- CORRECT：操作基于截图中的实际界面元素做出正确判断并推进任务。
- INCORRECT：操作明显错误或偏离任务目标。
- INEFFECTIVE：操作无实质贡献或陷入无效循环。
- EXPLORATORY：操作是在不确定情况下做出的合理探索。

不要被 Agent 的思考误导，必须基于截图中的实际内容和操作位置判断。推理过程中不要输出坐标数字，应通过视觉内容描述操作位置。

## 输出格式
{
  "verdict": "CORRECT|INCORRECT|INEFFECTIVE|EXPLORATORY",
  "reasoning": "推理过程",
  "think_action_match": true,
  "evidence": {
    "history_context": "前序操作对当前步骤的影响分析",
    "task_progress": "当前操作对任务的推进程度评估",
    "think_action_analysis": "思考与动作一致性分析"
  }
}"""

STEP_USER_TEMPLATE = """## 任务信息
- 目标: {goal}

## 当前步骤
- 步骤: {step}
- 操作类型: {action_type}
- 操作详情: {action_details}
- Agent思考: {think}

## 当前截图
[请查看附带的截图]

请判断该操作是否正确，输出 JSON 格式的验证结果。"""

REFLECTION_FEEDBACK_TEMPLATE = """监督模型对你的操作做出了如下判定：

判定结果：{verdict}
推理过程：{reasoning}
思考与动作一致性：{think_action_match}
证据分析：
- 历史上下文：{history_context}
- 任务进展评估：{task_progress}
- 思考动作分析：{think_action_analysis}

请基于以上反馈，重新分析当前截图并给出正确的操作。注意：
1. 仔细观察截图中的实际界面元素
2. 不要重复相同的错误操作
3. 如果之前的操作目标不可行，尝试替代方案"""


def build_feedback(judgment: dict) -> str:
    evidence = judgment.get("evidence", {})
    return REFLECTION_FEEDBACK_TEMPLATE.format(
        verdict=judgment.get("verdict", "UNKNOWN"),
        reasoning=judgment.get("reasoning", ""),
        think_action_match=judgment.get("think_action_match", "N/A"),
        history_context=evidence.get("history_context", ""),
        task_progress=evidence.get("task_progress", ""),
        think_action_analysis=evidence.get("think_action_analysis", ""),
    )


def _format_action_details(action: dict, scale_factor: float, image_size: tuple) -> str:
    action_type = action.get("action_type", "unknown")
    parts = [action_type]

    def scale(x, y):
        if not image_size or image_size[0] == 0 or image_size[1] == 0:
            return x, y
        scaled_x = x / image_size[0] * scale_factor
        scaled_y = y / image_size[1] * scale_factor
        if scale_factor == 1:
            return f"{scaled_x:.3f}", f"{scaled_y:.3f}"
        return str(int(scaled_x)), str(int(scaled_y))

    if action_type in ("CLK", "LongPress", "DoubleCLK") and action.get("action_pos"):
        x, y = scale(*action["action_pos"][0])
        parts.append(f"坐标 ({x}, {y})")
    elif action_type == "SWIPE" and len(action.get("action_pos", [])) >= 2:
        start_x, start_y = scale(*action["action_pos"][0])
        end_x, end_y = scale(*action["action_pos"][1])
        parts.append(f"从 ({start_x}, {start_y}) 到 ({end_x}, {end_y})")
    elif action_type == "INPUT":
        parts.append(f'输入 "{action.get("input", "")}"')
    elif action_type == "REOPEN":
        parts.append(f'打开 {action.get("app_name", "")}')
    elif action_type == "WAIT":
        parts.append(f'{action.get("duration", 1000)}ms')
    elif action_type in ("CallUser", "SUCCESS"):
        parts.append(f'内容: {action.get("input", "")}')
    return " ".join(parts)


def _annotate_screenshot(image_b64: str, action: dict) -> str:
    action_type = action.get("action_type", "")
    action_pos = action.get("action_pos", [])
    color_map = {"CLK": "red", "DoubleCLK": "orange", "LongPress": "purple"}
    if action_type in color_map and action_pos:
        x, y = (int(value) for value in action_pos[0])
        image = Image.open(BytesIO(base64.b64decode(image_b64)))
        draw = ImageDraw.Draw(image)
        radius = max(int(min(image.size) * 0.015), 6)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color_map[action_type],
            outline="white",
            width=2,
        )
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    if action_type == "SWIPE" and len(action_pos) >= 2:
        x1, y1 = (int(value) for value in action_pos[0])
        x2, y2 = (int(value) for value in action_pos[1])
        image = Image.open(BytesIO(base64.b64decode(image_b64)))
        draw = ImageDraw.Draw(image)
        radius = max(int(min(image.size) * 0.015), 6)
        draw.line([(x1, y1), (x2, y2)], fill="green", width=max(radius // 2, 3))
        draw.ellipse(
            [x1 - radius, y1 - radius, x1 + radius, y1 + radius],
            fill="green",
            outline="white",
            width=2,
        )
        draw.ellipse(
            [x2 - radius, y2 - radius, x2 + radius, y2 + radius],
            fill="blue",
            outline="white",
            width=2,
        )
        delta_x, delta_y = x2 - x1, y2 - y1
        length = math.hypot(delta_x, delta_y)
        if length:
            unit_x, unit_y = delta_x / length, delta_y / length
            base_x = x2 - unit_x * radius * 2.5
            base_y = y2 - unit_y * radius * 2.5
            perpendicular_x, perpendicular_y = -unit_y * radius, unit_x * radius
            draw.polygon(
                [
                    (x2, y2),
                    (base_x + perpendicular_x, base_y + perpendicular_y),
                    (base_x - perpendicular_x, base_y - perpendicular_y),
                ],
                fill="blue",
            )
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image_b64


def _parse_json_response(raw: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    text = re.sub(r".*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    logger.warning("无法解析 JSON 响应: %s", text)
    return {}


class ReflectionSupervisor:
    def __init__(
        self,
        model_url: str,
        model_name: str,
        api_key: str = "",
        temperature: float = 0.0,
        scale_factor: float = 1000,
        image_window: int = 3,
        max_retries: int = 3,
        **kwargs,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.scale_factor = scale_factor
        self.image_window = image_window
        self.api_key = api_key or os.environ.get("API_KEY") or os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.client = OpenAI(base_url=model_url, api_key=self.api_key)
        scale_label = "1（归一化小数）" if scale_factor == 1 else str(int(scale_factor))
        self.system_prompt = SYSTEM_PROMPT.replace("__SCALE_FACTOR__", scale_label)
        self._messages = [{"role": "system", "content": self.system_prompt}]
        self._user_image_indices = []
        self._is_retry = False

    def judge_action(
        self,
        goal: str,
        step: int,
        action: dict,
        think: str,
        screenshot_b64: str,
    ) -> dict:
        if self._is_retry and len(self._messages) >= 2:
            self._messages.pop()
            self._messages.pop()
            if self._user_image_indices and self._user_image_indices[-1] >= len(self._messages):
                self._user_image_indices.pop()
        user_message = self._build_step_message(goal, step, action, think, screenshot_b64)
        self._apply_image_window()
        self._user_image_indices.append(len(self._messages))
        self._messages.append(user_message)
        raw, reasoning_content = self._call_model(self._messages)
        parsed = _parse_json_response(raw)
        self._messages.append({"role": "assistant", "content": raw})
        verdict = parsed.get("verdict", "CORRECT") if parsed else "CORRECT"
        self._is_retry = verdict not in ("CORRECT", "EXPLORATORY")
        if not parsed:
            return {
                "verdict": "CORRECT",
                "reasoning": raw,
                "reasoning_content": reasoning_content,
            }
        parsed["reasoning_content"] = reasoning_content
        return parsed

    def notify_step_committed(self):
        self._is_retry = False

    def _build_step_message(
        self,
        goal: str,
        step: int,
        action: dict,
        think: str,
        screenshot_b64: str,
    ) -> dict:
        image_size = None
        if screenshot_b64:
            image_size = Image.open(BytesIO(base64.b64decode(screenshot_b64))).size
        user_text = STEP_USER_TEMPLATE.format(
            goal=goal,
            step=step,
            action_type=action.get("action_type", "unknown"),
            action_details=_format_action_details(action, self.scale_factor, image_size),
            think=think or "无",
        )
        content = [{"type": "text", "text": user_text}]
        if screenshot_b64:
            annotated = _annotate_screenshot(screenshot_b64, action)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{annotated}"},
            })
        return {"role": "user", "content": content}

    def _apply_image_window(self):
        while len(self._user_image_indices) >= self.image_window:
            old_index = self._user_image_indices.pop(0)
            if old_index >= len(self._messages):
                continue
            message = self._messages[old_index]
            if not isinstance(message.get("content"), list):
                continue
            message["content"] = [
                {"type": "text", "text": "[图片已省略]"}
                if isinstance(item, dict) and item.get("type") == "image_url"
                else item
                for item in message["content"]
            ]

    def _call_model(self, messages: list, max_retries: int = 3) -> tuple:
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=16384,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                        "repetition_penalty": 1.05,
                        "frequency_penalty": 0.3,
                    },
                )
                message = response.choices[0].message
                content = message.content or ""
                reasoning_content = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or ""
                return content, reasoning_content
            except Exception as error:
                last_error = error
                error_text = str(error)
                if "request_size_limit" in error_text or "body length exceed" in error_text:
                    if self._evict_oldest_image():
                        logger.warning("请求体超限，已淘汰一张图片")
                        continue
                if attempt < max_retries - 1:
                    wait_seconds = 2 ** attempt
                    logger.warning("监督模型调用失败: %s，等待 %d 秒", error, wait_seconds)
                    time.sleep(wait_seconds)
        logger.error("监督模型调用失败: %s", last_error)
        return "", ""

    def _evict_oldest_image(self) -> bool:
        for message in self._messages:
            if not isinstance(message.get("content"), list):
                continue
            for index, item in enumerate(message["content"]):
                if isinstance(item, dict) and item.get("type") == "image_url":
                    message["content"][index] = {"type": "text", "text": "[图片已省略]"}
                    return True
        return False
