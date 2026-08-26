import ast
import base64
import logging
import os
import re
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from PIL import Image

from policy.base_policy import BasePolicy


SCALE_FACTOR = 1000

ACTION_NAME_MAP = {
    "Click": "CLK",
    "DoubleClick": "DoubleCLK",
    "LongPress": "LongPress",
    "Swipe": "SWIPE",
    "Drag": "SWIPE",
    "Type": "INPUT",
    "LaunchApp": "REOPEN",
    "Wait": "WAIT",
    "CallUser": "CallUser",
    "GetScreenshot": "get_screenshot",
    "PressBack": "BACK",
    "PressHome": "PressHome",
    "PressEnter": "PressEnter",
    "PressRecent": "PressMenu",
    "Answer": "CallUser",
    "Finished": "SUCCESS",
}


def _parse_think_action(response: str) -> Tuple[str, str]:
    think = ""
    action = ""
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if think_match:
        think = think_match.group(1).strip()
    action_match = re.search(r"<action>(.*?)</action>", response, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()
    return think, action


def _parse_action_call(action: str) -> Tuple[str, dict]:
    match = re.match(r"(\w+)\((.*)\)\s*$", action.strip(), re.DOTALL)
    if not match:
        return action.strip(), {}
    name, params_text = match.group(1), match.group(2).strip()
    if not params_text:
        return name, {}
    escaped = params_text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    try:
        tree = ast.parse(f"_({escaped})", mode="eval")
        params = {item.arg: ast.literal_eval(item.value) for item in tree.body.keywords}
        return name, params
    except (SyntaxError, ValueError):
        return name, {}


def _to_pixel(coordinate: Tuple[float, float], width: int, height: int) -> Tuple[int, int]:
    if not isinstance(coordinate, (tuple, list)) or len(coordinate) != 2:
        raise ValueError("坐标必须包含两个数值")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in coordinate):
        raise ValueError("坐标必须是数值")
    if any(value < 0 or value >= SCALE_FACTOR for value in coordinate):
        raise ValueError("坐标必须在 [0, 999] 范围内")
    return (
        int(coordinate[0] / SCALE_FACTOR * width),
        int(coordinate[1] / SCALE_FACTOR * height),
    )


class UIVenus2Policy(BasePolicy):
    def __init__(
        self,
        runtime_context,
        model_url: str = "",
        model_name: str = "",
        api_key: str = "",
        temperature: float = 0.6,
        **kwargs,
    ):
        self.logger = logging.getLogger(__name__)
        self.runtime_context = runtime_context
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("API_KEY") or os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.client = OpenAI(base_url=model_url, api_key=self.api_key, timeout=120.0)
        self.last_action = None
        self._current_messages: List[dict] = []
        self._current_image_size: Tuple[int, int] = (0, 0)

    def get_next_action(self, state: Dict[str, Any]) -> Optional[tuple]:
        messages = state["user_query"]
        screenshot = state["screenshot_str"]
        image = Image.open(BytesIO(base64.b64decode(screenshot)))
        self._current_image_size = image.size
        self._current_messages = list(messages)
        self._current_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Current Screenshot:\n"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + screenshot}},
            ],
        })
        response = self._call_model(self._current_messages)
        if not response:
            return None, None, None, None, None
        self._current_messages.append({"role": "assistant", "content": response})
        action, pred_action, raw_response, think, conclusion = self._build_result(response)
        self.runtime_context.pred_action.append(pred_action)
        return action, pred_action, raw_response, think, conclusion

    def retry_with_feedback(self, feedback: str, state: Dict[str, Any]) -> tuple:
        self._current_messages.append({"role": "user", "content": feedback})
        response = self._call_model(self._current_messages)
        if not response:
            return None, None, None, None, None
        self._current_messages.append({"role": "assistant", "content": response})
        action, pred_action, raw_response, think, conclusion = self._build_result(response)
        if self.runtime_context.pred_action:
            self.runtime_context.pred_action[-1] = pred_action
        else:
            self.runtime_context.pred_action.append(pred_action)
        return action, pred_action, raw_response, think, conclusion

    def report_result(self, success: bool) -> None:
        if not success and self.last_action:
            self.logger.warning("操作失败: %s", self.last_action)
        self.last_action = None
        self._current_messages = []

    def _build_result(self, response: str) -> tuple:
        self.logger.info("模型输出: %s", response)
        width, height = self._current_image_size
        think, action_text = _parse_think_action(response)
        action = self._parse_to_action_dict(action_text, width, height)
        self.last_action = action
        return action, response, response, think, action_text

    def _parse_to_action_dict(self, action_text: str, width: int, height: int) -> dict:
        action = {
            "action_type": None,
            "action_pos": [],
            "input": "",
            "duration": -1,
            "role": "SIPA",
            "timestamp": None,
            "extend": "",
        }
        name, params = _parse_action_call(action_text)
        action_type = ACTION_NAME_MAP.get(name)
        if not action_type:
            self.logger.warning("未知动作: %s", name)
            return action
        action["action_type"] = action_type
        if action_type in ("CLK", "DoubleCLK", "LongPress"):
            coordinate = params.get("box", params.get("point"))
            if coordinate:
                x, y = _to_pixel(coordinate, width, height)
                action["action_pos"] = [[x, y]]
        elif action_type == "SWIPE":
            if "start" in params and "end" in params:
                start_x, start_y = _to_pixel(params["start"], width, height)
                end_x, end_y = _to_pixel(params["end"], width, height)
                action["action_pos"] = [[start_x, start_y], [end_x, end_y]]
            if name == "Drag":
                action["duration"] = 1000
        elif action_type == "INPUT":
            text = params.get("content", "")
            text = re.sub(r"\(", "（", text)
            text = re.sub(r"\)", "）", text)
            action["input"] = re.sub(r"\|", "｜", text)
        elif action_type == "REOPEN":
            action["app_name"] = params.get("app", "")
        elif action_type == "WAIT":
            action["duration"] = 1000
        elif action_type in ("CallUser", "SUCCESS"):
            action["input"] = params.get("content", "")
        self.logger.info("动作解析 %s", action)
        return action

    def _call_model(self, messages: list, retry_times: int = 3) -> Optional[str]:
        for attempt in range(retry_times):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=16384,
                    extra_body={
                        "repetition_penalty": 1.05,
                        "frequency_penalty": 0.3,
                    },
                )
                message = response.choices[0].message
                content = message.content or ""
                reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
                if reasoning:
                    content = f"<think>{reasoning}</think>\n{content}"
                return content
            except Exception as error:
                self.logger.warning("模型调用失败 (%d/%d): %s", attempt + 1, retry_times, error)
                if attempt < retry_times - 1:
                    time.sleep(2 ** attempt)
        self.logger.error("模型调用失败，已重试 %d 次", retry_times)
        return None
