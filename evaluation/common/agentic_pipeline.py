#!/usr/bin/env python3
"""Agentic high-resolution vision through ordinary assistant/user text-action turns.

Shared agentic pipeline for the sspro/OS-World-G, HR-Bench, and PerceptionBench
entries (shared module). The API receives no native tool schema. The assistant emits a textual crop_image action
or a mode-specific coordinate, option, or short-text answer. A turn may contain up to two
crop actions, whose results are returned together in one user message. Crop outputs retain
an exact mapping to the original image.
"""

from __future__ import annotations

import ast
import base64
import json
import math
import os
import re
import secrets
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image


DEFAULT_API_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_CROP_UPSCALE_FACTOR = 4.0
DEFAULT_CROP_MAX_PIXELS = 4_000_000
DEFAULT_CROP_MAX_DIMENSION = 4096
MAX_CROP_IMAGE_CALLS = 3
MAX_CROP_IMAGE_CALLS_PER_TURN = 2
TRUNCATED_ASSISTANT_CONTENT_CHARS = 2_000
DEGENERATE_RESCUE_MAX_TOKENS = 512
RATE_LIMIT_BACKOFF_SECONDS = (10, 20, 40, 80)
COORDINATE_RE = re.compile(
    r"\[\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\]"
)
ACTION_BLOCK_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL | re.IGNORECASE)
MCQ_ANSWER_TAG_RE = re.compile(
    r"<answer>\s*(?:answer\s*[:=]\s*)?([A-D])\s*[.)]?\s*</answer>",
    re.IGNORECASE,
)
MCQ_ANSWER_LABEL_RE = re.compile(
    r"(?:^|\n)\s*(?:final\s+)?answer\s*[:=]\s*([A-D])\s*[.)]?\s*$",
    re.IGNORECASE,
)
MCQ_TRAILING_LETTER_RE = re.compile(r"(?:^|\n)\s*([A-D])\s*[.)]?\s*$", re.IGNORECASE)
VQA_ANSWER_TAG_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE
)

CROP_IMAGE_DESCRIPTION = (
    "Crop an image that has already been supplied in this conversation. image_path "
    "must be the exact original path or a path returned by an earlier crop_image call. "
    "x1, y1, x2, y2 are absolute pixel coordinates in that referenced image's LOCAL "
    "coordinate system, with (x1,y1) at the top-left and (x2,y2) at the bottom-right. "
    "The result provides a new image path, dimensions, and the crop region mapped into "
    "the ORIGINAL image. Returned crops are resized for inspection, so nested crops must "
    "use the returned width and height. Crops may be nested. The final answer must always "
    "use relative coordinates of the ORIGINAL image normalized to the [0,1000] range, "
    "never crop-local coordinates. When writing a crop_image action, format each of x1, "
    "y1, x2, and y2 with exactly two digits after the decimal point."
)

MCQ_CROP_IMAGE_DESCRIPTION = (
    "Crop an image that has already been supplied in this conversation. image_path "
    "must be the exact original path or a path returned by an earlier crop_image call. "
    "x1, y1, x2, y2 are absolute pixel coordinates in that referenced image's LOCAL "
    "coordinate system, with (x1,y1) at the top-left and (x2,y2) at the bottom-right. "
    "The result provides a new image path and dimensions. Returned crops are resized "
    "for inspection, so nested crops must use the returned width and height. Crops may "
    "be nested. When writing a crop_image action, format x1, y1, x2, and y2 with exactly "
    "two digits after the decimal point. The final answer must be one option letter."
)

VQA_CROP_IMAGE_DESCRIPTION = (
    "Crop an image that has already been supplied in this conversation. image_path "
    "must be the exact original path or a path returned by an earlier crop_image call. "
    "x1, y1, x2, y2 are absolute pixel coordinates in that referenced image's LOCAL "
    "coordinate system, with (x1,y1) at the top-left and (x2,y2) at the bottom-right. "
    "The result provides a new image path and dimensions. Returned crops are resized "
    "for inspection, so nested crops must use the returned width and height. Crops may "
    "be nested. When writing a crop_image action, format x1, y1, x2, and y2 with exactly "
    "two digits after the decimal point. The final answer must be a concise text value "
    "inside one <answer>...</answer> tag."
)

CROP_IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "crop_image",
        "description": CROP_IMAGE_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Exact original or previously returned crop path.",
                },
                "x1": {"type": "number", "description": "Left local pixel."},
                "y1": {"type": "number", "description": "Top local pixel."},
                "x2": {"type": "number", "description": "Right local pixel."},
                "y2": {"type": "number", "description": "Bottom local pixel."},
            },
            "required": ["image_path", "x1", "y1", "x2", "y2"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT_TEMPLATE = """You are an agentic GUI grounding model.

Task instruction:
{instruction}

Original image metadata:
- image_path: {image_path}
- resolution: {width}x{height} pixels (width x height)
- origin: [0,0] at the top-left

Goal:
Output the center point of the position corresponding to the task instruction. The final
coordinate MUST be relative to the ORIGINAL image and normalized to [0.00,1000.00], where
[0.00,0.00] is the original image's top-left and [1000.00,1000.00] is its bottom-right.
Never output absolute pixels or coordinates local to a crop. Preserve sub-unit precision:
the two final coordinate values MUST each have exactly two digits after the decimal point
(for example, [512.34,76.50]). If the task is infeasible (for example, the requested UI
element or action is absent from the image), use [-1.00,-1.00].

Available local action:
- crop_image: {crop_description}

Text-action protocol:
Every turn must first emit visible reasoning inside <think>...</think>, then end with
exactly one of these choices:
- To inspect one region:
  <think>brief reasoning for the crop</think>
  <action>crop_image(image_path="exact path", x1=100.00, y1=200.00, x2=500.00, y2=600.00)</action>
- To inspect two independent regions in the same turn:
  <think>brief reasoning for both crops</think>
  <action>crop_image(image_path="exact path", x1=100.00, y1=200.00, x2=500.00, y2=600.00)</action>
  <action>crop_image(image_path="exact path", x1=600.00, y1=200.00, x2=900.00, y2=600.00)</action>
- To return the final point:
  <think>brief reasoning for the final point</think>
  [x.xx,y.yy]

crop_image is a textual action interpreted locally after your answer. It is NOT an API
function call. Do not emit JSON, <tool_call> XML, or native tool_calls. Keep x1, y1, x2,
and y2 as four separately named numeric arguments.

Tool rules:
1. Before returning the final coordinate, you SHOULD make at least one successful
   crop_image call to inspect the target more precisely. A direct first-turn coordinate is
   discouraged. Skip cropping only when the task is clearly infeasible or cropping cannot
   reasonably improve localization.
2. You may make at most {max_tool_calls} crop_image calls in total. Never exceed this
   budget; use each crop purposefully and return the final coordinate as soon as the target
   is localized. A single assistant turn may contain one or two crop_image actions, and
   every action counts separately toward the total budget. Never emit more than two
   actions in one turn or mix crop actions with a final coordinate.
3. Tool arguments are local absolute pixels in the image_path passed to that call. Write
   x1, y1, x2, and y2 with exactly two digits after the decimal point, including trailing
   zeros (for example, x1=100.00). Two actions emitted in the same turn must refer only
   to image paths already available before that turn; the second action cannot depend on
   the result of the first action from the same turn.
4. Each result states the crop's region both in original-image pixels and original-image
   normalized [0,1000] coordinates. Use that mapping to convert any point found in a crop
   back to the ORIGINAL image's normalized [0,1000] coordinate system.
5. After crop_image actions, the following single user message contains all result
   metadata and all successful crop images in action order.

Final-answer rule:
When ready, output exactly <think>...</think> followed by a newline and [x.xx,y.yy], with
no prose, Markdown, or action wrapper after the coordinate. Both values must contain
exactly two digits after the decimal point, including trailing zeros. The point must be in
the ORIGINAL image's normalized [0,1000] coordinate system. If the task is infeasible, use
[-1.00,-1.00].
"""

DIRECT_GROUNDING_PROMPT_TEMPLATE = (
    "Output the center point of the position corresponding to the following "
    "instruction: \n{instruction}. \n\nThe output should just be the coordinates "
    "of a point, in the format [x,y]. Additionally, if the task is infeasible "
    "(e.g., the task is not related to the image), the output should be [-1,-1]."
)

# Grounding think/nothink runs intentionally use the exact same user prompt. The
# only mode difference belongs in the vLLM chat-template request parameters.
DIRECT_NOTHINK_PROMPT_TEMPLATE = DIRECT_GROUNDING_PROMPT_TEMPLATE

MCQ_SYSTEM_PROMPT_TEMPLATE = """You are an agentic high-resolution image question-answering model.

Multiple-choice task:
{instruction}

Original image metadata:
- image_path: {image_path}
- resolution: {width}x{height} pixels (width x height)
- origin: [0,0] at the top-left

Goal:
Determine the single correct option from A, B, C, and D. Inspect the original image and
use crop_image when magnifying one or more regions would improve confidence. Do not alter,
reorder, or invent options. The final answer must identify exactly one option letter.

Available local action:
- crop_image: {crop_description}

Text-action protocol:
Every turn must first emit visible reasoning inside <think>...</think>, then end with
exactly one of these choices:
- To inspect one region:
  <think>brief reasoning for the crop</think>
  <action>crop_image(image_path="exact path", x1=100.00, y1=200.00, x2=500.00, y2=600.00)</action>
- To inspect two independent regions in the same turn:
  <think>brief reasoning for both crops</think>
  <action>crop_image(image_path="exact path", x1=100.00, y1=200.00, x2=500.00, y2=600.00)</action>
  <action>crop_image(image_path="exact path", x1=600.00, y1=200.00, x2=900.00, y2=600.00)</action>
- To return the final option:
  <think>brief reasoning for the answer</think>
  <answer>A</answer>

crop_image is a textual action interpreted locally after your answer. It is NOT an API
function call. Do not emit JSON, <tool_call> XML, or native tool_calls. Keep x1, y1, x2,
and y2 as four separately named numeric arguments.

Tool rules:
1. Use crop_image when small text, fine visual attributes, charts, maps, or comparisons
   cannot be resolved confidently from the full image. Return the answer directly when
   cropping cannot reasonably help.
2. You may make at most {max_tool_calls} crop_image calls in total. A single assistant
   turn may contain one or two crop_image actions, and every action counts separately.
   Never mix crop actions with a final answer.
3. Tool arguments are local absolute pixels in the image_path passed to that call. Write
   x1, y1, x2, and y2 with exactly two digits after the decimal point. Two actions in the
   same turn may only refer to paths available before that turn.
4. After crop_image actions, the following user message contains all result metadata and
   all successful crop images in action order.

Final-answer rule:
When ready, output exactly <think>...</think> followed by a newline and one
<answer>LETTER</answer> tag, with LETTER replaced by A, B, C, or D. Do not include prose,
Markdown, coordinates, or an action after the answer tag.
"""

MCQ_DIRECT_NOTHINK_PROMPT_TEMPLATE = (
    "{instruction}\n\nAnswer directly with exactly one option using the format "
    "<answer>A</answer>, replacing A with the correct letter from A, B, C, or D."
)

DIRECT_THINK_PROMPT_TEMPLATE = DIRECT_GROUNDING_PROMPT_TEMPLATE

# SSPro direct prompts are selected by direct_prompt_variant="sspro".
# Preserve the separate SSPro and HR-Bench ("grounding") prompt variants.
SSPRO_DIRECT_NOTHINK_PROMPT_TEMPLATE = (
    "Output the center point of the position corresponding to the following "
    "instruction: \n{instruction}. \n\nThe output should just be the coordinates "
    "of a point, in the format [x,y]."
)

SSPRO_DIRECT_THINK_PROMPT_TEMPLATE = (
    "{instruction}\n\nFirst reason carefully about the image and the instruction inside "
    "<think>...</think>. Then output the final coordinates as [x,y] after the "
    "</think> tag. Do not use tools or request additional images."
)

MCQ_DIRECT_THINK_PROMPT_TEMPLATE = (
    "{instruction}\n\nFirst reason carefully about the image and the choices inside "
    "<think>...</think>. Then output exactly one final option using the format "
    "<answer>A</answer>, replacing A with the correct letter from A, B, C, or D. "
    "Do not use tools or request crops."
)

VQA_SYSTEM_PROMPT_TEMPLATE = """You are an agentic high-resolution chart question-answering model.

Chart question-answering task:
{instruction}

Original image metadata:
- image_path: {image_path}
- resolution: {width}x{height} pixels (width x height)
- origin: [0,0] at the top-left

Goal:
Answer the task from the chart image, the supplied text context, and any conversation
history in the task. Use crop_image when magnifying chart labels, legends, axes, values,
or other fine details would improve confidence. Follow the task's answer-format rules.

Available local action:
- crop_image: {crop_description}

Text-action protocol:
Every turn must first emit visible reasoning inside <think>...</think>, then end with
exactly one of these choices:
- To inspect one region:
  <think>brief reasoning for the crop</think>
  <action>crop_image(image_path="exact path", x1=100.00, y1=200.00, x2=500.00, y2=600.00)</action>
- To inspect two independent regions in the same turn:
  <think>brief reasoning for both crops</think>
  <action>crop_image(image_path="exact path", x1=100.00, y1=200.00, x2=500.00, y2=600.00)</action>
  <action>crop_image(image_path="exact path", x1=600.00, y1=200.00, x2=900.00, y2=600.00)</action>
- To return the final answer:
  <think>brief reasoning for the answer</think>
  <answer>SHORT ANSWER</answer>

crop_image is a textual action interpreted locally after your answer. It is NOT an API
function call. Do not emit JSON, <tool_call> XML, or native tool_calls. Keep x1, y1, x2,
and y2 as four separately named numeric arguments.

Tool rules:
1. Use crop_image when small chart text, fine visual attributes, or comparisons cannot be
   resolved confidently from the full image. Return the answer directly when cropping
   cannot reasonably help.
2. You may make at most {max_tool_calls} crop_image calls in total. A single assistant
   turn may contain one or two crop_image actions, and every action counts separately.
   Never mix crop actions with a final answer.
3. Tool arguments are local absolute pixels in the image_path passed to that call. Write
   x1, y1, x2, and y2 with exactly two digits after the decimal point. Two actions in the
   same turn may only refer to paths available before that turn.
4. After crop_image actions, the following user message contains all result metadata and
   all successful crop images in action order.

Final-answer rule:
When ready, output exactly <think>...</think> followed by a newline and one
<answer>SHORT ANSWER</answer> tag. The tag value must contain only the concise answer
requested by the task: no explanation, Markdown, confidence, or trailing commentary.
Use the literal answer Unanswerable when the requested information cannot be determined.
"""

VQA_DIRECT_NOTHINK_PROMPT_TEMPLATE = (
    "{instruction}\n\nReturn only the concise final answer inside exactly one "
    "<answer>...</answer> tag. Do not include reasoning or additional text."
)

VQA_DIRECT_THINK_PROMPT_TEMPLATE = (
    "{instruction}\n\nFirst reason carefully about the image and task inside "
    "<think>...</think>. Then output the final answer inside exactly one "
    "<answer>...</answer> tag, preserving any JSON, HTML, Markdown, LaTeX, "
    "coordinates, or other answer format requested by the task. Do not use tools "
    "or request crops."
)


class SlidingWindowRateLimiter:
    """Thread-safe request-start limiter for rolling second and minute windows."""

    def __init__(self, qpm: int = 60, rps: int = 5) -> None:
        if qpm < 1 or rps < 1:
            raise ValueError("qpm and rps must both be positive")
        self.qpm = qpm
        self.rps = rps
        self.minute_starts: deque[float] = deque()
        self.second_starts: deque[float] = deque()
        self.condition = threading.Condition()

    def acquire(self) -> None:
        with self.condition:
            while True:
                now = time.monotonic()
                while self.minute_starts and self.minute_starts[0] <= now - 60:
                    self.minute_starts.popleft()
                while self.second_starts and self.second_starts[0] <= now - 1:
                    self.second_starts.popleft()
                if len(self.minute_starts) < self.qpm and len(self.second_starts) < self.rps:
                    self.minute_starts.append(now)
                    self.second_starts.append(now)
                    return
                waits = []
                if len(self.minute_starts) >= self.qpm:
                    waits.append(self.minute_starts[0] + 60 - now)
                if len(self.second_starts) >= self.rps:
                    waits.append(self.second_starts[0] + 1 - now)
                self.condition.wait(timeout=max(min(waits), 0.01))


def _detect_degenerate(
    text: str | None, *, max_unit: int = 64, min_reps: int = 10
) -> dict[str, int] | None:
    """Whole text is one short unit repeated many times (a token-loop failure)."""
    if not text:
        return None
    n = len(text)
    for unit in range(1, min(max_unit, n // min_reps) + 1):
        if n % unit == 0 and text[:unit] * (n // unit) == text:
            return {"unit_chars": unit, "reps": n // unit}
    return None


def parse_point(text: str) -> list[float] | None:
    action = parse_text_action(text)
    if action and action["type"] == "final":
        return action["point"]
    return None


def _coordinate_after_think(text: str) -> re.Match[str] | None:
    """Return the last coordinate after the final closing think tag, if present."""
    value = text or ""
    think_closes = list(re.finditer(r"</think\s*>", value, re.IGNORECASE))
    if think_closes:
        suffix = value[think_closes[-1].end() :]
        suffix_matches = list(COORDINATE_RE.finditer(suffix))
        if suffix_matches:
            return suffix_matches[-1]
    return None


def _final_coordinate_match(text: str) -> re.Match[str] | None:
    """Prefer a coordinate after </think>, otherwise return the last one overall."""
    after_think = _coordinate_after_think(text)
    if after_think:
        return after_think
    value = text or ""
    matches = list(COORDINATE_RE.finditer(value))
    return matches[-1] if matches else None


def image_size(image_path: str) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def encode_image(image_path: str, *, jpeg_quality: int = 90) -> str:
    """Encode an API-only JPEG copy without changing source pixels or dimensions."""
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    with Image.open(image_path) as image:
        image.load()
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, "white")
            rgb.paste(rgba, mask=rgba.getchannel("A"))
        else:
            rgb = image.convert("RGB")

        buffer = BytesIO()
        rgb.save(
            buffer,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
            subsampling=2,
        )

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def text_from_delta(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            item if isinstance(item, str) else str(item.get("text", ""))
            for item in value
            if isinstance(item, (str, dict))
        )
    return str(value)


def _registered_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _safe_task_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return (cleaned or "task")[:160]


def _json_dumps_two_decimal_floats(value: Any) -> str:
    """Serialize JSON while rendering every finite float with exactly two decimals."""
    if isinstance(value, float) and math.isfinite(value):
        return f"{value:.2f}"
    if isinstance(value, dict):
        items = (
            f"{json.dumps(key, ensure_ascii=False)}: "
            f"{_json_dumps_two_decimal_floats(item)}"
            for key, item in value.items()
        )
        return "{" + ", ".join(items) + "}"
    if isinstance(value, (list, tuple)):
        items = (_json_dumps_two_decimal_floats(item) for item in value)
        return "[" + ", ".join(items) + "]"
    return json.dumps(value, ensure_ascii=False)


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise ValueError("Tool arguments must be a JSON object")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Tool arguments must decode to an object")
    return value


def _extract_call_source(text: str) -> str | None:
    """Extract the first supported function-style action with balanced parentheses."""
    match = re.search(r"\b(?:crop_image|crop|click|finished)\s*\(", text, re.IGNORECASE)
    if not match:
        return None
    start = match.start()
    open_index = text.find("(", start)
    depth = 0
    quote = None
    escaped = False
    for index in range(open_index, len(text)):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _literal_value(node: ast.AST) -> Any:
    # literal_eval can produce Ellipsis when a model copies the "..." placeholder.
    # Ellipsis is not JSON serializable and would break trajectory checkpoints.
    # Normalize it to the literal string "...".
    if isinstance(node, ast.Constant) and node.value is Ellipsis:
        return "..."
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise ValueError("Action arguments must be Python literals") from exc


def _parse_function_action(source: str) -> tuple[str, dict[str, Any]]:
    try:
        expression = ast.parse(source.strip(), mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Invalid action syntax: {source}") from exc
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise ValueError("Expected a function-style action")
    if any(keyword.arg is None for keyword in expression.keywords):
        raise ValueError("Action does not support **kwargs")
    arguments = {
        str(keyword.arg): _literal_value(keyword.value)
        for keyword in expression.keywords
    }
    positional = [_literal_value(item) for item in expression.args]
    name = expression.func.id
    if positional:
        if name.lower() in {"crop", "crop_image"}:
            keys = ("image_path", "x1", "y1", "x2", "y2")
        else:
            keys = ("box",)
        if len(positional) > len(keys):
            raise ValueError("Too many positional action arguments")
        for key, value in zip(keys, positional):
            arguments.setdefault(key, value)
    return name, arguments


def _repair_malformed_crop_call(source: str) -> dict[str, Any] | None:
    """Repair the observed `x1=100,336, x2=300,388` model shorthand."""
    if not re.match(r"\s*(?:crop_image|crop)\s*\(", source, re.IGNORECASE):
        return None
    path_match = re.search(
        r"image_path\s*=\s*(['\"])(.*?)\1",
        source,
        re.DOTALL | re.IGNORECASE,
    )
    number = r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    pair_match = re.search(
        rf"x1\s*=\s*{number}\s*,\s*{number}\s*,\s*"
        rf"x2\s*=\s*{number}\s*,\s*{number}",
        source,
        re.IGNORECASE,
    )
    if not pair_match:
        return None
    values = [float(item) for item in pair_match.groups()]
    arguments: dict[str, Any] = dict(zip(("x1", "y1", "x2", "y2"), values))
    if path_match:
        arguments["image_path"] = path_match.group(2)
    return arguments


def _parse_xml_tool_action(text: str) -> tuple[str, dict[str, Any]] | None:
    """Compatibility fallback for a fine-tuned model that still emits Qwen XML."""
    block = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL | re.IGNORECASE)
    if not block:
        return None
    function = re.search(
        r"<function=([^>]+)>(.*?)</function>",
        block.group(1),
        re.DOTALL | re.IGNORECASE,
    )
    if not function:
        return None
    arguments: dict[str, Any] = {}
    for key, raw_value in re.findall(
        r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
        function.group(2),
        re.DOTALL | re.IGNORECASE,
    ):
        value = raw_value.strip()
        try:
            parsed_value = ast.literal_eval(value)
            # Normalize copied Ellipsis placeholders to a JSON-serializable string.
            arguments[key.strip()] = "..." if parsed_value is Ellipsis else parsed_value
        except (ValueError, SyntaxError):
            arguments[key.strip()] = value
    return function.group(1).strip(), arguments


def _normalize_crop_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(arguments)
    box = normalized.pop("box", None)
    if box is not None and not all(key in normalized for key in ("x1", "y1", "x2", "y2")):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("crop_image box must contain x1, y1, x2, y2")
        normalized.update(dict(zip(("x1", "y1", "x2", "y2"), box)))

    first = normalized.get("x1")
    second = normalized.get("x2")
    if "y1" not in normalized and isinstance(first, (list, tuple)) and len(first) == 2:
        normalized["x1"], normalized["y1"] = first
    if "y2" not in normalized and isinstance(second, (list, tuple)) and len(second) == 2:
        normalized["x2"], normalized["y2"] = second
    return normalized


def parse_mcq_answer(text: str) -> str | None:
    """Parse one unambiguous final A-D answer from the supported output formats."""
    value = text or ""
    tag_matches = [match.group(1).upper() for match in MCQ_ANSWER_TAG_RE.finditer(value)]
    if tag_matches:
        return tag_matches[0] if len(set(tag_matches)) == 1 else None

    think_closes = list(re.finditer(r"</think\s*>", value, re.IGNORECASE))
    suffix = value[think_closes[-1].end() :] if think_closes else value
    label_matches = [
        match.group(1).upper() for match in MCQ_ANSWER_LABEL_RE.finditer(suffix)
    ]
    if label_matches:
        return label_matches[0] if len(set(label_matches)) == 1 else None

    trailing = MCQ_TRAILING_LETTER_RE.search(suffix)
    return trailing.group(1).upper() if trailing else None


def parse_vqa_answer(text: str) -> str | None:
    """Parse one concise free-form answer without consuming crop actions."""
    value = text or ""
    tag_matches = [match.group(1).strip() for match in VQA_ANSWER_TAG_RE.finditer(value)]
    if tag_matches:
        if any(not answer for answer in tag_matches):
            return None
        return tag_matches[0] if len(set(tag_matches)) == 1 else None

    try:
        json_value = json.loads(value.strip())
    except (json.JSONDecodeError, TypeError):
        json_value = None
    json_actions = json_value if isinstance(json_value, list) else [json_value]
    has_json_crop = any(
        isinstance(item, dict)
        and str(item.get("name") or item.get("action") or "").lower()
        in {"crop", "crop_image"}
        for item in json_actions
    )
    if (
        ACTION_BLOCK_RE.search(value)
        or re.search(r"<tool_call>.*?</tool_call>", value, re.DOTALL | re.IGNORECASE)
        or _extract_call_source(value) is not None
        or has_json_crop
        or re.search(r"</?answer\b", value, re.IGNORECASE)
    ):
        return None

    think_closes = list(re.finditer(r"</think\s*>", value, re.IGNORECASE))
    suffix = value[think_closes[-1].end() :] if think_closes else value
    suffix = re.sub(
        r"^\s*(?:final\s+)?answer\s*[:=]\s*",
        "",
        suffix,
        flags=re.IGNORECASE,
    ).strip()
    return suffix or None


def _parse_single_text_action(
    text: str,
    *,
    answer_mode: str = "point",
) -> dict[str, Any] | None:
    """Parse one textual crop_image action or mode-specific final answer."""
    value = text or ""
    if answer_mode == "mcq":
        option = parse_mcq_answer(value)
        if option is not None:
            return {
                "type": "final",
                "name": "Answer",
                "option": option,
                "source": value.strip(),
                "compatibility_format": (
                    "answer_tag" if MCQ_ANSWER_TAG_RE.search(value) else "answer_label"
                ),
            }
    elif answer_mode == "vqa":
        answer = parse_vqa_answer(value)
        if answer is not None:
            return {
                "type": "final",
                "name": "Answer",
                "answer": answer,
                "source": value.strip(),
                "compatibility_format": (
                    "answer_tag" if VQA_ANSWER_TAG_RE.search(value) else "plain_text"
                ),
            }
    elif answer_mode != "point":
        raise ValueError("answer_mode must be 'point', 'mcq', or 'vqa'")

    coordinate_after_think = (
        _coordinate_after_think(value) if answer_mode == "point" else None
    )
    if coordinate_after_think:
        return {
            "type": "final",
            "name": "Click",
            "point": [
                float(coordinate_after_think.group(1)),
                float(coordinate_after_think.group(2)),
            ],
            "source": coordinate_after_think.group(0),
            "compatibility_format": "coordinate_after_think",
        }

    trailing_coordinate = (
        _final_coordinate_match(value) if answer_mode == "point" else None
    )
    if trailing_coordinate and not value[trailing_coordinate.end() :].strip():
        return {
            "type": "final",
            "name": "Click",
            "point": [
                float(trailing_coordinate.group(1)),
                float(trailing_coordinate.group(2)),
            ],
            "source": trailing_coordinate.group(0),
            "compatibility_format": "trailing_coordinate",
        }

    action_blocks = ACTION_BLOCK_RE.findall(value)
    candidate = action_blocks[-1].strip() if action_blocks else value.strip()

    call_source = _extract_call_source(candidate)
    if call_source:
        repaired_arguments = None
        try:
            name, arguments = _parse_function_action(call_source)
        except ValueError:
            repaired_arguments = _repair_malformed_crop_call(call_source)
            name = "crop_image" if repaired_arguments else ""
            arguments = repaired_arguments or {}
        lowered = name.lower()
        if lowered in {"crop", "crop_image"}:
            try:
                normalized = _normalize_crop_arguments(arguments)
            except ValueError:
                return None
            action = {
                "type": "crop",
                "name": "crop_image",
                "arguments": normalized,
                "source": call_source,
            }
            if repaired_arguments:
                action["compatibility_format"] = "paired_coordinates"
            return action
        if lowered in {"click", "finished"} and answer_mode == "point":
            point = arguments.get("box", arguments.get("point"))
            if point is None and "x" in arguments and "y" in arguments:
                point = (arguments["x"], arguments["y"])
            if isinstance(point, (list, tuple)) and len(point) == 2:
                try:
                    parsed_point = [float(point[0]), float(point[1])]
                except (TypeError, ValueError):
                    return None
                return {
                    "type": "final",
                    "name": "Click",
                    "point": parsed_point,
                    "source": call_source,
                }

    try:
        json_action = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        json_action = None
    if isinstance(json_action, dict):
        name = str(json_action.get("name") or json_action.get("action") or "")
        arguments = json_action.get("arguments") or {}
        if name.lower() in {"crop", "crop_image"} and isinstance(arguments, dict):
            try:
                normalized = _normalize_crop_arguments(arguments)
            except ValueError:
                return None
            return {
                "type": "crop",
                "name": "crop_image",
                "arguments": normalized,
                "source": candidate,
                "compatibility_format": "json",
            }

    xml_action = _parse_xml_tool_action(candidate)
    if xml_action:
        name, arguments = xml_action
        if name.lower() in {"crop", "crop_image"}:
            try:
                normalized = _normalize_crop_arguments(arguments)
            except ValueError:
                return None
            return {
                "type": "crop",
                "name": "crop_image",
                "arguments": normalized,
                "source": candidate,
                "compatibility_format": "qwen_xml",
            }

    coordinate = _final_coordinate_match(value) if answer_mode == "point" else None
    if coordinate:
        return {
            "type": "final",
            "name": "Click",
            "point": [float(coordinate.group(1)), float(coordinate.group(2))],
            "source": coordinate.group(0),
            "compatibility_format": "coordinate",
        }
    return None


def parse_text_actions(
    text: str,
    *,
    answer_mode: str = "point",
) -> list[dict[str, Any]] | None:
    """Parse one mode-specific final answer or up to two crop actions."""
    if answer_mode not in {"point", "mcq", "vqa"}:
        raise ValueError("answer_mode must be 'point', 'mcq', or 'vqa'")
    value = text or ""
    action_matches = list(ACTION_BLOCK_RE.finditer(value))
    if action_matches:
        if len(action_matches) > MAX_CROP_IMAGE_CALLS_PER_TURN:
            return None
        if answer_mode == "vqa" and value[action_matches[-1].end() :].strip():
            return None
        has_final = {
            "point": lambda: _final_coordinate_match(value),
            "mcq": lambda: parse_mcq_answer(value),
            "vqa": lambda: parse_vqa_answer(value),
        }[answer_mode]()
        if has_final:
            return None
        actions = [
            _parse_single_text_action(
                match.group(1).strip(), answer_mode=answer_mode
            )
            for match in action_matches
        ]
        if any(action is None for action in actions):
            return None
        parsed_actions = [action for action in actions if action is not None]
        if len(parsed_actions) > 1 and any(
            action["type"] != "crop" for action in parsed_actions
        ):
            return None
        return parsed_actions

    xml_matches = list(
        re.finditer(
            r"<tool_call>.*?</tool_call>",
            value,
            re.DOTALL | re.IGNORECASE,
        )
    )
    if len(xml_matches) > 1:
        if len(xml_matches) > MAX_CROP_IMAGE_CALLS_PER_TURN:
            return None
        actions = [
            _parse_single_text_action(match.group(0), answer_mode=answer_mode)
            for match in xml_matches
        ]
        if any(action is None or action["type"] != "crop" for action in actions):
            return None
        return [action for action in actions if action is not None]

    try:
        json_batch = json.loads(value.strip())
    except (json.JSONDecodeError, TypeError):
        json_batch = None
    if isinstance(json_batch, list) and all(
        isinstance(item, dict) for item in json_batch
    ):
        if not 1 <= len(json_batch) <= MAX_CROP_IMAGE_CALLS_PER_TURN:
            return None
        actions = [
            _parse_single_text_action(
                json.dumps(item, ensure_ascii=False), answer_mode=answer_mode
            )
            for item in json_batch
        ]
        if any(action is None or action["type"] != "crop" for action in actions):
            return None
        return [action for action in actions if action is not None]

    action = _parse_single_text_action(value, answer_mode=answer_mode)
    return [action] if action is not None else None


def parse_text_action(
    text: str,
    *,
    answer_mode: str = "point",
) -> dict[str, Any] | None:
    """Parse model content while preserving the legacy single-action return shape."""
    actions = parse_text_actions(text, answer_mode=answer_mode)
    if not actions:
        return None
    if len(actions) == 1:
        return actions[0]
    return {
        "type": "crop_batch",
        "name": "crop_image",
        "actions": actions,
        "source": "\n".join(str(action.get("source") or "") for action in actions),
        "call_count": len(actions),
    }


class AgenticGroundingPipeline:
    """Multi-turn grounding agent using locally interpreted text actions."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        api_url: str = DEFAULT_API_URL,
        model: str = "kimi-k3",
        reasoning_effort: str = "high",
        temperature: float = 0.0,
        thinking_budget: int = 4_096,
        max_completion_tokens: int | None = None,
        max_tool_calls: int = MAX_CROP_IMAGE_CALLS,
        max_action_corrections: int = 2,
        api_image_quality: int = 90,
        crop_upscale_factor: float = DEFAULT_CROP_UPSCALE_FACTOR,
        crop_max_pixels: int = DEFAULT_CROP_MAX_PIXELS,
        crop_max_dimension: int = DEFAULT_CROP_MAX_DIMENSION,
        crop_root: str | Path = Path(__file__).resolve().parents[1] / "results" / "crops",
        persist_crops: bool = True,
        qpm: int = 60,
        rps: int = 5,
        timeout: float = 600.0,
        max_attempts: int = 5,
        workers_hint: int = 32,
        prompt_mode: str = "agentic",
        answer_mode: str = "point",
        direct_system_prompt: str | None = None,
        direct_prompt_passthrough: bool = False,
        direct_prompt_variant: str = "grounding",
        truncate_length_capped_turns: bool = True,
        client: httpx.Client | None = None,
        limiter: SlidingWindowRateLimiter | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY or api_key is required")
        if thinking_budget < 1 or max_tool_calls < 0 or max_action_corrections < 0:
            raise ValueError(
                "thinking_budget must be positive; tool and correction limits "
                "must be nonnegative"
            )
        if not 1 <= api_image_quality <= 100:
            raise ValueError("api_image_quality must be between 1 and 100")
        if not math.isfinite(crop_upscale_factor) or crop_upscale_factor <= 0:
            raise ValueError("crop_upscale_factor must be a finite positive number")
        if crop_max_pixels < 1:
            raise ValueError("crop_max_pixels must be positive")
        if crop_max_dimension < 1:
            raise ValueError("crop_max_dimension must be positive")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be a finite nonnegative number")
        if prompt_mode not in {"agentic", "direct_nothink", "direct_think"}:
            raise ValueError(
                "prompt_mode must be 'agentic', 'direct_nothink', or 'direct_think'"
            )
        if answer_mode not in {"point", "mcq", "vqa"}:
            raise ValueError("answer_mode must be 'point', 'mcq', or 'vqa'")
        self.api_url = api_url
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.max_completion_tokens = max_completion_tokens or thinking_budget
        # The interaction contract allows no more than three crop actions, even if an
        # older caller still supplies the previous default of five.
        self.max_tool_calls = min(max_tool_calls, MAX_CROP_IMAGE_CALLS)
        self.max_action_corrections = max_action_corrections
        self.api_image_quality = api_image_quality
        self.crop_upscale_factor = crop_upscale_factor
        self.crop_max_pixels = crop_max_pixels
        self.crop_max_dimension = crop_max_dimension
        self.crop_root = Path(crop_root).expanduser().resolve()
        self.persist_crops = persist_crops
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.prompt_mode = prompt_mode
        self.answer_mode = answer_mode
        self.direct_system_prompt = (
            direct_system_prompt.strip() if direct_system_prompt else None
        )
        self.direct_prompt_passthrough = bool(direct_prompt_passthrough)
        if direct_prompt_variant not in {"grounding", "sspro"}:
            raise ValueError("direct_prompt_variant must be 'grounding' or 'sspro'")
        self.direct_prompt_variant = direct_prompt_variant
        # False preserves SSPro replay and correction behavior for length-capped turns.
        self.truncate_length_capped_turns = bool(truncate_length_capped_turns)
        self.limiter = limiter or SlidingWindowRateLimiter(qpm=qpm, rps=rps)
        self.workers_hint = max(workers_hint, 1)
        self._branch_executor = ThreadPoolExecutor(max_workers=self.workers_hint)
        self._owns_client = client is None
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(timeout, connect=min(30.0, timeout)),
            limits=httpx.Limits(
                max_connections=max(workers_hint * 2, 8),
                max_keepalive_connections=max(workers_hint, 4),
            ),
            trust_env=False,
        )

    def close(self) -> None:
        self._branch_executor.shutdown(wait=True)
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "AgenticGroundingPipeline":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def build_system_prompt(
        self, instruction: str, image_path: str, width: int, height: int
    ) -> str:
        templates = {
            "point": SYSTEM_PROMPT_TEMPLATE,
            "mcq": MCQ_SYSTEM_PROMPT_TEMPLATE,
            "vqa": VQA_SYSTEM_PROMPT_TEMPLATE,
        }
        crop_descriptions = {
            "point": CROP_IMAGE_DESCRIPTION,
            "mcq": MCQ_CROP_IMAGE_DESCRIPTION,
            "vqa": VQA_CROP_IMAGE_DESCRIPTION,
        }
        template = templates[self.answer_mode]
        crop_description = crop_descriptions[self.answer_mode]
        return template.format(
            instruction=instruction.rstrip().rstrip("."),
            image_path=image_path,
            width=width,
            height=height,
            crop_description=crop_description,
            max_tool_calls=self.max_tool_calls,
            thinking_budget=self.thinking_budget,
        )

    def user_image_message(self, image_path: str, *, encode: bool) -> dict[str, Any]:
        image_url: dict[str, Any]
        if encode:
            image_url = {
                "url": encode_image(
                    image_path,
                    jpeg_quality=self.api_image_quality,
                )
            }
        else:
            image_url = {"path": image_path}
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": f"image_path: {image_path}"},
                {"type": "image_url", "image_url": image_url},
            ],
        }

    def direct_nothink_user_message(
        self,
        instruction: str,
        image_path: str,
        *,
        encode: bool,
    ) -> dict[str, Any]:
        """Build the mode-specific one-shot prompt with thinking disabled."""
        normalized_instruction = instruction.rstrip()
        if self.direct_prompt_passthrough:
            prompt = normalized_instruction
        else:
            if normalized_instruction.endswith("."):
                normalized_instruction = normalized_instruction[:-1]
            template = {
                "point": (
                    SSPRO_DIRECT_NOTHINK_PROMPT_TEMPLATE
                    if self.direct_prompt_variant == "sspro"
                    else DIRECT_NOTHINK_PROMPT_TEMPLATE
                ),
                "mcq": MCQ_DIRECT_NOTHINK_PROMPT_TEMPLATE,
                "vqa": VQA_DIRECT_NOTHINK_PROMPT_TEMPLATE,
            }[self.answer_mode]
            prompt = template.format(instruction=normalized_instruction)
        image_url = (
            {
                "url": encode_image(
                    image_path,
                    jpeg_quality=self.api_image_quality,
                )
            }
            if encode
            else {"path": image_path}
        )
        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": image_url},
                {"type": "text", "text": prompt},
            ],
        }

    def direct_think_user_message(
        self,
        instruction: str,
        image_path: str,
        *,
        encode: bool,
    ) -> dict[str, Any]:
        """Build a one-shot prompt that asks for reasoning without exposing tools."""
        normalized_instruction = instruction.rstrip()
        if self.direct_prompt_passthrough:
            prompt = normalized_instruction
        else:
            if normalized_instruction.endswith("."):
                normalized_instruction = normalized_instruction[:-1]
            template = {
                "point": (
                    SSPRO_DIRECT_THINK_PROMPT_TEMPLATE
                    if self.direct_prompt_variant == "sspro"
                    else DIRECT_THINK_PROMPT_TEMPLATE
                ),
                "mcq": MCQ_DIRECT_THINK_PROMPT_TEMPLATE,
                "vqa": VQA_DIRECT_THINK_PROMPT_TEMPLATE,
            }[self.answer_mode]
            prompt = template.format(instruction=normalized_instruction)
        image_url = (
            {
                "url": encode_image(
                    image_path,
                    jpeg_quality=self.api_image_quality,
                )
            }
            if encode
            else {"path": image_path}
        )
        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": image_url},
                {"type": "text", "text": prompt},
            ],
        }

    def crop_user_message(
        self,
        result: dict[str, Any],
        *,
        encode: bool,
    ) -> dict[str, Any]:
        return self.crop_results_user_message(
            [{"tool_call_index": 1, "result": result}],
            encode=encode,
        )

    def crop_results_user_message(
        self,
        events: list[dict[str, Any]],
        *,
        encode: bool,
    ) -> dict[str, Any]:
        """Return one user message containing every result and successful crop image."""
        if not events:
            raise ValueError("events must contain at least one crop result")

        content: list[dict[str, Any]] = []
        event_count = len(events)
        for position, event in enumerate(events, start=1):
            result = dict(event.get("result") or {})
            call_index = event.get("tool_call_index", position)
            if event_count == 1:
                heading = "crop_image action result:"
            else:
                heading = (
                    f"crop_image action result {position}/{event_count} "
                    f"(tool_call_index={call_index}):"
                )
            result_text = _json_dumps_two_decimal_floats(result)
            result_lines = [heading, result_text]
            image_path = result.get("image_path") if result.get("ok") else None
            if image_path:
                image_path = str(image_path)
                result_lines.append(f"image_path: {image_path}")
            content.append({"type": "text", "text": "\n".join(result_lines)})
            if image_path:
                image_url = (
                    {
                        "url": encode_image(
                            image_path,
                            jpeg_quality=self.api_image_quality,
                        )
                    }
                    if encode
                    else {"path": image_path}
                )
                content.append({"type": "image_url", "image_url": image_url})

        final_instruction = {
            "point": "the final [x.xx,y.yy]",
            "mcq": "the final <answer>LETTER</answer> with LETTER set to A, B, C, or D",
            "vqa": "the final <answer>SHORT ANSWER</answer>",
        }[self.answer_mode]
        content.append(
            {
                "type": "text",
                "text": (
                    "Inspect all successful crops and account for any failed actions. "
                    "Reply with <think>...</think>, then either one or two independent "
                    "crop_image <action> blocks whose coordinates have exactly two "
                    f"decimal places, or {final_instruction}. Do not mix crop actions "
                    "with a final answer."
                ),
            }
        )
        return {"role": "user", "content": content}

    @staticmethod
    def text_user_message(text: str) -> dict[str, Any]:
        return {"role": "user", "content": text}

    def _request_once(
        self,
        messages: list[dict[str, Any]],
        *,
        n: int = 1,
        temperature: float | None = None,
        template_overrides: dict[str, Any] | None = None,
        max_tokens_override: int | None = None,
    ) -> list[dict[str, Any]]:
        if n < 1:
            raise ValueError("n must be positive")
        request_temperature = self.temperature if temperature is None else temperature
        if not math.isfinite(request_temperature) or request_temperature < 0:
            raise ValueError("temperature must be a finite nonnegative number")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "n": n,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": max_tokens_override or self.max_completion_tokens,
            "temperature": request_temperature,
        }
        if self.prompt_mode == "direct_nothink":
            payload["chat_template_kwargs"] = {
                "thinking": False,
                "enable_thinking": False,
            }
        else:
            # Send reasoning_effort through chat_template_kwargs; top-level vLLM accepts
            # only its standard values and may reject template-specific values such as xhigh.
            payload["chat_template_kwargs"] = {
                "thinking": True,
                "enable_thinking": True,
                "reasoning_effort": self.reasoning_effort,
                "max_thinking_tokens": self.thinking_budget,
            }
        if template_overrides is not None:
            payload["chat_template_kwargs"] = template_overrides
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        choice_states: dict[int, dict[str, Any]] = {}
        event_count = 0
        metadata: dict[str, Any] = {}

        with self.client.stream(
            "POST", self.api_url, headers=headers, json=payload
        ) as response:
            if response.status_code >= 400:
                body = response.read().decode("utf-8", errors="replace")
                raise httpx.HTTPStatusError(
                    f"HTTP {response.status_code}: {body[:2000]}",
                    request=response.request,
                    response=response,
                )
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                event_count += 1
                if event.get("error") is not None:
                    error_payload = event["error"]
                    if isinstance(error_payload, dict):
                        error_message = str(
                            error_payload.get("message")
                            or error_payload.get("detail")
                            or json.dumps(error_payload, ensure_ascii=False)
                        )
                    else:
                        error_message = str(error_payload)
                    raise RuntimeError(f"Streaming API error: {error_message}")
                for key in (
                    "id",
                    "object",
                    "created",
                    "model",
                    "system_fingerprint",
                    "service_tier",
                    "usage",
                ):
                    if event.get(key) is not None:
                        metadata[key] = event[key]
                for choice in event.get("choices") or []:
                    choice_index = int(choice.get("index", 0))
                    state = choice_states.setdefault(
                        choice_index,
                        {
                            "reasoning_parts": [],
                            "content_parts": [],
                            "finish_reason": None,
                            "delta_fields": set(),
                        },
                    )
                    delta = choice.get("delta") or choice.get("message") or {}
                    state["delta_fields"].update(delta)
                    state["reasoning_parts"].append(
                        text_from_delta(
                            delta.get(
                                "reasoning_content",
                                delta.get("reasoning", delta.get("thinking")),
                            )
                        )
                    )
                    state["content_parts"].append(
                        text_from_delta(delta.get("content"))
                    )
                    if choice.get("finish_reason") is not None:
                        state["finish_reason"] = choice["finish_reason"]

        missing_choices = sorted(set(range(n)) - choice_states.keys())
        if missing_choices:
            raise RuntimeError(
                "Streaming API returned no completion events for choice "
                f"index(es): {missing_choices}"
            )

        responses: list[dict[str, Any]] = []
        for choice_index in range(n):
            state = choice_states.get(
                choice_index,
                {
                    "reasoning_parts": [],
                    "content_parts": [],
                    "finish_reason": None,
                    "delta_fields": set(),
                },
            )
            responses.append(
                {
                    "choice_index": choice_index,
                    "reasoning_content": "".join(state["reasoning_parts"]),
                    "content": "".join(state["content_parts"]),
                    "finish_reason": state["finish_reason"],
                    "event_count": event_count,
                    "stream_delta_fields": sorted(state["delta_fields"]),
                    "response_metadata": metadata,
                }
            )
        return responses

    def _request_with_retries(
        self,
        messages: list[dict[str, Any]],
        *,
        n: int = 1,
        temperature: float | None = None,
        template_overrides: dict[str, Any] | None = None,
        max_tokens_override: int | None = None,
    ) -> tuple[list[dict[str, Any]] | None, str | None, int, float]:
        started = time.perf_counter()
        response_data = None
        error = None
        attempts = 0
        rate_limit_retries = 0
        for attempts in range(1, self.max_attempts + 1):
            try:
                self.limiter.acquire()
                response_data = self._request_once(
                    messages,
                    n=n,
                    temperature=temperature,
                    template_overrides=template_overrides,
                    max_tokens_override=max_tokens_override,
                )
                error = None
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempts < self.max_attempts:
                    if (
                        isinstance(exc, httpx.HTTPStatusError)
                        and exc.response.status_code == 429
                    ):
                        delay = RATE_LIMIT_BACKOFF_SECONDS[
                            min(
                                rate_limit_retries,
                                len(RATE_LIMIT_BACKOFF_SECONDS) - 1,
                            )
                        ]
                        rate_limit_retries += 1
                    else:
                        delay = min(2**attempts, 20)
                    time.sleep(delay)
        return response_data, error, attempts, round(time.perf_counter() - started, 3)

    def _execute_crop(
        self,
        raw_arguments: Any,
        registry: dict[str, dict[str, Any]],
        crop_dir: Path,
        crop_index: int,
    ) -> dict[str, Any]:
        try:
            arguments = _parse_arguments(raw_arguments)
            source_path = _registered_path(str(arguments["image_path"]))
            if source_path not in registry:
                raise ValueError(
                    "image_path is not the original image or a path returned by crop_image"
                )
            source = registry[source_path]
            values = [float(arguments[key]) for key in ("x1", "y1", "x2", "y2")]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Crop coordinates must be finite")
            x1, y1, x2, y2 = values
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Expected x1 < x2 and y1 < y2")
            left = max(0, min(source["width"], math.floor(x1)))
            top = max(0, min(source["height"], math.floor(y1)))
            right = max(0, min(source["width"], math.ceil(x2)))
            bottom = max(0, min(source["height"], math.ceil(y2)))
            if right - left < 2 or bottom - top < 2:
                raise ValueError("Clamped crop must be at least 2x2 pixels")

            crop_dir.mkdir(parents=True, exist_ok=True)
            output_path = crop_dir / f"crop_{crop_index:02d}.png"
            with Image.open(source_path) as image:
                cropped = image.crop((left, top, right, bottom))
                crop_width, crop_height = cropped.size
                resize_scale = min(
                    self.crop_upscale_factor,
                    math.sqrt(
                        self.crop_max_pixels / (crop_width * crop_height)
                    ),
                    self.crop_max_dimension / crop_width,
                    self.crop_max_dimension / crop_height,
                )
                output_width = max(
                    1,
                    min(
                        self.crop_max_dimension,
                        math.floor(crop_width * resize_scale),
                    ),
                )
                output_height = max(
                    1,
                    min(
                        self.crop_max_dimension,
                        math.floor(crop_height * resize_scale),
                    ),
                )
                if output_width * output_height > self.crop_max_pixels:
                    if output_width >= output_height:
                        output_width = max(
                            1,
                            self.crop_max_pixels // output_height,
                        )
                    else:
                        output_height = max(
                            1,
                            self.crop_max_pixels // output_width,
                        )
                if (output_width, output_height) != cropped.size:
                    cropped = cropped.resize(
                        (output_width, output_height),
                        resample=Image.Resampling.LANCZOS,
                    )
                cropped.save(output_path, format="PNG")

            original = source["original_region"]
            parent_scale_x = (original[2] - original[0]) / source["width"]
            parent_scale_y = (original[3] - original[1]) / source["height"]
            original_region = [
                original[0] + left * parent_scale_x,
                original[1] + top * parent_scale_y,
                original[0] + right * parent_scale_x,
                original[1] + bottom * parent_scale_y,
            ]
            output_scale_x = (original_region[2] - original_region[0]) / output_width
            output_scale_y = (original_region[3] - original_region[1]) / output_height
            output_key = _registered_path(str(output_path))
            registry[output_key] = {
                "path": output_key,
                "width": output_width,
                "height": output_height,
                "original_region": original_region,
                "original_width": source["original_width"],
                "original_height": source["original_height"],
            }
            normalized_region = [
                original_region[0] * 1000 / source["original_width"],
                original_region[1] * 1000 / source["original_height"],
                original_region[2] * 1000 / source["original_width"],
                original_region[3] * 1000 / source["original_height"],
            ]
            rounded_original_region = [round(value, 2) for value in original_region]
            rounded_normalized_region = [round(value, 2) for value in normalized_region]
            return {
                "ok": True,
                "image_path": output_key,
                "width": output_width,
                "height": output_height,
                "crop_in_parent": [left, top, right, bottom],
                "crop_size_before_resize": [crop_width, crop_height],
                "resize_scale": round(resize_scale, 2),
                "original_image_region_pixels": rounded_original_region,
                "original_image_region_0_1000": rounded_normalized_region,
                "mapping_to_original_0_1000": {
                    "x": (
                        f"({original_region[0]:.2f} + crop_x * "
                        f"{output_scale_x:.2f}) * 1000 / "
                        f"{source['original_width']}"
                    ),
                    "y": (
                        f"({original_region[1]:.2f} + crop_y * "
                        f"{output_scale_y:.2f}) * 1000 / "
                        f"{source['original_height']}"
                    ),
                },
                "reminder": (
                    "Use the inspected evidence to return exactly one option letter."
                    if self.answer_mode == "mcq"
                    else (
                        "Use the inspected evidence to return only the concise answer."
                        if self.answer_mode == "vqa"
                        else "Final coordinates must be normalized [0,1000] in the original image."
                    )
                ),
            }
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _assistant_visible_content(response: dict[str, Any]) -> str:
        """Merge provider-separated reasoning into dataset-style visible content."""
        content = str(response.get("content") or "")
        reasoning = str(response.get("reasoning_content") or "").strip()
        paired_think = re.search(
            r"<think\b[^>]*>.*?</think\s*>",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        has_open_think = re.search(r"<think\b[^>]*>", content, re.IGNORECASE)
        has_close_think = re.search(r"</think\s*>", content, re.IGNORECASE)

        # Some chat templates prefill ``<think>`` in the generation prompt. The
        # completion then starts with the reasoning body and contains only the closing
        # tag. Reconstruct the complete assistant answer before saving it or feeding it
        # into the next agentic turn. Compare the first tag positions as a completion
        # may also contain a later empty ``<think></think>`` block.
        if has_close_think and (
            not has_open_think or has_close_think.start() < has_open_think.start()
        ):
            return f"<think>\n{content.lstrip()}"
        if paired_think or not reasoning:
            return content
        suffix = f"\n{content.lstrip()}" if content else ""
        return f"<think>\n{reasoning}\n</think>{suffix}"

    @classmethod
    def _assistant_reasoning_content(cls, response: dict[str, Any]) -> str:
        """Return explicit or reconstructed reasoning without think tags."""
        explicit = str(response.get("reasoning_content") or "").strip()
        if explicit:
            return explicit
        visible = cls._assistant_visible_content(response)
        match = re.search(
            r"<think\b[^>]*>(.*?)</think\s*>",
            visible,
            re.DOTALL | re.IGNORECASE,
        )
        return match.group(1).strip() if match else ""

    @classmethod
    def _assistant_api_message(cls, response: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": cls._assistant_visible_content(response),
        }

    @staticmethod
    def _truncated_assistant_api_message(message: dict[str, Any]) -> dict[str, Any]:
        content = message.get("content")
        if (
            isinstance(content, str)
            and len(content) > TRUNCATED_ASSISTANT_CONTENT_CHARS
        ):
            return {
                **message,
                "content": (
                    content[:TRUNCATED_ASSISTANT_CONTENT_CHARS] + " ...[truncated]"
                ),
            }
        return message

    def _run_trajectory(
        self,
        instruction: str,
        image_path: str,
        *,
        task_id: str | None = None,
        sample_index: int = 0,
        initial_sample_count: int = 1,
        sample_temperature: float | None = None,
        first_turn: tuple[
            dict[str, Any] | None,
            str | None,
            int,
            float,
        ]
        | None = None,
    ) -> dict[str, Any]:
        original_path = _registered_path(image_path)
        if not Path(original_path).is_file():
            raise FileNotFoundError(original_path)
        width, height = image_size(original_path)
        task_id = task_id or Path(original_path).stem
        conversation_id = secrets.token_hex(8)
        crop_dir = (
            self.crop_root
            / _safe_task_component(task_id)
            / f"conversation_{conversation_id}"
        )
        system_prompt = self.build_system_prompt(
            instruction, original_path, width, height
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            self.user_image_message(original_path, encode=True),
        ]
        trajectory_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            self.user_image_message(original_path, encode=False),
        ]
        registry = {
            original_path: {
                "path": original_path,
                "width": width,
                "height": height,
                "original_region": [0.0, 0.0, float(width), float(height)],
                "original_width": width,
                "original_height": height,
            }
        }
        rounds: list[dict[str, Any]] = []
        tool_events: list[dict[str, Any]] = []
        tool_count = 0
        tool_attempt_count = 0
        action_correction_count = 0
        current_image_path = original_path
        final_content = ""
        final_point = None
        final_option = None
        final_answer = None
        conversation_error = None

        try:
            max_model_turns = (
                self.max_tool_calls + self.max_action_corrections + 1
            )
            next_template_overrides: dict[str, Any] | None = None
            next_max_tokens_override: int | None = None
            for model_turn in range(1, max_model_turns + 1):
                if model_turn == 1 and first_turn is not None:
                    response, error, attempts, elapsed = first_turn
                else:
                    response_batch, error, attempts, elapsed = (
                        self._request_with_retries(
                            messages,
                            temperature=sample_temperature,
                            template_overrides=next_template_overrides,
                            max_tokens_override=next_max_tokens_override,
                        )
                    )
                    response = response_batch[0] if response_batch else None
                next_template_overrides = None
                next_max_tokens_override = None
                metadata = (response or {}).get("response_metadata") or {}
                usage = metadata.get("usage") or {}
                reasoning_tokens = (
                    usage.get("completion_tokens_details", {}).get("reasoning_tokens")
                    if isinstance(usage, dict)
                    else None
                )
                content = (response or {}).get("content") or ""
                visible_content = self._assistant_visible_content(response or {})
                normalized_reasoning = self._assistant_reasoning_content(response or {})
                degenerate = (
                    _detect_degenerate(content)
                    or _detect_degenerate(visible_content)
                )
                parsed_action = parse_text_action(
                    visible_content or content,
                    answer_mode=self.answer_mode,
                )
                if parsed_action is None:
                    parsed_actions = None
                elif parsed_action["type"] == "crop_batch":
                    parsed_actions = list(parsed_action["actions"])
                else:
                    parsed_actions = [parsed_action]
                round_record = {
                    "model_turn": model_turn,
                    "sample_index": sample_index,
                    "initial_request_n": initial_sample_count,
                    "native_tools_enabled": False,
                    "attempts": attempts,
                    "scheduler_elapsed_seconds": elapsed,
                    "error": error,
                    "response": response,
                    "parsed_action": parsed_action,
                    "parsed_actions": parsed_actions,
                    "reasoning_stats": {
                        "characters": len(
                            normalized_reasoning
                        ),
                        "reported_reasoning_tokens": reasoning_tokens,
                        "completion_tokens": (
                            usage.get("completion_tokens")
                            if isinstance(usage, dict)
                            else None
                        ),
                    },
                }
                if degenerate:
                    round_record["degenerate"] = degenerate
                rounds.append(round_record)
                if error is not None or response is None:
                    conversation_error = error
                    break

                trajectory_assistant = {
                    "role": "assistant",
                    "content": visible_content or None,
                    "reasoning_content": normalized_reasoning,
                    "finish_reason": response.get("finish_reason"),
                    "parsed_action": parsed_action,
                    "parsed_actions": parsed_actions,
                }
                trajectory_messages.append(trajectory_assistant)

                if parsed_actions is None:
                    final_content = visible_content
                    if action_correction_count >= self.max_action_corrections:
                        expected_final = {
                            "point": "coordinate",
                            "mcq": "multiple-choice answer",
                            "vqa": "short-text answer",
                        }[self.answer_mode]
                        conversation_error = (
                            "Model did not return a parseable crop action or final "
                            f"{expected_final} after "
                            f"{action_correction_count + 1} attempts"
                        )
                        break
                    action_correction_count += 1
                    truncated = (
                        (response or {}).get("finish_reason") == "length"
                        and self.truncate_length_capped_turns
                    )
                    if degenerate:
                        # Replay would retrigger the loop; discard it and force a
                        # short no-thinking retry on the next turn instead.
                        messages.append(
                            {
                                "role": "assistant",
                                "content": "(repeated degenerate output, discarded)",
                            }
                        )
                        next_template_overrides = {
                            "thinking": False,
                            "enable_thinking": False,
                        }
                        next_max_tokens_override = DEGENERATE_RESCUE_MAX_TOKENS
                    else:
                        api_assistant_message = self._assistant_api_message(response)
                        if truncated:
                            # Length-capped turns are typically degenerate repetition; a
                            # full replay would bloat the context and invite the same loop.
                            api_assistant_message = self._truncated_assistant_api_message(
                                api_assistant_message
                            )
                        messages.append(api_assistant_message)
                    final_format = {
                        "point": "[x.xx,y.yy] coordinate",
                        "mcq": (
                            "<answer>LETTER</answer> with LETTER set to A, B, C, or D"
                        ),
                        "vqa": "<answer>SHORT ANSWER</answer>",
                    }[self.answer_mode]
                    if degenerate:
                        correction = self.text_user_message(
                            "Your previous reply degenerated into a repeated token "
                            "loop and was discarded. Do not reason, plan, or request "
                            "crops now. Reply immediately with exactly "
                            f"{final_format} and nothing else."
                        )
                    elif truncated:
                        correction = self.text_user_message(
                            "Your previous reply hit the token limit and was cut off "
                            "before producing any crop action or final answer. Do not "
                            "reason, plan, or request crops now. Reply immediately with "
                            f"exactly {final_format} and nothing else."
                        )
                    else:
                        correction = self.text_user_message(
                            "Your previous answer was not valid. Reply with either one "
                            "or two independent crop actions, each in its own "
                            "<action>crop_image(...)</action> block, or one final "
                            f"{final_format}. Do not emit more than two actions or mix "
                            "actions with a final answer. Every crop coordinate value "
                            "must have exactly two digits after the decimal point."
                        )
                    messages.append(correction)
                    trajectory_messages.append(correction)
                    continue

                if parsed_actions[0]["type"] == "final":
                    final_content = visible_content
                    if self.answer_mode == "mcq":
                        final_option = parsed_actions[0]["option"]
                    elif self.answer_mode == "vqa":
                        final_answer = parsed_actions[0]["answer"]
                    else:
                        final_point = parsed_actions[0]["point"]
                    break

                messages.append(self._assistant_api_message(response))
                batch_source_image_path = current_image_path
                batch_events: list[dict[str, Any]] = []
                for crop_action in parsed_actions:
                    tool_attempt_count += 1
                    arguments = dict(crop_action.get("arguments") or {})
                    arguments.setdefault("image_path", batch_source_image_path)
                    budget_index = None
                    if tool_count >= self.max_tool_calls:
                        exhausted_final = {
                            "point": (
                                "Return the final "
                                "[x.xx,y.yy] now, using exactly two decimal places."
                            ),
                            "mcq": (
                                "Return the final "
                                "<answer>LETTER</answer> now, with LETTER set to A, B, C, or D."
                            ),
                            "vqa": (
                                "Return the final "
                                "<answer>SHORT ANSWER</answer> now."
                            ),
                        }[self.answer_mode]
                        result = {
                            "ok": False,
                            "error": (
                                "Crop budget exhausted: maximum is "
                                f"{self.max_tool_calls}. {exhausted_final}"
                            ),
                        }
                    else:
                        tool_count += 1
                        budget_index = tool_count
                        result = self._execute_crop(
                            arguments,
                            registry,
                            crop_dir,
                            tool_count,
                        )
                    event = {
                        "tool_call_index": tool_attempt_count,
                        "tool_budget_index": budget_index,
                        "model_turn": model_turn,
                        "batch_index": len(batch_events),
                        "protocol": "text_action",
                        "name": "crop_image",
                        "raw_action": crop_action.get("source"),
                        "arguments": arguments,
                        "result": result,
                    }
                    tool_events.append(event)
                    batch_events.append(event)
                    if result.get("ok"):
                        current_image_path = str(result["image_path"])

                api_user_message = self.crop_results_user_message(
                    batch_events,
                    encode=True,
                )
                trajectory_user_message = self.crop_results_user_message(
                    batch_events,
                    encode=False,
                )
                messages.append(api_user_message)
                trajectory_messages.append(trajectory_user_message)
            else:
                conversation_error = "Conversation exhausted the model-turn limit"
        finally:
            if not self.persist_crops and crop_dir.exists():
                import shutil

                shutil.rmtree(crop_dir, ignore_errors=True)

        reasoning_tokens = [
            item["reasoning_stats"].get("reported_reasoning_tokens")
            for item in rounds
        ]
        numeric_tokens = [
            value for value in reasoning_tokens if isinstance(value, (int, float))
        ]
        crop_paths = [
            event["result"]["image_path"]
            for event in tool_events
            if event["result"].get("ok")
        ]
        return {
            "task_id": task_id,
            "sample_index": sample_index,
            "conversation_id": conversation_id,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "instruction": instruction,
            "original_image_path": original_path,
            "original_image_width": width,
            "original_image_height": height,
            "answer_mode": self.answer_mode,
            "coordinate_system": (
                "original_image_relative_0_1000"
                if self.answer_mode == "point"
                else None
            ),
            "system_prompt": system_prompt,
            "initial_user_message": self.user_image_message(
                original_path, encode=False
            ),
            "tool_schema": None,
            "text_action_protocol": {
                "crop": (
                    '<think>...</think>\\n<action>crop_image(image_path="...", '
                    "x1=100.00, y1=200.00, x2=500.00, y2=600.00)</action>"
                ),
                "crop_batch": (
                    '<think>...</think>\\n<action>crop_image(image_path="...", '
                    "x1=100.00, y1=200.00, x2=500.00, y2=600.00)</action>\\n"
                    '<action>crop_image(image_path="...", x1=600.00, y1=200.00, '
                    "x2=900.00, y2=600.00)</action>"
                ),
                "final": (
                    {
                        "point": "<think>...</think>\\n[x.xx,y.yy]",
                        "mcq": "<think>...</think>\\n<answer>LETTER</answer>",
                        "vqa": "<think>...</think>\\n<answer>SHORT ANSWER</answer>",
                    }[self.answer_mode]
                ),
            },
            "request_config": {
                "api_url": self.api_url,
                "stream": True,
                "initial_request_n": initial_sample_count,
                "temperature": (
                    self.temperature
                    if sample_temperature is None
                    else sample_temperature
                ),
                "max_tokens_per_turn": self.max_completion_tokens,
                "reasoning_effort": self.reasoning_effort,
                "chat_template_kwargs": {
                    "thinking": True,
                    "reasoning_effort": self.reasoning_effort,
                    "max_thinking_tokens": self.thinking_budget,
                },
                "max_tool_calls": self.max_tool_calls,
                "max_tool_calls_per_turn": MAX_CROP_IMAGE_CALLS_PER_TURN,
                "max_action_corrections": self.max_action_corrections,
                "answer_mode": self.answer_mode,
                "api_image_format": "jpeg",
                "api_image_quality": self.api_image_quality,
                "crop_upscale_factor": self.crop_upscale_factor,
                "crop_max_pixels": self.crop_max_pixels,
                "crop_max_dimension": self.crop_max_dimension,
                "native_api_tools": False,
                "action_protocol": "assistant_text_then_user_crop_image",
                "proxy_mode": "direct (httpx trust_env=False)",
                "persist_crops": self.persist_crops,
                "crop_root": str(self.crop_root),
            },
            "messages": trajectory_messages,
            "rounds": rounds,
            "action_events": tool_events,
            "tool_events": tool_events,
            "model_turn_count": len(rounds),
            "crop_action_count": tool_count,
            "tool_call_count": tool_count,
            "tool_attempt_count": tool_attempt_count,
            "crop_image_paths": crop_paths,
            "crop_directory": str(crop_dir),
            "final_content": final_content,
            "prediction_relative_0_1000": final_point,
            "prediction_option": final_option,
            "prediction_text": final_answer,
            "reasoning_stats": {
                "per_turn_reported_reasoning_tokens": reasoning_tokens,
                "total_reported_reasoning_tokens": sum(numeric_tokens),
                "max_reported_reasoning_tokens_in_one_turn": (
                    max(numeric_tokens) if numeric_tokens else None
                ),
            },
            "error": conversation_error,
        }

    def _run_direct_many(
        self,
        instruction: str,
        image_path: str,
        *,
        sample_count: int,
        task_id: str | None = None,
        temperature: float | None = None,
    ) -> list[dict[str, Any]]:
        """Run one-shot inference with no crop or native tool interface."""
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        if self.prompt_mode not in {"direct_nothink", "direct_think"}:
            raise ValueError("_run_direct_many requires a direct prompt mode")
        thinking_enabled = self.prompt_mode == "direct_think"
        original_path = _registered_path(image_path)
        if not Path(original_path).is_file():
            raise FileNotFoundError(original_path)
        width, height = image_size(original_path)
        resolved_task_id = task_id or Path(original_path).stem
        message_builder = (
            self.direct_think_user_message
            if thinking_enabled
            else self.direct_nothink_user_message
        )
        api_message = message_builder(
            instruction,
            original_path,
            encode=True,
        )
        trajectory_message = message_builder(
            instruction,
            original_path,
            encode=False,
        )
        api_messages: list[dict[str, Any]] = []
        trajectory_messages: list[dict[str, Any]] = []
        if self.direct_system_prompt is not None:
            api_messages.append(
                {"role": "system", "content": self.direct_system_prompt}
            )
            trajectory_messages.append(
                {"role": "system", "content": self.direct_system_prompt}
            )
        api_messages.append(api_message)
        trajectory_messages.append(trajectory_message)
        responses, error, attempts, elapsed = self._request_with_retries(
            api_messages,
            n=sample_count,
            temperature=temperature,
        )
        choices = responses or [None] * sample_count
        if len(choices) != sample_count:
            raise RuntimeError(
                "vLLM returned an unexpected number of choices: "
                f"expected {sample_count}, got {len(choices)}"
            )

        trajectories: list[dict[str, Any]] = []
        for sample_index, response in enumerate(choices):
            visible_content = self._assistant_visible_content(response or {})
            normalized_reasoning = self._assistant_reasoning_content(response or {})
            parsed_final = parse_text_action(
                visible_content,
                answer_mode=self.answer_mode,
            )
            if parsed_final is not None and parsed_final.get("type") != "final":
                parsed_final = None
            point = (
                parsed_final.get("point")
                if parsed_final is not None and self.answer_mode == "point"
                else None
            )
            option = (
                parsed_final.get("option")
                if parsed_final is not None and self.answer_mode == "mcq"
                else None
            )
            answer = (
                parsed_final.get("answer")
                if parsed_final is not None and self.answer_mode == "vqa"
                else None
            )
            trajectory_error = error
            if trajectory_error is None and parsed_final is None:
                expected_final = {
                    "point": "[x,y] coordinate",
                    "mcq": "<answer>LETTER</answer>",
                    "vqa": "<answer>SHORT ANSWER</answer>",
                }[self.answer_mode]
                trajectory_error = f"Model did not return a parseable final {expected_final}"
            metadata = (response or {}).get("response_metadata") or {}
            usage = metadata.get("usage") or {}
            reasoning_tokens = (
                usage.get("completion_tokens_details", {}).get("reasoning_tokens")
                if isinstance(usage, dict)
                else None
            )
            round_record = {
                "model_turn": 1,
                "sample_index": sample_index,
                "initial_request_n": sample_count,
                "native_tools_enabled": False,
                "attempts": attempts,
                "scheduler_elapsed_seconds": elapsed,
                "error": error,
                "response": response,
                "parsed_action": parsed_final,
                "parsed_actions": [parsed_final] if parsed_final is not None else None,
                "reasoning_stats": {
                    "characters": len(normalized_reasoning),
                    "reported_reasoning_tokens": reasoning_tokens,
                    "completion_tokens": (
                        usage.get("completion_tokens")
                        if isinstance(usage, dict)
                        else None
                    ),
                },
            }
            trajectories.append(
                {
                    "task_id": resolved_task_id,
                    "sample_index": sample_index,
                    "conversation_id": secrets.token_hex(8),
                    "model": self.model,
                    "reasoning_effort": (
                        self.reasoning_effort if thinking_enabled else None
                    ),
                    "instruction": instruction,
                    "original_image_path": original_path,
                    "original_image_width": width,
                    "original_image_height": height,
                    "answer_mode": self.answer_mode,
                    "coordinate_system": (
                        "original_image_relative_0_1000"
                        if self.answer_mode == "point"
                        else None
                    ),
                    "system_prompt": self.direct_system_prompt,
                    "initial_user_message": trajectory_message,
                    "tool_schema": None,
                    "text_action_protocol": None,
                    "request_config": {
                        "api_url": self.api_url,
                        "stream": True,
                        "initial_request_n": sample_count,
                        "temperature": (
                            self.temperature
                            if temperature is None
                            else temperature
                        ),
                        "max_tokens_per_turn": self.max_completion_tokens,
                        "reasoning_effort": (
                            self.reasoning_effort if thinking_enabled else None
                        ),
                        "chat_template_kwargs": (
                            {
                                "thinking": True,
                                **(
                                    {"enable_thinking": True}
                                    if self.direct_prompt_variant != "sspro"
                                    else {}
                                ),
                                "reasoning_effort": self.reasoning_effort,
                                "max_thinking_tokens": self.thinking_budget,
                            }
                            if thinking_enabled
                            else {
                                "thinking": False,
                                "enable_thinking": False,
                            }
                        ),
                        "max_tool_calls": 0,
                        "answer_mode": self.answer_mode,
                        "native_api_tools": False,
                        "action_protocol": self.prompt_mode,
                        "direct_prompt_passthrough": self.direct_prompt_passthrough,
                        "proxy_mode": "direct (httpx trust_env=False)",
                    },
                    "messages": trajectory_messages
                    + [
                        {
                            "role": "assistant",
                            "content": visible_content or None,
                            "finish_reason": (
                                (response or {}).get("finish_reason")
                            ),
                        },
                    ],
                    "rounds": [round_record],
                    "action_events": [],
                    "tool_events": [],
                    "model_turn_count": 1,
                    "crop_action_count": 0,
                    "tool_call_count": 0,
                    "tool_attempt_count": 0,
                    "crop_image_paths": [],
                    "crop_directory": None,
                    "final_content": visible_content,
                    "prediction_relative_0_1000": point,
                    "prediction_option": option,
                    "prediction_text": answer,
                    "reasoning_stats": {
                        "per_turn_reported_reasoning_tokens": [reasoning_tokens],
                        "total_reported_reasoning_tokens": (
                            reasoning_tokens
                            if isinstance(reasoning_tokens, (int, float))
                            else 0
                        ),
                        "max_reported_reasoning_tokens_in_one_turn": (
                            reasoning_tokens
                            if isinstance(reasoning_tokens, (int, float))
                            else None
                        ),
                    },
                    "error": trajectory_error,
                }
            )
        return trajectories

    def run(
        self,
        instruction: str,
        image_path: str,
        *,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        if self.prompt_mode in {"direct_nothink", "direct_think"}:
            return self._run_direct_many(
                instruction,
                image_path,
                sample_count=1,
                task_id=task_id,
            )[0]
        return self._run_trajectory(
            instruction,
            image_path,
            task_id=task_id,
        )

    def run_many(
        self,
        instruction: str,
        image_path: str,
        *,
        sample_count: int,
        task_id: str | None = None,
        temperature: float | None = None,
    ) -> list[dict[str, Any]]:
        """Sample the first agent turn once with ``n`` and continue all branches."""
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        if self.prompt_mode in {"direct_nothink", "direct_think"}:
            return self._run_direct_many(
                instruction,
                image_path,
                sample_count=sample_count,
                task_id=task_id,
                temperature=temperature,
            )
        if sample_count == 1:
            return [
                self._run_trajectory(
                    instruction,
                    image_path,
                    task_id=task_id,
                    sample_temperature=temperature,
                )
            ]

        original_path = _registered_path(image_path)
        if not Path(original_path).is_file():
            raise FileNotFoundError(original_path)
        width, height = image_size(original_path)
        resolved_task_id = task_id or Path(original_path).stem
        system_prompt = self.build_system_prompt(
            instruction,
            original_path,
            width,
            height,
        )
        initial_messages = [
            {"role": "system", "content": system_prompt},
            self.user_image_message(original_path, encode=True),
        ]
        responses, error, attempts, elapsed = self._request_with_retries(
            initial_messages,
            n=sample_count,
            temperature=temperature,
        )
        first_responses = responses or [None] * sample_count
        if len(first_responses) != sample_count:
            raise RuntimeError(
                "vLLM returned an unexpected number of choices: "
                f"expected {sample_count}, got {len(first_responses)}"
            )

        def continue_branch(
            indexed_response: tuple[int, dict[str, Any] | None],
        ) -> dict[str, Any]:
            sample_index, response = indexed_response
            return self._run_trajectory(
                instruction,
                original_path,
                task_id=resolved_task_id,
                sample_index=sample_index,
                initial_sample_count=sample_count,
                sample_temperature=temperature,
                first_turn=(response, error, attempts, elapsed),
            )

        return list(
            self._branch_executor.map(
                continue_branch,
                enumerate(first_responses),
            )
        )

    def inference(
        self, instruction: str, image_path: str, *, task_id: str | None = None
    ) -> dict[str, Any]:
        """Agentic inference alias returning the complete trajectory."""
        return self.run(instruction, image_path, task_id=task_id)

    def inference_many(
        self,
        instruction: str,
        image_path: str,
        *,
        sample_count: int,
        task_id: str | None = None,
        temperature: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return independent agentic trajectories from one batched first request."""
        return self.run_many(
            instruction,
            image_path,
            sample_count=sample_count,
            task_id=task_id,
            temperature=temperature,
        )


__all__ = [
    "AgenticGroundingPipeline",
    "SlidingWindowRateLimiter",
    "CROP_IMAGE_TOOL",
    "CROP_IMAGE_DESCRIPTION",
    "MCQ_CROP_IMAGE_DESCRIPTION",
    "VQA_CROP_IMAGE_DESCRIPTION",
    "DEFAULT_CROP_MAX_PIXELS",
    "DEFAULT_CROP_MAX_DIMENSION",
    "MAX_CROP_IMAGE_CALLS_PER_TURN",
    "SYSTEM_PROMPT_TEMPLATE",
    "DIRECT_GROUNDING_PROMPT_TEMPLATE",
    "DIRECT_NOTHINK_PROMPT_TEMPLATE",
    "DIRECT_THINK_PROMPT_TEMPLATE",
    "SSPRO_DIRECT_NOTHINK_PROMPT_TEMPLATE",
    "SSPRO_DIRECT_THINK_PROMPT_TEMPLATE",
    "MCQ_SYSTEM_PROMPT_TEMPLATE",
    "MCQ_DIRECT_NOTHINK_PROMPT_TEMPLATE",
    "MCQ_DIRECT_THINK_PROMPT_TEMPLATE",
    "VQA_SYSTEM_PROMPT_TEMPLATE",
    "VQA_DIRECT_NOTHINK_PROMPT_TEMPLATE",
    "VQA_DIRECT_THINK_PROMPT_TEMPLATE",
    "parse_point",
    "parse_mcq_answer",
    "parse_vqa_answer",
    "parse_text_action",
    "parse_text_actions",
]
