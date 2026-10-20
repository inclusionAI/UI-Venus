#!/usr/bin/env python3
"""Shared helpers for the agentic crop/zoom-in text-action protocol.

Constants, rate limiting, image encoding and text-action parsing extracted from
the pipeline client. Evaluation entries import these helpers directly; the
multi-turn pipeline client itself lives in each benchmark's
``models/agentic_pipeline.py``.
"""

from __future__ import annotations

import ast
import base64
import json
import math
import re
import threading
import time
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_CROP_UPSCALE_FACTOR = 4.0
DEFAULT_CROP_MAX_PIXELS = 4_000_000
DEFAULT_CROP_MAX_DIMENSION = 4096
MAX_CROP_IMAGE_CALLS = 3
MAX_CROP_IMAGE_CALLS_PER_TURN = 2
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
            arguments[key.strip()] = ast.literal_eval(value)
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
    elif answer_mode != "point":
        raise ValueError("answer_mode must be either 'point' or 'mcq'")

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
    if answer_mode not in {"point", "mcq"}:
        raise ValueError("answer_mode must be either 'point' or 'mcq'")
    value = text or ""
    action_matches = list(ACTION_BLOCK_RE.finditer(value))
    if action_matches:
        if len(action_matches) > MAX_CROP_IMAGE_CALLS_PER_TURN:
            return None
        has_final = (
            _final_coordinate_match(value)
            if answer_mode == "point"
            else parse_mcq_answer(value)
        )
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


__all__ = [
    "ACTION_BLOCK_RE",
    "DEFAULT_CROP_MAX_DIMENSION",
    "DEFAULT_CROP_MAX_PIXELS",
    "DEFAULT_CROP_UPSCALE_FACTOR",
    "MAX_CROP_IMAGE_CALLS",
    "MAX_CROP_IMAGE_CALLS_PER_TURN",
    "SlidingWindowRateLimiter",
    "encode_image",
    "image_size",
    "parse_mcq_answer",
    "parse_point",
    "parse_text_action",
    "parse_text_actions",
]
