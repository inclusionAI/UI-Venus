# Copyright 2026 verl recipe contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Zoom-In textual crop-action protocol.

The rollout sends no native tool schema: ``crop_image`` is emitted as
assistant text and executed locally.
"""

from __future__ import annotations

import ast
import base64
import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_CROP_UPSCALE_FACTOR = 4.0
DEFAULT_CROP_MAX_PIXELS = 4_000_000
DEFAULT_MAX_CROP_ASPECT_RATIO = 100.0
MAX_CROP_IMAGE_CALLS = 3
MAX_CROP_IMAGE_CALLS_PER_TURN = 2

COORDINATE_RE = re.compile(
    r"\[\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\]"
)
ACTION_BLOCK_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL | re.IGNORECASE)

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

INVALID_ACTION_CORRECTION = (
    "Your previous answer was not valid. Reply with <think>...</think> "
    "followed by either one or two independent crop actions, each in "
    "its own <action>crop_image(...)</action> block, or one final "
    "[x.xx,y.yy] coordinate. Do not emit more than two actions or mix "
    "actions with a final coordinate. Every coordinate value must "
    "have exactly two digits after the decimal point."
)


def registered_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def safe_task_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return (cleaned or "task")[:160]


def extract_instruction(problem: str) -> str:
    """Extract the task instruction from a grounding ``problem`` string."""
    value = str(problem or "").strip()
    value = re.sub(r"^\s*<image>\s*", "", value, flags=re.IGNORECASE)
    lowered = value.lower()
    start_marker = "following instruction:"
    end_marker = "the output should just be"
    start = lowered.find(start_marker)
    if start >= 0:
        start += len(start_marker)
        end = lowered.find(end_marker, start)
        value = value[start : end if end >= 0 else None]
    return value.strip()


def build_system_prompt(
    instruction: str,
    image_path: str,
    width: int,
    height: int,
    *,
    max_tool_calls: int,
) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        instruction=instruction.rstrip().rstrip("."),
        image_path=image_path,
        width=width,
        height=height,
        crop_description=CROP_IMAGE_DESCRIPTION,
        max_tool_calls=max_tool_calls,
    )


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
    """Prefer a coordinate after ``</think>``, otherwise return the last one."""
    after_think = _coordinate_after_think(text)
    if after_think:
        return after_think
    value = text or ""
    matches = list(COORDINATE_RE.finditer(value))
    return matches[-1] if matches else None


def _reject_json_constant(value: str) -> Any:
    """Reject ``NaN``/``Infinity``/``-Infinity`` tokens in model-emitted JSON.

    Python's ``json.loads`` accepts these non-standard constants by default.
    Non-finite floats would otherwise flow into rollout state and crash the
    rollout server's ``allow_nan=False`` response serialization.
    """
    raise ValueError(f"Non-finite JSON constant {value} is not allowed in actions")


def _loads_action_json(text: str) -> Any:
    """``json.loads`` for model output that rejects NaN/Infinity constants."""
    return json.loads(text, parse_constant=_reject_json_constant)


def _finite_point(x: Any, y: Any) -> list[float] | None:
    """Build a final point, rejecting overflow values such as ``1e999`` -> inf."""
    point = [float(x), float(y)]
    if not all(math.isfinite(value) for value in point):
        return None
    return point


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise ValueError("Tool arguments must be a JSON object")
    value = _loads_action_json(raw)
    if not isinstance(value, dict):
        raise ValueError("Tool arguments must decode to an object")
    return json_safe_value(value)


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
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise ValueError("Action arguments must be Python literals") from exc
    return json_safe_value(value)


def json_safe_value(value: Any) -> Any:
    """Normalize a parsed action value to the JSON types accepted by rollout RPC."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Action arguments must contain finite numbers")
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Action argument object keys must be strings")
        return {key: json_safe_value(item) for key, item in value.items()}
    # ast.literal_eval accepts Ellipsis, bytes, complex numbers, and sets, but
    # rollout_infos must be JSON-serializable. Treat these as malformed actions.
    raise ValueError(f"Unsupported action argument type: {type(value).__name__}")


def _parse_function_action(source: str) -> tuple[str, dict[str, Any]]:
    try:
        expression = ast.parse(source.strip(), mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Invalid action syntax: {source}") from exc
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise ValueError("Expected a function-style action")
    if any(keyword.arg is None for keyword in expression.keywords):
        raise ValueError("Action does not support **kwargs")
    arguments = {str(keyword.arg): _literal_value(keyword.value) for keyword in expression.keywords}
    positional = [_literal_value(item) for item in expression.args]
    name = expression.func.id
    if positional:
        keys = ("image_path", "x1", "y1", "x2", "y2") if name.lower() in {"crop", "crop_image"} else ("box",)
        if len(positional) > len(keys):
            raise ValueError("Too many positional action arguments")
        for key, value in zip(keys, positional):
            arguments.setdefault(key, value)
    return name, arguments


def _repair_malformed_crop_call(source: str) -> dict[str, Any] | None:
    """Repair the observed ``x1=100,336, x2=300,388`` shorthand."""
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
    if not all(math.isfinite(item) for item in values):
        return None
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
            arguments[key.strip()] = json_safe_value(ast.literal_eval(value))
        except (ValueError, TypeError, SyntaxError):
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


def _parse_single_text_action(text: str) -> dict[str, Any] | None:
    """Parse one textual ``crop_image`` action or final coordinate."""
    value = text or ""
    coordinate_after_think = _coordinate_after_think(value)
    if coordinate_after_think:
        point = _finite_point(
            coordinate_after_think.group(1), coordinate_after_think.group(2)
        )
        if point is None:
            return None
        return {
            "type": "final",
            "name": "Click",
            "point": point,
            "source": coordinate_after_think.group(0),
            "compatibility_format": "coordinate_after_think",
        }

    trailing_coordinate = _final_coordinate_match(value)
    if trailing_coordinate and not value[trailing_coordinate.end() :].strip():
        point = _finite_point(
            trailing_coordinate.group(1), trailing_coordinate.group(2)
        )
        if point is None:
            return None
        return {
            "type": "final",
            "name": "Click",
            "point": point,
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
        if lowered in {"click", "finished"}:
            point = arguments.get("box", arguments.get("point"))
            if point is None and "x" in arguments and "y" in arguments:
                point = (arguments["x"], arguments["y"])
            if isinstance(point, (list, tuple)) and len(point) == 2:
                try:
                    parsed_point = _finite_point(point[0], point[1])
                except (TypeError, ValueError):
                    return None
                if parsed_point is None:
                    return None
                return {
                    "type": "final",
                    "name": "Click",
                    "point": parsed_point,
                    "source": call_source,
                }

    try:
        json_action = _loads_action_json(candidate)
    except (ValueError, TypeError):
        json_action = None
    if isinstance(json_action, dict):
        name = str(json_action.get("name") or json_action.get("action") or "")
        arguments = json_action.get("arguments") or {}
        if name.lower() in {"crop", "crop_image"} and isinstance(arguments, dict):
            try:
                normalized = _normalize_crop_arguments(arguments)
                # Reject overflow values (e.g. ``1e999`` decoded as inf).
                json_safe_value(normalized)
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

    coordinate = _final_coordinate_match(value)
    if coordinate:
        point = _finite_point(coordinate.group(1), coordinate.group(2))
        if point is None:
            return None
        return {
            "type": "final",
            "name": "Click",
            "point": point,
            "source": coordinate.group(0),
            "compatibility_format": "coordinate",
        }
    return None


def parse_text_actions(text: str) -> list[dict[str, Any]] | None:
    """Parse one final coordinate or up to two crop actions from model content."""
    value = text or ""
    action_matches = list(ACTION_BLOCK_RE.finditer(value))
    if action_matches:
        if len(action_matches) > MAX_CROP_IMAGE_CALLS_PER_TURN:
            return None
        trailing_text = value[action_matches[-1].end() :]
        if _final_coordinate_match(trailing_text):
            return None
        actions = [
            _parse_single_text_action(match.group(1).strip())
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
            _parse_single_text_action(match.group(0))
            for match in xml_matches
        ]
        if any(action is None or action["type"] != "crop" for action in actions):
            return None
        return [action for action in actions if action is not None]

    try:
        json_batch = _loads_action_json(value.strip())
    except (ValueError, TypeError):
        json_batch = None
    if isinstance(json_batch, list) and all(
        isinstance(item, dict) for item in json_batch
    ):
        if not 1 <= len(json_batch) <= MAX_CROP_IMAGE_CALLS_PER_TURN:
            return None
        actions = [
            _parse_single_text_action(json.dumps(item, ensure_ascii=False))
            for item in json_batch
        ]
        if any(action is None or action["type"] != "crop" for action in actions):
            return None
        return [action for action in actions if action is not None]

    action = _parse_single_text_action(value)
    return [action] if action is not None else None


def parse_text_action(text: str) -> dict[str, Any] | None:
    """Parse model content while preserving the legacy single-action return shape."""
    actions = parse_text_actions(text)
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


def assistant_visible_content(content: str, reasoning_content: str = "") -> str:
    """Merge provider-separated reasoning into dataset-style visible content."""
    content = str(content or "")
    reasoning = str(reasoning_content or "").strip()
    paired_think = re.search(
        r"<think\b[^>]*>.*?</think\s*>",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    has_open_think = re.search(r"<think\b[^>]*>", content, re.IGNORECASE)
    has_close_think = re.search(r"</think\s*>", content, re.IGNORECASE)
    if has_close_think and (not has_open_think or has_close_think.start() < has_open_think.start()):
        return f"<think>\n{content.lstrip()}"
    if paired_think or not reasoning:
        return content
    suffix = f"\n{content.lstrip()}" if content else ""
    return f"<think>\n{reasoning}\n</think>{suffix}"


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


def crop_results_user_text(events: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Build the Swift string-form equivalent of Zoom-In' multimodal result message."""
    if not events:
        raise ValueError("events must contain at least one crop result")

    parts: list[str] = []
    image_paths: list[str] = []
    event_count = len(events)
    for position, event in enumerate(events, start=1):
        result = dict(event.get("result") or {})
        # Strip data_uri from the text shown to the model — it is only
        # needed for cross-node image transfer, not for the conversation.
        result.pop('data_uri', None)
        call_index = event.get("tool_call_index", position)
        if event_count == 1:
            heading = "crop_image action result:"
        else:
            heading = (
                f"crop_image action result {position}/{event_count} "
                f"(tool_call_index={call_index}):"
            )
        result_lines = [heading, _json_dumps_two_decimal_floats(result)]
        image_path = result.get("image_path") if result.get("ok") else None
        if image_path:
            image_path = str(image_path)
            result_lines.append(f"image_path: {image_path}")
            image_paths.append(image_path)
        parts.append("\n".join(result_lines))
        if image_path:
            parts.append("<image>")

    parts.append(
        "Inspect all successful crops and account for any failed actions. "
        "Reply with <think>...</think>, then either one or two independent "
        "crop_image <action> blocks whose coordinates have exactly two "
        "decimal places, or the final [x.xx,y.yy]. Do not mix crop actions "
        "with a final coordinate."
    )
    return "".join(parts), image_paths


def crop_user_text(result: dict[str, Any]) -> str:
    return crop_results_user_text([{"tool_call_index": 1, "result": result}])[0]


def crop_failure_text(result: dict[str, Any]) -> str:
    return crop_results_user_text([{"tool_call_index": 1, "result": result}])[0]


def execute_crop(
    raw_arguments: Any,
    registry: dict[str, dict[str, Any]],
    crop_dir: Path,
    crop_index: int,
    *,
    crop_upscale_factor: float = DEFAULT_CROP_UPSCALE_FACTOR,
    crop_max_pixels: int = DEFAULT_CROP_MAX_PIXELS,
    max_aspect_ratio: float = DEFAULT_MAX_CROP_ASPECT_RATIO,
) -> dict[str, Any]:
    """Execute one Zoom-In crop and preserve the mapping to the original image."""
    try:
        arguments = _parse_arguments(raw_arguments)
        source_path = registered_path(str(arguments["image_path"]))
        if source_path not in registry:
            raise ValueError("image_path is not the original image or a path returned by crop_image")
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
        _crop_w, _crop_h = right - left, bottom - top
        _aspect = max(_crop_w, _crop_h) / max(1, min(_crop_w, _crop_h))
        if _aspect > max_aspect_ratio:
            raise ValueError(
                f"Crop aspect ratio {_aspect:.1f} exceeds the limit "
                f"{max_aspect_ratio:.0f}; choose a broader region")

        crop_dir.mkdir(parents=True, exist_ok=True)
        output_path = crop_dir / f"crop_{crop_index:02d}.png"
        with Image.open(source_path) as image:
            cropped = image.crop((left, top, right, bottom))
            crop_width, crop_height = cropped.size
            resize_scale = min(
                crop_upscale_factor,
                math.sqrt(crop_max_pixels / (crop_width * crop_height)),
            )
            # Floor both dimensions so rounding can never push the resized crop
            # above the total-pixel budget. Neither width nor height is capped
            # independently.
            output_width = max(1, math.floor(crop_width * resize_scale))
            output_height = max(1, math.floor(crop_height * resize_scale))
            if output_width * output_height > crop_max_pixels:
                raise RuntimeError("Resized crop exceeds crop_max_pixels")
            if (output_width, output_height) != cropped.size:
                cropped = cropped.resize(
                    (output_width, output_height),
                    resample=Image.Resampling.LANCZOS,
                )
            cropped.save(output_path, format="PNG")

        # Encode crop as base64 data URI for cross-node transfer.
        # The training node may not have access to the rollout node's
        # local filesystem; embedding the image as base64 makes the
        # conversation data self-contained.
        with open(output_path, 'rb') as f:
            crop_b64 = base64.b64encode(f.read()).decode('ascii')
        data_uri = f'data:image/png;base64,{crop_b64}'

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
        output_key = registered_path(str(output_path))
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
            "data_uri": data_uri,
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
            "reminder": "Final coordinates must be normalized [0,1000] in the original image.",
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def parse_bbox(ground_truth: Any) -> list[float] | None:
    value = ground_truth
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in bbox):
        return None
    return bbox


def point_in_bbox_accuracy(point: Any, ground_truth: Any) -> float:
    """Return 1 iff a normalized point lies inside the normalized GT bbox."""
    bbox = parse_bbox(ground_truth)
    if hasattr(point, "tolist"):
        point = point.tolist()
    if bbox is None or not isinstance(point, (list, tuple)) or len(point) != 2:
        return 0.0
    try:
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(x) or not math.isfinite(y):
        return 0.0
    return float(bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3])


def accuracy_reward(solution_str: str, ground_truth: Any) -> float:
    return point_in_bbox_accuracy(parse_point(solution_str), ground_truth)


# ---------------------------------------------------------------------------
# Process reward / format reward / heuristic PRM turn scoring
# ---------------------------------------------------------------------------

REFUSAL_BBOX = [-1.0, -1.0, -1.0, -1.0]

# Defaults (overridable via ZOOMIN_* env vars in the plugin).
DEFAULT_REPEAT_IOU_THRESH = 0.9
DEFAULT_REPEAT_PENALTY = 0.2
DEFAULT_NO_IMPROVE_PENALTY = 0.2
DEFAULT_MAX_REPEAT_PENALTY = 0.4
DEFAULT_PRM_IOU_WEIGHT = 0.6
DEFAULT_ZOOM_BONUS = 0.2
DEFAULT_ZOOM_BONUS_MIN_QUALITY = 0.2
DEFAULT_ZOOM_BONUS_HARD_AREA_RATIO = 0.05


def _is_refusal_bbox(bbox: list[float] | None) -> bool:
    return bbox is not None and list(bbox) == REFUSAL_BBOX


def bbox_iou(box_a: Any, box_b: Any) -> float:
    """IoU of two [x1, y1, x2, y2] boxes in the same coordinate system."""
    a = parse_bbox(box_a)
    b = parse_bbox(box_b)
    if a is None or b is None:
        return 0.0
    inter_x1, inter_y1 = max(a[0], b[0]), max(a[1], b[1])
    inter_x2, inter_y2 = min(a[2], b[2]), min(a[3], b[3])
    inter_w, inter_h = inter_x2 - inter_x1, inter_y2 - inter_y1
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter_area = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _region_contains_point(region: Any, x: float, y: float) -> bool:
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(v) for v in region)
    except (TypeError, ValueError):
        return False
    return x1 <= x <= x2 and y1 <= y <= y2


def _successful_crop_regions(tool_events: Any) -> list[tuple[int, list[float]]]:
    """Extract (model_turn, original_image_region_0_1000) for successful crops, in order."""
    regions: list[tuple[int, list[float]]] = []
    if not isinstance(tool_events, (list, tuple)):
        return regions
    for event in tool_events:
        if not isinstance(event, dict):
            continue
        result = event.get('result') or {}
        if not result.get('ok'):
            continue
        region = result.get('original_image_region_0_1000')
        parsed = parse_bbox(region)
        if parsed is None:
            continue
        try:
            turn = int(event.get('model_turn', 0))
        except (TypeError, ValueError):
            turn = 0
        regions.append((turn, parsed))
    return regions


def turn_prm_scores(
    tool_events: Any,
    ground_truth: Any,
    *,
    iou_weight: float = DEFAULT_PRM_IOU_WEIGHT,
) -> dict[int, float]:
    """Heuristic PRM score per assistant turn that produced crop(s).

    score = iou_weight * IoU(crop_region, gt_bbox)
          + (1 - iou_weight) * 1[gt center inside crop region]

    Returns {model_turn: best_score}. Empty for refusal samples or no successful
    crop. iou_weight must be in [0, 1].
    """
    gt = parse_bbox(ground_truth)
    if _is_refusal_bbox(gt) or gt is None:
        return {}
    iou_weight = min(1.0, max(0.0, float(iou_weight)))
    center_x = (gt[0] + gt[2]) / 2.0
    center_y = (gt[1] + gt[3]) / 2.0
    scores: dict[int, float] = {}
    for turn, region in _successful_crop_regions(tool_events):
        iou = bbox_iou(region, gt)
        center_bonus = 1.0 if _region_contains_point(region, center_x, center_y) else 0.0
        score = iou_weight * iou + (1.0 - iou_weight) * center_bonus
        if turn not in scores or score > scores[turn]:
            scores[turn] = score
    return scores


def crop_process_reward(
    tool_events: Any,
    ground_truth: Any,
    *,
    repeat_iou_thresh: float = DEFAULT_REPEAT_IOU_THRESH,
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
    no_improve_penalty: float = DEFAULT_NO_IMPROVE_PENALTY,
    max_repeat_penalty: float = DEFAULT_MAX_REPEAT_PENALTY,
) -> float:
    """Trajectory-level process reward from crop regions vs GT bbox.

    * base score: best IoU over all successful crops
    * repeat penalty: adjacent crops whose regions overlap more than
      ``repeat_iou_thresh`` cost ``repeat_penalty`` each (capped at
      ``max_repeat_penalty``)
    * no-improve penalty (Gui-Cursor False-Direction, crop edition): with >=2
      successful crops, if the LAST crop's IoU is worse than the FIRST one,
      the zoom chain did not converge on the target

    Result is clipped to [0, 1]. Refusal samples and crop-free trajectories
    score 0 (neutral: accuracy reward still dominates those cases).
    """
    gt = parse_bbox(ground_truth)
    if _is_refusal_bbox(gt) or gt is None:
        return 0.0
    regions = _successful_crop_regions(tool_events)
    if not regions:
        return 0.0

    ious = [bbox_iou(region, gt) for _, region in regions]
    reward = max(ious)

    repeat_total = 0.0
    for prev, curr in zip(regions, regions[1:]):
        if bbox_iou(prev[1], curr[1]) > repeat_iou_thresh:
            repeat_total += repeat_penalty
    repeat_total = min(repeat_total, max_repeat_penalty)
    reward -= repeat_total

    if len(ious) >= 2 and ious[-1] < ious[0]:
        reward -= no_improve_penalty

    return min(1.0, max(0.0, reward))


DEFAULT_CALL_COST = 0.05


def gated_crop_process_reward(
    tool_events: Any,
    ground_truth: Any,
    *,
    repeat_iou_thresh: float = DEFAULT_REPEAT_IOU_THRESH,
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
    no_improve_penalty: float = DEFAULT_NO_IMPROVE_PENALTY,
    max_repeat_penalty: float = DEFAULT_MAX_REPEAT_PENALTY,
    call_cost: float = DEFAULT_CALL_COST,
) -> float:
    """Gated process reward: crop scoring plus a per-call tool-usage cost.

    Identical to :func:`crop_process_reward` except that every successful
    crop_image call charges ``call_cost``:

        reward = max_i IoU(crop_i, GT)
               - repeat penalty (capped at ``max_repeat_penalty``)
               - no-improve penalty
               - call_cost * (# successful crops)

    Intended to be applied **only to trajectories whose final answer is
    correct** — the outcome gate lives in the ORM wrapper
    (``ZoomInCropProcessGated``). Together they change the incentive shape:

    * wrong answers never earn process credit (no "crop well, answer wrong"
      farming, and no credit for lucky wide-area search);
    * correct trajectories pay for every extra crop, so one well-placed crop
      outscores a longer search chain and direct answers keep their full
      accuracy credit without being dragged down.

    Refusal samples and crop-free trajectories score 0.
    """
    gt = parse_bbox(ground_truth)
    if _is_refusal_bbox(gt) or gt is None:
        return 0.0
    regions = _successful_crop_regions(tool_events)
    if not regions:
        return 0.0

    ious = [bbox_iou(region, gt) for _, region in regions]
    reward = max(ious)

    repeat_total = 0.0
    for prev, curr in zip(regions, regions[1:]):
        if bbox_iou(prev[1], curr[1]) > repeat_iou_thresh:
            repeat_total += repeat_penalty
    repeat_total = min(repeat_total, max_repeat_penalty)
    reward -= repeat_total

    if len(ious) >= 2 and ious[-1] < ious[0]:
        reward -= no_improve_penalty

    reward -= max(0.0, float(call_cost)) * len(regions)

    return min(1.0, max(0.0, reward))


def format_reward(completion: str) -> float:
    """Format compliance of ONE assistant message (usually the last one).

    * 1.0 — canonical path: coordinate right after ``</think>``, or an
      ``<action>`` block parsed by the strict ast parser (no repair/fallback)
    * 0.5 — parseable but only via a fallback path (repaired call, JSON/XML
      compatibility, or a bare coordinate without think-then-answer structure)
    * 0.0 — unparseable
    """
    action = parse_text_action(completion)
    if action is None:
        return 0.0
    action_type = action.get('type')
    if action_type == 'final':
        return 1.0 if action.get('compatibility_format') == 'coordinate_after_think' else 0.5
    if action_type == 'crop':
        return 1.0 if 'compatibility_format' not in action else 0.5
    if action_type == 'crop_batch':
        sub_actions = action.get('actions') or []
        if sub_actions and all('compatibility_format' not in a for a in sub_actions):
            return 1.0
        return 0.5
    return 0.5


def gt_area_ratio(ground_truth: Any) -> float | None:
    """GT bbox area as a fraction of the image (norm1000 space, image = 1e6 units²)."""
    gt = parse_bbox(ground_truth)
    if gt is None or _is_refusal_bbox(gt):
        return None
    width = max(0.0, gt[2] - gt[0])
    height = max(0.0, gt[3] - gt[1])
    return width * height / 1_000_000.0


def zoom_bonus_reward(
    completion: Any,
    tool_events: Any,
    ground_truth: Any,
    *,
    bonus: float = DEFAULT_ZOOM_BONUS,
    min_quality: float = DEFAULT_ZOOM_BONUS_MIN_QUALITY,
    hard_area_ratio: float = DEFAULT_ZOOM_BONUS_HARD_AREA_RATIO,
    hard_only: bool = True,
    iou_weight: float = DEFAULT_PRM_IOU_WEIGHT,
    final_point: Any = None,
) -> float:
    """Zoom-in bonus: reward *effective tool use* on top of a correct answer.

    Returns ``bonus`` iff ALL of the following hold, else 0.0:

    1. the final answer lands inside the GT bbox (``final_point`` from the
       scheduler's rollout_infos is preferred; falls back to parsing
       ``completion``);
    2. the sample is "hard" — GT bbox area ratio below ``hard_area_ratio``
       (skipped when ``hard_only=False``). Big targets should be answered
       directly, so zooming there earns nothing;
    3. at least one successful crop reached ``min_quality`` under the same
       heuristic PRM score used for turn filtering — a crop that actually
       overlaps the target. Cropping an irrelevant region earns nothing.

    Refusal samples always score 0: the protocol expects a direct ``[-1,-1]``
    answer without any crop there.
    """
    gt = parse_bbox(ground_truth)
    if gt is None or _is_refusal_bbox(gt):
        return 0.0
    if hard_only:
        ratio = gt_area_ratio(gt)
        if ratio is None or ratio >= hard_area_ratio:
            return 0.0
    point = final_point if final_point is not None else parse_point(str(completion or ''))
    if point_in_bbox_accuracy(point, gt) < 1.0:
        return 0.0
    scores = turn_prm_scores(tool_events, gt, iou_weight=iou_weight)
    if not scores or max(scores.values()) < min_quality:
        return 0.0
    return float(bonus)


# ---------------------------------------------------------------------------
# CoT transcript extraction for the external LLM-as-judge reward
# ---------------------------------------------------------------------------

THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>(.*?)</think\s*>", re.DOTALL | re.IGNORECASE)
COT_SCORE_RE = re.compile(r"Score:\s*\**\s*([01](?:\.\d+)?)\s*\**", re.IGNORECASE)
COT_SCORE_FALLBACK_RE = re.compile(r"\[\[\s*([01](?:\.\d+)?)\s*\]\]")

# Defaults (overridable via ZOOMIN_COT_RM_* env vars in the plugin).
DEFAULT_COT_MAX_SEGMENT_CHARS = 1500
DEFAULT_COT_MAX_ACTION_CHARS = 600
DEFAULT_COT_MAX_TOTAL_CHARS = 6000
DEFAULT_COT_MAX_TASK_CHARS = 400


def message_content_text(content: Any) -> str:
    """Flatten message content into plain text.

    Multimodal content (list of parts) keeps only textual items — image
    payloads / data URIs are intentionally dropped, since the CoT judge is
    a text-only model and base64 images would blow up the prompt.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def extract_think_blocks(text: str) -> list[str]:
    """Return the stripped contents of every ``<think>...</think>`` block."""
    return [block.strip() for block in THINK_BLOCK_RE.findall(text or "") if block.strip()]


def extract_task_from_system_prompt(system_prompt: str) -> str:
    """Recover the task instruction embedded by ``build_system_prompt``."""
    text = system_prompt or ""
    match = re.search(r"Task instruction:\s*\n(.*?)(?:\n\n|\nOriginal image metadata:)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()[:DEFAULT_COT_MAX_TASK_CHARS]


def _cot_action_summary(text: str, max_action_chars: int) -> str:
    """Summarize the action part of one assistant turn (crop calls or final point)."""
    action = parse_text_action(text)
    if action is None:
        blocks = ACTION_BLOCK_RE.findall(text)
        if blocks:
            return "; ".join(block.strip() for block in blocks)[:max_action_chars]
        bare = _final_coordinate_match(text)
        return bare.group(0) if bare else ""
    if action.get("type") == "final":
        return f"final answer [{action['point'][0]:.2f},{action['point'][1]:.2f}]"
    raw = str(action.get("source") or "").strip()
    if raw:
        return raw[:max_action_chars]
    return ""


def cot_event_feedback(event: dict[str, Any]) -> str:
    """One-line text summary of a crop tool event for the judge (no images/paths)."""
    result = event.get("result") or {}
    if result.get("ok"):
        region = result.get("original_image_region_0_1000")
        parsed = parse_bbox(region)
        if parsed is not None:
            region_text = (f"[{parsed[0]:.2f},{parsed[1]:.2f},"
                           f"{parsed[2]:.2f},{parsed[3]:.2f}]")
            dims = result.get("width"), result.get("height")
            dims_text = f"{dims[0]}x{dims[1]}" if all(isinstance(d, (int, float)) for d in dims) else ""
            base = f"crop executed; region mapped to original-image [0,1000] coordinates {region_text}"
            return f"{base}; crop size {dims_text}" if dims_text else base
        return "crop executed"
    error = str(result.get("error") or "unknown error")
    return f"crop failed: {error[:200]}"


def _cot_feedback_by_turn(tool_events: Any) -> dict[int, list[str]]:
    feedback: dict[int, list[str]] = {}
    if not isinstance(tool_events, (list, tuple)):
        return feedback
    for event in tool_events:
        if not isinstance(event, dict):
            continue
        try:
            turn = int(event.get("model_turn", 0))
        except (TypeError, ValueError):
            turn = 0
        feedback.setdefault(turn, []).append(cot_event_feedback(event))
    return feedback


def build_cot_transcript(
    messages: Any,
    tool_events: Any = None,
    *,
    max_segment_chars: int = DEFAULT_COT_MAX_SEGMENT_CHARS,
    max_action_chars: int = DEFAULT_COT_MAX_ACTION_CHARS,
    max_total_chars: int = DEFAULT_COT_MAX_TOTAL_CHARS,
) -> str:
    """Flatten a trajectory's assistant turns into a text-only CoT transcript.

    Layout per assistant turn::

        [Turn N]
        think: <reasoning text>          (all think blocks, truncated)
        action: <crop call / final point> (structured summary)
        feedback: <crop execution result> (from tool_events, if provided)

    Only text survives: user turns (which may embed images / base64 data
    URIs) are skipped entirely, and crop feedback is rebuilt from
    ``tool_events`` metadata instead of raw message content.
    """
    if not isinstance(messages, (list, tuple)):
        return ""
    feedback_by_turn = _cot_feedback_by_turn(tool_events)
    parts: list[str] = []
    total_len = 0
    turn_index = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        turn_index += 1
        text = message_content_text(message.get("content", ""))
        if not text.strip():
            continue
        thinks = extract_think_blocks(text)
        action_summary = _cot_action_summary(text, max_action_chars)
        if not thinks and not action_summary:
            continue
        lines = [f"[Turn {turn_index}]"]
        if thinks:
            merged = "\n".join(thinks)
            lines.append(f"think: {merged[:max_segment_chars]}")
        if action_summary:
            lines.append(f"action: {action_summary}")
        for feedback in feedback_by_turn.get(turn_index, []):
            lines.append(f"feedback: {feedback}")
        block = "\n".join(lines)
        if total_len + len(block) > max_total_chars:
            tail = "\n[transcript truncated]"
            remaining = max_total_chars - total_len - len(tail)
            if remaining > 200:
                parts.append(block[:remaining] + tail)
            break
        parts.append(block)
        total_len += len(block) + 1
    return "\n\n".join(parts)


def has_think_blocks(messages: Any) -> bool:
    """Whether any assistant turn contains a non-empty ``<think>...
</think>

`` block."""
    if not isinstance(messages, (list, tuple)):
        return False
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        if extract_think_blocks(message_content_text(message.get("content", ""))):
            return True
    return False


def extract_cot_score(judge_output: str) -> float | None:
    """Parse the judge's final ``Score: 0.xx`` (fallback: ``[[0.xx]]``).

    Out-of-[0,1] values (e.g. ``Score: 1.5``) count as parse failures and
    return ``None`` so the caller can fall back to the neutral score instead
    of silently clamping.
    """
    text = judge_output or ""
    match = COT_SCORE_RE.search(text) or COT_SCORE_FALLBACK_RE.search(text)
    if match is None:
        return None
    try:
        score = float(match.group(1))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    return score


__all__ = [
    "COT_SCORE_FALLBACK_RE",
    "COT_SCORE_RE",
    "CROP_IMAGE_DESCRIPTION",
    "DEFAULT_COT_MAX_ACTION_CHARS",
    "DEFAULT_COT_MAX_SEGMENT_CHARS",
    "DEFAULT_COT_MAX_TASK_CHARS",
    "DEFAULT_COT_MAX_TOTAL_CHARS",
    "DEFAULT_CROP_MAX_PIXELS",
    "DEFAULT_CROP_UPSCALE_FACTOR",
    "DEFAULT_MAX_CROP_ASPECT_RATIO",
    "DEFAULT_MAX_REPEAT_PENALTY",
    "DEFAULT_NO_IMPROVE_PENALTY",
    "DEFAULT_PRM_IOU_WEIGHT",
    "DEFAULT_REPEAT_IOU_THRESH",
    "DEFAULT_REPEAT_PENALTY",
    "DEFAULT_ZOOM_BONUS",
    "DEFAULT_ZOOM_BONUS_HARD_AREA_RATIO",
    "DEFAULT_ZOOM_BONUS_MIN_QUALITY",
    "INVALID_ACTION_CORRECTION",
    "MAX_CROP_IMAGE_CALLS",
    "MAX_CROP_IMAGE_CALLS_PER_TURN",
    "REFUSAL_BBOX",
    "SYSTEM_PROMPT_TEMPLATE",
    "THINK_BLOCK_RE",
    "accuracy_reward",
    "assistant_visible_content",
    "bbox_iou",
    "build_cot_transcript",
    "build_system_prompt",
    "cot_event_feedback",
    "crop_failure_text",
    "crop_process_reward",
    "crop_results_user_text",
    "crop_user_text",
    "execute_crop",
    "extract_cot_score",
    "extract_instruction",
    "extract_task_from_system_prompt",
    "extract_think_blocks",
    "format_reward",
    "gt_area_ratio",
    "has_think_blocks",
    "message_content_text",
    "parse_bbox",
    "parse_point",
    "parse_text_action",
    "parse_text_actions",
    "point_in_bbox_accuracy",
    "registered_path",
    "safe_task_component",
    "turn_prm_scores",
    "zoom_bonus_reward",
]
