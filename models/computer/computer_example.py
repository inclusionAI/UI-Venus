#!/usr/bin/env python3
"""Standalone UI-Venus-2 computer inference on prerecorded screenshots.

This module calls an OpenAI-compatible multimodal endpoint, safely parses the
returned Computer action, and records normalized action data for a caller to
execute elsewhere. It does not depend on an external desktop runtime.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


NORM = 999

SYSTEM_PROMPT = r"""**You are a GUI Agent.**
Your role is to analyze the user's task, provide clear and accurate answers to their questions, and execute the task with precise actions on a desktop operating system. The password of the computer is {sudo_password}.

### Available Actions
You may execute one of the following functions. Coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).
- Click(box=(x1, y1)), or Click()
> Perform a left-click at `box`, or at the current cursor position when `box` is omitted.
- DoubleClick(box=(x1, y1)), or DoubleClick()
> Perform a double-click (selects a word in text). Use `box` to move first, or omit it to act at the current cursor position.
- TripleClick(box=(x1, y1)), or TripleClick()
> Perform a triple-click (selects a line or the content of a single-line input). Use `box` to move first, or omit it to act at the current cursor position.
- RightClick(box=(x1, y1)), or RightClick()
> Perform a right-click to open a context menu. Use `box` to move first, or omit it to act at the current cursor position.
- MiddleClick(box=(x1, y1)), or MiddleClick()
> Perform a middle-click (for example, open a link in a new tab). Use `box` to move first, or omit it to act at the current cursor position.
- Hover(box=(x1, y1))
> Move the cursor immediately to the coordinate WITHOUT clicking.
- Drag(end=(x2, y2), start=(x1, y1))
> Drag to `end` using a fixed 0.5-second drag. `start` is optional; omit it to begin from the current cursor position.
- Swipe(amount=-5, axis='vertical')
> Scroll at the current cursor position. `amount` is an integer from -4096 to 4096 and controls magnitude and direction: vertical positive scrolls up and negative scrolls down; horizontal positive scrolls right and negative scrolls left.
- Type(content='')
> Type the provided text into the focused field. Each `\n` presses Enter.
- Hotkey(keys=['ctrl', 'c'], repeat=1)
> Press 1 to 128 listed keys as a keyboard shortcut. Use `repeat=N` from 1 to 128 to press the shortcut N times.
- KeyDown(keys=['shift'])
> Press 1 to 128 listed keys in order and keep them held across later actions and model turns until a matching `KeyUp`.
- KeyUp(keys=['shift'])
> Release 1 to 128 listed keys in order.
- MouseDown(box=(x1, y1)), or MouseDown()
> Optionally move to `box`, then press and hold the left mouse button across later actions and model turns.
- MouseUp(box=(x1, y1)), or MouseUp()
> Optionally move to `box`, then release the left mouse button.
- Sequence(actions=[Click(box=(x1, y1)), Hotkey(keys=['ctrl', 's'])])
> Execute 2 to 32 actions in order as one open-loop model turn. Nested `Sequence` is not allowed, and `CallUser` or `Finished` may appear only as the final action.
- Wait()
> Wait for the current page, animation, or content to finish loading.
- CallUser(content='')
> Request user takeover or report failure when the task cannot be completed or additional information is required.
- Finished(content='')
> Mark the task as completed successfully and optionally report details in `content`.

### Instructions
- Make sure you understand the task goal to avoid wrong actions.
- Prefer one atomic action per turn. Use `Sequence` only when every child action is already known and no intermediate screenshot is needed; its children execute open-loop.
- `KeyDown`, `KeyUp`, `MouseDown`, and `MouseUp` preserve input state across turns. Release held input explicitly when it is no longer needed.
- Any `keys` list may contain at most 128 non-empty key names; each key name is limited to 1,024 characters.
- `Swipe` is the only scrolling action and always scrolls at the current cursor position.
- Make sure you carefully examine the current screenshot. Sometimes the summarized history might not be reliable, over-claiming some effects.
- To submit/search after typing into a field, end the text with a newline — `Type(content='query\n')` — which types the text and presses Enter in one action.
- To replace the existing content of an input field, use `TripleClick` to select it, then `Type` the new content.
- To open a submenu/dropdown, use `Hover` over the parent item to reveal it, then `Click` the desired entry.
- To use a context menu, `RightClick` the target to open it, then `Click` the desired entry.
- To hold a modifier during another action, use `KeyDown`, the target action, and `KeyUp`. Put them in one `Sequence` only when no intermediate screenshot is needed.
- After launching an app, running a command, downloading, or any slow operation, use `Wait()` to let it finish before continuing.
- To press a key or shortcut several times, use `repeat`, e.g. `Hotkey(keys=['down'], repeat=5)` or `Hotkey(keys=['ctrl', 'z'], repeat=3)`, instead of repeating the action.
- Consider exploring the screen by using the `Swipe` action to scroll and reveal additional content.
- Use `Hotkey` for keyboard shortcuts: copy (`ctrl+c`), paste (`ctrl+v`), save (`ctrl+s`), undo (`ctrl+z`), find (`ctrl+f`), etc.
- If the task cannot be completed or additional information is needed, use `CallUser`. Use `Finished` only after successful completion.

### Output Format
<think> your thinking process </think>
<action> the next action </action>

### User Task
{user_task}"""


class ComputerActionError(ValueError):
    """Model output is not a safe, supported Computer action."""


@dataclass(frozen=True)
class ActionCall:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    actions: tuple["ActionCall", ...] = ()


@dataclass(frozen=True)
class _Schema:
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()


SCHEMAS = {
    "Click": _Schema(optional=("box",)),
    "DoubleClick": _Schema(optional=("box",)),
    "TripleClick": _Schema(optional=("box",)),
    "RightClick": _Schema(optional=("box",)),
    "MiddleClick": _Schema(optional=("box",)),
    "Hover": _Schema(required=("box",)),
    "Drag": _Schema(required=("end",), optional=("start",)),
    "Swipe": _Schema(required=("amount", "axis")),
    "Type": _Schema(optional=("content",)),
    "Hotkey": _Schema(required=("keys",), optional=("repeat",)),
    "KeyDown": _Schema(required=("keys",)),
    "KeyUp": _Schema(required=("keys",)),
    "MouseDown": _Schema(optional=("box",)),
    "MouseUp": _Schema(optional=("box",)),
    "Wait": _Schema(),
    "CallUser": _Schema(optional=("content",)),
    "Finished": _Schema(optional=("content",)),
}
COORD_ARGS = {"box", "start", "end"}
TERMINALS = {"CallUser", "Finished"}


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        value = node.value
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise ComputerActionError(f"unsupported literal: {type(value).__name__}")
        return value
    if isinstance(node, ast.Tuple):
        return tuple(_literal(item) for item in node.elts)
    if isinstance(node, ast.List):
        return [_literal(item) for item in node.elts]
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and not isinstance(node.operand.value, bool)
    ):
        return node.operand.value if isinstance(node.op, ast.UAdd) else -node.operand.value
    raise ComputerActionError("only literal strings, finite numbers, lists, and tuples are allowed")


def _parse_call(node: ast.AST, *, allow_sequence: bool) -> ActionCall:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ComputerActionError("action must be a direct function call")
    if node.args:
        raise ComputerActionError("positional arguments are not allowed")
    keyword_nodes: dict[str, ast.AST] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            raise ComputerActionError("**kwargs are not allowed")
        if keyword.arg in keyword_nodes:
            raise ComputerActionError(f"duplicate argument: {keyword.arg}")
        keyword_nodes[keyword.arg] = keyword.value

    if node.func.id == "Sequence":
        if not allow_sequence:
            raise ComputerActionError("nested Sequence is not allowed")
        if set(keyword_nodes) != {"actions"} or not isinstance(keyword_nodes["actions"], ast.List):
            raise ComputerActionError("Sequence requires actions=[...]")
        call = ActionCall(
            "Sequence",
            actions=tuple(_parse_call(item, allow_sequence=False) for item in keyword_nodes["actions"].elts),
        )
    else:
        call = ActionCall(node.func.id, {key: _literal(value) for key, value in keyword_nodes.items()})
    _validate(call)
    return call


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (isinstance(value, int) or math.isfinite(value))
    )


def _validate(call: ActionCall) -> None:
    if call.name == "Sequence":
        if call.kwargs or not 2 <= len(call.actions) <= 32:
            raise ComputerActionError("Sequence must contain between 2 and 32 actions")
        for index, child in enumerate(call.actions):
            if child.name == "Sequence":
                raise ComputerActionError("nested Sequence is not allowed")
            _validate(child)
            if child.name in TERMINALS and index != len(call.actions) - 1:
                raise ComputerActionError("terminal action must be last in Sequence")
        return

    schema = SCHEMAS.get(call.name)
    if schema is None:
        raise ComputerActionError(f"unsupported Computer action: {call.name!r}")
    if call.actions:
        raise ComputerActionError("atomic action cannot contain child actions")
    actual = set(call.kwargs)
    missing = set(schema.required) - actual
    unknown = actual - set(schema.required) - set(schema.optional)
    if missing:
        raise ComputerActionError(f"{call.name} missing arguments: {sorted(missing)}")
    if unknown:
        raise ComputerActionError(f"{call.name} has unknown arguments: {sorted(unknown)}")

    for key, value in call.kwargs.items():
        if key in COORD_ARGS:
            if not isinstance(value, tuple) or len(value) != 2 or not all(_is_number(v) for v in value):
                raise ComputerActionError(f"{call.name}.{key} must be a two-number tuple")
            if not all(0 <= v <= NORM for v in value):
                raise ComputerActionError(f"{call.name}.{key} coordinates must be in [0, 999]")
        elif key == "keys":
            if not isinstance(value, list) or not value or any(not isinstance(v, str) or not v.strip() for v in value):
                raise ComputerActionError(f"{call.name}.keys must be a non-empty string list")
        elif key == "repeat":
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ComputerActionError("Hotkey.repeat must be a positive integer")
        elif key == "amount":
            if not isinstance(value, int) or isinstance(value, bool):
                raise ComputerActionError("Swipe.amount must be an integer")
        elif key == "axis":
            if value not in {"vertical", "horizontal"}:
                raise ComputerActionError("Swipe.axis must be 'vertical' or 'horizontal'")
        elif key == "content":
            if not isinstance(value, str):
                raise ComputerActionError(f"{call.name}.content must be a string")


def parse_action_call(text: str) -> ActionCall:
    """Safely parse one Computer action without executing it."""
    if not isinstance(text, str) or not text.strip():
        raise ComputerActionError("action text must be a non-empty string")
    try:
        node = ast.parse(text.strip(), mode="eval").body
    except (SyntaxError, ValueError, TypeError, RecursionError) as exc:
        raise ComputerActionError(f"invalid action syntax: {exc}") from exc
    return _parse_call(node, allow_sequence=True)


def parse_response(text: str, reasoning: str = "") -> tuple[str, str, ActionCall]:
    """Extract thought/action and validate the action; never execute model text."""
    import re

    action_blocks = re.findall(r"<action>\s*(.*?)\s*</action>", text, re.I | re.S)
    if action_blocks:
        if len(action_blocks) != 1:
            raise ComputerActionError("expected exactly one <action> block")
        action_text = action_blocks[0]
    elif "<action" in text.lower() or "</action>" in text.lower():
        raise ComputerActionError("malformed <action> block")
    else:
        action_text = text.strip()
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.I | re.S)
    thought = reasoning.strip() or (think_match.group(1).strip() if think_match else "")
    return thought, action_text, parse_action_call(action_text)


def normalized_point(value: Sequence[float], width: int, height: int) -> list[int]:
    """Convert a validated 0..999 point to pixels, clamping both endpoints."""
    if width <= 0 or height <= 0:
        raise ComputerActionError("image dimensions must be positive")
    if len(value) != 2 or not all(_is_number(v) and 0 <= v <= NORM for v in value):
        raise ComputerActionError("coordinate must contain two numbers in [0, 999]")
    return [
        max(0, min(width - 1, int(value[0] * width / NORM))),
        max(0, min(height - 1, int(value[1] * height / NORM))),
    ]


def normalize_action(call: ActionCall, width: int, height: int) -> dict[str, Any]:
    """Return JSON-safe action data with pixel coordinates and terminal state."""
    if call.name == "Sequence":
        actions = []
        for child in call.actions:
            actions.append(normalize_action(child, width, height))
            if child.name in TERMINALS:
                break
        result: dict[str, Any] = {"name": "Sequence", "actions": actions}
        if actions and "terminal" in actions[-1]:
            result["terminal"] = actions[-1]["terminal"]
        return result
    args: dict[str, Any] = {}
    for key, value in call.kwargs.items():
        args[key] = normalized_point(value, width, height) if key in COORD_ARGS else value
    result: dict[str, Any] = {"name": call.name, "arguments": args}
    if call.name == "Finished":
        result["terminal"] = "success"
    elif call.name == "CallUser":
        result["terminal"] = "needs_user"
    return result


def _image_bytes(image: str | Path | bytes) -> bytes:
    return image if isinstance(image, bytes) else Path(image).read_bytes()


def image_size(image: str | Path | bytes) -> tuple[int, int]:
    from PIL import Image
    from io import BytesIO

    with Image.open(BytesIO(_image_bytes(image))) as opened:
        return opened.size


def image_content(image: str | Path | bytes, label: str) -> list[dict[str, Any]]:
    encoded = base64.b64encode(_image_bytes(image)).decode("ascii")
    return [
        {"type": "text", "text": label},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
    ]


def build_messages(
    task: str,
    history: Sequence[Mapping[str, Any]],
    current_image: str | Path | bytes,
    n_img: int = 0,
    sudo_password: str = "password",
) -> list[dict[str, Any]]:
    """Build system + all assistant text + the last n_img history images."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(user_task=task, sudo_password=sudo_password)}]
    image_start = max(0, len(history) - max(0, n_img))
    for index, turn in enumerate(history):
        content: Any = ""
        if index >= image_start and n_img > 0:
            content = image_content(turn["image"], "History Screenshot:\n")
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": turn["accepted_response"]})
    messages.append({"role": "user", "content": image_content(current_image, "Current Screenshot:\n")})
    return messages


def call_model(client: Any, model_name: str, messages: list[dict[str, Any]], *, temperature: float, top_p: float, max_tokens: int, enable_thinking: bool = False) -> tuple[str, str]:
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if enable_thinking:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
    response = client.chat.completions.create(**kwargs)
    if not response.choices:
        raise RuntimeError("model response has no choices")
    choice = response.choices[0]
    if getattr(choice, "finish_reason", None) not in {None, "stop"}:
        raise RuntimeError(f"model response ended incompletely: {choice.finish_reason!r}")
    message = choice.message
    content = message.content or ""
    reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or ""
    if reasoning and "<think>" not in content:
        content = f"<think>\n{reasoning.strip()}\n</think>\n{content.strip()}"
    return content, reasoning


class VenusComputerAgent:
    """Stateful offline-screenshot inference API with accepted-only history."""

    def __init__(self, client: Any, model_name: str, *, temperature: float = 0.0, top_p: float = 0.7, max_tokens: int = 4096, n_img: int = 2, sudo_password: str = "password", enable_thinking: bool = False, parse_retries: int = 1):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.n_img = max(0, n_img)
        self.sudo_password = sudo_password
        self.enable_thinking = enable_thinking
        self.parse_retries = max(0, parse_retries)
        self.history: list[dict[str, Any]] = []

    def infer(self, task: str, screenshot: str | Path | bytes) -> dict[str, Any]:
        messages = build_messages(task, self.history, screenshot, self.n_img, self.sudo_password)
        last_error: Exception | None = None
        for _ in range(self.parse_retries + 1):
            content, reasoning = call_model(
                self.client, self.model_name, messages,
                temperature=self.temperature, top_p=self.top_p,
                max_tokens=self.max_tokens, enable_thinking=self.enable_thinking,
            )
            try:
                thought, action_text, call = parse_response(content, reasoning)
                break
            except ComputerActionError as exc:
                last_error = exc
        else:
            raise ComputerActionError(f"invalid action after {self.parse_retries + 1} attempts: {last_error}") from last_error

        width, height = image_size(screenshot)
        parts = []
        if thought:
            parts.append(f"<think>{thought}</think>")
        parts.append(f"<action>{action_text}</action>")
        accepted_response = "\n".join(parts)
        turn = {
            "turn": len(self.history) + 1,
            "think": thought,
            "action": action_text,
            "parsed_action": normalize_action(call, width, height),
            "raw_response": content,
            "accepted_response": accepted_response,
            "image": screenshot,
        }
        self.history.append(turn)
        return {key: value for key, value in turn.items() if key not in {"image", "accepted_response"}}


def _load_input(args: argparse.Namespace) -> tuple[str, list[Path], Path | None]:
    if args.input_file:
        source = Path(args.input_file).resolve()
        data = json.loads(source.read_text(encoding="utf-8"))
        task = args.task or data["task"]
        screenshots = [Path(path) if Path(path).is_absolute() else source.parent / path for path in data["screenshots"]]
        return task, [path.resolve() for path in screenshots], source
    if not args.task or not args.screenshot:
        raise SystemExit("provide --input-file, or both --task and --screenshot")
    return args.task, [Path(path).resolve() for path in args.screenshot], None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--task")
    parser.add_argument("--screenshot", action="append", help="repeat for an offline sequence")
    parser.add_argument("--input-file", help="JSON with task and screenshots")
    parser.add_argument("--output-file", default="results/computer/output.json")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--n-img", type=int, default=2, help="number of additional history images")
    parser.add_argument("--sudo-password", default="password")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    from openai import OpenAI

    task, screenshots, input_path = _load_input(args)
    client = OpenAI(base_url=args.model_url, api_key=args.api_key, timeout=args.timeout)
    agent = VenusComputerAgent(
        client, args.model_name, temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens, n_img=args.n_img, sudo_password=args.sudo_password,
        enable_thinking=args.enable_thinking, parse_retries=args.parse_retries,
    )
    turns = []
    for screenshot in screenshots:
        turn = agent.infer(task, screenshot)
        if input_path:
            try:
                screenshot_label = str(screenshot.relative_to(input_path.parent))
            except ValueError:
                screenshot_label = str(screenshot)
        else:
            screenshot_label = str(screenshot)
        turn["screenshot"] = screenshot_label
        turns.append(turn)
        print(f"Turn {turn['turn']}: {turn['action']}")
        terminal = turn["parsed_action"].get("terminal")
        if terminal:
            break

    output = Path(args.output_file).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"task": task, "model_name": args.model_name, "turns": turns}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
