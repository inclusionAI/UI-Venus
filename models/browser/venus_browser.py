#!/usr/bin/env python3
"""Minimal standalone Venus-style browser agent.

Install:
  pip install openai playwright

Start Chrome with a CDP port, then configure the agent:
  export CDP_URL="http://127.0.0.1:9222"
  export LLM_API_URL="https://your-endpoint/v1"
  export LLM_API_KEY="your-key"       # optional for local endpoints
  export LLM_MODEL="your-vision-model"

Run:
  bash scripts/browser.sh \
    "Open https://example.com and report the page title"

Artifacts are saved to results/browser/<timestamp>/ by the unified script.
The example is fully standalone.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import re
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any


WIDTH, HEIGHT, NORM = 1024, 768, 1000
URL_RE = re.compile(r"https?://[^\s<>\"']+", re.I)
SUPPORTED_ACTIONS = {
    "click", "drag", "scroll", "type", "launch", "wait", "geturl",
    "finished", "takenote", "calluser", "longpress", "pressback",
    "presshome", "pressenter", "hover", "doubleclick", "hotkey",
    "selectoption",
}

PROMPT = """**You are a GUI Browser Agent.**
Your task is to analyze a given user task, review current screenshot and previous actions, and determine the next action to complete the task.

### Available Actions
You may execute one of the following functions:
- Click(point=(x1,y1))
> Perform a tap action at the specified screen coordinate. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).
- Drag(start=(x1,y1), end=(x2,y2))
> Perform a drag action by long-pressing at the start coordinate for a few seconds and then dragging to the end coordinate. This is typically used for adjusting element layouts, moving sliders, solving slider captchas, etc. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).
- Scroll(point=(x1, y1), direction='up/down/left/right')
> Perform a scroll action on coordinate (x1, y1). This is typically used for scrolling to find content, switching tabs, pulling down the notification shade, etc. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999), scroll direction='up/down/left/right'.
- Type(content='')
> Enter the specified text into the currently active input field.
- Launch(url='')
> Launch the target url. Use this action when the target website is not currently visible on the screen.
- Wait()
> Wait for the current page, animation, or content to finish loading.
- GetUrl()
> Get the URL of the current browser tab. The URL is returned at the beginning of the next user message.
- Finished(content='')
> Mark the task as completed and inform the user of the task execution status.
- TakeNote(content='')
> Record important information from screenshots avoiding forgetting.
- CallUser(content='')
> Request user takeover or additional information when needed, for example, when there are multiple on-screen options that satisfy the requirement.
- LongPress(point=(x1,y1), duration=20)
> Press and hold the specified screen coordinate for `duration` seconds. `duration` must be a positive number and defaults to 20 when omitted. This can trigger additional options, such as copy, forward, or delete. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).
- PressBack()
> Return to the previous page.
- PressHome()
> Press the Home key to scroll to the top of the current page.
- PressEnter()
> Perform an Enter key action.
- Hover(point=(x1,y1))
> Perform a hover action at the specified screen coordinate. This can be used to reveal additional information or options, such as tooltips, dropdown menus, etc. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).
- DoubleClick(point=(x1,y1))
> Perform a double tap action at the specified screen coordinate. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).
- Hotkey(keys=('ctrl', 'c'))
> Press combination keys. Keys with comma and wrap each key in single quotes. Do not use more than 3 keys in one Hotkey action. Use `Hotkey(keys=('ctrl', 'tab'))` for the next browser tab or add `shift` for the previous tab.
- SelectOption(index=3)
> Choose an option from the last clicked native HTML select element. Use this only when the current user message provides an explicit native select option list. Do not use keyboard navigation hotkeys for native select dropdowns. For custom dropdowns or visible menu items in the screenshot, click the visible option directly.

### Instruction
- Today is {current_date}.
- Make sure you understand the task goal to avoid wrong actions.
- Make sure you carefully examine the the current screenshot. Sometimes the summarized history might not be reliable, over-claiming some effects.
- If additional information is needed during task execution, use `CallUser` to interact with the user.
- Consider exploring the screen by using the `Scroll` action with different directions to reveal additional content.
- Try to use simple language when searching.
- If you meet ERR_CONNECTION_CLOSED or 404 NOT FOUND error, please type the website key word in https://www.google.com to find the correct url.
- The official website of cryptpad is https://cryptpad.fr/ .
- Distinguish textbox from button: never `Type` into a button. If no textbox is visible, try clicking the search icon first — the input field may appear afterward.
- Strictly avoid repeating the same action when the webpage remains unchanged — you may have executed the wrong action. Continuous use of `Wait()` is also NOT allowed.

# Very Important
Take Notes:
- You are forgetful and will forget all information from the current screenshot before you scroll to next one. When you see important information(e.g. partial step info) for completing the task in the current screenshot, RECORD it using `TakeNote(content='...')` before you scrolling it down.
- The information needed for a task is often distributed across multiple pages. Even partial information should be taken note of — do not wait until all information is seen.
- Before you take `scroll` action, make sure you have taken notes for all important information in the current screenshot.

Apply Filters:
- If filters are available on the page, prioritize using filters for precise searching rather than using the search function for fuzzy searching.

### Output Format
<think> your thinking process </think>
<action> the next action </action>

### User Task
{task}
"""


def parse_action(text: str, reasoning: str = "") -> tuple[str, dict, str]:
    """Parse one function action without executing model-generated code."""
    blocks = re.findall(r"<action>\s*(.*?)\s*</action>", text, re.I | re.S)
    if len(blocks) != 1:
        raise ValueError("expected exactly one <action> block")
    call = ast.parse(blocks[0], mode="eval").body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise ValueError("action must be a simple function call")
    if call.args:
        raise ValueError("use named arguments")
    args = {}
    for item in call.keywords:
        if item.arg is None or item.arg in args:
            raise ValueError("invalid action arguments")
        args[item.arg] = ast.literal_eval(item.value)
    think = re.search(r"<think>\s*(.*?)\s*</think>", text, re.I | re.S)
    thought = reasoning.strip() or (think.group(1).strip() if think else "")
    return call.func.id.lower(), args, thought


def point(value: Any, width: int = WIDTH, height: int = HEIGHT) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("point must contain two coordinates")
    x, y = map(float, value)
    if not (0 <= x < NORM and 0 <= y < NORM):
        raise ValueError("coordinates must be within [0, 999]")
    return min(width - 1, round(x / NORM * width)), min(
        height - 1, round(y / NORM * height)
    )


# Kept as a public name to make the tiny parser easy to unit test/reuse.
normalized_point = point


def viewport(page) -> tuple[int, int]:
    size = getattr(page, "viewport_size", None)
    if not size:
        try:
            size = page.evaluate(
                "() => ({width: window.innerWidth, height: window.innerHeight})"
            )
        except Exception:
            size = None
    if not isinstance(size, dict):
        return WIDTH, HEIGHT
    return int(size.get("width", WIDTH)), int(size.get("height", HEIGHT))


def page_point(page, value: Any) -> tuple[int, int]:
    width, height = viewport(page)
    return point(value, width, height)


def select_at(page, x: int, y: int) -> dict | None:
    return page.evaluate(
        """({x, y}) => {
            const element = document.elementFromPoint(x, y);
            const select = element && element.closest('select');
            if (!select) return null;
            if (!select.dataset.venusMiniId) {
                select.dataset.venusMiniId = `select-${Date.now()}`;
            }
            return {
                id: select.dataset.venusMiniId,
                options: Array.from(select.options).map((option, index) => ({
                    index, text: option.text, value: option.value,
                    selected: option.selected, disabled: option.disabled,
                })),
            };
        }""",
        {"x": x, "y": y},
    )


def execute(page, name: str, args: dict, state: dict):
    """Return (finished, answer, feedback, active_page)."""
    name = name.replace("_", "")
    if name not in SUPPORTED_ACTIONS:
        raise ValueError(f"unsupported action: {name}")
    feedback = ""
    if name == "click":
        x, y = page_point(page, args.get("point"))
        selected = select_at(page, x, y)
        page.mouse.click(x, y)
        if selected:
            state["select"] = selected
            feedback = "Available options for the select element:\n" + "\n".join(
                f"{item['index']}. text={item['text']!r}, value={item['value']!r}"
                for item in selected["options"]
            )
        else:
            state.pop("select", None)
    elif name == "doubleclick":
        page.mouse.dblclick(*page_point(page, args.get("point")))
    elif name == "hover":
        page.mouse.move(*page_point(page, args.get("point")))
    elif name == "type":
        page.keyboard.insert_text(str(args.get("content", "")))
    elif name == "scroll":
        x, y = page_point(page, args.get("point", (500, 500)))
        direction = str(args.get("direction", "down")).lower()
        width, height = viewport(page)
        deltas = {
            "up": (0, -height * 2 // 3), "down": (0, height * 2 // 3),
            "left": (-width * 2 // 3, 0), "right": (width * 2 // 3, 0),
        }
        if direction not in deltas:
            raise ValueError("invalid scroll direction")
        page.mouse.move(x, y)
        page.mouse.wheel(*deltas[direction])
    elif name == "drag":
        start = page_point(page, args.get("start"))
        end = page_point(page, args.get("end"))
        page.mouse.move(*start)
        page.mouse.down()
        page.mouse.move(*end, steps=10)
        page.mouse.up()
    elif name == "launch":
        url = str(args.get("url", ""))
        if not url.startswith(("http://", "https://")):
            raise ValueError("Launch requires an http(s) URL")
        target = page.context.new_page()
        try:
            target.goto(url, wait_until="domcontentloaded", timeout=60_000)
        except Exception:
            target.close()
            raise
        page = target
    elif name == "pressback":
        page.go_back(wait_until="domcontentloaded", timeout=30_000)
    elif name == "presshome":
        page.keyboard.press("Home")
    elif name == "pressenter":
        page.keyboard.press("Enter")
    elif name == "hotkey":
        keys = args.get("keys", ())
        keys = (keys,) if isinstance(keys, str) else keys
        if not isinstance(keys, (tuple, list)) or not 1 <= len(keys) <= 3:
            raise ValueError("Hotkey requires one to three keys")
        mapping = {
            "ctrl": "ControlOrMeta", "cmd": "Meta", "meta": "Meta",
            "alt": "Alt", "shift": "Shift", "enter": "Enter",
            "tab": "Tab", "escape": "Escape", "backspace": "Backspace",
            "delete": "Delete", "arrowup": "ArrowUp",
            "arrowdown": "ArrowDown", "arrowleft": "ArrowLeft",
            "arrowright": "ArrowRight",
        }
        chord = "+".join(mapping.get(str(k).lower(), str(k)) for k in keys)
        if not chord:
            raise ValueError("Hotkey requires keys")
        page.keyboard.press(chord)
    elif name == "longpress":
        duration = float(args.get("duration", 20))
        if duration <= 0:
            raise ValueError("LongPress duration must be positive")
        page.mouse.move(*page_point(page, args.get("point")))
        page.mouse.down()
        try:
            page.wait_for_timeout(round(duration * 1000))
        finally:
            page.mouse.up()
    elif name == "selectoption":
        selected = state.get("select")
        if not selected:
            raise ValueError("no native select was clicked in the previous step")
        index = int(args.get("index"))
        result = page.evaluate(
            """({id, index}) => {
                const select = document.querySelector(
                    `select[data-venus-mini-id="${id}"]`
                );
                if (!select) return {ok: false, error: 'select is gone'};
                if (index < 0 || index >= select.options.length) {
                    return {ok: false, error: 'option index out of range'};
                }
                if (select.options[index].disabled) {
                    return {ok: false, error: 'option is disabled'};
                }
                select.selectedIndex = index;
                select.dispatchEvent(new Event('input', {bubbles: true}));
                select.dispatchEvent(new Event('change', {bubbles: true}));
                return {ok: true, text: select.options[index].text};
            }""",
            {"id": selected["id"], "index": index},
        )
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "SelectOption failed"))
        feedback = f"Selected option: {result['text']}"
        state.pop("select", None)
    elif name == "wait":
        page.wait_for_timeout(3_000)
    elif name == "geturl":
        feedback = f"Current URL: {page.url}"
    elif name == "takenote":
        note = str(args.get("content", "")).strip()
        if note:
            state["notes"].append(note)
    elif name == "calluser":
        question = str(args.get("content", "Please provide more information."))
        feedback = f"User response: {input(question + ' ')}"
    elif name == "finished":
        return True, str(args.get("content", "")), "", page
    else:
        raise ValueError(f"unsupported action: {name}")
    if name not in ("wait", "longpress", "calluser"):
        page.wait_for_timeout(500)
    if page.context.pages:
        page = page.context.pages[-1]
    return False, "", feedback, page


def image_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def start_url(task: str) -> str:
    match = URL_RE.search(task)
    return match.group().rstrip(".,;:!?)]}") if match else "https://www.google.com/"


def append_jsonl(path: Path, item: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")


def make_user(image: str, feedback: str = "", notes: list | None = None) -> dict:
    content = []
    if feedback:
        content.append({"type": "text", "text": feedback})
    if notes:
        content.append({
            "type": "text",
            "text": "Notes from earlier steps:\n- " + "\n- ".join(notes),
        })
    content.append({"type": "image_url", "image_url": {"url": image}})
    return {"role": "user", "content": content}


def run(args: argparse.Namespace) -> tuple[int, Path]:
    try:
        from openai import OpenAI
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Install dependencies first:\n"
            "  pip install openai playwright"
        ) from error

    task = " ".join(args.task).strip()
    model = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "")).strip()
    if not model:
        raise SystemExit("Set LLM_MODEL to an image-capable chat model.")
    client = OpenAI(
        base_url=os.getenv(
            "LLM_API_URL",
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        ),
        api_key=os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "empty")),
        timeout=300,
    )

    run_dir = Path(args.output).resolve() / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    shots = run_dir / "screenshots"
    shots.mkdir(parents=True)
    history_file = run_dir / "history.jsonl"
    history = deque(maxlen=2)  # two previous (screenshot, response) turns
    state = {"notes": []}
    system_prompt = PROMPT.format(
        current_date=datetime.now().date().isoformat(), task=task
    )
    feedback = answer = ""
    finished = False
    steps = 0
    cdp_url = os.getenv("CDP_URL", "http://127.0.0.1:9222")
    for variable in ("NO_PROXY", "no_proxy"):
        entries = [item.strip() for item in os.getenv(variable, "").split(",") if item.strip()]
        for host in ("127.0.0.1", "localhost"):
            if host not in entries:
                entries.append(host)
        os.environ[variable] = ",".join(entries)
    print(f"Task: {task}\nCDP: {cdp_url}\nArtifacts: {run_dir}")

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.connect_over_cdp(cdp_url)
        except Exception as error:
            raise RuntimeError(
                f"Cannot connect to Chrome at {cdp_url}. Start Chrome with "
                "--remote-debugging-port=9222 and verify /json/version."
            ) from error
        if not browser.contexts:
            raise RuntimeError("The CDP browser has no default context")
        context = browser.contexts[0]
        page = context.pages[-1] if context.pages else context.new_page()
        try:
            try:
                page.goto(start_url(task), wait_until="domcontentloaded", timeout=60_000)
            except Exception as error:
                feedback = f"Initial navigation failed: {error}"

            for step in range(1, args.max_steps + 1):
                steps = step
                page = context.pages[-1]  # follow tabs opened by clicks
                shot = shots / f"step_{step:03d}.png"
                page.screenshot(path=str(shot), animations="disabled")
                user = make_user(image_url(shot), feedback, state["notes"])
                messages = [{"role": "system", "content": system_prompt}]
                for old_user, old_reply in history:
                    messages += [old_user, {"role": "assistant", "content": old_reply}]
                messages.append(user)

                record = {
                    "step": step, "url_before": page.url,
                    "screenshot": str(shot.relative_to(run_dir)),
                    "input_feedback": feedback,
                    "notes": list(state["notes"]),
                }
                feedback = ""
                try:
                    request = dict(
                        model=model, messages=messages,
                        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
                        temperature=0,
                    )
                    thinking_enabled = os.getenv("LLM_THINKING", "false").lower() in (
                        "1", "true", "yes"
                    )
                    request["extra_body"] = {
                        "chat_template_kwargs": {
                            "enable_thinking": thinking_enabled
                        }
                    }
                    response = client.chat.completions.create(**request)
                    msg = response.choices[0].message
                    reasoning = str(getattr(msg, "reasoning_content", "") or "")
                    reply = str(msg.content or reasoning).strip()
                    record["model_response"] = reply
                    if response.usage:
                        record["prompt_tokens"] = response.usage.prompt_tokens
                        record["completion_tokens"] = response.usage.completion_tokens
                except Exception as error:
                    feedback = record["error"] = f"Model call failed: {error}"
                    append_jsonl(history_file, record)
                    continue

                history.append((user, reply))
                try:
                    name, action_args, thought = parse_action(reply, reasoning)
                    record.update(action=name, action_args=action_args, thought=thought)
                    print(f"[{step}] {name}({action_args})")
                    finished, answer, feedback, page = execute(
                        page, name, action_args, state
                    )
                    record["url_after"] = page.url
                except Exception as error:
                    record["error"] = f"Action failed: {error}"
                    feedback = (
                        "Previous action execution failed.\n"
                        f"Action: {reply}\nError: {error}\n"
                        "The screenshot and failed action were preserved. Inspect "
                        "the current page and retry or choose another action."
                    )
                append_jsonl(history_file, record)
                if finished:
                    break

            context.pages[-1].screenshot(path=str(run_dir / "final.png"))
        finally:
            # Leaving sync_playwright disconnects this client. Do not close the
            # CDP browser/context: the user owns that Chrome process and profile.
            pass

    result = {"task": task, "finished": finished, "answer": answer,
              "steps": steps, "history": "history.jsonl"}
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Answer: {answer or '(no final answer)'}\nSaved: {run_dir}")
    return (0 if finished else 2), run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one browser vision task.")
    parser.add_argument("task", nargs="+", help="natural-language browser task")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--output", default="results/browser")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    return args


if __name__ == "__main__":
    code, _ = run(parse_args())
    raise SystemExit(code)
