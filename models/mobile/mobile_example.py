"""Run multi-turn UI-Venus-2 Mobile inference over prerecorded screenshots.

Usage:
    bash scripts/mobile.sh
"""

import argparse
import ast
import base64
import json
import math
import re
import struct
from pathlib import Path


SYSTEM_PROMPT = """**You are a GUI Agent.** Your role is to analyze the user's task, provide clear and accurate answers to their questions, and execute the task with precise actions.

### Available Actions
You may execute one of the following functions:

- Click(point=(x1, y1))
> Perform a tap action at the specified screen coordinate. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).

- Drag(start=(x1, y1), end=(x2, y2))
> Perform a drag action by long-pressing at the start coordinate for a few seconds and then dragging to the end coordinate. This is typically used for adjusting app layouts, moving sliders, solving slider captchas, etc. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).

- Swipe(start=(x1, y1), end=(x2, y2))
> Perform a swipe action by dragging from the start coordinate to the end coordinate. This is typically used for scrolling to find content, switching tabs, pulling down the notification shade, etc. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).

- DoubleClick(point=(x1, y1))
> Perform a double tap action at the specified screen coordinate. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).

- LongPress(point=(x1, y1))
> Perform a long-press action at the specified screen coordinate for a certain duration. This can be used to trigger additional options, such as copy, forward, delete, etc. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).

- Type(content='')
> Enter the specified text into the currently active input field.

- LaunchApp(app='')
> Launch the target app. Use this action when the target app is not currently visible on the screen.

- Wait()
> Wait for the current page, animation, or content to finish loading.

- CallUser(content='')
> Request user takeover or additional information when needed, for example, when there are multiple on-screen options that satisfy the requirement.

- GetScreenshot()
> Take a screenshot and save it to the device's photo album.

- PressBack()
> Return to the previous screen.

- PressHome()
> Return to the system home screen.

- PressEnter()
> Perform an Enter key action.

- PressRecent()
> Open the system recent apps screen.

- Answer(content='')
> Answer the user's questions as requested.

- Finished(content='')
> Mark the task as completed and inform the user of the task execution status.

### Instructions
- Make sure you understand the task goal to avoid wrong actions.
- Make sure you carefully examine the current screenshot. Sometimes the summarized history might not be reliable, over-claiming some effects.
- If additional information is needed during task execution, use `CallUser` to interact with the user.
- Consider exploring the screen by using the `Swipe` action with different directions to reveal additional content.
- To copy text: first select the exact text you want to copy, which usually also brings up the text selection bar, then click the `copy` button in bar.
- To paste text into a text box, first long press the text box, then usually the text selection bar will appear with a `paste` button in it.

### Output Format
<think> your thinking process </think>
<action> the next action </action>

### User Task
{user_task}"""

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
NORM = 1000


def encode_image(image_path):
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def get_image_size(image_path):
    header = image_path.read_bytes()[:24]
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Only PNG screenshots are supported: {image_path}")
    return struct.unpack(">II", header[16:24])


def image_content(image_path, label):
    return [
        {"type": "text", "text": label},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image(image_path)}"},
        },
    ]


def parse_response(response):
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    action_match = re.search(r"<action>(.*?)</action>", response, re.DOTALL)
    think = think_match.group(1).strip() if think_match else ""
    action = action_match.group(1).strip() if action_match else ""
    return think, action


def parse_action_call(action):
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
    except (SyntaxError, ValueError):
        params = {}
    return name, params


def to_pixel(point, width, height):
    if not isinstance(point, (tuple, list)) or len(point) != 2:
        raise ValueError("point must contain two coordinates")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in point):
        raise ValueError("coordinates must be finite numbers")
    if any(value < 0 or value >= NORM for value in point):
        raise ValueError("coordinates must be within [0, 999]")
    return [
        int(point[0] / NORM * width),
        int(point[1] / NORM * height),
    ]


def normalize_action(action, width, height):
    name, params = parse_action_call(action)
    action_type = ACTION_NAME_MAP.get(name)
    result = {
        "action_type": action_type,
        "action_pos": [],
        "input": "",
        "duration": -1,
    }
    if action_type in {"CLK", "DoubleCLK", "LongPress"}:
        point = params.get("point", params.get("box"))
        if point:
            result["action_pos"] = [to_pixel(point, width, height)]
    elif action_type == "SWIPE":
        if "start" in params and "end" in params:
            result["action_pos"] = [
                to_pixel(params["start"], width, height),
                to_pixel(params["end"], width, height),
            ]
        if name == "Drag":
            result["duration"] = 1000
    elif action_type == "INPUT":
        text = params.get("content", "")
        result["input"] = text.replace("(", "（").replace(")", "）").replace("|", "｜")
    elif action_type == "REOPEN":
        result["app_name"] = params.get("app", "")
    elif action_type == "WAIT":
        result["duration"] = 1000
    elif action_type in {"CallUser", "SUCCESS"}:
        result["input"] = params.get("content", "")
    return name, result


def build_messages(task, history, current_image, n_img):
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(user_task=task)}]
    image_start = max(0, len(history) - n_img)
    for index, turn in enumerate(history):
        content = ""
        if n_img > 0 and index >= image_start:
            content = image_content(turn["image_path"], "History Screenshot:")
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": turn["raw_response"]})
    messages.append({"role": "user", "content": image_content(current_image, "Current Screenshot:\n")})
    return messages


def call_model(client, model_name, messages, temperature, max_tokens):
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"repetition_penalty": 1.05, "frequency_penalty": 0.3},
    )
    content = response.choices[0].message.content or ""
    reasoning = getattr(response.choices[0].message, "reasoning_content", None)
    reasoning = reasoning or getattr(response.choices[0].message, "reasoning", None)
    if reasoning and "<think>" not in content:
        content = f"<think>{reasoning}</think>\n{content}"
    return content


def save_result(output_path, task, model_name, turns):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "task": task,
        "model_name": model_name,
        "turns": [
            {
                "turn": turn["turn"],
                "screenshot": turn["screenshot"],
                "think": turn["think"],
                "action": turn["action"],
                "parsed_action": turn["parsed_action"],
                "raw_response": turn["raw_response"],
            }
            for turn in turns
        ],
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--n-img", type=int, default=0)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    from openai import OpenAI

    input_path = Path(args.input_file).resolve()
    output_path = Path(args.output_file).resolve()
    example = json.loads(input_path.read_text(encoding="utf-8"))
    task = example["task"]
    screenshots = [(input_path.parent / path).resolve() for path in example["screenshots"]]

    client = OpenAI(base_url=args.model_url, api_key=args.api_key, timeout=120.0)
    history = []
    for index, image_path in enumerate(screenshots, start=1):
        messages = build_messages(task, history, image_path, max(0, args.n_img))
        raw_response = call_model(
            client,
            args.model_name,
            messages,
            args.temperature,
            args.max_tokens,
        )
        think, action = parse_response(raw_response)
        width, height = get_image_size(image_path)
        action_name, parsed_action = normalize_action(action, width, height)
        turn = {
            "turn": index,
            "screenshot": str(image_path.relative_to(input_path.parent)),
            "image_path": image_path,
            "think": think,
            "action": action,
            "parsed_action": parsed_action,
            "raw_response": raw_response,
        }
        history.append(turn)
        save_result(output_path, task, args.model_name, history)
        print(f"Turn {index}: {action}")
        if action_name == "Finished":
            break


if __name__ == "__main__":
    main()
