#!/usr/bin/env python3
# coding: utf-8
"""Run vLLM inference on one CAPTCHA image and parse the returned action DSL.

Usage:
    bash scripts/captcha.sh

This module:
1. Converts one local image into a vLLM/OpenAI-compatible multimodal message.
2. Runs one inference request through the vLLM OpenAI-compatible API.
3. Parses Click, LongPress, Type, and Drag actions from the model output.
4. Saves the raw output, reasoning, inference metadata, and parsed actions as one JSON object.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional, Sequence

from PIL import Image

from captcha_prompt import SYS_PROMPT, USER_PROMPT


LOGGER = logging.getLogger("infer_captcha")

# ---------------------------------------------------------------------------
# CAPTCHA action DSL parsing
# ---------------------------------------------------------------------------

# Coordinates may be 0-999 integers, 0-1 decimals, or absolute-pixel decimals.
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"

# Use one combined expression to preserve the original action order.
# Keep Type as a separate action, so Click+Type becomes two adjacent dictionaries.
_ACTION_PATTERN = re.compile(
    r"(?P<click>(?P<click_kind>Click|LongPress)\s*\(\s*"
    r"(?:box|bbox)\s*=\s*[\(\[]\s*(?P<cx>" + _NUMBER + r")\s*,\s*"
    r"(?P<cy>" + _NUMBER + r")\s*[\)\]]\s*\))"
    r"|(?P<type>Type\s*\(\s*content\s*=\s*(?P<quote>['\"])"
    r"(?P<content>.*?)(?P=quote)\s*\))"
    r"|(?P<drag>Drag\s*\(\s*start\s*=\s*[\(\[]\s*"
    r"(?P<sx>" + _NUMBER + r")\s*,\s*(?P<sy>" + _NUMBER + r")\s*[\)\]]"
    r"\s*,\s*end\s*=\s*[\(\[]\s*(?P<ex>" + _NUMBER + r")\s*,\s*"
    r"(?P<ey>" + _NUMBER + r")\s*[\)\]]\s*\))",
    re.DOTALL,
)


def _number(value: str) -> int | float:
    """Convert a captured coordinate to int or float."""
    parsed = float(value)
    return int(parsed) if parsed.is_integer() and "." not in value else parsed


def parse_actions(text: Any) -> list[dict[str, Any]]:
    """Extract Click, LongPress, Type, and Drag actions in occurrence order.

    Prefer the first ``<action>...</action>`` block. If the model omits the
    action tag, scan the full text to tolerate slightly malformed output.
    """
    source = "" if text is None else str(text)
    tagged = re.search(
        r"<action\b[^>]*>(.*?)</action>",
        source,
        re.DOTALL | re.IGNORECASE,
    )
    action_text = tagged.group(1) if tagged else source
    actions: list[dict[str, Any]] = []

    # finditer returns matches by text position, preserving multi-action order.
    for match in _ACTION_PATTERN.finditer(action_text):
        if match.group("click"):
            actions.append(
                {
                    "type": match.group("click_kind"),
                    "x": _number(match.group("cx")),
                    "y": _number(match.group("cy")),
                }
            )
        elif match.group("type"):
            actions.append({"type": "Type", "content": match.group("content")})
        else:
            actions.append(
                {
                    "type": "Drag",
                    "sx": _number(match.group("sx")),
                    "sy": _number(match.group("sy")),
                    "ex": _number(match.group("ex")),
                    "ey": _number(match.group("ey")),
                }
            )
    return actions


# ---------------------------------------------------------------------------
# Image loading and OpenAI multimodal input construction
# ---------------------------------------------------------------------------

def _detect_image_mime(prefix: bytes, path: str) -> str:
    """Prefer file signatures when detecting MIME to handle incorrect extensions."""
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if prefix.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if prefix.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if prefix.startswith(b"RIFF") and prefix[8:12] == b"WEBP":
        return "image/webp"
    if prefix.startswith(b"BM"):
        return "image/bmp"
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


def image_to_data_url(path: str) -> str:
    """Encode a local image as a Base64 data URL accepted by the vLLM OpenAI API."""
    with open(path, "rb") as image_file:
        data = image_file.read()
    mime = _detect_image_mime(data[:32], path)
    return "data:%s;base64,%s" % (mime, base64.b64encode(data).decode("ascii"))


def read_image_size(path: str) -> tuple[int, int]:
    """Read image dimensions with Pillow."""
    with Image.open(path) as image:
        return image.size


def default_system_prompt(coord_scale: float) -> str:
    """Return the original system prompt from the real-world v1 dataset."""

    # The default prompt explicitly defines a 0-999 coordinate range. A caller
    # using another coordinate space must also provide a matching system prompt.
    if coord_scale != 999:
        raise ValueError(
            "数据集 system prompt 固定使用 0-999 坐标；"
            "如需其他坐标体系，请同时传入 --system-prompt"
        )
    return SYS_PROMPT


def build_messages(
    image_path: str,
    task: str,
    coord_scale: float,
    system_prompt: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Build one multimodal chat conversation for a single image."""
    user_content: list[dict[str, Any]] = []

    # The dataset user prompt is ``<image>``. OpenAI messages represent the image
    # as a structured image_url, so the literal placeholder must not be repeated.
    user_text = task.replace("<image>", "").strip()
    if user_text:
        user_content.append({"type": "text", "text": user_text})
    user_content.append(
        {
            "type": "image_url",
            "image_url": {"url": image_to_data_url(image_path)},
        }
    )
    return [
        {
            "role": "system",
            "content": system_prompt
            if system_prompt is not None
            else default_system_prompt(coord_scale),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def _response_text(content: Any) -> str:
    """Handle both string and segmented API message.content responses."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                pieces.append(str(item.get("text", item.get("content", ""))))
        return "\n".join(piece for piece in pieces if piece).strip()
    return "" if content is None else str(content).strip()


def _request_payload(
    args: argparse.Namespace,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a vLLM OpenAI-compatible chat completion request body."""
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "presence_penalty": args.presence_penalty,
        "chat_template_kwargs": {
            # Pass this explicitly to avoid relying on an implicit server default.
            "enable_thinking": args.enable_thinking,
        },
    }
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.seed is not None:
        payload["seed"] = args.seed
    return payload


# ---------------------------------------------------------------------------
# vLLM OpenAI-compatible API
# ---------------------------------------------------------------------------

def infer_with_vllm_api(
    args: argparse.Namespace,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run inference on one image through the vLLM OpenAI-compatible API."""
    base_url = args.base_url.rstrip("/")

    # Accept both ``.../v1`` and full ``.../v1/chat/completions`` URLs.
    endpoint = (
        base_url
        if base_url.endswith("/chat/completions")
        else base_url + "/chat/completions"
    )
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "empty"
    body = json.dumps(_request_payload(args, messages), ensure_ascii=False).encode(
        "utf-8"
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
    }
    last_error: Optional[BaseException] = None
    started = time.perf_counter()
    for attempt in range(1, args.max_retries + 1):
        # Create a fresh Request for each retry rather than reusing a consumed object.
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=args.request_timeout,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            choices = payload.get("choices") or []
            if not choices:
                raise RuntimeError("API 响应没有 choices")
            choice = choices[0]
            message = choice.get("message") or {}
            response_model = payload.get("model", args.model)
            if isinstance(response_model, str) and os.path.isabs(response_model):
                response_model = args.model
            return {
                "model_output": _response_text(message.get("content")),
                "reasoning_content": message.get(
                    "reasoning_content",
                    message.get("reasoning"),
                ),
                "inference": {
                    "backend": "vllm-api",
                    "model": response_model,
                    "seconds": round(time.perf_counter() - started, 4),
                    "attempts": attempt,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": payload.get("usage"),
                    "request_id": payload.get("id"),
                },
            }
        except urllib.error.HTTPError as error:
            # Most 4xx parameter errors are permanent; retry rate limits, timeouts, and 5xx errors.
            response_body = error.read().decode("utf-8", errors="replace")[:1000]
            last_error = RuntimeError("HTTP %d: %s" % (error.code, response_body))
            retryable = error.code in {408, 409, 425, 429} or error.code >= 500
            if not retryable:
                break
        except (urllib.error.URLError, TimeoutError, ValueError, RuntimeError) as error:
            last_error = error
        if attempt < args.max_retries:
            # Cap exponential backoff at eight seconds; one-image tasks need no concurrent backoff.
            time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError("vLLM API 推理失败: %s" % last_error) from last_error


# ---------------------------------------------------------------------------
# Result persistence and command-line entry point
# ---------------------------------------------------------------------------

def write_result(result: dict[str, Any], output_path: str) -> None:
    """Save one-image JSON atomically to avoid partial output after interruption."""
    output = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    file_descriptor, temporary = tempfile.mkstemp(
        prefix=Path(output).name + ".",
        suffix=".tmp",
        dir=os.path.dirname(output),
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as destination:
            json.dump(result, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
        # Keep both files in one directory so os.replace can complete atomically.
        os.replace(temporary, output)
    except BaseException:
        # Remove the temporary file for every exception, including KeyboardInterrupt, then re-raise.
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    """Define command-line arguments for one-image inference."""
    parser = argparse.ArgumentParser(
        description="单张 CAPTCHA 图片推理与动作解析（vLLM API）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="一张本地 CAPTCHA 图片")
    parser.add_argument("--model", required=True, help="vLLM 服务端模型名")

    # vLLM API connection arguments.
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=None, help="默认读取 OPENAI_API_KEY")
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--max-retries", type=int, default=3)

    # Generation arguments.
    parser.add_argument("--max-tokens", type=int, default=40960)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=996)
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="显式向模型 chat template 传入 enable_thinking=true",
    )
    thinking_group.add_argument(
        "--no-enable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="显式向模型 chat template 传入 enable_thinking=false",
    )
    parser.set_defaults(enable_thinking=True)
    parser.add_argument(
        "--task",
        default=USER_PROMPT,
        help="user 输入；默认 <image> 仅发送图片，可传自定义文本",
    )
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument(
        "--coord-scale",
        type=float,
        default=999.0,
        help="模型坐标上界：999、1 或 0（绝对像素）",
    )
    parser.add_argument(
        "--output",
        default="test_result.json",
        help="结果 JSON 文件",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    """Perform inexpensive argument validation before loading models or making requests."""
    image_path = os.path.abspath(os.path.expanduser(args.image))
    args.image = os.path.relpath(image_path)
    args.output = os.path.relpath(
        os.path.abspath(os.path.expanduser(args.output))
    )
    if not os.path.isfile(image_path):
        parser.error("--image 不存在或不是文件: %s" % args.image)
    if args.request_timeout <= 0:
        parser.error("--request-timeout 必须 > 0")
    if args.max_retries < 1:
        parser.error("--max-retries 必须 >= 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens 必须 >= 1")
    if args.temperature < 0:
        parser.error("--temperature 必须 >= 0")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p 必须在 (0, 1] 范围内")
    return image_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one-image inference, then save and print the result object."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    image_path = _validate_args(args, parser)
    try:
        # Image dimensions are recorded in the result but are not used for inference.
        width, height = read_image_size(image_path)
        messages = build_messages(
            image_path=image_path,
            task=args.task,
            coord_scale=args.coord_scale,
            system_prompt=args.system_prompt,
        )
        inference_result = infer_with_vllm_api(args, messages)
        output_directory = os.path.dirname(os.path.abspath(args.output))
        result = {
            "image": os.path.relpath(image_path, output_directory),
            "image_size": [width, height],
            "coord_scale": args.coord_scale,
            "enable_thinking": args.enable_thinking,
            "task": args.task,
            **inference_result,
        }
        # Save both raw text and structured actions to simplify parser debugging.
        result["parsed_actions"] = parse_actions(result["model_output"])
        if not result["parsed_actions"]:
            LOGGER.warning("模型输出中没有解析出受支持的 CAPTCHA 动作")
        write_result(result, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        LOGGER.info("单图结果已保存: %s", args.output)
        return 0
    except KeyboardInterrupt:
        LOGGER.error("用户中断")
        return 130
    except Exception as error:
        LOGGER.error("%s", error)
        if args.log_level == "DEBUG":
            LOGGER.exception("详细错误")
        return 1


if __name__ == "__main__":
    sys.exit(main())
