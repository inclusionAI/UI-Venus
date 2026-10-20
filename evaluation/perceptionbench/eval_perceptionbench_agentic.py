#!/usr/bin/env python3
"""Agentic PerceptionBench QA evaluation with crop/zoom-in support.

This script adapts the text-action crop_image protocol (helpers in
``models/agentic_helpers.py``) to PerceptionBench's free-form
visual-question-answering benchmark.  The model receives a visual question,
optionally requests crop_image text actions to inspect regions at higher
resolution, and then returns the final answer inside an ``<answer>...</answer>``
tag.  Judging uses the ``eval/judge_prompt_zoom_in.txt`` template, so the judge
is aware of the agentic protocol.  The prediction endpoint must be a local
``vllm serve`` OpenAI-compatible server; reasoning effort is passed exclusively
through ``chat_template_kwargs``.

Reliability / metrology features (P0 hardening):

* ``parse_qa_text_action`` extracts a free-form answer from ``<answer>`` tags
  or a trailing ``Final answer:`` line, and drives the multi-turn crop loop.
* ``decode_judge_robust`` accepts the bracketed ``[reason]/[judge]`` format,
  a JSON object ``{"judge": bool, "reason": str}``, or a trailing bare
  ``True/False``; outputs that still cannot be parsed are retried once with a
  format reminder and then classified as ``parse_error``.
* Judge-side failures (``parse_error`` / ``judge_error``) are recorded with
  ``judge_status`` and excluded from the primary accuracy denominator instead
  of being silently counted as model errors.
* ``run_gold_pass`` judges each reference answer against itself (all must be
  True) and ``DOUBLE_JUDGE_SAMPLE`` runs a second independent judgment on a
  random subset to measure judge agreement.
* The runner supports resume (streaming JSONL), ``LIMIT`` for smoke tests, and
  optional ``JUDGE_QPM``/``JUDGE_RPS`` rate limiting.

Usage: configure the endpoint variables (MODEL / OPENAI_API_KEY /
OPENAI_BASE_URL, normally exported by the run wrapper from an env file).
Run from the ``perceptionbench`` directory::

    python eval_perceptionbench_agentic.py

Optional environment overrides:
    AGENTIC_MAX_TOOL_CALLS   default 3       – crop budget per question
    AGENTIC_MAX_CORRECTIONS  default 2       – unparseable-response retries
    AGENTIC_THINKING_BUDGET  optional override; falls back to THINKING_BUDGET, then 4096
    AGENTIC_TEMPERATURE      default 0.0     – sampling temperature
    REASONING_EFFORT         default xhigh   – passed via chat_template_kwargs
    JUDGE_TEMPERATURE        default 0.0     – judge sampling temperature
    JUDGE_MODEL / JUDGE_BASE_URL / JUDGE_API_KEY
                             no built-in default; must be set via env
                             (see JUDGE_DEFAULT_* constants; env overrides win)
    JUDGE2_MODEL / JUDGE2_TEMPERATURE        – optional second judge (audit)
    GOLD_PASS / GOLD_PASS_SAMPLE             – judge-vs-reference self check
    DOUBLE_JUDGE_SAMPLE      default 300     – random subset re-judged (0=off)
    LIMIT                    default 0       – only evaluate first N records
    USE_HINT                 default 0       – append the dataset ``hint`` field
    JUDGE_QPM / JUDGE_RPS    default 0       – rate-limit judge calls (0=off)
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import secrets
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

# ---------------------------------------------------------------------------
# Path bootstrap so that local imports work from any working directory.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_EVAL_DIR = os.path.join(_HERE, "eval")
_MODELS_DIR = os.path.join(_HERE, "models")
for _p in (_MODELS_DIR, _EVAL_DIR, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from artifacts import prediction_basename, shard_tag as prediction_shard_tag

# Judge decode helpers shared with the bare-ask evaluator.
from eval import (                                                  # noqa: E402
    is_failed,
    normalize_escape,
)

# Reusable components of the agentic crop/zoom-in protocol.
from agentic_helpers import (                                       # noqa: E402
    ACTION_BLOCK_RE,
    DEFAULT_CROP_MAX_DIMENSION,
    DEFAULT_CROP_MAX_PIXELS,
    DEFAULT_CROP_UPSCALE_FACTOR,
    MAX_CROP_IMAGE_CALLS,
    MAX_CROP_IMAGE_CALLS_PER_TURN,
    SlidingWindowRateLimiter,
    encode_image,
    image_size,
    _json_dumps_two_decimal_floats,
    _normalize_crop_arguments,
    _parse_arguments,
    _parse_single_text_action,
    _parse_xml_tool_action,
    _registered_path,
    _safe_task_component,
)

try:
    from tqdm import tqdm as _tqdm
except ImportError:                                                 # pragma: no cover
    def _tqdm(iterable, **_kw):
        return iterable

from openai import OpenAI                                           # noqa: E402


# ---------------------------------------------------------------------------
# Local chat_with_retry (non-streaming counterpart of eval.py's
# stream_predict_with_retry, with identical backoff / fallback semantics).
# ---------------------------------------------------------------------------
_CHAT_MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))


class _StreamMessage:
    __slots__ = ("content", "reasoning_content", "reasoning")

    def __init__(self, content, reasoning):
        self.content = content
        self.reasoning_content = reasoning
        self.reasoning = reasoning


class _StreamChoice:
    __slots__ = ("message", "finish_reason")

    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason


class _StreamCompletion:
    """Minimal duck-type of ``ChatCompletion`` assembled from streamed chunks."""

    __slots__ = ("choices", "usage")

    def __init__(self, choices, usage):
        self.choices = choices
        self.usage = usage


def chat_with_retry(client, max_retries: int = _CHAT_MAX_RETRIES, **kw):
    """chat.completions.create over a streaming connection, with retries.

    Streaming keeps bytes flowing while the model thinks; non-streaming
    requests can sit idle for minutes on long thinking traces and get killed
    by gateway idle timeouts (observed as HTTP 504 on some endpoints).
    Returns a minimal duck-typed completion exposing
    ``choices[0].message.content`` / ``reasoning_content`` and ``usage``.
    ``max_retries=-1`` retries forever; the same
    max_tokens -> max_completion_tokens fallback as
    ``eval.stream_predict_with_retry`` is applied once when the endpoint
    demands it.
    """
    last, attempt, fallback_done = None, 0, False
    while True:
        try:
            content_parts, reasoning_parts = [], []
            finish_reason, usage = None, None
            stream = client.chat.completions.create(**kw, stream=True)
            for chunk in stream:
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                if not getattr(chunk, "choices", None):
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if getattr(delta, "content", None):
                    content_parts.append(delta.content)
                reasoning = getattr(delta, "reasoning_content", None) or getattr(
                    delta, "reasoning", None
                )
                if reasoning:
                    reasoning_parts.append(reasoning)
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
            message = _StreamMessage(
                "".join(content_parts), "".join(reasoning_parts) or None
            )
            return _StreamCompletion([_StreamChoice(message, finish_reason)], usage)
        except Exception as e:  # noqa: BLE001
            err = str(e).lower()
            if (
                not fallback_done
                and "max_tokens" in err
                and "max_completion_tokens" in err
            ):
                kw["max_completion_tokens"] = kw.pop("max_tokens")
                fallback_done = True
                continue
            last = e
            if max_retries >= 0 and attempt >= max_retries:
                raise last
            time.sleep(min(2 ** min(attempt, 3), 8))
            attempt += 1


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
MAX_TOOL_CALLS = int(os.getenv("AGENTIC_MAX_TOOL_CALLS", str(MAX_CROP_IMAGE_CALLS)))
MAX_ACTION_CORRECTIONS = int(os.getenv("AGENTIC_MAX_CORRECTIONS", "2"))
THINKING_BUDGET = int(os.getenv("AGENTIC_THINKING_BUDGET") or os.getenv("THINKING_BUDGET") or "4096")
AGENTIC_TEMPERATURE = float(os.getenv("AGENTIC_TEMPERATURE", "0.0"))
PRED_TEMPERATURE = os.getenv("PRED_TEMPERATURE")
TOP_P = os.getenv("TOP_P")
TOP_K = os.getenv("TOP_K")
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "16384"))
# Send reasoning_effort through chat_template_kwargs; vLLM rejects top-level xhigh.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "xhigh")

# Configure the judge separately through JUDGE_MODEL / JUDGE_BASE_URL / JUDGE_API_KEY.
# No judge defaults: missing settings fail at judging. Different judges are not interchangeable.
JUDGE_DEFAULT_MODEL = os.getenv("JUDGE_DEFAULT_MODEL")
JUDGE_DEFAULT_BASE_URL = os.getenv("JUDGE_DEFAULT_BASE_URL")
JUDGE_DEFAULT_API_KEY = os.getenv("JUDGE_DEFAULT_API_KEY")
DEFAULT_API_IMAGE_QUALITY = int(os.getenv("API_IMAGE_QUALITY", "90"))

_JUDGE_TEMPLATE_PATH = os.path.join(_EVAL_DIR, "judge_prompt_zoom_in.txt")
with open(_JUDGE_TEMPLATE_PATH, encoding="utf-8") as _fh:
    JUDGE_ZOOM_IN_TEMPLATE = _fh.read()

# Whether the dataset ``hint`` field is appended to the instruction (default off
# to preserve the paper's protocol; enable with USE_HINT=1).
_USE_HINT = os.getenv("USE_HINT", "0") == "1"

# Regexes used by the QA answer parser
ANSWER_TAG_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.DOTALL | re.IGNORECASE,
)
ANSWER_PREFIX_RE = re.compile(
    r"^(?:final\s+)?answer\s*[:=]\s*",
    re.IGNORECASE,
)
ANSWER_LABEL_RE = re.compile(
    r"(?:^|\n)\s*(?:final\s+)?answer\s*[:=]\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)
THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)
THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Crop-image and system prompt for QA mode
# ---------------------------------------------------------------------------
QA_CROP_IMAGE_DESCRIPTION = (
    "Crop an image that has already been supplied in this conversation. "
    "image_path must be the exact original path or a path returned by an "
    "earlier crop_image call.  x1, y1, x2, y2 are absolute pixel coordinates "
    "in that referenced image's LOCAL coordinate system, with (x1,y1) at the "
    "top-left and (x2,y2) at the bottom-right.  The result provides a new "
    "image path and dimensions.  Returned crops are resized for inspection, "
    "so nested crops must use the returned width and height.  Crops may be "
    "nested.  When writing a crop_image action, format x1, y1, x2, and y2 "
    "with exactly two digits after the decimal point.  Use crop_image to "
    "zoom into the smallest detail (small text, fine visual attributes, "
    "chart labels, maps) before answering."
)

QA_SYSTEM_PROMPT_TEMPLATE = """You are an agentic high-resolution image question-answering model.

Task instruction:
{instruction}

Original image metadata:
{images_metadata}

Goal:
Answer the question about the image(s) concisely and correctly.  Use crop_image
when small text, fine visual attributes, charts, maps, or comparisons cannot be
resolved confidently from the full image.  The final answer text inside the
<answer> tag must directly answer the question — do not restate the question or
include reasoning inside the tag.  If the question asks for a number, output
only the number.  If it asks for a short text, output only that text.

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
  <answer>your concise answer text</answer>

crop_image is a textual action interpreted locally after your answer.  It is
NOT an API function call.  Do not emit JSON, XML, or native tool_calls.  Keep
x1, y1, x2, and y2 as four separately named numeric arguments.

Tool rules:
1. Use crop_image when small text, fine visual attributes, charts, maps, or
   comparisons cannot be resolved confidently from the full image.  Return the
   answer directly when cropping cannot reasonably help.
2. You may make at most {max_tool_calls} crop_image calls in total.  A single
   assistant turn may contain one or two crop_image actions, and every action
   counts separately.  Never mix crop actions with a final answer.
3. Tool arguments are local absolute pixels in the image_path passed to that
   call.  Write x1, y1, x2, and y2 with exactly two digits after the decimal
   point, including trailing zeros (for example, x1=100.00).  Two actions
   emitted in the same turn must refer only to image paths already available
   before that turn; the second action cannot depend on the result of the
   first action from the same turn.
4. Each result states the crop's region in the original image's local pixel
   coordinates and lists the returned crop dimensions.  Use those dimensions
   when nesting further crops.
5. After crop_image actions, the following single user message contains all
   result metadata and all successful crop images in action order.

Final-answer rule:
When ready, output exactly <think>...</think> followed by a newline and
<answer>concise answer text</answer>.  The text inside the tag is the only
content used for grading, so keep it factual and to the point.  Do not include
prose, Markdown, coordinates, or an action after the answer tag.  Do not add
"Answer:" or "Final answer:" labels inside the tag.
"""

QA_DIRECT_THINK_PROMPT_TEMPLATE = (
    "{instruction}\n\nFirst reason carefully about the image and the question "
    "inside <think>...</think>. Then output exactly one final answer using the "
    "format <answer>your concise answer text</answer>. Do not use tools or "
    "request crops."
)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------
def decode_data_uri_to_image(uri: str) -> Image.Image:
    """Decode a ``data:image/...;base64,...`` URI into a PIL Image."""
    if not uri.startswith("data:"):
        raise ValueError("Expected a data: URI")
    header, b64 = uri.split(",", 1)
    image_bytes = base64.b64decode(b64)
    return Image.open(BytesIO(image_bytes))


def save_image_to_temp(
    image: Image.Image,
    *,
    directory: Path,
    file_name: str,
) -> str:
    """Save *image* into *directory* as PNG and return the absolute path."""
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / file_name
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        rgb = Image.new("RGB", rgba.size, "white")
        rgb.paste(rgba, mask=rgba.getchannel("A"))
    else:
        rgb = image.convert("RGB") if image.mode != "RGB" else image
    rgb.save(output_path, format="PNG")
    return str(output_path.resolve())


def encode_from_path(path: str, *, jpeg_quality: int = DEFAULT_API_IMAGE_QUALITY) -> str:
    """Encode a local image file as a JPEG data URI (reuses agentic_helpers encoder)."""
    return encode_image(path, jpeg_quality=jpeg_quality)


# ---------------------------------------------------------------------------
# QA answer / action parsing
# ---------------------------------------------------------------------------
def parse_qa_answer(text: str | None) -> str | None:
    """Extract the final free-form QA answer from the model output.

    Resolution order:
      1. Content of the last ``<answer>...</answer>`` tag.
      2. Non-empty text after the last ``</think>``.
      3. The value of a trailing ``Answer: X`` / ``Final answer: X`` label.
      4. The full stripped text as a last resort.
    """
    value = (text or "").strip()
    if not value:
        return None

    matches = list(ANSWER_TAG_RE.finditer(value))
    if matches:
        answer = matches[-1].group(1).strip()
        answer = ANSWER_PREFIX_RE.sub("", answer).strip()
        return answer or None

    think_closes = list(THINK_CLOSE_RE.finditer(value))
    if think_closes:
        suffix = value[think_closes[-1].end():].strip()
        if suffix:
            return ANSWER_PREFIX_RE.sub("", suffix).strip() or None

    label_match = ANSWER_LABEL_RE.search(value)
    if label_match:
        answer = label_match.group(1).strip()
        return answer or None

    # Fall back to the whole text (some models emit no reasoning at all).
    return value


def _parse_crop_actions_from_blocks(
    value: str,
    action_matches: list[re.Match[str]],
) -> dict[str, Any] | None:
    """Convert ``<action>`` blocks into crop action dictionaries."""
    if len(action_matches) > MAX_CROP_IMAGE_CALLS_PER_TURN:
        return None
    actions: list[dict[str, Any]] = []
    for match in action_matches:
        action = _parse_single_text_action(match.group(1).strip(), answer_mode="point")
        if action is None or action["type"] != "crop":
            return None
        actions.append(action)
    if not actions:
        return None
    if len(actions) == 1:
        return actions[0]
    return {
        "type": "crop_batch",
        "name": "crop_image",
        "actions": actions,
        "source": value.strip(),
        "call_count": len(actions),
    }


def parse_qa_text_action(text: str | None) -> dict[str, Any] | None:
    """Parse one final QA answer, one crop action, or a batch of crop actions.

    Returns a dict shaped like the actions produced by
    ``agentic_helpers.parse_text_action`` but for the QA answer mode:

    - ``{"type": "final", "name": "Answer", "answer": str, ...}``
    - ``{"type": "crop", "name": "crop_image", "arguments": dict, ...}``
    - ``{"type": "crop_batch", "name": "crop_image", "actions": [...], ...}``
    - ``None`` if the content cannot be interpreted.
    """
    value = text or ""

    # 1. Explicit <answer>...</answer> tag → final answer.
    answer_matches = list(ANSWER_TAG_RE.finditer(value))
    if answer_matches:
        # Refuse to mix crop actions with a final answer.
        if ACTION_BLOCK_RE.search(value):
            return None
        answer = answer_matches[-1].group(1).strip()
        answer = ANSWER_PREFIX_RE.sub("", answer).strip()
        if not answer:
            return None
        return {
            "type": "final",
            "name": "Answer",
            "answer": answer,
            "source": value.strip(),
            "compatibility_format": "answer_tag",
        }

    # 2. Content after </think>: might be an answer or crop actions.
    think_closes = list(THINK_CLOSE_RE.finditer(value))
    if think_closes:
        suffix = value[think_closes[-1].end():]
        action_blocks_in_suffix = list(ACTION_BLOCK_RE.finditer(suffix))
        if action_blocks_in_suffix:
            return _parse_crop_actions_from_blocks(value, action_blocks_in_suffix)
        suffix_text = suffix.strip()
        if suffix_text:
            answer = ANSWER_PREFIX_RE.sub("", suffix_text).strip()
            if answer:
                return {
                    "type": "final",
                    "name": "Answer",
                    "answer": answer,
                    "source": value.strip(),
                    "compatibility_format": "answer_after_think",
                }

    # 3. Action blocks anywhere in the text → crops.
    action_blocks = list(ACTION_BLOCK_RE.finditer(value))
    if action_blocks:
        return _parse_crop_actions_from_blocks(value, action_blocks)

    # 4. Qwen-style XML tool action (legacy compatibility).
    xml_action = _parse_xml_tool_action(value)
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
                "source": value.strip(),
                "compatibility_format": "qwen_xml",
            }

    # 5. Try bare function call (model omitted the <action> wrapper).
    # Use "point" answer_mode so the parser doesn't look for MCQ letters; only
    # accept crop-type results since coordinates are not valid QA answers.
    single = _parse_single_text_action(value, answer_mode="point")
    if single is not None and single.get("type") == "crop":
        return single

    # 6. Open/unclosed followed by bare answer text.
    think_opens = list(THINK_OPEN_RE.finditer(value))
    if think_opens and not think_closes:
        pass

    # 7. Fallback: let the QA answer extractor try the raw text.
    answer = parse_qa_answer(value)
    if answer:
        return {
            "type": "final",
            "name": "Answer",
            "answer": answer,
            "source": value.strip(),
            "compatibility_format": "fallback",
        }
    return None


# ---------------------------------------------------------------------------
# Agentic QA pipeline
# ---------------------------------------------------------------------------
class AgenticQAPipeline:
    """Multi-turn agentic QA agent with locally interpreted crop_image actions.

    The pipeline mirrors the grounding pipeline (``models/agentic_pipeline.py``) but targets
    free-form visual question answering rather than GUI grounding.  It uses an
    OpenAI-compatible client (the same one as ``eval/eval.py``) instead of a
    raw httpx streaming client, so the configuration story is identical to the
    local ``eval.py`` runner.
    """

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        judge_model: str,
        crop_root: str | Path = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = AGENTIC_TEMPERATURE,
        max_tool_calls: int = MAX_TOOL_CALLS,
        max_action_corrections: int = MAX_ACTION_CORRECTIONS,
        thinking_budget: int = THINKING_BUDGET,
        api_image_quality: int = DEFAULT_API_IMAGE_QUALITY,
        crop_upscale_factor: float = DEFAULT_CROP_UPSCALE_FACTOR,
        crop_max_pixels: int = DEFAULT_CROP_MAX_PIXELS,
        crop_max_dimension: int = DEFAULT_CROP_MAX_DIMENSION,
        max_retries: int | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.judge_model = judge_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        # Mirror agentic_helpers' hard cap so users can't accidentally raise it.
        self.max_tool_calls = min(max_tool_calls, MAX_CROP_IMAGE_CALLS)
        self.max_action_corrections = max(0, max_action_corrections)
        # Optional grace mode preserves the advertised training crop budget,
        # but permits one additional crop (up to four) during evaluation and
        # one final-answer-only turn. Use only for separately labeled recovery runs.
        # Disabled by default; the standard evaluation behavior is unchanged.
        self.grace_mode = os.getenv("AGENTIC_GRACE_MODE", "0") == "1"
        self.grace_tool_calls = 1 if self.grace_mode else 0
        self.thinking_budget = max(1, thinking_budget)
        self.api_image_quality = api_image_quality
        self.crop_upscale_factor = crop_upscale_factor
        self.crop_max_pixels = crop_max_pixels
        self.crop_max_dimension = crop_max_dimension
        self.max_retries = max_retries
        # Use the OS temp dir so we don't pollute the repository tree and crops
        # survive until the process exits.
        self.crop_root = Path(crop_root).expanduser().resolve() if crop_root else (
            Path(tempfile.gettempdir()) / "perceptionbench_agentic_crops"
        )
        self._lock = threading.Lock()
        self._crop_counter = 0

    # ------------------------------------------------------------------ utils
    def _next_crop_index(self) -> int:
        with self._lock:
            self._crop_counter += 1
            return self._crop_counter

    @staticmethod
    def _normalize_answer_text(value: str) -> str:
        """Collapse whitespace for compact final answers."""
        return re.sub(r"\s+", " ", (value or "").strip())

    # ------------------------------------------------------------ prompt build
    def build_system_prompt(
        self,
        instruction: str,
        images_info: list[dict[str, Any]],
    ) -> str:
        metadata_lines = []
        for position, info in enumerate(images_info, start=1):
            metadata_lines.append(
                f"- image_path: {info['path']}\n"
                f"  index: {position}\n"
                f"  resolution: {info['width']}x{info['height']} pixels "
                "(width x height)\n"
                f"  origin: [0,0] at the top-left"
            )
        images_metadata = "\n".join(metadata_lines) if metadata_lines else "(no images)"
        return QA_SYSTEM_PROMPT_TEMPLATE.format(
            instruction=instruction.rstrip().rstrip("."),
            images_metadata=images_metadata,
            crop_description=QA_CROP_IMAGE_DESCRIPTION,
            max_tool_calls=self.max_tool_calls,
        )

    def build_initial_user_message(
        self,
        images_info: list[dict[str, Any]],
        *,
        encode: bool,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for position, info in enumerate(images_info, start=1):
            content.append({
                "type": "text",
                "text": f"image_{position}_path: {info['path']}",
            })
            image_url: dict[str, Any]
            if encode:
                image_url = {
                    "url": encode_from_path(
                        info["path"], jpeg_quality=self.api_image_quality
                    )
                }
            else:
                image_url = {"path": info["path"]}
            content.append({"type": "image_url", "image_url": image_url})
        return {"role": "user", "content": content}

    @staticmethod
    def text_user_message(text: str) -> dict[str, Any]:
        return {"role": "user", "content": text}

    def crop_results_user_message(
        self,
        events: list[dict[str, Any]],
        *,
        encode: bool,
    ) -> dict[str, Any]:
        """Build a user message delivering crop results and crop images.

        This is a QA-flavoured counterpart of
        ``AgenticGroundingPipeline.crop_results_user_message``.
        """
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
                        "url": encode_from_path(
                            image_path, jpeg_quality=self.api_image_quality
                        )
                    }
                    if encode
                    else {"path": image_path}
                )
                content.append({"type": "image_url", "image_url": image_url})

        content.append(
            {
                "type": "text",
                "text": (
                    "Inspect all successful crops and account for any failed actions. "
                    "Reply with <think>...</think>, then either one or two independent "
                    "crop_image <action> blocks whose coordinates have exactly two "
                    "decimal places, or the final <answer>concise answer text"
                    "</answer>. Do not mix crop actions with a final answer."
                ),
            }
        )
        return {"role": "user", "content": content}

    # ------------------------------------------------------------ crop exec
    def _execute_crop(
        self,
        raw_arguments: Any,
        registry: dict[str, dict[str, Any]],
        crop_dir: Path,
        crop_index: int,
    ) -> dict[str, Any]:
        """Execute a single crop_image text action.

        Mirrors ``AgenticGroundingPipeline._execute_crop`` but is independent of
        the grounding pipeline so it can be used standalone.
        """
        try:
            arguments = _parse_arguments(raw_arguments) if not isinstance(raw_arguments, dict) else dict(raw_arguments)
            source_path = _registered_path(str(arguments["image_path"]))
            if source_path not in registry:
                raise ValueError(
                    "image_path is not the original image or a path returned by crop_image"
                )
            source = registry[source_path]
            values = [float(arguments[key]) for key in ("x1", "y1", "x2", "y2")]
            if not all(math.isfinite(v) for v in values):
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
            output_path = crop_dir / f"crop_{crop_index:04d}.png"
            with Image.open(source_path) as image:
                cropped = image.crop((left, top, right, bottom))
                crop_width, crop_height = cropped.size
                resize_scale = min(
                    self.crop_upscale_factor,
                    math.sqrt(self.crop_max_pixels / max(1, crop_width * crop_height)),
                    self.crop_max_dimension / max(1, crop_width),
                    self.crop_max_dimension / max(1, crop_height),
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
                        output_width = max(1, self.crop_max_pixels // output_height)
                    else:
                        output_height = max(1, self.crop_max_pixels // output_width)
                if (output_width, output_height) != cropped.size:
                    cropped = cropped.resize(
                        (output_width, output_height),
                        resample=Image.Resampling.LANCZOS,
                    )
                cropped.save(output_path, format="PNG")

            original_region = list(source["original_region"])
            parent_scale_x = (original_region[2] - original_region[0]) / source["width"]
            parent_scale_y = (original_region[3] - original_region[1]) / source["height"]
            mapped_region = [
                original_region[0] + left * parent_scale_x,
                original_region[1] + top * parent_scale_y,
                original_region[0] + right * parent_scale_x,
                original_region[1] + bottom * parent_scale_y,
            ]
            output_key = _registered_path(str(output_path))
            registry[output_key] = {
                "path": output_key,
                "width": output_width,
                "height": output_height,
                "original_region": mapped_region,
                "original_width": source["original_width"],
                "original_height": source["original_height"],
            }
            return {
                "ok": True,
                "image_path": output_key,
                "width": output_width,
                "height": output_height,
                "crop_in_parent": [left, top, right, bottom],
                "crop_size_before_resize": [crop_width, crop_height],
                "resize_scale": round(resize_scale, 2),
                "original_image_region_pixels": [round(v, 2) for v in mapped_region],
                "reminder": (
                    "Use the inspected evidence to return the final answer inside"
                    " <answer>...</answer>."
                ),
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------ model call
    def _chat_once(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
    ) -> tuple[str | None, str | None, Any]:
        """Send one chat completion request and return (content, error, raw)."""
        kw: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        sampling_temp = (
            temperature if temperature is not None else self.temperature
        )
        if sampling_temp is not None:
            kw["temperature"] = float(sampling_temp)
        if PRED_TEMPERATURE is not None:
            kw["temperature"] = float(PRED_TEMPERATURE)
        if TOP_P is not None:
            kw["top_p"] = float(TOP_P)
        extra_body: dict[str, Any] = {}
        if TOP_K is not None:
            extra_body["top_k"] = int(TOP_K)
        # Enable thinking for the vLLM endpoint when a budget is configured;
        # Send reasoning_effort only through chat_template_kwargs.
        if self.thinking_budget > 0:
            extra_body["chat_template_kwargs"] = {
                "thinking": True,
                "enable_thinking": True,
                "reasoning_effort": REASONING_EFFORT,
                "max_thinking_tokens": self.thinking_budget,
            }
        if extra_body:
            kw["extra_body"] = extra_body

        retry_kwargs = {}
        if self.max_retries is not None:
            retry_kwargs["max_retries"] = self.max_retries
        raw = chat_with_retry(self.client, **retry_kwargs, **kw)
        msg = raw.choices[0].message
        content = msg.content or ""
        # Some endpoints expose reasoning_content as a separate attribute; merge
        # it into the visible content for downstream parsing if content is empty.
        if not content.strip():
            reasoning = getattr(msg, "reasoning_content", None) or getattr(
                msg, "reasoning", None
            )
            if reasoning:
                content = str(reasoning)
        return content, None, raw

    # ----------------------------------------------------------- full trajectory
    def run_trajectory(
        self,
        problem: str,
        image_paths: list[str],
        *,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the agentic multi-turn trajectory for one record.

        Returns a trajectory dict similar to the grounding pipeline's output but keyed
        to the QA answer mode.  The most important fields are ``prediction``
        (extracted answer string) and ``error`` (``None`` on success).
        """
        # Normalize and prepare all images.
        resolved_paths = [_registered_path(p) for p in image_paths]
        for p in resolved_paths:
            if not Path(p).is_file():
                raise FileNotFoundError(p)

        images_info: list[dict[str, Any]] = []
        for path in resolved_paths:
            w, h = image_size(path)
            images_info.append({
                "path": path,
                "width": w,
                "height": h,
                "original_region": [0.0, 0.0, float(w), float(h)],
                "original_width": w,
                "original_height": h,
            })

        registry = {info["path"]: info for info in images_info}
        resolved_task_id = task_id or secrets.token_hex(8)
        conversation_id = secrets.token_hex(8)
        crop_dir = (
            self.crop_root
            / _safe_task_component(resolved_task_id)
            / f"conversation_{conversation_id}"
        )

        system_prompt = self.build_system_prompt(problem, images_info)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            self.build_initial_user_message(images_info, encode=True),
        ]

        rounds: list[dict[str, Any]] = []
        tool_events: list[dict[str, Any]] = []
        tool_count = 0
        tool_attempt_count = 0
        action_correction_count = 0
        final_content = ""
        final_answer: str | None = None
        conversation_error: str | None = None
        max_model_turns = self.max_tool_calls + self.max_action_corrections + 1
        if self.grace_mode:
            # Grace mode reserves two extra turns for an additional crop and final answer.
            max_model_turns += 2

        try:
            for model_turn in range(1, max_model_turns + 1):
                content, error, raw = self._chat_once(messages)
                rounds.append({
                    "model_turn": model_turn,
                    "response_content": content,
                    "error": error,
                })
                if error is not None:
                    conversation_error = error
                    break
                if not content:
                    conversation_error = "Empty response from model"
                    break

                parsed_action = parse_qa_text_action(content)
                round_record = {
                    "model_turn": model_turn,
                    "content": content,
                    "parsed_action": parsed_action,
                    "error": error,
                }
                rounds[-1] = round_record

                if parsed_action is None:
                    final_content = content
                    if action_correction_count >= self.max_action_corrections:
                        conversation_error = (
                            "Model did not return a parseable crop action or final "
                            f"answer after {action_correction_count + 1} attempts"
                        )
                        break
                    action_correction_count += 1
                    messages.append({"role": "assistant", "content": content})
                    correction = self.text_user_message(
                        "Your previous answer was not valid. Reply with "
                        "<think>...</think> followed by either one or two independent crop "
                        "actions, each in its own "
                        "<action>crop_image(...)</action> block, or the final "
                        "<answer>concise answer text</answer>. Do not emit more "
                        "than two actions or mix actions with a final answer. "
                        "Every crop coordinate value must have exactly two digits "
                        "after the decimal point."
                    )
                    messages.append(correction)
                    continue

                if parsed_action["type"] == "final":
                    final_content = content
                    final_answer = parsed_action.get("answer")
                    break

                # Crop / crop_batch path
                messages.append({"role": "assistant", "content": content})
                if parsed_action.get("type") == "crop_batch":
                    crop_actions = list(parsed_action["actions"])
                else:
                    crop_actions = [parsed_action]

                batch_events: list[dict[str, Any]] = []
                for crop_action in crop_actions:
                    tool_attempt_count += 1
                    arguments = dict(crop_action.get("arguments") or {})
                    if "image_path" not in arguments:
                        arguments["image_path"] = images_info[0]["path"]
                    budget_index: int | None = None
                    if tool_count >= self.max_tool_calls + self.grace_tool_calls:
                        result = {
                            "ok": False,
                            "error": (
                                "Crop budget exhausted: maximum is "
                                f"{self.max_tool_calls}. Return "
                                "<think>...</think> followed by the final "
                                "<answer>concise answer text</answer> now."
                            ),
                        }
                    else:
                        tool_count += 1
                        budget_index = tool_count
                        crop_index = self._next_crop_index()
                        result = self._execute_crop(
                            arguments,
                            registry,
                            crop_dir,
                            crop_index,
                        )
                    event = {
                        "tool_call_index": tool_attempt_count,
                        "tool_budget_index": budget_index,
                        "model_turn": model_turn,
                        "protocol": "text_action",
                        "name": "crop_image",
                        "raw_action": crop_action.get("source"),
                        "arguments": arguments,
                        "result": result,
                    }
                    tool_events.append(event)
                    batch_events.append(event)

                crop_message = self.crop_results_user_message(
                    batch_events, encode=True
                )
                messages.append(crop_message)
            else:
                conversation_error = "Conversation exhausted the model-turn limit"
        finally:
            # Keep crops on disk so debugging is possible; the OS temp cleaner
            # handles long-term cleanup.  Callers that want eager removal can
            # delete ``crop_dir`` afterwards using ``crop_directory`` below.
            pass

        # Grace-mode recovery: request a final answer when the budget is exhausted.
        # Further crops are forbidden; a parseable <answer> counts as recovered.
        if (
            self.grace_mode
            and final_answer is None
            and conversation_error != "Empty response from model"
        ):
            messages.append(
                self.text_user_message(
                    "Tool budget is fully used and no more crop actions are "
                    "allowed. You must answer right now based on the images "
                    "already available. Reply with a short <think>...</think> "
                    "followed by exactly one final <answer>concise answer "
                    "text</answer>. Do not emit any crop_image action."
                )
            )
            content, error, raw = self._chat_once(messages)
            if error is None and content:
                parsed_action = parse_qa_text_action(content)
                rounds.append(
                    {
                        "model_turn": model_turn + 1,
                        "content": content,
                        "parsed_action": parsed_action,
                        "error": error,
                        "grace_forced_final": True,
                    }
                )
                if parsed_action is not None and parsed_action.get("type") == "final":
                    final_content = content
                    final_answer = parsed_action.get("answer")
                    conversation_error = None

        prediction = self._normalize_answer_text(final_answer) if final_answer else None
        crop_paths = [
            ev["result"]["image_path"]
            for ev in tool_events
            if ev["result"].get("ok")
        ]
        return {
            "task_id": resolved_task_id,
            "conversation_id": conversation_id,
            "model": self.model,
            "instruction": problem,
            "original_image_paths": resolved_paths,
            "images_width": [info["width"] for info in images_info],
            "images_height": [info["height"] for info in images_info],
            "answer_mode": "qa",
            "system_prompt": system_prompt,
            "messages": messages,
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
            "prediction": prediction,
            "error": conversation_error,
        }


# ---------------------------------------------------------------------------
# Judge wrapper using judge_prompt_zoom_in.txt (does NOT modify eval.py)
# ---------------------------------------------------------------------------
# Robust verdict extraction.  Accepted forms:
#   [reason] <text> [judge] True|False     (bracketed, case/whitespace tolerant)
#   {"judge": true/false, "reason": "..."} (JSON object)
#   <any text> True|False                  (trailing bare boolean)
JUDGE_VERDICT_RE = re.compile(
    r"\[\s*judge\s*\]\s*[:\-]?\s*(true|false)\b", re.IGNORECASE
)
JUDGE_REASON_RE = re.compile(
    r"\[\s*reason\s*\]\s*(.*?)(?=\[\s*judge\s*\]|\s*$)",
    re.DOTALL | re.IGNORECASE,
)
TRAILING_BOOL_RE = re.compile(r"\b(true|false)\b[.\s]*$", re.IGNORECASE)


def decode_judge_robust(resp: Any) -> tuple[bool | None, str, str]:
    """Return ``(verdict, reason, status)``.

    ``verdict`` is ``True``/``False`` when the output could be decoded and
    ``None`` otherwise.  ``status`` is ``"ok"`` or ``"parse_error"``; a
    ``parse_error`` is a judge-side failure and must never be conflated with a
    model error (callers record it under ``judge_status``).
    """
    text = str(resp or "").strip()
    if not text:
        return None, "", "parse_error"

    match = JUDGE_VERDICT_RE.search(text)
    if match:
        verdict = match.group(1).lower() == "true"
        reason_match = JUDGE_REASON_RE.search(text)
        reason = reason_match.group(1).strip() if reason_match else text[:300]
        return verdict, reason[:300], "ok"

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("judge"), bool):
            reason = str(obj.get("reason") or "").strip()
            return obj["judge"], reason[:300], "ok"
    except (json.JSONDecodeError, TypeError):
        pass

    trailing = TRAILING_BOOL_RE.search(text)
    if trailing:
        return trailing.group(1).lower() == "true", text[:300], "ok"

    return None, text[:300], "parse_error"


def judge_general_qa_zoom_in_robust(
    client: OpenAI,
    judge_model: str,
    question: str,
    pred: str,
    reference: Any,
    *,
    template: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    parse_retries: int = 1,
    max_retries: int | None = None,
    limiter: SlidingWindowRateLimiter | None = None,
) -> dict[str, Any]:
    """Judge *pred* against *reference* using the zoom-in template.

    Returns ``{"verdict": bool|None, "reason": str, "status": str, "raw": str}``
    where ``status`` is one of ``"ok" | "parse_error" | "judge_error"``.

    * ``parse_error``: the judge returned text that could not be decoded; it is
      retried once with an explicit format reminder before being marked.
    * ``judge_error``: the API call failed even after ``chat_with_retry``.
    * ``verdict`` is ``None`` for both failure statuses, so callers can keep
      judge-side failures out of the accuracy denominator.
    """
    ref = str(reference if reference is not None else "").strip()
    tpl = template or JUDGE_ZOOM_IN_TEMPLATE
    prompt = (
        tpl.replace("{problem}", normalize_escape(str(question)).strip())
        .replace("{reference_answer}", normalize_escape(ref).strip())
        .replace("{assistant_answer}", normalize_escape(str(pred)).strip())
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    last_exc: Exception | None = None

    for attempt in range(parse_retries + 1):
        try:
            if limiter is not None:
                limiter.acquire()
            kw: dict[str, Any] = {
                "model": judge_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            if max_retries is not None:
                kw["max_retries"] = max_retries
            try:
                r = chat_with_retry(client, **kw)
            except TypeError:
                # Older chat_with_retry signatures didn't accept max_retries.
                r = chat_with_retry(
                    client,
                    model=judge_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=messages,
                )
            content = r.choices[0].message.content or ""
            verdict, reason, status = decode_judge_robust(content)
            if status == "ok":
                return {
                    "verdict": verdict,
                    "reason": reason,
                    "status": "ok",
                    "raw": content,
                }
            if attempt < parse_retries:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": content or ""},
                    {
                        "role": "user",
                        "content": (
                            "Your previous output could not be parsed. Respond again "
                            "strictly in this format:\n"
                            "[reason]\n<brief justification, max 100 tokens>\n"
                            "[judge]\nTrue or False\n"
                            "Your output must end with exactly True or False."
                        ),
                    },
                ]
                continue
            return {
                "verdict": None,
                "reason": f"unparseable judge output: {content[:200]}",
                "status": "parse_error",
                "raw": content,
            }
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            break
    return {
        "verdict": None,
        "reason": f"judge API error: {last_exc}",
        "status": "judge_error",
        "raw": None,
    }


def judge_general_qa_zoom_in(
    client: OpenAI,
    judge_model: str,
    question: str,
    pred: str,
    reference: Any,
    *,
    max_retries: int | None = None,
) -> tuple[int, str]:
    """Compatibility wrapper mirroring ``eval.judge_general_qa``'s return shape.

    Uses temperature 0.3 (the paper's judge temperature) and maps any judge-side
    failure to ``(0, reason)`` exactly like ``eval.py``.
    """
    result = judge_general_qa_zoom_in_robust(
        client,
        judge_model,
        question,
        pred,
        reference,
        temperature=0.3,
        max_retries=max_retries,
    )
    if result["status"] != "ok":
        return 0, result["reason"]
    return (1 if result["verdict"] else 0), result["reason"]


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------
def prepare_record(
    rec: dict[str, Any],
    *,
    session_dir: Path,
    index_key: Any,
) -> tuple[str, list[str], Path, str]:
    """Decode record images to temp files; return (problem, image_paths, crop_dir, task_id)."""
    task_id = (
        f"idx_{rec.get('index', index_key)}"
        f"_{_safe_task_component(str(rec.get('source_bmk', 'NA')))}"
    )
    record_dir = session_dir / _safe_task_component(task_id)
    image_uris = rec.get("image") or []
    if isinstance(image_uris, str):
        image_uris = [image_uris]
    image_paths: list[str] = []
    for i, uri in enumerate(image_uris, start=1):
        if not isinstance(uri, str):
            continue
        # Some runners rewrite image entries to "[image saved to <path>]";
        # unwrap the marker so the underlying file path is used.
        marker = re.match(
            r"^\[\s*image\s+saved\s+to\s+(.+?)\s*\]$", uri, re.IGNORECASE | re.DOTALL
        )
        if marker:
            uri = marker.group(1).strip()
        if not uri.startswith("data:"):
            # Treat non-data URIs as already-existing paths.
            image_paths.append(_registered_path(uri))
            continue
        pil_image = decode_data_uri_to_image(uri)
        saved_path = save_image_to_temp(
            pil_image,
            directory=record_dir,
            file_name=f"image_{i:02d}.png",
        )
        image_paths.append(saved_path)
    problem = re.sub(
        r"<\|image_(\d+)\|>",
        lambda m: f"[image_{m.group(1)}]",
        str(rec.get("problem", "") or ""),
    )
    # Only strip a leading run of placeholder markers.  Inline references such
    # as "in image 2" are preserved so the model can map them to the image
    # metadata enumerated in the system prompt.
    problem = re.sub(r"^(?:\[image_\d+\]\s*)+", "", problem).strip()
    if _USE_HINT:
        hint = rec.get("hint")
        if isinstance(hint, str) and hint.strip():
            problem = f"{problem}\n\nHint: {hint.strip()}"
    return problem, image_paths, record_dir, task_id


def process_record(
    rec: dict[str, Any],
    pipeline: AgenticQAPipeline,
    *,
    session_dir: Path,
    index_key: Any,
    judge_client: OpenAI | None = None,
    judge2_client: OpenAI | None = None,
    judge2_model: str | None = None,
    judge2_temperature: float = 0.3,
    double_judge: bool = False,
    judge_temperature: float = 0.0,
    judge_parse_retries: int = 1,
    judge_max_retries: int | None = None,
    judge_limiter: SlidingWindowRateLimiter | None = None,
) -> dict[str, Any]:
    """Run one record through the agentic pipeline and judge the answer.

    The returned record carries a ``judge_status`` field that distinguishes
    model-side failures (``no_answer`` / ``runtime_error``) from judge-side
    failures (``parse_error`` / ``judge_error``).  Judge-side failures have
    ``judge_result=None`` so the aggregator can exclude them from the primary
    accuracy denominator instead of counting them as model errors.
    """
    try:
        problem, image_paths, crop_dir, task_id = prepare_record(
            rec, session_dir=session_dir, index_key=index_key
        )
        if not image_paths:
            return {
                "index": rec.get("index"),
                "error_category": rec.get("error_category"),
                "source_bmk": rec.get("source_bmk"),
                "prediction": "[error] no image in record",
                "judge_result": None,
                "judge_status": "runtime_error",
                "judge_reason": "no image available",
                "agentic": {"error": "no image available"},
            }
        trajectory = pipeline.run_trajectory(
            problem,
            image_paths,
            task_id=task_id,
        )
        pred = trajectory.get("prediction") or trajectory.get("final_content") or ""
        if trajectory.get("error") and not pred:
            pred = f"[error] {trajectory['error']}"
    except Exception as exc:  # noqa: BLE001
        return {
            "index": rec.get("index"),
            "error_category": rec.get("error_category"),
            "source_bmk": rec.get("source_bmk"),
            "prediction": f"[error] {type(exc).__name__}: {exc}",
            "judge_result": None,
            "judge_status": "runtime_error",
            "judge_reason": f"runtime error: {exc}",
            "agentic": {"error": f"runtime error: {exc}"},
        }

    base: dict[str, Any] = {
        "index": rec.get("index"),
        "error_category": rec.get("error_category"),
        "source_bmk": rec.get("source_bmk"),
        "prediction": pred,
        "final_answer": trajectory.get("prediction"),
        "agentic": {
            "model_turn_count": trajectory.get("model_turn_count"),
            "crop_action_count": trajectory.get("crop_action_count"),
            "crop_image_paths": trajectory.get("crop_image_paths"),
            "crop_directory": trajectory.get("crop_directory"),
            "error": trajectory.get("error"),
            # Text-only trajectory for inspection; base64 images and encoded messages are omitted.
            "instruction": trajectory.get("instruction"),
            "system_prompt": trajectory.get("system_prompt"),
            "rounds": trajectory.get("rounds"),
            "tool_events": trajectory.get("tool_events"),
            "final_content": trajectory.get("final_content"),
        },
    }

    if is_failed(pred):
        base.update({
            "judge_result": 0,
            "judge_status": "no_answer",
            "judge_reason": "failed to obtain answer",
        })
        return base

    if os.getenv("PB_SKIP_JUDGE") == "1":
        # Inference-only mode: retain predictions and trajectories for later judging.
        base.update({
            "judge_result": None,
            "judge_status": "skipped",
            "judge_reason": "judge disabled (PB_SKIP_JUDGE=1)",
        })
        return base

    gt = rec.get("answer")
    ref = gt.get("answer") if isinstance(gt, dict) else gt
    judge_result = judge_general_qa_zoom_in_robust(
        judge_client or pipeline.client,
        pipeline.judge_model,
        problem,
        pred,
        ref,
        temperature=judge_temperature,
        parse_retries=judge_parse_retries,
        max_retries=judge_max_retries,
        limiter=judge_limiter,
    )
    base["judge_result"] = (
        (1 if judge_result["verdict"] else 0)
        if judge_result["status"] == "ok"
        else None
    )
    base["judge_status"] = judge_result["status"]
    base["judge_reason"] = judge_result["reason"]

    if double_judge and judge2_client is not None:
        judge2_result = judge_general_qa_zoom_in_robust(
            judge2_client,
            judge2_model or pipeline.judge_model,
            problem,
            pred,
            ref,
            temperature=judge2_temperature,
            parse_retries=judge_parse_retries,
            max_retries=judge_max_retries,
            limiter=judge_limiter,
        )
        base["judge2_result"] = (
            (1 if judge2_result["verdict"] else 0)
            if judge2_result["status"] == "ok"
            else None
        )
        base["judge2_status"] = judge2_result["status"]
        base["judge2_reason"] = judge2_result["reason"]
    return base


def run_gold_pass(
    client: OpenAI,
    judge_model: str,
    data: list[dict[str, Any]],
    *,
    sample_size: int = 200,
    seed: int = 42,
    concurrency: int = 4,
    judge_temperature: float = 0.0,
    judge_parse_retries: int = 1,
    judge_max_retries: int | None = None,
    judge_limiter: SlidingWindowRateLimiter | None = None,
) -> dict[str, Any]:
    """Judge each reference answer against itself; all should be True.

    This is a cheap self-check of the judge harness: if a non-trivial fraction
    of gold (reference vs reference) judgments is not True, the judge prompt or
    the endpoint is misbehaving and the main results cannot be trusted.

    ``sample_size <= 0`` (or >= len(data)) checks the whole dataset.
    """
    import random

    rng = random.Random(seed)
    if sample_size <= 0 or sample_size >= len(data):
        items = list(data)
    else:
        items = rng.sample(data, sample_size)

    def check(rec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        gt = rec.get("answer")
        ref = gt.get("answer") if isinstance(gt, dict) else gt
        result = judge_general_qa_zoom_in_robust(
            client,
            judge_model,
            rec.get("problem", ""),
            ref,
            ref,
            temperature=judge_temperature,
            parse_retries=judge_parse_retries,
            max_retries=judge_max_retries,
            limiter=judge_limiter,
        )
        return rec, result

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        checks = list(
            _tqdm(executor.map(check, items), total=len(items), desc="Gold pass")
        )

    failures: list[dict[str, Any]] = []
    passed = 0
    judged_ok = 0
    for rec, result in checks:
        if result["status"] != "ok":
            failures.append({
                "index": rec.get("index"),
                "status": result["status"],
                "reason": result["reason"],
            })
            continue
        judged_ok += 1
        if result["verdict"] is True:
            passed += 1
        else:
            failures.append({
                "index": rec.get("index"),
                "status": "wrong_gold_verdict",
                "reason": result["reason"],
            })
    return {
        "sample_size": len(items),
        "passed": passed,
        "judged_ok": judged_ok,
        "failures": failures,
        "pass_rate": passed / len(items) if items else 0.0,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def _print_env_config() -> None:
    print("Agentic PerceptionBench eval configuration:")
    print(f"  MODEL               = {os.getenv('MODEL')!r}")
    print(f"  JUDGE_MODEL         = {os.getenv('JUDGE_MODEL', JUDGE_DEFAULT_MODEL)}")
    print(f"  JUDGE2_MODEL        = {os.getenv('JUDGE2_MODEL') or '(same as judge)'}")
    print(f"  JUDGE_TEMPERATURE   = {os.getenv('JUDGE_TEMPERATURE', '0.0')}")
    print(f"  JUDGE2_TEMPERATURE  = {os.getenv('JUDGE2_TEMPERATURE', '0.3')}")
    print(f"  DATASET             = {os.getenv('DATASET', 'PerceptionBench.jsonl')}")
    print(f"  CONCURRENCY         = {os.getenv('CONCURRENCY', '4')}")
    print(f"  MAX_TOKENS          = {DEFAULT_MAX_TOKENS}")
    print(f"  MAX_TOOL_CALLS      = {MAX_TOOL_CALLS}")
    print(f"  MAX_CORRECTIONS     = {MAX_ACTION_CORRECTIONS}")
    print(f"  THINKING_BUDGET     = {THINKING_BUDGET}")
    print(f"  TEMPERATURE         = {AGENTIC_TEMPERATURE}")
    print(f"  LIMIT               = {os.getenv('LIMIT', '0')}")
    print(f"  GOLD_PASS           = {os.getenv('GOLD_PASS', '1')}")
    print(f"  GOLD_PASS_SAMPLE    = {os.getenv('GOLD_PASS_SAMPLE', '200')}")
    print(f"  DOUBLE_JUDGE_SAMPLE = {os.getenv('DOUBLE_JUDGE_SAMPLE', '300')}")
    if os.getenv("PB_SKIP_JUDGE") == "1":
        print("  PB_SKIP_JUDGE       = 1  (inference only; judging, gold pass, and double judge disabled)")
    print(f"  JUDGE_QPM/RPS       = {os.getenv('JUDGE_QPM', '0')}/{os.getenv('JUDGE_RPS', '0')}")
    print(f"  USE_HINT            = {os.getenv('USE_HINT', '0')}")
    print(f"  NUM_SHARDS/INDEX    = {os.getenv('NUM_SHARDS', '1')}/{os.getenv('SHARD_INDEX', '0')}")
    print(f"  CROP_ROOT           = {os.getenv('CROP_ROOT') or '<project>/tmp/<MODEL>'}")
    print(f"  JUDGE_TEMPLATE      = {_JUDGE_TEMPLATE_PATH}")
    print()


def main() -> None:
    _print_env_config()

    model = os.getenv("MODEL")
    if not model:
        raise SystemExit(
            "MODEL environment variable is required (see .env.example)"
        )
    judge_model = os.getenv("JUDGE_MODEL", JUDGE_DEFAULT_MODEL)
    judge2_model = os.getenv("JUDGE2_MODEL") or None
    judge_temperature = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))
    judge2_temperature = float(os.getenv("JUDGE2_TEMPERATURE", "0.3"))
    judge_parse_retries = int(os.getenv("JUDGE_PARSE_RETRIES", "1"))
    judge_max_retries = int(os.getenv("JUDGE_MAX_RETRIES", "2"))
    dataset = os.getenv("DATASET", "PerceptionBench.jsonl")
    dataset_path = dataset if os.path.isabs(dataset) else os.path.join(
        _PROJECT_ROOT, dataset
    )
    concurrency = int(os.getenv("CONCURRENCY", "4"))
    max_tokens = DEFAULT_MAX_TOKENS
    limit = int(os.getenv("LIMIT", "0"))
    seed = int(os.getenv("SEED", "42"))
    skip_judge = os.getenv("PB_SKIP_JUDGE") == "1"
    gold_pass_enabled = os.getenv("GOLD_PASS", "1") == "1" and not skip_judge
    gold_pass_sample = int(os.getenv("GOLD_PASS_SAMPLE", "200"))
    double_judge_sample = 0 if skip_judge else int(os.getenv("DOUBLE_JUDGE_SAMPLE", "300"))
    judge_qpm = int(os.getenv("JUDGE_QPM", "0"))
    judge_rps = int(os.getenv("JUDGE_RPS", "0"))

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY environment variable is required")
    api_timeout = float(os.getenv("API_TIMEOUT", "600"))
    client = (
        OpenAI(api_key=api_key, base_url=base_url, timeout=api_timeout)
        if base_url
        else OpenAI(api_key=api_key, timeout=api_timeout)
    )

    judge_api_key = os.getenv("JUDGE_API_KEY") or JUDGE_DEFAULT_API_KEY
    judge_base_url = os.getenv("JUDGE_BASE_URL") or JUDGE_DEFAULT_BASE_URL
    if not judge_api_key:
        raise SystemExit(
            "JUDGE_API_KEY environment variable is required "
            "(set it in your .env file or export it before running)"
        )
    judge_client = (
        OpenAI(api_key=judge_api_key, base_url=judge_base_url, timeout=api_timeout)
        if judge_base_url
        else OpenAI(api_key=judge_api_key, timeout=api_timeout)
    )
    judge2_client = judge_client
    if judge2_model:
        judge2_client = (
            OpenAI(api_key=judge_api_key, base_url=judge_base_url, timeout=api_timeout)
            if judge_base_url
            else OpenAI(api_key=judge_api_key, timeout=api_timeout)
        )

    with open(dataset_path, encoding="utf-8") as fh:
        data = [json.loads(line) for line in fh if line.strip()]
    if limit > 0:
        data = data[:limit]

    # Sharding: run NUM_SHARDS processes, each handling every NUM_SHARDS-th
    # record (by position).  Each shard writes its own output/scores files, so
    # shards can run concurrently and resume independently.
    num_shards = int(os.getenv("NUM_SHARDS", "1"))
    shard_index = int(os.getenv("SHARD_INDEX", "0"))
    shard_tag = prediction_shard_tag(num_shards, shard_index)
    if num_shards > 1:
        data = [
            rec for pos, rec in enumerate(data) if pos % num_shards == shard_index
        ]

    print(
        f"Loaded {len(data)} items. Model={model!r} Judge={judge_model!r} "
        f"Concurrency={concurrency}"
        + (f" Shard={shard_index}/{num_shards}" if num_shards > 1 else ""),
        flush=True,
    )

    judge_limiter = None
    if judge_qpm > 0 and judge_rps > 0:
        judge_limiter = SlidingWindowRateLimiter(qpm=judge_qpm, rps=judge_rps)
        print(f"Judge rate limiter enabled: {judge_qpm} qpm / {judge_rps} rps", flush=True)

    # Gold pass: judge each reference answer against itself (all must be True).
    gold_stats = None
    if gold_pass_enabled:
        gold_stats = run_gold_pass(
            judge_client,
            judge_model,
            data,
            sample_size=gold_pass_sample,
            seed=seed,
            concurrency=concurrency,
            judge_temperature=judge_temperature,
            judge_parse_retries=judge_parse_retries,
            judge_max_retries=judge_max_retries,
            judge_limiter=judge_limiter,
        )
        rate = gold_stats["pass_rate"]
        print(
            f"Gold pass: {gold_stats['passed']}/{gold_stats['sample_size']} "
            f"({rate*100:.2f}%), judge-ok={gold_stats['judged_ok']}",
            flush=True,
        )
        if rate < 0.99:
            print(
                "WARNING: gold pass rate < 99%; the judge harness may be "
                "unreliable. Inspect gold_pass_failures before trusting scores.",
                flush=True,
            )
        if gold_stats["failures"]:
            gold_fail_path = os.path.join(
                _PROJECT_ROOT,
                "results",
                f"gold_pass_failures_{judge_model.replace('/', '_')}.jsonl",
            )
            os.makedirs(os.path.dirname(gold_fail_path), exist_ok=True)
            with open(gold_fail_path, "w", encoding="utf-8") as fh:
                for fail in gold_stats["failures"]:
                    fh.write(json.dumps(fail, ensure_ascii=False) + "\n")
            print(f"Gold pass failures saved to {gold_fail_path}", flush=True)

    # Crops default to <project>/tmp/<model>/ so they survive reboots and are
    # separated per evaluated model; override with the CROP_ROOT env var.
    crop_root = os.getenv("CROP_ROOT") or os.path.join(
        _PROJECT_ROOT, "tmp", model.replace("/", "_")
    )
    pipeline = AgenticQAPipeline(
        client,
        model=model,
        judge_model=judge_model,
        max_tokens=max_tokens,
        crop_root=crop_root,
    )

    import random

    rng = random.Random(seed)
    double_indices: set[Any] = set()
    if double_judge_sample > 0:
        double_indices = set(
            rng.sample(
                [rec.get("index") for rec in data],
                min(double_judge_sample, len(data)),
            )
        )
        print(f"Double-judge sample: {len(double_indices)} records", flush=True)

    session_dir = pipeline.crop_root / f"session_{int(time.time())}{shard_tag}"
    out_dir = os.getenv("PB_RESULTS_DIR") or os.path.join(_PROJECT_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(
        out_dir,
        prediction_basename(model, num_shards, shard_index, os.getenv("PB_OUTPUT_SUFFIX", "")),
    )

    # Resume: reuse records already written to the output JSONL.
    done: dict[Any, dict[str, Any]] = {}
    if os.path.exists(out_base + ".jsonl"):
        with open(out_base + ".jsonl", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                done[rec.get("index")] = rec
        print(f"Resuming: {len(done)} records already present", flush=True)
    todo = [rec for rec in data if rec.get("index") not in done]
    print(f"Remaining: {len(todo)} items", flush=True)

    results = list(done.values())
    if todo:
        def run_one(rec: dict[str, Any]) -> dict[str, Any]:
            return process_record(
                rec,
                pipeline,
                session_dir=session_dir,
                index_key=rec.get("index"),
                judge_client=judge_client,
                judge2_client=judge2_client,
                judge2_model=judge2_model,
                judge2_temperature=judge2_temperature,
                double_judge=rec.get("index") in double_indices,
                judge_temperature=judge_temperature,
                judge_parse_retries=judge_parse_retries,
                judge_max_retries=judge_max_retries,
                judge_limiter=judge_limiter,
            )

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(run_one, rec): rec for rec in todo}
            out_fh = open(out_base + ".jsonl", "a", encoding="utf-8")
            try:
                for fut in _tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="PerceptionBench (agentic)",
                ):
                    try:
                        res = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        rec = futures[fut]
                        res = {
                            "index": rec.get("index"),
                            "error_category": rec.get("error_category"),
                            "source_bmk": rec.get("source_bmk"),
                            "prediction": f"[error] {exc}",
                            "judge_result": None,
                            "judge_status": "runtime_error",
                            "judge_reason": f"executor error: {exc}",
                            "agentic": {"error": f"executor error: {exc}"},
                        }
                    results.append(res)
                    out_fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                    out_fh.flush()
            finally:
                out_fh.close()

    # Aggregation.  Judge-side failures (parse_error / judge_error) are excluded
    # from the primary metric and reported separately; the paper-style metric
    # counts them as wrong for comparability with eval.py.
    from collections import Counter

    tot, cor = Counter(), Counter()
    judged = 0
    correct = 0
    for r in results:
        if r.get("judge_status") != "ok":
            continue
        judged += 1
        result_val = r.get("judge_result")
        if result_val == 1:
            correct += 1
        c = r.get("error_category") or "?"
        tot[c] += 1
        cor[c] += int(result_val or 0)
    overall_primary = correct / judged if judged else 0.0
    overall_paper = correct / len(results) if results else 0.0

    counts = {
        "total": len(results),
        "judged_ok": judged,
        "model_no_answer": sum(
            1 for r in results if r.get("judge_status") == "no_answer"
        ),
        "runtime_error": sum(
            1 for r in results if r.get("judge_status") == "runtime_error"
        ),
        "judge_parse_error": sum(
            1 for r in results if r.get("judge_status") == "parse_error"
        ),
        "judge_api_error": sum(
            1 for r in results if r.get("judge_status") == "judge_error"
        ),
    }

    double_stats = None
    doubled = [
        r
        for r in results
        if r.get("judge2_status") == "ok" and r.get("judge_result") is not None
    ]
    if doubled:
        agreed = sum(
            1 for r in doubled if r.get("judge2_result") == r.get("judge_result")
        )
        double_stats = {
            "sample": len(doubled),
            "agreement": agreed / len(doubled),
            "disagreements": len(doubled) - agreed,
        }

    print()
    print(
        f"Overall (excl. judge failures): {overall_primary*100:.2f}%  "
        f"({correct}/{judged})"
    )
    print(
        f"Overall (paper-style, judge failures as wrong): {overall_paper*100:.2f}%  "
        f"({correct}/{len(results)})"
    )
    print("Counts:", counts)
    if double_stats:
        print(
            f"Double-judge agreement: {double_stats['agreement']*100:.2f}%  "
            f"({double_stats['sample'] - double_stats['disagreements']}/{double_stats['sample']})"
        )
    print()
    for c in sorted(tot, key=lambda x: -tot[x]):
        print(f"  {c:32s} {cor[c]/tot[c]*100:5.1f}%  ({cor[c]}/{tot[c]})")

    scores = {
        "overall": overall_primary,
        "overall_judge_failures_as_wrong": overall_paper,
        "per_category": {c: cor[c] / tot[c] for c in tot},
        "counts": counts,
        "double_judge": double_stats,
        "gold_pass": (
            {k: v for k, v in gold_stats.items() if k != "failures"}
            if gold_stats is not None
            else None
        ),
        "config": {
            "model": model,
            "judge_model": judge_model,
            "judge2_model": judge2_model,
            "judge_temperature": judge_temperature,
            "judge2_temperature": judge2_temperature,
            "dataset": dataset_path,
            "prompt_mode": "agentic",
            "num_shards": num_shards,
            "shard_index": shard_index,
            "crop_root": str(pipeline.crop_root),
            "max_tool_calls": MAX_TOOL_CALLS,
            "max_action_corrections": MAX_ACTION_CORRECTIONS,
            "thinking_budget": THINKING_BUDGET,
            "agentic_temperature": AGENTIC_TEMPERATURE,
            "judge_template": _JUDGE_TEMPLATE_PATH,
            "double_judge_sample": double_judge_sample,
            "gold_pass_sample": gold_pass_sample,
            "seed": seed,
        },
    }
    with open(out_base + "_scores.json", "w", encoding="utf-8") as fh:
        json.dump(scores, fh, indent=2, ensure_ascii=False)
    print("saved to", out_base + ".jsonl")


if __name__ == "__main__":
    main()
