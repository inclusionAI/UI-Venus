#!/usr/bin/env python3
"""Evaluate the agentic crop pipeline on the official HR-Bench MCQ protocol."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

# Add the shared pipeline directory when a wrapper runs from this entry directory.
KIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if KIT_ROOT not in sys.path:
    sys.path.insert(0, KIT_ROOT)
from common.agentic_pipeline import (
    DEFAULT_API_URL,
    DEFAULT_CROP_MAX_DIMENSION,
    DEFAULT_CROP_MAX_PIXELS,
    AgenticGroundingPipeline,
    SlidingWindowRateLimiter,
    parse_mcq_answer,
)


CHOICES = ("A", "B", "C", "D")
CATEGORIES = ("single", "cross")
CYCLES = (0, 1, 2, 3)
PRINT_LOCK = threading.Lock()
YES_NO_RE = re.compile(r"^(yes|no)\b", re.IGNORECASE)


def print_threadsafe(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def json_fallback(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=json_fallback)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path")
    parser.add_argument("--split", choices=("4k", "8k"))
    parser.add_argument("--image_cache")
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--combine_reports", nargs="+")
    parser.add_argument("--max_samples", type=int)

    parser.add_argument("--model")
    parser.add_argument("--api_url", default=DEFAULT_API_URL)
    parser.add_argument(
        "--prompt_mode",
        choices=("agentic", "direct_nothink", "direct_think"),
        default="agentic",
    )
    parser.add_argument("--reasoning_effort", default="high")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking_budget", type=int, default=4096)
    # Match the wrapper budget: THINKING_BUDGET + 1024 includes visible answer tokens.
    # Keep this separate from SSPro, where None means thinking_budget.
    parser.add_argument("--max_completion_tokens", type=int, default=5120)
    parser.add_argument("--max_tool_calls", type=int, default=3)
    parser.add_argument("--max_action_corrections", type=int, default=4)
    parser.add_argument("--api_image_quality", type=int, default=95)
    parser.add_argument("--crop_max_pixels", type=int, default=DEFAULT_CROP_MAX_PIXELS)
    parser.add_argument(
        "--crop_max_dimension", type=int, default=DEFAULT_CROP_MAX_DIMENSION
    )
    parser.add_argument("--crop_root", default="/tmp/hrbench_crops")
    parser.add_argument("--discard_crops", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--qpm", type=int, default=600)
    parser.add_argument("--rps", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--no_retry_errors", action="store_true")
    parser.add_argument("--pass_at", type=int, default=1)
    parser.add_argument("--pass_at_temperature", type=float, default=1.0)

    parser.add_argument(
        "--judge_mode", choices=("strict", "official", "both"), default="both"
    )
    parser.add_argument("--judge_model")
    parser.add_argument("--judge_api_url")
    parser.add_argument("--judge_temperature", type=float, default=0.2)
    parser.add_argument("--judge_max_tokens", type=int, default=32)
    parser.add_argument("--judge_workers", type=int, default=16)
    parser.add_argument("--judge_qpm", type=int, default=600)
    parser.add_argument("--judge_rps", type=int, default=8)
    parser.add_argument("--judge_timeout", type=float, default=120.0)
    parser.add_argument("--judge_max_attempts", type=int, default=5)
    return parser.parse_args()


def source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def resolve_tsv_path(dataset_path: Path) -> Path:
    if dataset_path.suffix.lower() == ".tsv":
        return dataset_path
    if dataset_path.suffix.lower() == ".parquet":
        tsv_path = dataset_path.with_suffix(".tsv")
        if tsv_path.is_file():
            return tsv_path
        raise FileNotFoundError(
            "HR-Bench image localization streams the TSV to avoid materializing the "
            f"multi-gigabyte Parquet image dictionary; missing sibling TSV: {tsv_path}"
        )
    raise ValueError("--dataset_path must point to an HR-Bench .tsv or .parquet file")


def image_extension_and_size(raw: bytes) -> tuple[str, list[int]]:
    with Image.open(BytesIO(raw)) as image:
        image.load()
        image_format = (image.format or "png").lower()
        extension = {"jpeg": "jpg", "tiff": "tif"}.get(image_format, image_format)
        return extension, [int(image.width), int(image.height)]


def persist_image(encoded: str, image_dir: Path) -> tuple[str, list[int]]:
    raw = base64.b64decode(encoded)
    digest = hashlib.sha256(raw).hexdigest()
    extension, size = image_extension_and_size(raw)
    path = image_dir / f"{digest}.{extension}"
    if not path.is_file():
        image_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        with temporary.open("wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    return str(path.resolve()), size


def normalized_task(row: dict[str, str], split: str, image_path: str, size: list[int]) -> dict[str, Any]:
    index = int(row["index"])
    return {
        "eval_key": f"hrbench{split}:{index}",
        "order": index,
        "index": index,
        "group_index": index // 4,
        "split": split,
        "answer": row["answer"].strip().upper(),
        "question": row["question"].strip(),
        "options": {choice: row[choice] for choice in CHOICES},
        "category": row["category"].strip(),
        "cycle_category": int(row["cycle_category"]),
        "image_path": image_path,
        "img_size": size,
    }


def validate_tasks(tasks: list[dict[str, Any]], *, require_complete_groups: bool = True) -> None:
    if not tasks:
        raise ValueError("HR-Bench contains no selected tasks")
    indices = [task["index"] for task in tasks]
    if len(indices) != len(set(indices)):
        raise ValueError("HR-Bench indices must be unique")
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        if task["answer"] not in CHOICES:
            raise ValueError(f"Invalid answer at index {task['index']}: {task['answer']}")
        if task["category"] not in CATEGORIES:
            raise ValueError(f"Invalid category at index {task['index']}: {task['category']}")
        if task["cycle_category"] not in CYCLES:
            raise ValueError(
                f"Invalid cycle_category at index {task['index']}: {task['cycle_category']}"
            )
        groups[task["group_index"]].append(task)

    for group_index, group in groups.items():
        if require_complete_groups and len(group) != 4:
            raise ValueError(f"Group {group_index} has {len(group)} rows instead of 4")
        if len(group) == 4:
            if {task["cycle_category"] for task in group} != set(CYCLES):
                raise ValueError(f"Group {group_index} does not contain cycles 0,1,2,3")
            if len({task["question"] for task in group}) != 1:
                raise ValueError(f"Group {group_index} changes question across cycles")
            if len({task["image_path"] for task in group}) != 1:
                raise ValueError(f"Group {group_index} changes image across cycles")
            correct_texts = {
                task["options"][task["answer"]].strip() for task in group
            }
            if len(correct_texts) != 1:
                raise ValueError(f"Group {group_index} changes correct answer content")


def load_hrbench_tasks(
    dataset_path: Path | None,
    split: str,
    image_cache: Path,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    if max_samples is not None and (max_samples < 1 or max_samples % 4):
        raise ValueError("--max_samples must be a positive multiple of 4")
    # Reuse a complete manifest when the source TSV is unavailable.
    # Manifest loading validates split, completeness, and image existence.
    tsv_path: Path | None = None
    if dataset_path is not None:
        candidate = resolve_tsv_path(dataset_path)
        if candidate.is_file():
            tsv_path = candidate
    signature = source_signature(tsv_path) if tsv_path is not None else None
    manifest_path = image_cache / f"hrbench_{split}_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        manifest_tasks = manifest.get("tasks") or []
        source_ok = signature is None or manifest.get("source") == signature
        if (
            source_ok
            and manifest.get("split") == split
            and manifest.get("complete") is True
            and all(Path(task.get("image_path", "")).is_file() for task in manifest_tasks)
        ):
            tasks = list(manifest_tasks)
            if max_samples is not None:
                tasks = tasks[:max_samples]
            validate_tasks(tasks)
            print(f"Loaded {len(tasks)} localized tasks from {manifest_path}", flush=True)
            return tasks
    if tsv_path is None:
        raise FileNotFoundError(
            f"HR-Bench TSV unavailable and no valid manifest: dataset_path={dataset_path}, "
            f"manifest={manifest_path}"
        )

    csv.field_size_limit(sys.maxsize)
    tasks: list[dict[str, Any]] = []
    image_dir = image_cache / "images"
    active_group: int | None = None
    active_encoded: str | None = None
    active_image_path = ""
    active_size: list[int] = [0, 0]
    completed_source = True
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"index", "answer", "question", "category", "cycle_category", "image", *CHOICES}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing HR-Bench columns: {sorted(missing)}")
        for row in reader:
            index = int(row["index"])
            group_index = index // 4
            encoded = row["image"]
            if group_index != active_group:
                active_group = group_index
                active_encoded = encoded
                active_image_path, active_size = persist_image(encoded, image_dir)
            elif encoded != active_encoded:
                raise ValueError(f"Group {group_index} changes image across cycles")
            tasks.append(
                normalized_task(row, split, active_image_path, list(active_size))
            )
            if max_samples is not None and len(tasks) >= max_samples:
                completed_source = False
                break

    validate_tasks(tasks)
    if completed_source:
        write_json_atomic(
            manifest_path,
            {
                "version": 1,
                "source": signature,
                "split": split,
                "complete": True,
                "tasks": tasks,
            },
        )
    print(
        f"Localized {len(tasks)} HR-Bench {split.upper()} tasks from {tsv_path}",
        flush=True,
    )
    return tasks


def build_question_prompt(task: dict[str, Any]) -> str:
    options = "\n".join(
        f"{choice}. {task['options'][choice]}" for choice in CHOICES
    )
    return (
        f"Question: {task['question']}\n"
        f"Options:\n{options}\n"
        "Please select the correct answer from the options above."
    )


def extract_prediction_option(text: str, options: dict[str, str]) -> tuple[str | None, str]:
    parsed = parse_mcq_answer(text)
    if parsed in options:
        return parsed, "structured"

    normalized = re.sub(r"[.()\[\],:;!*#{}]", " ", str(text))
    tokens = normalized.split()
    mentioned = {choice for choice in options if choice in tokens}
    if len(mentioned) == 1:
        return next(iter(mentioned)), "standalone_letter"

    lowered = str(text).lower()
    text_matches = [
        choice
        for choice, option_text in options.items()
        if str(option_text).strip().lower() in lowered
    ]
    if len(text_matches) == 1:
        return text_matches[0], "option_text"
    return None, "unparseable"


def trajectory_candidate(trajectory: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    raw = str(trajectory.get("final_content") or "")
    prediction = trajectory.get("prediction_option")
    parse_method = "trajectory"
    if prediction not in CHOICES:
        prediction, parse_method = extract_prediction_option(raw, task["options"])
    return {
        "prediction_option": prediction,
        "parse_method": parse_method,
        "strict_score": int(prediction == task["answer"]),
        "wrong_format": prediction is None,
        "error": trajectory.get("error"),
        "final_content": raw,
        "trajectory": trajectory,
    }


def evaluate_one(
    pipeline: AgenticGroundingPipeline,
    task: dict[str, Any],
    pass_at: int,
    pass_at_temperature: float,
) -> dict[str, Any]:
    prompt = build_question_prompt(task)
    task_id = task["eval_key"].replace(":", "_")
    try:
        if pass_at == 1:
            trajectories = [
                pipeline.inference(
                    instruction=prompt,
                    image_path=task["image_path"],
                    task_id=task_id,
                )
            ]
        else:
            trajectories = pipeline.inference_many(
                instruction=prompt,
                image_path=task["image_path"],
                sample_count=pass_at,
                task_id=task_id,
                temperature=pass_at_temperature,
            )
        candidates = [trajectory_candidate(item, task) for item in trajectories]
        if len(candidates) != pass_at:
            raise RuntimeError(
                f"Expected {pass_at} candidates but received {len(candidates)}"
            )
        selected = candidates[0]
        error = selected.get("error")
        exception_traceback = None
    except Exception as exc:
        error = f"Unhandled {type(exc).__name__}: {exc}"
        exception_traceback = traceback.format_exc()
        candidates = [
            {
                "prediction_option": None,
                "parse_method": "unparseable",
                "strict_score": 0,
                "wrong_format": True,
                "error": error,
                "final_content": "",
                "trajectory": None,
            }
        ]
        selected = candidates[0]

    candidate_summaries = [
        {key: value for key, value in candidate.items() if key != "trajectory"}
        for candidate in candidates
    ]
    return {
        "eval_key": task["eval_key"],
        "order": task["order"],
        "index": task["index"],
        "group_index": task["group_index"],
        "split": task["split"],
        "category": task["category"],
        "cycle_category": task["cycle_category"],
        "question": task["question"],
        "options": task["options"],
        "answer": task["answer"],
        "answer_text": task["options"][task["answer"]],
        "image_path": task["image_path"],
        "img_size": task["img_size"],
        "prompt_to_evaluate": prompt,
        "pass_at": pass_at,
        "num_candidates": len(candidates),
        "prediction_option": selected["prediction_option"],
        "parse_method": selected["parse_method"],
        "strict_score": selected["strict_score"],
        "wrong_format": selected["wrong_format"],
        "pass_at_hit": int(any(item["strict_score"] for item in candidates)),
        "error": error,
        "exception_traceback": exception_traceback,
        "final_content": selected["final_content"],
        "trajectory": selected["trajectory"],
        "additional_trajectories": [
            candidate["trajectory"] for candidate in candidates[1:]
        ],
        "candidates": candidate_summaries,
    }


def load_progress(path: Path, fingerprint: str) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Ignoring invalid progress line {line_number} in {path}", flush=True)
                continue
            if record.get("run_fingerprint") == fingerprint and record.get("eval_key"):
                completed[record["eval_key"]] = record
    return completed


def run_fingerprint(args: argparse.Namespace, tasks: list[dict[str, Any]]) -> str:
    value = {
        "version": 1,
        "split": tasks[0]["split"],
        "tasks": [
            {
                "eval_key": task["eval_key"],
                "question": task["question"],
                "options": task["options"],
                "answer": task["answer"],
                "category": task["category"],
                "cycle_category": task["cycle_category"],
                "image_path": task["image_path"],
            }
            for task in tasks
        ],
        "model": args.model,
        "api_url": args.api_url,
        "prompt_mode": args.prompt_mode,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "thinking_budget": args.thinking_budget,
        "max_completion_tokens": args.max_completion_tokens,
        "max_tool_calls": args.max_tool_calls,
        "max_action_corrections": args.max_action_corrections,
        "api_image_quality": args.api_image_quality,
        "crop_max_pixels": args.crop_max_pixels,
        "crop_max_dimension": args.crop_max_dimension,
        "pass_at": args.pass_at,
        "pass_at_temperature": args.pass_at_temperature,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_inference_stage(
    args: argparse.Namespace,
    tasks: list[dict[str, Any]],
    fingerprint: str,
) -> list[dict[str, Any]]:
    output_path = Path(args.log_path)
    progress_path = output_path.with_name(f".{output_path.stem}.inference.progress.jsonl")
    completed = load_progress(progress_path, fingerprint)
    pending = [
        task
        for task in tasks
        if task["eval_key"] not in completed
        or (
            not args.no_retry_errors
            and (completed[task["eval_key"]].get("error") or completed[task["eval_key"]].get("wrong_format"))
        )
    ]
    pending_keys = {task["eval_key"] for task in pending}
    resumed_results = [
        completed[task["eval_key"]]
        for task in tasks
        if task["eval_key"] in completed and task["eval_key"] not in pending_keys
    ]
    correct_finished = sum(
        int(result.get("strict_score", 0)) for result in resumed_results
    )
    print(
        f"Inference: {len(tasks)} total, {len(tasks) - len(pending)} resumed, "
        f"{len(pending)} pending; correct={correct_finished}/{len(resumed_results)}, "
        f"accuracy={correct_finished / len(resumed_results):.2%}"
        if resumed_results
        else (
            f"Inference: {len(tasks)} total, 0 resumed, {len(pending)} pending; "
            "correct=0/0, accuracy=0.00%"
        ),
        flush=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with AgenticGroundingPipeline(
        api_url=args.api_url,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        thinking_budget=args.thinking_budget,
        max_completion_tokens=args.max_completion_tokens,
        max_tool_calls=args.max_tool_calls,
        max_action_corrections=args.max_action_corrections,
        api_image_quality=args.api_image_quality,
        crop_max_pixels=args.crop_max_pixels,
        crop_max_dimension=args.crop_max_dimension,
        crop_root=args.crop_root,
        persist_crops=not args.discard_crops,
        qpm=args.qpm,
        rps=args.rps,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
        workers_hint=args.workers,
        prompt_mode=args.prompt_mode,
        answer_mode="mcq",
    ) as pipeline, progress_path.open("a", encoding="utf-8") as progress:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task: dict[Future[dict[str, Any]], dict[str, Any]] = {
                executor.submit(
                    evaluate_one,
                    pipeline,
                    task,
                    args.pass_at,
                    args.pass_at_temperature,
                ): task
                for task in pending
            }
            finished = len(tasks) - len(pending)
            for future in as_completed(future_to_task):
                result = future.result()
                result["run_fingerprint"] = fingerprint
                progress.write(json.dumps(result, ensure_ascii=False, default=json_fallback) + "\n")
                progress.flush()
                os.fsync(progress.fileno())
                completed[result["eval_key"]] = result
                finished += 1
                correct_finished += int(result.get("strict_score", 0))
                running_accuracy = correct_finished / finished if finished else 0.0
                print_threadsafe(
                    f"[inference {finished}/{len(tasks)}] {result['eval_key']}: "
                    f"prediction={result['prediction_option']}, gt={result['answer']}, "
                    f"strict={result['strict_score']} | "
                    f"correct={correct_finished}/{finished}, "
                    f"accuracy={running_accuracy:.2%}"
                    + (f", error={result['error']}" if result.get("error") else "")
                )
    return [completed[task["eval_key"]] for task in tasks]


def official_judge_prompt(result: dict[str, Any]) -> str:
    options = "\n".join(
        f"{choice}. {result['options'][choice]}" for choice in CHOICES
    )
    question_prompt = (
        f"{result['question']}\nOptions:\n{options}\n"
        "Please select the correct answer from the options above. \n"
    )
    ground_truth = f"{result['answer']}. {result['answer_text']}"
    content = (
        f"Given the following question: {question_prompt}, the correct answer is: "
        f"{ground_truth}. Does the following answer correctly answers the question, "
        f"answer:{result['final_content']}?"
    )
    return (
        "You are a helpful and precise assistant for checking the quality of the "
        f"answer. Please answer in only yes or no. {content}"
    )


def response_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            item if isinstance(item, str) else str(item.get("text", ""))
            for item in value
            if isinstance(item, (str, dict))
        )
    return str(value or "")


class OfficialHRBenchJudge:
    """Official HR-Bench Yes/No judge over an OpenAI-compatible local endpoint."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.api_url = args.judge_api_url or args.api_url
        self.model = args.judge_model or args.model
        self.temperature = args.judge_temperature
        self.max_tokens = args.judge_max_tokens
        self.max_attempts = args.judge_max_attempts
        self.api_key = os.environ.get("OPENAI_API_KEY") or "dummy"
        self.limiter = SlidingWindowRateLimiter(args.judge_qpm, args.judge_rps)
        self.client = httpx.Client(
            timeout=httpx.Timeout(args.judge_timeout, connect=min(30.0, args.judge_timeout)),
            limits=httpx.Limits(
                max_connections=max(args.judge_workers * 2, 8),
                max_keepalive_connections=max(args.judge_workers, 4),
            ),
            trust_env=False,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "OfficialHRBenchJudge":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def judge(self, result: dict[str, Any]) -> dict[str, Any]:
        prompt = official_judge_prompt(result)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": 0.8,
            "top_k": 40,
            "max_tokens": self.max_tokens,
            "n": 1,
            "stream": False,
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        error = None
        answer = ""
        attempts = 0
        for attempts in range(1, self.max_attempts + 1):
            try:
                self.limiter.acquire()
                response = self.client.post(self.api_url, headers=headers, json=payload)
                if response.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}: {response.text[:2000]}",
                        request=response.request,
                        response=response,
                    )
                body = response.json()
                choices = body.get("choices") or []
                if not choices:
                    raise RuntimeError("Judge response contains no choices")
                answer = response_text((choices[0].get("message") or {}).get("content")).strip()
                error = None
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempts < self.max_attempts:
                    time.sleep(min(2**attempts, 20))
        match = YES_NO_RE.match(answer.lower())
        score = 1 if match and match.group(1).lower() == "yes" else 0
        format_error = error is None and match is None
        return {
            "eval_key": result["eval_key"],
            "official_judge_score": score,
            "official_judge_answer": answer,
            "official_judge_error": error,
            "official_judge_format_error": format_error,
            "official_judge_attempts": attempts,
            "official_judge_prompt": prompt,
        }


def judge_fingerprint(args: argparse.Namespace, inference_fingerprint: str) -> str:
    value = {
        "version": 1,
        "inference_fingerprint": inference_fingerprint,
        "judge_model": args.judge_model or args.model,
        "judge_api_url": args.judge_api_url or args.api_url,
        "temperature": args.judge_temperature,
        "max_tokens": args.judge_max_tokens,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_judge_stage(
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    inference_fingerprint: str,
) -> list[dict[str, Any]]:
    if args.judge_mode == "strict":
        return results
    output_path = Path(args.log_path)
    progress_path = output_path.with_name(f".{output_path.stem}.judge.progress.jsonl")
    fingerprint = judge_fingerprint(args, inference_fingerprint)
    completed = load_progress(progress_path, fingerprint)
    pending = [result for result in results if result["eval_key"] not in completed]
    pending_keys = {result["eval_key"] for result in pending}
    resumed_records = [
        completed[result["eval_key"]]
        for result in results
        if result["eval_key"] in completed and result["eval_key"] not in pending_keys
    ]
    correct_finished = sum(
        int(record.get("official_judge_score", 0)) for record in resumed_records
    )
    print(
        f"Judge: {len(results)} total, {len(results) - len(pending)} resumed, "
        f"{len(pending)} pending; correct={correct_finished}/{len(resumed_records)}, "
        f"accuracy={correct_finished / len(resumed_records):.2%}"
        if resumed_records
        else (
            f"Judge: {len(results)} total, 0 resumed, {len(pending)} pending; "
            "correct=0/0, accuracy=0.00%"
        ),
        flush=True,
    )
    with OfficialHRBenchJudge(args) as judge, progress_path.open("a", encoding="utf-8") as progress:
        with ThreadPoolExecutor(max_workers=args.judge_workers) as executor:
            future_to_result = {
                executor.submit(judge.judge, result): result for result in pending
            }
            finished = len(results) - len(pending)
            for future in as_completed(future_to_result):
                record = future.result()
                record["run_fingerprint"] = fingerprint
                progress.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress.flush()
                os.fsync(progress.fileno())
                completed[record["eval_key"]] = record
                finished += 1
                correct_finished += int(record.get("official_judge_score", 0))
                running_accuracy = correct_finished / finished if finished else 0.0
                print_threadsafe(
                    f"[judge {finished}/{len(results)}] {record['eval_key']}: "
                    f"score={record['official_judge_score']}, "
                    f"answer={record['official_judge_answer']!r} | "
                    f"correct={correct_finished}/{finished}, "
                    f"accuracy={running_accuracy:.2%}"
                    + (
                        f", error={record['official_judge_error']}"
                        if record.get("official_judge_error")
                        else ""
                    )
                )
    for result in results:
        judge_record = dict(completed[result["eval_key"]])
        result["judge_fingerprint"] = judge_record.pop("run_fingerprint", None)
        result.update(judge_record)
    return results


def score_bucket(correct: int, total: int) -> dict[str, Any]:
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
    }


def aggregate_score(results: list[dict[str, Any]], score_field: str) -> dict[str, Any]:
    cycles: dict[str, dict[str, Any]] = {}
    for cycle in CYCLES:
        cycle_results = [result for result in results if result["cycle_category"] == cycle]
        categories: dict[str, Any] = {}
        categories["all"] = score_bucket(
            sum(int(result.get(score_field, 0)) for result in cycle_results),
            len(cycle_results),
        )
        for category in CATEGORIES:
            subset = [result for result in cycle_results if result["category"] == category]
            categories[category] = score_bucket(
                sum(int(result.get(score_field, 0)) for result in subset), len(subset)
            )
        cycles[str(cycle)] = categories

    average: dict[str, Any] = {}
    for category in ("all", *CATEGORIES):
        accuracies = [
            cycles[str(cycle)][category]["accuracy"]
            for cycle in CYCLES
            if cycles[str(cycle)][category]["accuracy"] is not None
        ]
        average[category] = {
            "accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
            "cycle_accuracies": accuracies,
        }
    return {"cycles": cycles, "average": average}


def group_diagnostics(results: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[int(result["group_index"])].append(result)
    complete = [group for group in groups.values() if len(group) == 4]
    all_correct = sum(all(item["strict_score"] for item in group) for group in complete)
    content_consistent = 0
    for group in complete:
        predicted_texts = {
            item["options"].get(item["prediction_option"])
            for item in group
            if item.get("prediction_option") in CHOICES
        }
        if len(predicted_texts) == 1 and all(item.get("prediction_option") for item in group):
            content_consistent += 1
    return {
        "complete_groups": len(complete),
        "all_four_cycles_correct": all_correct,
        "all_four_cycles_correct_rate": all_correct / len(complete) if complete else 0.0,
        "prediction_content_consistent": content_consistent,
        "prediction_content_consistency_rate": (
            content_consistent / len(complete) if complete else 0.0
        ),
    }


def build_report(
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    inference_fingerprint: str,
) -> dict[str, Any]:
    results.sort(key=lambda item: item["order"])
    metrics: dict[str, Any] = {
        "strict": aggregate_score(results, "strict_score"),
        "pass_at": aggregate_score(results, "pass_at_hit"),
        "diagnostics": {
            "wrong_format_num": sum(bool(item.get("wrong_format")) for item in results),
            "inference_error_num": sum(bool(item.get("error")) for item in results),
            "average_crop_calls": (
                sum(
                    int((item.get("trajectory") or {}).get("crop_action_count", 0))
                    for item in results
                )
                / len(results)
                if results
                else 0.0
            ),
            "groups": group_diagnostics(results),
        },
    }
    if args.judge_mode != "strict":
        metrics["official_judge"] = aggregate_score(results, "official_judge_score")
        metrics["diagnostics"].update(
            {
                "judge_error_num": sum(
                    bool(item.get("official_judge_error")) for item in results
                ),
                "judge_format_error_num": sum(
                    bool(item.get("official_judge_format_error")) for item in results
                ),
                "strict_judge_disagreement_num": sum(
                    int(item.get("strict_score", 0))
                    != int(item.get("official_judge_score", 0))
                    for item in results
                ),
            }
        )
        primary_metric = "official_judge"
    else:
        primary_metric = "strict"
    metrics["primary_metric"] = primary_metric
    metrics["primary_accuracy"] = metrics[primary_metric]["average"]["all"]["accuracy"]
    return {
        "benchmark": f"HRBench{results[0]['split'].upper()}",
        "run_fingerprint": inference_fingerprint,
        "config": {
            "dataset_path": (
                str(Path(args.dataset_path).resolve()) if args.dataset_path else None
            ),
            "image_cache": str(Path(args.image_cache).resolve()),
            "model": args.model,
            "api_url": args.api_url,
            "prompt_mode": args.prompt_mode,
            "answer_mode": "mcq",
            "reasoning_effort": args.reasoning_effort,
            "temperature": args.temperature,
            "thinking_budget": args.thinking_budget,
            "max_completion_tokens": args.max_completion_tokens,
            "max_tool_calls": args.max_tool_calls,
            "max_action_corrections": args.max_action_corrections,
            "api_image_quality": args.api_image_quality,
            "crop_max_pixels": args.crop_max_pixels,
            "crop_max_dimension": args.crop_max_dimension,
            "workers": args.workers,
            "pass_at": args.pass_at,
            "pass_at_temperature": args.pass_at_temperature,
            "judge_mode": args.judge_mode,
            "judge_model": args.judge_model or args.model,
            "judge_api_url": args.judge_api_url or args.api_url,
            "judge_temperature": args.judge_temperature,
        },
        "metrics": metrics,
        "details": results,
    }


def combine_reports(report_paths: list[str], output_path: Path) -> None:
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in report_paths]
    splits: dict[str, Any] = {}
    primary_values: list[float] = []
    strict_values: list[float] = []
    judge_values: list[float] = []
    for report in reports:
        name = report["benchmark"]
        metrics = report["metrics"]
        splits[name] = {
            "primary_metric": metrics["primary_metric"],
            "primary_accuracy": metrics["primary_accuracy"],
            "strict_accuracy": metrics["strict"]["average"]["all"]["accuracy"],
            "official_judge_accuracy": (
                metrics.get("official_judge", {}).get("average", {}).get("all", {}).get("accuracy")
            ),
        }
        primary_values.append(float(metrics["primary_accuracy"]))
        strict_values.append(float(splits[name]["strict_accuracy"]))
        if splits[name]["official_judge_accuracy"] is not None:
            judge_values.append(float(splits[name]["official_judge_accuracy"]))
    summary = {
        "benchmark": "HRBench",
        "splits": splits,
        "average": {
            "primary_accuracy": sum(primary_values) / len(primary_values),
            "strict_accuracy": sum(strict_values) / len(strict_values),
            "official_judge_accuracy": (
                sum(judge_values) / len(judge_values) if judge_values else None
            ),
        },
    }
    write_json_atomic(output_path, summary)
    print(f"Wrote combined HR-Bench summary to {output_path}", flush=True)


def validate_args(args: argparse.Namespace) -> None:
    if args.combine_reports:
        return
    if not args.split or not args.image_cache:
        raise ValueError("--split and --image_cache are required")
    if not args.model:
        raise ValueError("--model is required (vllm --served-model-name)")
    positive = {
        "workers": args.workers,
        "qpm": args.qpm,
        "rps": args.rps,
        "thinking_budget": args.thinking_budget,
        "max_completion_tokens": args.max_completion_tokens,
        "crop_max_pixels": args.crop_max_pixels,
        "crop_max_dimension": args.crop_max_dimension,
        "pass_at": args.pass_at,
        "judge_workers": args.judge_workers,
        "judge_qpm": args.judge_qpm,
        "judge_rps": args.judge_rps,
        "judge_max_tokens": args.judge_max_tokens,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    if not math.isfinite(args.pass_at_temperature) or args.pass_at_temperature <= 0:
        raise ValueError("--pass_at_temperature must be finite and positive")
    if not math.isfinite(args.judge_temperature) or args.judge_temperature < 0:
        raise ValueError("--judge_temperature must be finite and nonnegative")


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_path = Path(args.log_path)
    if args.combine_reports:
        combine_reports(args.combine_reports, output_path)
        return
    dataset_path = Path(args.dataset_path) if args.dataset_path else None
    tasks = load_hrbench_tasks(
        dataset_path,
        args.split,
        Path(args.image_cache),
        args.max_samples,
    )
    fingerprint = run_fingerprint(args, tasks)
    results = run_inference_stage(args, tasks, fingerprint)
    results = run_judge_stage(args, results, fingerprint)
    report = build_report(args, results, fingerprint)
    write_json_atomic(output_path, report)
    print(
        f"Finished {report['benchmark']}: primary={report['metrics']['primary_accuracy']:.4f}, "
        f"strict={report['metrics']['strict']['average']['all']['accuracy']:.4f}, "
        f"output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
