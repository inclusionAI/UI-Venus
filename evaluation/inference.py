"""Message construction and crash-resilient dataset inference."""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Optional

from .backends import BackendError, BaseBackend
from .data import (
    Dataset,
    public_sample_metadata,
    sample_system_prompt,
    sample_user_text,
)


logger = logging.getLogger(__name__)

_OUTPUT_FIELDS = frozenset(
    {
        "sample_index",
        "sample_id",
        "image",
        "captcha_type",
        "model_output",
        "reasoning_content",
        "usage",
        "finish_reason",
        "request_id",
        "inference",
        "api_diagnostics",
    }
)


def _detect_image_mime(data: bytes) -> str:
    """Identify a supported image from its bytes rather than its filename."""

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    raise ValueError("unsupported image signature")


def image_to_data_url(path: str | Path) -> str:
    """Read an image, detect its MIME type, and return a base64 data URL."""

    image_path = Path(path).expanduser()
    try:
        data = image_path.read_bytes()
    except OSError as error:
        raise ValueError("cannot read image: %s" % image_path) from error
    mime = _detect_image_mime(data)
    payload = base64.b64encode(data).decode("ascii")
    return "data:%s;base64,%s" % (mime, payload)


def _validate_prompt(value: Optional[str], name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise TypeError("%s must be a string or None" % name)


def build_messages(
    sample: Mapping[str, Any],
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    api_format: str = "openai",
    image_transport: str = "data-url",
) -> list[dict[str, Any]]:
    """Build one multimodal chat request without exposing ground truth fields."""

    if not isinstance(sample, Mapping):
        raise TypeError("sample must be a mapping")
    _validate_prompt(system_prompt, "system_prompt")
    _validate_prompt(user_prompt, "user_prompt")
    if api_format not in ("openai", "antchat"):
        raise ValueError("api_format must be 'openai' or 'antchat'")
    if image_transport not in ("data-url", "path"):
        raise ValueError("image_transport must be 'data-url' or 'path'")

    image_path = sample.get("_image_path")
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("sample must contain a non-empty _image_path")

    system_text = (
        system_prompt if system_prompt is not None else sample_system_prompt(sample)
    )
    original_user_text = sample_user_text(sample)
    user_text = original_user_text
    if user_prompt is not None:
        user_text = user_prompt.replace("{user_text}", original_user_text)

    messages: list[dict[str, Any]] = []
    if system_text:
        system_content: Any = system_text
        if image_transport == "path":
            system_content = [{"type": "text", "text": system_text}]
        messages.append({"role": "system", "content": system_content})

    if image_transport == "path":
        image_part: dict[str, Any] = {
            "type": "image",
            "path": str(Path(image_path).expanduser().resolve()),
        }
    elif api_format == "antchat":
        data_url = image_to_data_url(image_path)
        image_part: dict[str, Any] = {
            "type": "image_url",
            "image_url": data_url,
        }
    else:
        data_url = image_to_data_url(image_path)
        image_part = {
            "type": "image_url",
            "image_url": {"url": data_url},
        }
    user_content: list[dict[str, Any]] = []
    if image_transport == "path":
        # Match the ordering used by Qwen/UI-Venus multimodal templates.
        user_content.append(image_part)
    if user_text:
        user_content.append({"type": "text", "text": user_text})
    if image_transport != "path":
        user_content.append(image_part)
    messages.append({"role": "user", "content": user_content})
    return messages


def _json_copy(value: Any) -> Any:
    """Return a JSON-safe copy, rejecting NaN and arbitrary Python objects."""

    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    return json.loads(encoded)


def _duration(value: Any, fallback: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        converted = float(value)
        if math.isfinite(converted) and converted >= 0:
            return converted
    return fallback


def _attempts(value: Any, fallback: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return fallback


def _success_record(
    sample: Mapping[str, Any],
    response: Mapping[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    output = response.get("model_output", "")
    reasoning = response.get("reasoning_content", "")
    if output is None:
        output = ""
    if reasoning is None:
        reasoning = ""
    if not isinstance(output, str) or not isinstance(reasoning, str):
        raise ValueError("backend text fields must be strings")

    usage = response.get("usage", {})
    if usage is None:
        usage = {}
    if not isinstance(usage, Mapping):
        raise ValueError("backend usage must be a mapping")
    usage = _json_copy(dict(usage))
    finish_reason = _json_copy(response.get("finish_reason"))
    request_id = _json_copy(response.get("request_id"))
    attempts = _attempts(response.get("attempts"), 1)
    latency = _duration(response.get("latency"), elapsed)

    record = public_sample_metadata(sample)
    record.update(
        {
            "model_output": output,
            "reasoning_content": reasoning,
            "usage": usage,
            "finish_reason": finish_reason,
            "request_id": request_id,
            "inference": {
                "latency_seconds": round(latency, 6),
                "attempts": attempts,
            },
            "api_diagnostics": {"success": True},
        }
    )
    return record


def _failure_record(
    sample: Mapping[str, Any],
    elapsed: float,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    safe_diagnostics = _json_copy(dict(diagnostics))
    safe_diagnostics["success"] = False
    attempts = _attempts(safe_diagnostics.get("attempts"), 0)
    record = public_sample_metadata(sample)
    record.update(
        {
            "model_output": "",
            "reasoning_content": "",
            "usage": {},
            "finish_reason": None,
            "inference": {
                "latency_seconds": round(elapsed, 6),
                "attempts": attempts,
            },
            "api_diagnostics": safe_diagnostics,
        }
    )
    return record


def _infer_one(
    sample: Mapping[str, Any],
    backend: BaseBackend,
    system_prompt: Optional[str],
    user_prompt: Optional[str],
    api_format: str,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        messages = build_messages(
            sample,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_format=api_format,
            image_transport=getattr(backend, "image_transport", "data-url"),
        )
        response = backend.infer(messages)
        if not isinstance(response, Mapping):
            raise ValueError("backend response must be a mapping")
        return _success_record(sample, response, time.perf_counter() - started_at)
    except BackendError as error:
        if error.fatal:
            raise
        return _failure_record(
            sample,
            time.perf_counter() - started_at,
            error.diagnostics,
        )
    except Exception as error:
        # Never persist exception messages: they can contain URLs, prompts, or keys.
        return _failure_record(
            sample,
            time.perf_counter() - started_at,
            {
                "error_type": "inference_error",
                "cause_type": type(error).__name__,
                "attempts": 0,
                "retryable": False,
                "status_code": None,
            },
        )


def _infer_batch(
    samples: Sequence[Mapping[str, Any]],
    backend: BaseBackend,
    system_prompt: Optional[str],
    user_prompt: Optional[str],
    api_format: str,
) -> list[dict[str, Any]]:
    started_at = time.perf_counter()
    try:
        conversations = [
            build_messages(
                sample,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                api_format=api_format,
                image_transport=getattr(backend, "image_transport", "data-url"),
            )
            for sample in samples
        ]
        responses = backend.infer_batch(conversations)
        if isinstance(responses, (str, bytes)) or not isinstance(responses, Sequence):
            raise TypeError("backend infer_batch() must return a sequence")
        if len(responses) != len(samples):
            raise ValueError("backend infer_batch() returned the wrong number of responses")
        if not all(isinstance(response, Mapping) for response in responses):
            raise TypeError("every batch response must be a mapping")
        elapsed = time.perf_counter() - started_at
        return [
            _success_record(sample, response, elapsed)
            for sample, response in zip(samples, responses)
        ]
    except BackendError as error:
        if error.fatal:
            raise
        elapsed = time.perf_counter() - started_at
        return [
            _failure_record(sample, elapsed, error.diagnostics) for sample in samples
        ]
    except Exception as error:
        elapsed = time.perf_counter() - started_at
        diagnostics = {
            "error_type": "inference_error",
            "cause_type": type(error).__name__,
            "attempts": 0,
            "retryable": False,
            "status_code": None,
        }
        return [_failure_record(sample, elapsed, diagnostics) for sample in samples]


def _validate_runner_arguments(
    dataset: Dataset,
    backend: BaseBackend,
    output_path: str | Path,
    concurrency: int,
    resume: bool,
    system_prompt: Optional[str],
    user_prompt: Optional[str],
    api_format: str,
) -> Path:
    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be a Dataset")
    if not callable(getattr(backend, "infer", None)):
        raise TypeError("backend must provide infer(messages)")
    batch_size = getattr(backend, "batch_size", None)
    if batch_size is not None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError("backend batch_size must be a positive integer or None")
        if not callable(getattr(backend, "infer_batch", None)):
            raise TypeError("backend with batch_size must provide infer_batch(conversations)")
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    max_concurrency = getattr(backend, "max_concurrency", None)
    if max_concurrency is not None:
        if (
            not isinstance(max_concurrency, int)
            or isinstance(max_concurrency, bool)
            or max_concurrency < 1
        ):
            raise ValueError("backend max_concurrency must be a positive integer or None")
        if concurrency > max_concurrency:
            raise ValueError(
                "backend %s supports concurrency at most %d"
                % (type(backend).__name__, max_concurrency)
            )
    image_transport = getattr(backend, "image_transport", "data-url")
    if image_transport not in ("data-url", "path"):
        raise ValueError("backend image_transport must be 'data-url' or 'path'")
    if not isinstance(resume, bool):
        raise TypeError("resume must be a boolean")
    _validate_prompt(system_prompt, "system_prompt")
    _validate_prompt(user_prompt, "user_prompt")
    if api_format not in ("openai", "antchat"):
        raise ValueError("api_format must be 'openai' or 'antchat'")
    if not isinstance(output_path, (str, os.PathLike)) or not str(output_path):
        raise TypeError("output_path must be a non-empty path")

    path = Path(output_path).expanduser().resolve()
    if path == dataset.annotation_path.resolve():
        raise ValueError("output_path must not overwrite the annotation file")

    seen_indices: set[int] = set()
    seen_ids: set[str] = set()
    for position, sample in enumerate(dataset.samples):
        index = sample.get("_source_index")
        sample_id = sample.get("_sample_id")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("dataset sample %d has an invalid _source_index" % position)
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("dataset sample %d has an invalid _sample_id" % position)
        if index in seen_indices or sample_id in seen_ids:
            raise ValueError("dataset contains duplicate sample identity")
        seen_indices.add(index)
        seen_ids.add(sample_id)
    return path


def _load_resume_records(
    path: Path,
    dataset: Dataset,
) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    if not path.is_file():
        raise ValueError("output_path is not a file: %s" % path)

    by_index = {
        int(sample["_source_index"]): sample for sample in dataset.samples
    }
    records: dict[int, dict[str, Any]] = {}
    try:
        raw_lines = path.read_bytes().splitlines(keepends=True)
    except OSError as error:
        raise ValueError("cannot read resume JSONL: %s" % path) from error
    nonempty_positions = [
        position for position, raw_line in enumerate(raw_lines) if raw_line.strip()
    ]
    last_nonempty_position = nonempty_positions[-1] if nonempty_positions else None
    for position, raw_line in enumerate(raw_lines):
        line_number = position + 1
        if not raw_line.strip():
            continue
        try:
            line = raw_line.decode("utf-8")
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            is_unterminated_tail = (
                position == last_nonempty_position
                and not raw_line.endswith((b"\n", b"\r"))
            )
            if is_unterminated_tail:
                logger.warning(
                    "Dropping incomplete final resume record at line %d", line_number
                )
                break
            raise ValueError(
                "invalid resume JSONL at line %d: %s" % (line_number, error)
            ) from error
        if not isinstance(record, dict):
            raise ValueError("resume JSONL line %d must be an object" % line_number)
        unknown_fields = set(record).difference(_OUTPUT_FIELDS)
        if unknown_fields:
            raise ValueError(
                "resume JSONL line %d has unsupported fields: %s"
                % (line_number, ", ".join(sorted(unknown_fields)))
            )

        index = record.get("sample_index")
        sample_id = record.get("sample_id")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("resume JSONL line %d has invalid sample_index" % line_number)
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("resume JSONL line %d has invalid sample_id" % line_number)
        sample = by_index.get(index)
        if sample is None:
            raise ValueError(
                "resume JSONL line %d references unknown sample_index %d"
                % (line_number, index)
            )
        if sample_id != sample["_sample_id"]:
            raise ValueError(
                "resume JSONL line %d has conflicting sample_id/sample_index"
                % line_number
            )
        if index in records:
            raise ValueError("duplicate resume sample_index %d" % index)
        if record.get("image") != sample["images"][0]:
            raise ValueError("resume JSONL line %d has conflicting image" % line_number)
        if record.get("captcha_type") != sample["captcha_type"]:
            raise ValueError(
                "resume JSONL line %d has conflicting captcha_type" % line_number
            )
        records[index] = record
    return records


def _record_line(record: Mapping[str, Any]) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=False,
        allow_nan=False,
        separators=(",", ":"),
    ) + "\n"


def _append_record(handle: Any, record: Mapping[str, Any]) -> None:
    # Serialize before touching the file, so serialization errors cannot leave a tail.
    line = _record_line(record)
    handle.write(line)
    handle.flush()
    os.fsync(handle.fileno())


def _atomic_rewrite(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(_record_line(record))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def infer_dataset(
    dataset: Dataset,
    backend: BaseBackend,
    output_path: str | Path,
    concurrency: int = 1,
    resume: bool = False,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    api_format: str = "openai",
) -> list[dict[str, Any]]:
    """Infer a dataset with bounded concurrency and durable per-sample commits."""

    path = _validate_runner_arguments(
        dataset,
        backend,
        output_path,
        concurrency,
        resume,
        system_prompt,
        user_prompt,
        api_format,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_resume_records(path, dataset) if resume else {}
    # Normalize an existing checkpoint before appending. Besides restoring
    # deterministic order, this guarantees a trailing newline even when a
    # hand-written or interrupted JSONL ended directly after its last object.
    if resume and path.exists():
        _atomic_rewrite(path, [completed[index] for index in sorted(completed)])
    pending_samples = [
        sample
        for sample in dataset.samples
        if int(sample["_source_index"]) not in completed
    ]
    total = len(dataset.samples)
    logger.info(
        "Inference start: total=%d pending=%d resumed=%d concurrency=%d",
        total,
        len(pending_samples),
        len(completed),
        concurrency,
    )

    mode = "a" if resume else "w"
    newly_completed = 0
    failed = sum(
        1
        for record in completed.values()
        if isinstance(record.get("api_diagnostics"), Mapping)
        and record["api_diagnostics"].get("success") is False
    )

    def commit_record(
        output: Any,
        sample: Mapping[str, Any],
        record: Mapping[str, Any],
    ) -> None:
        nonlocal newly_completed, failed
        index = int(sample["_source_index"])
        if int(record["sample_index"]) != index:
            raise RuntimeError("worker returned a conflicting sample_index")
        _append_record(output, record)
        completed[index] = dict(record)
        newly_completed += 1
        diagnostics = record.get("api_diagnostics")
        if isinstance(diagnostics, Mapping) and diagnostics.get("success") is False:
            failed += 1
        finished_count = len(completed)
        if newly_completed % 10 == 0 or finished_count == total:
            logger.info(
                "Inference progress: %d/%d complete, failures=%d",
                finished_count,
                total,
                failed,
            )

    batch_size = getattr(backend, "batch_size", None)
    if batch_size is not None:
        try:
            with path.open(mode, encoding="utf-8") as output:
                for offset in range(0, len(pending_samples), batch_size):
                    samples = pending_samples[offset : offset + batch_size]
                    records = _infer_batch(
                        samples,
                        backend,
                        system_prompt,
                        user_prompt,
                        api_format,
                    )
                    for sample, record in zip(samples, records):
                        commit_record(output, sample, record)
        except BaseException:
            logger.warning(
                "Inference interrupted after %d/%d durable record(s); resume is safe",
                len(completed),
                total,
            )
            raise
    else:
        executor = ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="venus-infer"
        )
        in_flight: dict[Future[dict[str, Any]], Mapping[str, Any]] = {}
        iterator = iter(pending_samples)

        def submit_next() -> bool:
            try:
                sample = next(iterator)
            except StopIteration:
                return False
            future = executor.submit(
                _infer_one,
                sample,
                backend,
                system_prompt,
                user_prompt,
                api_format,
            )
            in_flight[future] = sample
            return True

        try:
            with path.open(mode, encoding="utf-8") as output:
                for _ in range(min(concurrency, len(pending_samples))):
                    submit_next()

                while in_flight:
                    done, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
                    for future in done:
                        sample = in_flight.pop(future)
                        record = future.result()
                        commit_record(output, sample, record)
                        submit_next()
        except BaseException:
            for future in in_flight:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            logger.warning(
                "Inference interrupted after %d/%d durable record(s); resume is safe",
                len(completed),
                total,
            )
            raise
        else:
            executor.shutdown(wait=True)

    sorted_records = [completed[index] for index in sorted(completed)]
    if len(sorted_records) != total:
        raise RuntimeError(
            "inference completed with %d/%d records" % (len(sorted_records), total)
        )
    _atomic_rewrite(path, sorted_records)
    logger.info(
        "Inference complete: %d records, failures=%d, output=%s",
        total,
        failed,
        path,
    )
    return sorted_records


__all__ = ["build_messages", "image_to_data_url", "infer_dataset"]
