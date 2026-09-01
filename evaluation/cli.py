"""Command-line interface for validation, inference and offline scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import urllib.parse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from . import __version__
from .backends import BACKEND_REGISTRY, BackendError, build_backend
from .data import Dataset, InvalidDatasetError, load_dataset
from .inference import infer_dataset
from .metrics import (
    InvalidPredictionsError,
    evaluate_dataset,
    load_prediction_records,
    write_json,
    write_jsonl,
)
from .reporting import write_html_report
from .scoring import (
    DEFAULT_COORD_SCALE,
    DEFAULT_DRAG_DISTANCE_REL_TOLERANCE,
    DEFAULT_DRAG_Y_TOLERANCE,
)


LOGGER = logging.getLogger("venusbench_captcha")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = str(REPOSITORY_ROOT / "instruction" / "VenusBench-CAPTCHA.json")
RUN_SIGNATURE_SCHEMA_VERSION = 4
_MODEL_FULL_HASH_LIMIT = 16 * 1024 * 1024
_MODEL_SAMPLE_BYTES = 1024 * 1024
_MODEL_FINGERPRINT_IGNORED_PARTS = frozenset({".git", "__pycache__"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--annotations",
        default=DEFAULT_ANNOTATIONS,
        help="annotation JSON array",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=None,
        help="only include these captcha_type values",
    )
    parser.add_argument(
        "--sample-indices",
        nargs="+",
        type=int,
        default=None,
        help="only include these original zero-based annotation indices",
    )
    parser.add_argument("--limit", type=int, default=None, help="maximum selected samples")


def _add_scoring_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--coord-scale",
        type=float,
        default=DEFAULT_COORD_SCALE,
        help="model coordinate upper bound; use 1 for normalized floats or 0 for pixels",
    )
    parser.add_argument(
        "--drag-distance-rel-tolerance",
        "--drag-dist-rel-tol",
        type=float,
        default=DEFAULT_DRAG_DISTANCE_REL_TOLERANCE,
        help="strict relative tolerance for signed horizontal drag distance",
    )
    parser.add_argument(
        "--drag-y-tolerance",
        "--drag-y-tol",
        type=float,
        default=DEFAULT_DRAG_Y_TOLERANCE,
        help="strict drag endpoint y tolerance in original-image pixels",
    )


def _output_prefix(predictions_path: str) -> Path:
    path = Path(predictions_path).expanduser()
    name = path.name
    if name.endswith(".jsonl"):
        name = name[:-6]
    elif name.endswith(".json"):
        name = name[:-5]
    if not name:
        name = "predictions"
    return path.parent / name


def _score_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Optional[Path]]:
    prefix = _output_prefix(args.predictions)
    metrics_path = Path(args.metrics_output) if args.metrics_output else Path(str(prefix) + ".metrics.json")
    details_path = Path(args.details_output) if args.details_output else Path(str(prefix) + ".scored.jsonl")
    if args.no_html:
        html_path = None
    else:
        html_path = Path(args.html_output) if args.html_output else Path(str(prefix) + ".report.html")
    return metrics_path, details_path, html_path


def _validate_artifact_paths(
    args: argparse.Namespace,
    dataset: Dataset,
    *,
    predictions_is_output: bool,
    include_reports: bool,
    include_manifest: bool,
) -> None:
    """Prevent output artifacts from overwriting each other or benchmark data."""

    artifacts: dict[str, Path] = {
        "predictions": Path(args.predictions).expanduser().resolve(),
    }
    output_names = {"predictions"} if predictions_is_output else set()
    if include_reports:
        metrics_path, details_path, html_path = _score_output_paths(args)
        artifacts["metrics"] = metrics_path.expanduser().resolve()
        artifacts["details"] = details_path.expanduser().resolve()
        output_names.update(("metrics", "details"))
        if html_path is not None:
            artifacts["html"] = html_path.expanduser().resolve()
            output_names.add("html")
    if include_manifest:
        artifacts["manifest"] = _manifest_path(args.predictions).expanduser().resolve()
        output_names.add("manifest")

    by_path: dict[Path, list[str]] = {}
    for name, path in artifacts.items():
        by_path.setdefault(path, []).append(name)
    collisions = [names for names in by_path.values() if len(names) > 1]
    if collisions:
        raise ValueError(
            "artifact paths must be distinct: %s"
            % "; ".join("/".join(names) for names in collisions)
        )

    protected = {dataset.annotation_path.resolve()}
    protected.update(path.resolve() for path in dataset.source_image_paths)
    for argument_name in (
        "system_prompt_file",
        "user_prompt_file",
        "extra_body_file",
    ):
        input_value = getattr(args, argument_name, None)
        if input_value:
            protected.add(Path(input_value).expanduser().resolve())
    for name in output_names:
        if artifacts[name] in protected:
            raise ValueError(
                "%s output must not overwrite benchmark or configuration inputs" % name
            )


def _add_report_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--metrics-output", default=None, help="summary JSON path")
    parser.add_argument("--details-output", default=None, help="per-sample scored JSONL path")
    parser.add_argument("--html-output", default=None, help="visual HTML report path")
    parser.add_argument("--no-html", action="store_true", help="do not generate HTML")


def _read_text_argument(value: Optional[str], file_value: Optional[str]) -> Optional[str]:
    if file_value:
        return Path(file_value).expanduser().read_text(encoding="utf-8")
    return value


def _reject_json_constant(constant: str) -> None:
    raise ValueError("non-finite JSON value: %s" % constant)


def _read_json_object(value: Optional[str], file_value: Optional[str]) -> dict[str, Any]:
    if file_value:
        raw = Path(file_value).expanduser().read_text(encoding="utf-8")
    elif value:
        raw = value
    else:
        return {}
    parsed = json.loads(raw, parse_constant=_reject_json_constant)
    if not isinstance(parsed, dict):
        raise ValueError("extra request body must be a JSON object")
    return parsed


def _load_selected_dataset(args: argparse.Namespace, check_images: bool) -> Dataset:
    return load_dataset(
        args.annotations,
        captcha_types=args.types,
        sample_indices=args.sample_indices,
        limit=args.limit,
        check_images=check_images,
    )


def _print_summary(summary: Mapping[str, Any]) -> None:
    per_type = summary["per_captcha_type"]
    print(
        "\n%-22s %7s %8s %10s"
        % ("captcha_type", "correct", "total", "pass@1")
    )
    print("-" * 53)
    for captcha_type, stats in per_type.items():
        print(
            "%-22s %7d %8d %9.2f%%"
            % (
                captcha_type,
                stats["correct"],
                stats["total"],
                100 * stats["pass_at_1"],
            )
        )
    overall = summary["overall"]
    macro = summary["macro_average"]
    print("-" * 53)
    print(
        "pass@1 (micro): %.2f%% (%d/%d), macro pass@1 over %d types: %.2f%%"
        % (
            100 * overall["pass_at_1"],
            overall["correct"],
            overall["total"],
            macro["categories"],
            100 * macro["pass_at_1"],
        )
    )
    print(
        "parsed: %d/%d; inference failures: %d"
        % (overall["parsed"], overall["total"], overall["call_failed"])
    )
    warnings = summary.get("warnings") or []
    if warnings:
        print("warnings: %d" % len(warnings))
        for warning in warnings[:10]:
            print("  - %s" % warning)
        if len(warnings) > 10:
            print("  - ... and %d more" % (len(warnings) - 10))


def _evaluate_and_write(
    args: argparse.Namespace,
    dataset: Dataset,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary, details = evaluate_dataset(
        dataset,
        records,
        coord_scale=args.coord_scale,
        drag_distance_relative_tolerance=args.drag_distance_rel_tolerance,
        drag_y_tolerance=args.drag_y_tolerance,
    )
    summary["evaluation"] = {
        "tool": "VenusBench-CAPTCHA evaluation",
        "version": __version__,
        "generated_at": _utc_now(),
        "predictions": str(Path(args.predictions).expanduser().resolve()),
    }
    metrics_path, details_path, html_path = _score_output_paths(args)
    write_json(metrics_path, summary)
    write_jsonl(details_path, details)
    if html_path is not None:
        write_html_report(
            html_path,
            dataset,
            summary,
            details,
            coord_scale=args.coord_scale,
        )
    _print_summary(summary)
    print("metrics: %s" % metrics_path)
    print("details: %s" % details_path)
    if html_path is not None:
        print("report:  %s" % html_path)
    return summary


def _manifest_path(predictions_path: str) -> Path:
    return Path(str(_output_prefix(predictions_path)) + ".manifest.json").resolve()


def _selected_images_sha256(dataset: Dataset) -> str:
    """Fingerprint selected image identities and bytes for safe resume."""

    digest = hashlib.sha256()
    for sample in dataset.samples:
        sample_id = sample["_sample_id"].encode("utf-8")
        image_path = Path(sample["_image_path"])
        try:
            image_bytes = image_path.read_bytes()
        except OSError as error:
            raise ValueError("cannot fingerprint image: %s" % image_path) from error
        image_digest = hashlib.sha256(image_bytes).digest()
        digest.update(len(sample_id).to_bytes(8, "big"))
        digest.update(sample_id)
        digest.update(image_digest)
    return digest.hexdigest()


def _local_model_reference(value: str) -> str:
    """Canonicalize an existing local checkpoint while preserving Hub IDs."""

    candidate = Path(value).expanduser()
    return str(candidate.resolve()) if candidate.exists() else value


def _hash_model_file(digest: Any, path: Path, size: int) -> None:
    """Hash a small artifact fully or stable samples from a large weight shard."""

    with path.open("rb") as handle:
        if size <= _MODEL_FULL_HASH_LIMIT:
            while True:
                chunk = handle.read(_MODEL_SAMPLE_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
            return

        last_offset = max(0, size - _MODEL_SAMPLE_BYTES)
        offsets = sorted(
            {
                0,
                max(0, size // 4 - _MODEL_SAMPLE_BYTES // 2),
                max(0, size // 2 - _MODEL_SAMPLE_BYTES // 2),
                max(0, 3 * size // 4 - _MODEL_SAMPLE_BYTES // 2),
                last_offset,
            }
        )
        for offset in offsets:
            handle.seek(offset)
            chunk = handle.read(min(_MODEL_SAMPLE_BYTES, size - offset))
            digest.update(offset.to_bytes(8, "big"))
            digest.update(len(chunk).to_bytes(8, "big"))
            digest.update(chunk)


def _local_model_artifact_fingerprint(value: str) -> Optional[dict[str, Any]]:
    """Fingerprint an existing local checkpoint without rereading all large weights."""

    candidate = Path(value).expanduser()
    if not candidate.exists():
        return None
    root = candidate.resolve()
    if root.is_file():
        paths = [root]
        kind = "local_file"
    elif root.is_dir():
        try:
            paths = sorted(
                (
                    path
                    for path in root.rglob("*")
                    if path.is_file()
                    and not _MODEL_FINGERPRINT_IGNORED_PARTS.intersection(
                        path.relative_to(root).parts
                    )
                ),
                key=lambda path: path.relative_to(root).as_posix(),
            )
        except OSError as error:
            raise ValueError("cannot enumerate local model artifacts: %s" % root) from error
        kind = "local_directory"
    else:
        raise ValueError("local model reference is not a file or directory: %s" % root)

    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        try:
            before = path.stat()
            relative_name = (
                root.name
                if kind == "local_file"
                else path.relative_to(root).as_posix()
            )
            name = relative_name.encode("utf-8")
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(before.st_size.to_bytes(8, "big"))
            digest.update(before.st_mtime_ns.to_bytes(8, "big", signed=True))
            _hash_model_file(digest, path, before.st_size)
            after = path.stat()
        except OSError as error:
            raise ValueError("cannot fingerprint local model artifact: %s" % path) from error
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise ValueError("local model artifact changed while fingerprinting: %s" % path)
        total_bytes += before.st_size

    return {
        "kind": kind,
        "strategy": "full-small-sampled-large-v1",
        "sha256": digest.hexdigest(),
        "files": len(paths),
        "bytes": total_bytes,
    }


def _signature_payload(
    args: argparse.Namespace,
    dataset: Dataset,
    system_prompt: Optional[str],
    user_prompt: Optional[str],
    generation_params: Mapping[str, Any],
    extra_body: Mapping[str, Any],
) -> dict[str, Any]:
    def prompt_fingerprint(value: Optional[str]) -> Optional[dict[str, Any]]:
        if value is None:
            return None
        encoded = value.encode("utf-8")
        return {
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "characters": len(value),
        }

    payload: dict[str, Any] = {
        "signature_schema_version": RUN_SIGNATURE_SCHEMA_VERSION,
        "tool_version": __version__,
        "dataset_sha256": dataset.sha256,
        "selected_images_sha256": _selected_images_sha256(dataset),
        "selected_sample_ids": [sample["_sample_id"] for sample in dataset.samples],
        "backend": args.backend,
        "generation_params": generation_params,
        "system_prompt_override": prompt_fingerprint(system_prompt),
        "user_prompt_override": prompt_fingerprint(user_prompt),
    }
    if args.backend == "vllm":
        artifact_fingerprint = _local_model_artifact_fingerprint(args.model)
        payload.update(
            {
                "model_name_or_path": _local_model_reference(args.model),
                "model_artifacts": artifact_fingerprint,
                "local_vllm": {
                    "revision": args.revision,
                    "dtype": args.dtype,
                    "tensor_parallel_size": args.tensor_parallel,
                    "batch_size": args.batch_size,
                    "max_model_len": args.max_model_len,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "trust_remote_code": args.trust_remote_code,
                    "min_pixels": args.min_pixels,
                    "max_pixels": args.max_pixels,
                },
            }
        )
        return payload

    parsed_url = urllib.parse.urlsplit(args.base_url)
    safe_host = parsed_url.hostname or ""
    if ":" in safe_host and not safe_host.startswith("["):
        safe_host = "[%s]" % safe_host
    if parsed_url.port is not None:
        safe_host += ":%d" % parsed_url.port
    safe_origin = urllib.parse.urlunsplit(
        (parsed_url.scheme, safe_host, "", "", "")
    )
    extra_body_bytes = json.dumps(
        extra_body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload.update(
        {
            "base_url_origin": safe_origin,
            "base_url_path_sha256": hashlib.sha256(
                parsed_url.path.encode("utf-8")
            ).hexdigest(),
            "base_url_query_sha256": (
                hashlib.sha256(parsed_url.query.encode("utf-8")).hexdigest()
                if parsed_url.query
                else None
            ),
            "model": args.model,
            "api_format": args.api_format,
            # Extra request bodies can contain provider-specific credentials under
            # arbitrary keys. Persist only a fingerprint, never their contents.
            "extra_body_fingerprint": {
                "sha256": hashlib.sha256(extra_body_bytes).hexdigest(),
                "bytes": len(extra_body_bytes),
            },
        }
    )
    return payload


def _run_signature(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _matching_resume_manifest(
    args: argparse.Namespace,
    payload: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    """Validate and return an existing resume manifest without mutating it."""

    predictions_path = Path(args.predictions).expanduser().resolve()
    if not args.resume or not predictions_path.exists():
        return None
    path = _manifest_path(args.predictions)
    if not path.is_file():
        raise ValueError("cannot safely resume without matching manifest: %s" % path)
    existing = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(existing, dict)
        or existing.get("run_signature") != _run_signature(payload)
    ):
        raise ValueError("resume manifest does not match this dataset/model/config")
    return existing


def _prepare_manifest(
    args: argparse.Namespace,
    dataset: Dataset,
    payload: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    path = _manifest_path(args.predictions)
    signature = _run_signature(payload)
    existing = _matching_resume_manifest(args, payload)
    if existing is not None:
        manifest = existing
        manifest["resumed_at"] = _utc_now()
        manifest["status"] = "running"
    else:
        manifest = {
            "schema_version": RUN_SIGNATURE_SCHEMA_VERSION,
            "tool_version": __version__,
            "created_at": _utc_now(),
            "status": "running",
            "run_signature": signature,
            "dataset": {
                "annotations": str(dataset.annotation_path),
                "sha256": dataset.sha256,
                "selected_samples": len(dataset.samples),
            },
            "config": dict(payload),
        }
    write_json(path, manifest)
    return path, manifest


def _generation_params(args: argparse.Namespace) -> dict[str, Any]:
    if args.backend == "vllm":
        params: dict[str, Any] = {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "presence_penalty": args.presence_penalty,
            "seed": args.seed,
        }
        if args.enable_thinking is not None:
            params["enable_thinking"] = args.enable_thinking
        return params

    params: dict[str, Any] = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "presence_penalty": args.presence_penalty,
    }
    if args.top_k is not None:
        params["top_k"] = args.top_k
    if args.seed is not None:
        params["seed"] = args.seed
    if args.enable_thinking is not None:
        params["chat_template_kwargs"] = {"enable_thinking": args.enable_thinking}
    return params


def _validate_scoring_configuration(args: argparse.Namespace) -> None:
    values = (
        ("coord_scale", args.coord_scale),
        ("drag_distance_rel_tolerance", args.drag_distance_rel_tolerance),
        ("drag_y_tolerance", args.drag_y_tolerance),
    )
    for name, value in values:
        if not math.isfinite(value):
            raise ValueError("%s must be finite" % name)
    if args.coord_scale < 0:
        raise ValueError("coord_scale must be 0 (pixels) or a positive upper bound")
    if args.drag_distance_rel_tolerance <= 0 or args.drag_y_tolerance <= 0:
        raise ValueError("drag tolerances must be positive")


def _validate_run_configuration(args: argparse.Namespace) -> None:
    if args.concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        raise ValueError("request_timeout must be finite and positive")
    if args.max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if args.max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        raise ValueError("top_p must be finite and in (0, 1]")
    if not math.isfinite(args.presence_penalty) or not -2 <= args.presence_penalty <= 2:
        raise ValueError("presence_penalty must be finite and in [-2, 2]")
    if args.top_k is not None and args.top_k < -1:
        raise ValueError("top_k must be greater than or equal to -1")
    if args.seed is not None and args.seed < 0:
        raise ValueError("seed must be non-negative when provided")
    if args.backend == "vllm":
        if args.concurrency != 1:
            raise ValueError("vllm offline inference requires --concurrency 1")
        if args.api_format != "openai":
            raise ValueError("vllm offline inference requires --api-format openai")
        for name in ("tensor_parallel", "batch_size", "max_model_len"):
            if getattr(args, name) < 1:
                raise ValueError("%s must be positive" % name)
        if (
            not math.isfinite(args.gpu_memory_utilization)
            or not 0 < args.gpu_memory_utilization <= 1
        ):
            raise ValueError("gpu_memory_utilization must be finite and in (0, 1]")
        for name in ("min_pixels", "max_pixels"):
            value = getattr(args, name)
            if value is not None and value <= 0:
                raise ValueError("%s must be positive" % name)
        if (
            args.min_pixels is not None
            and args.max_pixels is not None
            and args.min_pixels > args.max_pixels
        ):
            raise ValueError("min_pixels must not exceed max_pixels")


def command_validate(args: argparse.Namespace) -> int:
    dataset = _load_selected_dataset(args, check_images=not args.skip_image_check)
    counts = Counter(sample["captcha_type"] for sample in dataset.samples)
    print("valid samples: %d" % len(dataset.samples))
    print("annotation sha256: %s" % dataset.sha256)
    for captcha_type, count in sorted(counts.items()):
        print("  %-22s %d" % (captcha_type, count))
    return 0


def command_score(args: argparse.Namespace) -> int:
    _validate_scoring_configuration(args)
    dataset = _load_selected_dataset(args, check_images=args.check_images)
    _validate_artifact_paths(
        args,
        dataset,
        predictions_is_output=False,
        include_reports=True,
        include_manifest=False,
    )
    records = load_prediction_records(args.predictions)
    _evaluate_and_write(args, dataset, records)
    return 0


def command_run(args: argparse.Namespace) -> int:
    _validate_scoring_configuration(args)
    _validate_run_configuration(args)
    predictions_path = Path(args.predictions).expanduser()
    if predictions_path.exists() and not args.resume and not args.overwrite:
        raise ValueError(
            "predictions already exist; pass --resume or --overwrite: %s"
            % predictions_path
        )
    dataset = _load_selected_dataset(args, check_images=True)
    _validate_artifact_paths(
        args,
        dataset,
        predictions_is_output=True,
        include_reports=not args.inference_only,
        include_manifest=True,
    )
    system_prompt = _read_text_argument(args.system_prompt, args.system_prompt_file)
    user_prompt = _read_text_argument(args.user_prompt, args.user_prompt_file)
    if args.coord_scale != 999.0 and system_prompt is None:
        raise ValueError(
            "non-default --coord-scale requires --system-prompt or "
            "--system-prompt-file with matching coordinate instructions"
        )
    extra_body = _read_json_object(args.extra_body_json, args.extra_body_file)
    if args.backend == "vllm" and extra_body:
        raise ValueError("--extra-body-json/--extra-body-file require an API backend")
    generation_params = _generation_params(args)
    payload = _signature_payload(
        args, dataset, system_prompt, user_prompt, generation_params, extra_body
    )
    # Refuse a mismatched resume before spending time and GPU memory loading a
    # local checkpoint. _prepare_manifest repeats the check immediately before
    # changing the manifest status.
    _matching_resume_manifest(args, payload)
    if args.backend == "vllm":
        backend = build_backend(
            args.backend,
            model_name_or_path=_local_model_reference(args.model),
            generation_params=generation_params,
            tensor_parallel_size=args.tensor_parallel,
            batch_size=args.batch_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        expected_artifacts = payload.get("model_artifacts")
        if expected_artifacts is not None:
            loaded_artifacts = _local_model_artifact_fingerprint(args.model)
            if loaded_artifacts != expected_artifacts:
                raise ValueError("local model artifacts changed while loading the model")
    else:
        backend = build_backend(
            args.backend,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=args.request_timeout,
            max_retries=args.max_retries,
            generation_params=generation_params,
            extra_body=extra_body,
        )
    manifest_path, manifest = _prepare_manifest(args, dataset, payload)
    try:
        records = infer_dataset(
            dataset,
            backend,
            predictions_path,
            concurrency=args.concurrency,
            resume=args.resume,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_format=args.api_format,
        )
    except BaseException:
        manifest["status"] = "interrupted"
        manifest["updated_at"] = _utc_now()
        write_json(manifest_path, manifest)
        raise

    manifest["status"] = "complete"
    manifest["completed_at"] = _utc_now()
    manifest["prediction_records"] = len(records)
    write_json(manifest_path, manifest)
    print("predictions: %s" % predictions_path)
    print("manifest:    %s" % manifest_path)
    if not args.inference_only:
        _evaluate_and_write(args, dataset, records)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate, run and score VenusBench-CAPTCHA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--verbose", action="store_true", help="enable debug logs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="validate annotation, action and image integrity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_dataset_arguments(validate_parser)
    validate_parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="skip image existence/decode/dimension checks",
    )
    validate_parser.set_defaults(handler=command_validate)

    score_parser = subparsers.add_parser(
        "score",
        help="score an existing prediction JSON/JSONL file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_dataset_arguments(score_parser)
    _add_scoring_arguments(score_parser)
    _add_report_arguments(score_parser)
    score_parser.add_argument("--predictions", required=True)
    score_parser.add_argument(
        "--check-images", action="store_true", help="also decode and verify every image"
    )
    score_parser.set_defaults(handler=command_score)

    run_parser = subparsers.add_parser(
        "run",
        help="run local vLLM or an OpenAI-compatible API and score predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_dataset_arguments(run_parser)
    _add_scoring_arguments(run_parser)
    _add_report_arguments(run_parser)
    run_parser.add_argument("--predictions", default="results/predictions.jsonl")
    run_parser.add_argument(
        "--backend",
        choices=tuple(sorted(BACKEND_REGISTRY)),
        default="vllm",
    )
    run_parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    run_parser.add_argument(
        "--model-name-or-path",
        "--model_name_or_path",
        "--model",
        dest="model",
        required=True,
        help="local checkpoint/Hugging Face model ID, or served model name",
    )
    run_parser.add_argument("--api-key", default=None, help="defaults to OPENAI_API_KEY")
    run_parser.add_argument("--api-format", choices=("openai", "antchat"), default="openai")
    run_parser.add_argument("--concurrency", type=int, default=1)
    run_parser.add_argument("--request-timeout", type=float, default=300.0)
    run_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help=(
            "infrastructure retries before a valid completion; an empty "
            "completion is terminal"
        ),
    )
    run_parser.add_argument(
        "--max-new-tokens",
        "--max_new_tokens",
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=25600,
    )
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--top-p", type=float, default=0.95)
    run_parser.add_argument("--top-k", type=int, default=20)
    run_parser.add_argument("--presence-penalty", type=float, default=0.0)
    run_parser.add_argument("--seed", type=int, default=996)
    run_parser.add_argument(
        "--tensor-parallel",
        "--tensor-parallel-size",
        dest="tensor_parallel",
        type=int,
        default=1,
        help="GPUs used by one local vLLM engine",
    )
    run_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="conversations submitted per local vLLM chat batch",
    )
    run_parser.add_argument(
        "--max-model-len",
        type=int,
        default=50000,
        help="local vLLM context length; configure API servers separately",
    )
    run_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="fraction of GPU memory reserved by local vLLM",
    )
    run_parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="vLLM model dtype",
    )
    run_parser.add_argument("--revision", default=None, help="local/Hugging Face revision")
    run_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow checkpoint-provided Python code during local loading",
    )
    run_parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="optional minimum visual tokenization area",
    )
    run_parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="optional maximum visual tokenization area",
    )
    thinking = run_parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="enable the model chat template's thinking mode",
    )
    thinking.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="disable the model chat template's thinking mode",
    )
    run_parser.set_defaults(enable_thinking=None)
    system = run_parser.add_mutually_exclusive_group()
    system.add_argument("--system-prompt", default=None, help="replace dataset system prompts")
    system.add_argument(
        "--system-prompt-file",
        default=None,
        help="UTF-8 file that replaces dataset system prompts",
    )
    user = run_parser.add_mutually_exclusive_group()
    user.add_argument(
        "--user-prompt",
        default=None,
        help="replace user text; {user_text} inserts the dataset text",
    )
    user.add_argument(
        "--user-prompt-file",
        default=None,
        help="UTF-8 user prompt template file",
    )
    extra = run_parser.add_mutually_exclusive_group()
    extra.add_argument("--extra-body-json", default=None, help="extra request-body JSON object")
    extra.add_argument("--extra-body-file", default=None, help="extra request-body JSON file")
    mode = run_parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="resume a manifest-matched run")
    mode.add_argument("--overwrite", action="store_true", help="rerun and replace predictions")
    run_parser.add_argument("--inference-only", action="store_true", help="skip scoring/reporting")
    run_parser.set_defaults(handler=command_run)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    try:
        return int(args.handler(args))
    except (
        BackendError,
        InvalidDatasetError,
        InvalidPredictionsError,
        ValueError,
        OSError,
    ) as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
