"""Prediction alignment, status accounting and machine-readable metrics."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Optional, Sequence

from .data import Dataset, public_sample_metadata
from .scoring import (
    DEFAULT_COORD_SCALE,
    DEFAULT_DRAG_DISTANCE_REL_TOLERANCE,
    DEFAULT_DRAG_Y_TOLERANCE,
    MIN_POINT_SEPARATION_PIXELS,
    evaluate_prediction,
    ground_truth_mode,
)


RESULT_STATUSES = (
    "correct",
    "wrong",
    "parse_error",
    "empty_response",
    "api_error",
    "missing_prediction",
)


class InvalidPredictionsError(ValueError):
    """Raised when prediction records cannot be aligned unambiguously."""


def load_prediction_records(path: str | Path) -> list[dict[str, Any]]:
    """Read either a JSON array/object or newline-delimited JSON objects."""

    prediction_path = Path(path).expanduser().resolve()
    try:
        text = prediction_path.read_text(encoding="utf-8")
    except OSError as error:
        raise InvalidPredictionsError(
            "cannot read predictions %s: %s" % (prediction_path, error)
        ) from error
    stripped = text.lstrip()
    if not stripped:
        return []

    if stripped.startswith("["):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise InvalidPredictionsError("invalid prediction JSON: %s" % error) from error
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise InvalidPredictionsError("prediction JSON must be an array of objects")
        return value

    # A pretty-printed single JSON object is accepted when the complete file
    # parses. Otherwise each non-empty line is treated as one JSONL record.
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        details = value.get("details")
        if isinstance(details, list) and all(isinstance(item, dict) for item in details):
            return details
        return [value]

    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise InvalidPredictionsError(
                "invalid JSONL at line %d: %s" % (line_number, error)
            ) from error
        if not isinstance(record, dict):
            raise InvalidPredictionsError(
                "prediction JSONL line %d must be an object" % line_number
            )
        records.append(record)
    return records


def _record_index(record: Mapping[str, Any]) -> Optional[int]:
    values = []
    for key in ("sample_index", "_source_index", "index"):
        if key not in record:
            continue
        value = record.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise InvalidPredictionsError("%s must be an integer" % key)
        values.append(value)
    if not values:
        return None
    if len(set(values)) != 1:
        raise InvalidPredictionsError("prediction record contains conflicting indices")
    return values[0]


def _record_sample_id(record: Mapping[str, Any]) -> Optional[str]:
    values = []
    for key in ("sample_id", "_sample_id"):
        if key not in record:
            continue
        value = record.get(key)
        if not isinstance(value, str) or not value:
            raise InvalidPredictionsError("%s must be a non-empty string" % key)
        values.append(value)
    if not values:
        return None
    if len(set(values)) != 1:
        raise InvalidPredictionsError("prediction record contains conflicting sample ids")
    return values[0]


def align_predictions(
    dataset: Dataset,
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, Mapping[str, Any]], list[str]]:
    """Align predictions by sample_id or source index and detect ambiguity."""

    id_to_index = {
        str(sample["_sample_id"]): int(sample["_source_index"])
        for sample in dataset.samples
    }
    valid_indices = {int(sample["_source_index"]) for sample in dataset.samples}
    aligned: dict[int, Mapping[str, Any]] = {}
    warnings = []

    identities = []
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise InvalidPredictionsError("prediction record %d must be an object" % position)
        sample_id = _record_sample_id(record)
        source_index = _record_index(record)
        identities.append((sample_id, source_index))
    keyed_modes = {
        sample_id is not None or source_index is not None
        for sample_id, source_index in identities
    }
    if len(keyed_modes) > 1:
        raise InvalidPredictionsError(
            "prediction file must use identities for every record or file order for every record"
        )

    for position, (record, identity) in enumerate(zip(records, identities)):
        sample_id, source_index = identity
        if sample_id is not None:
            if sample_id not in id_to_index:
                warnings.append("ignored unknown sample_id %s" % sample_id)
                continue
            id_index = id_to_index[sample_id]
            if source_index is not None and source_index != id_index:
                raise InvalidPredictionsError(
                    "record %d has conflicting sample_id and sample_index" % position
                )
            source_index = id_index
        elif source_index is None:
            if position >= len(dataset.samples):
                warnings.append("ignored unkeyed extra record at position %d" % position)
                continue
            source_index = int(dataset.samples[position]["_source_index"])

        if source_index not in valid_indices:
            warnings.append("ignored prediction for unselected sample_index %s" % source_index)
            continue
        if source_index in aligned:
            raise InvalidPredictionsError(
                "duplicate prediction for sample_index %d" % source_index
            )
        aligned[source_index] = record
    if records and keyed_modes == {True} and not aligned:
        raise InvalidPredictionsError(
            "no keyed prediction records match the selected dataset"
        )
    return aligned, warnings


def _model_output(record: Mapping[str, Any]) -> Any:
    for key in (
        "model_output",
        "prediction",
        "output",
        "response",
        "raw_response",
        "content",
    ):
        if key in record:
            return record.get(key)
    return None


def _is_api_error(record: Mapping[str, Any]) -> bool:
    diagnostics = record.get("api_diagnostics")
    if isinstance(diagnostics, Mapping) and diagnostics.get("success") is False:
        return True
    return bool(record.get("api_error"))


def _copy_optional_result_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = {}
    for key in (
        "reasoning_content",
        "inference",
        "api_diagnostics",
        "request_id",
    ):
        if key in record:
            fields[key] = record[key]
    return fields


def _evaluate_one(
    sample: Mapping[str, Any],
    record: Optional[Mapping[str, Any]],
    coord_scale: float,
    drag_distance_relative_tolerance: float,
    drag_y_tolerance: float,
) -> dict[str, Any]:
    detail = public_sample_metadata(sample)
    expected_mode = ground_truth_mode(sample)
    if record is None:
        detail.update(
            {
                "status": "missing_prediction",
                "correct": False,
                "mode": expected_mode,
                "reason": "missing_prediction",
                "model_output": None,
                "predicted_actions": [],
                "parser_errors": [],
                "call_failed": False,
            }
        )
        return detail

    output = _model_output(record)
    detail.update(_copy_optional_result_fields(record))
    detail["model_output"] = output
    if _is_api_error(record):
        diagnostics = record.get("api_diagnostics")
        error_type = (
            diagnostics.get("error_type")
            if isinstance(diagnostics, Mapping)
            else None
        )
        status = "empty_response" if error_type == "empty_response" else "api_error"
        detail.update(
            {
                "status": status,
                "correct": False,
                "mode": expected_mode,
                "reason": status,
                "predicted_actions": [],
                "parser_errors": [],
                "call_failed": True,
            }
        )
        return detail
    if not isinstance(output, str) or not output.strip():
        detail.update(
            {
                "status": "empty_response",
                "correct": False,
                "mode": expected_mode,
                "reason": "empty_response",
                "predicted_actions": [],
                "parser_errors": [],
                "call_failed": False,
            }
        )
        return detail

    score = evaluate_prediction(
        output,
        sample,
        coord_scale=coord_scale,
        drag_distance_relative_tolerance=drag_distance_relative_tolerance,
        drag_y_tolerance=drag_y_tolerance,
    )
    if score.correct:
        status = "correct"
    elif score.reason in ("format_error", "no_supported_action"):
        status = "parse_error"
    else:
        status = "wrong"
    detail.update(
        {
            "status": status,
            "correct": score.correct,
            "mode": score.mode,
            "reason": score.reason,
            "predicted_actions": list(score.actions),
            "parser_errors": list(score.parser_errors),
            "call_failed": False,
        }
    )
    return detail


def _aggregate(details: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(details)
    statuses = Counter(str(row["status"]) for row in rows)
    total = len(rows)
    correct = statuses["correct"]
    call_failed = sum(bool(row.get("call_failed")) for row in rows)
    valid_total = total - call_failed
    pass_at_1 = correct / total if total else 0.0
    valid_accuracy = correct / valid_total if valid_total else None
    parsed = sum(
        1
        for row in rows
        if row["status"] in ("correct", "wrong")
    )
    return {
        "total": total,
        "correct": correct,
        "pass_at_1": pass_at_1,
        # Retained as a compatibility alias for existing result consumers.
        "accuracy": pass_at_1,
        "call_failed": call_failed,
        "valid_total": valid_total,
        "valid_accuracy": valid_accuracy,
        "parsed": parsed,
        "parse_rate": parsed / total if total else 0.0,
        "statuses": {status: statuses[status] for status in RESULT_STATUSES},
        "failure_reasons": dict(
            sorted(
                Counter(
                    str(row.get("reason", "unknown"))
                    for row in rows
                    if not row.get("correct")
                ).items()
            )
        ),
    }


def evaluate_dataset(
    dataset: Dataset,
    records: Sequence[Mapping[str, Any]],
    coord_scale: float = DEFAULT_COORD_SCALE,
    drag_distance_relative_tolerance: float = DEFAULT_DRAG_DISTANCE_REL_TOLERANCE,
    drag_y_tolerance: float = DEFAULT_DRAG_Y_TOLERANCE,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate predictions and return a summary plus per-sample details."""

    aligned, warnings = align_predictions(dataset, records)
    details = [
        _evaluate_one(
            sample,
            aligned.get(int(sample["_source_index"])),
            coord_scale,
            drag_distance_relative_tolerance,
            drag_y_tolerance,
        )
        for sample in dataset.samples
    ]

    by_type_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_mode_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        by_type_rows[str(detail["captcha_type"])].append(detail)
        by_mode_rows[str(detail.get("mode") or "unknown")].append(detail)
    per_type = {
        name: _aggregate(rows) for name, rows in sorted(by_type_rows.items())
    }
    per_mode = {
        name: _aggregate(rows) for name, rows in sorted(by_mode_rows.items())
    }
    type_pass_at_1 = [metrics["pass_at_1"] for metrics in per_type.values()]
    valid_type_accuracies = [
        metrics["valid_accuracy"]
        for metrics in per_type.values()
        if metrics["valid_accuracy"] is not None
    ]
    macro_pass_at_1 = fmean(type_pass_at_1) if type_pass_at_1 else 0.0

    summary = {
        "dataset": {
            "name": dataset.name,
            "annotations": str(dataset.annotation_path),
            "sha256": dataset.sha256,
        },
        "scoring": {
            "primary_metric": "pass@1",
            "candidate_policy": "first_completion_only",
            "coord_scale": coord_scale,
            "drag_distance_relative_tolerance": drag_distance_relative_tolerance,
            "drag_y_tolerance_pixels": drag_y_tolerance,
            "minimum_point_separation_pixels": MIN_POINT_SEPARATION_PIXELS,
            "text_match": "strip_then_exact",
            "strict_action_set": True,
        },
        "overall": _aggregate(details),
        "macro_average": {
            "categories": len(per_type),
            "pass_at_1": macro_pass_at_1,
            # Retained as a compatibility alias for existing result consumers.
            "accuracy": macro_pass_at_1,
            "valid_accuracy": (
                fmean(valid_type_accuracies) if valid_type_accuracies else None
            ),
        },
        "per_captcha_type": per_type,
        "per_interaction_mode": per_mode,
        "warnings": warnings,
    }
    return summary, details


def _atomic_write(path: Path, content: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_json(path: str | Path, value: Any) -> None:
    _atomic_write(
        Path(path),
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        + "\n",
    )


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=False, allow_nan=False) + "\n"
        for record in records
    )
    _atomic_write(Path(path), content)
