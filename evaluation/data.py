"""Dataset loading, validation and stable sample identity."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .scoring import InvalidGroundTruthError, validate_ground_truth


REQUIRED_FIELDS = frozenset(
    (
        "images",
        "messages",
        "image_size",
        "captcha_type",
        "action_raw",
        "action_raw_rect",
        "inorder",
    )
)


class InvalidDatasetError(ValueError):
    """Raised when annotations or referenced images violate the contract."""


@dataclass(frozen=True)
class Dataset:
    annotation_path: Path
    sha256: str
    samples: tuple[dict[str, Any], ...]
    source_image_paths: tuple[Path, ...]

    @property
    def name(self) -> str:
        return self.annotation_path.stem


def _text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def message_text(sample: Mapping[str, Any], role: str) -> str:
    """Extract text for a role from string or segmented message content."""

    for message in sample.get("messages") or []:
        if isinstance(message, Mapping) and message.get("role") == role:
            return _text_content(message.get("content"))
    return ""


def sample_system_prompt(sample: Mapping[str, Any]) -> str:
    return message_text(sample, "system").strip()


def sample_user_text(sample: Mapping[str, Any]) -> str:
    """Return user text without the image placeholder used by SFT data."""

    text = message_text(sample, "user")
    return text.replace("<image>", "").strip()


def make_sample_id(source_index: int, recorded_image_path: str) -> str:
    """Build a stable, readable id without changing the released annotations."""

    return "%04d:%s" % (source_index, recorded_image_path)


def resolve_image_path(annotation_path: Path, recorded_path: str) -> Path:
    """Resolve an image relative to its annotation file, never the process CWD."""

    path = Path(recorded_path).expanduser()
    if not path.is_absolute():
        path = annotation_path.parent / path
    return path.resolve()


def _validate_messages(messages: Any) -> None:
    if not isinstance(messages, list) or not messages:
        raise InvalidDatasetError("messages must be a non-empty list")
    roles = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise InvalidDatasetError("messages[%d] must be an object" % index)
        role = message.get("role")
        if role not in ("system", "user"):
            raise InvalidDatasetError(
                "messages[%d].role must be system or user" % index
            )
        roles.append(role)
    if "user" not in roles:
        raise InvalidDatasetError("messages must contain a user message")


def validate_sample_structure(
    sample: Mapping[str, Any],
    annotation_path: Path,
    source_index: int,
    check_image: bool = False,
) -> Path:
    """Validate schema, strict ground truth, path and optionally image size."""

    missing = sorted(REQUIRED_FIELDS.difference(sample))
    if missing:
        raise InvalidDatasetError("missing fields: %s" % ", ".join(missing))

    images = sample.get("images")
    if (
        not isinstance(images, list)
        or len(images) != 1
        or not isinstance(images[0], str)
        or not images[0]
    ):
        raise InvalidDatasetError("images must contain exactly one non-empty path")
    if not isinstance(sample.get("captcha_type"), str) or not sample.get("captcha_type"):
        raise InvalidDatasetError("captcha_type must be a non-empty string")
    if not isinstance(sample.get("inorder"), bool):
        raise InvalidDatasetError("inorder must be a boolean")
    _validate_messages(sample.get("messages"))

    try:
        validate_ground_truth(sample)
    except InvalidGroundTruthError as error:
        raise InvalidDatasetError(str(error)) from error

    image_path = resolve_image_path(annotation_path, images[0])
    if check_image:
        if not image_path.is_file():
            raise InvalidDatasetError("image does not exist: %s" % image_path)
        try:
            from PIL import Image
        except ImportError as error:
            raise InvalidDatasetError(
                "Pillow is required for --check-images (pip install Pillow)"
            ) from error
        try:
            with Image.open(image_path) as image:
                actual_size = tuple(image.size)
                image.verify()
        except Exception as error:
            raise InvalidDatasetError("cannot decode image %s: %s" % (image_path, error)) from error
        expected_size = tuple(sample["image_size"])
        if actual_size != expected_size:
            raise InvalidDatasetError(
                "image_size %s differs from decoded size %s for %s"
                % (expected_size, actual_size, image_path)
            )
    return image_path


def _normalize_indices(indices: Optional[Iterable[int]], total: int) -> Optional[set[int]]:
    if indices is None:
        return None
    selected = set(indices)
    invalid = sorted(index for index in selected if index < 0 or index >= total)
    if invalid:
        raise InvalidDatasetError(
            "sample indices outside [0, %d]: %s" % (total - 1, invalid)
        )
    return selected


def load_dataset(
    annotation_path: str | Path,
    captcha_types: Optional[Sequence[str]] = None,
    sample_indices: Optional[Iterable[int]] = None,
    limit: Optional[int] = None,
    check_images: bool = False,
) -> Dataset:
    """Load an annotation array and enrich copies with private runtime fields."""

    path = Path(annotation_path).expanduser().resolve()
    try:
        raw_bytes = path.read_bytes()
    except OSError as error:
        raise InvalidDatasetError("cannot read annotations %s: %s" % (path, error)) from error
    try:
        raw_samples = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidDatasetError("invalid UTF-8 JSON in %s: %s" % (path, error)) from error
    if not isinstance(raw_samples, list):
        raise InvalidDatasetError("annotation root must be a JSON array")
    if limit is not None and limit < 0:
        raise InvalidDatasetError("limit must be non-negative")

    # Keep every syntactically discoverable source image protected, even when
    # the caller evaluates only a subset. This prevents report/output paths
    # from accidentally overwriting an unselected benchmark image.
    source_image_paths = set()
    for raw_sample in raw_samples:
        if not isinstance(raw_sample, Mapping):
            continue
        images = raw_sample.get("images")
        if not isinstance(images, list):
            continue
        for recorded_path in images:
            if isinstance(recorded_path, str) and recorded_path:
                source_image_paths.add(resolve_image_path(path, recorded_path))

    selected_indices = _normalize_indices(sample_indices, len(raw_samples))
    selected_types = set(captcha_types) if captcha_types else None
    known_types = {
        raw_sample.get("captcha_type")
        for raw_sample in raw_samples
        if isinstance(raw_sample, Mapping)
        and isinstance(raw_sample.get("captcha_type"), str)
        and raw_sample.get("captcha_type")
    }
    if selected_types is not None:
        unknown_types = sorted(selected_types.difference(known_types))
        if unknown_types:
            raise InvalidDatasetError(
                "unknown captcha_type selection: %s; available: %s"
                % (", ".join(unknown_types), ", ".join(sorted(known_types)))
            )
    samples = []
    errors = []
    seen_ids = set()
    for source_index, raw_sample in enumerate(raw_samples):
        if selected_indices is not None and source_index not in selected_indices:
            continue
        if not isinstance(raw_sample, Mapping):
            errors.append("[%d] sample must be an object" % source_index)
            continue
        if selected_types is not None and raw_sample.get("captcha_type") not in selected_types:
            continue
        if limit is not None and len(samples) >= limit:
            break
        try:
            image_path = validate_sample_structure(
                raw_sample,
                path,
                source_index,
                check_image=check_images,
            )
        except InvalidDatasetError as error:
            errors.append("[%d] %s" % (source_index, error))
            continue

        sample = copy.deepcopy(dict(raw_sample))
        sample_id = make_sample_id(source_index, sample["images"][0])
        if sample_id in seen_ids:
            errors.append("[%d] duplicate sample_id %s" % (source_index, sample_id))
            continue
        seen_ids.add(sample_id)
        sample["_source_index"] = source_index
        sample["_sample_id"] = sample_id
        sample["_image_path"] = str(image_path)
        sample["_annotation_path"] = str(path)
        samples.append(sample)

    if errors:
        preview = "\n".join("  - " + error for error in errors[:20])
        suffix = "\n  - ... and %d more" % (len(errors) - 20) if len(errors) > 20 else ""
        raise InvalidDatasetError(
            "%d invalid sample(s) in %s:\n%s%s"
            % (len(errors), path, preview, suffix)
        )
    if not samples:
        raise InvalidDatasetError("dataset selection is empty")

    return Dataset(
        annotation_path=path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        samples=tuple(samples),
        source_image_paths=tuple(sorted(source_image_paths, key=str)),
    )


def public_sample_metadata(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Fields copied into prediction and scored-result JSONL records."""

    return {
        "sample_index": sample["_source_index"],
        "sample_id": sample["_sample_id"],
        "image": sample["images"][0],
        "captcha_type": sample["captcha_type"],
    }
