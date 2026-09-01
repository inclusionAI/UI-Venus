"""Strict sample-level scoring for VenusBench-CAPTCHA."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from .actions import ParsedActions, parse_actions, parse_click_type_groups


POINT_ACTION_TYPES = frozenset(("Click", "LongPress"))
DEFAULT_COORD_SCALE = 999.0
DEFAULT_DRAG_DISTANCE_REL_TOLERANCE = 0.05
DEFAULT_DRAG_Y_TOLERANCE = 5.0
MIN_POINT_SEPARATION_PIXELS = 1.0


class InvalidGroundTruthError(ValueError):
    """Raised when a sample cannot be scored safely."""


@dataclass(frozen=True)
class ScoreResult:
    """The deterministic outcome of scoring one model response."""

    correct: bool
    mode: str
    reason: str
    actions: tuple[dict[str, Any], ...]
    parser_errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["actions"] = list(value["actions"])
        value["parser_errors"] = list(value["parser_errors"])
        return value


@dataclass(frozen=True)
class _GroundTruth:
    mode: str
    width: float
    height: float
    rects: tuple[Any, ...]
    actions: tuple[dict[str, Any], ...]


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_bbox(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(_is_number(item) for item in value)
        and value[0] <= value[2]
        and value[1] <= value[3]
    )


def _regular_rects(raw_rects: Any, width: float, height: float) -> tuple[tuple[float, ...], ...]:
    if _is_bbox(raw_rects):
        values = [raw_rects]
    elif isinstance(raw_rects, (list, tuple)):
        values = list(raw_rects)
    else:
        values = []
    if not values:
        raise InvalidGroundTruthError("action_raw_rect must contain at least one bbox")

    rects = []
    for index, rect in enumerate(values):
        if not _is_bbox(rect):
            raise InvalidGroundTruthError(
                "action_raw_rect[%d] must be [x1, y1, x2, y2]" % index
            )
        normalized = tuple(float(item) for item in rect)
        if not (
            0 <= normalized[0] <= normalized[2] <= width
            and 0 <= normalized[1] <= normalized[3] <= height
        ):
            raise InvalidGroundTruthError(
                "action_raw_rect[%d] is outside image_size" % index
            )
        rects.append(normalized)
    return tuple(rects)


def _bingo_rects(raw_rects: Any, width: float, height: float) -> tuple[Any, ...]:
    if not isinstance(raw_rects, (list, tuple)) or not raw_rects:
        raise InvalidGroundTruthError("bingo action_raw_rect must not be empty")
    candidates: Iterable[Any]
    if len(raw_rects) == 2 and all(_is_bbox(item) for item in raw_rects):
        candidates = [raw_rects]
    else:
        candidates = raw_rects

    answers = []
    for answer_index, answer in enumerate(candidates):
        if (
            not isinstance(answer, (list, tuple))
            or len(answer) != 2
            or not all(_is_bbox(item) for item in answer)
        ):
            raise InvalidGroundTruthError(
                "bingo answer %d must contain exactly two bboxes" % answer_index
            )
        answers.append(
            _regular_rects(answer, width, height)
        )
    return tuple(answers)


def _prepare_ground_truth(sample: Mapping[str, Any]) -> _GroundTruth:
    image_size = sample.get("image_size")
    if (
        not isinstance(image_size, (list, tuple))
        or len(image_size) != 2
        or not all(_is_number(value) and value > 0 for value in image_size)
    ):
        raise InvalidGroundTruthError("image_size must be [positive_width, positive_height]")
    width, height = float(image_size[0]), float(image_size[1])

    action_raw = sample.get("action_raw")
    if not isinstance(action_raw, str) or not action_raw.strip():
        raise InvalidGroundTruthError("action_raw must be a non-empty string")

    if sample.get("captcha_type") == "bingo":
        return _GroundTruth(
            mode="bingo",
            width=width,
            height=height,
            rects=_bingo_rects(sample.get("action_raw_rect"), width, height),
            actions=(),
        )

    rects = _regular_rects(sample.get("action_raw_rect"), width, height)
    parsed = parse_actions(action_raw, allow_unwrapped=True)
    if parsed.has_errors:
        raise InvalidGroundTruthError(
            "action_raw contains unsupported syntax: %s" % ", ".join(parsed.errors)
        )
    if not parsed:
        raise InvalidGroundTruthError("action_raw contains no supported action")

    actions = tuple(dict(action) for action in parsed)
    action_types = [action["type"] for action in actions]
    if "Drag" in action_types:
        if len(actions) != 1 or action_types[0] != "Drag":
            raise InvalidGroundTruthError("drag GT must contain exactly one Drag")
        mode = "drag"
    elif "Type" in action_types:
        groups, leftovers = parse_click_type_groups(actions)
        if not groups or leftovers or len(actions) != 2 * len(groups):
            raise InvalidGroundTruthError("text GT must contain only Click+Type pairs")
        if len(rects) != len(groups):
            raise InvalidGroundTruthError(
                "Click+Type pair count (%d) differs from bbox count (%d)"
                % (len(groups), len(rects))
            )
        mode = "text"
    else:
        if any(action_type not in POINT_ACTION_TYPES for action_type in action_types):
            raise InvalidGroundTruthError("point GT supports only Click and LongPress")
        if len(actions) != len(rects):
            raise InvalidGroundTruthError(
                "point action count (%d) differs from bbox count (%d)"
                % (len(actions), len(rects))
            )
        mode = "point"

    return _GroundTruth(mode, width, height, rects, actions)


def validate_ground_truth(sample: Mapping[str, Any]) -> bool:
    """Validate all fields that affect scoring before model inference starts."""

    _prepare_ground_truth(sample)
    return True


def ground_truth_mode(sample: Mapping[str, Any]) -> str:
    """Return ``point``, ``text``, ``drag`` or ``bingo`` after validation."""

    return _prepare_ground_truth(sample).mode


def denormalize(
    x: float,
    y: float,
    width: float,
    height: float,
    coord_scale: float = DEFAULT_COORD_SCALE,
) -> tuple[float, float]:
    """Convert model coordinates into the annotation's original pixel space."""

    if coord_scale <= 0:
        return float(x), float(y)
    return float(x) * width / coord_scale, float(y) * height / coord_scale


def point_in_bbox(x: float, y: float, bbox: Sequence[float]) -> bool:
    """Return whether a point lies in a bbox, including its boundary."""

    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _point_locations_are_distinct(
    actions: Iterable[Mapping[str, Any]],
    ground_truth: _GroundTruth,
    coord_scale: float,
) -> bool:
    """Require separate GUI targets to use separate original-pixel locations."""

    points = [
        denormalize(
            action["x"],
            action["y"],
            ground_truth.width,
            ground_truth.height,
            coord_scale,
        )
        for action in actions
        if action.get("type") in POINT_ACTION_TYPES
    ]
    minimum_squared = MIN_POINT_SEPARATION_PIXELS**2
    for index, first in enumerate(points):
        for second in points[index + 1 :]:
            squared_distance = (first[0] - second[0]) ** 2 + (
                first[1] - second[1]
            ) ** 2
            if squared_distance < minimum_squared:
                return False
    return True


def _spatial_values(action: Mapping[str, Any]) -> tuple[float, ...]:
    if action.get("type") in POINT_ACTION_TYPES:
        return float(action["x"]), float(action["y"])
    if action.get("type") == "Drag":
        return (
            float(action["sx"]),
            float(action["sy"]),
            float(action["ex"]),
            float(action["ey"]),
        )
    return ()


def _coordinates_in_range(actions: Iterable[Mapping[str, Any]], coord_scale: float) -> bool:
    if coord_scale <= 0:
        return True
    return all(
        0 <= value <= coord_scale
        for action in actions
        for value in _spatial_values(action)
    )


def _perfect_matching(adjacency: Sequence[Sequence[int]], right_count: int) -> bool:
    if len(adjacency) != right_count:
        return False
    right_to_left = [-1] * right_count

    def augment(left: int, seen: list[bool]) -> bool:
        for right in adjacency[left]:
            if seen[right]:
                continue
            seen[right] = True
            if right_to_left[right] == -1 or augment(right_to_left[right], seen):
                right_to_left[right] = left
                return True
        return False

    return all(augment(left, [False] * right_count) for left in range(len(adjacency)))


def _score_points(
    predicted: Sequence[Mapping[str, Any]],
    ground_truth: _GroundTruth,
    inorder: bool,
    coord_scale: float,
) -> tuple[bool, str]:
    if len(predicted) != len(ground_truth.actions):
        return False, "point_action_count_mismatch"
    if any(action.get("type") not in POINT_ACTION_TYPES for action in predicted):
        return False, "unexpected_action_type"
    if not _point_locations_are_distinct(predicted, ground_truth, coord_scale):
        return False, "duplicate_point_location"

    if inorder:
        for predicted_action, gt_action, rect in zip(
            predicted, ground_truth.actions, ground_truth.rects
        ):
            if predicted_action["type"] != gt_action["type"]:
                return False, "point_action_type_mismatch"
            point = denormalize(
                predicted_action["x"],
                predicted_action["y"],
                ground_truth.width,
                ground_truth.height,
                coord_scale,
            )
            if not point_in_bbox(*point, rect):
                return False, "point_target_mismatch"
        return True, "correct"

    adjacency = []
    for predicted_action in predicted:
        point = denormalize(
            predicted_action["x"],
            predicted_action["y"],
            ground_truth.width,
            ground_truth.height,
            coord_scale,
        )
        adjacency.append(
            [
                index
                for index, (gt_action, rect) in enumerate(
                    zip(ground_truth.actions, ground_truth.rects)
                )
                if predicted_action["type"] == gt_action["type"]
                and point_in_bbox(*point, rect)
            ]
        )
    if _perfect_matching(adjacency, len(ground_truth.actions)):
        return True, "correct"
    return False, "point_target_mismatch"


def _score_text(
    predicted: Sequence[Mapping[str, Any]],
    ground_truth: _GroundTruth,
    inorder: bool,
    coord_scale: float,
) -> tuple[bool, str]:
    predicted_groups, predicted_leftovers = parse_click_type_groups(predicted)
    gt_groups, _ = parse_click_type_groups(ground_truth.actions)
    if predicted_leftovers or len(predicted) != 2 * len(predicted_groups):
        return False, "text_actions_must_be_click_type_pairs"
    if len(predicted_groups) != len(gt_groups):
        return False, "click_type_pair_count_mismatch"
    if not _point_locations_are_distinct(predicted, ground_truth, coord_scale):
        return False, "duplicate_point_location"

    def matches(
        predicted_group: tuple[Mapping[str, Any], str],
        gt_index: int,
    ) -> bool:
        predicted_click, predicted_text = predicted_group
        _, gt_text = gt_groups[gt_index]
        point = denormalize(
            predicted_click["x"],
            predicted_click["y"],
            ground_truth.width,
            ground_truth.height,
            coord_scale,
        )
        return (
            predicted_text.strip() == gt_text.strip()
            and point_in_bbox(*point, ground_truth.rects[gt_index])
        )

    if inorder:
        if all(matches(group, index) for index, group in enumerate(predicted_groups)):
            return True, "correct"
        return False, "click_type_content_or_target_mismatch"

    adjacency = [
        [index for index in range(len(gt_groups)) if matches(group, index)]
        for group in predicted_groups
    ]
    if _perfect_matching(adjacency, len(gt_groups)):
        return True, "correct"
    return False, "click_type_content_or_target_mismatch"


def _score_drag(
    predicted: Sequence[Mapping[str, Any]],
    ground_truth: _GroundTruth,
    coord_scale: float,
    distance_relative_tolerance: float,
    y_tolerance: float,
) -> tuple[bool, str]:
    if len(predicted) != 1 or predicted[0].get("type") != "Drag":
        return False, "drag_requires_exactly_one_action"
    action = predicted[0]
    start = denormalize(
        action["sx"], action["sy"], ground_truth.width, ground_truth.height, coord_scale
    )
    end = denormalize(
        action["ex"], action["ey"], ground_truth.width, ground_truth.height, coord_scale
    )
    if not point_in_bbox(*start, ground_truth.rects[0]):
        return False, "drag_start_miss"

    gt = ground_truth.actions[0]
    predicted_dx = end[0] - start[0]
    gt_dx = float(gt["ex"]) - float(gt["sx"])
    if abs(gt_dx) <= 1e-12:
        if abs(predicted_dx) > 1e-12:
            return False, "drag_distance_mismatch"
    else:
        if predicted_dx * gt_dx <= 0:
            return False, "drag_direction_mismatch"
        if abs(predicted_dx - gt_dx) / abs(gt_dx) >= distance_relative_tolerance:
            return False, "drag_distance_mismatch"
    if abs(end[1] - float(gt["ey"])) >= y_tolerance:
        return False, "drag_end_y_mismatch"
    return True, "correct"


def _score_bingo(
    predicted: Sequence[Mapping[str, Any]],
    ground_truth: _GroundTruth,
    coord_scale: float,
) -> tuple[bool, str]:
    if len(predicted) != 2 or any(action.get("type") != "Click" for action in predicted):
        return False, "bingo_requires_exactly_two_clicks"
    if not _point_locations_are_distinct(predicted, ground_truth, coord_scale):
        return False, "duplicate_point_location"
    points = [
        denormalize(
            action["x"], action["y"], ground_truth.width, ground_truth.height, coord_scale
        )
        for action in predicted
    ]
    for first_rect, second_rect in ground_truth.rects:
        direct = point_in_bbox(*points[0], first_rect) and point_in_bbox(
            *points[1], second_rect
        )
        reverse = point_in_bbox(*points[0], second_rect) and point_in_bbox(
            *points[1], first_rect
        )
        if direct or reverse:
            return True, "correct"
    return False, "bingo_target_mismatch"


def evaluate_actions(
    predicted: ParsedActions,
    sample: Mapping[str, Any],
    coord_scale: float = DEFAULT_COORD_SCALE,
    drag_distance_relative_tolerance: float = DEFAULT_DRAG_DISTANCE_REL_TOLERANCE,
    drag_y_tolerance: float = DEFAULT_DRAG_Y_TOLERANCE,
) -> ScoreResult:
    """Score already-parsed actions against a sample's strict ground truth."""

    if not math.isfinite(coord_scale) or coord_scale < 0:
        raise ValueError("coord_scale must be finite and non-negative")
    if coord_scale == 0:
        coord_scale = 0.0
    if (
        not math.isfinite(drag_distance_relative_tolerance)
        or not math.isfinite(drag_y_tolerance)
        or drag_distance_relative_tolerance <= 0
        or drag_y_tolerance <= 0
    ):
        raise ValueError("drag tolerances must be positive")

    ground_truth = _prepare_ground_truth(sample)
    actions = tuple(dict(action) for action in predicted)
    if predicted.has_errors:
        return ScoreResult(
            False,
            ground_truth.mode,
            "format_error",
            actions,
            tuple(predicted.errors),
        )
    if not actions:
        return ScoreResult(False, ground_truth.mode, "no_supported_action", actions)
    if not _coordinates_in_range(actions, coord_scale):
        return ScoreResult(False, ground_truth.mode, "coordinate_out_of_range", actions)
    if ground_truth.mode == "point":
        correct, reason = _score_points(
            actions, ground_truth, bool(sample.get("inorder")), coord_scale
        )
    elif ground_truth.mode == "text":
        correct, reason = _score_text(
            actions, ground_truth, bool(sample.get("inorder")), coord_scale
        )
    elif ground_truth.mode == "drag":
        correct, reason = _score_drag(
            actions,
            ground_truth,
            coord_scale,
            drag_distance_relative_tolerance,
            drag_y_tolerance,
        )
    else:
        correct, reason = _score_bingo(actions, ground_truth, coord_scale)
    return ScoreResult(correct, ground_truth.mode, reason, actions)


def evaluate_prediction(
    model_output: Any,
    sample: Mapping[str, Any],
    coord_scale: float = DEFAULT_COORD_SCALE,
    drag_distance_relative_tolerance: float = DEFAULT_DRAG_DISTANCE_REL_TOLERANCE,
    drag_y_tolerance: float = DEFAULT_DRAG_Y_TOLERANCE,
) -> ScoreResult:
    """Parse and score one raw model response."""

    return evaluate_actions(
        parse_actions(model_output),
        sample,
        coord_scale=coord_scale,
        drag_distance_relative_tolerance=drag_distance_relative_tolerance,
        drag_y_tolerance=drag_y_tolerance,
    )
