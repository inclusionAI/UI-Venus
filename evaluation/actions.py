"""Parsing utilities for the Venus GUI action DSL.

The parser intentionally accepts small formatting variations (whitespace,
``box``/``bbox`` and round/square coordinate delimiters) while preserving a
strict action set. Model predictions must contain exactly one complete
``<action>`` block. Unsupported function-like actions and additional action
blocks are recorded so the scorer can reject them.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Iterable, Mapping


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_STRING_LITERAL = r"(?:'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")"

_ACTION_PATTERN = re.compile(
    r"(?P<point>(?P<point_type>Click|LongPress)\s*\(\s*(?:box|bbox)\s*=\s*"
    r"[\(\[]\s*(?P<point_x>" + _NUMBER + r")\s*,\s*"
    r"(?P<point_y>" + _NUMBER + r")\s*[\)\]]\s*\))"
    r"|(?P<type>Type\s*\(\s*content\s*=\s*(?P<type_literal>"
    + _STRING_LITERAL
    + r")\s*\))"
    r"|(?P<drag>Drag\s*\(\s*start\s*=\s*[\(\[]\s*"
    r"(?P<drag_sx>" + _NUMBER + r")\s*,\s*"
    r"(?P<drag_sy>" + _NUMBER + r")\s*[\)\]]\s*,\s*end\s*=\s*"
    r"[\(\[]\s*(?P<drag_ex>" + _NUMBER + r")\s*,\s*"
    r"(?P<drag_ey>" + _NUMBER + r")\s*[\)\]]\s*\))",
    re.DOTALL,
)

_ACTION_BLOCK = re.compile(
    r"<action(?:\s[^>]*)?>(.*?)</action\s*>",
    re.DOTALL | re.IGNORECASE,
)
_ACTION_OPEN = re.compile(r"<action(?:\s[^>]*)?>", re.IGNORECASE)
_ACTION_CLOSE = re.compile(r"</action\s*>", re.IGNORECASE)
_ACTION_FRAGMENT = re.compile(r"</?\s*action\b", re.IGNORECASE)
_ACTION_LIKE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


class ParsedActions(list):
    """A list of parsed actions with format-error metadata."""

    def __init__(
        self,
        actions: Iterable[Mapping[str, Any]] = (),
        errors: Iterable[str] = (),
    ) -> None:
        super().__init__(actions)
        self.errors = tuple(errors)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    @property
    def unrecognized_actions(self) -> tuple[str, ...]:
        """Compatibility alias used by the original CAPTCHA evaluator."""

        return self.errors

    @property
    def has_unrecognized_actions(self) -> bool:
        return self.has_errors


def _number(value: str) -> int | float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("coordinate must be finite")
    if parsed.is_integer() and not any(char in value.lower() for char in (".", "e")):
        return int(parsed)
    return parsed


def _string_value(literal: str) -> str:
    try:
        value = ast.literal_eval(literal)
    except (SyntaxError, ValueError):
        return literal[1:-1]
    return value if isinstance(value, str) else str(value)


def parse_actions(text: Any, *, allow_unwrapped: bool = False) -> ParsedActions:
    """Extract supported actions in their original execution order.

    By default, exactly one well-formed ``<action>`` block is required and only
    its contents are parsed. Surrounding text, including any ``<think>`` markup,
    is ignored and never validated. ``allow_unwrapped`` exists only for trusted
    source annotations such as the released ``action_raw`` field. A malformed
    or repeated action wrapper is always a format error.
    """

    source = "" if text is None else str(text)
    errors: list[str] = []
    blocks = list(_ACTION_BLOCK.finditer(source))

    if blocks:
        action_text = blocks[0].group(1).strip()
        if len(blocks) > 1:
            errors.extend("additional_action_block" for _ in blocks[1:])
        outside_parts = []
        cursor = 0
        for block in blocks:
            outside_parts.append(source[cursor:block.start()])
            cursor = block.end()
        outside_parts.append(source[cursor:])
        outside = "".join(outside_parts)
        if _ACTION_FRAGMENT.search(outside):
            errors.append("malformed_additional_action_block")
        if _ACTION_OPEN.search(action_text) or _ACTION_CLOSE.search(action_text):
            errors.append("nested_action_block")
    else:
        action_text = source if allow_unwrapped else ""
        if _ACTION_FRAGMENT.search(source):
            errors.append("malformed_action_block")
        elif not allow_unwrapped:
            errors.append("missing_action_block")

    actions: list[dict[str, Any]] = []
    matched_spans: list[tuple[int, int]] = []
    for match in _ACTION_PATTERN.finditer(action_text):
        matched_spans.append(match.span())
        try:
            if match.group("point") is not None:
                actions.append(
                    {
                        "type": match.group("point_type"),
                        "x": _number(match.group("point_x")),
                        "y": _number(match.group("point_y")),
                    }
                )
            elif match.group("type") is not None:
                actions.append(
                    {
                        "type": "Type",
                        "content": _string_value(match.group("type_literal")),
                    }
                )
            else:
                actions.append(
                    {
                        "type": "Drag",
                        "sx": _number(match.group("drag_sx")),
                        "sy": _number(match.group("drag_sy")),
                        "ex": _number(match.group("drag_ex")),
                        "ey": _number(match.group("drag_ey")),
                    }
                )
        except ValueError:
            errors.append("non_finite_coordinate")

    for candidate in _ACTION_LIKE.finditer(action_text):
        if any(start <= candidate.start() < end for start, end in matched_spans):
            continue
        errors.append("unsupported_action:%s" % candidate.group(1))

    return ParsedActions(actions, errors)


def parse_click_type_groups(
    actions: Iterable[Mapping[str, Any]],
) -> tuple[list[tuple[Mapping[str, Any], str]], list[Mapping[str, Any]]]:
    """Split actions into adjacent ``Click`` + ``Type`` pairs and leftovers."""

    action_list = list(actions)
    groups: list[tuple[Mapping[str, Any], str]] = []
    leftovers: list[Mapping[str, Any]] = []
    index = 0
    while index < len(action_list):
        action = action_list[index]
        if (
            action.get("type") == "Click"
            and index + 1 < len(action_list)
            and action_list[index + 1].get("type") == "Type"
        ):
            groups.append((action, str(action_list[index + 1].get("content", ""))))
            index += 2
            continue
        leftovers.append(action)
        index += 1
    return groups, leftovers
