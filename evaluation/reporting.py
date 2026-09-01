"""Self-contained visual HTML reports for CAPTCHA evaluation results.

The report intentionally has no runtime dependencies: screenshots are embedded
as data URLs, overlays are SVG, and filtering is performed by a small inline
script.  Values originating in annotations or model responses are escaped
before being inserted into HTML.
"""

from __future__ import annotations

import base64
import html
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_STATUS_ORDER = (
    "correct",
    "wrong",
    "parse_error",
    "empty_response",
    "api_error",
    "missing_prediction",
)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _display_text(value: Any) -> str:
    """Render arbitrary JSON-like content without allowing HTML markup."""

    if value is None:
        text = "—"
    elif isinstance(value, str):
        text = value if value else "—"
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        except (TypeError, ValueError):
            text = str(value)
    return html.escape(text, quote=True)


def _as_finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _format_number(value: float) -> str:
    # SVG attributes receive only values converted and checked by
    # _as_finite_number, so no annotation text can become markup here.
    return format(value, ".6g")


def _percent(value: Any) -> str:
    number = _as_finite_number(value)
    return "N/A" if number is None else "%.2f%%" % (number * 100.0)


def _integer(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(
        content, (str, bytes, bytearray)
    ):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, Mapping) and item.get("type") == "text":
                pieces.append(str(item.get("text", "")))
        return "\n".join(piece for piece in pieces if piece)
    return str(content)


def _user_text(sample: Mapping[str, Any]) -> str:
    """Extract user text from OpenAI-style string or segmented messages."""

    parts: list[str] = []
    messages = sample.get("messages")
    if isinstance(messages, Sequence) and not isinstance(
        messages, (str, bytes, bytearray)
    ):
        for message in messages:
            if isinstance(message, Mapping) and message.get("role") == "user":
                text = _content_text(message.get("content"))
                if text:
                    parts.append(text)
    return "\n".join(parts).replace("<image>", "").strip()


def _recorded_image(sample: Mapping[str, Any]) -> str | None:
    images = sample.get("images")
    if (
        isinstance(images, Sequence)
        and not isinstance(images, (str, bytes, bytearray))
        and images
        and isinstance(images[0], str)
    ):
        return images[0]
    image = sample.get("image")
    return image if isinstance(image, str) and image else None


def _resolve_image(annotation_path: Path, sample: Mapping[str, Any]) -> Path | None:
    recorded = _recorded_image(sample)
    if recorded is None:
        return None
    path = Path(recorded).expanduser()
    if not path.is_absolute():
        path = annotation_path.parent / path
    return path.resolve()


def _image_mime(data: bytes) -> str | None:
    """Identify common browser-displayable image formats from magic bytes."""

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if data.startswith(b"\x00\x00\x01\x00"):
        return "image/x-icon"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"avif", b"avis") or b"avif" in data[8:32]:
            return "image/avif"
    return None


def _embedded_image(
    annotation_path: Path, sample: Mapping[str, Any]
) -> tuple[str | None, str]:
    image_path = _resolve_image(annotation_path, sample)
    if image_path is None:
        return None, "No image path in annotation"
    try:
        data = image_path.read_bytes()
    except OSError as error:
        return None, "Could not read %s: %s" % (image_path.name, error)
    mime = _image_mime(data)
    if mime is None:
        return None, "Unsupported image signature: %s" % image_path.name
    encoded = base64.b64encode(data).decode("ascii")
    return "data:%s;base64,%s" % (mime, encoded), image_path.name


def _image_size(sample: Mapping[str, Any]) -> tuple[float, float]:
    size = sample.get("image_size")
    if (
        isinstance(size, Sequence)
        and not isinstance(size, (str, bytes, bytearray))
        and len(size) == 2
    ):
        width = _as_finite_number(size[0])
        height = _as_finite_number(size[1])
        if width is not None and height is not None and width > 0 and height > 0:
            return width, height
    return 1.0, 1.0


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 4
    ):
        return None
    numbers = tuple(_as_finite_number(item) for item in value)
    if any(item is None for item in numbers):
        return None
    x1, y1, x2, y2 = numbers
    if x1 > x2 or y1 > y2:
        return None
    return x1, y1, x2, y2


def _flatten_bboxes(value: Any) -> list[tuple[float, float, float, float]]:
    direct = _bbox(value)
    if direct is not None:
        return [direct]
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        boxes: list[tuple[float, float, float, float]] = []
        for item in value:
            boxes.extend(_flatten_bboxes(item))
        return boxes
    return []


def _denormalize(
    x: Any, y: Any, width: float, height: float, coord_scale: float
) -> tuple[float, float] | None:
    x_number = _as_finite_number(x)
    y_number = _as_finite_number(y)
    if x_number is None or y_number is None:
        return None
    if coord_scale > 0:
        return x_number * width / coord_scale, y_number * height / coord_scale
    return x_number, y_number


def _svg_overlay(
    sample: Mapping[str, Any],
    detail: Mapping[str, Any],
    coord_scale: float,
    marker_index: int,
) -> str:
    width, height = _image_size(sample)
    elements: list[str] = []
    click_order_labels: list[str] = []

    for index, (x1, y1, x2, y2) in enumerate(
        _flatten_bboxes(sample.get("action_raw_rect")), 1
    ):
        elements.append(
            '<rect class="gt-box" x="%s" y="%s" width="%s" height="%s">'
            '<title>GT bbox %d</title></rect>'
            % tuple(
                list(
                    map(
                        _format_number,
                        (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)),
                    )
                )
                + [index]
            )
        )

    actions = detail.get("predicted_actions")
    if not isinstance(actions, Sequence) or isinstance(
        actions, (str, bytes, bytearray)
    ):
        actions = ()
    marker_id = "drag-arrow-%d" % marker_index
    has_drag = any(
        isinstance(action, Mapping) and action.get("type") == "Drag"
        for action in actions
    )
    if has_drag:
        elements.insert(
            0,
            '<defs><marker id="%s" markerWidth="8" markerHeight="8" '
            'refX="7" refY="4" orient="auto" markerUnits="strokeWidth">'
            '<path d="M0,0 L8,4 L0,8 Z" fill="#e11d48"></path>'
            "</marker></defs>" % marker_id,
        )

    show_click_order = sample.get("inorder") is True
    click_order = 0
    for index, action in enumerate(actions, 1):
        if not isinstance(action, Mapping):
            continue
        action_type = str(action.get("type", ""))
        if action_type in ("Click", "LongPress"):
            if action_type == "Click":
                click_order += 1
            point = _denormalize(
                action.get("x"), action.get("y"), width, height, coord_scale
            )
            if point is None:
                continue
            x, y = point
            if action_type == "Click" and show_click_order:
                # Use fixed-size, text-only HTML labels so high-resolution
                # images do not enlarge the marker and the underlying symbol
                # remains visible around the number glyph.
                left = x * 100.0 / width
                top = y * 100.0 / height
                click_order_labels.append(
                    '<span class="pred-click-order" style="left:%s%%;top:%s%%" '
                    'aria-label="Prediction click order %d" '
                    'title="Prediction click %d: (%.2f, %.2f)">%d</span>'
                    % (
                        _format_number(left),
                        _format_number(top),
                        click_order,
                        click_order,
                        x,
                        y,
                        click_order,
                    )
                )
            else:
                css_class = (
                    "pred-longpress" if action_type == "LongPress" else "pred-click"
                )
                radius = max(4.0, min(width, height) * 0.012)
                elements.append(
                    '<circle class="%s" cx="%s" cy="%s" r="%s">'
                    '<title>Prediction %d: %s (%.2f, %.2f)</title></circle>'
                    % (
                        css_class,
                        _format_number(x),
                        _format_number(y),
                        _format_number(radius),
                        index,
                        action_type,
                        x,
                        y,
                    )
                )
                if action_type == "LongPress":
                    elements.append(
                        '<circle class="pred-longpress-ring" cx="%s" cy="%s" '
                        'r="%s"></circle>'
                        % (
                            _format_number(x),
                            _format_number(y),
                            _format_number(radius * 1.9),
                        )
                    )
        elif action_type == "Drag":
            start = _denormalize(
                action.get("sx"), action.get("sy"), width, height, coord_scale
            )
            end = _denormalize(
                action.get("ex"), action.get("ey"), width, height, coord_scale
            )
            if start is None or end is None:
                continue
            sx, sy = start
            ex, ey = end
            elements.append(
                '<line class="pred-drag" x1="%s" y1="%s" x2="%s" y2="%s" '
                'marker-end="url(#%s)"><title>Prediction %d: Drag '
                '(%.2f, %.2f) → (%.2f, %.2f)</title></line>'
                % (
                    _format_number(sx),
                    _format_number(sy),
                    _format_number(ex),
                    _format_number(ey),
                    marker_id,
                    index,
                    sx,
                    sy,
                    ex,
                    ey,
                )
            )
            radius = max(3.0, min(width, height) * 0.009)
            elements.append(
                '<circle class="drag-start" cx="%s" cy="%s" r="%s"></circle>'
                % (_format_number(sx), _format_number(sy), _format_number(radius))
            )

    svg = (
        '<svg class="overlay" viewBox="0 0 %s %s" preserveAspectRatio="none" '
        'aria-label="Ground-truth and prediction overlay">%s</svg>'
        % (_format_number(width), _format_number(height), "".join(elements))
    )
    return svg + "".join(click_order_labels)


def _detail_for_sample(
    sample: Mapping[str, Any],
    position: int,
    details: Sequence[Mapping[str, Any]],
    by_id: Mapping[str, Mapping[str, Any]],
    by_index: Mapping[int, Mapping[str, Any]],
) -> Mapping[str, Any]:
    sample_id = sample.get("_sample_id")
    if sample_id is not None and str(sample_id) in by_id:
        return by_id[str(sample_id)]
    source_index = sample.get("_source_index")
    if isinstance(source_index, int) and not isinstance(source_index, bool):
        if source_index in by_index:
            return by_index[source_index]
    if position < len(details):
        return details[position]
    return {
        "sample_index": source_index if source_index is not None else position,
        "sample_id": sample_id if sample_id is not None else str(position),
        "captcha_type": sample.get("captcha_type", "unknown"),
        "status": "missing_prediction",
        "reason": "missing_prediction",
        "model_output": None,
        "predicted_actions": [],
    }


def _status_class(status: str) -> str:
    safe = "".join(character if character.isalnum() else "-" for character in status)
    return "status-" + (safe or "unknown")


def _sample_card(
    annotation_path: Path,
    sample: Mapping[str, Any],
    detail: Mapping[str, Any],
    coord_scale: float,
    marker_index: int,
) -> str:
    captcha_type = str(detail.get("captcha_type") or sample.get("captcha_type") or "unknown")
    status = str(detail.get("status") or "unknown")
    sample_index = detail.get("sample_index", sample.get("_source_index", marker_index))
    sample_id = detail.get("sample_id", sample.get("_sample_id", sample_index))
    image_url, image_note = _embedded_image(annotation_path, sample)
    width, height = _image_size(sample)
    ratio = _format_number(width) + " / " + _format_number(height)
    if image_url is None:
        visual = (
            '<div class="missing-image" style="aspect-ratio:%s">'
            '<span>%s</span></div>' % (ratio, _escape(image_note))
        )
    else:
        # The base64 value exists only in src; the human-readable label is the
        # original filename, never the encoded payload.
        visual = (
            '<img src="%s" alt="CAPTCHA screenshot: %s" width="%s" height="%s" '
            'loading="lazy">'
            % (
                image_url,
                _escape(image_note),
                _format_number(width),
                _format_number(height),
            )
        )
    overlay = _svg_overlay(sample, detail, coord_scale, marker_index)

    reasoning = detail.get("reasoning_content")
    if reasoning is None:
        reasoning = detail.get("reasoning")
    fields = (
        ("User task", _user_text(sample)),
        ("GT action", sample.get("action_raw")),
        ("Model output", detail.get("model_output")),
        ("Reason", detail.get("reason")),
        ("Parser errors", detail.get("parser_errors")),
        ("Reasoning", reasoning),
        ("Inference", detail.get("inference")),
        ("Backend diagnostics", detail.get("api_diagnostics")),
    )
    field_html = "".join(
        '<div class="field"><dt>%s</dt><dd><pre>%s</pre></dd></div>'
        % (_escape(label), _display_text(value))
        for label, value in fields
    )

    return (
        '<article class="sample-card" data-captcha-type="%s" data-status="%s">'
        '<header><div><span class="sample-index">#%s</span> '
        '<span class="type-pill">%s</span></div>'
        '<span class="status-pill %s">%s</span></header>'
        '<div class="sample-id" title="Sample ID">%s</div>'
        '<div class="sample-grid"><div><div class="image-stage" style="aspect-ratio:%s">'
        '%s%s</div><div class="legend"><span class="legend-gt">GT bbox</span>'
        '<span class="legend-click">Click</span><span class="legend-long">LongPress</span>'
        '<span class="legend-drag">Drag</span></div></div>'
        '<dl class="detail-fields">%s</dl></div></article>'
        % (
            _escape(captcha_type),
            _escape(status),
            _escape(sample_index),
            _escape(captcha_type),
            _status_class(status),
            _escape(status),
            _escape(sample_id),
            ratio,
            visual,
            overlay,
            field_html,
        )
    )


def _metric_cards(summary: Mapping[str, Any]) -> str:
    overall = summary.get("overall")
    if not isinstance(overall, Mapping):
        overall = {}
    macro = summary.get("macro_average")
    if not isinstance(macro, Mapping):
        macro = {}
    statuses = overall.get("statuses")
    if not isinstance(statuses, Mapping):
        statuses = {}

    status_names = list(_STATUS_ORDER)
    status_names.extend(
        sorted(str(name) for name in statuses if str(name) not in _STATUS_ORDER)
    )
    status_html = "".join(
        '<div class="status-stat"><span>%s</span><strong>%d</strong></div>'
        % (_escape(name), _integer(statuses.get(name, 0)))
        for name in status_names
    )
    return (
        '<section class="metrics-grid">'
        '<div class="metric-card"><span>Pass@1 (micro)</span><strong>%s</strong>'
        '<small>%d / %d correct</small></div>'
        '<div class="metric-card"><span>Macro pass@1</span><strong>%s</strong>'
        '<small>across %d CAPTCHA types</small></div>'
        '<div class="metric-card status-card"><span>Status counts</span>'
        '<div class="status-grid">%s</div></div></section>'
        % (
            _percent(overall.get("pass_at_1", overall.get("accuracy"))),
            _integer(overall.get("correct")),
            _integer(overall.get("total")),
            _percent(macro.get("pass_at_1", macro.get("accuracy"))),
            _integer(macro.get("categories")),
            status_html,
        )
    )


def _per_type_table(summary: Mapping[str, Any]) -> str:
    per_type = summary.get("per_captcha_type")
    if not isinstance(per_type, Mapping):
        per_type = {}
    rows = []
    for name, raw_stats in sorted(per_type.items(), key=lambda item: str(item[0])):
        stats = raw_stats if isinstance(raw_stats, Mapping) else {}
        rows.append(
            "<tr><th scope=\"row\">%s</th><td>%d</td><td>%d</td>"
            "<td>%s</td><td>%d / %d</td></tr>"
            % (
                _escape(name),
                _integer(stats.get("correct")),
                _integer(stats.get("total")),
                _percent(stats.get("pass_at_1", stats.get("accuracy"))),
                _integer(stats.get("parsed")),
                _integer(stats.get("total")),
            )
        )
    if not rows:
        rows.append('<tr><td colspan="5" class="empty-cell">No category metrics</td></tr>')
    return (
        '<section class="panel"><h2>Per CAPTCHA type</h2><div class="table-wrap">'
        '<table><thead><tr><th>CAPTCHA type</th><th>Correct</th><th>Total</th>'
        '<th>Pass@1</th><th>Parsed</th></tr></thead>'
        '<tbody>%s</tbody></table></div></section>' % "".join(rows)
    )


def _filter_controls(captcha_types: Iterable[str], statuses: Iterable[str]) -> str:
    type_options = "".join(
        '<option value="%s">%s</option>' % (_escape(value), _escape(value))
        for value in sorted(set(captcha_types))
    )
    ordered_statuses = [name for name in _STATUS_ORDER if name in statuses]
    ordered_statuses.extend(sorted(set(statuses).difference(ordered_statuses)))
    status_options = "".join(
        '<option value="%s">%s</option>' % (_escape(value), _escape(value))
        for value in ordered_statuses
    )
    return (
        '<section class="filters panel" aria-label="Sample filters"><label>'
        'CAPTCHA type<select id="type-filter"><option value="">All types</option>%s</select>'
        '</label><label>Status<select id="status-filter"><option value="">All statuses</option>%s</select>'
        '</label><button id="clear-filters" type="button">Clear</button>'
        '<span id="visible-count" aria-live="polite"></span></section>'
        % (type_options, status_options)
    )


_CSS = r"""
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:#f3f5f9; color:#18212f; }
* { box-sizing:border-box; }
body { margin:0; }
main { width:min(1500px, calc(100% - 32px)); margin:0 auto; padding:32px 0 64px; }
.page-header h1 { margin:0; font-size:clamp(1.65rem, 3vw, 2.4rem); letter-spacing:-.035em; }
.page-header p { margin:.55rem 0 0; color:#667085; overflow-wrap:anywhere; }
.metrics-grid { display:grid; grid-template-columns:minmax(190px, .75fr) minmax(190px, .75fr) minmax(360px, 2fr); gap:14px; margin:24px 0 18px; }
.metric-card,.panel,.sample-card { background:#fff; border:1px solid #e2e7ef; border-radius:14px; box-shadow:0 2px 10px rgba(16,24,40,.045); }
.metric-card { padding:18px; display:flex; flex-direction:column; gap:5px; }
.metric-card>span { color:#667085; font-size:.8rem; text-transform:uppercase; letter-spacing:.06em; }
.metric-card>strong { font-size:2rem; letter-spacing:-.04em; }
.metric-card small { color:#667085; }
.status-card { min-width:0; }
.status-grid { display:grid; grid-template-columns:repeat(3,minmax(85px,1fr)); gap:8px 14px; margin-top:5px; }
.status-stat { display:flex; justify-content:space-between; gap:8px; color:#667085; font-size:.84rem; }
.status-stat strong { color:#18212f; }
.panel { padding:18px; margin:18px 0; }
.panel h2 { margin:0 0 14px; font-size:1.05rem; }
.table-wrap { overflow:auto; }
table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
th,td { padding:10px 12px; border-bottom:1px solid #edf0f5; text-align:right; white-space:nowrap; }
th:first-child,td:first-child { text-align:left; }
thead th { color:#667085; font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; }
tbody th { font-weight:600; }
.empty-cell { text-align:center!important; color:#667085; }
.filters { display:flex; align-items:end; gap:14px; position:sticky; top:8px; z-index:10; }
.filters label { display:flex; flex-direction:column; gap:5px; color:#667085; font-size:.78rem; font-weight:600; }
select,button { border:1px solid #ccd3df; border-radius:8px; background:#fff; color:#18212f; padding:9px 34px 9px 10px; font:inherit; }
button { padding:9px 14px; cursor:pointer; }
button:hover { background:#f7f8fa; }
#visible-count { margin-left:auto; color:#667085; font-variant-numeric:tabular-nums; }
.samples { display:grid; gap:18px; }
.sample-card { padding:18px; overflow:hidden; }
.sample-card[hidden] { display:none; }
.sample-card header { display:flex; align-items:center; justify-content:space-between; gap:12px; }
.sample-index { color:#667085; font-variant-numeric:tabular-nums; }
.type-pill,.status-pill { display:inline-block; border-radius:999px; padding:4px 9px; font-size:.76rem; font-weight:700; }
.type-pill { color:#344054; background:#eef2f7; }
.status-pill { color:#344054; background:#eef2f7; }
.status-correct { color:#067647; background:#ecfdf3; }
.status-wrong { color:#b42318; background:#fef3f2; }
.status-parse-error,.status-empty-response { color:#b54708; background:#fffaeb; }
.status-api-error,.status-missing-prediction { color:#6941c6; background:#f4f3ff; }
.sample-id { color:#98a2b3; font: .75rem ui-monospace, SFMono-Regular, Consolas, monospace; overflow-wrap:anywhere; margin:7px 0 16px; }
.sample-grid { display:grid; grid-template-columns:minmax(280px, .9fr) minmax(360px, 1.3fr); gap:20px; align-items:start; }
.image-stage { position:relative; width:100%; overflow:hidden; background:#e8ecf2; border-radius:10px; border:1px solid #d7dde7; }
.image-stage img { width:100%; height:100%; object-fit:fill; display:block; }
.missing-image { width:100%; height:100%; display:grid; place-items:center; padding:24px; color:#667085; background:repeating-linear-gradient(135deg,#f2f4f7,#f2f4f7 12px,#e8ecf1 12px,#e8ecf1 24px); text-align:center; overflow-wrap:anywhere; }
.overlay { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
.gt-box { fill:rgba(18,183,106,.12); stroke:#12b76a; stroke-width:3; vector-effect:non-scaling-stroke; }
.pred-click-order { position:absolute; z-index:2; transform:translate(-50%,-50%); color:#e11d48; font:800 10px/1 ui-monospace,SFMono-Regular,Consolas,monospace; font-variant-numeric:tabular-nums; -webkit-text-stroke:1px rgba(255,255,255,.96); paint-order:stroke fill; text-shadow:0 0 1px #fff; pointer-events:none; }
.pred-click { fill:rgba(240,68,56,.25); stroke:#f04438; stroke-width:3; vector-effect:non-scaling-stroke; }
.pred-longpress { fill:rgba(127,86,217,.28); stroke:#7f56d9; stroke-width:3; vector-effect:non-scaling-stroke; }
.pred-longpress-ring { fill:none; stroke:#7f56d9; stroke-width:2; stroke-dasharray:5 4; vector-effect:non-scaling-stroke; }
.pred-drag { stroke:#e11d48; stroke-width:4; fill:none; vector-effect:non-scaling-stroke; }
.drag-start { fill:#fff; stroke:#e11d48; stroke-width:3; vector-effect:non-scaling-stroke; }
.legend { display:flex; flex-wrap:wrap; gap:12px; color:#667085; font-size:.75rem; margin-top:8px; }
.legend span::before { content:""; display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }
.legend-gt::before { background:#12b76a; }.legend-click::before { background:#f04438; }.legend-long::before { background:#7f56d9; }.legend-drag::before { background:#e11d48; }
.detail-fields { margin:0; display:grid; gap:10px; min-width:0; }
.field { min-width:0; }
.field dt { color:#667085; font-size:.76rem; font-weight:700; text-transform:uppercase; letter-spacing:.04em; margin-bottom:3px; }
.field dd { margin:0; }
pre { margin:0; padding:9px 11px; border-radius:8px; background:#f7f8fa; border:1px solid #edf0f4; white-space:pre-wrap; overflow-wrap:anywhere; font: .78rem/1.45 ui-monospace, SFMono-Regular, Consolas, monospace; max-height:190px; overflow:auto; }
@media (max-width:900px) { .metrics-grid { grid-template-columns:1fr 1fr; }.status-card { grid-column:1/-1; }.sample-grid { grid-template-columns:1fr; }.filters { position:static; flex-wrap:wrap; } }
@media (max-width:560px) { main { width:min(100% - 18px, 1500px); padding-top:18px; }.metrics-grid { grid-template-columns:1fr; }.status-card { grid-column:auto; }.status-grid { grid-template-columns:1fr 1fr; }.filters { align-items:stretch; }.filters label { width:100%; }select { width:100%; }#visible-count { margin-left:0; width:100%; } }
"""


_SCRIPT = r"""
(() => {
  const typeFilter = document.getElementById('type-filter');
  const statusFilter = document.getElementById('status-filter');
  const clear = document.getElementById('clear-filters');
  const count = document.getElementById('visible-count');
  const cards = Array.from(document.querySelectorAll('.sample-card'));
  const apply = () => {
    let visible = 0;
    for (const card of cards) {
      const show = (!typeFilter.value || card.dataset.captchaType === typeFilter.value) &&
                   (!statusFilter.value || card.dataset.status === statusFilter.value);
      card.hidden = !show;
      if (show) visible += 1;
    }
    count.textContent = `${visible} / ${cards.length} samples`;
  };
  typeFilter.addEventListener('change', apply);
  statusFilter.addEventListener('change', apply);
  clear.addEventListener('click', () => {
    typeFilter.value = '';
    statusFilter.value = '';
    apply();
  });
  apply();
})();
"""


def _atomic_write(path: Path, content: str) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_html_report(
    path: str | Path,
    dataset: Any,
    summary: Mapping[str, Any],
    details: Sequence[Mapping[str, Any]],
    coord_scale: float,
) -> None:
    """Write one atomic, self-contained visual evaluation report.

    Missing or unsupported screenshots are represented by per-sample
    placeholders and never abort generation of the remaining report.
    """

    annotation_path = Path(getattr(dataset, "annotation_path", ".")).expanduser().resolve()
    samples = list(getattr(dataset, "samples", ()) or ())
    detail_rows = [detail for detail in details if isinstance(detail, Mapping)]
    by_id = {
        str(detail["sample_id"]): detail
        for detail in detail_rows
        if detail.get("sample_id") is not None
    }
    by_index = {
        int(detail["sample_index"]): detail
        for detail in detail_rows
        if isinstance(detail.get("sample_index"), int)
        and not isinstance(detail.get("sample_index"), bool)
    }

    paired: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for position, raw_sample in enumerate(samples):
        if not isinstance(raw_sample, Mapping):
            continue
        detail = _detail_for_sample(
            raw_sample, position, detail_rows, by_id, by_index
        )
        paired.append((raw_sample, detail))

    captcha_types = [
        str(detail.get("captcha_type") or sample.get("captcha_type") or "unknown")
        for sample, detail in paired
    ]
    statuses = [str(detail.get("status") or "unknown") for _, detail in paired]
    cards = "".join(
        _sample_card(annotation_path, sample, detail, coord_scale, index)
        for index, (sample, detail) in enumerate(paired)
    )
    if not cards:
        cards = '<div class="panel empty-cell">No evaluated samples</div>'

    dataset_summary = summary.get("dataset")
    if not isinstance(dataset_summary, Mapping):
        dataset_summary = {}
    dataset_name = dataset_summary.get("name", getattr(dataset, "name", "Dataset"))
    annotations = dataset_summary.get("annotations", annotation_path)
    document = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%s · CAPTCHA evaluation report</title>
<style>%s</style>
</head>
<body><main>
<header class="page-header"><h1>%s · Evaluation report</h1><p>Annotations: %s</p></header>
%s
%s
%s
<section class="samples" aria-label="Per-sample evaluation details">%s</section>
</main><script>%s</script></body></html>
""" % (
        _escape(dataset_name),
        _CSS,
        _escape(dataset_name),
        _escape(annotations),
        _metric_cards(summary),
        _per_type_table(summary),
        _filter_controls(captcha_types, statuses),
        cards,
        _SCRIPT,
    )
    _atomic_write(Path(path), document)


__all__ = ["write_html_report"]
