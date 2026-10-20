"""Completion evidence for grounding reports; independent of scoring."""
import hashlib
import json
from collections import Counter


def result_key(result):
    return (result["img_path"], result["instruction_style"], result["gt_type"],
            result.get("lang", result.get("language")), result["prompt_to_evaluate"])


def coverage_digest(keys):
    payload = json.dumps(sorted(keys), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_coverage(results, expected_keys):
    """Reject missing, duplicate, or foreign records, including stale checkpoints."""
    actual = Counter(result_key(row) for row in results)
    expected = Counter(expected_keys)
    if actual != expected or any(n != 1 for n in actual.values()):
        missing = sum((expected - actual).values())
        extra = sum((actual - expected).values())
        raise RuntimeError(f"Incomplete task coverage: missing={missing}, extra={extra}; "
                           "checkpoints retained, no final report published")


def completion_metadata(results, expected_keys, max_samples):
    require_coverage(results, expected_keys)
    return {"status": "complete", "scope": "smoke" if max_samples is not None else "full",
            "expected_tasks": len(expected_keys), "completed_tasks": len(results),
            "task_ids_sha256": coverage_digest(expected_keys)}


def validate_completion(report, require_full=False):
    """Validate self-consistency; reports without explicit evidence are not reusable."""
    state = report.get("completion", {})
    rows = report.get("details")
    if not isinstance(rows, list) or not isinstance(state, dict):
        raise ValueError("Missing report details or completion metadata")
    if state.get("status") != "complete" or state.get("scope") not in ("full", "smoke"):
        raise ValueError("Missing successful completion state")
    if require_full and (state["scope"] != "full" or not rows):
        raise ValueError("Smoke or empty output cannot satisfy a full evaluation")
    keys = [result_key(r) for r in rows]
    if (state.get("expected_tasks") != len(rows) or state.get("completed_tasks") != len(rows)
            or len(set(keys)) != len(keys) or state.get("task_ids_sha256") != coverage_digest(keys)):
        raise ValueError("Result coverage does not match completion metadata")
    if report.get("metrics", {}).get("overall", {}).get("num_total") != len(rows):
        raise ValueError("Metrics and detail counts disagree")
    return state
