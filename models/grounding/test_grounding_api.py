"""Grounding API smoke tests.

Usage from the repository root:
  GROUNDING_BASE_URL=http://127.0.0.1:8000/v1 \
  GROUNDING_API_KEY=empty \
  GROUNDING_MODEL=your-model-name \
  python models/grounding/test_grounding_api.py

The default run evaluates three bundled samples through the same non-thinking
path as eval_single_benchmark.py. Set GROUNDING_TEST_PARAMETERS=true to add
logprobs parameter checks.
"""

import json
import os
import sys
import time

from PIL import Image

# --- Import paths for the local eval_single_benchmark and repository models package ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
for p in (SCRIPT_DIR, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_single_benchmark import eval_sample_positive_gt  # noqa: E402
from models.grounding.ui_venus2_gd import DEFAULT_USER_PROMPT, Qwen35GroundModel, encode_image  # noqa: E402

# --- API endpoint configuration ---
# Provided through environment variables; bring your own OpenAI-compatible API service:
#   GROUNDING_BASE_URL  Endpoint URL
#   GROUNDING_API_KEY   API key; use "empty" when authentication is disabled
#   GROUNDING_MODEL     Model name
BASE_URL = os.environ.get("GROUNDING_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("GROUNDING_API_KEY", "empty")
MODEL_NAME = os.environ.get("GROUNDING_MODEL", "your-model-name")
TEST_PARAMETERS = os.environ.get("GROUNDING_TEST_PARAMETERS", "false").lower() == "true"

TEST_IMGS = os.path.join(SCRIPT_DIR, "test_cases", "images")
TEST_ANN = os.path.join(SCRIPT_DIR, "test_cases", "annotations", "test_samples.json")
SMOKE_IMG = os.path.join(TEST_IMGS, "vivado_windows", "screenshot_2024-12-10_00-12-57.png")


def smoke_messages(text):
    with Image.open(SMOKE_IMG) as image:
        encoded = encode_image(image)
    return [{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "min_pixels": 3136,
                "max_pixels": 12845056,
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            },
            {"type": "text", "text": DEFAULT_USER_PROMPT.format(instruction=text)},
        ],
    }]


def tc1_connectivity(client):
    """TC1: Test connectivity with a text-only chat completion."""
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=smoke_messages("search report in vivado"),
        max_tokens=1024,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    elapsed = time.time() - t0
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None) or (
        getattr(msg, "model_extra", None) or {}
    ).get("reasoning")
    print(f"[TC1] OK ({elapsed:.1f}s) content={msg.content!r}")
    print(f"[TC1] reasoning field present: {bool(reasoning)}")
    return {"ok": True, "elapsed": elapsed, "reasoning_present": bool(reasoning)}


def tc2_grounding_e2e(model):
    """TC2: Run end-to-end grounding evaluation on three samples."""
    samples = json.load(open(TEST_ANN))
    results = []
    for s in samples:
        img_path = os.path.join(TEST_IMGS, s["img_filename"])
        s["img_size"] = list(s["img_size"])

        t0 = time.time()
        result_dict, _img, _ = model.ground(
            s["instruction"], img_path, need_logprobs=False
        )
        elapsed = time.time() - t0

        correctness = eval_sample_positive_gt(s, result_dict)
        raw = result_dict.get("raw_response", "")
        point = result_dict.get("point")

        # Validate the coordinate space by checking raw response values against [0,1000].
        import re
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", raw)]
        coord_in_1000 = bool(nums) and all(-1 <= n <= 1000 for n in nums)

        results.append({
            "id": s["id"],
            "ui_type": s["ui_type"],
            "instruction": s["instruction"],
            "bbox": s["bbox"],
            "raw_response": raw,
            "point_norm": point,
            "correctness": correctness,
            "model_result": result_dict.get("result"),
            "api_error": bool(result_dict.get("api_error")),
            "elapsed": round(elapsed, 1),
            "coord_in_0_1000": coord_in_1000,
        })
        print(f"[TC2] {s['id']}: {correctness} ({elapsed:.1f}s) "
              f"raw={raw!r} point={point}")
    return results


def tc3_parameter_sanity(client):
    """TC3: Validate API parameter combinations against the implementation."""
    findings = {}

    # P1: logprobs=True + top_logprobs=3, as used by ground(need_logprobs=True).
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=smoke_messages("search report in vivado"),
            max_tokens=1024, logprobs=True, top_logprobs=3,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        has_lp = resp.choices[0].logprobs is not None
        findings["P1_logprobs"] = f"accepted, logprobs returned: {has_lp}"
    except Exception as e:
        findings["P1_logprobs"] = f"REJECTED: {type(e).__name__}: {e}"
    print(f"[TC3] P1 logprobs=True,top_logprobs=3 -> {findings['P1_logprobs']}")

    # P2: logprobs=False + top_logprobs=0, as used by ground(need_logprobs=False).
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=smoke_messages("search report in vivado"),
            max_tokens=1024, logprobs=False, top_logprobs=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        findings["P2_logprobs_off"] = "accepted"
    except Exception as e:
        findings["P2_logprobs_off"] = f"REJECTED: {type(e).__name__}: {e}"
    print(f"[TC3] P2 logprobs=False,top_logprobs=0 -> {findings['P2_logprobs_off']}")

    return findings


def main():
    from openai import OpenAI

    if MODEL_NAME == "your-model-name":
        print("[HINT] 请先通过环境变量配置你的 API 服务，例如：")
        print("  export GROUNDING_BASE_URL='http://127.0.0.1:8000/v1'")
        print("  export GROUNDING_API_KEY='your-api-key'")
        print("  export GROUNDING_MODEL='your-model-name'")
        print("[HINT] Configure your API service via env vars first "
              "(GROUNDING_BASE_URL / GROUNDING_API_KEY / GROUNDING_MODEL).\n")

    print("=" * 70)
    print(f"Endpoint: {BASE_URL}  Model: {MODEL_NAME}")
    print("=" * 70)

    model = Qwen35GroundModel(
        base_url=BASE_URL, api_key=API_KEY, model_name=MODEL_NAME,
    )
    r2 = tc2_grounding_e2e(model)

    if TEST_PARAMETERS:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY, max_retries=0, timeout=120)
        r1 = tc1_connectivity(client)
        r3 = tc3_parameter_sanity(client)
    else:
        print("[TC1/TC3] skipped; set GROUNDING_TEST_PARAMETERS=true to enable parameter probes")
        r1 = {"ok": True, "skipped": True}
        r3 = {}

    n_correct = sum(1 for r in r2 if r["correctness"] == "correct")
    print("\n" + "=" * 70)
    print(f"SUMMARY: grounding {n_correct}/{len(r2)} correct")
    print("=" * 70)

    parameter_errors = [value for value in r3.values() if "REJECTED:" in value or "ERROR:" in value]
    invalid_outputs = any(result["api_error"] or result["model_result"] == "wrong_format" for result in r2)
    if not r1["ok"] or invalid_outputs or parameter_errors:
        raise SystemExit(1)

    return {"tc1": r1, "tc2": r2, "tc3": r3}


if __name__ == "__main__":
    main()
