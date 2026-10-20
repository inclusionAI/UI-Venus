#!/usr/bin/env python3
"""Rejudge selected agentic results in place using the original zoom-in judge.

Load endpoint configuration from the environment before running. By default only
judge_error records are retried; use --statuses skipped,judge_error for deferred
judging. --limit restricts the number of records and --workers controls load.
The input is backed up as <results>.bak_rejudge before updating verdicts.
This uses the agentic judge template and robust decoder from the evaluator."""
import argparse
import importlib.util
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_PB_PATH = os.path.join(_HERE, "..", "eval_perceptionbench_agentic.py")


def load_pb_module():
    spec = importlib.util.spec_from_file_location("pbmod", os.path.abspath(_PB_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--dataset", default="PerceptionBench.jsonl")
    ap.add_argument("--workers", type=int, default=2,
                    help="Concurrent judge calls (try 1-2 when rate limited)")
    ap.add_argument("--max-retries", type=int, default=10)
    ap.add_argument("--statuses", default="judge_error",
                    help="Comma-separated judge_status values to retry; "
                         "default: judge_error; use skipped,judge_error for deferred judging")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N matching records")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    pb = load_pb_module()
    from openai import OpenAI
    # Judge settings come exclusively from the environment, as in the main evaluator.
    # Missing JUDGE_API_KEY is an error; no credentials are embedded.
    judge_api_key = os.getenv("JUDGE_API_KEY") or pb.JUDGE_DEFAULT_API_KEY
    if not judge_api_key:
        raise SystemExit(
            "JUDGE_API_KEY environment variable is required "
            "(set it in your .env file or export it before running)"
        )
    judge_client = OpenAI(
        api_key=judge_api_key,
        base_url=os.getenv("JUDGE_BASE_URL") or pb.JUDGE_DEFAULT_BASE_URL,
        timeout=int(os.getenv("API_TIMEOUT", "120")),
    )
    judge_model = os.getenv("JUDGE_MODEL") or pb.JUDGE_DEFAULT_MODEL
    temperature = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))

    dataset = {}
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            dataset[d["index"]] = d

    results = [json.loads(l) for l in open(args.results, encoding="utf-8")]
    target_statuses = {s.strip() for s in args.statuses.split(",") if s.strip()}
    bad_pos = [i for i, r in enumerate(results) if r.get("judge_status") in target_statuses]
    if args.limit:
        bad_pos = bad_pos[: args.limit]
    print(f"{args.results}: {len(results)} records, {len(bad_pos)} to rejudge "
          f"(statuses={sorted(target_statuses)}, judge={judge_model}, "
          f"temp={temperature}, workers={args.workers})", flush=True)
    if not bad_pos:
        return

    def rejudge(pos):
        r = results[pos]
        d = dataset.get(r["index"])
        if d is None:
            return pos, None
        ans = d.get("answer")
        ref = ans.get("answer") if isinstance(ans, dict) else ans
        out = pb.judge_general_qa_zoom_in_robust(
            judge_client, judge_model, d.get("problem", ""), r["prediction"], ref,
            temperature=temperature, max_retries=args.max_retries)
        if out["status"] == "ok":
            return pos, out
        return pos, None

    fixed, still_bad = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for n, (pos, out) in enumerate(ex.map(rejudge, bad_pos), 1):
            if out is not None:
                results[pos]["judge_result"] = 1 if out["verdict"] else 0
                results[pos]["judge_reason"] = out["reason"]
                results[pos]["judge_status"] = "ok"
                fixed += 1
            else:
                still_bad += 1
            if n % 10 == 0 or n == len(bad_pos):
                print(f"  progress {n}/{len(bad_pos)}  fixed {fixed}", flush=True)

    if fixed:
        shutil.copy(args.results, args.results + ".bak_rejudge")
        with open(args.results, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Recompute the displayed scores after updating judgments.
    total = len(results)
    correct = sum(1 for r in results if r.get("judge_result") == 1)
    jf = sum(1 for r in results if not isinstance(r.get("judge_result"), int))
    denom = total - jf
    print(f"Fixed {fixed}; still failing {still_bad}"
          + (" (rerun this command to retry)" if still_bad else ""))
    if denom:
        print(f"After rejudging: overall(excl)={correct / denom:.4f}  paper={correct / total:.4f}  "
              f"(correct={correct}, judge_fail={jf}, total={total})")


if __name__ == "__main__":
    sys.exit(main())
