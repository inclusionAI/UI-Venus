#!/usr/bin/env python3
# Adapted from MoonshotAI/PerceptionBench (Apache-2.0):
# https://github.com/MoonshotAI/PerceptionBench/blob/master/eval/eval.py
# Modified by the LBS contributors: multiprocessing, endpoint configuration,
# agentic evaluation integration, and resumable result handling.
# See the repository-root LICENSE and NOTICE. Runtime judge templates are separate.
"""Evaluator for PerceptionBench: free-form predictions, judged 0/1 by an LLM (general_qa).

Usage: MODEL=your-model python eval/eval.py   (all configuration via env vars, see .env.example)

Supports dual-API mode: prediction model and judge can use different endpoints
via JUDGE_API_KEY / JUDGE_BASE_URL env vars.
Uses multiprocessing to avoid thread-safety issues with certain API endpoints.
"""
import os, re, json, time, multiprocessing as mp
from openai import OpenAI

# Use fork on macOS so workers inherit dataset via copy-on-write
try:
    mp.set_start_method('fork', force=True)
except RuntimeError:
    pass

try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

PH = re.compile(r"<\|image_(\d+)\|>")
_HERE = os.path.dirname(os.path.abspath(__file__))
JUDGE_TEMPLATE = open(os.path.join(_HERE, "judge_prompt.txt"), encoding="utf-8").read()
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
PRED_TEMPERATURE = os.getenv("PRED_TEMPERATURE")
TOP_P = os.getenv("TOP_P")
TOP_K = os.getenv("TOP_K")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "120"))
# Concurrent streaming requests per worker process (two-level parallelism:
# CONCURRENCY processes x STREAMS_PER_PROC threads each, e.g. 4x4 = 16 in-flight).
STREAMS_PER_PROC = int(os.getenv("STREAMS_PER_PROC", "1"))
# Client-side image compression: downscale data-URI images to MAX_IMAGE_DIM and
# re-encode as JPEG. Off by default (baseline runs used original images; enable
# only when the backend cannot handle full-resolution images).
COMPRESS_IMAGES = os.getenv("COMPRESS_IMAGES", "0") == "1"
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "2048"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))
# Optional thinking control for vLLM-deployed models; all passed via
# chat_template_kwargs (vLLM top-level reasoning_effort rejects xhigh).
# ENABLE_THINKING: unset = server default; "0"/"false"/"off"/"no" = disable; else enable.
# THINKING_TOKEN_BUDGET: unset = not passed; integer = max_thinking_tokens.
# REASONING_EFFORT: unset uses the server default; supported values depend on the model template.
ENABLE_THINKING = os.getenv("ENABLE_THINKING")
THINKING_TOKEN_BUDGET = os.getenv("THINKING_TOKEN_BUDGET")
REASONING_EFFORT = os.getenv("REASONING_EFFORT")


def maybe_compress_image(uri):
    """Downscale/re-encode a data-URI image when COMPRESS_IMAGES=1."""
    if not COMPRESS_IMAGES or not isinstance(uri, str) or not uri.startswith("data:"):
        return uri
    try:
        import base64, io
        from PIL import Image
        header, b64 = uri.split(",", 1)
        im = Image.open(io.BytesIO(base64.b64decode(b64)))
        w, h = im.size
        if max(w, h) <= MAX_IMAGE_DIM and header.startswith("data:image/jpeg"):
            return uri
        im.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM))
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=JPEG_QUALITY)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return uri

# ---- Module-level globals inherited by forked worker processes ----
_POOL_DATA = None  # dataset list, set in main() before Pool creation


def stream_predict_with_retry(client, max_retries=MAX_RETRIES, **kw):
    """Streaming chat.completions.create with retries and token speed tracking.

    Returns (content, completion_tokens, tokens_per_sec).
    Streaming keeps the connection alive, avoiding Connection errors on slow endpoints.
    """
    last, attempt, _fallback_done = None, 0, False
    while True:
        try:
            kw_stream = {**kw, "stream": True}
            t_start = time.time()
            stream = client.chat.completions.create(**kw_stream)
            content_parts = []
            t_first = None
            usage = None
            for chunk in stream:
                if t_first is None:
                    t_first = time.time()
                if chunk.choices and chunk.choices[0].delta.content:
                    content_parts.append(chunk.choices[0].delta.content)
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage = chunk.usage
            t_end = time.time()
            content = "".join(content_parts)
            # Token count: prefer usage, otherwise estimate from content
            if usage and hasattr(usage, "completion_tokens") and usage.completion_tokens:
                comp_tokens = usage.completion_tokens
            else:
                comp_tokens = max(len(content) // 4, 1)
            # Token speed: based on generation time (first chunk to end)
            gen_time = max(t_end - (t_first or t_start), 0.1)
            tps = comp_tokens / gen_time
            return content, comp_tokens, tps
        except Exception as e:
            msg = str(e).lower()
            if not _fallback_done and "max_tokens" in msg and "max_completion_tokens" in msg:
                kw["max_completion_tokens"] = kw.pop("max_tokens")
                _fallback_done = True
                continue
            last = e
            if max_retries >= 0 and attempt >= max_retries:
                raise last
            time.sleep(min(2 ** min(attempt, 3), 8))
            attempt += 1


def build_messages(rec):
    """Interleave text and images from problem (+ <|image_N|> placeholders) + image list."""
    imgs = rec.get("image") or []
    imgs = imgs if isinstance(imgs, list) else []
    imgs = [maybe_compress_image(u) for u in imgs]
    prob = str(rec.get("problem", "") or "")
    content, last, used = [], 0, set()
    for m in PH.finditer(prob):
        n = int(m.group(1))
        if 1 <= n <= len(imgs):
            used.add(n - 1)
            seg = prob[last:m.start()]
            if seg:
                content.append({"type": "text", "text": seg})
            content.append({"type": "image_url", "image_url": {"url": imgs[n - 1]}})
            last = m.end()
    if prob[last:]:
        content.append({"type": "text", "text": prob[last:]})
    for k, img in enumerate(imgs):
        if k not in used:
            content.append({"type": "image_url", "image_url": {"url": img}})
    msgs = []
    system = rec.get("system")
    if isinstance(system, str) and system.strip():
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": content if content else prob})
    return msgs


def normalize_escape(s):
    wl = " 0123456789-"
    s = re.sub(rf"\\+[{wl}]", r"\\\\" + " ", s)
    s = re.sub(rf"\\+(?![{wl}])", r"\\", s)
    return re.sub(r"\\+n", "\n", s)


def decode_judge(resp):
    resp = str(resp).strip()
    if "[reason]" not in resp or "[judge]" not in resp:
        return False, "No [reason] or [judge] in output"
    reason = resp.split("[judge]")[0].split("[reason]")[-1].strip()
    verdict = resp.split("[judge]")[-1].strip()
    return ("true" in verdict.lower()), reason


def judge_general_qa(judge_client, judge_model, question, pred, reference):
    ref = str(reference if reference is not None else "").strip()
    prompt = (JUDGE_TEMPLATE
              .replace("{problem}", normalize_escape(str(question)).strip())
              .replace("{reference_answer}", normalize_escape(ref).strip())
              .replace("{assistant_answer}", normalize_escape(str(pred)).strip()))
    kw = dict(model=judge_model, temperature=0.3,
              max_tokens=1024,
              messages=[{"role": "user", "content": prompt}])
    try:
        r = judge_client.chat.completions.create(**kw)
    except Exception as e:
        msg = str(e).lower()
        if "max_tokens" in msg and "max_completion_tokens" in msg:
            kw["max_completion_tokens"] = kw.pop("max_tokens")
            r = judge_client.chat.completions.create(**kw)
        else:
            raise
    ok, reason = decode_judge(r.choices[0].message.content)
    return (1 if ok else 0), reason[:200]


def is_failed(pred):
    return (pred is None) or (isinstance(pred, str) and (not pred.strip() or pred.startswith("[error]")))


def run_one(rec, pred_client, judge_client, model, judge_model, max_tokens):
    pred, comp_tokens, tps = "", 0, 0.0
    try:
        kw = dict(model=model, messages=build_messages(rec), max_tokens=max_tokens)
        if PRED_TEMPERATURE is not None:
            kw["temperature"] = float(PRED_TEMPERATURE)
        if TOP_P is not None:
            kw["top_p"] = float(TOP_P)
        extra = {}
        if TOP_K is not None:
            extra["top_k"] = int(TOP_K)
        ctk = {}
        if ENABLE_THINKING is not None:
            ctk["enable_thinking"] = ENABLE_THINKING.strip().lower() not in ("0", "false", "off", "no")
        if THINKING_TOKEN_BUDGET is not None:
            ctk["max_thinking_tokens"] = int(THINKING_TOKEN_BUDGET)
        if REASONING_EFFORT is not None:
            ctk["reasoning_effort"] = REASONING_EFFORT.strip().lower()
        if ctk:
            extra["chat_template_kwargs"] = ctk
        if extra:
            kw["extra_body"] = extra
        pred, comp_tokens, tps = stream_predict_with_retry(pred_client, **kw)
    except Exception as e:
        pred = f"[error] {e}"
    if is_failed(pred):
        score, reason = 0, "failed to obtain answer"
    else:
        gt = rec.get("answer")
        ref = gt.get("answer") if isinstance(gt, dict) else gt
        try:
            score, reason = judge_general_qa(judge_client, judge_model, rec.get("problem", ""), pred, ref)
        except Exception as e:
            score, reason = 0, f"judge error: {e}"
    return {"index": rec.get("index"), "error_category": rec.get("error_category"),
            "source_bmk": rec.get("source_bmk"), "prediction": pred,
            "judge_result": int(score), "judge_reason": reason,
            "comp_tokens": comp_tokens, "tps": round(tps, 1)}


# ---- Multiprocessing worker functions ----

def _worker_init(model, judge_model, max_tokens):
    """Create per-process OpenAI clients. Dataset is inherited via fork COW."""
    global _POOL_MODEL, _POOL_JUDGE_MODEL, _POOL_MAX_TOKENS, _POOL_PRED_CLIENT, _POOL_JUDGE_CLIENT
    _POOL_MODEL = model
    _POOL_JUDGE_MODEL = judge_model
    _POOL_MAX_TOKENS = max_tokens
    _POOL_PRED_CLIENT = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        timeout=API_TIMEOUT,
    )
    judge_api_key = os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
    judge_base_url = os.getenv("JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    _POOL_JUDGE_CLIENT = OpenAI(api_key=judge_api_key, base_url=judge_base_url, timeout=API_TIMEOUT)


def _worker_run(idx):
    """Process a single record by index. Data is in _POOL_DATA (via COW)."""
    rec = _POOL_DATA[idx]
    return run_one(rec, _POOL_PRED_CLIENT, _POOL_JUDGE_CLIENT,
                   _POOL_MODEL, _POOL_JUDGE_MODEL, _POOL_MAX_TOKENS)


def _worker_run_chunk(idxs):
    """Process a chunk of indices with up to STREAMS_PER_PROC concurrent threads.

    The OpenAI client (httpx-based) is thread-safe, so worker threads share the
    per-process clients created in _worker_init.
    """
    if STREAMS_PER_PROC <= 1 or len(idxs) <= 1:
        return [_worker_run(i) for i in idxs]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=STREAMS_PER_PROC) as ex:
        return list(ex.map(_worker_run, idxs))


def main():
    global _POOL_DATA

    dataset = os.getenv("DATASET", "PerceptionBench.jsonl")
    model = os.getenv("MODEL")
    if not model:
        raise SystemExit("MODEL environment variable is required (see .env.example)")
    judge = os.getenv("JUDGE_MODEL", "gpt-oss-120b")
    num_procs = int(os.getenv("CONCURRENCY", "4"))
    streams = int(os.getenv("STREAMS_PER_PROC", "1"))
    max_tokens = int(os.getenv("MAX_TOKENS", "65536"))

    pred_url = os.getenv("OPENAI_BASE_URL", "")
    judge_url = os.getenv("JUDGE_BASE_URL") or pred_url
    dual = pred_url != judge_url
    print(f"Loading dataset: {dataset}", flush=True)
    data = [json.loads(l) for l in open(dataset, encoding="utf-8")]
    print(f"Loaded {len(data)} items. Model={model}[{pred_url}], "
          f"Judge={judge}[{judge_url}], Processes={num_procs}, Streams/proc={streams}, "
          f"Dual={dual}", flush=True)
    think_info = "server-default" if ENABLE_THINKING is None else ENABLE_THINKING
    if THINKING_TOKEN_BUDGET is not None:
        think_info += f" (budget={THINKING_TOKEN_BUDGET})"
    if REASONING_EFFORT is not None:
        think_info += f" effort={REASONING_EFFORT}"
    print(f"Thinking={think_info}, Compress={COMPRESS_IMAGES}", flush=True)

    # Set _POOL_DATA before creating Pool so forked children inherit via COW
    _POOL_DATA = data

    os.makedirs("results", exist_ok=True)
    # OUT_SUFFIX lets sharded runs keep distinct output files while MODEL stays
    # the exact name the backend expects (e.g. OUT_SUFFIX=_shard0of8).
    out_suffix = os.getenv("OUT_SUFFIX", "")
    out_prefix = f"results/PerceptionBench_{model.replace('/', '_')}{out_suffix}"
    out_jsonl = out_prefix + ".jsonl"

    # Resume: load existing results if the output file already exists
    done_indices = set()
    results = []
    if os.path.exists(out_jsonl):
        with open(out_jsonl, "r", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                results.append(rec)
                done_indices.add(rec["index"])
        print(f"Resuming: {len(done_indices)} items already done", flush=True)

    todo_indices = [i for i in range(len(data)) if i not in done_indices]
    # Re-key by actual index field to match data
    todo_indices = [i for i in range(len(data)) if data[i].get("index") not in done_indices]
    print(f"Remaining: {len(todo_indices)} items", flush=True)

    if not todo_indices:
        print("All items already completed.", flush=True)
    else:
        out_fh = open(out_jsonl, "a", encoding="utf-8")
        completed = len(done_indices)
        t_start = time.time()

        chunks = [todo_indices[i:i + streams] for i in range(0, len(todo_indices), streams)]
        with mp.Pool(processes=num_procs, initializer=_worker_init,
                      initargs=(model, judge, max_tokens)) as pool:
            for chunk_res in pool.imap_unordered(_worker_run_chunk, chunks, chunksize=1):
                for res in chunk_res:
                    results.append(res)
                    out_fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                    out_fh.flush()
                    completed += 1
                    elapsed = time.time() - t_start
                    rate = (completed - len(done_indices)) / max(elapsed, 0.1)
                    eta = (len(todo_indices) - (completed - len(done_indices))) / max(rate, 0.01) if rate > 0 else 0
                    tps = res.get("tps", 0)
                    ctok = res.get("comp_tokens", 0)
                    print(f"  [{completed}/{len(data)}] idx={res['index']} cat={res['error_category']} "
                          f"judge={res['judge_result']} tok={ctok} {tps:.1f}t/s {rate:.1f}q/s eta={eta/60:.0f}min", flush=True)

        out_fh.close()

    from collections import Counter
    tot, cor = Counter(), Counter()
    for r in results:
        c = r["error_category"] or "?"
        tot[c] += 1; cor[c] += r["judge_result"]
    overall = sum(cor.values()) / max(1, sum(tot.values()))
    print(f"\nOverall: {overall*100:.2f}%  ({sum(cor.values())}/{sum(tot.values())})", flush=True)
    for c in sorted(tot, key=lambda x: -tot[x]):
        print(f"  {c:32s} {cor[c]/tot[c]*100:5.1f}%  ({cor[c]}/{tot[c]})", flush=True)

    json.dump({"overall": overall, "per_category": {c: cor[c] / tot[c] for c in tot}},
              open(out_prefix + "_scores.json", "w"), indent=2, ensure_ascii=False)
    print("saved to", out_jsonl, flush=True)


if __name__ == "__main__":
    main()
