"""Evaluate ScreenSpot-Pro and OS-World-G with the shared text-action pipeline.

Task construction, judgments, aggregation, and checkpoint keys are defined in
sspro_eval_common.py. This entry adapts trajectories and runs worker processes.
Each result can include complete trajectories; image references contain paths
rather than base64 payloads. See evaluation/README.md for wrapper commands."""

import argparse
import json
import logging
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from sspro_eval_common import (
    GT_TYPES,
    INSTRUCTION_STYLES,
    LANGUAGES,
    build_tasks_to_run,
    eval_sample_negative_gt,
    eval_sample_positive_gt,
    evaluate,
    sample_id_of,
)

# Add the shared pipeline directory when the wrapper uses this directory as cwd.
KIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if KIT_ROOT not in sys.path:
    sys.path.insert(0, KIT_ROOT)
from common.agentic_pipeline import DEFAULT_API_URL, AgenticGroundingPipeline
from common.grounding_completion import completion_metadata, require_coverage

logging.basicConfig(level=logging.INFO)
random.seed(114514)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCREENSPOT_IMGS = os.path.join(REPO_ROOT, "Screenspot-pro", "images")
DEFAULT_SCREENSPOT_TEST = os.path.join(REPO_ROOT, "Screenspot-pro", "annotations")

# Sentinel coordinates for an infeasible task.
INFEASIBLE_POINT_0_1000 = (-1.0, -1.0)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="ScreenSpot-Pro evaluation with the agentic text-action "
                    "grounding pipeline (common/agentic_pipeline.py)."
    )

    # API configuration.
    parser.add_argument('--api_url', type=str, default=DEFAULT_API_URL,
                        help="OpenAI-compatible chat/completions API URL")
    parser.add_argument('--api_key', type=str, default=None,
                        help="API key; None reads OPENAI_API_KEY")
    parser.add_argument('--model', type=str, required=True,
                        help="Model ID exposed by the API endpoint")

    # Dataset paths; direct entry defaults to Screenspot-pro beside this file.
    parser.add_argument('--screenspot_imgs', type=str, default=DEFAULT_SCREENSPOT_IMGS,
                        help="ScreenSpot-Pro image directory")
    parser.add_argument('--screenspot_test', type=str, default=DEFAULT_SCREENSPOT_TEST,
                        help="ScreenSpot-Pro annotation JSON directory")
    parser.add_argument('--task', type=str, default='all',
                        help="Comma-separated task filenames or 'all' (default; excludes "
                             "test_50/test_10 debug subsets that duplicate standard tasks)")
    parser.add_argument('--inst_style', type=str, default='instruction',
                        choices=INSTRUCTION_STYLES + ['all'],
                        help="Instruction style to use.")
    parser.add_argument('--language', type=str, default='en',
                        choices=LANGUAGES + ['all'])
    parser.add_argument('--gt_type', type=str, default='positive',
                        choices=GT_TYPES + ['all'])
    parser.add_argument('--max_samples', type=int, default=None,
                        help="Run the first N samples for a smoke test; None runs all samples")

    # Outputs and resume.
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--checkpoint_interval', type=int, default=10)

    # Concurrency.
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Concurrent inference threads (the pipeline httpx client is thread-safe)")
    parser.add_argument('--num_processes', type=int, default=1,
                        help="Worker processes: divide samples into N shards with separate checkpoints, "
                             "then merge at completion")

    # Inference modes.
    parser.add_argument('--prompt_mode', type=str, default='agentic',
                        choices=['agentic', 'direct_nothink', 'direct_think'],
                        help="agentic = multi-turn crop trajectories; direct_nothink / direct_think "
                             "= single-pass inference without crops")
    parser.add_argument('--max_tool_calls', type=int, default=3,
                        help="Total crop_image budget in agentic mode (protocol limit: 3)")
    parser.add_argument('--max_action_corrections', type=int, default=2,
                        help="Maximum correction rounds for unparseable outputs")
    parser.add_argument('--reasoning_effort', type=str, default='high')
    parser.add_argument('--thinking_budget', type=int, default=4096,
                        help="max_thinking_tokens")
    parser.add_argument('--max_completion_tokens', type=int, default=None,
                        help="Per-turn max_tokens; None uses thinking_budget")
    parser.add_argument('--temperature', type=float, default=0.0)

    # Multiple samples and majority voting.
    parser.add_argument('--sample_count', type=int, default=1,
                        help="Trajectories per sample; >1 generates n first turns and continues each branch, "
                             "then scores by majority vote")
    parser.add_argument('--sample_temperature', type=float, default=None,
                        help="Sampling temperature; None uses 0.6 when sample_count>1, otherwise "
                             "--temperature")

    # Crop execution settings.
    parser.add_argument('--crop_root', type=str,
                        default=os.path.join(REPO_ROOT, 'crops_agentic'),
                        help="Directory for generated crop images")
    parser.add_argument('--persist_crops', action='store_true',
                        help="Retain crop images after a trajectory (default: remove them)")
    parser.add_argument('--crop_upscale_factor', type=float, default=4.0)
    parser.add_argument('--crop_max_pixels', type=int, default=4_000_000)
    parser.add_argument('--crop_max_dimension', type=int, default=4096)
    parser.add_argument('--api_image_quality', type=int, default=90,
                        help="JPEG quality for images sent to the API (1-100)")

    # Rate limits and retries.
    parser.add_argument('--qpm', type=int, default=60)
    parser.add_argument('--rps', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=600.0)
    parser.add_argument('--max_attempts', type=int, default=5)

    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_model(args):
    sample_temperature = args.sample_temperature
    if sample_temperature is None:
        sample_temperature = 0.6 if args.sample_count > 1 else args.temperature
    args.sample_temperature = sample_temperature

    return AgenticGroundingPipeline(
        api_key=args.api_key,
        api_url=args.api_url,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        thinking_budget=args.thinking_budget,
        max_completion_tokens=args.max_completion_tokens,
        max_tool_calls=args.max_tool_calls,
        max_action_corrections=args.max_action_corrections,
        api_image_quality=args.api_image_quality,
        crop_upscale_factor=args.crop_upscale_factor,
        crop_max_pixels=args.crop_max_pixels,
        crop_max_dimension=args.crop_max_dimension,
        crop_root=args.crop_root,
        persist_crops=args.persist_crops,
        qpm=args.qpm,
        rps=args.rps,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
        workers_hint=max(args.num_workers, args.sample_count),
        prompt_mode=args.prompt_mode,
        answer_mode='point',
        # Preserve the SSPro direct prompts, including the nothink infeasibility convention.
        # The default grounding variant is used for HR-Bench and is not interchangeable.
        direct_prompt_variant='sspro',
    )


# ---------------------------------------------------------------------------
# Trajectory -> compare-script response adapter
# ---------------------------------------------------------------------------
def _is_infeasible_point(point):
    """The sentinel [-1.00,-1.00] means the model considers the target infeasible."""
    return (
        isinstance(point, (list, tuple))
        and len(point) == 2
        and abs(point[0] - INFEASIBLE_POINT_0_1000[0]) < 1e-6
        and abs(point[1] - INFEASIBLE_POINT_0_1000[1]) < 1e-6
    )


def trajectory_to_response(trajectory):
    """Adapt a trajectory to the evaluator response schema.

The point is normalized to [0,1] or None; result is positive, negative,
or None when there is no valid final answer."""
    point_0_1000 = trajectory.get("prediction_relative_0_1000")
    if _is_infeasible_point(point_0_1000):
        result, point = "negative", None
    elif point_0_1000 is not None:
        result = "positive"
        point = [point_0_1000[0] / 1000.0, point_0_1000[1] / 1000.0]
    else:
        result, point = None, None

    return {
        "point": point,
        "result": result,
        "raw_response": trajectory.get("final_content") or "",
        "error": trajectory.get("error"),
        "conversation_id": trajectory.get("conversation_id"),
        "model_turn_count": trajectory.get("model_turn_count"),
        "crop_action_count": trajectory.get("crop_action_count"),
        "tool_attempt_count": trajectory.get("tool_attempt_count"),
        "crop_image_paths": trajectory.get("crop_image_paths"),
        "reasoning_stats": trajectory.get("reasoning_stats"),
    }


def judge_sample(sample, response):
    """Score one trajectory using the positive or negative ground-truth rule.

A refusal on a positive task is wrong, while a missing valid answer is
wrong_format. Negative tasks use eval_sample_negative_gt unchanged."""
    if sample["gt_type"] == "positive":
        if response["result"] == "negative":
            return "wrong"
        return eval_sample_positive_gt(sample, response)
    if sample["gt_type"] == "negative":
        return eval_sample_negative_gt(sample, response)
    raise ValueError("Wrong instruction type")


def majority_correctness(labels):
    """Use strict majority: correct first, then wrong_format, otherwise wrong."""
    n = len(labels)
    correct = sum(1 for x in labels if x == "correct")
    wrong_format = sum(1 for x in labels if x == "wrong_format")
    if correct * 2 > n:
        return "correct"
    if wrong_format * 2 > n:
        return "wrong_format"
    return "wrong"


# ---------------------------------------------------------------------------
# Single-sample inference
# ---------------------------------------------------------------------------
def run_inference(model, instruction, img_path, args, filename, sample):
    """Run sample_count trajectories and return (responses, trajectories)."""
    task_id = (
        f"{filename.replace('/', '_').replace('.png', '')}"
        f"_{sample['gt_type']}_{sample['instruction_style']}_{sample['language']}"
    )
    if args.sample_count <= 1:
        trajectories = [
            model.run(instruction, img_path, task_id=task_id)
        ]
    else:
        trajectories = model.run_many(
            instruction,
            img_path,
            sample_count=args.sample_count,
            task_id=task_id,
            temperature=args.sample_temperature,
        )
    responses = [trajectory_to_response(t) for t in trajectories]
    return responses, trajectories


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _run_config_dict(args, proc_id=None):
    config = {
        "api_url": args.api_url,
        "model": args.model,
        "max_samples": args.max_samples,
        "task": args.task,
        "gt_type": args.gt_type,
        "language": args.language,
        "inst_style": args.inst_style,
        "screenspot_imgs": os.path.abspath(args.screenspot_imgs),
        "screenspot_test": os.path.abspath(args.screenspot_test),
        "prompt_mode": args.prompt_mode,
        "answer_mode": "point",
        "max_tool_calls": args.max_tool_calls,
        "max_action_corrections": args.max_action_corrections,
        "reasoning_effort": args.reasoning_effort,
        "thinking_budget": args.thinking_budget,
        "max_completion_tokens": args.max_completion_tokens,
        "temperature": args.temperature,
        "sample_count": args.sample_count,
        "sample_temperature": args.sample_temperature,
        "crop_root": args.crop_root,
        "persist_crops": args.persist_crops,
        "crop_upscale_factor": args.crop_upscale_factor,
        "crop_max_pixels": args.crop_max_pixels,
        "crop_max_dimension": args.crop_max_dimension,
        "api_image_quality": args.api_image_quality,
        "qpm": args.qpm,
        "rps": args.rps,
        "timeout": args.timeout,
        "max_attempts": args.max_attempts,
        "num_workers": args.num_workers,
        "num_processes": args.num_processes,
        "model_type": "agentic_pipeline",
    }
    if proc_id is not None:
        config["proc_id"] = proc_id
    return config


def _task_keys(tasks, args):
    return [(os.path.join(args.screenspot_imgs, t["img_filename"]),
             t["instruction_style"], t["gt_type"], t["language"],
             t["prompt_to_evaluate"]) for t in tasks]


def _write_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "w") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def _run_single_process(tasks_to_run, args, proc_id, num_procs):
    """Run inference for the subset assigned to one worker process."""
    model = build_model(args)
    proc_tag = f"[P{proc_id}/{num_procs}]" if num_procs > 1 else ""
    print(f"{proc_tag} Load model success: agentic_pipeline AgenticGroundingPipeline")
    print(f"{proc_tag} Prompt mode: {args.prompt_mode}, "
          f"sample_count: {args.sample_count}")

    # Assign each process its own slice of the tasks.
    total = len(tasks_to_run)
    shard_size = (total + num_procs - 1) // num_procs
    start = proc_id * shard_size
    end = min(start + shard_size, total)
    my_tasks = list(tasks_to_run[start:end])

    # Use separate checkpoint and log paths for each process.
    ckpt_path = args.checkpoint_path
    if ckpt_path and num_procs > 1:
        base, ext = os.path.splitext(ckpt_path)
        ckpt_path = f"{base}_p{proc_id}{ext}"

    log_path = args.log_path
    if log_path and num_procs > 1:
        base, ext = os.path.splitext(log_path)
        log_path = f"{base}_p{proc_id}{ext}"

    # Load checkpoint
    results = []
    processed_samples = set()

    if ckpt_path and os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, 'r') as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data.get("results", [])
                logging.info(f"{proc_tag} Loaded {len(results)} results from checkpoint")
                for result in results:
                    processed_samples.add(sample_id_of(result))
        except Exception as e:
            logging.error(f"{proc_tag} Error loading checkpoint: {e}")

    expected_keys = set(_task_keys(my_tasks, args))
    if (len(processed_samples) != len(results)
            or not processed_samples.issubset(expected_keys)):
        model.close()
        raise RuntimeError("Checkpoint has duplicate or out-of-shard tasks; retain it and "
                           "use a separate results directory for a changed task selection")

    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path) or '.', exist_ok=True)

    # Filter completed samples before inference.
    pending_tasks = []
    for sample in my_tasks:
        filename = sample["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)
        sample_id = (
            img_path,
            sample["instruction_style"],
            sample["gt_type"],
            sample["language"],
            sample["prompt_to_evaluate"]
        )
        if sample_id in processed_samples:
            continue
        pending_tasks.append((sample, img_path))

    print(f"{proc_tag} Total: {len(my_tasks)}, Pending: {len(pending_tasks)} "
          f"(skipped {len(my_tasks) - len(pending_tasks)})")

    def _json_safe(obj):
        # A model can copy "...", which literal_eval converts to Ellipsis.
        # Normalize Ellipsis recursively to None before writing JSON,
        # so serialization cannot corrupt an interrupted checkpoint.
        if obj is Ellipsis:
            return None
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        return obj

    def _save_checkpoint():
        checkpoint_data = {
            "results": _json_safe(results),
            "total_tasks": len(my_tasks),
            "completed_tasks": len(results)
        }
        # Write to a temporary file and rename atomically to preserve the previous checkpoint.
        tmp_path = ckpt_path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False)
        os.replace(tmp_path, ckpt_path)
        logging.info(f"{proc_tag} Saved checkpoint at {len(results)}/{len(my_tasks)}")

    def process_one(task_tuple):
        sample, img_path = task_tuple
        filename = sample["img_filename"]
        responses, trajectories = run_inference(
            model,
            instruction=sample["prompt_to_evaluate"],
            img_path=img_path,
            args=args,
            filename=filename,
            sample=sample,
        )

        img_size = sample["img_size"]
        sample_records = []
        for index, (response, trajectory) in enumerate(
                zip(responses, trajectories)):
            point = response.get("point")
            point_in_pixel = (
                [point[0] * img_size[0], point[1] * img_size[1]]
                if point else None
            )
            correctness = judge_sample(sample, response)
            sample_records.append({
                "sample_index": trajectory.get("sample_index", index),
                "pred": point_in_pixel,
                "correctness": correctness,
                "result": response.get("result"),
                "error": response.get("error"),
                "conversation_id": response.get("conversation_id"),
                "model_turn_count": response.get("model_turn_count"),
                "crop_action_count": response.get("crop_action_count"),
                "tool_attempt_count": response.get("tool_attempt_count"),
                "total_reasoning_tokens": (
                    (response.get("reasoning_stats") or {})
                    .get("total_reported_reasoning_tokens")
                ),
            })

        correctness = (
            sample_records[0]["correctness"]
            if len(sample_records) == 1
            else majority_correctness(
                [record["correctness"] for record in sample_records])
        )
        # Prefer a correct trajectory, otherwise the first trajectory with a predicted point.
        chosen = next(
            (r for r in sample_records if r["correctness"] == "correct"),
            None,
        ) or next(
            (r for r in sample_records if r["pred"] is not None),
            sample_records[0],
        )
        chosen_index = chosen["sample_index"]
        chosen_trajectory = trajectories[
            next(i for i, r in enumerate(sample_records)
                 if r["sample_index"] == chosen_index)
        ]

        sample_result = {
            "img_path": img_path,
            "group": sample.get("group"),
            "platform": sample["platform"],
            "application": sample["application"],
            "lang": sample["language"],
            "instruction_style": sample["instruction_style"],
            "prompt_to_evaluate": sample["prompt_to_evaluate"],
            "gt_type": sample["gt_type"],
            "ui_type": sample["ui_type"],
            "task_filename": sample["task_filename"],
            "pred": chosen["pred"],
            "raw_response": chosen_trajectory.get("final_content") or "",
            "inference_mode": args.prompt_mode,
            "model_type": "agentic_pipeline",
            "sample_count": len(sample_records),
            "samples": sample_records,
            "trajectories": trajectories,
        }

        if sample["gt_type"] == "positive":
            sample_result.update({"bbox": sample["bbox"]})
        sample_result.update({"correctness": correctness})
        return sample_result

    # Concurrent inference loop.
    results_lock = threading.Lock()
    completed_since_checkpoint = 0

    pbar_desc = f"{proc_tag}" if proc_tag else None
    pbar = tqdm(total=len(pending_tasks), desc=pbar_desc)

    try:
        if args.num_workers <= 1:
            for task_tuple in pending_tasks:
                try:
                    sample_result = process_one(task_tuple)
                except Exception as e:
                    logging.error(f"{proc_tag} Error: {e}")
                    pbar.update(1)
                    continue
                with results_lock:
                    results.append(sample_result)
                    completed_since_checkpoint += 1
                    if (ckpt_path and completed_since_checkpoint
                            >= args.checkpoint_interval):
                        completed_since_checkpoint = 0
                        _save_checkpoint()
                pbar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                future_to_task = {
                    executor.submit(process_one, t): t for t in pending_tasks
                }
                for future in as_completed(future_to_task):
                    try:
                        sample_result = future.result()
                    except Exception as e:
                        logging.error(f"{proc_tag} Error: {e}")
                        pbar.update(1)
                        continue
                    with results_lock:
                        results.append(sample_result)
                        completed_since_checkpoint += 1
                        if (ckpt_path and completed_since_checkpoint
                                >= args.checkpoint_interval):
                            completed_since_checkpoint = 0
                            _save_checkpoint()
                    pbar.update(1)
    finally:
        pbar.close()
        try:
            if ckpt_path:
                _save_checkpoint()
        finally:
            model.close()

    completion = completion_metadata(results, _task_keys(my_tasks, args), args.max_samples)

    # Save this process output.
    if log_path:
        result_report = evaluate(results)
        result_report["run_config"] = _run_config_dict(args, proc_id=proc_id)
        result_report["completion"] = completion
        _write_json(log_path, result_report)

    return results


def main(args):
    print(f"Load model success: agentic_pipeline AgenticGroundingPipeline")
    print(f"Prompt mode: {args.prompt_mode}, sample_count: {args.sample_count}")
    print(f"Processes: {args.num_processes}, Workers per process: {args.num_workers}")

    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be positive")
    if args.num_processes > 1 and not args.checkpoint_path:
        args.checkpoint_path = os.path.splitext(args.log_path)[0] + ".ckpt.json"

    tasks_to_run = build_tasks_to_run(args)
    print(f"Total tasks: {len(tasks_to_run)}")
    if not tasks_to_run:
        raise ValueError("No tasks selected; no final report published")
    random.shuffle(tasks_to_run)
    if args.max_samples is not None:
        tasks_to_run = tasks_to_run[:args.max_samples]
        print(f"--max_samples: running {len(tasks_to_run)} tasks")

    num_procs = args.num_processes if args.num_processes > 0 else 1

    if num_procs == 1:
        _run_single_process(tasks_to_run, args, proc_id=0, num_procs=1)
        logging.info("Evaluation of ScreenSpot finished.")
        return

    # Multiprocess execution.
    import multiprocessing

    ctx = multiprocessing.get_context('spawn')
    manager = ctx.Manager()
    shared_tasks = manager.list(tasks_to_run)

    processes = []
    for proc_id in range(num_procs):
        p = ctx.Process(
            target=_run_single_process,
            args=(shared_tasks, args, proc_id, num_procs),
        )
        processes.append(p)
        p.start()
        print(f"Started process {proc_id}/{num_procs} (pid={p.pid})")

    for p in processes:
        p.join()
        print(f"Process (pid={p.pid}) finished, exitcode={p.exitcode}")

    manager.shutdown()
    failed = [p.pid for p in processes if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"Worker processes failed: {failed}; checkpoints retained, "
                           "no final report published")

    # ------------------------------------------------------------------
    # Merge per-process checkpoints.
    # ------------------------------------------------------------------
    all_results = []
    base_ckpt, ext = os.path.splitext(args.checkpoint_path)
    for proc_id in range(num_procs):
        shard_ckpt = f"{base_ckpt}_p{proc_id}{ext}"
        with open(shard_ckpt, 'r') as f:
            shard_data = json.load(f)
        shard_results = shard_data["results"]
        shard_size = (len(tasks_to_run) + num_procs - 1) // num_procs
        assigned = tasks_to_run[proc_id * shard_size:(proc_id + 1) * shard_size]
        require_coverage(shard_results, _task_keys(assigned, args))
        all_results.extend(shard_results)
        print(f"Merged {len(shard_results)} results from {shard_ckpt}")

    print(f"Total merged results: {len(all_results)}")

    completion = completion_metadata(all_results, _task_keys(tasks_to_run, args), args.max_samples)
    result_report = evaluate(all_results)
    result_report["run_config"] = _run_config_dict(args)

    result_report["completion"] = completion
    _write_json(args.log_path, result_report)

    if args.checkpoint_path:
        merged_ckpt = {
            "results": all_results,
            "total_tasks": len(tasks_to_run),
            "completed_tasks": len(all_results),
        }
        _write_json(args.checkpoint_path, merged_ckpt)
        logging.info(f"Saved merged checkpoint with {len(all_results)} results")

    logging.info("Evaluation of ScreenSpot finished.")


if __name__ == "__main__":
    main(parse_args())
