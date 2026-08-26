"""
eval_single_benchmark.py
========================

Single-benchmark grounding evaluation script (direct inference).

Features:
     Direct grounding via Qwen35GroundModel (models/grounding/ui_venus2_gd.py);
     no zoom-in / test-time-scaling logic.
     Supports OpenAI-compatible API endpoints via --base_url / --api_key / --model_name.
     Thread-level concurrent inference via --num_workers.
     Multi-process sharding via --num_processes, with automatic result merging.
     Checkpoint & resume support via --checkpoint_path / --checkpoint_interval.

Usage (the endpoint, key, model name, and data paths are placeholders):

  python3 models/grounding/eval_single_benchmark.py \
      --base_url http://127.0.0.1:8000/v1 \
      --api_key your-api-key \
      --model_name your-model-name \
      --task all \
      --language en \
      --gt_type positive \
      --inst_style instruction \
      --num_processes 2 \
      --num_workers 4 \
      --norm_type 0-1000 \
      --checkpoint_interval 20 \
      --log_path results/screenspot_pro.json \
      --checkpoint_path results_mid/screenspot_pro.json \
      --screenspot_imgs /path/to/Screenspot-pro/images \
      --screenspot_test /path/to/Screenspot-pro/annotations

"""

import copy
import itertools
import threading

import json
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm
import random
from PIL import Image

logging.basicConfig(level=logging.INFO)
random.seed(114514)

GT_TYPES = ['positive', 'negative']
INSTRUCTION_STYLES = ['instruction', 'action', 'description']
LANGUAGES = ['en', 'cn']


def _ensure_parent_directory(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="ScreenSpot-Pro evaluation: direct grounding via OpenAI-compatible API."
    )

    # --- API endpoint configuration ---
    parser.add_argument('--base_url', type=str,
                        default='http://127.0.0.1/v1',
                        help="OpenAI 兼容 API 的 base URL")
    parser.add_argument('--api_key', type=str,
                        default='empty',
                        help="API key")
    parser.add_argument('--model_name', type=str,
                        default='qwen3.6-model',
                        help="API 端点的模型名称")

    # --- Data paths (defaults use bundled smoke samples; use full datasets for evaluation) ---
    parser.add_argument('--screenspot_imgs', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'test_cases', 'images'),
                        help="标注样本图片目录 (默认: 随附 test_cases/images)")
    parser.add_argument('--screenspot_test', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'test_cases', 'annotations'),
                        help="标注 JSON 目录 (默认: 随附 test_cases/annotations)")
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--inst_style', type=str, required=True,
                        choices=INSTRUCTION_STYLES + ['all'],
                        help="Instruction style to use.")
    parser.add_argument('--language', type=str, required=True,
                        choices=LANGUAGES + ['all'], default='en')
    parser.add_argument('--gt_type', type=str, required=True,
                        choices=GT_TYPES + ['all'])
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--num_workers', type=int, default=2,
                        help="并发推理线程数 (OpenAI 客户端线程安全)")
    parser.add_argument('--num_processes', type=int, default=2,
                        help="多进程数: 将样本等比划分为 N 份分别运行独立进程，"
                             "每个进程内部再用 --num_workers 线程并发。"
                             "进程间通过各自的 checkpoint 文件隔离，"
                             "运行结束后自动合并结果。")

    # --- Inference configuration ---
    parser.add_argument('--norm_type', type=str, default='0-1000',
                        choices=['pixel', '0-1000'],
                        help="坐标归一化方式: 'pixel' = 模型输出重采样像素坐标; "
                             "'0-1000' = 模型输出 [0, 1000] 归一化坐标(默认)")

    # --- Custom prompts (None selects the built-in default) ---
    parser.add_argument('--system_prompt', type=str, default=None,
                        help="自定义 system prompt。默认 None 时使用 user-only 消息(无 system role)。"
                             "设置后将在所有请求中附加该文本作为 system message。")
    parser.add_argument('--user_prompt', type=str, default=None,
                        help="自定义 user prompt 模板，用于初始 grounding。"
                             "支持 {instruction} 占位符，运行时替换为实际 instruction。"
                             "默认 None 时使用内置 prompt。")

    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_model(args):
    import sys

    # Add the repository root to sys.path and load through the namespace package.
    # models/grounding/ui_venus2_gd.py
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from models.grounding.ui_venus2_gd import Qwen35GroundModel

    model = Qwen35GroundModel(
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        norm_type=args.norm_type,
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
        debug=args.debug,
    )

    return model


# ---------------------------------------------------------------------------
# Evaluation metrics (identical to original)
# ---------------------------------------------------------------------------
def collect_results_to_eval(results, platform=None, group=None, application=None,
                            language=None, gt_type=None, instruction_style=None, ui_type=None):
    filtered_results = []
    for sample in results:
        if (platform is None or sample.get("platform") == platform) and \
           (group is None or sample.get("group") == group) and \
           (application is None or sample.get("application") == application) and \
           (language is None or sample.get("language") == language) and \
           (gt_type is None or sample.get("gt_type") == gt_type) and \
           (instruction_style is None or sample.get("instruction_style") == instruction_style) and \
           (ui_type is None or sample.get("ui_type") == ui_type):
            filtered_results.append(sample)
    return filtered_results


def make_combinations(results, platform=False, group=None, application=False,
                      language=False, gt_type=False, instruction_style=False, ui_type=False):
    unique_values = {
        "platform": set(), "group": set(), "application": set(),
        "language": set(), "gt_type": set(),
        "instruction_style": set(), "ui_type": set(),
    }
    for sample in results:
        if platform:
            unique_values["platform"].add(sample.get("platform"))
        if group:
            unique_values["group"].add(sample.get("group"))
        if application:
            unique_values["application"].add(sample.get("application"))
        if language:
            unique_values["language"].add(sample.get("language"))
        if gt_type:
            unique_values["gt_type"].add(sample.get("gt_type"))
        if instruction_style:
            unique_values["instruction_style"].add(sample.get("instruction_style"))
        if ui_type:
            unique_values["ui_type"].add(sample.get("ui_type"))

    filtered_values = {key: list(value) for key, value in unique_values.items() if value}
    if not filtered_values:
        return []

    attribute_combinations = list(itertools.product(*filtered_values.values()))
    combinations = []
    for combination in attribute_combinations:
        combinations.append(dict(zip(filtered_values.keys(), combination)))
    return combinations


def calc_metric_for_result_list(results):
    num_total = len(results)
    correct_num = sum(1 for res in results if res["correctness"] == "correct")
    wrong_format_num = sum(1 for res in results if res["correctness"] == "wrong_format")

    text_results = collect_results_to_eval(results, ui_type="text")
    icon_results = collect_results_to_eval(results, ui_type="icon")

    text_correct = sum(1 for res in text_results if res["correctness"] == "correct")
    text_total = len(text_results)
    icon_correct = sum(1 for res in icon_results if res["correctness"] == "correct")
    icon_total = len(icon_results)
    metrics = {
        "num_correct_action": correct_num,
        "num_total": num_total,
        "wrong_format_num": wrong_format_num,
        "action_acc": correct_num / num_total if num_total > 0 else 0,
        "text_acc": text_correct / text_total if text_total > 0 else 0,
        "icon_acc": icon_correct / icon_total if icon_total > 0 else 0
    }
    return metrics


def _get_img_size(img_path):
    try:
        with Image.open(img_path) as im:
            return im.size
    except Exception as e:
        logging.error(f"Failed to read img_size for {img_path}: {e}")
        return None


def eval_sample_positive_gt(sample, response):
    bbox = sample.get("bbox", [-1, -1, -1, -1])
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
    img_size = sample.get("img_size")
    if img_size is None:
        return "wrong_format"
    click_point = response["point"]
    if click_point is None:
        return "wrong_format"
    # Fix: handle refusal case - both GT and prediction are [-1,-1] (normalized as [-0.001,-0.001])
    if bbox == [-1, -1, -1, -1] and click_point[0] < 0 and click_point[1] < 0:
        return "correct"
    if bbox != [-1, -1, -1, -1]:
        bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1],
                bbox[2] / img_size[0], bbox[3] / img_size[1]]
    if (bbox[0] <= click_point[0] <= bbox[2]) and \
       (bbox[1] <= click_point[1] <= bbox[3]):
        return "correct"
    else:
        return "wrong"


def eval_sample_negative_gt(sample, response):
    if response["result"] == "negative":
        return "correct"
    elif response["result"] == "positive":
        return "wrong"
    else:
        return "wrong_format"


def evaluate_fine_grained(results):
    combinations = make_combinations(results, platform=True, application=True,
                                     instruction_style=True, gt_type=True)
    evaluation_result = {}
    for combo in combinations:
        platform = combo.get("platform")
        application = combo.get("application")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        filtered_results = collect_results_to_eval(
            results=results, platform=platform, application=application,
            instruction_style=inst_style, gt_type=gt_type)
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        key = f"plat:{platform} app:{application} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics
    return evaluation_result


def evaluate_seeclick_paper_style(results):
    combinations = make_combinations(results, platform=True,
                                     instruction_style=True, gt_type=True)
    evaluation_result = {}
    for combo in combinations:
        platform = combo.get("platform")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        filtered_results = collect_results_to_eval(
            results=results, platform=platform,
            instruction_style=inst_style, gt_type=gt_type)
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        key = f"plat:{platform} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics
    return evaluation_result


def evaluate_leaderboard_detailed_style(results):
    combinations = make_combinations(results, application=True)
    evaluation_result = {}
    for combo in combinations:
        application = combo.get("application")
        filtered_results = collect_results_to_eval(results=results, application=application)
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        key = f"app:{application}"
        evaluation_result[key] = metrics
    return evaluation_result


def evaluate_leaderboard_simple_style(results):
    combinations = make_combinations(results, group=True)
    evaluation_result = {}
    for combo in combinations:
        group = combo.get("group")
        filtered_results = collect_results_to_eval(results=results, group=group)
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        key = f"group:{group}"
        evaluation_result[key] = metrics
    return evaluation_result


def evaluate(results):
    result_report = {
        "details": [],
        "metrics": {}
    }
    result_report["metrics"]["fine_grained"] = evaluate_fine_grained(results)
    result_report["metrics"]["seeclick_style"] = evaluate_seeclick_paper_style(results)
    result_report["metrics"]["leaderboard_simple_style"] = evaluate_leaderboard_simple_style(results)
    result_report["metrics"]["leaderboard_detailed_style"] = evaluate_leaderboard_detailed_style(results)
    result_report["metrics"]["overall"] = calc_metric_for_result_list(results)
    result_report["details"] = results
    return result_report


# ---------------------------------------------------------------------------
# Single-sample inference (direct grounding)
# ---------------------------------------------------------------------------
def run_inference(model, instruction, img_path):
    """
    Run one direct grounding call.

    Returns:
        result_dict (dict): Contains at least ``point`` (normalized [x, y] or None) and ``raw_response``.
    """
    # ground() returns (result_dict, display_image, system_message).
    result_dict, _display_image, _system_message = model.ground(
        instruction, img_path, need_logprobs=False
    )
    return result_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _build_tasks_to_run(args):
    """Build the complete task list shared by all processes."""
    if args.task == "all":
        task_filenames = [
            os.path.splitext(f)[0]
            for f in os.listdir(args.screenspot_test)
            if f.endswith(".json")
        ]
    else:
        task_filenames = args.task.split(",")

    # Support subdirectories by reading every JSON file when task_filename is a directory.
    expanded_task_filenames = []
    for task_filename in task_filenames:
        task_path = os.path.join(args.screenspot_test, task_filename)
        if os.path.isdir(task_path):
            # Read all JSON files in this directory.
            for f in os.listdir(task_path):
                if f.endswith(".json"):
                    expanded_task_filenames.append(os.path.join(task_filename, os.path.splitext(f)[0]))
        else:
            expanded_task_filenames.append(task_filename)
    task_filenames = expanded_task_filenames

    if args.inst_style == "all":
        inst_styles = INSTRUCTION_STYLES
    else:
        inst_styles = args.inst_style.split(",")

    if args.language == "all":
        languages = LANGUAGES
    else:
        languages = args.language.split(",")

    if args.gt_type == "all":
        gt_types = GT_TYPES
    else:
        gt_types = args.gt_type.split(",")

    tasks_to_run = []
    for task_filename in task_filenames:
        dataset = task_filename + ".json"
        with open(os.path.join(args.screenspot_test, dataset), 'r') as f:
            task_data = json.load(f)

        for inst_style in inst_styles:
            for gt_type in gt_types:
                for lang in languages:
                    for task_instance in task_data:
                        task_instance = copy.deepcopy(task_instance)
                        task_instance["task_filename"] = task_filename
                        task_instance["gt_type"] = gt_type
                        task_instance["instruction_style"] = inst_style
                        task_instance["language"] = lang
                        if lang == "cn":
                            if inst_style != 'instruction' or gt_type != 'positive':
                                raise AttributeError(
                                    "Only positive samples and 'instruction' style "
                                    "are supported for Chinese instructions."
                                )
                            task_instance["prompt_to_evaluate"] = task_instance["instruction_cn"]
                        elif lang == "en":
                            if "instruction" in task_instance.keys():
                                task_instance["prompt_to_evaluate"] = task_instance["instruction"]
                            elif "prompt_to_evaluate" in task_instance.keys():
                                task_instance["prompt_to_evaluate"] = task_instance["prompt_to_evaluate"]
                            elif "name" in task_instance.keys():
                                task_instance["prompt_to_evaluate"] = task_instance["name"]
                            else:
                                raise KeyError(f"No instruction/prompt_to_evaluate/name found in sample: {list(task_instance.keys())}")
                        tasks_to_run.append(task_instance)
        print(f"Num of sample in {task_filename}: "
              f"{len(task_data)} * {len(inst_styles)} * {len(gt_types)} * {len(languages)} "
              f"= {len(task_data) * len(inst_styles) * len(gt_types) * len(languages)}")
    return tasks_to_run


def _run_single_process(tasks_to_run, args, proc_id, num_procs):
    """Run the subset assigned to one process."""
    model = build_model(args)
    proc_tag = f"[P{proc_id}/{num_procs}]" if num_procs > 1 else ""
    print(f"{proc_tag} Load model success: Qwen35GroundModel (direct)")

    # Split tasks evenly among processes.
    total = len(tasks_to_run)
    shard_size = (total + num_procs - 1) // num_procs
    start = proc_id * shard_size
    end = min(start + shard_size, total)
    my_tasks = tasks_to_run[start:end]

    # Build a separate checkpoint path for this process.
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
                results = [
                    result for result in checkpoint_data.get("results", [])
                    if not result.get("api_error")
                ]
                for result in results:
                    result.pop("thinking", None)
                logging.info(f"{proc_tag} Loaded {len(results)} results from checkpoint")
                for result in results:
                    sample_id = (
                        result["img_path"],
                        result["instruction_style"],
                        result["gt_type"],
                        result["lang"],
                        result["prompt_to_evaluate"]
                    )
                    processed_samples.add(sample_id)
        except Exception as e:
            logging.error(f"{proc_tag} Error loading checkpoint: {e}")

    if ckpt_path:
        _ensure_parent_directory(ckpt_path)

    # Filter out completed samples before inference.
    pending_tasks = []
    for sample in my_tasks:
        if "img_filename" in sample.keys():
            filename = sample["img_filename"]
        else:
            filename = sample["image_path"]
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

    def _save_checkpoint():
        checkpoint_data = {
            "results": results,
            "total_tasks": len(my_tasks),
            "completed_tasks": len(results)
        }
        with open(ckpt_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        logging.info(f"{proc_tag} Saved checkpoint at {len(results)}/{len(my_tasks)}")

    def process_one(task_tuple):
        sample, img_path = task_tuple
        response = run_inference(
            model,
            instruction=sample["prompt_to_evaluate"],
            img_path=img_path,
        )

        point = response.get("point")
        img_size = sample.get("img_size") or _get_img_size(img_path)
        sample["img_size"] = img_size
        # Fix: preserve [-1, -1] sentinel value (refusal) through coordinate transformation
        # Model outputs [-1, -1] in 0-1000 space -> normalized to [-0.001, -0.001] in [0,1] space
        # We must NOT scale this sentinel value, otherwise it becomes [-1.92, -1.08] etc.
        if point and point[0] < 0 and point[1] < 0:
            point_in_pixel = [-1, -1]  # Keep exact sentinel value for refusal
        else:
            point_in_pixel = [point[0] * img_size[0], point[1] * img_size[1]] if (point and img_size) else None

        sample_result = {
            "img_path": img_path,
            "group": sample.get("group"),
            "platform": sample.get("platform"),
            "application": sample.get("application"),
            "lang": sample.get("language"),
            "instruction_style": sample.get("instruction_style"),
            "prompt_to_evaluate": sample.get("prompt_to_evaluate"),
            "gt_type": sample.get("gt_type", "positive"),
            "ui_type": sample.get("ui_type"),
            "task_filename": sample.get("task_filename"),
            "pred": point_in_pixel,
            "raw_response": response.get("raw_response", ""),
            "model_type": "Qwen35GroundModel",
        }

        for extra_key in [
            "perplexity", "perplexity_content",
            "perplexity_x", "perplexity_y",
            "api_error",
        ]:
            if extra_key in response:
                sample_result[extra_key] = response[extra_key]

        if sample["gt_type"] == "positive":
            correctness = eval_sample_positive_gt(sample, response)
            sample_result.update({"bbox": sample.get("bbox", [-1, -1, -1, -1])})
        elif sample["gt_type"] == "negative":
            correctness = eval_sample_negative_gt(sample, response)
        else:
            raise ValueError("Wrong instruction type")

        sample_result.update({"correctness": correctness})
        return sample_result

    # Concurrent inference loop.
    results_lock = threading.Lock()
    completed_since_checkpoint = 0
    task_errors = []

    pbar_desc = f"{proc_tag}" if proc_tag else None
    pbar = tqdm(total=len(pending_tasks), desc=pbar_desc)

    if args.num_workers <= 1:
        for task_tuple in pending_tasks:
            try:
                sample_result = process_one(task_tuple)
            except Exception as e:
                logging.error(f"{proc_tag} Error: {e}")
                task_errors.append(str(e))
                pbar.update(1)
                continue
            with results_lock:
                results.append(sample_result)
                completed_since_checkpoint += 1
                if ckpt_path and completed_since_checkpoint >= args.checkpoint_interval:
                    completed_since_checkpoint = 0
                    _save_checkpoint()
            pbar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_task = {
                executor.submit(process_one, t): t for t in pending_tasks
            }
            for future in as_completed(future_to_task):
                task_tuple = future_to_task[future]
                try:
                    sample_result = future.result()
                except Exception as e:
                    logging.error(f"{proc_tag} Error: {e}")
                    task_errors.append(str(e))
                    pbar.update(1)
                    continue
                with results_lock:
                    results.append(sample_result)
                    completed_since_checkpoint += 1
                    if ckpt_path and completed_since_checkpoint >= args.checkpoint_interval:
                        completed_since_checkpoint = 0
                        _save_checkpoint()
                pbar.update(1)
    pbar.close()

    # Save final results for this process.
    if log_path:
        result_report = evaluate(results)
        result_report["run_config"] = {
            "base_url": args.base_url,
            "model_name": args.model_name,
            "num_workers": args.num_workers,
            "num_processes": args.num_processes,
            "proc_id": proc_id,
            "norm_type": args.norm_type,
            "system_prompt": args.system_prompt,
            "user_prompt": args.user_prompt,
        }
        _ensure_parent_directory(log_path)
        with open(log_path, 'w') as f:
            json.dump(result_report, f, indent=2, ensure_ascii=False)

    if ckpt_path:
        _save_checkpoint()
        logging.info(f"{proc_tag} Final checkpoint: {len(results)} results")

    api_errors = sum(bool(result.get("api_error")) for result in results)
    if task_errors:
        raise RuntimeError(
            f"{len(task_errors)} grounding task(s) failed; first error: {task_errors[0]}"
        )
    if api_errors:
        raise RuntimeError(f"{api_errors} grounding request(s) failed at the API layer")

    return results


def main(args):
    print(f"Load model success: Qwen35GroundModel (direct)")
    print(f"Processes: {args.num_processes}, Workers per process: {args.num_workers}")

    tasks_to_run = _build_tasks_to_run(args)
    print(f"Total tasks: {len(tasks_to_run)}")
    random.shuffle(tasks_to_run)

    num_procs = args.num_processes if args.num_processes > 0 else 1

    if num_procs == 1:
        # ---- Single-process mode ----
        _run_single_process(tasks_to_run, args, proc_id=0, num_procs=1)
        logging.info("Evaluation of ScreenSpot finished.")
        return

    # ---- Multi-process mode ----
    import multiprocessing

    # Use spawn to avoid sharing GIL/OpenAI client state through fork.
    ctx = multiprocessing.get_context('spawn')
    manager = ctx.Manager()
    # Shared task list.
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

    failed_processes = [p.pid for p in processes if p.exitcode != 0]
    if failed_processes:
        raise RuntimeError(f"Grounding worker processes failed: {failed_processes}")

    all_results = []
    if args.checkpoint_path:
        base_path, ext = os.path.splitext(args.checkpoint_path)
        results_key = "results"
    else:
        base_path, ext = os.path.splitext(args.log_path)
        results_key = "details"

    for proc_id in range(num_procs):
        shard_path = f"{base_path}_p{proc_id}{ext}"
        if os.path.exists(shard_path):
            with open(shard_path, 'r') as f:
                shard_data = json.load(f)
                shard_results = shard_data.get(results_key, [])
                all_results.extend(shard_results)
            print(f"Merged {len(shard_results)} results from {shard_path}")

    print(f"Total merged results: {len(all_results)}")

    # Evaluate & save final merged result
    result_report = evaluate(all_results)
    result_report["run_config"] = {
        "base_url": args.base_url,
        "model_name": args.model_name,
        "num_workers": args.num_workers,
        "num_processes": args.num_processes,
        "norm_type": args.norm_type,
        "system_prompt": args.system_prompt,
        "user_prompt": args.user_prompt,
    }

    _ensure_parent_directory(args.log_path)
    with open(args.log_path, 'w') as f:
        json.dump(result_report, f, indent=2, ensure_ascii=False)

    # Merge checkpoints.
    if args.checkpoint_path:
        merged_ckpt = {
            "results": all_results,
            "total_tasks": len(tasks_to_run),
            "completed_tasks": len(all_results),
        }
        with open(args.checkpoint_path, 'w') as f:
            json.dump(merged_ckpt, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved merged checkpoint with {len(all_results)} results")

    logging.info("Evaluation of ScreenSpot finished.")


if __name__ == "__main__":
    main(parse_args())
