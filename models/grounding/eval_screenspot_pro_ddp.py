import copy
import itertools

import datetime
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
import json
import re
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm
import tempfile
import shutil
import time

from ui_venus1_5_gd import UIVenusGroundV15

logging.basicConfig(level=logging.INFO)
torch.manual_seed(114514)

GT_TYPES = ['positive', 'negative']
INSTRUCTION_STYLES = ['instruction', 'action', 'description']
LANGUAGES = ['en', 'cn']

def parse_args():
    # Modified parse_args function
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=False)
    parser.add_argument('--screenspot_imgs', type=str, required=True)
    parser.add_argument('--screenspot_test', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--inst_style', type=str, required=True, choices=INSTRUCTION_STYLES + ['all'], help="Instruction style to use.")
    parser.add_argument('--language', type=str, required=True, choices=LANGUAGES + ['all'], default='en', help="Language to use.")
    parser.add_argument('--gt_type', type=str, required=True, choices=GT_TYPES + ['all'], help="Ground truth type: 'positive' or 'negative'.")
    parser.add_argument('--log_path', type=str, required=True)
    
    args = parser.parse_args()
    # Get local_rank from environment variable (if exists)
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.local_rank = -1
    
    return args

def init_distributed_mode(args):
    # Initialize distributed environment
    # Prefer getting local_rank from environment variable (if not set via parse_args)
    if hasattr(args, 'local_rank') and args.local_rank != -1:
        local_rank = args.local_rank
    elif 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = -1
        
    if local_rank != -1:

        os.environ['NCCL_BLOCKING_WAIT'] = '0'  # Allow non-blocking wait
        os.environ['NCCL_TIMEOUT'] = '3600000'  # Set NCCL timeout to 1 hour (milliseconds)
        os.environ['NCCL_DEBUG'] = 'INFO'       # Increase log level to help debugging

        dist.init_process_group(
            backend='nccl',  # Use NCCL backend
            init_method='env://',
            timeout=datetime.timedelta(hours=1)  # Set process group timeout to 1 hour
        )
        # Set current device
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        print(f"NCCL timeout setting: {os.environ.get('NCCL_TIMEOUT')} milliseconds")

        # Update local_rank in args
        args.local_rank = local_rank
        return rank, world_size, local_rank
    else:
        # Non-distributed mode
        return 0, 1, 0

def build_model(args):
    model_type = args.model_type
    model_name_or_path = args.model_name_or_path

    model_maps = {
        'ui_venus_v15': UIVenusGroundV15,
    }

    assert model_type.lower() in model_maps, 'Unsupported model type: {}'.format(model_type)

    model = model_maps[model_type.lower()]()
    if model_name_or_path:
        model.load_model(model_name_or_path=model_name_or_path)
    else:
        model.load_model()
    model.set_generation_config(temperature=0, max_new_tokens=256, do_sample=False)
    
    # Move model to the corresponding device - ensure using correct local_rank
    local_rank = getattr(args, 'local_rank', -1)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and local_rank != -1 else "cpu")
    model.model.to(device)
    
    return model

def collect_results_to_eval(results, platform=None, group=None, application=None, language=None, gt_type=None, instruction_style=None, ui_type=None):
    """
    Filters the results based on provided values. None means include all (ignore filtering this attribute).

    Parameters:
        results (list): A list of dictionaries containing sample results.
    
    Returns:
        list: A filtered list of dictionaries based on the given criteria.
    """
    filtered_results = []

    for sample in results:
        # Check each filter condition; if None, consider it as passed
        if (platform is None or sample.get("platform") == platform) and \
           (group is None or sample.get("group") == group) and \
           (application is None or sample.get("application") == application) and \
           (language is None or sample.get("language") == language) and \
           (gt_type is None or sample.get("gt_type") == gt_type) and \
           (instruction_style is None or sample.get("instruction_style") == instruction_style) and \
           (ui_type is None or sample.get("ui_type") == ui_type):
            filtered_results.append(sample)

    return filtered_results


def make_combinations(results, platform=False, group=None, application=False, language=False, gt_type=False, instruction_style=False, ui_type=False):
    """
    Returns a list of combinations of values for attributes where the corresponding parameter is set to True.
    """
    # Initialize a dictionary to store unique values for each attribute
    unique_values = {
        "platform": set(),
        "group": set(),
        "application": set(),
        "language": set(),
        "gt_type": set(),
        "instruction_style": set(),
        "ui_type": set(),
    }

    # Collect unique values from the results
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

    # Filter out the attributes that are set to False (no need for combinations)
    filtered_values = {key: list(value) for key, value in unique_values.items() if value}
    if not filtered_values:
        return []

    # Generate all combinations of the selected attributes using itertools.product
    attribute_combinations = list(itertools.product(*filtered_values.values()))

    # Convert combinations into dictionaries with corresponding attribute names
    combinations = []
    for combination in attribute_combinations:
        combinations.append(dict(zip(filtered_values.keys(), combination)))

    return combinations


def calc_metric_for_result_list(results):
    """Calculates the metrics for a simple result list."""
    num_total = len(results)
    correct_num = sum(1 for res in results if res["correctness"] == "correct")
    wrong_format_num = sum(1 for res in results if res["correctness"] == "wrong_format")

    # Calculate text and icon specific metrics using collect_results_to_eval
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


def eval_sample_positive_gt(sample, response):
    bbox = sample["bbox"]
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]  # x1, y1, x2, y2
    # bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # x1, y1, w, h
    img_size = sample["img_size"]
    #bbox = [max(bbox[0]-10,0) / img_size[0], max(bbox[1]-10,0) / img_size[1], min(bbox[2]+10,img_size[0]) / img_size[0], min(bbox[3]+10,img_size[1]) / img_size[1]]
    bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
    
    click_point = response["point"]  # may be none
    # print(click_point)
    if click_point is None:
        return "wrong_format"
    # Check if the predicted point falls in the ground truth box
    if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
        return "correct"
    else:
        return "wrong"
    
def eval_sample_negative_gt(sample, response):
    if response["result"] == "negative":
        return "correct"
    elif response["result"] == "positive":
        return "wrong"
    else: ## response["result"] == wrong_format
        return "wrong_format"

def evaluate_fine_grained(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        platform=True, 
        application=True,
        instruction_style=True, 
        gt_type=True
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        platform = combo.get("platform")
        application = combo.get("application")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            platform=platform,
            application=application,
            instruction_style=inst_style,
            gt_type=gt_type
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"plat:{platform} app:{application} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_seeclick_paper_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        platform=True, 
        instruction_style=True, 
        gt_type=True
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        platform = combo.get("platform")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            platform=platform,
            instruction_style=inst_style,
            gt_type=gt_type
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"plat:{platform} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_leaderboard_detailed_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        application=True,
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        application = combo.get("application")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            application=application,
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"app:{application}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_leaderboard_simple_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        group=True,
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        group = combo.get("group")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            group=group,
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"group:{group}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_overall(results):
    """
    Evaluates the overall metrics for all results without any filtering.
    
    Parameters:
        results (list): A list of dictionaries containing sample results.
        
    Returns:
        dict: A dictionary containing the overall metrics.
    """
    # Calculate metrics for the entire result set
    metrics = calc_metric_for_result_list(results)
    
    return metrics


def evaluate(results):
    """Collect results and calculate metrics. You can comment out function calls or add new ones based on your need.
    """
    result_report = {
        "details": [],  # Store detailed information for each sample
        "metrics": {}
    }

    # TODO: comment out function calls based on your need
    result_report["metrics"]["fine_grained"] = evaluate_fine_grained(results)
    result_report["metrics"]["seeclick_style"] = evaluate_seeclick_paper_style(results)
    result_report["metrics"]["leaderboard_simple_style"] = evaluate_leaderboard_simple_style(results)
    result_report["metrics"]["leaderboard_detailed_style"] = evaluate_leaderboard_detailed_style(results)
    result_report["metrics"]["overall"] = evaluate_overall(results)

    # Save detailed results
    result_report["details"] = results

    return result_report


def main(args):
    # Initialize distributed environment
    rank, world_size, local_rank = init_distributed_mode(args)
    
    # Only main process outputs log information
    if rank == 0:
        print("Initializing distributed environment...")
        print(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
    
    if rank == 0:
        temp_dir = tempfile.mkdtemp()  # Use system default temp directory
        temp_dir_bytes = temp_dir.encode('utf-8')
        temp_dir_len = len(temp_dir_bytes)
        dist.broadcast(torch.tensor([temp_dir_len], dtype=torch.int32).cuda(), src=0)
        dist.broadcast(torch.ByteTensor(list(temp_dir_bytes)).cuda(), src=0)
    else:
        temp_dir_len_tensor = torch.tensor([0], dtype=torch.int32).cuda()
        dist.broadcast(temp_dir_len_tensor, src=0)
        temp_dir_len = temp_dir_len_tensor.item()
        temp_dir_bytes_tensor = torch.ByteTensor([0] * temp_dir_len).cuda()
        dist.broadcast(temp_dir_bytes_tensor, src=0)
        temp_dir = bytes(temp_dir_bytes_tensor.tolist()).decode('utf-8')

    os.makedirs(temp_dir, exist_ok=True)
    print(f'temp_dir: {temp_dir}')

    # Build and load model
    model = build_model(args)
    if rank == 0:
        print("Load model success")
    
    # Load task data
    if args.task == "all":
        task_filenames = [
            os.path.splitext(f)[0]
            for f in os.listdir(args.screenspot_test)
            if f.endswith(".json")
        ]
    else:
        task_filenames = args.task.split(",")

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
        # Create task instance list
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
                            if inst_style!= 'instruction' or gt_type != 'positive':
                                # TODO: Translate the data
                                raise AttributeError("Only positive samples and 'instruction' style are supported for Chinese instructions.")
                            task_instance["prompt_to_evaluate"] = task_instance["instruction_cn"]
                        elif lang == "en":
                            task_instance["prompt_to_evaluate"] = task_instance["instruction"]

                        tasks_to_run.append(task_instance)
        
        if rank == 0:
            print(f"Num of sample in {task_filename}: {len(task_data)} * {len(inst_styles)} * {len(gt_types)} * {len(languages)} = {len(task_data) * len(inst_styles) * len(gt_types) * len(languages)}")
    
    if rank == 0:
        print(f"Total tasks: {len(tasks_to_run)}")
        
    # Split data by rank
    chunk_size = len(tasks_to_run) // world_size
    remainder = len(tasks_to_run) % world_size
    
    if rank < remainder:
        start_idx = rank * (chunk_size + 1)
        end_idx = start_idx + chunk_size + 1
    else:
        start_idx = rank * chunk_size + remainder
        end_idx = start_idx + chunk_size
    
    # Get tasks to be processed by current process
    local_tasks = tasks_to_run[start_idx:end_idx]
    if rank == 0:
        print(f"Each process handles approximately {len(local_tasks)} tasks")
    
    # Process local tasks
    local_results = []
    for sample in tqdm(local_tasks, desc=f"Process {rank}"):
        filename = sample["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)

        response = model.inference(instruction=sample["prompt_to_evaluate"], image_path=img_path)

        point = response["point"]
        tmp_img = Image.open(img_path)
        img_size = tmp_img.size
        sample["img_size"] = img_size
        point_in_pixel = [point[0] * img_size[0], point[1] * img_size[1]] if point else None
        
        sample_result = {
            "img_path": img_path, 
            "group": sample["group"] if "group" in sample else None,
            "platform": sample["platform"] if "platform" in sample else None,
            "application": sample["application"] if "application" in sample else None,
            "lang": sample["language"] if "language" in sample else None,
            "instruction_style": sample["instruction_style"] if "instruction_style" in sample else None,
            "prompt_to_evaluate": sample["prompt_to_evaluate"] if "prompt_to_evaluate" in sample else None,
            "gt_type": sample["gt_type"] if "gt_type" in sample else 'positive',
            "ui_type": sample["ui_type"] if "ui_type" in sample else None,
            "task_filename": sample["task_filename"], 
            "pred": point_in_pixel, 
            "raw_response": response["raw_response"],
            "org_info": sample
        }
        
        if sample["gt_type"] == "positive":
            correctness = eval_sample_positive_gt(sample, response)
            sample_result.update({
                "bbox": sample["bbox"], 
            })
        elif sample["gt_type"] == "negative":
            correctness = eval_sample_negative_gt(sample, response)
        else:
            raise ValueError("Wrong instruction type")

        if rank == 0:  # Only print detailed info on main process to avoid output clutter
            print(correctness, point, sample['bbox'])

        sample_result.update({
            "correctness": correctness,
        })
        local_results.append(sample_result)
    
    # Result aggregation: each process saves results to temp file, then main process merges
    local_result_file = os.path.join(temp_dir, f"results_rank_{rank}.json")
    
    # Save local results
    with open(local_result_file, 'w') as f:
        json.dump(local_results, f, indent=4)
        
    # Main process merges all results
    if rank == 0:
        all_results = []
        for i in range(world_size):
            rank_result_file = os.path.join(temp_dir, f"results_rank_{i}.json")

            while not os.path.exists(rank_result_file):
                time.sleep(5)

            with open(rank_result_file, 'r') as f:
                rank_results = json.load(f)
                all_results.extend(rank_results)
        
        # Calculate evaluation metrics
        result_report = evaluate(all_results)
        
        # Save final results
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
        with open(args.log_path, 'w') as f:
            json.dump(result_report, f, indent=4)
        
        logging.info(f"Evaluation of ScreenSpot finished. Total samples: {len(all_results)}")
        
    else:
        # Other ranks wait for rank0 process to complete
        main_rank_file = os.path.join(temp_dir, "results_rank_0.json")
        while not os.path.exists(main_rank_file):
            time.sleep(2)
    
    dist.barrier()


if __name__ == "__main__":
    main(parse_args())
