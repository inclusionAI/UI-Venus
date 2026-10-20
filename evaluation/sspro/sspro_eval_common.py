"""Shared ScreenSpot-Pro task construction, judgments, and score aggregation.

Used by agentic, direct, and multiple-sample evaluation. Each details record
contains image/task metadata, prediction, raw response, and correctness
(correct, wrong, or wrong_format). Positive tasks also include a bbox."""

import copy
import itertools
import json
import os

GT_TYPES = ['positive', 'negative']
INSTRUCTION_STYLES = ['instruction', 'action', 'description']
LANGUAGES = ['en', 'cn']


# ---------------------------------------------------------------------------
# Task construction
# ---------------------------------------------------------------------------
def build_tasks_to_run(args):
    """Build tasks from task, screenspot_test, inst_style, language, and gt_type."""
    if args.task == "all":
        # Exclude test_50/test_10 from "all": both duplicate samples in the official tasks.
        task_filenames = [
            os.path.splitext(f)[0]
            for f in os.listdir(args.screenspot_test)
            if f.endswith(".json")
            and os.path.splitext(f)[0] not in {"test_50", "test_10"}
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
                            task_instance["prompt_to_evaluate"] = task_instance["instruction"]
                        tasks_to_run.append(task_instance)
        print(f"Num of sample in {task_filename}: "
              f"{len(task_data)} * {len(inst_styles)} * {len(gt_types)} * {len(languages)} "
              f"= {len(task_data) * len(inst_styles) * len(gt_types) * len(languages)}")
    return tasks_to_run


def sample_id_of(result):
    """Return a sample key shared by task records and resumable result records."""
    return (
        result["img_path"],
        result["instruction_style"],
        result["gt_type"],
        result.get("lang", result.get("language")),
        result["prompt_to_evaluate"],
    )


# ---------------------------------------------------------------------------
# Per-sample judgement
# ---------------------------------------------------------------------------
def eval_sample_positive_gt(sample, response):
    bbox = sample["bbox"]
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
    img_size = sample["img_size"]
    bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1],
            bbox[2] / img_size[0], bbox[3] / img_size[1]]
    click_point = response["point"]
    if click_point is None:
        return "wrong_format"
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


# ---------------------------------------------------------------------------
# Metrics
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
