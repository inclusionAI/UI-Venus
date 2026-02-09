"""
UI-Venus 1.5 Grounding Evaluation Script (Single-node Auto Mode)
"""
import copy
import itertools
import torch
import json
import re
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
torch.manual_seed(114514)

GT_TYPES = ['positive', 'negative']
INSTRUCTION_STYLES = ['instruction', 'action', 'description']
LANGUAGES = ['en', 'cn']


def parse_args():
    parser = argparse.ArgumentParser(description='UI-Venus 1.5 Grounding Evaluation (Auto Mode)')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Model path')
    parser.add_argument('--screenspot_imgs', type=str, required=True, help='Image directory')
    parser.add_argument('--screenspot_test', type=str, required=True, help='Annotation directory')
    parser.add_argument('--task', type=str, required=True, help='Task name or "all"')
    parser.add_argument('--inst_style', type=str, required=True, choices=INSTRUCTION_STYLES + ['all'])
    parser.add_argument('--language', type=str, required=True, choices=LANGUAGES + ['all'], default='en')
    parser.add_argument('--gt_type', type=str, required=True, choices=GT_TYPES + ['all'])
    parser.add_argument('--log_path', type=str, required=True, help='Result save path')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cuda:0, cpu')
    
    args = parser.parse_args()
    return args


def build_model(args):
    """Build UI-Venus 1.5 model (using device_map=auto)"""
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    from transformers.generation import GenerationConfig
    
    model_path = args.model_name_or_path
    device = args.device
    
    logging.info(f"Loading model: {model_path}")
    logging.info(f"Device mode: {device}")
    
    # Use device_map="auto" to automatically distribute across GPUs
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "auto" else None,
        attn_implementation="flash_attention_2"
    ).eval()
    
    # If not using device_map="auto", move to specific device
    if device != "auto":
        model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Create wrapper class
    class UIVenusWrapper:
        def __init__(self, model, processor, tokenizer):
            self.model = model
            self.processor = processor
            self.tokenizer = tokenizer
            self.generation_config = {
                'max_new_tokens': 256,
                'do_sample': False,
                'temperature': 0,
            }
        
        def set_generation_config(self, **kwargs):
            self.generation_config.update(kwargs)
        
        def inference(self, instruction, image_path, do_not_use_refusal=False):
            from qwen_vl_utils import process_vision_info
            
            # Build prompt - using the new prompt format
            if instruction.endswith('.'):
                instruction = instruction[:-1]
            
            if do_not_use_refusal:
                prompt = f"Output the center point of the position corresponding to the following instruction: \n{instruction}. \n\nThe output should just be the coordinates of a point, in the format [x,y]."
            else:
                prompt = f"Output the center point of the position corresponding to the following instruction: \n{instruction}. \n\nThe output should just be the coordinates of a point, in the format [x,y]. Additionally, if the task is infeasible (e.g., the task is not related to the image), the output should be [-1,-1]."
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Process input
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            
            # Generate
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.generation_config.get('max_new_tokens', 128)
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # Parse output
            raw_output = output_text[0]
            point = self._parse_point(raw_output)
            
            # Determine result status
            if point is None:
                result_status = "wrong_format"
            elif point == [-1, -1]:
                result_status = "infeasible"
            else:
                result_status = "positive"
            
            return {
                "result": result_status,
                "raw_response": raw_output,
                "point": point
            }
        
        def _parse_point(self, text):
            """Parse coordinate point"""
            pattern1 = r"\[\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*\]"
            pattern2 = r"\[\s*-?\d+\s*,\s*-?\d+\s*\]"
            pattern3 = r"\[\s*-?\d+\s*,\s*-?\d+\s*\],\s*\[\s*-?\d+\s*,\s*-?\d+\s*\]"
            
            text = text.strip()
            try:
                if re.fullmatch(pattern1, text, re.DOTALL):
                    box = eval(text)
                    point = [(box[0] + box[2])/2, (box[1] + box[3])/2]
                elif re.fullmatch(pattern2, text, re.DOTALL):
                    point = eval(text)
                elif re.fullmatch(pattern3, text.replace(' ', ''), re.DOTALL):
                    bbox = eval('['+text+']')
                    point = [(bbox[0][0] + bbox[1][0])/2, (bbox[0][1] + bbox[1][1])/2]
                else:
                    point = list(map(int, text.split(']')[0].split('[')[1].split(',')))
                
                # Check for infeasible marker
                if point == [-1, -1]:
                    return [-1, -1]
                
                # Normalize to 0-1
                abs_x = float(point[0]/1000)
                abs_y = float(point[1]/1000)
                return [abs_x, abs_y]
            except:
                return None
    
    return UIVenusWrapper(model, processor, tokenizer)


def collect_results_to_eval(results, platform=None, group=None, application=None, 
                           language=None, gt_type=None, instruction_style=None, ui_type=None):
    """Filter results based on conditions"""
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
    """Generate attribute combinations"""
    unique_values = {
        "platform": set(),
        "group": set(),
        "application": set(),
        "language": set(),
        "gt_type": set(),
        "instruction_style": set(),
        "ui_type": set(),
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
    combinations = [dict(zip(filtered_values.keys(), combo)) for combo in attribute_combinations]
    return combinations


def calc_metric_for_result_list(results):
    """Calculate metrics"""
    num_total = len(results)
    correct_num = sum(1 for res in results if res["correctness"] == "correct")
    wrong_format_num = sum(1 for res in results if res["correctness"] == "wrong_format")

    text_results = collect_results_to_eval(results, ui_type="text")
    icon_results = collect_results_to_eval(results, ui_type="icon")

    text_correct = sum(1 for res in text_results if res["correctness"] == "correct")
    text_total = len(text_results)
    icon_correct = sum(1 for res in icon_results if res["correctness"] == "correct")
    icon_total = len(icon_results)
    
    return {
        "num_correct_action": correct_num,
        "num_total": num_total,
        "wrong_format_num": wrong_format_num,
        "action_acc": correct_num / num_total if num_total > 0 else 0,
        "text_acc": text_correct / text_total if text_total > 0 else 0,
        "icon_acc": icon_correct / icon_total if icon_total > 0 else 0
    }


def eval_sample_positive_gt(sample, response):
    """Evaluate positive sample"""
    bbox = sample["bbox"]
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
    img_size = sample["img_size"]
    bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
    
    click_point = response["point"]
    if click_point is None:
        return "wrong_format"
    if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
        return "correct"
    return "wrong"


def eval_sample_negative_gt(sample, response):
    """Evaluate negative sample"""
    point = response["point"]
    if point is None:
        return "wrong_format"
    if point == [-1, -1]:
        return "correct"
    return "wrong"


def evaluate_fine_grained(results):
    combinations = make_combinations(results, platform=True, application=True, instruction_style=True, gt_type=True)
    evaluation_result = {}
    for combo in combinations:
        filtered_results = collect_results_to_eval(
            results=results,
            platform=combo.get("platform"),
            application=combo.get("application"),
            instruction_style=combo.get("instruction_style"),
            gt_type=combo.get("gt_type")
        )
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        key = f"plat:{combo.get('platform')} app:{combo.get('application')} inst_style:{combo.get('instruction_style')} gt_type:{combo.get('gt_type')}"
        evaluation_result[key] = metrics
    return evaluation_result


def evaluate_seeclick_paper_style(results):
    combinations = make_combinations(results, platform=True, instruction_style=True, gt_type=True)
    evaluation_result = {}
    for combo in combinations:
        filtered_results = collect_results_to_eval(
            results=results,
            platform=combo.get("platform"),
            instruction_style=combo.get("instruction_style"),
            gt_type=combo.get("gt_type")
        )
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        key = f"plat:{combo.get('platform')} inst_style:{combo.get('instruction_style')} gt_type:{combo.get('gt_type')}"
        evaluation_result[key] = metrics
    return evaluation_result


def evaluate_leaderboard_detailed_style(results):
    combinations = make_combinations(results, application=True)
    evaluation_result = {}
    for combo in combinations:
        filtered_results = collect_results_to_eval(results=results, application=combo.get("application"))
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        key = f"app:{combo.get('application')}"
        evaluation_result[key] = metrics
    return evaluation_result


def evaluate_leaderboard_simple_style(results):
    combinations = make_combinations(results, group=True)
    evaluation_result = {}
    for combo in combinations:
        filtered_results = collect_results_to_eval(results=results, group=combo.get("group"))
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        key = f"group:{combo.get('group')}"
        evaluation_result[key] = metrics
    return evaluation_result


def evaluate_overall(results):
    return calc_metric_for_result_list(results)


def evaluate(results):
    """Aggregate evaluation results"""
    result_report = {
        "details": [],
        "metrics": {}
    }
    result_report["metrics"]["fine_grained"] = evaluate_fine_grained(results)
    result_report["metrics"]["seeclick_style"] = evaluate_seeclick_paper_style(results)
    result_report["metrics"]["leaderboard_simple_style"] = evaluate_leaderboard_simple_style(results)
    result_report["metrics"]["leaderboard_detailed_style"] = evaluate_leaderboard_detailed_style(results)
    result_report["metrics"]["overall"] = evaluate_overall(results)
    result_report["details"] = results
    return result_report


def main(args):
    model = build_model(args)
    logging.info("Model loaded successfully")

    # Load task data
    if args.task == "all":
        task_filenames = [
            os.path.splitext(f)[0]
            for f in os.listdir(args.screenspot_test)
            if f.endswith(".json")
        ]
    else:
        task_filenames = args.task.split(",")

    inst_styles = INSTRUCTION_STYLES if args.inst_style == "all" else args.inst_style.split(",")
    languages = LANGUAGES if args.language == "all" else args.language.split(",")
    gt_types = GT_TYPES if args.gt_type == "all" else args.gt_type.split(",")

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
                                raise AttributeError("Only positive samples and 'instruction' style are supported for Chinese.")
                            task_instance["prompt_to_evaluate"] = task_instance["instruction_cn"]
                        elif lang == "en":
                            task_instance["prompt_to_evaluate"] = task_instance["instruction"]
                        tasks_to_run.append(task_instance)
        
        logging.info(f"Task {task_filename}: {len(task_data)} * {len(inst_styles)} * {len(gt_types)} * {len(languages)} = {len(task_data) * len(inst_styles) * len(gt_types) * len(languages)}")
    
    logging.info(f"Total tasks: {len(tasks_to_run)}")

    # Inference
    results = []
    for sample in tqdm(tasks_to_run, desc="Evaluation progress"):
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
            "group": sample.get("group"),
            "platform": sample.get("platform"),
            "application": sample.get("application"),
            "lang": sample.get("language"),
            "instruction_style": sample.get("instruction_style"),
            "prompt_to_evaluate": sample.get("prompt_to_evaluate"),
            "gt_type": sample.get("gt_type", 'positive'),
            "ui_type": sample.get("ui_type"),
            "task_filename": sample["task_filename"], 
            "pred": point_in_pixel, 
            "raw_response": response["raw_response"]
        }
        
        if sample["gt_type"] == "positive":
            correctness = eval_sample_positive_gt(sample, response)
            sample_result["bbox"] = sample["bbox"]
        elif sample["gt_type"] == "negative":
            correctness = eval_sample_negative_gt(sample, response)
        else:
            raise ValueError("Wrong gt_type")

        sample_result["correctness"] = correctness
        results.append(sample_result)
        
        # Free memory
        del response
        
    # Save results
    result_report = evaluate(results)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, 'w') as f:
        json.dump(result_report, f, indent=4)
    
    # Print overall results
    overall = result_report["metrics"]["overall"]
    logging.info("=" * 60)
    logging.info(f"Evaluation complete! Total samples: {overall['num_total']}")
    logging.info(f"Accuracy: {overall['action_acc']:.4f}")
    logging.info(f"Text accuracy: {overall['text_acc']:.4f}")
    logging.info(f"Icon accuracy: {overall['icon_acc']:.4f}")
    logging.info(f"Format errors: {overall['wrong_format_num']}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main(parse_args())
