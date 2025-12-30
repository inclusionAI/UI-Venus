import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info  
from transformers.generation import GenerationConfig
import json
import re
import os
from PIL import Image
import tempfile

def bbox_2_point(bbox, dig=2):
    point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    point = [f"{item:.{dig}f}" for item in point]
    return f"({point[0]},{point[1]})"

def bbox_2_bbox(bbox, dig=2):
    bbox_str = [f"{item:.{dig}f}" for item in bbox]
    return f"({bbox_str[0]},{bbox_str[1]},{bbox_str[2]},{bbox_str[3]})"

def pred_2_point(s):
    floats = re.findall(r'-?\d+\.?\d*', s)
    floats = [float(num) for num in floats]
    if len(floats) == 2:
        return floats
    elif len(floats) == 4:
        return [(floats[0] + floats[2]) / 2, (floats[1] + floats[3]) / 2]
    else:
        return None

def extract_bbox(s):
    """Extract bounding box from string, supporting JSON and special token formats."""
    try:
        json_block = None
        m = re.search(r"```json(.*?)```", s, re.DOTALL)
        if m:
            json_block = m.group(1).strip()
        else:
            json_block = s.strip()
        data = json.loads(json_block)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "bbox_2d" in item:
                    bbox = item["bbox_2d"]
                    if isinstance(bbox, list) and len(bbox) == 4:
                        return (bbox[0], bbox[1]), (bbox[2], bbox[3])
        elif isinstance(data, dict) and "bbox_2d" in data:
            bbox = data["bbox_2d"]
            if isinstance(bbox, list) and len(bbox) == 4:
                return (bbox[0], bbox[1]), (bbox[2], bbox[3])
    except Exception:
        pass

    pattern1 = r"<\|box_start\|>\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]<\|box_end\|>"
    pattern2 = r"<\|box_start\|>\(\s*(\d+),\s*(\d+)\s*\),\(\s*(\d+),\s*(\d+)\s*\)<\|box_end\|>"
    pattern3 = r"\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]"  

    matches = re.findall(pattern1, s)
    if not matches:
        matches = re.findall(pattern2, s)
    if not matches:
        matches = re.findall(pattern3, s)

    if matches:
        last_match = matches[-1]
        return (int(last_match[0]), int(last_match[1])), (int(last_match[2]), int(last_match[3]))

    return None

def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    return temp_file.name

class Qwen3VLModel:
    def __init__(self):
        self.model = None
        self.processor = None

    def load_model(self, model_name_or_path="/root/Qwen_3-VL-8B_Instruct/"):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
            trust_remote_code=True
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

        self.generation_config = GenerationConfig.from_pretrained(model_name_or_path, trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_length=2048,
            do_sample=False,
            temperature=0.7
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)

    def inference(self, instruction, image_path):

        prompt_origin = 'Output the center point of the position corresponding to the instruction: {}. The output should just be the coordinates of a point, in the format [x,y]. Additionally, if you think the task is infeasible (e.g., the task is not related to the image), the output should be [-1,-1].'
        full_prompt = prompt_origin.format(instruction)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.generation_config.get("max_new_tokens", 128),
                do_sample=self.generation_config.get("do_sample", False),
                temperature=self.generation_config.get("temperature", 0.7)
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,  
            clean_up_tokenization_spaces=False
        )[0]

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        pred_bbox = extract_bbox(response)
        if pred_bbox is not None:
            (x1, y1), (x2, y2) = pred_bbox
            pred_bbox_normalized = [pos / 1000.0 for pos in [x1, y1, x2, y2]]
            click_point = [
                (pred_bbox_normalized[0] + pred_bbox_normalized[2]) / 2,
                (pred_bbox_normalized[1] + pred_bbox_normalized[3]) / 2
            ]
            result_dict["bbox"] = pred_bbox_normalized
            result_dict["point"] = click_point
        else:
            click_point_raw = pred_2_point(response)
            if click_point_raw:
                result_dict["point"] = [x / 1000.0 for x in click_point_raw]

        return result_dict
