import json
import os
import re
import tempfile
import base64
from io import BytesIO
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
import torch

import openai
from qwen_vl_utils import process_vision_info

def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def bbox_2_point(bbox, dig=2):
    point = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    point = [f"{item:.2f}" for item in point]
    return "({},{})".format(point[0], point[1])

def bbox_2_bbox(bbox, dig=2):
    bbox = [f"{item:.2f}" for item in bbox]
    return "({},{},{},{})".format(bbox[0], bbox[1], bbox[2], bbox[3])

def pred_2_point(s):
    floats = re.findall(r'-?\d+\.?\d*', s)
    floats = [float(num) for num in floats]
    if len(floats) == 2:
        return floats
    elif len(floats) == 4:
        return [(floats[0]+floats[2])/2, (floats[1]+floats[3])/2]
    else:
        return None

def extract_bbox(s):
    match = re.search(r'<\|box_start\|\>(.*?)<\|box_end\|\>', s)
    if match:
        extracted_text = match.group(1)
        cleaned_text = re.sub(r'[()\[\]]', '', extracted_text)
        pattern = r"(\d+),\s*(\d+),\s*(\d+),\s*(\d+)"
        numbers = re.findall(pattern, cleaned_text)
        if numbers:
            x1, y1, x2, y2 = numbers[0]
            return (int(x1), int(y1)), (int(x2), int(y2))
    return None

def extract_bbox_simple(s):
    pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    matches = re.findall(pattern, s)
    if matches:
        last_match = matches[-1]
        return (int(last_match[0]), int(last_match[1])), (int(last_match[2]), int(last_match[3]))
    return None

def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    return temp_file.name

class OSAtlas7BModel():
    def load_model(self, model_name_or_path="/root/OS-Atlas-Base-7B/", device="cuda"):
        self.device = device
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            device_map=device, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        self.generation_config = GenerationConfig.from_pretrained("/root/OS-Atlas-Base-7B/", trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_length=4096,
            do_sample=False,
            temperature=0.0
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)

    def inference(self, instruction, image_path):
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
        prompt_origin = 'In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with bbox)?. Additionally, if you think the task is infeasible (e.g., the task is not related to the image), output: <｜box_start｜>-1,-1,-1,-1<｜box_end｜>.'
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
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text_input], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        if '<|box_start|>' in response and '<|box_end|>' in response:
            pred_bbox = extract_bbox(response)
            if pred_bbox is not None:
                (x1, y1), (x2, y2) = pred_bbox
                pred_bbox = [pos / 1000 for pos in [x1, y1, x2, y2]]
                x1, y1, x2, y2 = pred_bbox
                if 0 < x1 < 1 and 0 < y1 < 1 and 0 < x2 < 1 and 0 < y2 < 1:
                    click_point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
                    result_dict["bbox"] = pred_bbox
                    result_dict["point"] = click_point
        else:
            click_point = pred_2_point(response)
            if click_point is not None:
                x, y = click_point
                if 0 < x < 1 and 0 < y < 1:
                    click_point = [x / 1000, y / 1000]
                    result_dict["point"] = click_point
        
        return result_dict

    def layout_gen(self, name, explanation, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        instruction = f"An functional region named \"{name}\" where {explanation}."
        prompt = f"In this UI screenshot, what is the position of the functional region corresponding to the instruction \"{instruction}\" (with bbox)?"
        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text_input], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        pred_bbox = extract_bbox(response)
        if pred_bbox is None:
            pred_bbox = extract_bbox_simple(response)
    
        if pred_bbox is not None:
            (x1, y1), (x2, y2) = pred_bbox
            pred_bbox = [pos / 1000 for pos in [x1, y1, x2, y2]]
            x1, y1, x2, y2 = pred_bbox
            if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:                
                result_dict["bbox"] = pred_bbox
        
        return result_dict
