import torch
from transformers import AutoModelForImageTextToText, AutoProcessor,AutoTokenizer
from transformers.generation import GenerationConfig
import json
import re
import os
from PIL import Image
import io
from io import BytesIO
import base64
import ast

from qwen_vl_utils import process_vision_info,smart_resize

import math

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def resize_image(image, resized_width, resized_height):
    resized_image = image.resize((resized_width, resized_height), Image.LANCZOS)

    buffered = BytesIO()
    resized_image.save(buffered, format=image.format or "JPEG")

    return resized_image, base64.b64encode(buffered.getvalue()).decode('utf-8')

# bbox -> point (str)
def bbox_2_point(bbox, dig=2):
    # bbox [left, top, right, bottom]
    point = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    point = [f"{item:.2f}" for item in point]
    point_str = "({},{})".format(point[0], point[1])
    return point_str

# bbox -> bbox (str)
def bbox_2_bbox(bbox, dig=2):
    bbox = [f"{item:.2f}" for item in bbox]
    bbox_str = "({},{},{},{})".format(bbox[0], bbox[1], bbox[2], bbox[3])
    return bbox_str

# point (str) -> point
def pred_2_point(s):
    floats = re.findall(r'-?\d+\.?\d*', s)
    floats = [float(num) for num in floats]
    if len(floats) == 2:
        return floats
    elif len(floats) == 4:
        return [(floats[0]+floats[2])/2, (floats[1]+floats[3])/2]
    else:
        return None

# bbox (qwen str) -> bbox
def extract_bbox(s):
    pattern = r"<\|box_start\|\>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|\>"
    matches = re.findall(pattern, s)
    if matches:
        # Get the last match and return as tuple of integers
        last_match = matches[-1]
        return (int(last_match[0]), int(last_match[1])), (int(last_match[2]), int(last_match[3]))
    return None


def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    return temp_file.name

def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output


class Qwen3VLModel():
    def load_model(self, model_name_or_path="/root/model/Qwen3-VL-30B-A3B-Instruct", device="cuda"):
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2"
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        # Setting default generation config
        self.generation_config = GenerationConfig.from_pretrained(model_name_or_path, trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_length=2048,
            do_sample=True,
            temperature=0.9
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)

    def inference(self, instruction, image_path, pass_k=1):
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        # prompt_origin = 'Output the center point of the position corresponding to the instruction: "{}". The output should just be the coordinates of a point, in the format [x,y].'
        prompt_origin = 'Output the center point of the position corresponding to the instruction: {}. The output should just be the coordinates of a point, in the format [x,y]. Additionally, if you think the task is infeasible (e.g., the task is not related to the image), the output should be [-1,-1].'
        
        if instruction[-1] == '.':
            instruction = instruction[:-1]

        full_prompt = prompt_origin.format(instruction)
        
        max_pixels = 4800000
        min_pixels = 2000000
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": image_path,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels
                    },
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]

        # Preparation for inference
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

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
                
        pattern1 = r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]"
        pattern2 = r"\[\s*\d+\s*,\s*\d+\s*\]"
        pattern3 = r"\[\s*\d+\s*,\s*\d+\s*\], \[\s*\d+\s*,\s*\d+\s*\]"

        input_height = inputs['image_grid_thw'][0][1]*14
        input_width = inputs['image_grid_thw'][0][2]*14
        
        if not re.fullmatch(pattern2, output_text[0], re.DOTALL):
            pass

        if re.fullmatch(pattern1, output_text[0], re.DOTALL):
            box = eval(output_text[0])
            point = (box[0] + box[2])/2, (box[1] + box[3])/2
        elif re.fullmatch(pattern2, output_text[0], re.DOTALL):
            point = eval(output_text[0])
        elif re.fullmatch(pattern3, output_text[0], re.DOTALL):
            bbox = eval('['+output_text[0]+']')
            point = (bbox[0][0] + bbox[1][0])/2, (bbox[0][1] + bbox[1][1])/2
        else:
            try:
                point = list(map(int,output_text[0].split(']')[0].split('[')[1].split(',')))
            except:
                point = [0,0]
        
        try:
            abs_y = float(point[1]/1000)
            abs_x = float(point[0]/1000)
        except:
            abs_x, abs_y = [0,0]

        point = [abs_x,abs_y]
    
        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": output_text,
            "point": point
        }
        
        return result_dict
