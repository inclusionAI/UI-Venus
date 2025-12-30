import os
import re

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, GenerationConfig



class MiniCPMVModel():
    def __init__(self):
        pass

    def load_model(self, model_name_or_path="/root/MiniCPM-V-4_5_8B/", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained("/root/MiniCPM-V-4_5_8B/", trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device).eval()

        # Setting default generation config
        self.override_generation_config = GenerationConfig.from_pretrained("/root/MiniCPM-V-4_5_8B/", trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            max_length=None,
        )
    
    def set_generation_config(self, **kwargs):
        self.override_generation_config.update(kwargs)
        self.model.generation_config = GenerationConfig(**self.override_generation_config)

    def inference(self, instruction, image_path):
        if isinstance(image_path, str):
            image_path = image_path
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."
        
        # Prepare query
        prompt_origin = 'Outline the position corresponding to the instruction: {}. The output should be only [x1,y1,x2,y2]. Additionally, if you think the task is infeasible (e.g., the task is not related to the image), the output should be [-1,-1,-1,-1].'
        full_prompt = prompt_origin.format(instruction)
        
        msgs = [{'role': 'user', 'content': [image, full_prompt]}]

        response = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer
        )
        bbox = extract_first_bounding_box(response)
        if bbox:
            click_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        else:
            # Try matching a point
            click_point = extract_first_point(response)

        result_dict = {
            "result": "positive",
            "bbox": bbox,
            "point": click_point,
            "raw_response": response
        }
        
        return result_dict


def extract_first_bounding_box(text):
    # Regular expression pattern to match the first bounding box in the format <box>x0 y0 x1 y1</box>
    pattern = r"<box>(\d+) (\d+) (\d+) (\d+)</box>"
    
    # Search for the first match in the text with the DOTALL flag to support multi-line text
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        bbox = [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
        return [pos / 1000 for pos in bbox]
    
    return None


def extract_first_point(text):
    # Regular expression pattern to match the first point
    pattern = r"(\d+) (\d+)"
    
    # Search for the first match in the text with the DOTALL flag to support multi-line text
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        bbox = [int(match.group(1)), int(match.group(2))]
        return [pos / 1000 for pos in bbox]
    
    return None