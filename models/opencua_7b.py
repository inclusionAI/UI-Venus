# File: models/grounding/opencua_7b.py

import os
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
import base64
import re
import json


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


class OpenCUA_7B:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.device = None
        self.generation_config = {
            "max_new_tokens": 512,
            "temperature": 0.0,
        }

    def load_model(self, model_name_or_path="OpenCUA/OpenCUA-7B"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype='auto',
            device_map="cuda",
            trust_remote_code=True
        ).eval()
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.device = next(self.model.parameters()).device



    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)

    def inference(self, instruction: str, image_path: str):
        assert os.path.exists(image_path), f"Image not found: {image_path}"

        min_pixels = 2000000
        max_pixels = 4800000

        SYSTEM_PROMPT = (
            "You are a GUI agent. You are given a task and a screenshot of the screen. "
            "You need to perform a series of pyautogui actions to complete the task."
        )
        enhanced_instruction = (
        f"{instruction}\n"
        "If you think the task is infeasible (e.g., the task is not related to the image), "
        "output: `pyautogui.click(x=-1, y=-1)`.\n"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/png;base64,{encode_image(image_path)}",
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels},
                    {"type": "text", "text": enhanced_instruction},
                ],
            },
        ]

        # === Step 2: Tokenize input ===
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        # === Step 3: Preprocess image ===
        image = Image.open(image_path).convert('RGB')
        info = self.image_processor.preprocess(images=[image],min_pixels=min_pixels,
    max_pixels=max_pixels)
        pixel_values = torch.tensor(info['pixel_values']).to(dtype=torch.bfloat16, device=self.model.device)
        grid_thws = torch.tensor(info['image_grid_thw']).to(self.model.device)

        # === Step 4: Generate response ===
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    grid_thws=grid_thws,
                    max_new_tokens=self.generation_config["max_new_tokens"],
                    temperature=self.generation_config["temperature"],
                )

            prompt_len = input_ids.shape[1]
            generated_ids = generated_ids[:, prompt_len:]
            raw_response = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        except Exception:
            raw_response = ""

        input_height = info['image_grid_thw'][0][1]*14
        input_width = info['image_grid_thw'][0][2]*14


        # === Step 5: Parse coordinates from pyautogui.click or [x1,y1,x2,y2] ===
        try:
            image_size = image.size  # (width, height)

            # Case 1: pyautogui.click(x=xxx, y=yyy)
            match = re.search(r"pyautogui\.click\s*\(\s*x\s*=\s*(\d+\.?\d*)\s*,\s*y\s*=\s*(\d+\.?\d*)", raw_response, re.IGNORECASE)
            if match:
                x_px = float(match.group(1))
                y_px = float(match.group(2))
                W, H = input_width, input_height

                px_norm = max(0.0, min(1.0, x_px / W))
                py_norm = max(0.0, min(1.0, y_px / H))
                point = [px_norm, py_norm]
                return {
                    "result": "positive",
                    "raw_response": raw_response,
                    "bbox": None,
                    "point": point,
                    "format": "click_xy"
                }

            # Case 2: [x1, y1, x2, y2]
            match = re.search(r"\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]", raw_response)
            if match:
                x1, y1, x2, y2 = map(float, match.groups())
                W, H = input_width, input_height
                bbox_norm = [x1/W, y1/H, x2/W, y2/H]
                bbox_norm = [max(0.0, min(1.0, c)) for c in bbox_norm]
                cx = (bbox_norm[0] + bbox_norm[2]) / 2
                cy = (bbox_norm[1] + bbox_norm[3]) / 2
                point = [cx, cy]
                return {
                    "result": "positive",
                    "raw_response": raw_response,
                    "bbox": bbox_norm,
                    "point": point,
                    "format": "x1y1x2y2"
                }

            match = re.search(r'"?x"?[ :]+(\d+\.?\d*)[,\n}\]]+\s*"?y"?[ :]+(\d+\.?\d*)', raw_response)
            if match:
                x_px = float(match.group(1))
                y_px = float(match.group(2))
                W, H = input_width, input_height
                px_norm = max(0.0, min(1.0, x_px / W))
                py_norm = max(0.0, min(1.0, y_px / H))
                point = [px_norm, py_norm]
                return {
                    "result": "positive",
                    "raw_response": raw_response,
                    "bbox": None,
                    "point": point,
                    "format": "dict_xy"
                }

            raise ValueError("No coordinate pattern found")

        except Exception as e:
            print(f"[Parsing Error] {e}")
            return {
                "result": "negative",
                "raw_response": raw_response,
                "bbox": None,
                "point": None,
                "format": "error"
            }
