# models/grounding/gta1_7b_2507.py

from PIL import Image
from qwen_vl_utils import process_vision_info, smart_resize
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.generation import GenerationConfig
import torch
import re
import os


def extract_coordinates(raw_string):
    try:
        matches = re.findall(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)", raw_string)
        return [tuple(map(int, match)) for match in matches][0]
    except Exception:
        return (0, 0) 


SYSTEM_PROMPT = '''
You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {height} and width {width}. For elements with area, return the center point.

If you think the task is infeasible (e.g., the task is not related to the image), the output should be (-1,-1).

Output the coordinate pair exactly:
(x,y)
'''.strip()


class GTA1_7B_2507():
    def __init__(self):
        self.model = None
        self.processor = None
        self.generation_config = None

    def load_model(self, model_name_or_path="HelloKKMe/GTA1-7B"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda"
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            min_pixels=3136,
            max_pixels=4096 * 2160
        )

        self.set_generation_config(max_new_tokens=32, temperature=0.0, do_sample=False)
    def set_generation_config(self, **kwargs):
        if self.model is None:
            raise RuntimeError("Model not loaded yet. Call load_model() first.")
        gen_config = self.model.generation_config
        for k, v in kwargs.items():
            setattr(gen_config, k, v)
        self.model.generation_config = gen_config

    def inference(self, instruction, image_path):
        assert os.path.exists(image_path), f"Image not found: {image_path}"

        # --- 1. Load original image ---
        image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = image.size  

        # --- 2. Resize for model input ---
        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size
        factor = patch_size * merge_size

        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=factor,
            min_pixels=self.processor.image_processor.min_pixels,
            max_pixels=self.processor.image_processor.max_pixels,
        )
        resized_image = image.resize((resized_width, resized_height))

        # Scaling factor (from model input image -> original image)
        scale_x = orig_width / resized_width
        scale_y = orig_height / resized_height

        # --- 3. Construct message ---
        system_msg = {
            "role": "system",
            "content": SYSTEM_PROMPT.format(height=resized_height, width=resized_width)
        }
        user_msg = {
            "role": "user",
            "content": [
                {"type": "image", "image": resized_image},
                {"type": "text", "text": instruction}
            ]
        }

        # --- 4. Tokenize & Prepare Inputs ---
        image_inputs, video_inputs = process_vision_info([system_msg, user_msg])
        text = self.processor.apply_chat_template([system_msg, user_msg], tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        # --- 5. Generate ---
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=32, do_sample=False, temperature=1.0, use_cache=True)
        generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        # --- 6. Extract and rescale coordinates ---
        pred_x_pixel, pred_y_pixel = extract_coordinates(output_text)

        
        pred_x_orig = pred_x_pixel * scale_x
        pred_y_orig = pred_y_pixel * scale_y

        
        if orig_width > 0 and orig_height > 0:
            point_norm = [pred_x_orig / orig_width, pred_y_orig / orig_height]
        else:
            point_norm = None

       
        if pred_x_pixel == 0 and pred_y_pixel == 0 and "0,0" not in output_text:
            point_norm = None

        result_dict = {
            "result": "positive",
            "raw_response": output_text.strip(),
            "point": point_norm  
        }

        return result_dict
