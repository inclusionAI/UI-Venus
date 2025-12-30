# models/grounding/holo1_5_7b.py

from PIL import Image
from pydantic import BaseModel, Field
from typing import Literal, Any, Dict
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
import torch
import os
import json



class ClickAbsoluteAction(BaseModel):
    """Click at absolute coordinates."""
    action: Literal["click_absolute"] = "click_absolute"
    x: int = Field(..., description="The x coordinate, number of pixels from the left edge.")
    y: int = Field(..., description="The y coordinate, number of pixels from the top edge.")


ChatMessage: type = Dict[str, Any]


def get_chat_messages(task: str, image: Image.Image) -> list[ChatMessage]:
    """Create the prompt structure for navigation task"""
    prompt = f"""Localize an element on the GUI image according to the provided target and output a click position.
 * You must output a valid JSON following the format: {ClickAbsoluteAction.model_json_schema()}.
 * If the task is infeasible (e.g., the element does not exist or the task is unrelated to the image), output {{'action': 'click_absolute', 'x': -1, 'y': -1}}.
 * Do NOT output any other format.
 Your target is:"""

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{prompt}\n{task}"},
            ],
        },
    ]


class Holo1_5_7B():
    def __init__(self):
        self.model = None
        self.processor = None
        self.generation_config = None

    def load_model(self, model_name_or_path="Hcompany/Holo1.5-7B"):
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path
        )

        self.set_generation_config(max_new_tokens=256, temperature=0.0, do_sample=False)

    def set_generation_config(self, **kwargs):
        gen_config = self.model.generation_config
        for k, v in kwargs.items():
            setattr(gen_config, k, v)
        self.model.generation_config = gen_config

    def inference(self, instruction, image_path):
        assert os.path.exists(image_path), f"Image not found: {image_path}"
        assert isinstance(instruction, str) and len(instruction.strip()) > 0, "Invalid instruction."


        image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = image.size


        image_processor_config = self.processor.image_processor
        factor = image_processor_config.patch_size * image_processor_config.merge_size

        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=factor,
            min_pixels=image_processor_config.min_pixels,
            max_pixels=image_processor_config.max_pixels,
        )
        processed_image = image.resize((resized_width, resized_height), resample=Image.Resampling.LANCZOS)


        scale_x = orig_width / resized_width
        scale_y = orig_height / resized_height


        messages = get_chat_messages(instruction, processed_image)


        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt],
            images=[processed_image],
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)


        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_output = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


        try:
            action_data = json.loads(raw_output)
            action = ClickAbsoluteAction(**action_data)
            pred_x_pixel, pred_y_pixel = action.x, action.y
        except Exception:
            pred_x_pixel = pred_y_pixel = -1


        if pred_x_pixel < 0 or pred_y_pixel < 0:
            point_norm = None
        else:

            pred_x_orig = pred_x_pixel * scale_x
            pred_y_orig = pred_y_pixel * scale_y


            point_norm = [
                pred_x_orig / orig_width if orig_width > 0 else 0,
                pred_y_orig / orig_height if orig_height > 0 else 0
            ]

        result_dict = {
            "result": "positive",
            "raw_response": raw_output.strip(),
            "point": point_norm  
        }

        return result_dict
