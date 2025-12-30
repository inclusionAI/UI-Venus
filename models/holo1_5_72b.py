

from PIL import Image
from pydantic import BaseModel, Field
from typing import Literal, Any, Dict, List, Optional
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
import torch
import os
import json
import logging

from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info



class ClickAbsoluteAction(BaseModel):
    """Click at absolute coordinates."""
    action: Literal["click_absolute"] = "click_absolute"
    x: int = Field(..., description="The x coordinate, number of pixels from the left edge.")
    y: int = Field(..., description="The y coordinate, number of pixels from the top edge.")


ChatMessage: type = Dict[str, Any]


def get_chat_messages(task: str, image: Image.Image) -> list[ChatMessage]:
    """Create the prompt structure for navigation task"""
    prompt = f"""Localize an element on the GUI image according to the provided target and output a click position.
 * You must output a valid JSON following the format: {ClickAbsoluteAction.model_json_schema()}
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


class Holo1_5_72B():
    def __init__(self, logger: logging.Logger = None):
        self.model: Optional[LLM] = None
        self.processor = None
        self.generation_config = None
        self.sampling_params: Optional[SamplingParams] = None
        self.logger = logger or logging.getLogger(__name__)

    def load_model(self, model_name_or_path: str = "Hcompany/Holo1.5-72B"):
        self.logger.info(f"Loading Holo1.5 model from {model_name_or_path}...")

        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

  
        img_proc = self.processor.image_processor
        self.logger.info(f"Image processor: min_pixels={img_proc.min_pixels}, max_pixels={img_proc.max_pixels}")


        self.model = LLM(
            model=model_name_or_path,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=16, 
            gpu_memory_utilization=0.9,
        )


        self.set_generation_config(max_new_tokens=256, temperature=0.0, do_sample=False)

        self.logger.info(f"Holo1.5 model {model_name_or_path} loaded successfully with vLLM.")

    def set_generation_config(self, **kwargs):
        """
        Compatibility method. Maps HF-style generation args to vLLM SamplingParams.
        """

        params_dict = {
            "max_tokens": getattr(self.sampling_params, "max_tokens", 256),
            "temperature": getattr(self.sampling_params, "temperature", 0.0),
            "top_p": getattr(self.sampling_params, "top_p", 1.0),
            "repetition_penalty": getattr(self.sampling_params, "repetition_penalty", 1.0)
        }


        if "max_new_tokens" in kwargs:
            params_dict["max_tokens"] = kwargs["max_new_tokens"]
        if "temperature" in kwargs:
            params_dict["temperature"] = float(kwargs["temperature"])
        if "top_p" in kwargs:
            params_dict["top_p"] = float(kwargs["top_p"])
        if "repetition_penalty" in kwargs:
            params_dict["repetition_penalty"] = float(kwargs["repetition_penalty"])
        if "do_sample" in kwargs:
            params_dict["temperature"] = params_dict["temperature"] if kwargs["do_sample"] else 0.0


        self.sampling_params = SamplingParams(**params_dict)
        self.logger.info(f"Updated generation config: {self.sampling_params}")

    def inference(self, instruction: str, image_path: str):
        assert os.path.exists(image_path), f"Image not found: {image_path}"
        assert isinstance(instruction, str) and len(instruction.strip()) > 0, "Invalid instruction."

        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call .load_model() first.")


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


        image_inputs, _ = process_vision_info(messages)
        assert len(image_inputs) == 1, "Only one image is supported"
        image_input = image_inputs[0]


        llm_inputs = {
            "prompt": text_prompt,
            "multi_modal_data": {"image": image_input}
        }


        try:
            outputs = self.model.generate([llm_inputs], sampling_params=self.sampling_params)
            raw_output = outputs[0].outputs[0].text.strip()
        except Exception as e:
            self.logger.error(f"vLLM generation failed: {e}")
            raw_output = "{}"

        try:
            action_data = json.loads(raw_output)
            action = ClickAbsoluteAction(**action_data)
            pred_x_pixel, pred_y_pixel = action.x, action.y
        except Exception as e:
            self.logger.warning(f"[Parse Error] Failed to parse JSON: {e}, Raw output: {raw_output}")
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
