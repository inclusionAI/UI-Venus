# models/grounding/ui_tars_72b_vllm.py

import os
import re
import tempfile
from PIL import Image
import torch
import logging
from typing import Dict, Any, Optional

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from ast import literal_eval


def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    return temp_file.name


def extract_point(response: str) -> Optional[tuple]:
    if '(' in response and ')' in response:
        try:
            coords = response[response.find('(')+1 : response.find(')')]
            return literal_eval(coords)
        except Exception:
            return None
    else:
        return None


class UITARSModel_72B():
    def __init__(self, logger: logging.Logger = None):
        self.model: Optional[LLM] = None
        self.processor = None
        self.generation_config = {}
        self.sampling_params: Optional[SamplingParams] = None
        self.logger = logger or logging.getLogger(__name__)

    def load_model(self, model_name_or_path: str = "/root/UI-TARS-72B-DPO/"):
        self.logger.info(f"Loading UI-TARS model from {model_name_or_path}...")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

        # Initialize vLLM model
        self.model = LLM(
            model=model_name_or_path,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=8, 
            gpu_memory_utilization=0.9
        )

        # Set default generation config
        self.set_generation_config(max_length=4096, do_sample=False, temperature=0.0)
        self.logger.info(f"UI-TARS model {model_name_or_path} loaded successfully with vLLM.")

    def set_generation_config(self, **kwargs):
        """
        Compatibility method: map HF-style generation args to vLLM SamplingParams.
        """
        params_dict = {
            "max_tokens": getattr(self.sampling_params, "max_tokens", 128),
            "temperature": getattr(self.sampling_params, "temperature", 0.0),
            "top_p": getattr(self.sampling_params, "top_p", 1.0),
            "repetition_penalty": getattr(self.sampling_params, "repetition_penalty", 1.0),
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
            if not kwargs["do_sample"]:
                params_dict["temperature"] = 0.0

        self.sampling_params = SamplingParams(**params_dict)
        self.logger.info(f"Updated generation config: {self.sampling_params}")

    def inference(self, instruction: str, image_path: str) -> Dict[str, Any]:
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Please call .load_model() first.")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": "Output only the coordinate of one box in your response. "
                                + instruction +
                                " Additionally, if you think the task is infeasible (e.g., the task is not related to the image), the output should be (-1,-1)."
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = process_vision_info(messages)
        assert len(image_inputs) == 1, "Only one image supported"
        image_input = image_inputs[0]

        llm_inputs = {
            "prompt": text,
            "multi_modal_data": {"image": image_input}
        }

        try:
            outputs = self.model.generate([llm_inputs], sampling_params=self.sampling_params)
            response = outputs[0].outputs[0].text.strip()
        except Exception as e:
            self.logger.error(f"vLLM generation failed: {e}")
            response = "(0,0)"

        click_xy = extract_point(response)
        if click_xy:
            point = (click_xy[0] / 1000.0, click_xy[1] / 1000.0)
        else:
            point = (0.0, 0.0)

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": point
        }

        return result_dict
