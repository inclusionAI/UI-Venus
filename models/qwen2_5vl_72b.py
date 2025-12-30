import os
import torch
from typing import Dict, Any, List, Optional
from PIL import Image
import logging

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from qwen_vl_utils import process_vision_info, smart_resize


class Qwen2_5_VL_72B:
    def __init__(
        self,
        model_path: str = None,
        logger: logging.Logger = None,
        device: str = "cuda",
        **kwargs
    ):
        """
        Initialize configuration for the model. Does NOT load the model yet.

        Args:
            model_path (str): Path to the pretrained model.
            logger (logging.Logger): Logger instance.
            device (str): Default device ('cuda').
            **kwargs: Additional config parameters.
        """
        self.model_path = model_path
        self.logger = logger or logging.getLogger(__name__)
        self.device = device

        # Default configuration (can be overridden by kwargs)
        self.config = {
            "max_model_len": 4096,
            "max_num_seqs": 8,
            "tensor_parallel_size": 16,
            "gpu_memory_utilization": 0.9,
            "max_tokens": 128,
            "temperature": 0.0,
            "top_p": 0.95
        }
        # Override defaults
        self.config.update({k: v for k, v in kwargs.items() if k in self.config})

        # Placeholder for model & processor
        self.model: Optional[LLM] = None
        self.processor = None
        self.sampling_params: Optional[SamplingParams] = None

        self.logger.info("UI_Venus_Ground_72B_VLLM initialized with config (model not loaded).")

    def load_model(self, model_name_or_path: str = None):
        """
        Load the vLLM model and processor.

        Args:
            model_path (str): Model path to load. If None, uses self.model_path.
        """
        load_path = model_name_or_path
        if not load_path:
            raise ValueError("Model path must be provided either in __init__ or in load_model().")

        if self.model is not None:
            self.logger.warning("Model already loaded. Reinitializing...")
        
        self.processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)

        # Initialize vLLM model
        self.model = LLM(
            model=load_path,
            tensor_parallel_size=self.config["tensor_parallel_size"],
            gpu_memory_utilization=self.config["gpu_memory_utilization"],
            trust_remote_code=True,
            dtype="bfloat16"
        )

        # Create sampling params
        self.sampling_params = SamplingParams(
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            repetition_penalty=1.05,
            stop_token_ids=[],
        )

        self.logger.info(f"Model successfully loaded from: {load_path}")
        self.logger.info(f"SamplingParams: {self.sampling_params}")


    def set_generation_config(self, **kwargs):
        """Compatibility method for setting generation parameters."""
        if self.sampling_params is None:
            orig_params = {
                "max_tokens": 128,
                "temperature": 0.0,
                "top_p": 0.95,
                "repetition_penalty": 1.05,
            }
        else:
            orig_params = {
                "max_tokens": self.sampling_params.max_tokens,
                "temperature": self.sampling_params.temperature,
                "top_p": self.sampling_params.top_p,
                "repetition_penalty": self.sampling_params.repetition_penalty,
            }

        mapped_kwargs = {
            ("max_tokens" if k != "max_new_tokens" else "max_tokens"): v
            for k, v in kwargs.items()
        }
        mapped_kwargs = {"max_tokens" if k=="max_new_tokens" else k: v for k, v in kwargs.items()}

        updated_params = {**orig_params, **mapped_kwargs}
        self.sampling_params = SamplingParams(**updated_params)

        if self.logger:
            self.logger.info(f"Generation config updated: {self.sampling_params}")



    def inference(self, instruction: str, image_path: str) -> Dict[str, Any]:
        """
        Run grounding inference: locate object in image based on instruction.

        Args:
            instruction (str): Text instruction describing the target object.
            image_path (str): Path to input image file.

        Returns:
            Dict containing result, bbox, point, raw_response, etc.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Please call .load_model() first.")

        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        # Format prompt
        prompt_origin = 'Output the bounding box in the image corresponding to the instruction "{}" with grounding. The output should be only [x1,y1,x2,y2]. Additionally, if you think the task is infeasible (e.g., the task is not related to the image), the output should be [-1,-1,-1,-1].'
        full_prompt = prompt_origin.format(instruction)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "min_pixels": self.processor.image_processor.min_pixels,
                        "max_pixels": self.processor.image_processor.max_pixels,
                    },
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Extract image inputs
        image_inputs, video_inputs = process_vision_info(messages)
        assert len(image_inputs) == 1, "Only one image supported per request"
        image_input = image_inputs[0]

        # Prepare input for vLLM
        llm_inputs = {
            "prompt": text,
            "multi_modal_data": {"image": image_input}
        }

        # Generate
        outputs = self.model.generate([llm_inputs], sampling_params=self.sampling_params)
        output_text = outputs[0].outputs[0].text.strip()

        # Parse output as list [x1,y1,x2,y2]
        try:
            box = eval(output_text)
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError("Parsed box is not a 4-element list/tuple")
        except Exception as e:
            self.logger.warning(f"Failed to parse bounding box from output: {output_text}, error: {e}")
            box = [0, 0, 0, 0]

        # Get resized dimensions used by model
        orig_width, orig_height = image_input.size
        resized_height, resized_width = smart_resize(
            orig_height,
            orig_width,
            factor=28,
            min_pixels=self.processor.image_processor.min_pixels,
            max_pixels=99999999,
        )
        w, h = resized_width, resized_height

        # Normalize coordinates to [0, 1]
        try:
            abs_x1 = float(box[0]) / w
            abs_y1 = float(box[1]) / h
            abs_x2 = float(box[2]) / w
            abs_y2 = float(box[3]) / h
            normalized_box = [abs_x1, abs_y1, abs_x2, abs_y2]
        except Exception as e:
            self.logger.warning(f"Failed to normalize bbox: {box}, error: {e}")
            normalized_box = [0, 0, 0, 0]

        # Compute center point
        point = [(normalized_box[0] + normalized_box[2]) / 2, (normalized_box[1] + normalized_box[3]) / 2]

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": output_text,
            "bbox": normalized_box,
            "point": point
        }

        return result_dict
