import re
from PIL import Image
from io import BytesIO
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
import torch
import os


CLICK_REGEXES = [
                re.compile(r"click\s*\(\s*x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)\s*\)", re.IGNORECASE),
                re.compile(r"click\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", re.IGNORECASE),
            ]


def parse_xy_from_text(text: str):
    if "click" not in text.lower():
        return [-1, -1]
    for rx in CLICK_REGEXES:
        m = rx.search(text)
        if m:
            try:
                return int(m.group(1)), int(m.group(2))
            except Exception:
                continue
    return [-1, -1]


class GTA1_32B():
    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.processor = None
        self.prompt_tok = None
        self.sampling_params = None

    def load_model(self, model_name_or_path="/root/GTA1-32B/"):
        self.llm = LLM(
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            tokenizer_mode="slow",
            trust_remote_code=True,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": 1},
            tensor_parallel_size=8,
        )

        self.prompt_tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = self.llm.get_tokenizer()
        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

        self.sampling_params = SamplingParams(max_tokens=512, temperature=0.0)
    def set_generation_config(self, **kwargs):
        max_tokens = kwargs.get("max_new_tokens", 512)
        temperature = kwargs.get("temperature", 0.0)
        top_p = kwargs.get("top_p", 1.0)

        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )

    def inference(self, instruction, image_path):
        assert os.path.exists(image_path), f"Image not found: {image_path}"
        assert isinstance(instruction, str) and len(instruction.strip()) > 0, "Invalid instruction."

        image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = image.size

        factor = self.processor.patch_size * self.processor.merge_size
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=factor,
            min_pixels=self.processor.min_pixels,
            max_pixels=self.processor.max_pixels,
        )
        resized_image = image.resize((resized_width, resized_height))

        scale_x = orig_width / resized_width
        scale_y = orig_height / resized_height

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
            {"role": "user", "content": [
                {"type": "image", "image": image_path}, 
                {"type": "text", "text": enhanced_instruction},
            ]},
        ]

        prompt_text = self.prompt_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        prompt_text, n = re.subn(
            r"<\|media_begin\|>.*?<\|media_end\|>",
            "<|vision_start|><|image_pad|><|vision_end|>",
            prompt_text,
            flags=re.S
        )
        if n == 0:
            raise RuntimeError("Cannot find <|media_begin|>...<|media_end|> token.")

        try:
            outputs = self.llm.generate(
                [{
                    "prompt": prompt_text,
                    "multi_modal_data": {"image": [resized_image]}
                }],
                sampling_params=self.sampling_params
            )
            response_text = outputs[0].outputs[0].text
        except Exception:
            response_text = ""

        x_pixel, y_pixel = parse_xy_from_text(response_text)

        if x_pixel == -1 or y_pixel == -1 or x_pixel <= 0 or y_pixel <= 0:
            point_norm = None
        else:
            x_orig = x_pixel * scale_x
            y_orig = y_pixel * scale_y
            point_norm = [x_orig / orig_width, y_orig / orig_height]

        result_dict = {
            "result": "positive",      
            "raw_response": response_text,
            "point": point_norm       
        }

        return result_dict
