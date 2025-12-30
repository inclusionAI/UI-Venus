import os
import re
import tempfile
from PIL import Image
import torch

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig

from qwen_vl_utils import process_vision_info
from ast import literal_eval


def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    return temp_file.name

def extract_point(response):
    if '(' in response and ')' in response:
        try:
            cordinates = response[response.find('(')+1:response.find(')')]
            return literal_eval(cordinates)
        except:
            return None
    else:
        return None


class UITARSModel():
    def load_model(self, model_name_or_path="bytedance-research/UI-TARS-7B-DPO", device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )

        self.generation_config = dict()
        self.set_generation_config(
            max_length=4096,
            do_sample=False,
            temperature=0.0
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)


    def inference(self, instruction, image_path):
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": "Output only the coordinate of one box in your response. " + instruction + "Additionally, if you think the task is infeasible (e.g., the task is not related to the image), the output should be (-1,-1)."},
                ],
            }
        ]

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
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        click_xy = extract_point(response)
        if click_xy:
            point = (click_xy[0]/1000, click_xy[1]/1000)
        else:
            point = (0,0)

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": point
        }

        return result_dict