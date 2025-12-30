import re
import os
import tempfile
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from transformers.generation import GenerationConfig

def parse_pyautogui(pred):
   pattern = r'pyautogui\.click\((?:x=)?(\d+\.\d+)(?:,\s*y=|,\s*)(\d+\.\d+)\)'
   match = re.search(pattern, pred)
   if match:
       x, y = match.groups()
       return [float(x), float(y)]
   else:
       return [0, 0]

def image_to_temp_filename(image):
   temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
   image.save(temp_file.name)
   return temp_file.name

class AguvisModel():
   chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
   def __init__(self, device="cuda"):
       self.device = device
       self.grounding_system_message = "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task."
       self.user_message = """Please complete the following tasks by clicking using `pyautogui.click`:\n"""
       self.infeasible_instruction = """If the target element is not visible or the task cannot be performed on the current screen, output: `pyautogui.click(x=-1, y=-1)`."""
   def load_model(self, model_name_or_path="/root/Aguvis-7B-720P/"):
       self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name_or_path)
       self.processor = Qwen2VLProcessor.from_pretrained(model_name_or_path)
       self.tokenizer = self.processor.tokenizer
       self.model.to(self.device)
       self.model.tie_weights()
   def set_generation_config(self, **kwargs):
       pass
   def inference(self, instruction, image_path):
       assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
       messages = [
           {
               "role": "system",
               "content": self.grounding_system_message
           },
           {
               "role": "user",
               "content": [
                   {
                       "type": "image",
                       "image": image_path,
                   },
                   {"type": "text", "text": self.user_message + instruction + self.infeasible_instruction},
               ],
           }
       ]
       # Preparation for inference
       text_input = self.processor.apply_chat_template(
           messages, tokenize=False, add_generation_prompt=True, chat_template=self.chat_template
       )
       text_input += "<|im_start|>assistant<|recipient|>all\n"
       image_inputs, video_inputs = process_vision_info(messages)
       inputs = self.processor(
           text=[text_input],
           images=image_inputs,
           videos=video_inputs,
           padding=True,
           return_tensors="pt",
       )
       inputs = inputs.to(self.device)
       cont = self.model.generate(**inputs, do_sample=False, max_new_tokens=1024)
       cont_toks = cont.tolist()[0][len(inputs.input_ids[0]) :]
       text_outputs = self.tokenizer.decode(cont_toks, skip_special_tokens=True).strip()
       result_dict = {
           "result": "positive",
           "format": "x1y1x2y2",
           "raw_response": text_outputs,
           "bbox": None,
           "point": None
       }
       click_point = parse_pyautogui(text_outputs)
       result_dict["point"] = click_point
       return result_dict