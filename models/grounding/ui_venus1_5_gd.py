"""
UI-Venus Grounding v1.5 - UI Element Localization Model
"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from transformers.generation import GenerationConfig
import re
import os
from PIL import Image
from qwen_vl_utils import process_vision_info


class UIVenusGroundV15:
    """UI-Venus Grounding v1.5 Model"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
    
    def load_model(self, model_name_or_path="UI-Venus/UI-Venus-1.5-Pro", device="cuda"):
        """Load model
        
        Args:
            model_name_or_path: Model path
            device: Device (cuda / cpu)
        """
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2"
        ).to(device).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        # Default generation config - use deterministic output for evaluation
        self.generation_config = GenerationConfig.from_pretrained(model_name_or_path, trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0
        )

    def set_generation_config(self, **kwargs):
        """Set generation configuration"""
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)

    def inference(self, instruction, image_path, do_not_use_refusal=False):
        """Perform inference
        
        Args:
            instruction: UI element description
            image_path: Image path (str) or PIL.Image.Image object
            do_not_use_refusal: If True, don't include refusal option in prompt
            
        Returns:
            dict: Result dictionary containing result, raw_response, point
        """
        # Validate input
        if isinstance(image_path, str):
            assert os.path.exists(image_path) and os.path.isfile(image_path), f"Invalid image path: {image_path}"
        elif not isinstance(image_path, Image.Image):
            raise ValueError("image must be a file path (str) or a PIL.Image.Image object")
        
        # Build prompt
        if instruction.endswith('.'):
            instruction = instruction[:-1]
        
        if do_not_use_refusal:
            prompt = "Output the center point of the position corresponding to the following instruction: \n{}. \n\nThe output should just be the coordinates of a point, in the format [x,y]."
        else:
            prompt = "Output the center point of the position corresponding to the following instruction: \n{}. \n\nThe output should just be the coordinates of a point, in the format [x,y]. Additionally, if the task is infeasible (e.g., the task is not related to the image), the output should be [-1,-1]."
        
        full_prompt = prompt.format(instruction)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]

        # Process input
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
        inputs = inputs.to(self.model.device)

        # Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Parse output
        raw_output = output_text[0]
        point = self._parse_point(raw_output)
        
        # Determine result status
        if point is None:
            result_status = "wrong_format"
        elif point == [-1, -1]:
            result_status = "infeasible"
        else:
            result_status = "positive"
        
        return {
            "result": result_status,
            "raw_response": raw_output,
            "point": point
        }
    
    def _parse_point(self, text):
        """Parse coordinate point
        
        Supported formats:
        - [x, y] -> point coordinate
        - [x1, y1, x2, y2] -> bbox center point
        - [x1, y1], [x2, y2] -> center of two points
        
        Returns:
            Normalized [x, y] coordinates or [-1, -1] for infeasible, None for parse error
        """
        pattern_bbox = r"\[\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*\]"
        pattern_point = r"\[\s*-?\d+\s*,\s*-?\d+\s*\]"
        pattern_two_points = r"\[\s*-?\d+\s*,\s*-?\d+\s*\],\s*\[\s*-?\d+\s*,\s*-?\d+\s*\]"
        
        text = text.strip()
        
        try:
            if re.fullmatch(pattern_bbox, text, re.DOTALL):
                # [x1, y1, x2, y2] -> center point
                box = eval(text)
                point = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
            elif re.fullmatch(pattern_point, text, re.DOTALL):
                # [x, y]
                point = eval(text)
            elif re.fullmatch(pattern_two_points, text.replace(' ', ''), re.DOTALL):
                # [x1, y1], [x2, y2]
                bbox = eval('[' + text + ']')
                point = [(bbox[0][0] + bbox[1][0]) / 2, (bbox[0][1] + bbox[1][1]) / 2]
            else:
                # Try to extract numbers from text like "[123,456]..."
                point = list(map(int, text.split(']')[0].split('[')[1].split(',')))
            
            # Check for infeasible marker
            if point == [-1, -1]:
                return [-1, -1]
            
            # Normalize to 0-1 (assuming coordinates based on 1000x1000)
            abs_x = float(point[0] / 1000)
            abs_y = float(point[1] / 1000)
            return [abs_x, abs_y]
        except Exception:
            return None