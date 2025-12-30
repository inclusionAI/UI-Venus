import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor,AutoTokenizer
from transformers.generation import GenerationConfig


from qwen_vl_utils import process_vision_info,smart_resize


class GUI_G2_7B():
    def load_model(self, model_name_or_path="/root/ckpt/huggingface/"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        self.generation_config = GenerationConfig.from_pretrained(model_name_or_path, trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_length=2048,
            do_sample=False,
            temperature=0.0
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)


    
    def inference(self, instruction, image_path):
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
        
        prompt_origin = 'Outline the position corresponding to the instruction: {}. The output should be only [x1,y1,x2,y2]. Additionally, if you think the task is infeasible (e.g., the task is not related to the image), the output should be [-1,-1,-1,-1].'
        full_prompt = prompt_origin.format(instruction)

        min_pixels = 2000000
        max_pixels = 4800000
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": image_path,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels
                    },
                    {"type": "text", "text": full_prompt},
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
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
                
        input_height = inputs['image_grid_thw'][0][1]*14
        input_width = inputs['image_grid_thw'][0][2]*14

        try:
            box = eval(output_text[0])
            abs_y1 = float(box[1]/input_height)
            abs_x1 = float(box[0]/input_width)
            abs_y2 = float(box[3]/input_height)
            abs_x2 = float(box[2]/input_width)
            box = [abs_x1,abs_y1,abs_x2,abs_y2]
        except:
            box = [0,0,0,0]

        point = [(box[0]+box[2])/2,(box[1]+box[3])/2]
        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": output_text,
            "bbox": box,
            "point": point
        }
        
        return result_dict

