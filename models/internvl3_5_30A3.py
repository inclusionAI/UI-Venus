import os
import re

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, GenerationConfig, LogitsProcessorList, LogitsProcessor

import torch
import torch.nn.functional as F

def compute_lambda(first_prop_clamped: torch.Tensor, k: float) -> torch.Tensor:
    batch_size, N = first_prop_clamped.size()
    exponent = -1 / (k - 1)

    with torch.cuda.amp.autocast(enabled=False):
        first_prop_fp64 = first_prop_clamped.double() 
        S = torch.sum(first_prop_fp64 ** exponent, dim=1)
        numerator = N - 1
        ratio = numerator / S
        lambda_val = k * (ratio ** (k - 1)).unsqueeze(1)

    return lambda_val.to(first_prop_clamped)  


class RolloutLogitsWarper(torch.nn.Module):
    def __init__(self, lamda=0.0000528, k=10):
        super().__init__()
        self.lamda = lamda * 10 / 8
        self.k = k

    def forward(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        first_prop = F.softmax(scores, dim=1)
        first_prop_clamped = torch.clamp(first_prop, min=1e-9)  

        with torch.cuda.amp.autocast(enabled=False):
            first_prop_float = first_prop_clamped.double()
            lambda_val = compute_lambda(first_prop_float, self.k)

            a = 1 / (self.k - 1)
            lambda_k = lambda_val / self.k
            x = lambda_k / first_prop_float
            larger_delta_x = lambda_k - first_prop_float
            epsilon = larger_delta_x / first_prop_float

            approx_2nd = (
                -a * epsilon 
                - (a * (a - 1) / 2) * (epsilon ** 2)
            )

            approx_3rd = approx_2nd - (
                (a * (a - 1) * (a - 2) / 6) * (epsilon ** 3)
            )

            exact = 1 - x.pow(a)

            threshold = 0.0001
            condition_3rd = torch.abs(epsilon) < threshold 

            second_prop = torch.where(
                condition_3rd,
                approx_3rd,
                exact
            )

            second_prop_clamped = torch.clamp(second_prop, min=1e-9)
            summed = second_prop_clamped.sum(dim=1, keepdim=True)

            combined_prop = (second_prop_clamped + first_prop_float) / 2
            scores_processed = torch.log(second_prop_clamped).to(scores)

        return scores_processed



def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def internvl_preprocess_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values



class InternVL3_5_30A3Model():
    def __init__(self):
        pass

    def load_model(self, model_name_or_path="/root/model/InternVL3_5-30B-A3B-v14-20251009-161639-hf", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device).eval()

        self.override_generation_config = dict()
        self.set_generation_config(
            max_new_tokens=50,
            do_sample=False,
            temperature=0.9,
            max_length=None
        )
    
    def set_generation_config(self, **kwargs):
        self.override_generation_config.update(kwargs)
        self.model.generation_config = GenerationConfig(**self.override_generation_config)

    def inference(self, instruction, image_path):
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
        image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."

        pixel_values = internvl_preprocess_image(image, max_num=12).to(torch.bfloat16).cuda()

        grounding_prompt = f"<image>\n{instruction}. If the element does not exist or the task is unrelated to the image, output '<action>\nclick(x=-1, y=-1)\n</action>'."
        response = self.model.chat(self.tokenizer, pixel_values, grounding_prompt, self.override_generation_config)

        bbox = extract_last_bounding_box(response)
        click_point = extract_pyautogui_point(response)

        
        if not click_point and bbox:
            click_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

        result_dict = {
            "result": "positive",
            "bbox": bbox,
            "point": click_point,
            "raw_response": response
        }
        
        return result_dict

    def ground_only_positive_pass_k(self, instruction, image, pass_k):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."

        pixel_values1 = internvl_preprocess_image(image, max_num=12).to(torch.bfloat16).cuda()  # tile size 12
        
        pixel_values = torch.cat([pixel_values1] * pass_k, dim=0)
        mylogits_processor = RolloutLogitsWarper()
        logits_processor = LogitsProcessorList([mylogits_processor])
        grounding_prompt = [f"<image>\n{instruction}"] *pass_k

        
        response_list = self.model.batch_chat(self.tokenizer, pixel_values, grounding_prompt, self.override_generation_config, num_patches_list=[pixel_values1.shape[0]]*pass_k)

        result_dict_list = []
        for response in response_list:
            bbox = extract_last_bounding_box(response)
            click_point = extract_pyautogui_point(response)

            
            if not click_point and bbox:
                click_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

            if click_point is None:
                click_point = [0, 0]

            result_dict = {
                "result": "positive",
                "bbox": bbox,
                "point": click_point,
                "raw_response": response
            }
            result_dict_list.append(result_dict)
        
        return result_dict_list


def extract_last_bounding_box(text):
    # Match bounding box in the format [[x0,y0,x1,y1]]
    pattern = r'\[\[(\d+),(\d+),(\d+),(\d+)\]\]'
    
    # Search for the last match in the text
    matches = re.findall(pattern, text)
    if matches:
        # Get the last match and return as tuple of integers
        match = matches[-1]
        bbox = [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
        return [pos / 1000 for pos in bbox]
    
    return None


def extract_first_point(text):
    # Match point in the format [[x0,y0]]
    pattern = r"\[\[(\d+),(\d+)\]\]"
    
    # Search for the first match in the text with the DOTALL flag to support multi-line text
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        bbox = [int(match.group(1)), int(match.group(2))]
        return [pos / 1000 for pos in bbox]
    
    return None

def extract_pyautogui_point(text):
    match = re.search(r'x=(-?\d+\.\d+),\s*y=(-?\d+\.\d+)', text)

    result = None
    if match:
        x = float(match.group(1))
        y = float(match.group(2))
        result = [x, y]

    return result
  