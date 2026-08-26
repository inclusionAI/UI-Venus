"""
ui_venus2_gd.py

Direct grounding model for Qwen3.5-style API endpoints:
  - Output coordinates directly in [0, 1000] space (no function-call wrapping)
  - Use a simple user-only prompt (no NousFnCallPrompt / ComputerUse)
  - Return [x, y] or [x1, y1, x2, y2] as plain text
  - Thinking mode is DISABLED by default for grounding tasks

Key properties (aligned with qwen35.py):
  1. No system message / function-call wrapping in grounding prompts
  2. Default user_prompt matches qwen35.py (UI-Venus) format
  3. Default norm_type = "0-1000"
  4. Post-processing aligned with qwen35.py's pred_2_point logic
"""

import base64
import io
import json
import math
import os
import re
import time

import numpy as np
from openai import OpenAI
from PIL import Image


# ---------------------------------------------------------------------------
# Vendored smart_resize (from transformers Qwen2-VL, pure-python, no torch)
# ---------------------------------------------------------------------------
def smart_resize(height, width, factor=28, min_pixels=4 * 28 * 28, max_pixels=16384 * 28 * 28):
    """Rescales dimensions so that:
    1. Both dims are divisible by *factor*.
    2. Total pixels is within [min_pixels, max_pixels].
    3. Aspect ratio is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > 200:
        height, width = int(min(height, width)), int(max(height, width))
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.ceil(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height / beta / factor) * factor
        w_bar = math.ceil(width / beta / factor) * factor
    return h_bar, w_bar


# ---------------------------------------------------------------------------
# Qwen35-style coordinate extraction (aligned with qwen35.py pred_2_point)
# ---------------------------------------------------------------------------
def extract_coordinates_qwen35(response):
    """
    Extract [x, y] coordinates from Qwen3.5-style model response.

    Handles:
      - "[x,y]"              → point
      - "[x1,y1,x2,y2]"      → center of bbox
      - "[x1,y1], [x2,y2]"   → center of two-point bbox
      - "[-1,-1]"            → infeasible (returns None)

    Returns:
        (list[float, float], str) or (None, None)
    """
    text = response.strip()

    # Pattern: [x1,y1,x2,y2] (4 numbers in one bracket pair)
    m = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    if m:
        x1, y1, x2, y2 = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
        # If all zeros or [-1,-1,...], treat as infeasible
        if x1 == -1 and y1 == -1:
            return None, "infeasible"
        return [(x1 + x2) / 2, (y1 + y2) / 2], "bbox4_center"

    # Pattern: [x1,y1], [x2,y2] (two bracket pairs)
    m = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*,\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    if m:
        x1, y1, x2, y2 = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
        if x1 == -1 and y1 == -1:
            return None, "infeasible"
        return [(x1 + x2) / 2, (y1 + y2) / 2], "two_point_center"

    # Pattern: [x, y] (2 numbers in one bracket pair)
    m = re.search(r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]', text)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        if x == -1 and y == -1:
            return None, "infeasible"
        return [x, y], "point2d"

    # Fallback: try pred_2_point logic (extract all numbers)
    floats = re.findall(r'-?\d+\.?\d*', text)
    floats = [float(num) for num in floats]
    if len(floats) == 2:
        if floats[0] == -1 and floats[1] == -1:
            return None, "infeasible"
        return floats, "fallback_2nums"
    elif len(floats) >= 4:
        if floats[0] == -1 and floats[1] == -1:
            return None, "infeasible"
        return [(floats[0] + floats[2]) / 2, (floats[1] + floats[3]) / 2], "fallback_4nums"

    return None, None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
def encode_image(image):
    if isinstance(image, (str, os.PathLike)):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Default prompts (aligned with qwen35.py / UI-Venus)
# ---------------------------------------------------------------------------
DEFAULT_USER_PROMPT = (
    "Output the center point of the position corresponding to the following instruction: \n"
    "{instruction}. \n\n"
    "The output should just be the coordinates of a point, in the format [x,y]. "
    "Additionally, if the task is infeasible (e.g., the task is not related to the image), "
    "the output should be [-1,-1]."
)


class Qwen35GroundModel:
    """
    Direct grounding model for Qwen3.5-style API endpoints.

    Key design choices (aligned with qwen35.py):
      * No system message / function-call wrapping — user-only messages
      * Model outputs coordinates in [0, 1000] space
      * Post-processing follows qwen35.py's pred_2_point logic
      * norm_type defaults to "0-1000"
    """

    def __init__(
        self,
        base_url="http://localhost:8400/v1",
        api_key="empty",
        model_name="Qwen/Qwen2.5-VL-72B-Instruct",
        norm_type="0-1000",
        system_prompt=None,
        user_prompt=None,
        debug=False,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.generation_config = {}

        self.norm_type = norm_type
        self.debug = debug

        # Prompts: None → use qwen35 defaults
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt or DEFAULT_USER_PROMPT

    def load_model(self):
        pass

    def set_generation_config(self, **kwargs):
        self.generation_config = kwargs

    # ------------------------------------------------------------------
    # Coordinate normalization helpers
    # ------------------------------------------------------------------
    def _normalize_coords(self, coords, ref_w, ref_h):
        if self.norm_type == "0-1000":
            return [coords[0] / 1000.0, coords[1] / 1000.0]
        else:
            return [coords[0] / ref_w, coords[1] / ref_h]

    # ------------------------------------------------------------------
    # Prompt building helpers
    # ------------------------------------------------------------------
    def _build_system_messages(self):
        """Build system message list if system_prompt is set, else empty list."""
        if self.system_prompt is not None:
            return [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]
        return []

    def _build_user_text(self, instruction):
        """Build user text, replacing {instruction} placeholder."""
        # Strip trailing period from instruction (aligned with qwen35.py)
        inst = instruction
        if inst and inst[-1] == '.':
            inst = inst[:-1]
        return self.user_prompt.replace("{instruction}", inst)

    def _build_messages(self, encoded_string, instruction, image_format="png"):
        """
        Build the full message list for API call.
        Qwen35 style: user-only (no system), unless system_prompt is explicitly set.
        """
        user_text = self._build_user_text(instruction)
        system_msgs = self._build_system_messages()

        user_msg = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "min_pixels": 3136,
                    "max_pixels": 12845056,
                    "image_url": {"url": f"data:image/{image_format};base64,{encoded_string}"},
                },
                {"type": "text", "text": user_text},
            ],
        }

        return system_msgs + [user_msg]

    # ------------------------------------------------------------------
    # Debug helper: print messages and response in readable form
    # ------------------------------------------------------------------
    def _truncate_for_print(self, obj, max_str_len=200):
        """Recursively truncate long strings (e.g. base64 image data) for printing."""
        if isinstance(obj, str):
            if len(obj) > max_str_len:
                return obj[:100] + f"...(truncated, total {len(obj)} chars)..."
            return obj
        if isinstance(obj, list):
            return [self._truncate_for_print(item, max_str_len) for item in obj]
        if isinstance(obj, dict):
            return {k: self._truncate_for_print(v, max_str_len) for k, v in obj.items()}
        return obj

    def _debug_print_messages(self, messages, label=""):
        """Print the messages sent to the model in a readable format."""
        readable = self._truncate_for_print(messages)
        prefix = f"[DEBUG] {label} " if label else "[DEBUG] "
        print(f"\n{'='*60}")
        print(f"{prefix}Sending messages to model ({self.model_name}):")
        print(json.dumps(readable, indent=2, ensure_ascii=False))
        print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Endpoint call
    # ------------------------------------------------------------------
    def _call_endpoint(self, messages, temperature=0, top_p=1.0, return_raw=False, need_logprobs=True):
        max_retries = 3
        timeout = 120
        max_tokens = 1024
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        # Debug: print input messages
        if self.debug:
            self._debug_print_messages(messages, label="Input messages")

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=timeout,
                    max_tokens=max_tokens,
                    logprobs=need_logprobs,
                    top_logprobs=3 if need_logprobs else 0,
                    extra_body=extra_body,
                )
                content = response.choices[0].message.content
                if self.debug:
                    print(f"\n{'-'*60}")
                    print(f"[DEBUG] Model output (temperature={temperature}, top_p={top_p}):")
                    print(content)
                    print(f"{'-'*60}\n")
                return response if return_raw else content
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Error calling API after {max_retries} attempts: {e}")
                    return "Error: Unable to get a response from the model after multiple attempts."

    def get_token_probs_between_strings(self, response, start_str, end_str):
        """Find all tokens (and their probabilities) between two marker strings."""
        if not response.choices[0].logprobs:
            return [], "Logprobs not enabled in response.", ''

        logprobs_list = response.choices[0].logprobs.content

        full_text = ""
        token_map = []

        current_idx = 0
        for item in logprobs_list:
            token_str = item.token
            start = current_idx
            end = current_idx + len(token_str)
            token_map.append({"start": start, "end": end, "data": item})
            full_text += token_str
            current_idx = end

        start_str_loc = full_text.find(start_str)
        if start_str_loc == -1:
            return [], f"Start string not found: '{start_str}'", ''
        start_boundary_idx = start_str_loc + len(start_str)

        end_str_loc = full_text.find(end_str, start_boundary_idx)
        if end_str_loc == -1:
            return [], f"End string not found: '{end_str}'", ''
        end_boundary_idx = end_str_loc

        if start_boundary_idx >= end_boundary_idx:
            return [], "No tokens between markers.", ''

        tokens_in_between = []
        tokens_in_between_content = ''
        for t_map in token_map:
            token_start = t_map['start']
            token_data = t_map['data']
            if start_boundary_idx <= token_start < end_boundary_idx:
                prob_percent = math.exp(token_data.logprob) * 100
                tokens_in_between_content += token_data.token
                tokens_in_between.append({
                    "token": token_data.token,
                    "logprob": token_data.logprob,
                    "probability": prob_percent,
                    "top_logprobs": token_data.top_logprobs[:3],
                })
            if token_start >= end_boundary_idx:
                break

        return tokens_in_between, None, tokens_in_between_content

    def calculate_perplexity(self, token_data_list):
        if not token_data_list:
            print("Warning: empty token list, cannot compute perplexity.")
            return None

        logprobs = [data['logprob'] for data in token_data_list]
        sum_logprobs = np.sum(logprobs)
        N = len(logprobs)
        ANLL = -sum_logprobs / N
        perplexity = math.exp(ANLL)
        return perplexity

    # ------------------------------------------------------------------
    # Direct grounding (single call)
    # ------------------------------------------------------------------
    def ground(self, instruction, image, need_logprobs=True):
        if isinstance(image, str):
            assert os.path.exists(image) and os.path.isfile(image), "Invalid input image path."
            with Image.open(image) as source_image:
                image_format = {
                    "PNG": "png",
                    "JPEG": "jpeg",
                    "WEBP": "webp",
                    "GIF": "gif",
                }.get(source_image.format)
                input_image = source_image.copy()
            if image_format:
                encoded_string = encode_image(image)
            else:
                image_format = "png"
                encoded_string = encode_image(input_image)
        else:
            assert isinstance(image, Image.Image)
            input_image = image
            image_format = "png"
            encoded_string = encode_image(input_image)

        resized_height, resized_width = smart_resize(
            input_image.height, input_image.width,
            min_pixels=3136, max_pixels=12845056,
        )
        display_image = input_image.resize((resized_width, resized_height))

        messages = self._build_messages(encoded_string, instruction, image_format=image_format)

        response_ori = self._call_endpoint(messages, return_raw=True, need_logprobs=need_logprobs)

        # Handle API failure: _call_endpoint returns a str error message
        # instead of a response object when all retries are exhausted.
        if isinstance(response_ori, str):
            print(f"[ground] API call failed, returning error result: {response_ori[:100]}")
            return {
                "result": "wrong_format",
                "api_error": True,
                "format": "x1y1x2y2",
                "raw_response": response_ori,
                "bbox": None,
                "point": None,
            }, display_image, None
        # print(response_ori)
        response = response_ori.choices[0].message.content

        if response is None:
            response = ""

        try:
            coordinates, method = extract_coordinates_qwen35(response)
            if coordinates is None:
                if method == "infeasible":
                    # [-1,-1] → model says task infeasible
                    result_dict = {
                        "result": "negative",
                        "format": "x1y1x2y2",
                        "raw_response": response,
                        "bbox": None,
                        "point": [-0.001, -0.001],  # normalized [-1,-1]
                    }
                    return result_dict, display_image, None
                raise ValueError("No coordinates found in response")

            ppl_value = ppl_value_x = ppl_value_y = None
            results, error, tokens_in_between_content = self.get_token_probs_between_strings(response_ori, '[', ']')
            results_x, _, _ = self.get_token_probs_between_strings(response_ori, '[', ',')
            results_y, _, _ = self.get_token_probs_between_strings(response_ori, ',', ']')
            if not error and results:
                ppl_value = self.calculate_perplexity(results)
                ppl_value_x = self.calculate_perplexity(results_x)
                ppl_value_y = self.calculate_perplexity(results_y)

            result_dict = {
                "result": "positive",
                "format": "x1y1x2y2",
                "raw_response": response,
                "bbox": None,
                "perplexity": ppl_value,
                "perplexity_x": ppl_value_x,
                "perplexity_y": ppl_value_y,
                "perplexity_content": tokens_in_between_content,
                "point": self._normalize_coords(coordinates, resized_width, resized_height),
                "parse_method": method,
            }
        except Exception as e:
            print(f"ground() failed to parse response: {e}")
            result_dict = {
                "result": "wrong_format",
                "format": "x1y1x2y2",
                "raw_response": response,
                "bbox": None,
                "point": None
            }

        return result_dict, display_image, None
