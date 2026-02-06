# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SeeAct agent for Android."""
from android_world.agents import base_agent
from android_world.agents import seeact_utils
from android_world.agents import maiui_convert2jsonaction as mobile_agent_utils
from android_world.env import actuation_qwen3
from android_world.env import interface
from android_world.env import adb_utils
from android_world.agents.maiui_prompt import MAI_MOBILE_SYS_PROMPT, MAI_MOBILE_SYS_PROMPT_ASK_USER_MCP
from android_world.agents.unified_memory import TrajStep, TrajMemory
from android_world.agents.maiui_utils import pil_to_base64, safe_pil_to_bytes

from android_world.env import tools
from android_world.agents import new_json_action as json_action
from PIL import Image
import numpy as np
import base64
import json
import pprint
import os
import time
from qwen_vl_utils import smart_resize
from io import BytesIO
import re

from android_world.agents.coordinate_resize import update_image_size_
import traceback

def extract_tag_content(tag_name, data):
    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, data, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG") 
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def image_to_base64(image_path):
  dummy_image = Image.open(image_path)
  MIN_PIXELS=3136
  MAX_PIXELS=10035200
  resized_height, resized_width  = smart_resize(dummy_image.height,
      dummy_image.width,
      factor=28,
      min_pixels=MIN_PIXELS,
      max_pixels=MAX_PIXELS,)
  dummy_image = dummy_image.resize((resized_width, resized_height))
  return f"data:image/png;base64,{pil_to_base64(dummy_image)}"


def fetch_resized_image(screenshot_file):
    screenshot = Image.open(screenshot_file)
    width, height = screenshot.size
    MIN_PIXELS=3136
    MAX_PIXELS=10035200
    current_image_ele = {'width': width, 'height': height, }
    resized_height = height
    resized_width = width
    # resized_height, resized_width  = smart_resize(height,
    #     width,
    #     factor=32,
    #     min_pixels=MIN_PIXELS,
    #     max_pixels=MAX_PIXELS,)
    current_image_ele['resized_width'] = resized_width
    current_image_ele['resized_height'] = resized_height
    return screenshot, resized_width, resized_height, current_image_ele

def rescale_coordinates(point, width, height):
    point = (round(point[0]/999*width), round(point[1]/999*height))
    return point




# MAI-UI 

from typing import Any, Dict, List, Optional, Tuple
import copy

from openai import OpenAI
# Constants
SCALE_FACTOR = 999


def mask_image_urls_for_logging(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create a copy of messages with image URLs masked for logging.

    Args:
        messages: List of message dictionaries that may contain image URLs.

    Returns:
        Deep copy of messages with image URLs replaced by "[IMAGE_DATA]".
    """
    messages_masked = copy.deepcopy(messages)
    for message in messages_masked:
        content = message.get("content", [])
        if content and isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "image_url" in item:
                    item["image_url"]["url"] = "[IMAGE_DATA]"
    return messages_masked


def parse_tagged_text(text: str) -> Dict[str, Any]:
    """
    Parse text containing XML-style tags to extract thinking and tool_call content.

    Args:
        text: Text containing <thinking> and <tool_call> tags.

    Returns:
        Dictionary with keys:
            - "thinking": Content inside <thinking> tags (str or None)
            - "tool_call": Parsed JSON content inside <tool_call> tags (dict or None)

    Raises:
        ValueError: If tool_call content is not valid JSON.
    """
    # Handle thinking model output format (uses </think> instead of </thinking>)
    if "</think>" in text and "</thinking>" not in text:
        text = text.replace("</think>", "</thinking>")
        text = "<thinking>" + text

    # Define regex pattern with non-greedy matching
    pattern = r"<thinking>(.*?)</thinking>.*?<tool_call>(.*?)</tool_call>"

    result: Dict[str, Any] = {
        "thinking": None,
        "tool_call": None,
    }

    # Use re.DOTALL to match newlines
    match = re.search(pattern, text, re.DOTALL)
    if match:
        result = {
            "thinking": match.group(1).strip().strip('"'),
            "tool_call": match.group(2).strip().strip('"'),
        }

    result['dummy_tool_call'] = result["tool_call"]

    # Parse tool_call as JSON
    if result["tool_call"]:
        try:
            result["tool_call"] = json.loads(result["tool_call"])
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in tool_call: {e}")

    return result


def parse_action_to_structure_output(text: str) -> Dict[str, Any]:
    """
    Parse model output text into structured action format.

    Args:
        text: Raw model output containing thinking and tool_call tags.

    Returns:
        Dictionary with keys:
            - "thinking": The model's reasoning process
            - "action_json": Parsed action with normalized coordinates

    Note:
        Coordinates are normalized to [0, 1] range by dividing by SCALE_FACTOR.
    """
    text = text.strip()

    results = parse_tagged_text(text)
    thinking = results["thinking"]
    # json
    tool_call = results["tool_call"]
    # str
    dummy_tool_call = results["dummy_tool_call"]

    action = tool_call["arguments"]

    # Normalize coordinates from SCALE_FACTOR range to [0, 1]
    if "coordinate" in action:
        coordinates = action["coordinate"]
        if len(coordinates) == 2:
            point_x, point_y = coordinates
        elif len(coordinates) == 4:
            x1, y1, x2, y2 = coordinates
            point_x = (x1 + x2) / 2
            point_y = (y1 + y2) / 2
        else:
            raise ValueError(
                f"Invalid coordinate format: expected 2 or 4 values, got {len(coordinates)}"
            )
        point_x = point_x / SCALE_FACTOR
        point_y = point_y / SCALE_FACTOR
        action["coordinate"] = [point_x, point_y]
    
    if "start_coordinate" in action:
        coordinates = action["start_coordinate"]
        if len(coordinates) == 2:
            point_x, point_y = coordinates
        elif len(coordinates) == 4:
            x1, y1, x2, y2 = coordinates
            point_x = (x1 + x2) / 2
            point_y = (y1 + y2) / 2
        else:
            raise ValueError(
                f"Invalid coordinate format: expected 2 or 4 values, got {len(coordinates)}"
            )
        point_x = point_x / SCALE_FACTOR
        point_y = point_y / SCALE_FACTOR
        action["start_coordinate"] = [point_x, point_y]
    
    if "end_coordinate" in action:
        coordinates = action["end_coordinate"]
        if len(coordinates) == 2:
            point_x, point_y = coordinates
        elif len(coordinates) == 4:
            x1, y1, x2, y2 = coordinates
            point_x = (x1 + x2) / 2
            point_y = (y1 + y2) / 2
        else:
            raise ValueError(
                f"Invalid coordinate format: expected 2 or 4 values, got {len(coordinates)}"
            )
        point_x = point_x / SCALE_FACTOR
        point_y = point_y / SCALE_FACTOR
        action["end_coordinate"] = [point_x, point_y]

    return {
        "thinking": thinking,
        "action_json": action,
        'dummy_tool_call': dummy_tool_call
    }



class MAIUI(base_agent.EnvironmentInteractingAgent):
  """mobile agent for Android."""

  def __init__(self, env: interface.AsyncEnv, 
                 llm_base_url, src_format, name: str = "model", output_path = ""):
        super().__init__(env, name)
        self.src_format = src_format
        self._actions = []
        self._screenshots = []
        self._summarys = []
        self._thoughts = []
        self.output_result = {}
        self.output_path = output_path
        if self.output_path and not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        self.add_thought = True
        self._text_actions = []

        # self.url = llm_base_url

        self.output_list = []
        self._response = []
        self.task_name = {}

        # 核心配置：复现参考代码中的 history_n 逻辑

        default_conf = {
            "history_n": 3,
            "temperature": 0.0,
            "top_k": -1,
            "top_p": 1.0,
            "max_tokens": 2048,
        }
        self.top_k = -1
        self.max_tokens = 2048
        self.top_p=1.0
        self.temperature = 0
        self.history_n = 3  # 保留最近3步的上下文
        self.message_history = []  # 存储结构化对话
        
        self.step_counter = 0
        self.traj_memory = TrajMemory(
            task_goal="",
            task_id="",
            steps=[],
        )

        self.llm_base_url = llm_base_url
        self.model_name = name

        self.llm = OpenAI(
            base_url=self.llm_base_url,
            api_key="empty",
        )
  def reset(self, go_home: bool = False) -> None:
        super().reset(go_home)
        self.traj_memory = TrajMemory(
            task_goal="",
            task_id="",
            steps=[],
        )
        self.message_history = []
        self.step_counter = 0
        print("Agent state reset.")

        self.env.hide_automation_ui()
        self._actions.clear()
        self._text_actions.clear()
        self._screenshots.clear()  # TODO
        self._summarys.clear()
        self._thoughts.clear()
        self._response.clear()
  
  def initialize_chrome(self):
        print("Running additional chrome initialization...")
        # handle chrome initialization problem for browser tasks
        adb_utils.launch_app("chrome", self.env.controller)
        time.sleep(5)

        tool_controller = tools.AndroidToolController(env=self.env.controller)
        time.sleep(2)

        first_op = False
        try:
            print("try first variant...")
            tool_controller.click_element("Use without an account")
            time.sleep(5.0)
            first_op = True
        except:
            print("Failed to click 'Use without an account' button.")
            pass
        
        if not first_op:
            print("try second variant...")
            try:
                tool_controller.click_element("Accept & continue")
            except:
                pass
            time.sleep(3.0)
            try:
                tool_controller.click_element("No thanks")
            except:
                pass
            time.sleep(5.0)
            
        adb_utils.press_home_button(self.env.controller)
        time.sleep(2.0)
        print("Done additional chrome initialization")
    
  def get_task_name(self, suite):
        for name, instances in suite.items():
            self.task_name[instances[0].goal] = name

  @property
  def observations(self) -> List[Dict[str, Any]]:
        """Return list of observations from trajectory memory."""
        return [
            {
                "screenshot": step.screenshot_bytes,
                "accessibility_tree": step.accessibility_tree,
            }
            for step in self.traj_memory.steps
        ]

  @property
  def history_images(self) -> List[bytes]:
        """Return list of screenshot bytes from trajectory memory."""
        return [step.screenshot_bytes for step in self.traj_memory.steps]



  def system_prompt(self):
        return MAI_MOBILE_SYS_PROMPT
  
  @property
  def history_responses(self) -> List[str]:
        """
        Generate formatted history responses for context.

        Returns:
            List of formatted response strings with thinking and tool_call tags.
        """
        history_responses = []

        for step in self.traj_memory.steps:
            thinking = step.thought
            structured_action = step.structured_action

            if not structured_action:
                continue

            action_json = copy.deepcopy(structured_action.get("action_json", {}))

            # Convert normalized coordinates back to SCALE_FACTOR range
            if "coordinate" in action_json:
                coordinates = action_json.get("coordinate", [])
                if len(coordinates) == 2:
                    point_x, point_y = coordinates
                elif len(coordinates) == 4:
                    x1, y1, x2, y2 = coordinates
                    point_x = (x1 + x2) / 2
                    point_y = (y1 + y2) / 2
                else:
                    continue
                action_json["coordinate"] = [
                    int(point_x * SCALE_FACTOR),
                    int(point_y * SCALE_FACTOR),
                ]

            tool_call_dict = {
                "name": "mobile_use",
                "arguments": action_json,
            }
            tool_call_json = json.dumps(tool_call_dict, separators=(",", ":"))
            history_responses.append(
                f"<thinking>\n{thinking}\n</thinking>\n<tool_call>\n{tool_call_json}\n</tool_call>"
            )

        return history_responses

  def mem2response(self, step: TrajStep) -> str:
        thinking = step.thought
        structured_action = step.structured_action

        if not structured_action:
            raise ValueError("No structured action found")

        action_json = copy.deepcopy(structured_action.get("action_json", {}))

        # Convert normalized coordinates back to SCALE_FACTOR range
        if "coordinate" in action_json:
            coordinates = action_json.get("coordinate", [])
            if len(coordinates) == 2:
                point_x, point_y = coordinates
            elif len(coordinates) == 4:
                x1, y1, x2, y2 = coordinates
                point_x = (x1 + x2) / 2
                point_y = (y1 + y2) / 2
            else:
                raise ValueError(f"Invalid coordinate format: expected 2 or 4 values, got {len(coordinates)}")
            action_json["coordinate"] = [
                int(point_x * SCALE_FACTOR),
                int(point_y * SCALE_FACTOR),
            ]

        tool_call_dict = {
            "name": "mobile_use",
            "arguments": action_json,
        }
        tool_call_json = json.dumps(tool_call_dict, separators=(",", ":"))
        return f"<thinking>\n{thinking}\n</thinking>\n<tool_call>\n{tool_call_json}\n</tool_call>"

  def mem2ask_user_response(self, step: TrajStep) -> str:
        return step.ask_user_response

  def mem2mcp_response(self, step: TrajStep) -> str:
        return step.mcp_response

  def _prepare_images(self, screenshot_bytes: bytes) -> List[Image.Image]:
        """
        Prepare image list including history and current screenshot.

        Args:
            screenshot_bytes: Current screenshot as bytes.

        Returns:
            List of PIL Images (history + current).
        """
        # Calculate how many history images to include
        if len(self.history_images) > 0:
            max_history = min(len(self.history_images), self.history_n - 1)
            recent_history = self.history_images[-max_history:] if max_history > 0 else []
        else:
            recent_history = []

        # Add current image bytes
        recent_history.append(screenshot_bytes)

        # Normalize input type
        if isinstance(recent_history, bytes):
            recent_history = [recent_history]
        elif isinstance(recent_history, np.ndarray):
            recent_history = list(recent_history)
        elif not isinstance(recent_history, list):
            raise TypeError(f"Unidentified images type: {type(recent_history)}")

        # Convert all images to PIL format
        images = []
        for image in recent_history:
            if isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, Image.Image):
                pass
            else:
                raise TypeError(f"Expected bytes or PIL Image, got {type(image)}")

            if image.mode != "RGB":
                image = image.convert("RGB")

            images.append(image)

        return images

  def _build_messages(
        self,
        instruction: str,
        images: List[Image.Image],
    ) -> List[Dict[str, Any]]:
        """
        Build the message list for the LLM API call.

        Args:
            instruction: Task instruction from user.
            images: List of prepared images.
        Returns:
            List of message dictionaries for the API.
        """
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt()}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": instruction}],
            },
        ]

        image_num = 0
        # history_responses = self.history_responses

        if len(self.traj_memory.steps) > 0:
            # Only the last (history_n - 1) history responses need images,
            start_image_idx = max(0, len(self.traj_memory.steps) - (self.history_n - 1))
            
            for history_idx, step in enumerate(self.traj_memory.steps):
                # Only include images for the last (history_n - 1) history responses
                should_include_image = (history_idx >= start_image_idx)
                
                if should_include_image:
                    # Add image before the assistant response
                    if image_num < len(images) - 1:
                        cur_image = images[image_num]
                        encoded_string = pil_to_base64(cur_image)
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{encoded_string}"},
                            }],
                        })
                    image_num += 1
                
                # Always add the assistant response (regardless of whether an image is included)
                history_response = self.mem2response(step)
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": history_response}],
                })

                # Add ask_user_response or mcp_response if present
                ask_user_response = self.mem2ask_user_response(step)
                if ask_user_response:
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": ask_user_response}],
                    })
                mcp_response = self.mem2mcp_response(step)
                if mcp_response:
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": mcp_response}],
                    })

            # Add current image (last one in images list)
            if image_num < len(images):
                cur_image = images[image_num]
                encoded_string = pil_to_base64(cur_image)
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_string}"},
                    }],
                })
        else:
            # No history, just add the current image
            cur_image = images[0]
            encoded_string = pil_to_base64(cur_image)
            messages.append({
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_string}"},
                }],
            })

        return messages


  def predict(
            self,
            instruction: str,
            obs: Dict[str, Any],
            **kwargs: Any,
        ) -> Tuple[str, Dict[str, Any], str]:
        """
        Predict the next action based on the current observation.

        Args:
            instruction: Task instruction/goal.
            obs: Current observation containing:
                - screenshot: PIL Image or bytes of current screen
                - ask_user_response: Optional response from asking user
                - mcp_response: Optional response from MCP tools
        Returns:
            Tuple of (prediction_text, action_dict) where:
                - prediction_text: Raw model response or error message
                - action_dict: Parsed action dictionary
        """
        # Set task goal if not already set
        if not self.traj_memory.task_goal:
            self.traj_memory.task_goal = instruction

        # Process screenshot
        screenshot_pil = obs["screenshot"]
        screenshot_bytes = safe_pil_to_bytes(screenshot_pil)

        # Prepare images
        images = self._prepare_images(screenshot_bytes)

        # Build messages
        messages = self._build_messages(instruction, images)

        # print('-------------------- input messages: ----------------------\n', messages)
        # Make API call with retry logic
        max_retries = 3
        prediction = None
        action_json = None

        for attempt in range(max_retries):
            try:
                # messages_print = mask_image_urls_for_logging(messages)
                # print(f"Messages (attempt {attempt + 1}):\n{messages_print}")

                response = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    extra_body={"repetition_penalty": 1.0, "top_k": self.top_k},
                    seed=42,
                )
                prediction = response.choices[0].message.content.strip()
                print(f"Raw response:\n{prediction}")

                # Parse response
                parsed_response = parse_action_to_structure_output(prediction)
                thinking = parsed_response["thinking"]
                action_json = parsed_response["action_json"]
                dummy_tool_call = parsed_response["dummy_tool_call"]

                print(f"Parsed response:\n{parsed_response}")
                break

            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                traceback.print_exc()
                prediction = None
                action_json = None

        # Return error if all retries failed
        if prediction is None or action_json is None:
            print("Max retry attempts reached, returning error flag.")
            return "llm client error", {"action": None}, ''

        # Create and store trajectory step
        traj_step = TrajStep(
            screenshot=screenshot_pil,
            accessibility_tree=obs.get("accessibility_tree"),
            prediction=prediction,
            action=action_json,
            conclusion="",
            thought=thinking,
            step_index=len(self.traj_memory.steps),
            agent_type="MAIMobileAgent",
            model_name=self.model_name,
            screenshot_bytes=screenshot_bytes,
            structured_action={"action_json": action_json},
        )
        self.traj_memory.steps.append(traj_step)

        return prediction, action_json, dummy_tool_call


  def step(
      self, goal: str, step_numb: bool = False) -> base_agent.AgentInteractionResult:
    result = {
        "ui_elements": None,
        "screenshot": None,
        "actionable_elements": None,
        "action_gen_payload": None,
        "action_gen_response": None,
        "action_ground_payload": None,
        "action_ground_response": None,
        "seeact_action": None,
        "action": None,
        "action_description": None,
    }
    step_idx = len(self._screenshots)
    state = self.get_post_transition_state()
    result["ui_elements"] = state.ui_elements
    result["screenshot"] = state.pixels
    screenshot = Image.fromarray(state.pixels)
    screenshot_file = f"screenshot_{step_idx}.png"
    
    if self.output_path:
      # Saving screenshot file
      if goal not in self.task_name:
        task_output_dir = os.path.join(self.output_path, goal.replace(" ", "_")[:50])
      else:
        task_output_dir = os.path.join(self.output_path, self.task_name[goal])
      screenshot_file = os.path.join(task_output_dir, f"screenshot_{step_idx}.png")
      if not os.path.exists(task_output_dir):
        os.makedirs(task_output_dir)
      screenshot.save(screenshot_file)
      
      # Saving action traj json
      with open(os.path.join(task_output_dir, "action.jsonl"), 'w', encoding='utf-8') as f:
        for item in self._actions:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    self._screenshots.append(screenshot)
  
    stage2_history = ''
    for idx, his in enumerate(self._summarys):
        if his is not None:
            stage2_history += 'Step ' + str(idx + 1) + ': ' + str(his.replace('\n', '').replace('"', '')) + '; '
    stage2_user_prompt = goal

    screenshot, resized_width, resized_height, current_image_ele = fetch_resized_image(screenshot_file)
    # action_response = ''
    action = None
    
    # system_prompt_part, user_prompt_part = build_system_messages(stage2_user_prompt, stage2_history)

    # print(system_prompt_part)
    # user_prompt_part['content'].append({'image': screenshot_file})
    
    
    # messages = [system_prompt_part, user_prompt_part]

    # import pdb; pdb.set_trace()
    # action_response, _, _ = self.vllm.predict_mm(
    #       "",
    #       [],
    #       messages=messages
    #   )
    
    raw_response, action_json, dummy_tool_call = self.predict(instruction=goal, obs = {'screenshot':screenshot})

    action_response = raw_response
    result["action_response"] = raw_response
    print('========== action_response ==========')
    pprint.pprint(action_response)

    dummy_action = None
    thought = None
    summary = None
    try:
      print(f"dummy_tool_call: {dummy_tool_call}")
      dummy_action = json.loads(dummy_tool_call)
    #   dummy_action = dummy_tool_call

      dummy_action['arguments']['action'] = dummy_action['arguments']['action'].replace('tap', 'click')
    #   Handling answer action
      if len(self._actions) > 0 and self._actions[-1]['arguments']['action'] == 'answer':
          dummy_action = {"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}}
          self.env.interaction_cache =  self._actions[-1]['arguments']['text']

    

      if dummy_action['arguments'].get('action') == 'swipe':
        arguments = dummy_action['arguments']
        
        direction_str = arguments.get('direction', '')
        has_start = 'coordinate' in arguments
        has_end = 'coordinate2' in arguments
        
        d_w, d_h = int(0.25 * resized_width), int(0.25 * resized_height)
        
        if has_start and not has_end and direction_str:
            start_x, start_y = arguments['coordinate']
            
            if direction_str == 'up':
                end_x = start_x
                end_y = start_y - d_h
            elif direction_str == 'down':
                end_x = start_x
                end_y = start_y + d_h
            elif direction_str == 'left':
                end_x = start_x - d_w
                end_y = start_y
            elif direction_str == 'right':
                end_x = start_x + d_w
                end_y = start_y

            else:
                raise ValueError(f"Unknown swipe direction: {direction_str}")
            
            end_x = max(0, min(end_x, resized_width))
            end_y = max(0, min(end_y, resized_height))
            
            arguments['coordinate2'] = (end_x, end_y)
        
        elif not has_start and not has_end and direction_str:
            start_x = resized_width // 2
            start_y = resized_height // 2
            
            if direction_str == 'up':
                end_x = start_x
                end_y = start_y - d_h
            elif direction_str == 'down':
                end_x = start_x
                end_y = start_y + d_h
            elif direction_str == 'left':
                end_x = start_x - d_w
                end_y = start_y
            elif direction_str == 'right':
                end_x = start_x + d_w
                end_y = start_y
            else:
                raise ValueError(f"Unknown swipe direction: {direction_str}")
            
            start_x = max(0, min(start_x, resized_width))
            start_y = max(0, min(start_y, resized_height))
            end_x = max(0, min(end_x, resized_width))
            end_y = max(0, min(end_y, resized_height))
            
            arguments['coordinate'] = (start_x, start_y)
            arguments['coordinate2'] = (end_x, end_y)
        
      if dummy_action['arguments'].get('action') == 'drag':
        arguments = dummy_action['arguments']
        if 'start_coordinate' in arguments:
            arguments['coordinate'] = arguments.pop('start_coordinate')
        if 'end_coordinate' in arguments:
            arguments['coordinate2'] = arguments.pop('end_coordinate')



      for key in dummy_action['arguments']:
        if key in ['coordinate', 'coordinate2']:
          x, y = dummy_action['arguments'][key]
          new_point = rescale_coordinates((x, y), resized_width, resized_height)
          dummy_action['arguments'][key] = new_point





      print(f"Rescaled coordinates (abs_resized) dummy_action: {dummy_action}")
      action, dummy_action_translated = mobile_agent_utils.convert_mobile_agent_action_to_json_action(
          dummy_action, current_image_ele, src_format=self.src_format, tgt_format='abs_origin'
      )

      result["dummy_action"] = dummy_action
      result["dummy_action_translated"] = dummy_action_translated
      result["action"] = action
        

    except seeact_utils.ParseActionError as e:
      action = json_action.JSONAction(action_type=json_action.UNKNOWN)
      result["seeact_action"] = None
      result["action"] = action
    except:
        traceback.print_exc()
        print(action_response)
        raise
    else:
      if step_numb == False:
        actuation_qwen3.execute_adb_action(
            action,
            [],
            self.env.logical_screen_size,
            self.env.controller
        )
      
      self._text_actions.append(summary)
      self._actions.append(dummy_action)
      self._summarys.append(summary)
      self._thoughts.append(thought)
      self._response.append(action_response)

    if self.output_path:
      if goal not in self.task_name:
        task_output_dir = os.path.join(self.output_path, goal.replace(" ", "_")[:50])
      else:
        task_output_dir = os.path.join(self.output_path, self.task_name[goal])
      if not os.path.exists(task_output_dir):
        os.makedirs(task_output_dir)
      screenshot.save(screenshot_file)
      with open(os.path.join(task_output_dir, "action.jsonl"), 'w', encoding='utf-8') as f:
        for item in self._actions:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    
    return base_agent.AgentInteractionResult(
        done=action.action_type == json_action.STATUS,
        data=result,
    )
