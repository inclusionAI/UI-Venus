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
from android_world.agents import mobile_agent_utils_new as mobile_agent_utils
from android_world.env import actuation_qwen3
from android_world.env import interface
from android_world.env import adb_utils
from android_world.env import tools
from android_world.agents import new_json_action as json_action
from PIL import Image
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
system_prompt = "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"mobile_use\", \"description\": \"Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen's resolution is 999x999.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\", \"parameters\": {\"properties\": {\"action\": {\"description\": \"The action to perform. The available actions are:\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `answer`: Output the answer.\\n* `system_button`: Press the system button.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.\", \"enum\": [\"click\", \"long_press\", \"swipe\", \"type\", \"answer\", \"system_button\", \"wait\", \"terminate\"], \"type\": \"string\"}, \"coordinate\": {\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.\", \"type\": \"array\"}, \"coordinate2\": {\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.\", \"type\": \"array\"}, \"text\": {\"description\": \"Required only by `action=type` and `action=answer`.\", \"type\": \"string\"}, \"time\": {\"description\": \"The seconds to wait. Required only by `action=long_press` and `action=wait`.\", \"type\": \"number\"}, \"button\": {\"description\": \"Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`\", \"enum\": [\"Back\", \"Home\", \"Menu\", \"Enter\"], \"type\": \"string\"}, \"status\": {\"description\": \"The status of the task. Required only by `action=terminate`.\", \"type\": \"string\", \"enum\": [\"success\", \"failure\"]}}, \"required\": [\"action\"], \"type\": \"object\"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n\n# Response format\n\nResponse format for every step:\n1) Thought: one concise sentence explaining the next move (no multi-step reasoning).\n2) Action: a short imperative describing what to do in the UI.\n3) A single <tool_call>...</tool_call> block containing only the JSON: {\"name\": <function-name>, \"arguments\": <args-json-object>}.\n\nRules:\n- Output exactly in the order: Thought, Action, <tool_call>.\n- Be brief: one sentence for Thought, one for Action.\n- Do not output anything else outside those three parts.\n- If finishing, use action=terminate in the tool call."

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


all_apps_str = """- simple calendar pro: A calendar app.\n  - settings: The Android system settings app for managing device settings such as Bluetooth, Wi-Fi, and brightness.\n  - markor: A note-taking app for creating, editing, deleting, and managing notes and folders.\n  - broccoli: A recipe management app.\n  - pro expense: An expense tracking app.\n  - simple sms messenger: An SMS app for sending, replying to, and resending text messages.\n  - opentracks: A sport tracking app for recording and analyzing activities.\n  - tasks: A task management app for tracking tasks, due dates, and priorities.\n  - clock: An app with stopwatch and timer functionality.\n  - joplin: A note-taking app.\n  - retro music: A music player app.\n  - simple gallery pro: An app for viewing images.\n  - camera: An app for taking photos and videos.\n  - chrome: A web browser app.\n  - contacts: An app for managing contact information.\n  - osmand: A maps and navigation app with support for adding location markers, favorites, and saving tracks.\n  - vlc: A media player app for playing media files.\n  - audio recorder: An app for recording and saving audio clips.\n  - files: A file manager app for the Android filesystem, used for deleting and moving files.\n  - simple draw pro: A drawing app for creating and saving drawings."""

DETAILED_TIPS = (
    'General:\n'
    '- If a previous action fails and the screen does not change, simply try again first.\n'
    '- For any pop-up window, such as a permission request, you need to close it (e.g., by clicking `Don\'t Allow` or `Accept & continue`) before proceeding. Never choose to add any account or log in.`\n'
    '- For requests that are questions (or chat messages), remember to use'
    ' the `answer` action to reply to user explicitly before finish!\n'
    '- If the desired state is already achieved (e.g., enabling Wi-Fi when'
    " it's already on), you can just complete the task.\n\n"
    'Action Related:\n'
    '- Use the `open_app` action whenever you want to open an app'
    ' (nothing will happen if the app is not installed), do not use the'
    ' app drawer to open an app unless all other ways have failed.'
    ' ALL avaliable apps are listed as follows, please use the exact names (in lowercase) as argument for the `open_app` action.\n'
    f'{all_apps_str}'
    '- Use the `type` action whenever you want to type'
    ' something (including password) instead of clicking characters on the'
    ' keyboard one by one. Sometimes there is some default text in the text'
    ' field you want to type in, remember to delete them before typing.\n'
    '- For `click`, `long_press` and `type`, the index parameter you'
    ' pick must be VISIBLE in the screenshot\n'
    '- Consider exploring the screen by using the `swipe`'
    ' action with different directions to reveal additional content. Or use search to quickly find a specific entry, if applicable.\n\n'
    'Text Related Operations:\n'
    '- When asked to save a file with a specific name, you can usually edit the name in the final step. For example, you can first record an audio clip then save it with a specific name.\n'
    '- Normally to select certain text on the screen: <i> Enter text selection'
    ' mode by long pressing the area where the text is, then some of the words'
    ' near the long press point will be selected (highlighted with two pointers'
    ' indicating the range) and usually a text selection bar will also appear'
    ' with options like `copy`, `paste`, `select all`, etc.'
    ' <ii> Select the exact text you need. Usually the text selected from the'
    ' previous step is NOT the one you want, you need to adjust the'
    ' range by dragging the two pointers. If you want to select all text in'
    ' the text field, simply click the `select all` button in the bar.\n'
    "- At this point, you don't have the ability to drag something around the"
    ' screen, so in general you can not select arbitrary text.\n'
    '- To delete some text: the most traditional way is to place the cursor'
    ' at the right place and use the backspace button in the keyboard to'
    ' delete the characters one by one (can long press the backspace to'
    ' accelerate if there are many to delete). Another approach is to first'
    ' select the text you want to delete, then click the backspace button'
    ' in the keyboard.\n'
    '- To copy some text: first select the exact text you want to copy, which'
    ' usually also brings up the text selection bar, then click the `copy`'
    ' button in bar.\n'
    '- To paste text into a text box, first long press the'
    ' text box, then usually the text selection bar will appear with a'
    ' `paste` button in it.\n'
    '- When typing into a text field, sometimes an auto-complete dropdown'
    ' list will appear. This usually indicating this is a enum field and you'
    ' should try to select the best match by clicking the corresponding one'
    ' in the list.\n\n'
)

def fetch_resized_image(screenshot_file):
    screenshot = Image.open(screenshot_file)
    width, height = screenshot.size
    MIN_PIXELS=3136
    MAX_PIXELS=10035200
    current_image_ele = {'width': width, 'height': height, }
    resized_height, resized_width  = smart_resize(height,
        width,
        factor=32,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,)
    current_image_ele['resized_width'] = resized_width
    current_image_ele['resized_height'] = resized_height
    return screenshot, resized_width, resized_height, current_image_ele
def rescale_coordinates(point, width, height):
    point = (round(point[0]/999*width), round(point[1]/999*height))
    return point

class Qwen3_VL(base_agent.EnvironmentInteractingAgent):
  """mobile agent for Android."""

  def __init__(self, env: interface.AsyncEnv, 
               vllm, src_format, api_key, 
               url, name: str = "Mobile_Agent", output_path = ""):
    super().__init__(env, name)
    self._actions = []
    self._screenshots = []
    self._summarys = []
    self._thoughts = []
    self.output_result = {}
    self.output_path = output_path
    if self.output_path and not os.path.exists(self.output_path):
      os.mkdir(self.output_path)
    self.vllm = vllm

    self.add_thought = True
    self._text_actions = []
    self.src_format = src_format

    self.url = url
    self.api_key = api_key

    self.output_list = []
    self._response = []
    self.task_name = {}

  def reset(self, go_home: bool = False) -> None:
    super().reset(go_home)
    self.env.hide_automation_ui()
    self._actions.clear()
    self._text_actions.clear()
    self._screenshots.clear() # TODO
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
    action_response = ''
    action = None
    
    system_prompt_part, user_prompt_part = build_system_messages(stage2_user_prompt, stage2_history)

    # print(system_prompt_part)
    user_prompt_part['content'].append({'image': screenshot_file})
    
    
    messages = [system_prompt_part, user_prompt_part]
    # import pdb; pdb.set_trace()
    action_response, _, _ = self.vllm.predict_mm(
          "",
          [],
          messages=messages
      )
    
    result["action_response"] = action_response
    print('========== action_response ==========')
    pprint.pprint(action_response)

    dummy_action = None
    thought = None
    summary = None
    try:
      if self.add_thought:

        # print('action_response: ', action_response)
        part1, part2 = action_response.split("Action:", 1)
        thought = part1.replace("Thought:", "").strip()
        action_part = part2.split("<tool_call>", 1)[0]
        summary = action_part.strip().strip('"')
        dummy_action = extract_tag_content('tool_call', action_response)
      else:
        dummy_action = extract_tag_content('tool_call', action_response)
        thought = None
        summary = None

      dummy_action = json.loads(dummy_action)
      dummy_action['arguments']['action'] = dummy_action['arguments']['action'].replace('tap', 'click')
      if len(self._actions) > 0 and self._actions[-1]['arguments']['action'] == 'answer':
          dummy_action = {"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}}
          self.env.interaction_cache =  self._actions[-1]['arguments']['text']

      for key in dummy_action['arguments']:
        if key in ['coordinate', 'coordinate2']:
          x, y = dummy_action['arguments'][key]
          new_point = rescale_coordinates((x, y), resized_width, resized_height)
          dummy_action['arguments'][key] = new_point

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
      # VenusBench-Mobile
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
  
def build_system_messages(user_prompt, history):
    system_prompt_part = {
        "role": "system",
        "content": [
            {
                "text": system_prompt
            }
        ]
    }

    user_prompt_part = {
        "role": "user",
        "content": [
            {
                "text": f"The user query: {user_prompt}.\nTask progress (You have done the following operation on the current device): {history}.\n"
            }
        ]
    }

    return system_prompt_part, user_prompt_part