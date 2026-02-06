"""AutoGLM Agent adapted for Android World framework."""

import base64
import json
import re
import time
import traceback
from io import BytesIO
from typing import Any, Callable

from PIL import Image

from android_world.agents import base_agent
from android_world.env import interface
from android_world.env import json_action_autoglm as json_action
from android_world.env import representation_utils
from android_env.proto import adb_pb2
from android_world.env import adb_utils
from android_env import env_interface


from datetime import datetime

SYSTEM_PROMPT = (
"""
# Setup
You are a professional Android operation agent assistant that can fulfill the user's high-level instructions. Given a screenshot of the Android interface at each step, you first analyze the situation, then plan the best course of action using Python-style pseudo-code.

# More details about the code
Your response format must be structured as follows:

Think first: Use <think>...</think> to analyze the current screen, identify key elements, and determine the most efficient action.
Provide the action: Use <answer>...</answer> to return a single line of pseudo-code representing the operation.

Your output should STRICTLY follow the format:
<think>
[Your thought]
</think>
<answer>
[Your operation code]
</answer>

- **Tap**
  Perform a tap action on a specified screen area. The element is a list of 2 integers, representing the coordinates of the tap point.
  **Example**:
  <answer>
  do(action="Tap", element=[x,y])
  </answer>
- **Type**
  Enter text into the currently focused input field.
  **Example**:
  <answer>
  do(action="Type", text="Hello World")
  </answer>
- **Swipe**
  Perform a swipe action with start point and end point.
  **Examples**:
  <answer>
  do(action="Swipe", start=[x1,y1], end=[x2,y2])
  </answer>
- **Long Press**
  Perform a long press action on a specified screen area.
  You can add the element to the action to specify the long press area. The element is a list of 2 integers, representing the coordinates of the long press point.
  **Example**:
  <answer>
  do(action="Long Press", element=[x,y])
  </answer>
- **Launch**
  Launch an app. Try to use launch action when you need to launch an app. Check the instruction to choose the right app before you use this action.
  **Example**:
  <answer>
  do(action="Launch", app="Settings")
  </answer>
- **Back**
  Press the Back button to navigate to the previous screen.
  **Example**:
  <answer>
  do(action="Back")
  </answer>
- **Finish**
  Terminate the program and optionally print a message.
  **Example**:
  <answer>
  finish(message="Task completed.")
  </answer>


REMEMBER:
- Think before you act: Always analyze the current UI and the best course of action before executing any step, and output in <think> part.
- Only ONE LINE of action in <answer> part per response: Each step must contain exactly one line of executable code.
- Generate execution code strictly according to format requirements.
"""
)

def image_to_base64(image) -> str:
    """Convert numpy array image to base64 string."""
    image_pil = Image.fromarray(image)
    buffer = BytesIO()
    image_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class MessageBuilder:
    """Helper class for building conversation messages."""

    @staticmethod
    def create_system_message(content: str) -> dict[str, Any]:
        """Create a system message."""
        return {"role": "system", "content": content}

    @staticmethod
    def create_user_message(
        text: str, image_base64: str | None = None
    ) -> dict[str, Any]:
        """
        Create a user message with optional image.

        Args:
            text: Text content.
            image_base64: Optional base64-encoded image.

        Returns:
            Message dictionary.
        """
        content = []

        if image_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                }
            )

        content.append({"type": "text", "text": text})

        return {"role": "user", "content": content}

    @staticmethod
    def create_assistant_message(content: str) -> dict[str, Any]:
        """Create an assistant message."""
        return {"role": "assistant", "content": content}

    @staticmethod
    def remove_images_from_message(message: dict[str, Any]) -> dict[str, Any]:
        """
        Remove image content from a message to save context space.

        Args:
            message: Message dictionary.

        Returns:
            Message with images removed.
        """
        if isinstance(message.get("content"), list):
            message["content"] = [
                item for item in message["content"] if item.get("type") == "text"
            ]
        return message

    @staticmethod
    def build_screen_info(current_app: str, **extra_info) -> str:
        """
        Build screen info string for the model.

        Args:
            current_app: Current app name.
            **extra_info: Additional info to include.

        Returns:
            JSON string with screen info.
        """
        info = {"current_app": current_app, **extra_info}
        return json.dumps(info, ensure_ascii=False)




class LLMClient:
    """Client for interacting with LLM server with streaming support."""
    
    def __init__(self, base_url: str, api_key: str = "EMPTY", lang: str = "en"):
        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            self.lang = lang
        except ImportError:
            raise ImportError("Please install openai package: pip install openai")
    
    def generate(
        self, 
        messages: list, 
        max_tokens: int = 3000, 
        temperature: float = 0.0,
        verbose: bool = True
    ) -> tuple[str, str, str]:
        """
        Generate response from LLM with streaming.
        
        Returns:
            Tuple of (thinking, action)
        """
        start_time = time.time()
        time_to_first_token = None
        
        stream = self.client.chat.completions.create(
            model="autoglm-phone-9b",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.85,
            frequency_penalty=0.2,
            stream=True,
        )

        raw_content = ""
        buffer = ""
        action_markers = ["finish(message=", "do(action="]
        in_action_phase = False
        first_token_received = False

        for chunk in stream:
            if len(chunk.choices) == 0:
                continue
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                raw_content += content

                # Record time to first token
                if not first_token_received:
                    time_to_first_token = time.time() - start_time
                    first_token_received = True

                if in_action_phase:
                    continue

                buffer += content

                # Check if any marker is fully present in buffer
                marker_found = False
                for marker in action_markers:
                    if marker in buffer:
                        thinking_part = buffer.split(marker, 1)[0]
                        if verbose:
                            print(thinking_part, end="", flush=True)
                            print()  # Newline after thinking
                        in_action_phase = True
                        marker_found = True
                        break

                if marker_found:
                    continue

                # Check if buffer ends with a prefix of any marker
                is_potential_marker = False
                for marker in action_markers:
                    for i in range(1, len(marker)):
                        if buffer.endswith(marker[:i]):
                            is_potential_marker = True
                            break
                    if is_potential_marker:
                        break

                if not is_potential_marker:
                    if verbose:
                        print(buffer, end="", flush=True)
                    buffer = ""

        if verbose:
            print()

        # Calculate total time
        total_time = time.time() - start_time

        # Print performance metrics
        if verbose:
            print("-" * 50)
            print(f"⏱️  Performance Metrics:")
            if time_to_first_token is not None:
                print(f"   Time to first token: {time_to_first_token:.3f}s")
            print(f"   Total inference time: {total_time:.3f}s")
            print("-" * 50)

        # Parse thinking and action
        thinking, action = self._parse_response(raw_content)
        
        return thinking, action, raw_content

    def _parse_response(self, content: str) -> tuple[str, str]:
        """Parse the model response into thinking and action parts."""
        # Rule 1: Check for finish(message=
        if "finish(message=" in content:
            parts = content.split("finish(message=", 1)
            thinking = parts[0].strip()
            action = "finish(message=" + parts[1]
            return thinking, action

        # Rule 2: Check for do(action=
        if "do(action=" in content:
            parts = content.split("do(action=", 1)
            thinking = parts[0].strip()
            action = "do(action=" + parts[1]
            return thinking, action

        # Rule 3: Fallback to legacy XML tag parsing
        if "<answer>" in content:
            parts = content.split("<answer>", 1)
            thinking = parts[0].replace("<think>", "").replace("</think>", "").strip()
            action = parts[1].replace("</answer>", "").strip()
            return thinking, action

        # Rule 4: No markers found, return content as action
        return "", content
_DEFAULT_TIMEOUT_SECS = 30

import ast
def get_current_app(env: env_interface.AndroidEnvInterface) -> str:
    """
    Get the currently focused app name.

    Args:
        env: The Android environment interface.

    Returns:
        The app name if recognized, otherwise "System Home".
    """
    response = adb_utils.issue_generic_request(
        ["shell", "dumpsys", "window"],
        env
    )
    
    if response.status != adb_pb2.AdbResponse.Status.OK:
        raise ValueError("Failed to execute dumpsys window command")
    
    output = response.generic.output.decode().strip()
    if not output:
        raise ValueError("No output from dumpsys window")

    # Parse window focus info
    for line in output.split("\n"):
        if "mCurrentFocus" in line or "mFocusedApp" in line:
            # Check each pattern in _PATTERN_TO_ACTIVITY
            for pattern, activity in adb_utils._PATTERN_TO_ACTIVITY.items():
                # Extract package name from activity string (format: package/activity)
                package = activity.split('/')[0]
                if package in line:
                    # Return the first pattern name (before any '|')
                    return pattern.split('|')[0]

    return "System Home"


class AutoGLMAgent(base_agent.EnvironmentInteractingAgent):
    """
    AutoGLM Agent adapted for Android World framework.
    
    This agent uses vision-language models to understand screen content
    and decide on actions to complete user tasks.
    
    Args:
        env: The Android environment interface.
        model_base_url: Base URL for the LLM API endpoint.
        name: Name identifier for the agent.
        max_steps: Maximum number of steps before stopping.
        wait_after_action_seconds: Wait time after each action.
        verbose: Whether to print detailed logs.
        lang: Language for UI messages ('en' or 'cn').
    """
    
    def __init__(
        self,
        env: interface.AsyncEnv,
        model_base_url: str = "http://127.0.0.1:8000/v1",
        name: str = "AutoGLM_Agent",
        max_steps: int = 100,
        wait_after_action_seconds: float = 2.0,
        verbose: bool = True,
        output_path: str = '',
        lang: str = "en",
    ):
        super().__init__(env, name)
        
        self.llm_client = LLMClient(model_base_url, lang=lang)
        self.max_steps = max_steps
        self.wait_after_action_seconds = wait_after_action_seconds
        self.verbose = verbose
        self.lang = lang
        self.output_path = output_path
        self.context = []  # Conversation context
        self.step_count = 0
        self.max_context_images = 8  
        self.max_context_turns = 20  
    
    def reset(self, go_home_on_reset: bool = False):
        """Reset agent state for a new task."""
        super().reset(go_home_on_reset)
        self.env.hide_automation_ui()
        self.context = []
        self.step_count = 0
    

    def _parse_action_string(self, response: str) -> dict[str, Any]:
        """
        Parse action from model response.

        Args:
            response: Raw response string from the model.

        Returns:
            Parsed action dictionary.

        Raises:
            ValueError: If the response cannot be parsed.
        """
        print(f"Parsing action: {response}")
        try:
            response = response.strip()
            
            if '\n' in response:
                lines = response.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('do(') or line.startswith('finish('):
                        response = line
                        break
                if response.startswith('do(') or response.startswith('finish('):
                    paren_count = 0
                    end_idx = 0
                    for i, char in enumerate(response):
                        if char == '(':
                            paren_count += 1
                        elif char == ')':
                            paren_count -= 1
                            if paren_count == 0:
                                end_idx = i + 1
                                break
                    if end_idx > 0:
                        response = response[:end_idx]
                print(f"Extracted first action: {response}")
            if response.startswith('do(action="Type"') or response.startswith(
                'do(action="Type_Name"'
            ):
                text = response.split("text=", 1)[1][1:-2]
                action = {"_metadata": "do", "action": "Type", "text": text}
                return action
            elif response.startswith("do"):
                # Use AST parsing instead of eval for safety
                try:
                    # Escape special characters (newlines, tabs, etc.) for valid Python syntax
                    response = response.replace('\n', '\\n')
                    response = response.replace('\r', '\\r')
                    response = response.replace('\t', '\\t')

                    tree = ast.parse(response, mode="eval")
                    if not isinstance(tree.body, ast.Call):
                        raise ValueError("Expected a function call")

                    call = tree.body
                    # Extract keyword arguments safely
                    action = {"_metadata": "do"}
                    for keyword in call.keywords:
                        key = keyword.arg
                        value = ast.literal_eval(keyword.value)
                        action[key] = value

                    return action
                except (SyntaxError, ValueError) as e:
                    raise ValueError(f"Failed to parse do() action: {e}")

            elif response.startswith("finish"):
                action = {
                    "_metadata": "finish",
                    "message": response.replace("finish(message=", "")[1:-2],
                }
            else:
                raise ValueError(f"Failed to parse action: {response}")
            return action
        except Exception as e:
            raise ValueError(f"Failed to parse action: {e}")



    def _convert_to_android_world_action(
        self, 
        action: dict, 
        screen_width: int, 
        screen_height: int
    ) -> json_action.JSONAction:
        """
        Convert AutoGLM action format to Android World JSONAction.
        
        AutoGLM uses coordinates in 0-1000 range, need to convert to actual pixels.
        """
        action_type = action.get('_metadata')
        
        if action_type == 'finish':
            message = action.get('message', 'Task completed')
            return json_action.JSONAction(
                action_type='status',
                goal_status='success',
                text=message
            )
        

        if action_type != 'do':
            raise ValueError(f"Unknown action metadata: {action_type}")
        
        action_name = action.get('action')
        
        # Helper function to convert relative coords (0-1000) to absolute pixels
        def convert_coords(element):
            x = int(element[0] / 1000 * screen_width)
            y = int(element[1] / 1000 * screen_height)
            # Clamp to screen bounds
            x = max(0, min(x, screen_width))
            y = max(0, min(y, screen_height))
            return x, y
        
        if action_name == 'Take_over':
            message = action.get('message', '')
            return json_action.JSONAction(action_type='answer', text=message)
        if action_name == 'Call_API':
            message = action.get('message', '')
            return json_action.JSONAction(action_type='answer', text=message)
        if action_name == 'Interact':
            message = action.get('message', '')
            return json_action.JSONAction(action_type='answer', text=message)
        if action_name == 'Note':
            message = action.get('message', '')
            return json_action.JSONAction(action_type='answer', text=message)

        if action_name == 'Tap':
            element = action.get('element')
            if not element:
                raise ValueError("Tap action requires 'element' parameter")
            x, y = convert_coords(element)
            return json_action.JSONAction(action_type='click', x=x, y=y)
        elif action_name == 'Double Tap':
            element = action.get('element')
            x, y = convert_coords(element)
            return json_action.JSONAction(action_type='double_tap', x=x, y=y)
        # Type, Type_Name
        elif action_name == 'Type':
            text = action.get('text', '')
            return json_action.JSONAction(action_type='input_text', text=text)
        
        elif action_name == 'Swipe':
            start = action.get('start')
            end = action.get('end')
            if not start or not end:
                raise ValueError("Swipe action requires 'start' and 'end' parameters")
            x1, y1 = convert_coords(start)
            x2, y2 = convert_coords(end)
            return json_action.JSONAction(
                action_type='swipe',
                x=x1, y=y1, x2=x2, y2=y2,
                duration="500"
            )
        
        elif action_name == 'Long Press':
            element = action.get('element')
            if not element:
                raise ValueError("Long Press action requires 'element' parameter")
            x, y = convert_coords(element)
            return json_action.JSONAction(action_type='long_press', x=x, y=y)
        
        elif action_name == 'Launch':
            app_name = action.get('app')
            if not app_name:
                raise ValueError("Launch action requires 'app' parameter")
            # Normalize app name: strip whitespace and convert to lowercase
            app_name = app_name.strip().lower()
            return json_action.JSONAction(action_type='open_app', app_name=app_name)
        

        if action_name == 'Wait':
            return json_action.JSONAction(action_type='wait')
        
        elif action_name == 'Back':
            return json_action.JSONAction(action_type='navigate_back')
        
        elif action_name == 'Home':
            return json_action.JSONAction(action_type='navigate_home')
        
        else:
            raise ValueError(f"Unsupported action: {action_name}")
    
    def _truncate_context(self):

        if len(self.context) <= 2: 
            return
 
        max_messages = 1 + self.max_context_turns * 2
        if len(self.context) > max_messages:
            self.context = [self.context[0]] + self.context[-(max_messages-1):]
            if self.verbose:
                print(f"⚠️  Context truncated to {len(self.context)} messages")
        
        image_count = 0
        for msg in self.context:
            if isinstance(msg.get('content'), list):
                for item in msg['content']:
                    if item.get('type') == 'image_url':
                        image_count += 1
        
        if image_count > self.max_context_images:
            removed = 0
            for i in range(1, len(self.context) - 1):
                msg = self.context[i]
                if isinstance(msg.get('content'), list):
                    has_image = any(item.get('type') == 'image_url' for item in msg['content'])
                    if has_image:
                        self.context[i] = MessageBuilder.remove_images_from_message(msg)
                        removed += 1
                        image_count -= 1
                        if image_count <= self.max_context_images:
                            break
            
            if self.verbose and removed > 0:
                print(f"⚠️  Removed images from {removed} old messages")
        
    def _print_context_to_model(self):
        """Print the complete text context that will be sent to the model."""
        print("\n" + "=" * 80)
        print(f"📨 FULL CONTEXT SENT TO MODEL (Step {self.step_count})")
        print(f"   Total messages: {len(self.context)}")
        print("=" * 80)
        
        for i, msg in enumerate(self.context):
            role = msg.get('role', 'unknown').upper()
            print(f"\n[Message {i+1}] {role}:")
            print("-" * 40)
            
            content = msg.get('content')
            
            if isinstance(content, str):
                # Simple text message
                print(content)
            elif isinstance(content, list):
                # Multi-part message (text + image)
                for part in content:
                    if part.get('type') == 'text':
                        print(part.get('text', ''))
                    elif part.get('type') == 'image_url':
                        print("[🖼️  IMAGE ATTACHED]")
            else:
                print(f"[Unknown content type: {type(content)}]")
        
        print("\n" + "=" * 80)
        print()


    def step(self, goal: str, step_numb: bool=False) -> base_agent.AgentInteractionResult:
        """
        Execute a single step of the agent.
        
        Args:
            goal: The user's task description.
            
        Returns:
            AgentInteractionResult indicating success and step data.
        """
        self.step_count += 1
        
        step_data = {
            'raw_screenshot': None,
            'action_prompt': None,
            'action_output': None,
            'thinking': None,
            'action_string': None,
            'action_output_json': None,
            'converted_action': None,
            'summary': None,
        }
        
        if self.verbose:
            print(f'\n{"="*50}')
            print(f"💭 Step {self.step_count}")
            print(f'{"="*50}')
        
        # Get current state
        state = self.get_post_transition_state()
        screen_width, screen_height = self.env.logical_screen_size
        step_data['raw_screenshot'] = state.pixels.copy()
        
        # Get current app
        try:
            current_app = get_current_app(env=self.env.controller)
        except:
            current_app = "Unknown"
        
        # Get screenshot as base64
        screenshot = state.pixels
        encoded_image = image_to_base64(screenshot)

        
        
        # Truncate context to avoid exceeding limits
        self._truncate_context()
        
        # Build messages for LLM
        try:
            is_first = len(self.context) == 0
            
            if is_first:
                # First step: add system message and task
                self.context.append(
                    MessageBuilder.create_system_message(SYSTEM_PROMPT)
                )
                
                screen_info = MessageBuilder.build_screen_info(current_app)
                text_content = f"{goal}\n\n{screen_info}"
                

                self.context.append(
                    MessageBuilder.create_user_message(
                        text=text_content, 
                        image_base64=encoded_image
                    )
                )
            else:
                # Subsequent steps: add new screen info
                screen_info = MessageBuilder.build_screen_info(current_app)
                text_content = f"** Screen Info **\n\n{screen_info}"
                
                self.context.append(
                    MessageBuilder.create_user_message(
                        text=text_content,
                        image_base64=encoded_image
                    )
                )
            
            step_data['action_prompt'] = json.dumps(self.context, ensure_ascii=False)
            
        except Exception as e:
            if self.verbose:
                traceback.print_exc()
            step_data['summary'] = f"Failed to build messages: {e}"
            return base_agent.AgentInteractionResult(False, step_data)
        
        # Get LLM response
        try:
            if self.verbose:
                print(f"\n🤖 Model Response:")
                print(f"{'-'*50}")
            
            thinking, action_string, raw_response = self.llm_client.generate(
                self.context, 
                verbose=self.verbose
            )
            
            
            step_data['action_output'] = raw_response
            step_data['thinking'] = thinking
            step_data['action_string'] = action_string
            
            if self.verbose:
                print(f"\n💭 Thinking: {thinking}")
                print(f"🎯 Action: {action_string}\n")
                
        except Exception as e:
            if self.verbose:
                traceback.print_exc()
            step_data['summary'] = f"LLM generation failed: {e}"
            return base_agent.AgentInteractionResult(False, step_data)
        
        # Remove image from context to save space (关键步骤!)
        self.context[-1] = MessageBuilder.remove_images_from_message(self.context[-1])
        # self._print_context_to_model()

        # Parse action string as JSON from action_string extracted from raw response.
        try:
            action_json = self._parse_action_string(action_string)
            step_data['action_output_json'] = action_json
            
            if self.verbose:
                print(f"📋 Parsed Action:")
                print(json.dumps(action_json, ensure_ascii=False, indent=2))
                print()
                
        except Exception as e:
            if self.verbose:
                traceback.print_exc()
            step_data['summary'] = f"Failed to parse action string: {e}"
            
            if len(self.context) > 1:
                self.context.pop()  
            
            return base_agent.AgentInteractionResult(False, step_data)
        
        # Check if finished
        if action_json.get('_metadata') == 'finish':
            message = action_json.get('message', 'Task completed')
            step_data['summary'] = message

            self.env.update_interaction_cache(message)

            # Add assistant response to context (关键步骤!)
            self.context.append(
                MessageBuilder.create_assistant_message(
                    f"{thinking}\n{action_string}"
                )
            )
            
            if self.verbose:
                print(f"\n🎉 {'='*48}")
                print(f"✅ Task Completed: {message}")
                print(f"{'='*50}\n")
            return base_agent.AgentInteractionResult(True, step_data)
        

        # Convert to Android World action
        try:
            converted_action = self._convert_to_android_world_action(
                action_json, screen_width, screen_height
            )
            step_data['converted_action'] = str(converted_action)
            
            if self.verbose:
                print(f"🔄 Converted Action: {converted_action}\n")
                
        except Exception as e:
            if self.verbose:
                traceback.print_exc()
            step_data['summary'] = f"Failed to convert action: {e}"
            
            # Add assistant response even on error
            self.context.append(
                MessageBuilder.create_assistant_message(
                    f"{thinking}\n{action_string}"
                )
            )
            return base_agent.AgentInteractionResult(False, step_data)
        
        # Handle answer action
        if converted_action.action_type == 'answer':
            self.env.update_interaction_cache(converted_action.text)
            return base_agent.AgentInteractionResult(True, step_data)


        # Execute action
        try:
            if step_numb == False:
                self.env.execute_action(converted_action, actuation_type='autoglm')
            
            if self.verbose:
                print(f"✓ Action executed successfully")
            
            # Wait after action
            if self.wait_after_action_seconds > 0:
                time.sleep(self.wait_after_action_seconds)
            
            step_data['summary'] = "Action executed successfully"
            
        except Exception as e:
            if self.verbose:
                traceback.print_exc()
            step_data['summary'] = f"Failed to execute action: {e}"
            
            # Add assistant response even on error
            self.context.append(
                MessageBuilder.create_assistant_message(
                    f"{thinking}\n{action_string}"
                )
            )
            print("return exception step_data")
            return base_agent.AgentInteractionResult(False, step_data)
        
        self.context.append(
            MessageBuilder.create_assistant_message(
                f"{thinking}\n{action_string}"
            )
        )
        

        return base_agent.AgentInteractionResult(False, step_data)
    

    def run(self, goal: str) -> str:
        """
        Run the agent to complete a task.
        
        Args:
            goal: Natural language description of the task.
            
        Returns:
            Final message from the agent.
        """
        self.reset()
        
        if self.verbose:
            print(f"\n{'='*50}")
            print(f"🎯 Task: {goal}")
            print(f"{'='*50}\n")
        
        while self.step_count < self.max_steps:
            result = self.step(goal)
            
            if result.done:
                return result.data.get('summary', 'Task completed')
        
        return f"Max steps ({self.max_steps}) reached without completion"
