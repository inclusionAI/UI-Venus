# Copyright 2024 The android_world Authors.
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
"""Represents an AutoGLM action for Android interaction, parsed from a JSON format."""
import dataclasses
import json
from typing import Optional
_JSON_SEPARATORS = (',', ':')
# AutoGLM action types
CLICK = 'click'  # Tap
DOUBLE_TAP = 'double_tap'
LONG_PRESS = 'long_press'
TYPE = 'input_text'
SWIPE = 'swipe'
LAUNCH = 'open_app'
BACK = 'navigate_back'
HOME = 'navigate_home'
WAIT = 'wait'
TAKE_OVER = 'take_over'
NOTE = 'note'
CALL_API = 'call_api'
INTERACT = 'interact'
FINISH = 'finish'
ANSWER = 'answer'
INDEX = 'index'
_ACTION_TYPES = (
    CLICK,
    DOUBLE_TAP,
    LONG_PRESS,
    TYPE,
    SWIPE,
    LAUNCH,
    BACK,
    HOME,
    WAIT,
    TAKE_OVER,
    NOTE,
    CALL_API,
    INTERACT,
    FINISH,
    ANSWER,
)
# Keys of AutoGLM JSON action
ACTION_TYPE = 'action_type'
ELEMENT = 'element'  # [x, y] in 0-1000 range
START = 'start'  # [x, y] for swipe start in 0-1000 range
END = 'end'  # [x, y] for swipe end in 0-1000 range
TEXT = 'text'
APP_NAME = 'app_name'
DURATION = 'duration'  # Duration string like "2 seconds"
DURATION_MS = 'duration_ms'  # Duration in milliseconds
MESSAGE = 'message'  # Message for finish, take_over, note, etc.
PRESS_ENTER = 'press_enter'  # Whether to press enter after typing
GOAL_STATUS = 'goal_status'  # Status for finish action
ACTION_KEYS = [
    ACTION_TYPE,
    ELEMENT,
    START,
    END,
    INDEX,
    TEXT,
    APP_NAME,
    DURATION,
    DURATION_MS,
    MESSAGE,
    PRESS_ENTER,
    GOAL_STATUS,
]
@dataclasses.dataclass()
class JSONAction:
  """Represents a parsed AutoGLM JSON action.
  # Example
  result_json = {'action_type': 'click', 'element': [500, 800]}
  action = AutoGLMJSONAction(**result_json)
  Attributes:
    action_type: The action type (click, double_tap, long_press, type, swipe, etc.).
    element: Coordinates [x, y] in 0-1000 range for tap/click actions.
    start: Start coordinates [x, y] in 0-1000 range for swipe actions.
    end: End coordinates [x, y] in 0-1000 range for swipe actions.
    text: The text to type, if action is type.
    app_name: The app name to launch, if the action type is 'launch'.
    duration: Duration string like "2 seconds" for wait actions.
    duration_ms: Duration in milliseconds for long_press or swipe actions.
    message: Message for finish, take_over, note, or interact actions.
    press_enter: Whether to press enter after typing.
    goal_status: Status of the goal for finish actions ('success', 'failed', 'paused').
  """
  action_type: Optional[str] = None
  element: Optional[list[int]] = None
  start: Optional[list[int]] = None
  end: Optional[list[int]] = None
  index: Optional[str | int] = None
  x: Optional[int] = None
  y: Optional[int] = None
  x2: Optional[int] = None
  y2: Optional[int] = None
  text: Optional[str] = None
  app_name: Optional[str] = None
  duration: Optional[str] = None
  duration_ms: Optional[int] = None
  message: Optional[str] = None
  press_enter: Optional[bool] = None
  goal_status: Optional[str] = None
  app_name: Optional[str] = None
  def __post_init__(self):
    if self.action_type not in _ACTION_TYPES:
      raise ValueError(f'Invalid action type: {self.action_type}')
    
    # Validate element coordinates
    if self.element is not None:
      if not isinstance(self.element, list) or len(self.element) != 2:
        raise ValueError(f'Invalid element coordinates: {self.element}')
      if not all(isinstance(coord, (int, float)) and 0 <= coord <= 1000 for coord in self.element):
        raise ValueError(f'Element coordinates must be in 0-1000 range: {self.element}')
    
    # Validate swipe coordinates
    if self.start is not None:
      if not isinstance(self.start, list) or len(self.start) != 2:
        raise ValueError(f'Invalid start coordinates: {self.start}')
      if not all(isinstance(coord, (int, float)) and 0 <= coord <= 1000 for coord in self.start):
        raise ValueError(f'Start coordinates must be in 0-1000 range: {self.start}')
    
    if self.end is not None:
      if not isinstance(self.end, list) or len(self.end) != 2:
        raise ValueError(f'Invalid end coordinates: {self.end}')
      if not all(isinstance(coord, (int, float)) and 0 <= coord <= 1000 for coord in self.end):
        raise ValueError(f'End coordinates must be in 0-1000 range: {self.end}')
      
    if self.index is not None:
      self.index = int(self.index)
      if self.x is not None or self.y is not None:
        raise ValueError('Either an index or a <x, y> should be provided.')
    # Ensure text is string
    if self.text is not None and not isinstance(self.text, str):
      self.text = str(self.text)
    
    # Validate duration_ms
    if self.duration_ms is not None and self.duration_ms < 0:
      raise ValueError(f'Invalid duration_ms: {self.duration_ms}')
  def __repr__(self) -> str:
    properties = []
    for key, value in self.__dict__.items():
      if value is not None:
        if isinstance(value, float):
          value = f'{value:.3f}'
        properties.append(f'{key}={value!r}')
    return f"AutoGLMJSONAction({', '.join(properties)})"
  def __eq__(self, other):
    if isinstance(other, JSONAction):
      return _compare_actions(self, other)
    return False
  def __ne__(self, other):
    return not self.__eq__(other)
  def json_str(self) -> str:
    """Convert action to JSON string."""
    non_null = {}
    for key, value in self.__dict__.items():
      if value is not None:
        non_null[key] = value
    return json.dumps(non_null, separators=_JSON_SEPARATORS)
def _compare_actions(a: JSONAction, b: JSONAction) -> bool:
  """Compares two AutoGLMJSONActions.
  Args:
    a: The first action.
    b: The second action.
  Returns:
    If the actions are equal.
  """
  # Ignore cases for text, app_name, and message
  if a.app_name is not None and b.app_name is not None:
    app_name_match = a.app_name.lower() == b.app_name.lower()
  else:
    app_name_match = a.app_name == b.app_name
  if a.text is not None and b.text is not None:
    text_match = a.text.lower() == b.text.lower()
  else:
    text_match = a.text == b.text
  if a.message is not None and b.message is not None:
    message_match = a.message.lower() == b.message.lower()
  else:
    message_match = a.message == b.message
  # Compare all fields
  return (
      a.action_type == b.action_type
      and a.element == b.element
      and a.start == b.start
      and a.end == b.end
      and text_match
      and app_name_match
      and a.duration == b.duration
      and a.duration_ms == b.duration_ms
      and message_match
      and a.press_enter == b.press_enter
      and a.goal_status == b.goal_status
  )