"""Tasks for general system tasks like interacting with settings."""

import dataclasses
import random
from typing import Any

from absl import logging
from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.utils import fuzzy_match_lib
import immutabledict
from android_world.policy.verification import VerifyPolicy
from android_world.env import representation_utils


class _Tomoto(task_eval.TaskEval):
  """Task for checking that the screen brightness has been set to {max_or_min}."""

  app_names = ('pomodoro',)
  complexity = 1
  schema = {
  }
  template = ''

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}


# Time
class GSATimingTomatoLongBreakTime(_Tomoto):
  schema = {}

  complexity = 2
  time_gap = 2

  template = 'Launch the Tomato app, ' \
  'let the long-break timer run for about {time_start}-{time_end} seconds, ' \
  'then hit Pause.'

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)

    # Clear data in Tomota
    activity = adb_utils.get_adb_activity("tomato")
    if activity is None:
      logging.warning("Tomato app not found or not installed. Skipping clear_app_data.")
      return
    
    package_name = adb_utils.extract_package_name(activity)
    if package_name is not None:
      adb_utils.clear_app_data(package_name, env.controller)

    
  def is_successful(self, env: interface.AsyncEnv) -> float:
    def is_tomoto_longbreak_time_in_range(time_str, time_start, time_end):
      try:
          second = int(time_str.split(':')[1])
          
          # 计算范围：60 - time_start 到 60 - time_end
          range_start = 60 - time_start
          range_end = 60 - time_end
          
          return range_end <= second <= range_start
      
      except (ValueError, IndexError):
          print("时间格式错误，请使用 'MM:SS' 格式")
          return False
    
    super().is_successful(env)
    ui_elements = representation_utils.xml_dump_to_ui_elements(
        adb_utils.uiautomator_dump(env.controller)
    )
    is14exist = False
    for element in ui_elements:
      if 'Tomato notification:' in element.content_description:
        #  Check Longbreak
         if 'Long break  ·  14 min remaining' not in element.content_description:
           return 0
         
      if '14:' in element.text:
        is14exist = True
        # Check time passing of Longbreak
        if is_tomoto_longbreak_time_in_range(
            time_str=element.text, 
            time_start=self.params['time_start'],
            time_end=self.params['time_end']
          ) == False:
          print(f"{element.text} time not in range")
          return 0

    if is14exist == False:
      print("14: not found in any element")
      return 0
    
    return 1.0



  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    time_start =random.randint(10, 20)
    return {'time_start': time_start,
            'time_end': time_start+cls.time_gap}



# Refusal
class RefusalTomotoSettingConflict1(_Tomoto):

  complexity = 2
  
  template = 'Change the Session Length in Tomoto app as {length}.'
  answer = 'I can\'t set the session length to {length}; the maximum allowed is 6.'

  schema = {
        "type": "object",
        "properties": {
            "length": {"type": "integer"}
        },
        "required": ["length"]
    }
  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ans = self.answer.format(length=self.params['length'])
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    ply = VerifyPolicy({})
    result = ply.verify_textual_prediction(
            goal=self.goal,
            prediction=env.interaction_cache,
            gt_answer = ans,
            judgement_type = 'refusal'
        )
    return result

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {'length': random.choice([7, 8, 9, 10, 11, 12])}


# Refusal
class RefusalTomotoSettingConflict2(_Tomoto):
  schema = {
      'color': {
          'type': 'string',
      }
  }

  complexity = 2
  

  template = 'Change the Color Theme in Tomoto app as {color}.'
  answer = (
    'Sorry, I can\'t set the color theme to {color}; '
    'that color isn\'t available in the Color Theme of Tomoto app.'
)
  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    # Provide default color if not in params
    color = self.params['color']
    ans = self.answer.format(color=color)
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    ply = VerifyPolicy({})
    result = ply.verify_textual_prediction(
            goal=self.goal,
            prediction=env.interaction_cache,
            gt_answer = ans,
            judgement_type = 'refusal'
        )
    return result

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    color = random.choice(['dark purple', 'black', 'dark red', 'pure red'])
    return {'color': color}

