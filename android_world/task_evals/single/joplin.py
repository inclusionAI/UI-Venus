
import random
from typing import Any
from android_world.task_evals.information_retrieval import datetime_utils as datetime_utils_ir
from android_world.task_evals.information_retrieval import task_app_utils

from absl import logging
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import file_validators
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils
from android_world.task_evals.information_retrieval.proto import task_pb2
from android_world.task_evals.information_retrieval import proto_utils
from android_world.task_evals.utils import sqlite_utils
from android_world.env import adb_utils
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.env import adb_utils
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals.information_retrieval import calendar_utils
from android_world.task_evals.information_retrieval import datetime_utils as datetime_utils_ir
from android_world.task_evals.information_retrieval import proto_utils
from android_world.task_evals.information_retrieval.proto import state_pb2
from android_world.task_evals.information_retrieval.proto import task_pb2
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import sqlite_utils
from android_world.utils import datetime_utils
import datetime
import random
import uuid


class _Joplin(task_eval.TaskEval):
  app_names = ("joplin",)
  template = ''
  schema = {}
  app_names = ()
  complexity = 3.0  # Overridden in the registry.
  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
  
  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return 0
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}

from android_world.policy.verification import VerifyPolicy

class GUIMJoplinDrawCircle(_Joplin):
  template = 'Draw a red circle using Joplin app. After drawing, leave the canvas on-screen so the user can see the result.'
  complexity = 2
  
  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    # Get the last screenshot path from instance
    if not self.last_screenshot_path:
      print("self.last_screenshot_path not exist !")
      return 0.0
    
    goal = self.template
    task_key = 'JoplinDrawCircleAndRectangle'
    ply = VerifyPolicy({})
    result = ply.verify_drawing_task(
        goal=goal,
        canvas_screenshot_path=self.last_screenshot_path,
        task_key=task_key,
    )
    return result


class GUIMJoplinDrawCircleAndRectangle(_Joplin):
  template = 'In Joplin app, first draw a circle, ' \
  'then draw a rectangle inside it ' \
  'so that all four vertices of the rectangle lie ' \
  'exactly on the circle’s circumference.' \
  'After drawing, leave the canvas on-screen so the user can see the result.'

  complexity = 3
  
  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    # Get the last screenshot path from instance
    if not self.last_screenshot_path:
      print("self.last_screenshot_path not exist !")
      return 0.0
    
    goal = self.template
    task_key = 'JoplinDrawCircleAndRectangle'
    ply = VerifyPolicy({})
    result = ply.verify_drawing_task(
        goal=goal,
        canvas_screenshot_path=self.last_screenshot_path,
        task_key=task_key,
    )
    return result