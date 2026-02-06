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


# This file has been modified from the original AndroidWorld project.
# Changes: Implementing tasks in VenusBench.

"""Tasks for Simple Draw Pro app."""

import os
import random
import yaml
from typing import Any
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import file_validators
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils
from android_world.env import adb_utils




import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from config import config

# Load local paths from config
_LOCAL_SIMPLE_DRAW_GUIM = config.get('local_paths.simple_draw_guim')

# Load remote paths from config
_REMOTE_SDCARD = config.get('remote_paths.sdcard')



class SimpleDrawProCreateDrawing(task_eval.TaskEval):
  """Task for checking that a new drawing has been created with a specific name."""

  app_names = ("simple draw pro",)
  complexity = 1.8
  schema = file_validators.CreateFile.schema
  template = (
      "Create a new drawing in Simple Draw Pro. Name it {file_name}. Save it in"
      " the Pictures folder within the sdk_gphone_x86_64 storage area."
  )

  def __init__(self, params: dict[str, Any]):
    super().__init__(params)
    self.initialized = False
    self.create_file_task = file_validators.CreateFile(
        params,
        file_utils.convert_to_posix_path(
            device_constants.EMULATOR_DATA, "Pictures"
        ),
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    file_name = self.params["file_name"]
    exists = file_utils.check_file_or_folder_exists(
        file_name, self.create_file_task.data_directory, env.controller
    )
    return 1.0 if exists else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    words = [
        "lorem",
        "ipsum",
        "dolor",
        "sit",
        "amet",
        "consectetur",
        "adipiscing",
        "elit",
    ]
    extensions = [".png", ".svg", ".jpg"]
    random_file_name = (
        "".join(random.choices(words, k=1))
        + "_"
        + user_data_generation.generate_random_file_name()
        + random.choice(extensions)
    )

    return {
        "file_name": random_file_name,
        "text": "",  # Unused.
    }

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.create_file_task.tear_down(env)



class _Draw(task_eval.TaskEval):
  """Task for checking that the screen brightness has been set to {max_or_min}."""

  app_names = ("simple draw pro",)
  complexity = 2
  schema = {
  }
  template = ''

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)

    adb_utils.clear_app_data(
      adb_utils.extract_package_name(adb_utils.get_adb_activity("simple draw pro")),
      env.controller,
    )


  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
  
    return 1.0 

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {'max_or_min': 'max' if random.choice([True, False]) else 'min'}

from android_world.policy.verification import VerifyPolicy

class GUIMDrawA(_Draw):
  schema = {}

  complexity = 2
  
  template = (
    'Draw a uppercase {color} character A in Draw app. '    
    'After drawing, leave the canvas on-screen so the user can see the result. '
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    color_pool = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
    return {'color': random.choice(color_pool)}
  

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    # Get the last screenshot path from instance
    if not self.last_screenshot_path:
      print("self.last_screenshot_path not exist !")
      return 0.0
    
    goal = self.template
    task_key = 'DrawA'
    ply = VerifyPolicy({})
    result = ply.verify_drawing_task(
        goal=goal,
        canvas_screenshot_path=self.last_screenshot_path,
        task_key=task_key,
    )
    return result
  

class GUIMDrawACHS(GUIMDrawA):
  template = (
    '在Draw应用中绘制一个大写的{color}字母A。'
    '绘制完成后，保持画布显示在屏幕上，以便用户可以看到结果。'
)

class GUIMDrawAENGVariation(GUIMDrawA):
  template = (
    'In the Draw app, sketch a capital {color} letter A. '
    'Once finished, keep the canvas visible on screen for the user to view the outcome. '
  )

class GUIMDrawRectangleHard(_Draw):
  schema = {}

  template = (
        'In the Draw app, on a white canvas, '
        'draw a large square with a {outer_color} line. '
        'Then, inside that {outer_color} square, '
        'draw a smaller square with a {inner_color} line. '
        'After drawing, leave the canvas on-screen so the user can see the result.'
    )
  
  complexity = 3


  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    color_pool = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
    # 确保内外颜色不同，避免视觉混淆
    outer_color = random.choice(color_pool)
    inner_color = random.choice([c for c in color_pool if c != outer_color])
    return {
        'outer_color': outer_color,
        'inner_color': inner_color
    }
  

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    # Get the last screenshot path from instance
    if not self.last_screenshot_path:
      print("self.last_screenshot_path not exist !")
      return 0.0
    
    goal = self.template
    task_key = 'DrawRectangleHard'
    ply = VerifyPolicy({})
    result = ply.verify_drawing_task(
        goal=goal,
        canvas_screenshot_path=self.last_screenshot_path,
        task_key=task_key,
    )
    return result
  

class GUIMEraseObject1(_Draw):
  schema = {}

  template = (
        'There is a picture named graphic.png located at the folder GUIM within sdk_gphone_x86_64 storage area.'
        'Use the Draw app to open the image, '
        'erase the green shape located below the three shapes in red, blue, and yellow, and stay on the screen after the erasing is completed.'
    )
  
  complexity = 3
  
  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.push_file_to_emulator(_LOCAL_SIMPLE_DRAW_GUIM, remote_path=_REMOTE_SDCARD, env=env.controller)

  

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    # Get the last screenshot path from instance
    if not self.last_screenshot_path:
      print("self.last_screenshot_path not exist !")
      return 0.0
    
    goal = self.template
    task_key = 'GUIMEraseObject1'
    ply = VerifyPolicy({})
    result = ply.verify_edit_task(
        goal=goal,
        canvas_screenshot_path=self.last_screenshot_path,
        task_key=task_key,
    )
    return result
  

class GUIMEraseObject1CHS(GUIMEraseObject1):
  template = (
    '在sdk_gphone_x86_64存储区的GUIM文件夹中有一张名为graphic.png的图片。'
    '使用Draw应用打开该图片，擦除位于红色、蓝色和黄色三个形状下方的绿色形状，'
    '擦除完成后保持屏幕显示。'
  )


class GUIMEraseObject1ENGVariation(GUIMEraseObject1):
  template = (
    'A file named graphic.png is stored in the GUIM folder within the sdk_gphone_x86_64 storage location. '
    'Open this image in the Draw app, remove the green shape positioned beneath the three red, blue, and yellow shapes, '
    'and remain on the screen after the removal is complete.'
  )

  

class GUIMEraseObject2(_Draw):
  schema = {}

  template = (
        'There is a picture named flower.png located at the folder GUIM within sdk_gphone_x86_64 storage area.'
        'Use the Draw app to open the image, erase the green stem below the white flower, as well as all the green leaves on the stem, and stay on the screen after the erasing is completed.'
    )
  
  complexity = 4
  
  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.push_file_to_emulator(_LOCAL_SIMPLE_DRAW_GUIM, remote_path=_REMOTE_SDCARD, env=env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    # Get the last screenshot path from instance
    if not self.last_screenshot_path:
      print("self.last_screenshot_path not exist !")
      return 0.0
    
    goal = self.template
    task_key = 'GUIMEraseObject2'
    ply = VerifyPolicy({})
    result = ply.verify_edit_task(
        goal=goal,
        canvas_screenshot_path=self.last_screenshot_path,
        task_key=task_key,
    )
    return result
  

class GUIMCircleObject1(_Draw):
  schema = {}

  template = (
        'There is a picture named fruit.png located at the folder GUIM within sdk_gphone_x86_64 storage area.'
        'Use the Draw app to open the image, '
        'circle the banana with a red pen, '
        'and stay on the screen after the task is completed.'
    )
  
  complexity = 3
  
  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.push_file_to_emulator(_LOCAL_SIMPLE_DRAW_GUIM, remote_path=_REMOTE_SDCARD, env=env.controller)


  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    # Get the last screenshot path from instance
    if not self.last_screenshot_path:
      print("self.last_screenshot_path not exist !")
      return 0.0
    
    goal = self.template
    task_key = 'GUIMCircleObject1'
    ply = VerifyPolicy({})
    result = ply.verify_edit_task(
        goal=goal,
        canvas_screenshot_path=self.last_screenshot_path,
        task_key=task_key,
    )
    return result
  

class GUIMCircleObject2(_Draw):
  schema = {}

  template = (
        'There is a picture named table.png '
        'located at the folder GUIM within sdk_gphone_x86_64 storage area.'
        'Use the Draw app to open the image, '
        'circle the eraser with a blue pen, '
        'circle the sharpener with a red pen, '
        'and stay on the screen after the task is completed.'
    )
  
  complexity = 3
  
  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.push_file_to_emulator(_LOCAL_SIMPLE_DRAW_GUIM, remote_path=_REMOTE_SDCARD, env=env.controller)


  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    # Get the last screenshot path from instance
    if not self.last_screenshot_path:
      print("self.last_screenshot_path not exist !")
      return 0.0
    
    goal = self.template
    task_key = 'GUIMCircleObject2'
    ply = VerifyPolicy({})
    result = ply.verify_edit_task(
        goal=goal,
        canvas_screenshot_path=self.last_screenshot_path,
        task_key=task_key,
    )
    return result