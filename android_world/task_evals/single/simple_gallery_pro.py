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

"""Tasks for the Simple Gallery Pro app."""

from typing import Any
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.utils import receipt_generator
from android_world.task_evals.utils import schema
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils
from android_world.policy.verification import VerifyPolicy
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

class SaveCopyOfReceiptTaskEval(task_eval.TaskEval):
  """Task using SimpleGalleryPro to save a copy of a receipt."""

  app_names = ("simple gallery pro",)

  complexity = 1.6

  template = (
      "In Simple Gallery Pro, copy {file_name} in DCIM and save a copy with the"
      " same name in Download"
  )

  schema = schema.no_params()

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    user_data_generation.clear_device_storage(env)
    receipt_image = self.params["receipt_image"]
    temp_storage_location = file_utils.convert_to_posix_path(
        file_utils.get_local_tmp_directory(), self.params["file_name"]
    )
    receipt_image.save(temp_storage_location)
    file_utils.copy_data_to_device(
        temp_storage_location,
        device_constants.GALLERY_DATA,
        env.controller,
    )

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    user_data_generation.clear_device_storage(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)

    if file_utils.check_file_or_folder_exists(
        target=self.params["file_name"],
        base_path=device_constants.DOWNLOAD_DATA,
        env=env.controller,
    ):
      return 1.0

    return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    receipt_image, _ = receipt_generator.create_receipt()
    return {
        "receipt_image": receipt_image,
        "file_name": (
            "receipt_"
            + user_data_generation.generate_random_file_name()
            + ".jpg"
        ),
    }
  
class GUIMChangePicture(SaveCopyOfReceiptTaskEval):
  schema = {}

  complexity = 3
  
  template = (
    'There is a picture named graphic.png located at the folder GUIM within sdk_gphone_x86_64 storage area.'
    'Use the Gallery app to open the image.'
    'Adjust the picture so that the green rectangle '
    'is at the top and the red circle is on the right side.'
    'After setting, do not save the picture. '
    'Just leave the canvas on-screen so the user can see the result. '
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.push_file_to_emulator(_LOCAL_SIMPLE_DRAW_GUIM, remote_path=_REMOTE_SDCARD, env=env.controller)


  def is_successful(self, env: interface.AsyncEnv) -> float:

    super().is_successful(env)
    
    goal = self.template
    task_key = 'GUIMChangePicture'
    
    ply = VerifyPolicy({})
    result = ply.verify_locating_interface(
        goal=goal,
        task_key=task_key,
        last_screenshot=self.last_screenshot,
    )
    return 1.0 if result else 0.0

