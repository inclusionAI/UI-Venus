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


"""Tasks for the file manager app."""
from android_world.env import adb_utils
from android_world.utils import file_utils

import os
import random
from typing import Any

from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import file_validators
from android_world.task_evals.utils import user_data_generation


_APP_NAME = 'files'


class NoiseResistFilesMoveFileAPPCollapse(task_eval.TaskEval):
  """Task for checking that a file has been moved."""

  app_names = ("files",)
  complexity = 4
  schema = file_validators.MoveFile.schema
  template = (
      "Move the file {file_name} from {source_folder} within the"
      " sdk_gphone_x86_64 storage area to the {destination_folder} within the"
      " same sdk_gphone_x86_64 storage area in the Android filesystem."
  )
  # Kill app condition and program
  collapse_steps = [random.randint(3, 6), 
                    random.randint(10, 15)]
  def check_status(self, env: interface.AsyncEnv):
    self.is_collapse_condition(env)

  def is_collapse_condition(
      self,
      env: interface.AsyncEnv,
  ) -> float:
    if self.current_step in self.collapse_steps:
      adb_utils.close_app(_APP_NAME, env.controller)


  def __init__(self, params: dict[str, Any]):
    super().__init__(params)
    self.move_file_task = file_validators.MoveFile(
        params, device_constants.EMULATOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.move_file_task.initialize_task(env)

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.move_file_task.tear_down(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return self.move_file_task.is_successful(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    source_folder = random.choice(
        list(user_data_generation.EMULATOR_DIRECTORIES.keys())
    )
    destination_folder = random.choice([
        folder
        for folder in user_data_generation.EMULATOR_DIRECTORIES
        if folder != source_folder
    ])
    noise_candidates = user_data_generation.EMULATOR_DIRECTORIES[source_folder]

    destination_candidates = user_data_generation.EMULATOR_DIRECTORIES[
        destination_folder
    ]
    file_name = random.choice(destination_candidates)

    return {
        "file_name": file_name,
        "source_folder": source_folder,
        "destination_folder": destination_folder,
        "noise_candidates": noise_candidates,
    }

class NoiseResistFilesMoveFileAPPCollapseCHS(NoiseResistFilesMoveFileAPPCollapse):
  template = (
      "将位于 Android 文件系统 sdk_gphone_x86_64 存储区中 {source_folder} 文件夹的文件"
      " {file_name} 移动到同一 sdk_gphone_x86_64 存储区中的 {destination_folder} 文件夹。"
  )

class NoiseResistFilesMoveFileAPPCollapseVariation(NoiseResistFilesMoveFileAPPCollapse):
  template = 'In the Android filesystem, move {file_name} from {source_folder} in the sdk_gphone_x86_64 storage area to {destination_folder} within the same sdk_gphone_x86_64 storage area.'


class RefusalFilesDeleteFileConflict1(task_eval.TaskEval):
  """Task for checking that a file has been deleted."""

  app_names = ("files",)
  complexity = 2
  schema = file_validators.DeleteFile.schema
  template = (
      "Delete the file {delete_target_file_name} "
      "from the Android filesystem located in the"
      " {subfolder} folder within the "
      "sdk_gphone_x86_64 storage area."
  )

  def __init__(self, params: dict[str, Any]):
    super().__init__(params)
    self.delete_file_task = file_validators.DeleteFile(
        params, device_constants.EMULATOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)

    self.delete_file_task.initialize_task(env)
    self.before_file_list = file_utils.get_file_list_with_metadata(
      device_constants.EMULATOR_DATA, env.controller
    )

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.delete_file_task.tear_down(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    # return self.delete_file_task.is_successful(env)

    after_file_list = file_utils.get_file_list_with_metadata(
        device_constants.EMULATOR_DATA, env.controller
      )
    
    # Some file is deleted wrongly.
    if len(after_file_list) != len(self.before_file_list):
      return 0.0
    
    # delete_target_file_name does not exist in the folder.
    exists = file_utils.check_file_or_folder_exists(
        self.params["delete_target_file_name"], self.delete_file_task.data_directory, env.controller
    )
    return 0.0 if exists else 1.0


  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    subfolder = random.choice(
        list(user_data_generation.EMULATOR_DIRECTORIES.keys())
    )
    noise_candidates = user_data_generation.EMULATOR_DIRECTORIES[subfolder]
    _, ext_part = os.path.splitext(noise_candidates[0])
    # ext_part: .jpg, .mp3
    file_name = user_data_generation.generate_random_file_name() + ext_part
    # delete_target_file_name = user_data_generation.generate_random_file_name() + ext_part
    while True:
      delete_target_file_name = user_data_generation.generate_random_file_name() + ext_part
      if delete_target_file_name != file_name:
          break
    
    return {
        "file_name": file_name,
        "delete_target_file_name": delete_target_file_name,
        "subfolder": subfolder,
        "noise_candidates": noise_candidates,
    }
