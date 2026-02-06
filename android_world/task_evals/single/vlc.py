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

"""Tasks for VLC player."""

import os
import random
from typing import Any

import yaml
from absl import logging
from android_world.env import interface
from android_world.env.setup_device import apps
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import sqlite_utils
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils
from android_world.policy.verification import VerifyPolicy
import re

from pathlib import Path
import sys
# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from config import config


_LOCAL_GUI_BROWSING_VIDEO = config.get('local_paths.gui_browsing_video')
_REMOTE_GUI_BROWSING = config.get('remote_paths.gui_browsing')


_DB_PATH = '/data/data/org.videolan.vlc/app_db/vlc_media.db'
_APP_NAME = 'vlc'




def _get_playlist_info_query() -> str:
  """Gets query for fetching playlists and their associated files."""
  return """
    SELECT
      Playlist.name AS playlist_name,
      Media.filename AS media_file_name,
      PlaylistMediaRelation.position AS order_in_playlist
    FROM
      PlaylistMediaRelation
    INNER JOIN Playlist ON Playlist.id_playlist = PlaylistMediaRelation.playlist_id
    INNER JOIN Media ON Media.id_media = PlaylistMediaRelation.media_id
    ORDER BY
      Playlist.name,
      PlaylistMediaRelation.position;
    """


def _clear_playlist_dbs(env: interface.AsyncEnv) -> None:
  """Clears all DBs related to playlists."""
  sqlite_utils.delete_all_rows_from_table('Playlist', _DB_PATH, env, _APP_NAME)
  sqlite_utils.delete_all_rows_from_table('Media', _DB_PATH, env, _APP_NAME)
  sqlite_utils.delete_all_rows_from_table(
      'PlaylistMediaRelation', _DB_PATH, env, _APP_NAME
  )


def _get_playlist_file_info(
    env: interface.AsyncEnv,
) -> list[sqlite_schema_utils.PlaylistInfo]:
  """Executes join query to fetch playlist file info."""
  with env.controller.pull_file(_DB_PATH, timeout_sec=3) as local_db_directory:
    local_db_path = file_utils.convert_to_posix_path(
        local_db_directory, os.path.split(_DB_PATH)[1]
    )
    return sqlite_utils.execute_query(
        _get_playlist_info_query(),
        local_db_path,
        sqlite_schema_utils.PlaylistInfo,
    )


class _VLC(task_eval.TaskEval):

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    user_data_generation.clear_internal_storage(env)
    file_utils.clear_directory(apps.VlcApp.videos_path, env.controller)

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    user_data_generation.clear_internal_storage(env)
    file_utils.clear_directory(apps.VlcApp.videos_path, env.controller)


class VlcCreatePlaylist(_VLC):
  """Task to create a playlist in VLC."""

  app_names = ['vlc']
  complexity = 2.8
  schema = {
      'type': 'object',
      'properties': {
          'playlist_name': {'type': 'string'},
          'files': {
              'type': 'array',
              'items': {'type': 'string'},
          },
      },
      'required': ['playlist_name', 'files'],
  }
  template = ''  # We'll directly use goal.
  HTML = ''  # Implementation overrides this.

  @property
  def goal(self) -> str:
    files = ', '.join(self.params['files'])
    playlist_name = self.params['playlist_name']
    return (
        f'Create a playlist titled "{playlist_name}" with the following files'
        f' in VLC (located in Internal Memory/VLCVideos), in order: {files}'
    )

  def setup_files(self, env: interface.AsyncEnv):
    for file in self.params['files']:
      user_data_generation.write_video_file_to_device(
          file,
          apps.VlcApp.videos_path,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )
    for file in self.params['noise_files']:
      user_data_generation.write_video_file_to_device(
          file,
          apps.VlcApp.videos_path,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )
  def setup_files_custom_path(self, env: interface.AsyncEnv, custom_path: str):
    for file in self.params['files']:
      user_data_generation.write_video_file_to_device(
          file,
          custom_path,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )

    for file in self.params['noise_files']:
      user_data_generation.write_video_file_to_device(
          file,
          custom_path,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )
  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    _clear_playlist_dbs(env)
    self.setup_files(env)

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    _clear_playlist_dbs(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    actual = _get_playlist_file_info(env)

    return float(
        sqlite_validators.verify_playlist(
            actual, self.params['playlist_name'], self.params['files']
        )
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    playlist_name = _generate_playlist_name()
    num_files = random.randint(2, 5)
    files = [generate_file_name() for _ in range(num_files)]
    return {
        'playlist_name': playlist_name,
        'files': files,
        'noise_files': [generate_file_name() for _ in range(num_files)],
    }



class FuncAssistVlcLocateOneFunctionality1(_VLC):
  """Task to Locate in VLC."""

  app_names = ['vlc']
  complexity = 2
  schema = {
      'type': 'object',
      'properties': {
          'playlist_name': {'type': 'string'},
          'files': {
              'type': 'array',
              'items': {'type': 'string'},
          },
      },
      'required': ['playlist_name', 'files'],
  }
  template = ''  # We'll directly use goal.
  HTML = ''  # Implementation overrides this.

  @property
  def goal(self) -> str:
    files = ', '.join(self.params['files'])
    playlist_name = self.params['playlist_name']
    return (
        f'I want to set video screen orientation of VLC app as Landscape reverse.'
        'Where I can do this? Locate the inferface, simply show me where to find it—don’t apply the change.'
    )

  def setup_files(self, env: interface.AsyncEnv):
    for file in self.params['files']:
      user_data_generation.write_video_file_to_device(
          file,
          apps.VlcApp.videos_path,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )
    for file in self.params['noise_files']:
      user_data_generation.write_video_file_to_device(
          file,
          apps.VlcApp.videos_path,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    _clear_playlist_dbs(env)
    self.setup_files(env)

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    _clear_playlist_dbs(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    """Check if the task is successful by verifying the screenshot.
    
    Args:
      env: The environment interface.
      
    Returns:
      1.0 if successful (screenshot shows the correct interface), 0.0 otherwise.
    """
    super().is_successful(env)
    
   
    
    goal = self.goal
    task_key = 'FuncAssistVlcLocateOneFunctionality1'
    
    try:
      ply = VerifyPolicy({})
      result = ply.verify_locating_interface(
          goal=goal,       
          task_key=task_key,
          last_screenshot=self.last_screenshot,
      )
      return 1.0 if result else 0.0
    except Exception as e:
      logging.warning(
          'Error during verification for FuncAssistVlcLocateOneFunctionality1: %s', e
      )
      return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    playlist_name = _generate_playlist_name()
    num_files = random.randint(2, 5)
    files = [generate_file_name() for _ in range(num_files)]
    return {
        'playlist_name': playlist_name,
        'files': files,
        'noise_files': [generate_file_name() for _ in range(num_files)],
    }


class FuncAssistVlcLocateOneFunctionality1CHS(FuncAssistVlcLocateOneFunctionality1):
  @property
  def goal(self) -> str:
    return (
        '我想把 VLC 播放器的画面方向设为“Landscape reverse”。'
'请告诉我具体在哪个界面可以设置，定位到那个界面，且不需要帮我设置。'
    )

class FuncAssistVlcLocateOneFunctionality1Variation(FuncAssistVlcLocateOneFunctionality1):
  @property
  def goal(self) -> str:
    return (
        f'I’d like to lock VLC’s video orientation '
'to “Landscape (reverse)”.'
'Where can I find this option? '
'Just point me to the exact screen—no need to change it for me.'
    )
  
class FuncAssistVlcExplainOneFunctionality1(_VLC):
  """Task to Locate in VLC."""

  app_names = ['vlc']
  complexity = 4
  schema = {
      'type': 'object',
      'properties': {
          'playlist_name': {'type': 'string'},
          'files': {
              'type': 'array',
              'items': {'type': 'string'},
          },
      },
      'required': ['playlist_name', 'files'],
  }
  template = ''  # We'll directly use goal.
  HTML = ''  # Implementation overrides this.

  @property
  def goal(self) -> str:
    files = ', '.join(self.params['files'])
    playlist_name = self.params['playlist_name']
    return (
        f'In the VLC app, if I want to adjust the audio, '
        'what settings can I change? Please tell me briefly.'
    )

  def setup_files(self, env: interface.AsyncEnv):
    for file in self.params['files']:
      user_data_generation.write_video_file_to_device(
          file,
          apps.VlcApp.videos_path,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )
    for file in self.params['noise_files']:
      user_data_generation.write_video_file_to_device(
          file,
          apps.VlcApp.videos_path,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    _clear_playlist_dbs(env)
    self.setup_files(env)

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    _clear_playlist_dbs(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    """Check if the task is successful by verifying the text prediction.
    
    Args:
      env: The environment interface.
      
    Returns:
      1.0 if successful (text prediction matches ground truth), 0.0 otherwise.
    """
    super().is_successful(env)
    
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.goal
    prediction = env.interaction_cache
    gt_answer = "View information, add to playlist, set as ringtone, remove, stop after this track, share, browse parent."
    
    try:
      ply = VerifyPolicy({})
      result = ply.verify_textual_prediction(
          goal=goal,
          prediction=prediction,
          gt_answer=gt_answer,
          judgement_type='funcassist'
      )
      return 1.0 if result else 0.0
    except Exception as e:
      logging.warning(
          'Error during verification for FuncAssistVlcExplainOneFunctionality1: %s', e
      )
      return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    playlist_name = _generate_playlist_name()
    num_files = random.randint(2, 5)
    files = [generate_file_name() for _ in range(num_files)]
    return {
        'playlist_name': playlist_name,
        'files': files,
        'noise_files': [generate_file_name() for _ in range(num_files)],
    }



class GUIBrowsingWatchVideo1(_VLC):
  """Task to Locate in VLC."""

  app_names = ['vlc']
  complexity = 5
  schema = {
      'type': 'object',
      'properties': {
          'playlist_name': {'type': 'string'},
          'files': {
              'type': 'array',
              'items': {'type': 'string'},
          },
      },
      'required': ['playlist_name', 'files'],
  }
  template = ''  # We'll directly use goal.
  HTML = ''  # Implementation overrides this.

  answer_list = [9,8]

  @property
  def goal(self) -> str:
    
    return (
        'There are some videos located at the folder GUIBrowsing/Video '
        'within the sdk_gphone_x86_64 storage area. '
        'Open the file cowboy.mp4 in the Movies folder and watch it. '
        'Identify the exact second (timestamp) at which the cowboy falls off the horse. '
        'Report only the integer second; '
        'output strictly in the format "X s" where X is the integer, '
        'where X is an integer.'
    )
  
  def is_successful(self, env: interface.AsyncEnv) -> float:
    if not env.interaction_cache:
        return 0.0

    m = re.search(r'(\d+) s', env.interaction_cache)
    if not m:
        return 0.0

    ts = int(m.group(1))

    return 1.0 if ts in {8, 9} else 0.0
  

  def setup_files(self, env: interface.AsyncEnv):
    for file in self.params['files']:
      user_data_generation.write_video_file_to_device(
          file,
          apps.VlcApp.videos_path,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )
    for file in self.params['noise_files']:
      user_data_generation.write_video_file_to_device(
          file,
          apps.VlcApp.videos_path,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    _clear_playlist_dbs(env)
    self.setup_files(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_VIDEO, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)


  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    _clear_playlist_dbs(env)



  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    playlist_name = _generate_playlist_name()
    num_files = random.randint(2, 5)
    files = [generate_file_name() for _ in range(num_files)]
    return {
        'playlist_name': playlist_name,
        'files': files,
        'noise_files': [generate_file_name() for _ in range(num_files)],
    }


class GUIBrowsingWatchVideo2(_VLC):
  """Task to Locate in VLC."""

  app_names = ['vlc']
  complexity = 5
  schema = {
      'type': 'object',
      'properties': {
          'playlist_name': {'type': 'string'},
          'files': {
              'type': 'array',
              'items': {'type': 'string'},
          },
      },
      'required': ['playlist_name', 'files'],
  }
  template = ''  # We'll directly use goal.
  HTML = ''  # Implementation overrides this.

  @property
  def goal(self) -> str:
    return (
        f'There are some videos located at '
        'the folder GUIBrowsing/Video within sdk_gphone_x86_64 storage area. ' 
        'Open the video with Gallery.'
        'After the cowboy falls off his horse, '
        'pause the video and display the frozen frame on the screen.'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    _clear_playlist_dbs(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_VIDEO, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)
    # self.setup_files(env)

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    _clear_playlist_dbs(env)


  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    
    
    
    goal = self.goal
    task_key = 'GUIBrowsingVlcWatchVideo2'
    
    try:
      ply = VerifyPolicy({})
      result = ply.verify_locating_interface_multiple_gt_images(
          goal=goal,
          task_key=task_key,
          last_screenshot=self.last_screenshot,
      )
      return 1.0 if result else 0.0
    except Exception as e:
      logging.warning(
          'Error during verification for GUIBrowsingVlcWatchVideo2: %s', e
      )
      return 0.0
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}


# 6 - 10 s
class GUIBrowsingWatchVideo3(_VLC):
  """Task to Locate in VLC."""

  app_names = ['vlc']
  complexity = 5
  schema = {
      'type': 'object',
      'properties': {
          'playlist_name': {'type': 'string'},
          'files': {
              'type': 'array',
              'items': {'type': 'string'},
          },
      },
      'required': ['playlist_name', 'files'],
  }
  template = ''  # We'll directly use goal.
  HTML = ''  # Implementation overrides this.
  answer = '6-10 s'
  answer_in_list = [6,10]

  @property
  def goal(self) -> str:
    return (
        'There are some videos in the folder GUIBrowsing/Video under the internal storage of the device “sdk_gphone_x86_64”.'
        'Open the file paddleboard.mp4 in the Movies folder and watch it. '
        'Find the continuous time segment (in seconds) in which the man is seated on the paddleboard.'
        'Give the answer strictly in the format start-end s, where both numbers are whole seconds.'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    _clear_playlist_dbs(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_VIDEO, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    _clear_playlist_dbs(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    m = re.search(r'(\d+)-(\d+) s', env.interaction_cache)
    task_done = (
        m is not None and
        int(m.group(1)) in (5, 6) and
        int(m.group(2)) in (9, 10)
    )

    if task_done:
      return 1.0
    else:
      return 0.0
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}



class GUIBrowsingWatchVideo4(_VLC):
  """Task to Locate in VLC."""

  app_names = ['vlc']
  complexity = 5
  schema = {
      'type': 'object',
      'properties': {
          'playlist_name': {'type': 'string'},
          'files': {
              'type': 'array',
              'items': {'type': 'string'},
          },
      },
      'required': ['playlist_name', 'files'],
  }
  template = ''  # We'll directly use goal.
  HTML = ''  # Implementation overrides this.

  @property
  def goal(self) -> str:
    return (
      'There are some videos located at the folder GUIBrowsing/Video within sdk_gphone_x86_64 storage area. '
      'Open the file anime_fireworks.mp4 in the Movies folder with Gallery and watch it. '
      'Locate the frame where fireworks burst on the bridge, '
      'pause the video, display the frame on the screen.'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    _clear_playlist_dbs(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_VIDEO, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    _clear_playlist_dbs(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    
    goal = self.goal
    task_key = 'GUIBrowsingVlcWatchVideo4'
    
    try:
      ply = VerifyPolicy({})
      result = ply.verify_locating_interface_multiple_gt_images(
          goal=goal,
          task_key=task_key,
          last_screenshot=self.last_screenshot,
      )
      return 1.0 if result else 0.0
    except Exception as e:
      logging.warning(
          'Error during verification for GUIBrowsingVlcWatchVideo4: %s', e
      )
      return 0.0
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}




class GUIBrowsingWatchVideo5(_VLC):
  """Task to Locate in VLC."""

  app_names = ['vlc']
  complexity = 5
  schema = {
      'type': 'object',
      'properties': {
          'playlist_name': {'type': 'string'},
          'files': {
              'type': 'array',
              'items': {'type': 'string'},
          },
      },
      'required': ['playlist_name', 'files'],
  }
  template = ''  # We'll directly use goal.
  HTML = ''  # Implementation overrides this.

  @property
  def goal(self) -> str:
    return (
        'There are some videos located at the folder GUIBrowsing/Video '
        'within sdk_gphone_x86_64 storage area. '
        'Open the file astronaut.mp4 in the Movies folder with Gallery and watch it. '
        'Locate the frame where the puppy is biting the first tennis ball, '
        'pause the video, display the frame on the screen.'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    _clear_playlist_dbs(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_VIDEO, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)
    

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    _clear_playlist_dbs(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    
    # Get the last screenshot path from instance
    if not self.last_screenshot_path:
      return 0.0
    
    goal = self.goal
    task_key = 'GUIBrowsingVlcWatchVideo5'
    
    try:
      ply = VerifyPolicy({})
      result = ply.verify_locating_interface_multiple_gt_images(
          goal=goal,
          task_key=task_key,
          last_screenshot=self.last_screenshot,
      ) 
      return 1.0 if result else 0.0
    except Exception as e:
      logging.warning(
          'Error during verification for GUIBrowsingVlcWatchVideo5: %s', e
      )
      return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}

class GUIBrowsingWatchVideo6(_VLC):
  """Task to Locate in VLC."""

  app_names = ['vlc']
  complexity = 5
  schema = {
      'type': 'object',
      'properties': {
          'playlist_name': {'type': 'string'},
          'files': {
              'type': 'array',
              'items': {'type': 'string'},
          },
      },
      'required': ['playlist_name', 'files'],
  }
  template = ''  # We'll directly use goal.
  HTML = ''  # Implementation overrides this.
  @property
  def goal(self) -> str:
    return (
        'There are some videos located at the folder GUIBrowsing/Video '
        'within sdk_gphone_x86_64 storage area. '
        'Open the file astronaut.mp4 in the Movies folder and watch it. '
        'Tell me the maximum number of tennis balls the puppy has in its mouth '
        'at any point in the video.'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    _clear_playlist_dbs(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_VIDEO, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    _clear_playlist_dbs(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)    
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.goal
    prediction = env.interaction_cache
    gt_answer = '2'
    
    try:
      ply = VerifyPolicy({})
      result = ply.verify_textual_prediction(
          goal=goal,
          prediction=prediction,
          gt_answer=gt_answer,
          judgement_type='guibrowsing'
      )
      return 1.0 if result else 0.0
    except Exception as e:
      logging.warning(
          'Error during verification for GUIBrowsingWatchVideo6: %s', e
      )
      return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}


class VlcCreateTwoPlaylists(task_eval.TaskEval):
  """Task to create two playlists in VLC."""

  app_names = ['vlc']
  complexity = 4.8
  schema = {
      'type': 'object',
      'properties': {
          'playlist_name1': {'type': 'string'},
          'files1': {
              'type': 'array',
              'items': {'type': 'string'},
          },
          'playlist_name2': {'type': 'string'},
          'files2': {
              'type': 'array',
              'items': {'type': 'string'},
          },
      },
      'required': ['playlist_name1', 'files1', 'playlist_name2', 'files2'],
  }
  template = ''  # Directly use goal.

  def __init__(self, params: dict[str, Any]):
    super().__init__(params)
    self.task1_params = {
        'playlist_name': params['playlist_name1'],
        'files': params['files1'],
        'noise_files': params['noise_files1'],
    }
    self.task2_params = {
        'playlist_name': params['playlist_name2'],
        'files': params['files2'],
        'noise_files': params['noise_files2'],
    }
    self.task1 = VlcCreatePlaylist(self.task1_params)
    self.task2 = VlcCreatePlaylist(self.task2_params)

  @property
  def goal(self) -> str:
    goal1 = (
        f'Create a playlist titled "{self.params["playlist_name1"]}" with the'
        ' following files in VLC (located in Internal Memory/VLCVideos), in'
        f' order: {", ".join(self.params["files1"])}'
    )
    goal2 = (
        f'create a playlist titled "{self.params["playlist_name2"]}" with the'
        f' following files in VLC, in order: {", ".join(self.params["files2"])}'
    )
    return f'{goal1}. And then, {goal2}.'

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    self.task1.initialize_task(env)
    self.task2.setup_files(env)  # Don't want to clear db.

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    self.task1.tear_down(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (self.task1.is_successful(env) + self.task2.is_successful(env)) / 2

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    playlist1_params = VlcCreatePlaylist.generate_random_params()
    playlist2_params = VlcCreatePlaylist.generate_random_params()
    return {
        'playlist_name1': playlist1_params['playlist_name'],
        'files1': playlist1_params['files'],
        'noise_files1': playlist1_params['noise_files'],
        'playlist_name2': playlist2_params['playlist_name'],
        'files2': playlist2_params['files'],
        'noise_files2': playlist2_params['noise_files'],
    }


class MultiRoundVlcCreateTwoPlaylistsReverse(task_eval.TaskEval):
  """Task to create two playlists in VLC."""

  app_names = ['vlc']
  complexity = 7
  schema = {
      'type': 'object',
      'properties': {
          'playlist_name1': {'type': 'string'},
          'files1': {
              'type': 'array',
              'items': {'type': 'string'},
          },
          'playlist_name2': {'type': 'string'},
          'files2': {
              'type': 'array',
              'items': {'type': 'string'},
          },
      },
      'required': ['playlist_name1', 'files1', 'playlist_name2', 'files2'],
  }
  template = ''  # Directly use goal.
  playlist_1st_verified = False

  round = 0
  max_round = 2

  def __init__(self, params: dict[str, Any]):
    super().__init__(params)
    self.task1_params = {
        'playlist_name': params['playlist_name1'],
        'files': params['files1'],
        'noise_files': params['noise_files1'],
    }
    self.task2_params = {
        'playlist_name': params['playlist_name2'],
        'files': params['files2'],
        'noise_files': params['noise_files2'],
    }
    self.task1 = VlcCreatePlaylist(self.task1_params)
    self.task2 = VlcCreatePlaylist(self.task2_params)

  def check_status(self, env: interface.AsyncEnv):
    # first round task not finished yet, keep checking.
    if self.playlist_1st_verified == False:
      task1_success = self.task1.is_successful(env)
      if task1_success:
        self.playlist_1st_verified = task1_success
        self.start_step_2ndround = self.current_step + 3
    
    # first round task finished, move to round 2
    if self.playlist_1st_verified and self.start_step_2ndround == self.current_step:
      self.round = 1

  @property
  def goal(self) -> str:
    if self.round == 0:
      goal1 = (
          f'Create a playlist titled "{self.params["playlist_name1"]}" with the'
          ' following files in VLC (located in Internal Memory/VLCVideos), in'
          f' order: {", ".join(self.params["files1"])}'
      )
      goal2 = (
          f'create a playlist titled "{self.params["playlist_name2"]}" with the'
          f' following files in VLC, in order: {", ".join(self.params["files2"])}'
      )
      return f'{goal1}. And then, {goal2}.'
    elif self.round == 1:
      return "Undo all playlist creation actions you performed."

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    
    self.task1.initialize_task(env)
    
    self.task2.setup_files(env)  # Do not need to clear db. Just create noise files of task2.

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    self.task1.tear_down(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    actual = _get_playlist_file_info(env)
    if sqlite_validators.playlist_exist(actual, self.params['playlist_name1']):
        return 0.0
    if sqlite_validators.playlist_exist(actual, self.params['playlist_name2']):
        return 0.0
    if not self.playlist_1st_verified:
      return 0.0

    return 1.0
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    playlist1_params = VlcCreatePlaylist.generate_random_params()
    playlist2_params = VlcCreatePlaylist.generate_random_params()
    return {
        'playlist_name1': playlist1_params['playlist_name'],
        'files1': playlist1_params['files'],
        'noise_files1': playlist1_params['noise_files'],
        'playlist_name2': playlist2_params['playlist_name'],
        'files2': playlist2_params['files'],
        'noise_files2': playlist2_params['noise_files'],
    }


#### Synthetic data ############################################################

def generate_file_name() -> str:
  """Generates a more realistic and descriptive video file name."""
  prefixes = [
      'clip',
      'footage',
      'scene',
      'recording',
      'highlight',
      'moment',
      'episode',
  ]
  suffixes = [
      '',
      'HD',
      '4K',
      'raw',
      'export',
  ]
  prefix = random.choice(prefixes)
  suffix = random.choice(suffixes)
  num = str(random.randint(1, 99))
  name = f'{prefix}_{num}_{suffix}.mp4'
  return user_data_generation.generate_modified_file_name(name)


def _generate_playlist_name() -> str:
  """Generates realistic and descriptive playlist names."""
  themes = [
      'Adventure',
      'Comedy',
      'Daily Routines',
      'Documentary Insights',
      'Epic Moments',
      'Family Gatherings',
      'Fitness Challenges',
      'Gaming Sessions',
      'How To',
      'Mystery and Thrills',
      'Recipe Collection',
      'Road Trips',
      'Summer Highlights',
      'Tech Reviews',
      'Travel Guide',
      'Ultimate Fails',
  ]
  qualifiers = [
      'Essentials',
      'Favorites',
      'Marathon',
      'Playlist',
      'Series',
      'Specials',
      'Ultimate Collection',
  ]
  # Select a random theme and qualifier
  theme = random.choice(themes)
  qualifier = random.choice(qualifiers)
  # Form the playlist name
  return f'{theme} {qualifier}'



class NoiseResistVlcCreatePlaylistWithOrientation(_VLC):
  """Task to create a playlist in VLC."""

  landscape_step = 6
  portrait_step = 12
  def check_status(self, env: interface.AsyncEnv):
    self.is_o_condition(env)

  # original code
  app_names = ['vlc']
  complexity = 2.8
  schema = {
      'type': 'object',
      'properties': {
          'playlist_name': {'type': 'string'},
          'files': {
              'type': 'array',
              'items': {'type': 'string'},
          },
      },
      'required': ['playlist_name', 'files'],
  }
  template = ''  # We'll directly use goal.
  HTML = ''  # Implementation overrides this.

  @property
  def goal(self) -> str:
    files = ', '.join(self.params['files'])
    playlist_name = self.params['playlist_name']
    return (
        f'Create a playlist titled "{playlist_name}" with the following files'
        f' in VLC (located in Internal Memory/VLCVideos), in order: {files}'
    )

  def setup_files(self, env: interface.AsyncEnv):
    for file in self.params['files']:
      user_data_generation.write_video_file_to_device(
          file,
          apps.VlcApp.videos_path,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )
    for file in self.params['noise_files']:
      user_data_generation.write_video_file_to_device(
          file,
          apps.VlcApp.videos_path,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    _clear_playlist_dbs(env)
    self.setup_files(env)

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    _clear_playlist_dbs(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    actual = _get_playlist_file_info(env)
    return float(
        sqlite_validators.verify_playlist(
            actual, self.params['playlist_name'], self.params['files']
        )
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    playlist_name = _generate_playlist_name()
    num_files = random.randint(2, 5)
    files = [generate_file_name() for _ in range(num_files)]
    return {
        'playlist_name': playlist_name,
        'files': files,
        'noise_files': [generate_file_name() for _ in range(num_files)],
    }


from android_world.env import adb_utils

class NoiseResistVlcCreateTwoPlaylistsWithCallandCollapse(task_eval.TaskEval):
  """Task to create two playlists in VLC."""


  call_coming_steps = [15,20]
  collapse_steps = [10]
  def check_status(self, env: interface.AsyncEnv):
    self.is_call_condition(env)
    self.is_collapse_condition(env)

  def is_collapse_condition(
      self,
      env: interface.AsyncEnv,
  ) -> float:
    if self.current_step in self.collapse_steps:
      adb_utils.close_app(_APP_NAME, env.controller)



  # Original code
  app_names = ['vlc']
  complexity = 5.8
  schema = {
      'type': 'object',
      'properties': {
          'playlist_name1': {'type': 'string'},
          'files1': {
              'type': 'array',
              'items': {'type': 'string'},
          },
          'playlist_name2': {'type': 'string'},
          'files2': {
              'type': 'array',
              'items': {'type': 'string'},
          },
      },
      'required': ['playlist_name1', 'files1', 'playlist_name2', 'files2'],
  }
  template = ''  # Directly use goal.

  def __init__(self, params: dict[str, Any]):
    super().__init__(params)
    self.task1_params = {
        'playlist_name': params['playlist_name1'],
        'files': params['files1'],
        'noise_files': params['noise_files1'],
    }
    self.task2_params = {
        'playlist_name': params['playlist_name2'],
        'files': params['files2'],
        'noise_files': params['noise_files2'],
    }
    self.task1 = VlcCreatePlaylist(self.task1_params)
    self.task2 = VlcCreatePlaylist(self.task2_params)

  @property
  def goal(self) -> str:
    goal1 = (
        f'Create a playlist titled "{self.params["playlist_name1"]}" with the'
        ' following files in VLC (located in Internal Memory/VLCVideos), in'
        f' order: {", ".join(self.params["files1"])}'
    )
    goal2 = (
        f'create a playlist titled "{self.params["playlist_name2"]}" with the'
        f' following files in VLC, in order: {", ".join(self.params["files2"])}'
    )
    return f'{goal1}. And then, {goal2}.'

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    self.task1.initialize_task(env)
    self.task2.setup_files(env)  # Don't want to clear db.

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    self.task1.tear_down(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return (self.task1.is_successful(env) + self.task2.is_successful(env)) / 2

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    playlist1_params = VlcCreatePlaylist.generate_random_params()
    playlist2_params = VlcCreatePlaylist.generate_random_params()
    return {
        'playlist_name1': playlist1_params['playlist_name'],
        'files1': playlist1_params['files'],
        'noise_files1': playlist1_params['noise_files'],
        'playlist_name2': playlist2_params['playlist_name'],
        'files2': playlist2_params['files'],
        'noise_files2': playlist2_params['noise_files'],
    }

class NoiseResistVlcCreateTwoPlaylistsWithCallandCollapseCHS(NoiseResistVlcCreateTwoPlaylistsWithCallandCollapse):
  @property
  def goal(self) -> str:

    goal1 = (
    f'在VLC中创建一个名为"{self.params["playlist_name1"]}"的播放列表，'
    f'包含以下文件（位于内存/VLCVideos文件夹中），按顺序排列：'
    f'{", ".join(self.params["files1"])}'
    )
    goal2 = (
        f'在VLC中创建一个名为"{self.params["playlist_name2"]}"的播放列表，'
        f'包含以下文件，按顺序排列：{", ".join(self.params["files2"])}'
    )


    return f'{goal1}。接着，{goal2}。'
  
class NoiseResistVlcCreateTwoPlaylistsWithCallandCollapseVariation(NoiseResistVlcCreateTwoPlaylistsWithCallandCollapse):
  @property
  def goal(self) -> str:
    goal1 = (
        f'In VLC, create a playlist named "{self.params["playlist_name1"]}" '
        f'containing the following media files from Internal Memory/VLCVideos, '
        f'arranged in this order: {", ".join(self.params["files1"])}'
    )
    goal2 = (
        f'In VLC, create a playlist named "{self.params["playlist_name2"]}" '
        f'with these files in the specified order: {", ".join(self.params["files2"])}'
    )

    return f'{goal1}. Next, {goal2}.'

class BrowsecompVlcFindVlcAPP1(_VLC):
  """Task to find VLC app."""

  app_names = ['vlc']
  complexity = 8
  schema = {
      'type': 'object',
      'properties': {
          'playlist_name': {'type': 'string'},
          'files': {
              'type': 'array',
              'items': {'type': 'string'},
          },
      },
      'required': ['playlist_name', 'files'],
  }
  template = (
        'Among the apps Markor, Joplin, Tasks, Broccoli, Simple Calendar Pro, VLC, and Audio Recorder, '
        'identify the one whose settings screen contains '
        'four exclamation-mark buttons, '
        'each used to explain the meaning of the setting it is next to. '
        'Provide the app name. '
    )
  HTML = ''  # Implementation overrides this.


  @property
  def goal(self) -> str:
    files = ', '.join(self.params['files'])
    playlist_name = self.params['playlist_name']
    return self.template

  def setup_files(self, env: interface.AsyncEnv):
    for file in self.params['files']:
      user_data_generation.write_video_file_to_device(
          file,
          apps.VlcApp.videos_path,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )
    for file in self.params['noise_files']:
      user_data_generation.write_video_file_to_device(
          file,
          apps.VlcApp.videos_path,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    _clear_playlist_dbs(env)
    self.setup_files(env)

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    _clear_playlist_dbs(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)    
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = 'VLC'
    
    try:
      ply = VerifyPolicy({})
      result = ply.verify_textual_prediction(
          goal=goal,
          prediction=prediction,
          gt_answer=gt_answer,
          judgement_type='guibrowsing'
      )
      return 1.0 if result else 0.0
    except Exception as e:
      logging.warning(
          'Error during verification for BrowsecompVlcFindVlcAPP1: %s', e
      )
      return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    playlist_name = _generate_playlist_name()
    num_files = random.randint(2, 5)
    files = [generate_file_name() for _ in range(num_files)]
    return {
        'playlist_name': playlist_name,
        'files': files,
        'noise_files': [generate_file_name() for _ in range(num_files)],
    }

class BrowsecompVlcFindVlcAPP1CHS(BrowsecompVlcFindVlcAPP1):
  template = ('在 Markor、Joplin、Tasks、Broccoli、Simple Calendar Pro、VLC 和 Audio Recorder '
              '这些应用中，找出设置界面包含四个感叹号按钮的那个应用；'
              '每个按钮用于解释其旁边的设置含义。请告诉我该应用的名称。' )

class BrowsecompVlcFindVlcAPP1Variation(BrowsecompVlcFindVlcAPP1):
  template = 'From the apps Markor, Joplin, Tasks, Broccoli, Simple Calendar Pro, VLC, and Audio Recorder, determine which one has a Settings screen containing four exclamation mark buttons, each explaining the adjacent setting. Give the name of that app.'

class GUIBrowsingRefusalVlcWatchVideo(_VLC):
  """Task to Locate in VLC."""

  app_names = ['vlc']
  complexity = 5
  schema = {
      'type': 'object',
      'properties': {
          'playlist_name': {'type': 'string'},
          'files': {
              'type': 'array',
              'items': {'type': 'string'},
          },
      },
      'required': ['playlist_name', 'files'],
  }
  template = ''  # We'll directly use goal.
  HTML = ''  # Implementation overrides this.

  @property
  def goal(self) -> str:
    files = ', '.join(self.params['files'])
    playlist_name = self.params['playlist_name']
    return (
        f'There are some videos located at '
        f'sdk_gphone_x86_64/GUIBrowsing/Video (Android file system). '
        f'Open the file coolguy.m4a in the Video folder and watch it. '
        'Tell me the exact second (timestamp) at which the boy falls off the car.'
    )

  def setup_files(self, env: interface.AsyncEnv):
    for file in self.params['files']:
      user_data_generation.write_video_file_to_device(
          file,
          apps.VlcApp.videos_path,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )
    for file in self.params['noise_files']:
      user_data_generation.write_video_file_to_device(
          file,
          apps.VlcApp.videos_path,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    _clear_playlist_dbs(env)
    self.setup_files(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_VIDEO, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    _clear_playlist_dbs(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.goal
    prediction = env.interaction_cache
    gt_answer = "There is no file named coolguy.m4a in the Video folder, so the corresponding operation cannot be performed. Please check whether the file name or file path is correct."
    
    try:
      ply = VerifyPolicy({})
      result = ply.verify_textual_prediction(
          goal=goal,
          prediction=prediction,
          gt_answer=gt_answer,
          judgement_type='refusal',
      )
      return 1.0 if result else 0.0
    except Exception as e:
      logging.warning(
          'Error during verification for FuncAssistVlcExplainOneFunctionality1: %s', e
      )
      return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    playlist_name = _generate_playlist_name()
    num_files = random.randint(2, 5)
    files = [generate_file_name() for _ in range(num_files)]
    return {
        'playlist_name': playlist_name,
        'files': files,
        'noise_files': [generate_file_name() for _ in range(num_files)],
    }
  


class GUIBrowsingVagueVlcWatchVideo(_VLC):
  """Task to Locate in VLC."""

  app_names = ['vlc']
  complexity = 5
  schema = {
      'type': 'object',
      'properties': {
          'playlist_name': {'type': 'string'},
          'files': {
              'type': 'array',
              'items': {'type': 'string'},
          },
      },
      'required': ['playlist_name', 'files'],
  }
  template = '' 
  HTML = ''  

  @property
  def goal(self) -> str:
    files = ', '.join(self.params['files'])
    playlist_name = self.params['playlist_name']
    return (
        f'There are some videos located at sdk_gphone_x86_64/GUIBrowsing/Video (Android file system). There is a video with two animals, but I forgot the name of that video.'
    )

  def setup_files(self, env: interface.AsyncEnv):
    for file in self.params['files']:
      user_data_generation.write_video_file_to_device(
          file,
          apps.VlcApp.videos_path,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )
    for file in self.params['noise_files']:
      user_data_generation.write_video_file_to_device(
          file,
          apps.VlcApp.videos_path,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    _clear_playlist_dbs(env)
    self.setup_files(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_VIDEO, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    _clear_playlist_dbs(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)   
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.goal
    prediction = env.interaction_cache
    gt_answer = 'cowboy.mp4'
    
    try:
      ply = VerifyPolicy({})
      result = ply.verify_textual_prediction(
          goal=goal,
          prediction=prediction,
          gt_answer=gt_answer,
          judgement_type='guibrowsing'
      )
      return 1.0 if result else 0.0
    except Exception as e:
      logging.warning(
          'Error during verification for GUIBrowsingVagueVlcWatchVideo: %s', e
      )
      return 0.0  

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    playlist_name = _generate_playlist_name()
    num_files = random.randint(2, 5)
    files = [generate_file_name() for _ in range(num_files)]
    return {
        'playlist_name': playlist_name,
        'files': files,
        'noise_files': [generate_file_name() for _ in range(num_files)],
    }
