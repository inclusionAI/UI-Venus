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
# Changes: Implementing tasks in VenusBench using audio recoder apps.


"""Tasks for AudioRecorder app."""
from android_world.policy.verification import VerifyPolicy
import os
import tempfile
from android_world.env import representation_utils

tempfile.tempdir = os.path.join(os.getcwd(), "temp")
from android_world.env import json_action, adb_utils, actuation
from mutagen.mp4 import MP4
import subprocess
import random
from typing import Any

from absl import logging
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import file_validators
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils


class _AudioRecorder(task_eval.TaskEval):
  """Base class for AudioRecorder tasks."""

  app_names = ("audio recorder",)


class AudioRecorderRecordAudio(_AudioRecorder):
  """Task for checking that one audio recording has been completed."""

  complexity = 1.2
  schema = {
      "type": "object",
      "properties": {},
      "required": [],
  }
  template = "Record an audio clip using Audio Recorder app and save it."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    try:
      self.before_recording = file_utils.get_file_list_with_metadata(
          device_constants.AUDIORECORDER_DATA, env.controller
      )
    except RuntimeError as exc:
      raise RuntimeError(
          "Failed to inspect recordings directory,"
          " {device_constants.AUDIORECORDER_DATA}, for Audio Recorder task."
          " Check to make sure Audio Recorder app is correctly installed."
      ) from exc

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    after_recording = [
        file
        for file in file_utils.get_file_list_with_metadata(
            device_constants.AUDIORECORDER_DATA, env.controller
        )
        if file.file_size > 0
    ]
    changed = []
    # Old recordings may be deleted and a new recording may reuse an existing
    # file name.
    for item in after_recording:
      if item not in self.before_recording:
        changed.append(item.file_name)
    logging.info("New or changed recording: %s", changed)

    # Check if a new audio recording is done by comparing directory contents
    one_new_file = len(changed) == 1
    return 1.0 if one_new_file else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    return {}


class AudioRecorderRecordAudioWithFileName(_AudioRecorder):
  """Task for checking that one audio recording with file_name has been completed."""

  complexity = 2
  schema = file_validators.CreateFile.schema
  template = (
      'Record an audio clip and save it with name "{file_name}" using Audio'
      " Recorder app."
  )

  def __init__(self, params: dict[str, Any]):
    """See base class."""
    super().__init__(params)
    self.initialized = False
    self.create_file_task = file_validators.CreateFile(
        params, device_constants.AUDIORECORDER_DATA
    )

  def _clear_audio_recorder_data(self, env: interface.AsyncEnv) -> None:
    """Clears all audio recorder data on device."""
    file_utils.clear_directory(
        device_constants.AUDIORECORDER_DATA, env.controller
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)
    self._clear_audio_recorder_data(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    file_name = self.params["file_name"]
    exists = file_utils.check_file_or_folder_exists(
        file_name + ".m4a", self.create_file_task.data_directory, env.controller
    )
    if not exists:
      logging.info("%s not found", file_name)
      return 0.0
    return 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    name = [
        "interview",
        "meeting",
        "lecture",
        "session",
        "note",
        "conference",
        "webinar",
        "workshop",
        "seminar",
        "briefing",
        "discussion",
        "talk",
        "presentation",
        "training",
        "guidance",
        "memo",
        "narration",
        "storytelling",
        "journal",
        "diary",
        "debate",
        "symposium",
        "roundtable",
        "consultation",
        "review",
    ]
    return {
        "file_name": user_data_generation.generate_modified_file_name(
            random.choice(name) + ".m4a"
        ),
        "text": "",  # Unused.
    }

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.create_file_task.tear_down(env)
    self._clear_audio_recorder_data(env)


# New Tasks
class GSATimingAudioRecorderPauseAudioRecordingTime(_AudioRecorder):
  """Task for checking that one audio recording has been completed."""

  complexity = 2
  schema = {
      "type": "object",
      "properties": {},
      "required": [],
  }
  time_gap = 2
  template = (
    'Record an audio clip in Audio Recorder, '
    'pausing at {time_start}-{time_end} seconds (do not save).'
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    try:
      self.before_recording = file_utils.get_file_list_with_metadata(
          device_constants.AUDIORECORDER_DATA, env.controller
      )
    except RuntimeError as exc:
      raise RuntimeError(
          "Failed to inspect recordings directory,"
          " {device_constants.AUDIORECORDER_DATA}, for Audio Recorder task."
          " Check to make sure Audio Recorder app is correctly installed."
      ) from exc

  
  
  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    ui_elements = representation_utils.xml_dump_to_ui_elements(
        adb_utils.uiautomator_dump(env.controller)
    )
    def is_time_in_range(element_text, time_start, time_end):
      minutes, seconds = map(int, element_text.split(':'))
      total_seconds = minutes * 60 + seconds
      return time_start <= total_seconds <= time_end
    
    for element in ui_elements:
      if element.resource_id == 'com.dimowner.audiorecorder:id/txt_name':
        if element.text != 'Paused':
          return 0
      if element.resource_id == 'com.dimowner.audiorecorder:id/txt_progress':
        if is_time_in_range(element_text=element.text, 
                            time_start=self.params['time_start'],
                            time_end=self.params['time_end']) == False:
          return 0
    return 1

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    time_start =random.randint(10, 20)
    return {'time_start': time_start,
            'time_end': time_start+cls.time_gap}

# resource_id text
# com.dimowner.audiorecorder:id/txt_progress 00:55
# com.dimowner.audiorecorder:id/txt_name Paused
# 



# New Tasks
class GSATimingAudioRecorderRecordAudioTime(_AudioRecorder):
  """Task for checking that one audio recording has been completed with specific file name and duration."""

  complexity = 3
  schema = {
      "type": "object",
      "properties": {
          "file_name": {"type": "string"},
          "time_start": {"type": "integer"},
          "time_end": {"type": "integer"},
      },
      "required": ["file_name", "time_start", "time_end"],
  }
  time_gap = 2
  template = (
    'Record a {time_start}-{time_end}-second audio clip in '
    'Audio Recorder and save it with the name "{file_name}".'
  )
  def _clear_audio_recorder_data(self, env: interface.AsyncEnv) -> None:
    """Clears all audio recorder data on device."""
    file_utils.clear_directory(
        device_constants.AUDIORECORDER_DATA, env.controller
    )
    
  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self._clear_audio_recorder_data(env)

    try:
      self.before_recording = file_utils.get_file_list_with_metadata(
          device_constants.AUDIORECORDER_DATA, env.controller
      )
    except RuntimeError as exc:
      raise RuntimeError(
          "Failed to inspect recordings directory,"
          " {device_constants.AUDIORECORDER_DATA}, for Audio Recorder task."
          " Check to make sure Audio Recorder app is correctly installed."
      ) from exc

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    file_name = self.params["file_name"]
    
    # Check if the file with the specified name exists
    file_exists = file_utils.check_file_or_folder_exists(
        file_name + ".m4a",
        device_constants.AUDIORECORDER_DATA,
        env.controller,
    )
    
    if not file_exists:
      logging.info("File %s.m4a not found", file_name)
      print("File %s.m4a not found", file_name)
      return 0.0
    
    # Get the full path to the file
    all_files = file_utils.get_file_list_with_metadata(
        device_constants.AUDIORECORDER_DATA, env.controller
    )
    
    target_file = None
    for file in all_files:
      if file.file_name == file_name + ".m4a":
        target_file = file
        break
    
    if target_file is None:
      logging.info("File %s.m4a not found in metadata", file_name)
      return 0.0
    
    # Check the duration of the audio file
    tmp_path = None
    try:
      with tempfile.NamedTemporaryFile(suffix='.m4a', delete=False) as tmp:
        tmp_path = tmp.name
      
      # Pull the file from device
      adb_utils.issue_generic_request(
          ['pull', target_file.full_path, tmp_path],
          env.controller
      )
      
      result = subprocess.run(
          ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
          '-of', 'default=noprint_wrappers=1:nokey=1', tmp_path],
          capture_output=True, text=True, timeout=10
      )
      
      duration = float(result.stdout.strip())
      
      logging.info("Audio file duration: %f seconds", duration)
      
      if duration >= self.params['time_start'] and duration <= self.params['time_end']:
        return 1.0
      else:
        logging.info(
            "Duration %f is not in range [%d, %d]",
            duration,
            self.params['time_start'],
            self.params['time_end']
        )
        return 0.0
    except Exception as e:
      logging.error("Error checking audio duration: %s", e)
      return 0.0
    finally:
      # Clean up temporary file
      if tmp_path is not None:
        try:
          os.remove(tmp_path)
        except OSError:
          pass

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    time_start = random.randint(10, 20)
    file_name = random.choice([
        "recording",
        "audio_clip",
        "voice_memo",
        "interview",
        "meeting",
        "lecture",
        "session",
        "note",
        "conference",
        "webinar",
    ])
    return {
        'file_name': file_name,
        'time_start': time_start,
        'time_end': time_start + cls.time_gap
    }




class RefusalAudioRecorderRecordAudioWithFileNameConflict1(_AudioRecorder):
  """Task for checking that one audio recording with file_name has been completed."""

  complexity = 2
  schema = file_validators.CreateFile.schema
  template = (
      'Record an audio clip and save it with name "{file_name}" using Audio'
      ' Recorder app. '
      'Set the Sample rate as 16kHz and the Bitrate as {wrong_bitrate}. '
  )


  answer = (
    'The instruction cannot be completed because the app doesn\'t support setting the required bitrate. The request conflicts with the app’s functionality.'
    )
  # 'The GUI agent must reply to the user that the instruction cannot be completed. '
  # 'For example, the agent can reply: I can\'t complete the request—the app doesn\'t support setting the bitrate to {wrong_bitrate}.'

  def __init__(self, params: dict[str, Any]):
    """See base class."""
    super().__init__(params)
    self.initialized = False
    self.create_file_task = file_validators.CreateFile(
        params, device_constants.AUDIORECORDER_DATA
    )

  def _clear_audio_recorder_data(self, env: interface.AsyncEnv) -> None:
    """Clears all audio recorder data on device."""
    file_utils.clear_directory(
        device_constants.AUDIORECORDER_DATA, env.controller
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)
    self._clear_audio_recorder_data(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    file_name = self.params["file_name"]
    exists = file_utils.check_file_or_folder_exists(
        file_name + ".m4a", self.create_file_task.data_directory, env.controller
    )
    if exists:
      logging.info("%s should not be created. ", file_name)
      return 0.0
    

    if not env.interaction_cache:
      return 0.0
    ply = VerifyPolicy({})
    ans = self.answer.format(self.params["wrong_bitrate"])

    result = ply.verify_textual_prediction(
            goal=self.goal,
            prediction=env.interaction_cache,
            gt_answer=ans,
            judgement_type = 'refusal'
        )
    return result


  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    name = [
        "interview",
        "meeting",
        "lecture",
        "session",
        "note",
        "conference",
        "webinar",
        "workshop",
        "seminar",
        "briefing",
        "discussion",
        "talk",
        "presentation",
        "training",
        "guidance",
        "memo",
        "narration",
        "storytelling",
        "journal",
        "diary",
        "debate",
        "symposium",
        "roundtable",
        "consultation",
        "review",
    ]
    return {
        "file_name": user_data_generation.generate_modified_file_name(
            random.choice(name) + ".m4a"
        ),
        "text": "",  # Unused.
        "wrong_bitrate": random.choice([191,193,194,195,196,197])
    }

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.create_file_task.tear_down(env)
    self._clear_audio_recorder_data(env)



class RefusalAudioRecorderRecordAudioWithFileNameConflict2(_AudioRecorder):
  """Task for checking that one audio recording with file_name has been completed."""

  complexity = 2
  schema = file_validators.CreateFile.schema
  template = (
      'Record an audio clip and save it with name "{file_name}" using Audio'
      " Recorder app. "
      "Set the Sample rate as {wrong_samplerate_str}kHz and the Bitrate as 192 kbps. "
  )
  answer = ('The instruction cannot be completed because the app doesn\'t support setting the required Sample rate. The request conflicts with the app’s functionality.'
    )
  
  def __init__(self, params: dict[str, Any]):
    """See base class."""
    super().__init__(params)
    self.initialized = False
    self.create_file_task = file_validators.CreateFile(
        params, device_constants.AUDIORECORDER_DATA
    )

  def _clear_audio_recorder_data(self, env: interface.AsyncEnv) -> None:
    """Clears all audio recorder data on device."""
    file_utils.clear_directory(
        device_constants.AUDIORECORDER_DATA, env.controller
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)
    self._clear_audio_recorder_data(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    file_name = self.params["file_name"]
    exists = file_utils.check_file_or_folder_exists(
        file_name + ".m4a", self.create_file_task.data_directory, env.controller
    )
    if exists:
      logging.info("%s should not be created. ", file_name)
      return 0.0

    if not env.interaction_cache:
      return 0.0
    ply = VerifyPolicy({})
    ans = self.answer.format(self.params["wrong_samplerate_str"])

    result = ply.verify_textual_prediction(
            goal=self.goal,
            prediction=env.interaction_cache,
            gt_answer=ans,
            judgement_type = 'refusal'
        )
    return result

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    name = [
        "interview",
        "meeting",
        "lecture",
        "session",
        "note",
        "conference",
        "webinar",
        "workshop",
        "seminar",
        "briefing",
        "discussion",
        "talk",
        "presentation",
        "training",
        "guidance",
        "memo",
        "narration",
        "storytelling",
        "journal",
        "diary",
        "debate",
        "symposium",
        "roundtable",
        "consultation",
        "review",
    ]
    return {
        "file_name": user_data_generation.generate_modified_file_name(
            random.choice(name) + ".m4a"
        ),
        "text": "",  # Unused.
        "wrong_samplerate_str": random.choice(['44.2', '44.3', '44.4', '44.5', '44.6', '44.7', '44.8'])
    }

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.create_file_task.tear_down(env)
    self._clear_audio_recorder_data(env)


class NoiseResistAudioRecorderRecordAudioWithFilenameCall(_AudioRecorder):
  complexity = 2.3
  schema = file_validators.CreateFile.schema
  template = (
      'Record an audio clip and save it with name "{file_name}" using Audio'
      " Recorder app."
  )

  def __init__(self, params: dict[str, Any]):
    """See base class."""
    super().__init__(params)
    self.initialized = False
    self.create_file_task = file_validators.CreateFile(
        params, device_constants.AUDIORECORDER_DATA
    )
    """call noising"""
    self.call_coming_steps = params.get("call_coming_steps", [5, 10, 15])
    self.phone_number = params.get("phone_number", "12345678")

  def _clear_audio_recorder_data(self, env: interface.AsyncEnv) -> None:
    """Clears all audio recorder data on device."""
    file_utils.clear_directory(
        device_constants.AUDIORECORDER_DATA, env.controller
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)
    self._clear_audio_recorder_data(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    file_name = self.params["file_name"]
    exists = file_utils.check_file_or_folder_exists(
        file_name + ".m4a", self.create_file_task.data_directory, env.controller
    )
    if not exists:
      logging.info("%s not found", file_name)
      return 0.0
    return 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    name = [
        "interview",
        "meeting",
        "lecture",
        "session",
        "note",
        "conference",
        "webinar",
        "workshop",
        "seminar",
        "briefing",
        "discussion",
        "talk",
        "presentation",
        "training",
        "guidance",
        "memo",
        "narration",
        "storytelling",
        "journal",
        "diary",
        "debate",
        "symposium",
        "roundtable",
        "consultation",
        "review",
    ]
    steps = sorted(random.sample(range(0, 21, 3), 3))
    phone_number = ''.join(random.choices('0123456789', k=8))

    return {
        "file_name": user_data_generation.generate_modified_file_name(
            random.choice(name) + ".m4a"
        ),
        "text": "",  # Unused.
        "call_coming_steps": steps,
        "phone_number": phone_number,
    }
  
  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.create_file_task.tear_down(env)
    self._clear_audio_recorder_data(env)

  def check_status(self, env: interface.AsyncEnv):
    self.is_call_condition(env)


class GSATimingAudioRecorderRecordAudioTimeCHS(GSATimingAudioRecorderRecordAudioTime):
  template = '在Audio Recorder应用中录制一段时长为 {time_start} 至 {time_end} 秒的音频，并将其保存。'

class GSATimingAudioRecorderRecordAudioTimeENGVariation(GSATimingAudioRecorderRecordAudioTime):
  template = 'Open the Audio Recorder app, capture an audio segment lasting between {time_start} and {time_end} seconds, and then save it.'
