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
# Changes: Implementing tasks such as MultiRoundMarkorCreateNoteAndSms.



"""Tasks that involve Markor and SMS."""


from absl import logging
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals.common_validators import sms_validators
from android_world.task_evals.single import markor
from android_world.utils import file_utils
import random

class MarkorCreateNoteAndSms(markor.Markor):
  """Task for checking that a new note in Markor has been created and then an SMS has been sent."""

  app_names = ("markor", "simple sms messenger")
  complexity = 1.8
  schema = {
      "type": "object",
      "properties": {
          "file_name": {"type": "string"},
          "text": {"type": "string"},
          "number": {"type": "string"},
      },
      "required": ["file_name", "text", "number"],
  }

  template = (
      "Create a new note in Markor named {file_name} with the following text:"
      " {text}. Share the entire content of the note with the phone number"
      " {number} via SMS using Simple SMS Messenger"
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.markor_task = markor.MarkorCreateNote(
        params={
            "file_name": self.params["file_name"],
            "text": self.params["text"],
        }
    )
    self.markor_task.initialize_task(env)

    self.sms_task = sms_validators.SimpleSMSSendSms(
        params={"number": self.params["number"], "message": self.params["text"]}
    )
    self.sms_task.initialize_task(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    markor_success = self.markor_task.is_successful(env)
    logging.info("Markor success: %s", markor_success)

    sms_success = self.sms_task.is_successful(env)
    logging.info("SMS success: %s", sms_success)

    return (markor_success + sms_success) / 2.0

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    self.markor_task.tear_down(env)
    self.sms_task.tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    markor_params = markor.MarkorCreateNote.generate_random_params()
    sms_params = sms_validators.SimpleSMSSendSms.generate_random_params()

    compound_params = {
        "file_name": markor_params["file_name"],
        "text": markor_params["text"],
        "number": sms_params["number"],
    }

    return compound_params


# Multiround

class MultiRoundMarkorCreateNoteAndSms(markor.Markor):
  """Task for checking that a new note in Markor has been created and then an SMS has been sent."""

  app_names = ("markor", "simple sms messenger")
  complexity = 7
  schema = {
      "type": "object",
      "properties": {
          "file_name": {"type": "string"},
          "text": {"type": "string"},
          "number": {"type": "string"},
      },
      "required": ["file_name", "text", "number"],
  }
  round = 0
  max_round = 3


  template = (
      "Create a new note in Markor named {file_name} with the following text:"
      " {text}"
  )
  template1 = (
    'Add the following text on a new line in the same note: {text_2nd}'
  )

  template2 = (
    "Bold the line that was just appended."
  )

  template3 = (
    "Share only the content inside the bold formatting to"
      " {number} via SMS using Simple SMS Messenger"
  )


  @property
  def goal(self) -> str:
    """The language goal constructed from the template with the params."""
    if self.round == 0:
      return self.template.format(**self.params)
    elif self.round == 1:
      return self.template1.format(**self.params)
    elif self.round == 2:
      return self.template2.format(**self.params)
    elif self.round == 3:
      return self.template3.format(**self.params)


  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    
    self.sms_task = sms_validators.SimpleSMSSendSms(
        params={"number": self.params["number"],
                "message": self.params["text_2nd"]}
    )
    self.sms_task.initialize_task(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    
    if self.round == 0:
      # Round 0: Create a new note in Markor named {file_name} with the following text: {text}
      if not file_utils.check_file_or_folder_exists(
          self.params["file_name"],
          device_constants.MARKOR_DATA,
          env.controller,
      ):
        return 0.0
      if not file_utils.check_file_content(
          file_utils.convert_to_posix_path(
              device_constants.MARKOR_DATA, self.params["file_name"]
          ),
          self.params["text"],
          env.controller,
      ):
        return 0.0
      return 1.0
      
    elif self.round == 1:
      # Round 1: Add the following text on a new line in the same note: {text_2nd}
      if not file_utils.check_file_or_folder_exists(
          self.params["file_name"],
          device_constants.MARKOR_DATA,
          env.controller,
      ):
        return 0.0
      if not file_utils.check_file_content(
          file_utils.convert_to_posix_path(
              device_constants.MARKOR_DATA, self.params["file_name"]
          ),
          f'''{self.params["text"]}
{self.params["text_2nd"]}''',
          env.controller,
      ):
        return 0.0
      return 1.0
      
    elif self.round == 2:
      # Round 2: Bold the line that was just appended
      if not file_utils.check_file_or_folder_exists(
          self.params["file_name"],
          device_constants.MARKOR_DATA,
          env.controller,
      ):
        return 0.0
      if not file_utils.check_file_content(
          file_utils.convert_to_posix_path(
              device_constants.MARKOR_DATA, self.params["file_name"]
          ),
          f'''{self.params["text"]}
**{self.params["text_2nd"]}**''',
          env.controller,
      ):
        return 0.0
      return 1.0
      
    elif self.round == 3:
      # Round 3: Share only the content inside the bold formatting to {number} via SMS
      sms_success = self.sms_task.is_successful(env)
      return sms_success

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    self.sms_task.tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    markor_params = markor.MarkorCreateNote.generate_random_params()
    sms_params = sms_validators.SimpleSMSSendSms.generate_random_params()
    markor_params_2nd = markor.MarkorCreateNote.generate_random_params()

    compound_params = {
        "file_name": markor_params["file_name"],
        "text": markor_params["text"],
        "number": sms_params["number"],
        'text_2nd': markor_params_2nd["text"],
    }
    return compound_params


# 失败  
class MultiRoundMarkorCreateNoteSummaryAndSms(markor.Markor):
  app_names = ("markor", "simple sms messenger")
  complexity = 7
  schema = {
      "type": "object",
      "properties": {
          "file_name": {"type": "string"},
          "text": {"type": "string"},
          "number": {"type": "string"},
          "project_name": {"type": "string"},
          "checklist": {"type": "string"},
          "file_name_renamed": {"type": "string"},
          "target_folder": {"type": "string"},
          "sms_message": {"type": "string"},
      },
      "required": ["file_name", "text", "number"],
  }
  round = 0
  max_round = 5

  template = (
      "Create a new note in Markor named {file_name} with the following text: {text}."
  )
  template1 = (
      "Prepend an H1 heading to the note with the title: Project - {project_name}."
  )
  template2 = (
      "Append the following checklist items to the end of the note (each as '- [ ] '): {checklist}."
  )
  template3 = (
      "Share only the checklist section of the note via SMS to {number} using Simple SMS Messenger."
  )
  template4 = (
      "Rename the note to {file_name_renamed} and move it into the folder {target_folder}."
  )

  @property
  def goal(self) -> str:
    if self.round == 0:
      return self.template.format(**self.params)
    elif self.round == 1:
      return self.template1.format(**self.params)
    elif self.round == 2:
      return self.template2.format(**self.params)
    elif self.round == 3:
      return self.template3.format(**self.params)
    elif self.round == 4:
      return self.template4.format(**self.params)

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.markor_task = markor.MarkorCreateNote(
        params={
            "file_name": self.params["file_name"],
            "text": self.params["text"],
        }
    )
    self.markor_task.initialize_task(env)

    self.sms_task = sms_validators.SimpleSMSSendSms(
        params={
            "number": self.params["number"],
            "message": self.params.get("sms_message", self.params["text"]),
        }
    )
    self.sms_task.initialize_task(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    markor_success = self.markor_task.is_successful(env)
    sms_success = self.sms_task.is_successful(env)
    return (markor_success + sms_success) / 2.0

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    self.markor_task.tear_down(env)
    self.sms_task.tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    markor_params = markor.MarkorCreateNote.generate_random_params()
    sms_params = sms_validators.SimpleSMSSendSms.generate_random_params()
    markor_params_renamed = markor.MarkorCreateNote.generate_random_params()

    subfolders = [
        "ProjectNotes",
        "ActionItems",
        "DailyLogs",
        "Ideas",
        "MeetingMinutes",
        "Personal",
        "Plans",
        "ReadingList",
        "Research",
        "Work",
    ]
    target_folder = random.choice(subfolders)

    project_names = [
        "Apollo",
        "Orion",
        "Nexus",
        "Atlas",
        "Zephyr",
        "Pegasus",
        "Aurora",
        "Aquila",
        "Helios",
        "Vega",
    ]
    project_name = random.choice(project_names)

    checklist_items = [
        "Define requirements",
        "Draft timeline",
        "Assign tasks",
        "Prepare resources",
        "Review milestones",
        "Schedule kickoff",
    ]
    # Join as a single string; agent will split/add lines with '- [ ] '
    checklist_text = "; ".join(random.sample(checklist_items, k=4))

    compound_params = {
        "file_name": markor_params["file_name"],
        "text": markor_params["text"],
        "number": sms_params["number"],
        "project_name": project_name,
        "checklist": checklist_text,
        "file_name_renamed": markor_params_renamed["file_name"],
        "target_folder": target_folder,
        # For the SMS round we share the checklist content
        "sms_message": checklist_text,
    }
    return compound_params