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
# Changes: Implementing tasks in VenusBench using markor app.


"""Tasks for Markor app."""
import re

import dataclasses
import datetime
import random
from typing import Any
import re

from absl import logging
from android_world.env import adb_utils
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import file_validators
from android_world.task_evals.single import vlc
from android_world.task_evals.utils import receipt_generator
from android_world.task_evals.utils import user_data_generation
from android_world.utils import datetime_utils
from android_world.utils import file_utils
from android_world.utils import fuzzy_match_lib
from android_world.task_evals.single import markor
from android_world.policy.verification import VerifyPolicy

@dataclasses.dataclass(frozen=True)
class _Note:
  name: str
  content: str


generate_random_sentence = lambda: random.choice(
    user_data_generation.RANDOM_SENTENCES
)


def _generate_random_note() -> _Note:
  """Generates a random note."""
  extensions = [".md", ]
  random_file_name = (
      user_data_generation.generate_random_file_name()
      + random.choice(extensions)
  )
  return _Note(random_file_name, generate_random_sentence())


class Markor(task_eval.TaskEval):
  app_names = ("markor",)

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    file_utils.clear_directory(device_constants.MARKOR_DATA, env.controller)

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    file_utils.clear_directory(device_constants.MARKOR_DATA, env.controller)


class MarkorMoveNote(Markor):
  """Task for checking that a file has been moved in Markor."""

  complexity = 1.4
  schema = file_validators.MoveFile.schema
  template = (
      "In Markor, move the note {file_name} from {source_folder} to"
      " {destination_folder}."
  )

  def __init__(self, params: dict[str, Any]):
    """Initialize the task."""
    super().__init__(params)
    self.move_file_task = file_validators.MoveFile(
        params, device_constants.MARKOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.move_file_task.initialize_task(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return self.move_file_task.is_successful(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    subfolders = [
        "BookNotes",
        "CodeSnippets",
        "DailyNotes",
        "FitnessPlans",
        "MeetingMinutes",
        "PersonalJournal",
        "RecipeCollections",
        "StudyGuides",
        "TravelItineraries",
        "WorkProjects",
    ]
    source_folder = random.choice(subfolders)
    destination_folder = random.choice(
        [folder for folder in subfolders if folder != source_folder]
    )
    file_name = _generate_random_note().name
    return {
        "file_name": file_name,
        "source_folder": source_folder,
        "destination_folder": destination_folder,
        "noise_candidates": _NOTE_TITLES,
    }

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.move_file_task.tear_down(env)


class MarkorCreateFolder(Markor):
  """Task for checking that a new folder in Markor has been created with a specific name."""

  complexity = 1
  schema = {
      "type": "object",
      "properties": {
          "folder_name": {"type": "string"},
      },
      "required": ["folder_name"],
  }
  template = "Create a new folder in Markor named {folder_name}."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    user_data_generation.generate_noise_files(
        "file",
        device_constants.MARKOR_DATA,
        env.controller,
        _NOTE_TITLES,
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    folder_name = self.params["folder_name"]

    exists = file_utils.check_file_or_folder_exists(
        folder_name, device_constants.MARKOR_DATA, env.controller
    )

    if not exists:
      logging.info("%s not found", folder_name)
      return 0.0

    return 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    random_folder_name = "folder_" + str(
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    return {"folder_name": random_folder_name}


class MarkorEditNote(Markor):
  """Task for editing an existing note in Markor."""

  complexity = 1.2
  schema = {
      "type": "object",
      "properties": {
          "file_name": {"type": "string"},
          "header": {"type": "string"},
          "footer": {"type": "string"},
          "replace_text": {"type": "string"},
          "edit_type": {
              "type": "string",
              "enum": ["header", "footer", "replace"],
          },
      },
      "required": ["file_name", "edit_type"],
  }

  @property
  def template(self) -> str:
    templates = {
        "header": (
            "Edit {file_name} in Markor. Add to the top of the note {header}"
        ),
        "footer": (
            "Edit {file_name} in Markor. Add to the bottom of the note {footer}"
        ),
        "replace": (
            "Edit {file_name} in Markor. Replace the text with {replace_text}"
        ),
    }

    if "edit_type" not in self.params and "edit_type" not in templates:
      return templates.get(
          self.params.get("edit_type"),
          "Invalid edit_type for {file_name} in Markor.",
      )
    return templates[self.params.get("edit_type")]

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    user_data_generation.generate_noise_files(
        self.params["file_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        _NOTE_TITLES,
    )
    self.original_content = file_utils.create_file(
        self.params["file_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        content=generate_random_sentence(),
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    res = adb_utils.issue_generic_request(
        [
            "shell",
            "cat",
            file_utils.convert_to_posix_path(
                device_constants.MARKOR_DATA, self.params["file_name"]
            ),
        ],
        env.controller,
    )
    file_contents = res.generic.output.decode().replace("\r", "").strip()
    logging.info("Retrieved file contents: %s", file_contents)

    if self.params["edit_type"] == "header":
      expected_content = self.params["header"] + "\n" + self.original_content
    elif self.params["edit_type"] == "footer":
      expected_content = self.original_content + "\n" + self.params["footer"]
    else:
      expected_content = self.params["replace_text"]

    is_match = fuzzy_match_lib.fuzzy_match(file_contents, expected_content)
    logging.info(
        "Is content match: %s.\nFound: %s\nExpected: %s",
        is_match,
        file_contents,
        expected_content,
    )

    return 1.0 if is_match else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    extensions = [".md", ]

    random_file_name = (
        "note_"
        + user_data_generation.generate_random_string(5)
        + random.choice(extensions)
    )

    edit_type = random.choice(["header", "footer", "replace"])

    params = {
        "file_name": random_file_name,
        "edit_type": edit_type,
    }

    if edit_type == "header":
      params["header"] = generate_random_sentence()
    elif edit_type == "footer":
      params["footer"] = generate_random_sentence()
    elif edit_type == "replace":
      params["replace_text"] = "\n".join(
          [generate_random_sentence() for _ in range(3)]
      )

    return params


class MarkorDeleteNote(Markor):
  """Task for checking that a note in Markor has been deleted."""

  complexity = 1
  schema = file_validators.DeleteFile.schema
  template = "Delete the note in Markor named {file_name}."

  def __init__(self, params: dict[str, Any]):
    """Initialize the task."""
    super().__init__(params)
    self.delete_file_task = file_validators.DeleteFile(
        params, device_constants.MARKOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.delete_file_task.initialize_task(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return self.delete_file_task.is_successful(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    file_name = user_data_generation.generate_random_file_name()
    return {"file_name": file_name, "noise_candidates": _NOTE_TITLES}

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.delete_file_task.tear_down(env)


class MarkorDeleteNewestNote(Markor):
  """Task for deleting the newest note in Markor."""

  complexity = 1
  schema = {}
  template = "Delete the newest note in Markor."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    # Generate some random notes in Markor.
    for _ in range(random.randint(2, 6)):
      note = _generate_random_note()
      file_utils.create_file(
          note.name,
          device_constants.MARKOR_DATA,
          env.controller,
          content=note.content,
      )
      # Advance system time so the change time for these initial notes can be
      # separated.
      datetime_utils.advance_system_time(
          datetime.timedelta(minutes=random.randint(-500, 500)), env.controller
      )

    file_list = file_utils.get_file_list_with_metadata(
        device_constants.MARKOR_DATA, env.controller
    )
    self.initial_file_list_sorted = sorted(
        file_list, key=lambda f: f.change_time
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    new_file_list = file_utils.get_file_list_with_metadata(
        device_constants.MARKOR_DATA, env.controller
    )
    new_file_list_sorted = sorted(new_file_list, key=lambda f: f.change_time)
    for i in range(len(new_file_list)):
      # Both file lists are ordered by file change time, so by simply checking
      # file names and their change time are the same, we can ensure all other
      # files have not been changed.
      if not (
          new_file_list_sorted[i].file_name
          == self.initial_file_list_sorted[i].file_name
          and new_file_list_sorted[i].change_time
          == self.initial_file_list_sorted[i].change_time
      ):
        return 0.0
    one_fewer_file = (
        len(new_file_list_sorted) == len(self.initial_file_list_sorted) - 1
    )
    return 1.0 if one_fewer_file else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    return {}



class MarkorDeleteAllNotes(Markor):
  """Task for deleting all notes in Markor."""

  # For this task's complexity, the agent may complete this task by deleting the
  # files one-by-one which envolves many steps (more than 10), but there is also
  # an optimal approach by first long pressing one file, then tapping to select
  # all others and deleting them all together.
  complexity = 1.4
  schema = {}
  template = "Delete all my notes in Markor."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    user_data_generation.generate_noise_files(
        user_data_generation.generate_random_string(5),
        device_constants.MARKOR_DATA,
        env.controller,
        _NOTE_TITLES,
        random.randint(2, 6),
    )

    file_list = file_utils.get_file_list_with_metadata(
        device_constants.MARKOR_DATA, env.controller
    )

    if not file_list:
      raise RuntimeError("Something went wrong, file was not created.")

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    file_list = file_utils.get_file_list_with_metadata(
        device_constants.MARKOR_DATA, env.controller
    )
    return 0.0 if file_list else 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    return {}



class VagueMarkorCreateNotewithCalculation(Markor):
  """Task for checking that a new note in Markor has been created with a specific name and text."""

  app_names = ("markor",)
  complexity = 4
  schema = file_validators.CreateFile.schema

  template = (
      "Create a new note in Markor named as {file_name}. "
      "The content of the note must be the integer result of {formula}, "
      "written as a plain string of digits only—no commas, spaces, words, or scientific notation."
  )

  def __init__(self, params: dict[str, Any]):
    """See base class."""
    super().__init__(params)

    self.create_file_task = file_validators.CreateFile(
        params, device_constants.MARKOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)

    self.create_file_task.initialize_task(env)  

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return self.create_file_task.is_successful(env) 

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    note = _generate_random_note()

    def generate_integer_expr() -> tuple[str, int]:
      while True:
          a, b, c = [random.randint(1000, 9999) for _ in range(3)]
          d, e, f = [random.randint(1000, 9999) for _ in range(3)]
          g, h, i = [random.randint(1000, 9999) for _ in range(3)]

          left   = a * b * c
          mid    = d * e * f
          right  = g * h * i

          res = left - mid - right
          if res > 0:
              expr = f"{a}*{b}*{c} - {d}*{e}*{f} - {g}*{h}*{i}"
              return expr, res
    expr, answer = generate_integer_expr()
    return {"file_name": note.name, "text": answer, 'formula': expr}

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.create_file_task.tear_down(env)


class MarkorCreateNote(Markor):
  """Task for checking that a new note in Markor has been created with a specific name and text."""

  app_names = ("markor",)
  complexity = 1.6
  schema = file_validators.CreateFile.schema
  template = (
      "Create a new note in Markor named {file_name} with the following text:"
      " {text}"
  )

  def __init__(self, params: dict[str, Any]):
    """See base class."""
    super().__init__(params)

    self.create_file_task = file_validators.CreateFile(
        params, device_constants.MARKOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)  # Delegate

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return self.create_file_task.is_successful(env)  # Delegate

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    note = _generate_random_note()
    return {"file_name": note.name, "text": note.content}

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.create_file_task.tear_down(env)


class MarkorCreateNoteFromClipboard(Markor):
  """Task for creating a note using text in clipboard in Markor."""

  app_names = ("markor", "clipper")
  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {
          "file_name": {"type": "string"},
          "file_content": {"type": "string"},
      },
      "required": ["file_name", "file_content"],
  }
  template = (
      "Create a note in Markor named {file_name}. Perform a paste operation in"
      " the note and save the note."
  )

  def __init__(self, params: dict[str, Any]):
    """Initialize the task."""
    super().__init__(params)
    if "file_content" not in params or not params["file_content"]:
      params["file_content"] = user_data_generation.generate_random_string(20)
    self.create_file_task = file_validators.CreateFile(
        {"file_name": params["file_name"], "text": params["file_content"]},
        device_constants.MARKOR_DATA,
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.set_clipboard_contents(
        self.params["file_content"], env.controller
    )
    if (
        adb_utils.get_clipboard_contents(env.controller)
        != self.params["file_content"]
    ):
      raise RuntimeError(
          "Something went wrong, clipboard not set up correctly."
      )
    self.create_file_task.initialize_task(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return self.create_file_task.is_successful(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    return {
        "file_name": _generate_random_note().name,
        "file_content": user_data_generation.generate_random_string(10),
    }

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.create_file_task.tear_down(env)


class MarkorMergeNotes(Markor):
  """Task for merging three existing notes into a new one."""

  complexity = 7.8
  schema = {
      "type": "object",
      "properties": {
          "file1_name": {"type": "string"},
          "file2_name": {"type": "string"},
          "file3_name": {"type": "string"},
          "new_file_name": {"type": "string"},
          "file1_content": {"type": "string"},
          "file2_content": {"type": "string"},
          "file3_content": {"type": "string"},
      },
      "required": [
          "file1_name",
          "file2_name",
          "file3_name",
          "new_file_name",
          "file1_content",
          "file2_content",
          "file3_content",
      ],
  }
  template = (
      "Merge the contents of Markor notes {file1_name}, {file2_name} and"
      " {file3_name} (in the same order) into a new Markor note named"
      " {new_file_name} and save it. Add a new line between the content of each"
      " note."
  )

  def __init__(self, params: dict[str, Any]):
    """Initialize the task."""
    super().__init__(params)
    self.create_file_task = file_validators.CreateFile(
        {
            "file_name": params["new_file_name"],
            # file_util.create_file with non-empty content will add a \n to the
            # end of the file.
            "text": (
                "\n\n".join([
                    self.params["file1_content"],
                    self.params["file2_content"],
                    self.params["file3_content"],
                ])
                + "\n"
            ),
        },
        device_constants.MARKOR_DATA,
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)
    file_utils.create_file(
        self.params["file1_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        content=self.params["file1_content"],
    )
    file_utils.create_file(
        self.params["file2_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        content=self.params["file2_content"],
    )
    file_utils.create_file(
        self.params["file3_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        content=self.params["file3_content"],
    )

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    file_utils.remove_single_file(
        self.params["file1_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    )
    file_utils.remove_single_file(
        self.params["file2_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    )
    file_utils.remove_single_file(
        self.params["file3_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    )
    self.create_file_task.tear_down(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not self.create_file_task.is_successful(env):
      return 0.0
    # The CreateFile task is using a fuzzy match in its is_successful function,
    # but here we want to explicitly check if the agent adds a blank line
    # between the notes. The following check only works based on the current way
    # we generate notes with the assumption that each file's content is a string
    # of length less than 20, consisting of letters and digits, ended with a \n.
    merged_file = (
        adb_utils.issue_generic_request(
            [
                "shell",
                "cat",
                file_utils.convert_to_posix_path(
                    device_constants.MARKOR_DATA, self.params["new_file_name"]
                ),
            ],
            env.controller,
        )
        .generic.output.decode()
        .replace("\r", "")
        .strip()
    )

    # merged_file should look like,
    # file1\n\nfile2\n\nfile3, where the first and third \n are inserted by
    # create_file in file_utils, the second and the forth \n should be inserted
    # by agent.
    content_split = merged_file.split("\n")
    are_notes_merged = (
        len(content_split) == 5
        and (not content_split[1])
        and (not content_split[3])
    )
    return 1.0 if are_notes_merged else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    return {
        "file1_name": _generate_random_note().name,
        "file2_name": _generate_random_note().name,
        "file3_name": _generate_random_note().name,
        "new_file_name": user_data_generation.generate_random_string(8),
        "file1_content": user_data_generation.generate_random_string(20),
        "file2_content": user_data_generation.generate_random_string(20),
        "file3_content": user_data_generation.generate_random_string(20),
    }


class MarkorChangeNoteContent(Markor):
  """Task for changing an existing note's content and renaming it."""

  complexity = 1.2
  schema = {
      "type": "object",
      "properties": {
          "original_name": {"type": "string"},
          "new_name": {"type": "string"},
          "updated_content": {"type": "string"},
      },
      "required": ["original_name", "new_name", "updated_content"],
  }
  template = (
      'Update the content of {original_name} to "{updated_content}" in Markor'
      " and change its name to {new_name}."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    file_utils.create_file(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        content=user_data_generation.generate_random_string(20),
    )
    user_data_generation.generate_noise_files(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        _NOTE_TITLES,
    )
    if not file_utils.check_file_or_folder_exists(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    ):
      raise RuntimeError("Something went wrong, file not created correctly.")

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    file_utils.remove_single_file(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if file_utils.check_file_or_folder_exists(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    ):
      return 0.0
    if not file_utils.check_file_or_folder_exists(
        self.params["new_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    ):
      return 0.0
    content_updated = file_utils.check_file_content(
        file_utils.convert_to_posix_path(
            device_constants.MARKOR_DATA, self.params["new_name"]
        ),
        self.params["updated_content"],
        env.controller,
    )
    return 1.0 if content_updated else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    original = _generate_random_note().name
    new = _generate_random_note().name
    return {
        "original_name": original,
        "new_name": new,
        "updated_content": user_data_generation.generate_random_string(20),
    }


class MarkorAddNoteHeader(Markor):
  """Task for adding a header to an existing note and renaming it."""

  complexity = 1.2
  schema = {
      "type": "object",
      "properties": {
          "original_name": {"type": "string"},
          "new_name": {"type": "string"},
          "header": {"type": "string"},
          "original_content": {"type": "string"},
      },
      "required": ["original_name", "new_name", "header", "original_content"],
  }
  template = (
      "Update the Markor note {original_name} by adding the following text,"
      ' along with a new blank line before the existing content: "{header}",'
      " and rename it to {new_name}."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    file_utils.create_file(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        content=self.params["original_content"],
    )
    user_data_generation.generate_noise_files(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        _NOTE_TITLES,
    )

    if not file_utils.check_file_or_folder_exists(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    ):
      raise RuntimeError("Something went wrong, file not created correctly.")

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    file_utils.remove_single_file(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if file_utils.check_file_or_folder_exists(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    ):
      return 0.0
    if not file_utils.check_file_or_folder_exists(
        self.params["new_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    ):
      return 0.0
    correct = file_utils.check_file_content(
        file_utils.convert_to_posix_path(
            device_constants.MARKOR_DATA, self.params["new_name"]
        ),
        self.params["header"] + "\n\n" + self.params["original_content"] + "\n",
        env.controller,
        exact_match=True,
    )
    return 1.0 if correct else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    return {
        "original_name": _generate_random_note().name,
        "original_content": generate_random_sentence(),
        "new_name": _generate_random_note().name,
        "header": user_data_generation.generate_random_string(20),
    }


class MarkorTranscribeReceipt(task_eval.TaskEval):
  """Task for creating a markdown file from a receipt image using Simple Gallery and Markor.

  This task involves viewing a receipt image in Simple Gallery and then
  creating a markdown file in Markor with details of the transactions
  listed in the image. The file should be named 'receipt.md' and include
  transactions with the format "Date, Item, Amount".
  """

  app_names = ("simple gallery pro", "markor")
  complexity = 1.8
  template = (
      "Create a file in Markor, called receipt.md with the transactions from"
      " the receipt.png. Use Simple Gallery to view the receipt. Please enter"
      ' transactions in csv format including the header "Date, Item, Amount".'
  )

  schema = file_validators.CreateFile.schema

  def __init__(self, params: dict[str, Any]):
    super().__init__(params)
    self.img = params.pop("img")
    self.create_file_task = file_validators.CreateFile(
        params, device_constants.MARKOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    """Initializes the task for creating a receipt markdown file."""
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)
    receipt_img_path = file_utils.convert_to_posix_path(
        file_utils.get_local_tmp_directory(), "receipt.png"
    )
    self.img.save(receipt_img_path)
    file_utils.copy_data_to_device(
        receipt_img_path,
        device_constants.GALLERY_DATA,
        env.controller,
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return self.create_file_task.is_successful(env)

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    self.create_file_task.tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    img, text = receipt_generator.create_receipt(random.randint(1, 5))
    text = "\n".join(text.split("\n")[2:])  # Remove header.
    return {
        "img": img,
        "file_name": "receipt.md",
        "text": text,
    }


class MarkorTranscribeVideo(Markor):
  """Task for transcribing a video using Markor."""

  complexity = 2
  schema = file_validators.CreateFile.schema
  app_names = ("markor", "vlc")

  template = (
      "Transcribe the contents of video {video_name} by watching it in VLC"
      " player (located in Download) and writing the sequence of strings shown"
      " on each frame to the text file {file_name} in Markor as a comma"
      ' separated list. For example, if the first frame shows the text "edna"'
      ' and the second frame shows the text "pineapple", then the text file'
      ' should contain only the following text: "edna, pineapple".'
  )

  def __init__(self, params: dict[str, Any]):
    super().__init__(params)
    self.create_file_task = file_validators.CreateFile(
        params, device_constants.MARKOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)
    user_data_generation.write_video_file_to_device(
        self.params["video_name"],
        device_constants.DOWNLOAD_DATA,
        env,
        messages=self.params["messages"],
        message_display_time=8,
    )
    for file in self.params["noise_files"]:
      user_data_generation.write_video_file_to_device(
          file,
          device_constants.DOWNLOAD_DATA,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return self.create_file_task.is_successful(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    messages = list(
        random.sample(
            user_data_generation.COMMON_GIVEN_NAMES, random.randint(2, 4)
        )
    )
    video_name = vlc.generate_file_name()
    text_file_name = f"{video_name.split('.')[0]}_transcription.txt"
    return {
        "file_name": text_file_name,
        "text": ",".join(messages),
        # Video specific.
        "messages": messages,
        "video_name": video_name,
        "noise_files": [
            vlc.generate_file_name() for _ in range(random.randint(5, 20))
        ],
    }

_NOTE_TITLES = [
    "grocery_list_weekly.md",
    "meeting_notes_project_team.md",
    "personal_goals_2024.md",
    "reading_list_2024.md",
    "research_paper_summary.md",
    "summer_vacation_plans.md",
    "budget_home_renovation.md",
    "april_workout_routine.md",
    "birthday_gift_ideas_mom.md",
    "recipe_homemade_pizza.md",
    "weekend_todo_list.md",
    "insurance_plan_comparison.md",
    "art_project_sketches.md",
    "python_learning_goals.md",
    "trip_reflections_recent.md",
    "startup_ideas_launch.md",
    "client_meetings_schedule.md",
    "favorite_book_quotes.md",
    "garden_layout_plan.md",
    "upcoming_presentation_outline.md",
]



# New Tasks of ANT-UI-BENCH

_ALGO_KEYWORDS = ["queue", "stack", "sort", "search", "complexity"]
_ALGO_NAMES = ['QuickSort', 'DIJKSTRA' ,'BubbleSort', 'MergeSort', 'BucketSort', 'SegmentTree']

class GUIBrowsingMarkorFindFilesPath(Markor):
  """Find the folder that contains multiple algorithm notes in Markor."""

  complexity = 6
  schema = {}                       # no extra hyper-parameters needed
  template = (
      "In Markor, find the folder that contains multiple notes about "
      "algorithms or data structures, and tell me the full path of that folder."
      "Return only the full path of that folder, ensuring it starts with 'Markor/' and is followed solely by folder names separated by slashes. "
      "Example format for reference (do not use this exact path): Markor/1ZLZOLI6Xh"
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)

    # 4 candidate folders
    folder_names = [
        user_data_generation.generate_random_string(10) for _ in range(4)
    ]

    self.target_folder_name = random.choice(folder_names)
    self.target_folder = None  
    data_directory = device_constants.MARKOR_DATA



    for folder in folder_names:
        file_directory = file_utils.convert_to_posix_path(
            data_directory, folder
        )
        file_utils.mkdir(file_directory, env.controller)

        note_count = random.randint(3, 5)

        for _ in range(note_count):
            note_name = user_data_generation.generate_random_file_name() + ".md"

            if folder == self.target_folder_name:
                content = "Notes on " + random.choice(_ALGO_KEYWORDS) + ": \n" \
                          + generate_random_sentence()
            else:
                content = generate_random_sentence()

            file_utils.create_file(
                note_name,
                file_directory,
                env.controller,
                content=content,
            )

    self.target_folder = file_utils.convert_to_posix_path(
        data_directory, self.target_folder_name
    )

    if self.target_folder is None:
      raise RuntimeError("Initialization error: no target folder")


  def is_successful(self, env: interface.AsyncEnv) -> float:
      super().is_successful(env)
      if not env.interaction_cache:
          return 0.0

      expected = f"Markor/{self.target_folder_name}"
      if env.interaction_cache.strip() == expected:
          return 1.0
      return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}   


class GUIBrowsingMarkorFindFilesPathHard(Markor):
  """Find the folder that contains multiple algorithm notes in Markor."""

  complexity = 10
  schema = {}                       # no extra hyper-parameters needed
  template = (
      "In Markor, find the folder that contains one note containing information about "
      "algorithms or data structures. "
      "Return only the full path of that folder, ensuring it starts with 'Markor/' and is followed solely by folder names separated by slashes. "
      "Example format for reference (do not use this exact path): Markor/1ZLZOLI6Xh/eE9FJKcpPi/zicdJ1kd5I"
      # "and tell me the full path of that folder."
  )
  level = 3
  num_dirs = 2

  def initialize_task(self, env: interface.AsyncEnv) -> None:
        super().initialize_task(env)

        data_root = (device_constants.MARKOR_DATA)

        target_depth = random.randint(1, self.level)
        target_index = random.randint(0, self.num_dirs - 1)

        self.target_folder = None  

        def build(current_depth: int, parent_posix: str):
            """
            current_depth: 当前正在创建的第几层 (1..level)
            parent_posix: 上一层目录的 POSIX 路径字符串
            """
            for idx in range(self.num_dirs):
                dir_name = user_data_generation.generate_random_string(10)
                dir_posix = file_utils.convert_to_posix_path(parent_posix, dir_name)

                file_utils.mkdir(dir_posix, env.controller)

                note_name = user_data_generation.generate_random_file_name() + ".md"
                if current_depth == target_depth and idx == target_index:
                    self.target_folder = dir_posix
                    content = "Notes on " + random.choice(_ALGO_KEYWORDS) + ": " + \
                              generate_random_sentence()
                else:
                    content = generate_random_sentence()

                file_utils.create_file(note_name, dir_posix, env.controller, content=content)

                if current_depth < self.level:
                    build(current_depth + 1, dir_posix)

        root_posix = file_utils.convert_to_posix_path(data_root)
        build(1, root_posix)

        print(f"target_folder: {self.target_folder}")
        if self.target_folder is None:
            raise RuntimeError("Initialization error: no target folder")
        
  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    target = re.search(r'Markor/.*', self.target_folder).group(0)

    if target == env.interaction_cache.strip():
      return 1.0
    else:
      return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}   


class GUIBrowsingMarkorFindCommonPackage(Markor):
  """Find the common package."""

  complexity = 10
  schema = {}                       # no extra hyper-parameters needed
  template = (
      "Find and tell me the common Python packages "
      "listed in all three markdown files inside Markor."
      "Tell me only those package names without version numbers."
      "If there are multiple package names, separate them with commas without spaces. "
      "For example: package1,package2,package3"
  )
  answer = []

  full_package_list = [
        "absl-py==2.1.0", "adbutils==2.9.3", "aiohappyeyeballs==2.6.1",
        "aiohttp==3.12.15", "aiosignal==1.4.0", "android-env==1.2.2",
        "annotated-types==0.7.0", "anyio==4.9.0", "argon2-cffi==25.1.0",
        "argon2-cffi-bindings==21.2.0", "arrow==1.3.0", "asttokens==3.0.0",
        "async-lru==2.0.5", "attrs==25.3.0", "av==14.4.0", "babel==2.17.0",
        "backoff==2.2.1", "beautifulsoup4==4.13.4", "bleach==6.2.0",
        "blinker==1.9.0", "cachetools==5.5.2", "certifi==2025.4.26",
        "cffi==1.17.1", "charset-normalizer==3.4.1", "click==8.2.1",
        "comm==0.2.2", "conda-pack==0.8.1", "contourpy==1.3.2",
        "cryptography==45.0.7", "cycler==0.12.1", "dashscope==1.24.2",
        "debugpy==1.8.14", "decorator==5.2.1", "defusedxml==0.7.1",
        "deprecation==2.1.0", "distro==1.9.0", "dm-env==1.6",
        "dm-tree==0.1.9", "eval_type_backport==0.2.2", "executing==2.2.0",
        "fastjsonschema==2.21.1", "filelock==3.18.0", "Flask==3.1.1",
        "fonttools==4.57.0", "fqdn==1.5.1", "frozenlist==1.7.0",
        "fsspec==2025.5.1", "fuzzywuzzy==0.18.0",
        "google-ai-generativelanguage==0.6.2", "google-api-core==2.24.2",
        "google-api-python-client==2.168.0", "google-auth==2.39.0",
        "google-auth-httplib2==0.2.0", "google-generativeai==0.5.1",
        "googleapis-common-protos==1.70.0", "grpcio==1.71.0",
        "grpcio-status==1.62.3", "grpcio-tools==1.62.3", "h11==0.16.0",
        "hf-xet==1.1.5", "httpcore==1.0.9", "httplib2==0.22.0",
        "httpx==0.28.1", "huggingface-hub==0.34.4", "idna==3.10",
        "immutabledict==2.0.0", "iniconfig==2.1.0", "ipykernel==6.29.5",
        "ipython==9.2.0", "ipython_pygments_lexers==1.1.1",
        "ipywidgets==8.1.7", "isoduration==20.11.0", "itsdangerous==2.2.0",
        "jedi==0.19.2", "Jinja2==3.1.6", "jiter==0.10.0", "joblib==1.5.1",
        "json5==0.12.0", "jsonlines==4.0.0", "jsonpointer==3.0.0",
        "jsonschema==4.24.0", "jsonschema-specifications==2025.4.1",
        "jupyter==1.1.1", "jupyter-console==6.6.3", "jupyter-events==0.12.0",
        "jupyter-lsp==2.2.5", "jupyter_client==8.6.3",
        "jupyter_core==5.8.1", "jupyter_server==2.16.0",
        "jupyter_server_terminals==0.5.3", "jupyterlab==4.4.3",
        "jupyterlab_pygments==0.3.0", "jupyterlab_server==2.27.3",
        "jupyterlab_widgets==3.0.15", "kiwisolver==1.4.8",
        "Levenshtein==0.27.1", "lxml==6.0.0", "MarkupSafe==3.0.2",
        "matplotlib==3.6.1", "matplotlib-inline==0.1.7", "mistune==3.1.3",
        "mpmath==1.3.0", "multidict==6.6.4", "nbclient==0.10.2",
        "nbconvert==7.16.6", "nbformat==5.10.4", "nest-asyncio==1.6.0",
        "networkx==3.5", "notebook==7.4.3", "notebook_shim==0.2.4",
        "numpy==1.26.3", "nvidia-cublas-cu12==12.6.4.1",
        "nvidia-cuda-cupti-cu12==12.6.80", "nvidia-cuda-nvrtc-cu12==12.6.77",
        "nvidia-cuda-runtime-cu12==12.6.77", "nvidia-cudnn-cu12==9.5.1.17",
        "nvidia-cufft-cu12==11.3.0.4", "nvidia-cufile-cu12==1.11.1.6",
        "nvidia-curand-cu12==10.3.7.77", "nvidia-cusolver-cu12==11.7.1.2",
        "nvidia-cusparse-cu12==12.5.4.2", "nvidia-cusparselt-cu12==0.6.3",
        "nvidia-nccl-cu12==2.26.2", "nvidia-nvjitlink-cu12==12.6.85",
        "nvidia-nvtx-cu12==12.6.77", "openai==1.82.1",
        "opencv-python==4.11.0.86", "overrides==7.7.0", "packaging==25.0",
        "pandas==2.1.4", "pandocfilters==1.5.1", "parso==0.8.4",
        "pexpect==4.9.0", "pillow==11.2.1", "platformdirs==4.3.8",
        "pluggy==1.5.0", "portpicker==1.6.0", "prometheus_client==0.22.1",
        "prompt_toolkit==3.0.51", "propcache==0.3.2", "proto-plus==1.26.1",
        "protobuf==5.29.4", "psutil==7.0.0", "ptyprocess==0.7.0",
        "pure_eval==0.2.3", "pyasn1==0.6.1", "pyasn1_modules==0.4.2",
        "pycparser==2.22", "pycryptodome==3.23.0", "pydantic==2.11.4",
        "pydantic_core==2.33.2", "pydub==0.25.1", "pygame==2.6.1",
        "Pygments==2.19.1", "pyparsing==3.2.3", "pyrsistent==0.20.0",
        "pytesseract==0.3.13", "pytest==8.3.5",
        "python-dateutil==2.9.0.post0", "python-dotenv==1.1.1",
        "python-json-logger==3.3.0", "python-Levenshtein==0.27.1",
        "pytz==2025.2", "PyYAML==6.0.2", "pyzmq==26.4.0",
        "qwen-agent==0.0.29", "qwen-vl-utils==0.0.11", "RapidFuzz==3.13.0",
        "referencing==0.36.2", "regex==2024.11.6", "requests==2.32.3",
        "retry2==0.9.5", "rfc3339-validator==0.1.4",
        "rfc3986-validator==0.1.1", "rpds-py==0.25.1", "rsa==4.9.1",
        "safetensors==0.5.3", "scikit-learn==1.7.0", "scipy==1.16.0",
        "Send2Trash==1.8.3", "six==1.17.0", "sniffio==1.3.1",
        "soupsieve==2.7", "stack-data==0.6.3", "sympy==1.14.0",
        "tenacity==9.1.2", "termcolor==3.0.1", "terminado==0.18.1",
        "threadpoolctl==3.6.0", "tiktoken==0.9.0", "tinycss2==1.4.0",
        "tokenizers==0.21.1", "torch==2.7.1", "torchvision==0.22.1",
        "tornado==6.5.1", "tqdm==4.67.1", "traitlets==5.14.3",
        "transformers==4.52.4", "triton==3.3.1",
        "types-python-dateutil==2.9.0.20250516", "typing-inspection==0.4.0",
        "typing_extensions==4.13.2", "tzdata==2025.2", "uiautomator2==3.3.3",
        "uri-template==1.3.0", "uritemplate==4.1.1", "urllib3==2.4.0",
        "wcwidth==0.2.13", "webcolors==24.11.1", "webencodings==0.5.1",
        "websocket-client==1.8.0", "Werkzeug==3.1.3",
        "widgetsnbextension==4.0.14", "wrapt==1.17.2", "yarl==1.20.1"
    ]
  
  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)


    pkgs = self.full_package_list.copy()
    random.shuffle(pkgs)

    a, b, c = pkgs[:70], pkgs[70:140], pkgs[140:210]
    extra = pkgs[210:215]         

    
    a += [extra[0], extra[1], extra[3], extra[4]]
    b += [extra[0], extra[2], extra[3], extra[4]]
    c += [extra[1], extra[2], extra[3], extra[4]]
    
   
    self.answer = [extra[3].split('==')[0], extra[4].split('==')[0]]

    print("answer: ", self.answer)
    
    random.shuffle(a)
    random.shuffle(b)
    random.shuffle(c)

 
    folder_names = ["ubuntu", "centos", "redhat", "linux"]
    target_folders = folder_names[:3]  
    lists = [a, b, c]

    for idx, folder in enumerate(folder_names):
        path = file_utils.convert_to_posix_path(device_constants.MARKOR_DATA, folder)
        file_utils.mkdir(path, env.controller)

        if idx < 3:   
            content = "\n".join(f"- {p}" for p in lists[idx])
            file_utils.create_file(
                f"packages_{idx}.md",
                path,
                env.controller,
                content=content
            )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0

    prediction = env.interaction_cache
    gt_answer = self.answer

    pred_packages = set([pkg.strip() for pkg in prediction.split(',')])
    
    answer_packages = set(gt_answer)
      
    if pred_packages == answer_packages:
      return 1
    else:
      return 0



  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}   





class FuncAssistMarkorJoplinTasks(Markor):
  # FuncAssist
  # Cross-app
  app_names = ("markor","joplin","tasks",)
  complexity = 6
  schema = file_validators.CreateFile.schema

  def __init__(self, params: dict[str, Any]):
    """See base class."""
    super().__init__(params)

    self.create_file_task = file_validators.CreateFile(
        params, device_constants.MARKOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)  # Delegate

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
    gt_answer = "Markor, Joplin and Tasks all allow users to create to-do items. Only Tasks allows users to set priority for to-do."
    
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
          'Error during verification for FuncAssistMarkorJoplinTasks: %s', e
      )
      return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    note = _generate_random_note()
    return {"file_name": note.name, "text": note.content}

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.create_file_task.tear_down(env)

  answer = (
    'Markor, Joplin and Tasks all allow users to create to-do items. '
    'Only Tasks allows users to set priority for to-do.'
  )


  @property
  def goal(self) -> str:
    return (
      'Among the three apps—Markor, Joplin, and Tasks,'
      'which one allows you to create to-do items? '
      'Which one lets you set a priority for each to-do ?')



class FuncAssistMarkorExplainOneFunctionality1(Markor):

  app_names = ("markor",)
  complexity = 4
  schema = file_validators.CreateFile.schema

  def __init__(self, params: dict[str, Any]):
    """See base class."""
    super().__init__(params)

    self.create_file_task = file_validators.CreateFile(
        params, device_constants.MARKOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)  # Delegate

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    
    if not env.interaction_cache:
      print(f'Agent did not return textual answer')
      return 0.0
    
    goal = self.goal
    prediction = env.interaction_cache
    gt_answer = 'To create a new note in Markor, follow these steps: 1. Click on the red plus button at the bottom right corner of the screen. 2. Enter the desired file name for the new note. 3. Choose to change the file type and template as well. 4. Click on the \'OK\' button to proceed with the creation of the new note.'
    
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
          'Error during verification for FuncAssistMarkorExplainOneFunctionality1: %s', e
      )
      return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    note = _generate_random_note()
    return {"file_name": note.name, "text": note.content}

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.create_file_task.tear_down(env)


  @property
  def goal(self) -> str:
    return f'I want to create a new note in Markor. Explore this APP and tell me the settings of creating new note.'


class FuncAssistMarkorExplainOneFunctionality3(Markor):

  app_names = ("markor",)
  complexity = 4
  schema = file_validators.CreateFile.schema

  def __init__(self, params: dict[str, Any]):
    """See base class."""
    super().__init__(params)

    self.create_file_task = file_validators.CreateFile(
        params, device_constants.MARKOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)  # Delegate

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.goal
    prediction = env.interaction_cache
    gt_answer = 'It is a checkbox icon, which is used for creating a new note or marking something as checked off.'
    
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
          'Error during verification for FuncAssistMarkorExplainOneFunctionality3: %s', e
      )
      return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    note = _generate_random_note()
    return {"file_name": note.name, "text": note.content}

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.create_file_task.tear_down(env)


  @property
  def goal(self) -> str:
    return f'Open Markor app. On its main screen, tap the third button in the bottom tab bar. A new button row will appear. Report what the first button in this new row does.'


class FuncAssistRefusalMarkorExplainOneFunctionality(Markor):

  app_names = ("markor",)
  complexity = 4
  schema = file_validators.CreateFile.schema

  def __init__(self, params: dict[str, Any]):
    """See base class."""
    super().__init__(params)

    self.create_file_task = file_validators.CreateFile(
        params, device_constants.MARKOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)  # Delegate

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
    gt_answer = "The bottom tab bar has only four tabs. There is no fifth tab."
    
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
          'Error during verification for FuncAssistRefusalMarkorExplainOneFunctionality: %s', e
      )
      return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    note = _generate_random_note()
    return {"file_name": note.name, "text": note.content}

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.create_file_task.tear_down(env)


  @property
  def goal(self) -> str:
    return f'Open Markor app. On its main screen, tap the fifth button in the bottom tab bar. A new button row will appear. Report what the second button in this new row does.'


class FuncAssistMarkorExplainOneFunctionality2(Markor):
  # Answer: cannot add a new file which type is latex and cannot compile it to pdf.
  app_names = ("markor",)
  complexity = 4
  schema = file_validators.CreateFile.schema

  def __init__(self, params: dict[str, Any]):
    """See base class."""
    super().__init__(params)

    self.create_file_task = file_validators.CreateFile(
        params, device_constants.MARKOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)  # Delegate

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.goal
    prediction = env.interaction_cache
    gt_answer = 'Users can choose None for the type when creating a file and then change the file extension to .tex to obtain a LaTeX document, but it cannot be compiled into a PDF file.'
    
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
          'Error during verification for FuncAssistMarkorExplainOneFunctionality2: %s', e
      )
      return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    note = _generate_random_note()
    return {"file_name": note.name, "text": note.content}

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.create_file_task.tear_down(env)


  @property
  def goal(self) -> str:
    return f'I want to create a new latex file in Markor. Can I create a latex and complie it to a PDF?'


class FuncAssistMarkorLocateOneFunctionality1(Markor):

  app_names = ("markor",)
  complexity = 2
  schema = file_validators.CreateFile.schema

  def __init__(self, params: dict[str, Any]):
    """See base class."""
    super().__init__(params)

    self.create_file_task = file_validators.CreateFile(
        params, device_constants.MARKOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)  # Delegate

  def is_successful(self, env: interface.AsyncEnv) -> float:
    """Check if the task is successful by verifying the screenshot.
    
    Args:
      env: The environment interface.
      
    Returns:
      1.0 if successful (screenshot shows the correct interface), 0.0 otherwise.
    """
    super().is_successful(env)
    
    
    goal = self.goal
    task_key = 'FuncAssistMarkorLocateOneFunctionality1'
    
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
          'Error during verification for FuncAssistMarkorLocateOneFunctionality1: %s', e
      )
      return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    note = _generate_random_note()
    return {"file_name": note.name, "text": note.content}

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.create_file_task.tear_down(env)

  @property
  def goal(self) -> str:
    return f'''I want to locate an interface in Markor, where I can config the Action Order of Markdown.'''


class NoiseResistMarkorCreateNoteWithOrientation(Markor):

  app_names = ("markor",)
  complexity = 4.6
  schema = file_validators.CreateFile.schema


  landscape_step = [5,15,25]
  portrait_step = [10,20,30]

  def __init__(self, params: dict[str, Any]):
    """See base class."""
    super().__init__(params)

    self.create_file_task = file_validators.CreateFile(
        params, device_constants.MARKOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)  # Delegate

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return self.create_file_task.is_successful(env)  # Delegate

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    note = _generate_random_note()
    return {"file_name": note.name, "text": note.content}

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.create_file_task.tear_down(env)
  



  def check_status(self, env: interface.AsyncEnv):
    self.is_o_condition(env)

  def is_o_condition(
      self,
      env: interface.AsyncEnv,
  ) -> float:
    if self.current_step in self.landscape_step:
      adb_utils.change_orientation('landscape', env=env.controller)

    if self.current_step in self.portrait_step:
      adb_utils.change_orientation('portrait', env=env.controller)      
    return ''
  

  template = (
      "Create a new note in Markor named {file_name} with the following text:"
      " {text}"
  )




class BrowsecompMarkorNoteUI(Markor):

  app_names = ("markor",)
  complexity = 8
  schema = file_validators.CreateFile.schema
  answer = "Markor"

  def __init__(self, params: dict[str, Any]):
    """See base class."""
    super().__init__(params)

    self.create_file_task = file_validators.CreateFile(
        params, device_constants.MARKOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)  # Delegate

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    if self.answer.lower() in env.interaction_cache.lower():
      return 1
    else:
      return 0
  

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    note = _generate_random_note()
    return {"file_name": note.name, "text": note.content}

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.create_file_task.tear_down(env)


  @property
  def goal(self) -> str:
    return f'''Among Files, Broccoli, Calendar, Markor, Pro Expense, Tasks and Draw, there is an app that has add, delete, and edit functions.
    The main interface of this app contains more functionalities than the others.
    What is this app?
    '''



class MultiRoundMarkorCreateNoteReverse(Markor):
  """Create some notes and then reverse."""

  app_names = ("markor",)
  complexity = 6
  schema = file_validators.CreateFile.schema

  
  template = (
    "In Markor, create three notes:\n"
    "{file_name1}: {text1}\n"
    "{file_name2}: {text2}\n"
    "{file_name3}: {text3}"
    )
  def __init__(self, params: dict[str, Any]):
    """See base class."""
    super().__init__(params)

    self.create_file_task = file_validators.CreateFile(
        params, device_constants.MARKOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    
    user_data_generation.generate_noise_files(
        user_data_generation.generate_random_string(5),
        device_constants.MARKOR_DATA,
        env.controller,
        _NOTE_TITLES,
        random.randint(5, 8),
    )

    file_list = file_utils.get_file_list_with_metadata(
        device_constants.MARKOR_DATA, env.controller
    )
    self.initial_file_list_sorted = sorted(
        file_list, key=lambda f: f.file_name
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    new_file_list = file_utils.get_file_list_with_metadata(
        device_constants.MARKOR_DATA, env.controller
    )
    new_file_list_sorted = sorted(new_file_list, key=lambda f: f.file_name)
    expected_remaining_list = self.initial_file_list_sorted

    if len(expected_remaining_list) != len(new_file_list_sorted):
      return 0.0
    
    for i in range(len(new_file_list_sorted)):
      current_file = new_file_list_sorted[i]
      expected_file = expected_remaining_list[i]
      
      if not (
          current_file.file_name == expected_file.file_name
          and current_file.change_time == expected_file.change_time
      ):
        return 0.0
    return 1

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    note1 = _generate_random_note()
    note2 = _generate_random_note()
    note3 = _generate_random_note()

    return {"file_name1": note1.name, "text1": note1.content, 
            "file_name2": note2.name, "text2": note2.content, 
            "file_name3": note3.name, "text3": note3.content
            }

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.create_file_task.tear_down(env)


  @property
  def goal(self) -> str:
    """The language goal constructed from the template with the params."""
    if self.current_step < 15:
      return self.template.format(**self.params)
    else:
      return (
        "Never mind. "
      "I don't wanna create these notes anymore. "
      "Reverse to the initial state."
      )



class RefusalMarkorDeleteMultipleNotesConflict1(Markor):
  """Task for deleting notes in Markor. 
  But these 3 notes do not exist in the folder."""

  complexity = 2
  schema = {}
  template = (
    "Delete the following notes in "
    "Algo directory of Markor: "
    "{note1}, {note2}, {note3}."
  )

  answer = (
    "The three notes you want to delete do not exist in Algo directory."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)

    # 4 candidate folders
    folder_names = ["Algo"]
    self.target_folder = None       # will be set below
    note_count = 8
    params = self.params  # 或者 getattr(self, "params", {})
    refused_notes = {params["note1"], params["note2"], params["note3"]}


    for folder in folder_names:
      data_directory = device_constants.MARKOR_DATA
      file_directory = file_utils.convert_to_posix_path(data_directory, folder)
      file_utils.mkdir(file_directory, env.controller)
      
      to_create = [name for name in _NOTE_TITLES if name not in refused_notes]
      random.shuffle(to_create)          # 随机排序
      for note_name in to_create:
          content = generate_random_sentence()
          file_utils.create_file(
              note_name,
              file_directory,
              env.controller,
              content=content,
          )
    self.before_file_list = file_utils.get_file_list_with_metadata(
      device_constants.MARKOR_DATA, env.controller
    )


  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    
    after_file_list = file_utils.get_file_list_with_metadata(
        device_constants.MARKOR_DATA, env.controller
      )
    if len(after_file_list) != len(self.before_file_list):
      return 0.0

    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    ply = VerifyPolicy({})
    result = ply.verify_textual_prediction(
            goal=self.goal,
            prediction=env.interaction_cache,
            gt_answer = self.answer,
            judgement_type = 'refusal'
        )
    return result

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    # return {'note1':_NOTE_TITLES[0],'note2':_NOTE_TITLES[1],'note3':_NOTE_TITLES[2]}
    chosen = random.sample(_NOTE_TITLES, 3)
    return {"note1": chosen[0], "note2": chosen[1], "note3": chosen[2]}


class VagueMarkorDeleteNewestTwoNotes(Markor):
  """Task for deleting the newest two note in Markor."""

  complexity = 2
  schema = {}
  template = "Delete the latest two notes."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    self.start_on_home_screen = False
    super().initialize_task(env)
    
    # Generate some random notes in Markor.
    for _ in range(random.randint(4, 8)):
      note = _generate_random_note()
      file_utils.create_file(
          note.name,
          device_constants.MARKOR_DATA,
          env.controller,
          content=note.content,
      )
      # Advance system time so the change time for these initial notes can be
      # separated.
      datetime_utils.advance_system_time(
          datetime.timedelta(minutes=random.randint(2, 50)), env.controller
      )

    file_list = file_utils.get_file_list_with_metadata(
        device_constants.MARKOR_DATA, env.controller
    )
    self.initial_file_list_sorted = sorted(
        file_list, key=lambda f: f.change_time
    )
    if len(self.initial_file_list_sorted) < 4:
        raise RuntimeError("Failed to initialize enough notes for the task.")

    # The initial screenshot of this task should be markor.
    # So the instruction does not indicate markor app.
    adb_utils.launch_app('markor', env.controller)


  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    new_file_list = file_utils.get_file_list_with_metadata(
        device_constants.MARKOR_DATA, env.controller
    )
    new_file_list_sorted = sorted(new_file_list, key=lambda f: f.change_time)
    two_fewer_files = (
        len(new_file_list_sorted) == len(self.initial_file_list_sorted) - 2
    )
    if not two_fewer_files:
        # 
        return 0.0
    
    expected_remaining_list = self.initial_file_list_sorted[:-2]
    
    for i in range(len(new_file_list_sorted)):
      current_file = new_file_list_sorted[i]
      expected_file = expected_remaining_list[i]
      
      if not (
          current_file.file_name == expected_file.file_name
          and current_file.change_time == expected_file.change_time
      ):
        return 0.0
    return 1
  

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    return {}



class VagueMarkorDeleteNewestTwoNotesCHS(VagueMarkorDeleteNewestTwoNotes):
  template = "删除最近创建的两条笔记。"

class VagueMarkorDeleteNewestTwoNotesVariation(VagueMarkorDeleteNewestTwoNotes):
  template = 'Remove the two most recent notes.'

class RefusalMarkorDeleteMultipleNotesAmbigious1(Markor):
  """Task for deleting all notes in Markor."""

  # For this task's complexity, the agent may complete this task by deleting the
  # files one-by-one which envolves many steps (more than 10), but there is also
  # an optimal approach by first long pressing one file, then tapping to select
  # all others and deleting them all together.
  complexity = 2
  schema = {}
  template = "Delete all 3 notes in Algo directory of Markor."
  # which is actually has 4 notes
  answer = 'There are 4 files in the Algo directory. Please specify which 3 files you want me to delete.'

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)

    # 4 candidate folders
    folder_names = ["Math", "Algo"]
    self.target_folder = None       # will be set below

    for folder in folder_names:
      
      data_directory = device_constants.MARKOR_DATA
      file_directory = file_utils.convert_to_posix_path(data_directory, folder)
      file_utils.mkdir(file_directory, env.controller)

      note_count = 4

      for _ in range(note_count):
        note_name = user_data_generation.generate_random_file_name() + ".md"
        content = generate_random_sentence()

        file_utils.create_file(
            note_name,
            file_directory,
            env.controller,
            content=content,
        )
    self.before_file_list = file_utils.get_file_list_with_metadata(
      device_constants.MARKOR_DATA, env.controller
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)

    # Check if any file is deleted wrongly.
    # As the user not specify which files should be deleted, 
    # no files should be deleted.
    after_file_list = file_utils.get_file_list_with_metadata(
        device_constants.MARKOR_DATA, env.controller
      )
    if len(after_file_list) != len(self.before_file_list):
      return 0.0
    

    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    ply = VerifyPolicy({})
    result = ply.verify_textual_prediction(
            goal=self.goal,
            prediction=env.interaction_cache,
            gt_answer = self.answer,
            judgement_type = 'refusal'
        )
    return result

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    return {}

class RefusalMarkorDeleteMultipleNotesAmbigious1CHS(RefusalMarkorDeleteMultipleNotesAmbigious1):
  template = "删除 Markor 中 Algo 目录下的全部 3 条笔记。"

class RefusalMarkorDeleteMultipleNotesAmbigious1Variation(RefusalMarkorDeleteMultipleNotesAmbigious1):
  template = 'Clear out the three notes residing in the Algo folder of Markor.'
  



_NEWS_NAMES = [
    'RIVIAN_STOCK_SURGE',
    'DEEPSEEK_MODEL_RELEASE',
    'CHINA_MARS_SAMPLE_RETURN',
    'TESLA_ROBOTAXI_DELAYS',
    'NVIDIA_ANTITRUST_PROBE',
    'OPENAI_LEADERSHIP_SHUFFLE',
    'GLOBAL_CHIP_DIPLOMACY',
    'APPLE_VISION_PRO_2_LEAK',
    'SAUDI_LINEAR_CITY_FUNDING',
    'EU_GREEN_STEEL_TARIFF'
]

class RefusalMarkorDeleteMultipleNotesAmbigious2(Markor):
  # For this task's complexity, the agent may complete this task by deleting the
  # files one-by-one which envolves many steps (more than 10), but there is also
  # an optimal approach by first long pressing one file, then tapping to select
  # all others and deleting them all together.
  complexity = 2
  schema = {}
  template = (
    "Delete all 3 algorithm notes in directory Favorite of Markor. "
    "These notes are all about algorithms."
  )
  # algorithm notes are not in Favorite, but in Like. ui agent should not delete files in Favorite

  answer = 'According to the names of the three notes in directory Favorite, these notes are not related to algorithms. Are you sure you want to delete these notes?'
  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)

    # 4 candidate folders
    folder_names = ["Favorite", "Like"]
    self.target_folder = None       # will be set below

    for folder in folder_names:
      
      data_directory = device_constants.MARKOR_DATA
      file_directory = file_utils.convert_to_posix_path(data_directory, folder)
      file_utils.mkdir(file_directory, env.controller)

      if folder == "Favorite":
        note_count = 3

        for name in random.sample(_NEWS_NAMES, k=note_count):
          note_name = name + ".md"
          content = generate_random_sentence()
          file_utils.create_file(
              note_name,
              file_directory,
              env.controller,
              content=content,
          )
      
      else:
        note_count = 3
        for name in random.sample(_ALGO_NAMES, k=note_count):
          note_name = name + ".md"
          content = generate_random_sentence()
          file_utils.create_file(
              note_name,
              file_directory,
              env.controller,
              content=content,
          )
    self.before_file_list = file_utils.get_file_list_with_metadata(
      device_constants.MARKOR_DATA, env.controller
    )
    
  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    ply = VerifyPolicy({})
    
    result = ply.verify_textual_prediction(
            goal=self.goal,
            prediction=env.interaction_cache,
            gt_answer = self.answer,
            judgement_type = 'refusal'
        )
  
    if result == 0:
      return 0

    after_file_list = file_utils.get_file_list_with_metadata(
        device_constants.MARKOR_DATA, env.controller
      )
    if len(after_file_list) != len(self.before_file_list):
      return 0.0
    
    return 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    return {}


# MultiRound
class MultiRoundMarkorCreateTwoNote(markor.Markor):

  app_names = ("markor",)
  complexity = 7
  schema = {
      "type": "object",
      "properties": {
          "file_name": {"type": "string"},
          "text": {"type": "string"},
      },
      "required": ["file_name", "text"],
  }
  round = 0
  max_round = 5

  template = (
      "Create a new note in Markor named {file_name} with the following text:"
      " {text}"
  )
  
  template1 = "Make the body text bold."

  template2 =( 
    "Create another note named {file_name_2nd} with the following text: "
    "{text_2nd}"
  )

  template3 = "Delete the body text of second note."

  template4 = "Copy the body content from the previous note into the current note."

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
    elif self.round == 4:
      return self.template4
      

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    user_data_generation.generate_noise_files(
        user_data_generation.generate_random_string(5),
        device_constants.MARKOR_DATA,
        env.controller,
        _NOTE_TITLES,
        random.randint(5, 8),
    )

    file_list = file_utils.get_file_list_with_metadata(
        device_constants.MARKOR_DATA, env.controller
    )
    self.initial_file_list_sorted = sorted(
        file_list, key=lambda f: f.file_name
    )

    
  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if self.round == 0:
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
        # template1 = "Make the body text bold."

    elif self.round == 1:
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
          f"**{self.params['text']}**",
          env.controller,
      ):
        return 0.0
  #       template2 =( 
  #   "Create another note named {file_name_2nd} with the following text: "
  #   "{text_2nd}."
  # )

    elif self.round == 2:
      # 1st note
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
          f"**{self.params['text']}**",
          env.controller,
      ):
        return 0.0
      
      # 2nd note
      if not file_utils.check_file_or_folder_exists(
          self.params["file_name_2nd"],
          device_constants.MARKOR_DATA,
          env.controller,
      ):
        return 0.0
      if not file_utils.check_file_content(
          file_utils.convert_to_posix_path(
              device_constants.MARKOR_DATA, self.params["file_name_2nd"]
          ),
          self.params['text_2nd'],
          env.controller,
      ):
        return 0.0
      
      # template3 = "Delete the body text."
    elif self.round == 3:
      # 1st note
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
          f"**{self.params['text']}**",
          env.controller,
      ):
        return 0.0
      
      # 2nd note
      if not file_utils.check_file_or_folder_exists(
          self.params["file_name_2nd"],
          device_constants.MARKOR_DATA,
          env.controller,
      ):
        return 0.0
      if not file_utils.check_file_content(
          file_utils.convert_to_posix_path(
              device_constants.MARKOR_DATA, self.params["file_name_2nd"]
          ),
         '',
          env.controller,
      ):
        return 0.0
    elif self.round == 4:
      # 1st note
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
          f"**{self.params['text']}**",
          env.controller,
      ):
        return 0.0
      
      # 2nd note
      if not file_utils.check_file_or_folder_exists(
          self.params["file_name_2nd"],
          device_constants.MARKOR_DATA,
          env.controller,
      ):
        return 0.0
      if not file_utils.check_file_content(
          file_utils.convert_to_posix_path(
              device_constants.MARKOR_DATA, self.params["file_name_2nd"]
          ),
         f"**{self.params['text']}**",
          env.controller,
      ):
        return 0.0
    return 1.0
   
  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    # self.markor_task.tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    markor_params = markor.MarkorCreateNote.generate_random_params()

    markor_params_2nd = markor.MarkorCreateNote.generate_random_params()

    subfolders = [
        "BookNotes",
        "CodeSnippets",
        "DailyNotes",
        "FitnessPlans",
        "MeetingMinutes",
        "PersonalJournal",
        "RecipeCollections",
        "StudyGuides",
        "TravelItineraries",
        "WorkProjects",
    ]

    target_folder = random.choice(subfolders)
    compound_params = {
        "file_name": markor_params["file_name"],
        "text": markor_params["text"],
        'file_name_2nd': markor_params_2nd["file_name"],
        'text_2nd': markor_params_2nd["text"],
    }

    return compound_params

# 失败：无法实现heading
class MultiRoundMarkorCreateNoteHeading(markor.Markor):
 
  app_names = ("markor",)
  complexity = 7
  schema = {
      "type": "object",
      "properties": {
          "file_name": {"type": "string"},
          "text": {"type": "string"},
      },
      "required": ["file_name", "text"],
  }
  round = 0
  max_round = 5



  template = (
      "Create a new note in Markor named {file_name} with the following text:"
      " {text}"
  )
  
  template1 = "Change the content to an H1 heading."

  template2 =( 
    'Add this to a new line in the note: {text_2nd}'
  )

  template3 = "Set the newly added content to H2 heading."

  template4 = "Delete the H1 heading content."

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
    elif self.round == 4:
      return self.template4
      

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)

    user_data_generation.generate_noise_files(
        user_data_generation.generate_random_string(5),
        device_constants.MARKOR_DATA,
        env.controller,
        _NOTE_TITLES,
        random.randint(5, 8),
    )

    file_list = file_utils.get_file_list_with_metadata(
        device_constants.MARKOR_DATA, env.controller
    )
    self.initial_file_list_sorted = sorted(
        file_list, key=lambda f: f.file_name
    )
    
  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if self.round == 0:
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
      
    elif self.round == 1:
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
          f"# {self.params['text']}",
          env.controller,
      ):
        return 0.0
      
    elif self.round == 2:
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
          f'''# {self.params['text']}
{self.params['text_2nd']}''',
          env.controller,
      ):
        return 0.0
    
    # template3 = "Set the newly added content to H2 heading."
    elif self.round == 3:
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
          f'''# {self.params['text']}
## {self.params['text_2nd']}''',
          env.controller,
      ):
        return 0.0
      
   
    elif self.round == 4:
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
          f'''## {self.params['text_2nd']}''',
          env.controller,
      ):
        return 0.0
    return 1.0
  
  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    markor_params = markor.MarkorCreateNote.generate_random_params()
    markor_params_2nd = markor.MarkorCreateNote.generate_random_params()

    subfolders = [
        "BookNotes",
        "CodeSnippets",
        "DailyNotes",
        "FitnessPlans",
        "MeetingMinutes",
        "PersonalJournal",
        "RecipeCollections",
        "StudyGuides",
        "TravelItineraries",
        "WorkProjects",
    ]

    target_folder = random.choice(subfolders)
    compound_params = {
        "file_name": markor_params["file_name"],
        "text": markor_params["text"],
        'text_2nd': markor_params_2nd["text"],
    }

    return compound_params



class MultiRoundMarkorAppendCopyRename(markor.Markor):

  app_names = ("markor",)
  complexity = 7
  schema = {
      "type": "object",
      "properties": {
          "file_name": {"type": "string"},
          "text": {"type": "string"},
      },
      "required": ["file_name", "text"],
  }
  round = 0
  max_round = 5

  template = (
      "Create a new note in Markor named {file_name} "
      "with the following text: {text}"
  )

  template1 = (
      "Append the following text to the end of the same note: {append_text}"
  )

  template2 = (
      "Create another note named {file_name_2nd} "
      "with the following text: {text_2nd}"
  )

  template3 = (
      "Append the entire content of the first note to "
      "the beginning of the second note, followed by a newline, "
      "then keep the original content of the second note."
  )

  template4 = (
      "Rename the second note to {file_name_renamed} and move it into the folder {target_folder}."
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
      return self.template3
    elif self.round == 4:
      return self.template4.format(**self.params)

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
   

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)

    if self.round == 0:
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
      
    elif self.round == 1:
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
{self.params['append_text']}''',
          env.controller,
      ):
        return 0.0
  
    elif self.round == 2:
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
{self.params['append_text']}''',
          env.controller,
      ):
        return 0.0
      
      # 2nd note
      if not file_utils.check_file_or_folder_exists(
          self.params["file_name_2nd"],
          device_constants.MARKOR_DATA,
          env.controller,
      ):
        return 0.0
      if not file_utils.check_file_content(
          file_utils.convert_to_posix_path(
              device_constants.MARKOR_DATA, self.params["file_name_2nd"]
          ),
          self.params['text_2nd'],
          env.controller,
      ):
        return 0.0
      
    elif self.round == 3:
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
{self.params['append_text']}''',
          env.controller,
      ):
        return 0.0
      
      # 2nd note
      if not file_utils.check_file_or_folder_exists(
          self.params["file_name_2nd"],
          device_constants.MARKOR_DATA,
          env.controller,
      ):
        return 0.0
      if not file_utils.check_file_content(
          file_utils.convert_to_posix_path(
              device_constants.MARKOR_DATA, self.params["file_name_2nd"]
          ),
          f'''{self.params["text"]}
{self.params['append_text']}
{self.params['text_2nd']}''',
          env.controller,
      ):
        return 0.0 
      
 
    elif self.round == 4:
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
{self.params['append_text']}''',
          env.controller,
      ):
        return 0.0
      
      # 2nd note
      # Should move to new folder.
      if file_utils.check_file_or_folder_exists(
          self.params["file_name_2nd"],
          device_constants.MARKOR_DATA,
          env.controller,
      ):
        return 0.0
      # Should move to new folder.
      if file_utils.check_file_or_folder_exists(
          self.params["file_name_renamed"],
          device_constants.MARKOR_DATA,
          env.controller,
      ):
        return 0.0
      
      if not file_utils.check_file_or_folder_exists(
          self.params["file_name_renamed"],
          file_utils.convert_to_posix_path(
              device_constants.MARKOR_DATA, self.params["target_folder"]
          ),
          env.controller,
      ):
        return 0.0

      if not file_utils.check_file_content(
          file_utils.convert_to_posix_path(
              device_constants.MARKOR_DATA, 
              self.params["target_folder"],
              self.params["file_name_renamed"],
          ),
          f'''{self.params["text"]}
{self.params['append_text']}
{self.params['text_2nd']}''',
          env.controller,
      ):
        return 0.0 
    return 1.0
  
  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
   

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    markor_params_1st = markor.MarkorCreateNote.generate_random_params()
    markor_params_1st_append = markor.MarkorCreateNote.generate_random_params()
    markor_params_2nd = markor.MarkorCreateNote.generate_random_params()
    markor_params_renamed = markor.MarkorCreateNote.generate_random_params()

    subfolders = [
        "BookNotes",
        "CodeSnippets",
        "DailyNotes",
        "FitnessPlans",
        "MeetingMinutes",
        "PersonalJournal",
        "RecipeCollections",
        "StudyGuides",
        "TravelItineraries",
        "WorkProjects",
    ]
    target_folder = random.choice(subfolders)

    compound_params = {
        "file_name": markor_params_1st["file_name"],
        "text": markor_params_1st["text"],
        "append_text": markor_params_1st_append["text"],
        "file_name_2nd": markor_params_2nd["file_name"],
        "text_2nd": markor_params_2nd["text"],
        "file_name_renamed": markor_params_renamed["file_name"],
        "target_folder": target_folder,
    }
    return compound_params

class MultiRoundMarkorAppendCopyRenameCHS(MultiRoundMarkorAppendCopyRename):
  template = (
      "在 Markor 中创建一个名为 {file_name} 的新笔记，内容如下：{text}。"
  )

  template1 = (
      "将以下文本追加到同一笔记的末尾：{append_text}。"
  )

  template2 = (
      "创建另一个名为 {file_name_2nd} 的新笔记，内容如下：{text_2nd}。"
  )

  template3 = (
    "将第一条笔记的全部内容复制并粘贴到第二条笔记的顶部，"
    "后跟一个换行符，然后保留第二条笔记的原有内容。"
  )

  template4 = (
      "将第二条笔记重命名为 {file_name_renamed} 并将其移动到文件夹 {target_folder} 中。"
  )

class MultiRoundMarkorAppendCopyRenameVariation(MultiRoundMarkorAppendCopyRename):
  template = (
      "In Markor, create a new note titled {file_name} containing the following text: {text}"
  )

  template1 = (
      "Add the following text to the end of that note: {append_text}"
  )

  template2 = (
      "Create another note called {file_name_2nd} with this text: {text_2nd}"
  )

 
  template3 = (
    "Take all the content from the first note and paste it at the beginning of the second note, "
    "followed by a newline, keeping the original content of the second note intact."
  )


  template4 = (
      "Change the name of the second note to {file_name_renamed} and relocate it to the folder {target_folder}."
  )

  

class MultiRoundRefusalMarkorCreateNoteHeading(markor.Markor):
 
  app_names = ("markor",)
  complexity = 7
  schema = {
      "type": "object",
      "properties": {
          "file_name": {"type": "string"},
          "text": {"type": "string"},
      },
      "required": ["file_name", "text"],
  }
  round = 0
  max_round = 5



  template = "Create a new note in Markor named {file_name}"
  
  template1 = "Delete the body text."

  template2 =( 
    "Continue adding the following text to the note: "
    "{text_2nd}"
  )

  template3 = "Delete the body text."

  template4 = "Set the newly added content to H3 heading."

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
    elif self.round == 4:
      return self.template4
      

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.markor_task = markor.MarkorCreateNote(
        params={
            "file_name": self.params["file_name"],
            "text": self.params["text"],
        }
    )
    self.markor_task.initialize_task(env)

    
  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    markor_success = self.markor_task.is_successful(env)
    logging.info("Markor success: %s", markor_success)


    return markor_success

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    self.markor_task.tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    markor_params = markor.MarkorCreateNote.generate_random_params()
    markor_params_2nd = markor.MarkorCreateNote.generate_random_params()

    subfolders = [
        "BookNotes",
        "CodeSnippets",
        "DailyNotes",
        "FitnessPlans",
        "MeetingMinutes",
        "PersonalJournal",
        "RecipeCollections",
        "StudyGuides",
        "TravelItineraries",
        "WorkProjects",
    ]

    target_folder = random.choice(subfolders)
    compound_params = {
        "file_name": markor_params["file_name"],
        "text": markor_params["text"],
        'text_2nd': markor_params_2nd["text"],
    }

    return compound_params


class MultiRoundMarkorAdd4NotesDeleteNote(markor.Markor):

  app_names = ("markor",)
  complexity = 7
  schema = {
      "type": "object",
      "properties": {
          "file_name": {"type": "string"},
          "text": {"type": "string"},
      },
      "required": ["file_name", "text"],
  }
  round = 0
  max_round = 5

  template = (
      "Create a new note in Markor named {file_name} with the following text:"
      " {text}"
  )
  
  template1 = ( 
    "Create another note named {file_name_2nd} with the following text: "
    "{text_2nd}"
  )

  template2 =( 
    "Create another note named {file_name_3rd} with the following text: "
    "{text_3rd}"
  )

  template3 = ( 
    "Create another note named {file_name_4th} with the following text: "
    "{text_4th}"
  )

  template4 = "Delete the frist two notes."

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
    elif self.round == 4:
      return self.template4
      

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    user_data_generation.generate_noise_files(
        user_data_generation.generate_random_string(5),
        device_constants.MARKOR_DATA,
        env.controller,
        _NOTE_TITLES,
        random.randint(2, 6),
    )

    file_list = file_utils.get_file_list_with_metadata(
        device_constants.MARKOR_DATA, env.controller
    )

    if not file_list:
      raise RuntimeError("Something went wrong, file was not created.")
    self.initial_file_list_sorted = sorted(
        file_list, key=lambda f: f.file_name
    )
    
  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if self.round == 0:
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
    elif self.round == 1:
      if not file_utils.check_file_or_folder_exists(
          self.params["file_name_2nd"],
          device_constants.MARKOR_DATA,
          env.controller,
      ):
        return 0.0
      if not file_utils.check_file_content(
          file_utils.convert_to_posix_path(
              device_constants.MARKOR_DATA, self.params["file_name_2nd"]
          ),
          self.params["text_2nd"],
          env.controller,
      ):
        return 0.0
    elif self.round == 2:
      if not file_utils.check_file_or_folder_exists(
          self.params["file_name_3rd"],
          device_constants.MARKOR_DATA,
          env.controller,
      ):
          return 0.0
      if not file_utils.check_file_content(
          file_utils.convert_to_posix_path(
              device_constants.MARKOR_DATA, self.params["file_name_3rd"]
          ),
          self.params["text_3rd"],
          env.controller,
      ):
          return 0.0
    elif self.round == 3:
      if not file_utils.check_file_or_folder_exists(
          self.params["file_name_4th"],
          device_constants.MARKOR_DATA,
          env.controller,
      ):
          return 0.0
      if not file_utils.check_file_content(
          file_utils.convert_to_posix_path(
              device_constants.MARKOR_DATA, self.params["file_name_4th"]
          ),
          self.params["text_4th"],
          env.controller,
      ):
          return 0.0
    elif self.round == 4:
      new_file_list = file_utils.get_file_list_with_metadata(
       device_constants.MARKOR_DATA, env.controller
      )
      new_file_list_sorted = sorted(new_file_list, key=lambda f: f.file_name)
      # Finally only 3rd file and 4th file added.
      if len(new_file_list_sorted)-len(self.initial_file_list_sorted) != 2:
        return 0.0

    
      if file_utils.check_file_or_folder_exists(
          self.params["file_name"],
          device_constants.MARKOR_DATA,
          env.controller,
      ):
          return 0.0
      if file_utils.check_file_or_folder_exists(
          self.params["file_name_2nd"],
          device_constants.MARKOR_DATA,
          env.controller,
      ):
          return 0.0

      
      if not file_utils.check_file_or_folder_exists(
          self.params["file_name_3rd"],
          device_constants.MARKOR_DATA,
          env.controller,
      ):
          return 0.0
      if not file_utils.check_file_or_folder_exists(
          self.params["file_name_4th"],
          device_constants.MARKOR_DATA,
          env.controller,
      ):
          return 0.0

    return 1.0

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    markor_params = markor.MarkorCreateNote.generate_random_params()

    markor_params_2nd = markor.MarkorCreateNote.generate_random_params()

    markor_params_3nd = markor.MarkorCreateNote.generate_random_params()

    markor_params_4th = markor.MarkorCreateNote.generate_random_params()

    subfolders = [
        "BookNotes",
        "CodeSnippets",
        "DailyNotes",
        "FitnessPlans",
        "MeetingMinutes",
        "PersonalJournal",
        "RecipeCollections",
        "StudyGuides",
        "TravelItineraries",
        "WorkProjects",
    ]

    target_folder = random.choice(subfolders)
    compound_params = {
        "file_name": markor_params["file_name"],
        "text": markor_params["text"],
        'file_name_2nd': markor_params_2nd["file_name"],
        'text_2nd': markor_params_2nd["text"],
        'file_name_3rd': markor_params_3nd["file_name"],
        'text_3rd': markor_params_3nd["text"],
        'file_name_4th': markor_params_4th["file_name"],
        'text_4th': markor_params_4th["text"],
    }

    return compound_params



class MultiRoundFuncAssistMarkor(markor.Markor):

  app_names = ("markor",)
  complexity = 8
  schema = {}
  round = 0
  max_round = 3

  template = "I want to create a new note in Markor. Explore this APP and tell me the detailed settings of creating new note."
  
  template1 = "I want to create a new latex file in Markor. Can I create a latex and complie it to a PDF?"

  template2 = "Open Markor app. On its main screen, tap the third button in the bottom tab bar. A new button row will appear. Report what the second button in this new row does."
   
  
  @property
  def goal(self) -> str:
    """The language goal constructed from the template with the params."""
    if self.round == 0:
      return self.template.format(**self.params)
    elif self.round == 1:
      return self.template1.format(**self.params)
    elif self.round == 2:
      return self.template2
      

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.markor_task = markor.MarkorCreateNote(
        params={}
    )
    self.markor_task.initialize_task(env)

    
  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    markor_success = self.markor_task.is_successful(env)
    logging.info("Markor success: %s", markor_success)


    return markor_success

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    self.markor_task.tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    markor_params = markor.MarkorCreateNote.generate_random_params()

    markor_params_2nd = markor.MarkorCreateNote.generate_random_params()

    subfolders = [
        "BookNotes",
        "CodeSnippets",
        "DailyNotes",
        "FitnessPlans",
        "MeetingMinutes",
        "PersonalJournal",
        "RecipeCollections",
        "StudyGuides",
        "TravelItineraries",
        "WorkProjects",
    ]

    target_folder = random.choice(subfolders)
    compound_params = {
    }

    return compound_params