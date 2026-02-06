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

"""Tasks for Tasks app."""

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
from android_world.policy.verification import VerifyPolicy
import datetime
import random
import uuid

_PRIMARY_KEY = '_id'
_TASK_TABLE = 'tasks'
_DB_PATH = '/data/data/org.tasks/databases/database'
_APP_NAME = 'tasks'

def add_tasks(
    rows: list[sqlite_schema_utils.Task], env: interface.AsyncEnv
) -> None:
  sqlite_utils.insert_rows_to_remote_db(
      rows,
      _PRIMARY_KEY,
      _TASK_TABLE,
      _DB_PATH,
      _APP_NAME,
      env,
  )
  adb_utils.close_app(_APP_NAME, env.controller)  # Register changes.

def clear_task_db(env: interface.AsyncEnv) -> None:
  """Clears the task database."""
  sqlite_utils.delete_all_rows_from_table(_TASK_TABLE, _DB_PATH, env, _APP_NAME)
  adb_utils.close_app(_APP_NAME, env.controller)  # Register changes.

def list_rows(
    env: interface.AsyncEnv,
) -> list[sqlite_schema_utils.Task]:
  return sqlite_utils.get_rows_from_remote_device(
      _TASK_TABLE,
      _DB_PATH,
      sqlite_schema_utils.Task,
      env,
  )

_TASKS = {
    'Grocery Shopping': (
        'Don\t forget milk, eggs, and bread. Also need to pick up snacks for'
        ' the kids.'
    ),
    'Finish Project Proposal': (
        'Deadline is Friday. Need to finalize budget and timeline sections.'
    ),
    'Schedule Dentist Appointment': (
        "Teeth cleaning overdue. Call Dr. Smith's office."
    ),
    'Water Plants': (
        'Check moisture level before watering. Fertilize succulents.'
    ),
    'Meal Prep for the Week': (
        'Make a grocery list based on planned meals. Cook chicken and chop'
        ' veggies on Sunday.'
    ),
    'Research Vacation Destinations': (
        'Looking for beach destinations with family-friendly activities.'
    ),
    "Read 'The Martian'": 'Started last week. Aim to finish by next weekend.',
    'Call Grandma': (
        'Catch up on family news. Ask for her famous cookie recipe.'
    ),
    'Change Air Filter': (
        'Last changed 3 months ago. Buy a new filter at the hardware store.'
    ),
    'Brainstorm Blog Post Ideas': (
        "Need 5 new topics for the next month's content calendar."
    ),
    "Renew Driver's License": (
        'Expires next month. Check DMV website for requirements.'
    ),
    'Organize Closet': (
        'Donate old clothes and shoes. Put winter clothes in storage.'
    ),
    'Submit Expense Report': (
        'Deadline is Wednesday. Attach receipts for all purchases.'
    ),
    'Attend Team Meeting': (
        'Agenda includes project updates and brainstorming new initiatives.'
    ),
    'Learn to Play Guitar': (
        'Practice chords for 30 minutes every day. Find online tutorials.'
    ),
    'Reply to Emails': (
        'Inbox is overflowing. Prioritize urgent messages and unsubscribe from'
        ' unwanted lists.'
    ),
    'Clean Out Fridge': (
        'Check expiration dates and discard old food. Wipe down shelves.'
    ),
    'Create Budget for Next Month': (
        'Track income and expenses. Set savings goals.'
    ),
    'Back Up Computer Files': (
        'Use external hard drive or cloud storage. Schedule regular backups.'
    ),
    'Take Dog to the Vet': 'Annual checkup and vaccinations due.',
}
 

def create_task_from_proto(
    task: state_pb2.TasksAppTask,
) -> sqlite_schema_utils.Task:
  """Creates a Task object from a state_pb2.TasksAppTask proto."""
  due_date_ts = 0
  hide_until_date_ts = 0
  completed_date_ts = 0
  if task.HasField('due_date'):
    due_date_ts = (
        calendar_utils.convert_datetime_to_unix_ts(task.due_date, task.due_time)
        * 1000
    )
  # created date is 1 week before due date.
  created_date_ts = due_date_ts - 7 * 3600
  if task.HasField('hide_until_date'):
    hide_until_date_ts = (
        calendar_utils.convert_datetime_to_unix_ts(
            task.hide_until_date, task.hide_until_time
        )
        * 1000
    )
  if task.HasField('completed_date'):
    completed_date_ts = (
        calendar_utils.convert_datetime_to_unix_ts(
            task.completed_date, task.completed_time
        )
        * 1000
    )
  notes = task.notes if task.HasField('notes') else None
  importance = int(task.importance) if task.HasField('importance') else 2
  return sqlite_schema_utils.Task(
      title=task.title,
      importance=importance,
      dueDate=due_date_ts,
      hideUntil=hide_until_date_ts,
      completed=completed_date_ts,
      created=created_date_ts,
      modified=created_date_ts,
      notes=notes,
      remoteId=str(uuid.uuid4().int),
      recurrence=None,
  )
def _generate_random_task():
  """Generates a single random sqlite_schema_utils.Task object."""
  new_task = state_pb2.TasksAppTask()
  new_task.title = random.choice(list(_TASKS.keys()))
  if random.choice([True, False]):
    new_task.notes = _TASKS[new_task.title]
  new_task.importance = str(random.choice([0, 1, 2, 3]))
  random_due_datetime = datetime_utils.generate_random_datetime()
  new_task.due_date = random_due_datetime.date().strftime(
      datetime_utils_ir.DATE_FORMAT
  )
  new_task.due_time = random_due_datetime.time().strftime('%H:%M')
  # Make sure the hide_until_date is before the due date
  random_hide_until_datetime = datetime_utils.generate_random_datetime(
      window_size=datetime.timedelta(days=5),
      window_center=random_due_datetime - datetime.timedelta(days=6),
  )
  new_task.hide_until_date = random_hide_until_datetime.date().strftime(
      '%B %d %Y'
  )
  new_task.hide_until_time = random_hide_until_datetime.time().strftime('%H:%M')
  is_completed = random.choice([True, False])
  if is_completed:
    # Make sure completed date is before current time
    random_completed_datetime = datetime_utils.generate_random_datetime(
        window_center=device_constants.DT - datetime.timedelta(days=14)
    )
    new_task.completed_date = random_completed_datetime.date().strftime(
        datetime_utils_ir.DATE_FORMAT
    )
    new_task.completed_time = random_completed_datetime.time().strftime('%H:%M')
  return create_task_from_proto(new_task)


def generate_random_tasks(
    num_tasks: int,
    exclusion_conditions: list[task_pb2.ExclusionCondition],
) -> list[sqlite_schema_utils.Task]:
  """Generates random tasks with the given exclusion conditions."""
  return sqlite_schema_utils.get_random_items(
      num_tasks,
      generate_item_fn=_generate_random_task,
      filter_fn=lambda x: check_task_conditions(x, exclusion_conditions),
  )

def check_task_conditions(
    task: sqlite_schema_utils.Task,
    exclusion_conditions: list[task_pb2.ExclusionCondition],
) -> bool:
  """Evaluates the specified task against a set of exclusion conditions.

  A task is considered eligible if it does not satisfy all of the conditions
  specified in the exclusion_conditions list. Each condition is checked against
  various fields of the task such as importance, completed date, due date, and
  title. The task is eligible if not all of these conditions are met, ensuring
  it doesn't fall under any exclusion criteria defined.

  Args:
    task: The task to check
    exclusion_conditions: All the conditions the task will be checked against,
      if they are all met, this task should be excluded and does not meet the
      conditions.

  Returns:
    A bool, True if the task does not meet all of the exclusion conditions,
    False otherwise.
  """
  if not exclusion_conditions:
    return True
  # Keeps track of whether an exclusion condition is met.
  all_conditions_met = True
  for condition in exclusion_conditions:
    if condition.field == 'due_date':
      condition_value = datetime_utils_ir.get_date(condition.value)
      due_datetime = datetime_utils.timestamp_to_localized_datetime(
          int(task.dueDate / 1000)
      )

      all_conditions_met = all_conditions_met and proto_utils.compare(
          due_datetime.date(), condition.operation, condition_value
      )
    if condition.field == 'completed_date':
      completed_datetime = datetime_utils.timestamp_to_localized_datetime(
          int(task.completed / 1000)
      )
      # Tasks app uses 0 in completed_date to indicate it's not complete.
      if condition.value == '0':
        all_conditions_met = all_conditions_met and proto_utils.compare(
            int(task.completed), condition.operation, 0
        )
      else:
        condition_value = datetime_utils_ir.get_date(condition.value)
        all_conditions_met = all_conditions_met and proto_utils.compare(
            completed_datetime.date(), condition.operation, condition_value
        )

    elif condition.field == 'title':
      all_conditions_met = all_conditions_met and proto_utils.compare(
          task.title.lower(), condition.operation, condition.value.lower()
      )
    elif condition.field == 'importance':
      all_conditions_met = all_conditions_met and proto_utils.compare(
          int(task.importance), condition.operation, int(condition.value)
      )

  return not all_conditions_met


class _Tasks(task_eval.TaskEval):
  """Base class for Tasks tasks."""

  app_names = ("tasks",)
  template = ''
  schema = {}
  complexity = 1.0  # Overridden in the registry.

  
  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().__init__(self.params)

    super().initialize_task(env)
   
    clear_task_db(env)
    tasks = []
    tasks += generate_random_tasks(
      20,
      [
        task_pb2.ExclusionCondition(
          field='due_date',
          value='October 15 2023',
          operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
        )
      ]
    )
    random.shuffle(tasks)
    add_tasks(tasks, env)

    self.before_task_list = list_rows(env=env)



  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return 0.0

  def tear_down(self, env: interface.AsyncEnv) -> None:
    task_app_utils.clear_task_db(env)
    super().tear_down(env)


def _maybe_replace_date(params: dict[str, Any]) -> None:
  """Maybe replaces date parameters with a natural language equivalent."""
  for param_name, param_value in params.items():
    if param_name == 'seed':
      continue
    if not isinstance(param_value, str):
      continue
    try:
      if not param_value:
        continue
      params[param_name] = datetime_utils_ir.generate_reworded_date(param_value)
    except ValueError:
      pass  # Skip if there's no date parameter.



class FuncAssistLocateTasksInterface1(_Tasks):
  """Task eval for Tasks app"""

  complexity = 2
  schema = {}
  template = 'Navigate to the setting in Tasks APP that allows you to change the task sort order and show me that interface.'
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    """Check if the task is successful by verifying the screenshot.
    
    Args:
      env: The environment interface.
      
    Returns:
      1.0 if successful (screenshot shows the correct interface), 0.0 otherwise.
    """
    super().is_successful(env)
    
    goal = self.template
    task_key = 'FuncAssistLocateTasksInterface1'
    
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
          'Error during verification for FuncAssistLocateTasksInterface1: %s', e
      )
      return 0.0



class FuncAssistTasksExplainOneFunctionality1(_Tasks):
  """Task eval for Tasks app"""

  complexity = 4
  schema = {}
  template = 'In the Tasks app, start creating a new task and examine every editable field or toggle that appears on the creation screen. Do not save the task. Instead, return a  list of every item you can configure, with a concise explanation of what each item does.'
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}
  
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
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = "Set the task's start and end times, whether it repeats, task priority (four levels), add a location, add tags, choose a local list, add subtasks, enable or disable reminders, add attachments, add a description, choose whether to add it to the calendar, and set a timer."
    
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
          'Error during verification for FuncAssistTasksExplainOneFunctionality1: %s', e
      )
      return 0.0


class AskTasksOneFunctionality1(_Tasks):
  """Task eval for Tasks app"""

  complexity = 3
  schema = {}
  
  template = 'In the Tasks app, can you sort the tasks by priority within the Recently Modified view?'
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return 0.0
  
TASKS = [
    "Finish quarterly budget report",
    "Call Mom on her birthday",
    "Book flights for Tokyo conference",
    "Replace living-room lightbulb",
    "Submit pull request for feature #2048",
    "Renew car insurance by June 30",
    "Schedule dentist appointment",
    "Buy anniversary gift",
    "Water the succulents every Monday",
    "Prepare presentation slides for Monday's meeting",
    "Cancel gym membership before next billing cycle",
    "Read chapter 5 of 'Clean Code'",
    "Update résumé and LinkedIn profile",
    "Send thank-you email to Sarah after interview",
    "Pick up dry-cleaning by Friday",
    "Renew passport before October trip",
    "Install latest Android security patch",
    "Organize Google Drive folder structure",
    "Review intern’s pull requests by EOD",
    "Pay electricity bill before due date",
    "Plan weekend hiking route",
    "Back up photos to external SSD",
    "Review Q3 marketing metrics",
    "Buy groceries for Taco Tuesday",
    "Write 500 words for daily journal",
    "Schedule team off-site for next month",
    "Register for online Python workshop",
    "Set up 2-factor auth on all work accounts",
    "Service bike before summer season",
    "Donate old clothes to Goodwill",
    "Draft proposal for client XYZ",
    "Test new espresso machine",
    "Follow up on overdue invoice #INV-2024-0156",
    "Prepare tax documents for accountant",
    "Reserve campsite at Yosemite"
]


class RefusalTasksAddOneTaskAmbigious1(_Tasks):
  """Task eval for Tasks app"""

  complexity = 2
  schema = {}
  template = (
    'Open the Tasks app and add a new task named "{task_name_str}". '
    'Set its priority to the highest level—remember to '
    'tap the blue circle for priority.'
  )
  answer = (
    'I cannot finish your instruction '
    'because the blue circle dose not present as the highest level priority.'
    'Instead, setting the task\'s priority should tap the red circle button.'
    'Please specify your instruction.'
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {'task_name_str': random.choice(TASKS)}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    
    after_task_list = list_rows(env=env)
    if len(after_task_list) != len(self.before_task_list):
      return 0
    
    ply = VerifyPolicy({})
    result = ply.verify_textual_prediction(
            goal=self.goal,
            prediction=env.interaction_cache,
            gt_answer= self.answer,
            judgement_type = 'refusal'
        )
    return result

  