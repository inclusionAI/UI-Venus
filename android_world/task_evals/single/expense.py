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


"""Tasks for managing expenses in an expense app."""
from android_world.utils import fuzzy_match_lib
from datetime import datetime, timedelta

import abc
import dataclasses
import random
from typing import Any, Optional

from absl import logging
from android_world.policy.verification import VerifyPolicy
from android_world.env import device_constants
from android_world.env import interface
from android_world.env.setup_device import apps
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import sqlite_utils
from android_world.task_evals.utils import user_data_generation
from android_world.utils import datetime_utils
from android_world.utils import file_utils
from android_world.env import representation_utils
from android_world.env import adb_utils
from android_world.task_evals.single import expense

_DB_PATH = '/data/data/com.arduia.expense/databases/accounting.db'
_TABLE_NAME = 'expense'
_APP_NAME = 'pro expense'
_DB_KEY = 'expense_id'

# How to represent recipes in text form.
_TEXT_REPRESENTATION_TYPE = 'text_representation_type'


def _get_random_timestamp() -> int:
  """Gets a timestep in the current month, up to the current day (Oct 15)."""
  return datetime_utils.create_random_october_2023_unix_ts(
      start_day=1, end_day=15
  )


class _Expense(task_eval.TaskEval, abc.ABC):
  """Base class for expense logic task evals."""

  # From TaskEval.
  schema = {}
  app_names = (_APP_NAME,)
  template = ''  # Unused, since we directly build goal in implementations.

  app_name_with_db = _APP_NAME
  db_key = _DB_KEY
  db_path = _DB_PATH
  table_name = _TABLE_NAME
  row_type = sqlite_schema_utils.Expense

  def initialize_task(self, env: interface.AsyncEnv):
    if not sqlite_utils.table_exists(self.table_name, self.db_path, env):
      apps.ExpenseApp.setup(env)
    print("_Expense.initialize_task")
    print("_Expense super().initialize_task")
    super().initialize_task(env)


  def _expense_name_is_entered(
      self, 
      expense_name: str,
      ui_elements: list[representation_utils.UIElement],
  ) -> bool:
    """Checks if UI elements contain requested contact info.

    Returns:
      True if contact form is filled out.
    """
    name_element = None
    
    "UIElement(text='Mortgage', content_description='', class_name='android.widget.EditText', bbox=BoundingBox(x_min=63, x_max=1017, y_min=352, y_max=509), bbox_pixels=BoundingBox(x_min=63, x_max=1017, y_min=352, y_max=509), hint_text=None, is_checked=False, is_checkable=False, is_clickable=True, is_editable=None, is_enabled=True, is_focused=False, is_focusable=True, is_long_clickable=True, is_scrollable=False, is_selected=False, is_visible=True, package_name='com.arduia.expense', resource_name=None, tooltip=None, resource_id='com.arduia.expense:id/edt_name', metadata=None)"
    for element in ui_elements:
      if (
          element.text
          and fuzzy_match_lib.fuzzy_match(element.text, expense_name)
      ):
        print(element)
        name_element = element

    if (
        name_element is None
    ):
      return False
    return True


class _ExpenseDeleteMultiple(_Expense, sqlite_validators.DeleteMultipleRows):
  """Task to delete multiple expenses in an expense tracking app."""

  complexity = 2
  n_rows = 3  # Default number of expenses to delete
  n_rows_noise = 0  # Default noise rows

  @property
  def goal(self) -> str:
    targets = self.params[sqlite_validators.ROW_OBJECTS]
    expense_names = [expense.name for expense in targets]
    expense_names_str = ', '.join(expense_names)
    return (
        f'Delete the following expenses from {_APP_NAME}: {expense_names_str}.'
    )

  def validate_deletion_integrity(
      self,
      before: list[sqlite_schema_utils.Expense],
      after: list[sqlite_schema_utils.Expense],
  ) -> bool:
    """Validates the integrity of the expense deletion."""
    return sqlite_validators.validate_rows_removal_integrity(
        before,
        after,
        [expense.expense_id for expense in self.rows_to_delete],
        self.db_key,
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a remove expense task."""

    expenses = []
    while len(expenses) < cls.n_rows + cls.n_rows_noise:
      candidate = _generate_expense()
      if not any([candidate.name == expense.name for expense in expenses]):
        expenses.append(candidate)

    if cls.n_rows_noise > 0:
      target_rows = expenses[: cls.n_rows]
      noise_rows = expenses[cls.n_rows :]
      return {
          sqlite_validators.ROW_OBJECTS: target_rows,
          sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
      }
    else:
      return {
          sqlite_validators.ROW_OBJECTS: expenses,
      }


class ExpenseDeleteSingle(_ExpenseDeleteMultiple):
  """Task to delete a single expense in an expense tracking app."""

  complexity = 1
  n_rows = 1
  n_rows_noise = 0





class ExpenseDeleteMultiple(_ExpenseDeleteMultiple):
  """Task to delete multiple expenses in an expense tracking app."""

  complexity = 2
  n_rows = 3
  n_rows_noise = 0



class _ExpenseDeleteDuplicates(_Expense, sqlite_validators.DeleteDuplicateRows):
  """Deduplicate expenses in the expense tracking app with some noise."""

  complexity = 1.2
  n_rows = 1  # Number of unique expenses to duplicate for the task
  n_rows_noise = 5  # Number of additional unique expenses to include as noise

  @property
  def goal(self) -> str:
    return (
        f'Delete all but one of any expenses in {_APP_NAME} that are exact'
        ' duplicates, ensuring at least one instance of each unique expense'
        ' remains.'
    )

  def validate_deletion_integrity(
      self,
      before: list[sqlite_schema_utils.Expense],
      after: list[sqlite_schema_utils.Expense],
  ) -> bool:
    """Validates the integrity of the expense deletion."""
    target1, target2 = self.rows_to_delete
    return sqlite_validators.validate_rows_removal_integrity(
        before, after, [target1.expense_id], self.db_key
    ) or sqlite_validators.validate_rows_removal_integrity(
        before, after, [target2.expense_id], self.db_key
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a remove duplicate expense task."""
    rows = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise + cls.n_rows,
        _generate_expense,
        replacement=False,
    )
    target = rows.pop()
    return {
        sqlite_validators.ROW_OBJECTS: [target, target],
        sqlite_validators.NOISE_ROW_OBJECTS: rows,
    }


class ExpenseDeleteDuplicates(_ExpenseDeleteDuplicates):
  """Deduplicate expenses in the expense tracking app with some noise."""

  n_rows = 1
  n_rows_noise = 5


class ExpenseDeleteDuplicates2(_ExpenseDeleteDuplicates):
  """Harder task to deduplicate expenses in the expense tracking app."""

  n_rows = 1
  n_rows_noise = 40
  complexity = 1.8

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a remove duplicate expense task."""
    assert cls.n_rows == 1
    noise = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise + cls.n_rows - 3,
        _generate_expense,
        replacement=False,
    )
    target = noise.pop()
    perturbations = random.sample(range(50, 1000), 3)
    target_varations = []
    for perturbation in perturbations:
      target_varations.append(
          dataclasses.replace(
              target,
              amount=target.amount + perturbation,
              created_date=_get_random_timestamp() * 1000,
              modified_date=_get_random_timestamp() * 1000,
          )
      )

    return {
        sqlite_validators.ROW_OBJECTS: [target, target],
        sqlite_validators.NOISE_ROW_OBJECTS: noise + target_varations,
    }


def _get_expense_rows_as_text(
    rows: list[sqlite_schema_utils.Expense],
    format_type: str,
    wrap_width: int | None = None,
) -> str:
  return sqlite_schema_utils.get_text_representation_of_rows(
      rows,
      [
          'name',
          'amount_dollars',
          'category_name',
          'note',
      ],
      format_type,
      'name',
      wrap_width=wrap_width,
  )


class _ExpenseAddMultiple(_Expense, sqlite_validators.AddMultipleRows):
  """Task to add multiple expenses in the Expense Tracking App."""

  complexity = 3
  n_rows = 3
  n_rows_noise = 10

  @property
  def goal(self) -> str:
    text_repr = _get_expense_rows_as_text(
        self.params[sqlite_validators.ROW_OBJECTS],
        self.params[_TEXT_REPRESENTATION_TYPE],
    )
    return f'Add the following expenses into the {_APP_NAME}:\n{text_repr}'

  def validate_addition_integrity(
      self,
      before: list[sqlite_schema_utils.Expense],
      after: list[sqlite_schema_utils.Expense],
      reference_rows: list[sqlite_schema_utils.Expense],
  ) -> bool:
    """Validates the integrity of the expense addition."""
    return sqlite_validators.validate_rows_addition_integrity(
        before,
        after,
        reference_rows,
        compare_fields=[
            'name',
            'amount',
            'category',
            'note',
        ],
        free_form_fields=[
            'name',
            'note',
        ],
    )

  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.Expense:
    return _generate_expense()

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for an add expense task."""
    target_rows = sqlite_schema_utils.get_random_items(
        cls.n_rows,
        cls._get_random_target_row,
        replacement=False,
    )
    noise_rows = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        cls._get_random_target_row,
        replacement=False,
        filter_fn=lambda r: all(r.name != t.name for t in target_rows),
    )
    return {
        sqlite_validators.ROW_OBJECTS: target_rows,
        sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
        _TEXT_REPRESENTATION_TYPE: random.choice(['csv', 'text_block']),
    }


class ExpenseAddSingle(_ExpenseAddMultiple):
  """Task to add a single expense in the Expense Tracking App."""
  
  complexity = 1.2
  n_rows = 1
  n_rows_noise = 10


class ExpenseAddSingleEnhanced(_ExpenseAddMultiple):
  """Task to add a single expense in the Expense Tracking App. Setting time and date."""

  complexity = 4
  n_rows = 1
  n_rows_noise = 10
  @property
  def goal(self) -> str:
    text_repr = _get_expense_rows_as_text(
        self.params[sqlite_validators.ROW_OBJECTS],
        self.params[_TEXT_REPRESENTATION_TYPE],
    )
    # return f'Add the following expenses into the {_APP_NAME}:\n{text_repr}. \nAnd set time at 1:20 PM. Set data as the last Monday of Sep 2025.'
    # time_str='1:20 PM'
    # date_str='the last Monday of September 2025'

    def random_time_str() -> str:
      """随机生成 00:00–23:59 之间的一个时间字符串，格式 HH:MM AM/PM"""
      hour_24 = random.randint(0, 23)
      minute = random.randint(0, 59)
      dt = datetime.strptime(f"{hour_24}:{minute:02d}", "%H:%M")
      return dt.strftime("%I:%M %p").lstrip("0")  # 去掉前导 0，如 09:30 -> 9:30 AM

    def random_date_str() -> str:
      """随机返回 2025 年中的任意一天，格式 2025-03-14"""
      start = datetime(2025, 1, 1)
      end   = datetime(2025, 12, 31)
      delta_days = (end - start).days
      random_day = start + timedelta(days=random.randint(0, delta_days))
      return random_day.strftime("%Y-%m-%d")


    time_str=random_time_str()
    date_str=random_date_str()
    
    return (
        f'Add the following expenses into {_APP_NAME}:\n{text_repr}.\n'
        f'Set the time to {time_str} and the date to {date_str}.'
    )

class BrowsecompExpenseLocateInterface(_ExpenseAddMultiple):
  """Task to locate an interface of the Expense Tracking App."""
  # 
  complexity = 5
  n_rows = 1
  n_rows_noise = 10

  ground_truth_inferface_file = 'file path of some sreenshot path of Pro Expense '
  @property
  def goal(self) -> str:
    text_repr = _get_expense_rows_as_text(
        self.params[sqlite_validators.ROW_OBJECTS],
        self.params[_TEXT_REPRESENTATION_TYPE],
    )
    return f'''
    I need to find an APP among OpenTracks, Broccoli, Expense, Markor and Calendar.
    这个app可以对存储的信息item设置类别，且类别是固定的，不能通过自定义来设置。
    这个app主界面有加号按钮可以用于添加new item. 
    Find this APP and show me the interface where I can edit the new information item.
    '''



class NoiseResistExpenseAddSingleADs(_ExpenseAddMultiple):
  """Task to add a single expense in the Expense Tracking App."""

  complexity = 3.0
  n_rows = 1
  n_rows_noise = 10
  # popup_step = 5
  
  APP_KEYWORDS = [
      'chrome',
      'settings',
      'youtube',
      'gmail',
      'camera',
      'audio recorder',
      'clock',
      'contacts',
      'markor',
      'messages',
      'phone',
      'simple calendar pro',
      'simple draw pro',
      'pro expense',
      'broccoli',
      'osmand',
      'tasks',
      'opentracks',
      'joplin',
      'vlc',
      'retro',
      'calcyou',
      'pomodoro',
  ]

  popup_step = {
      random.randint(3, 8): random.choice(APP_KEYWORDS),
      random.randint(12, 18): random.choice(APP_KEYWORDS),
  }
  def check_status(self, env: interface.AsyncEnv):
    self.is_ADs_condition(env)


# TODO
class NoiseResistExpenseAddSingleWithOrientation(_ExpenseAddMultiple):
  """Task to add a single expense in the Expense Tracking App."""

  complexity = 3.0
  n_rows = 1
  n_rows_noise = 10

  landscape_step = 5
  portrait_step = 10
  def check_status(self, env: interface.AsyncEnv):
    self.is_o_condition(env)



class NoiseResistExpenseAddSingleAPPNumb(_ExpenseAddMultiple):
  """Task to add a single expense in the Expense Tracking App. Agent actons maybe stuck."""

  complexity = 3.0
  n_rows = 1
  n_rows_noise = 10

  numb_start_step = -1
  numb_end_step = -1
  
  numb_steps = random.randint(8,15)
  numb_steps_count = numb_steps

  collapsed =False
  def check_status(self, env: interface.AsyncEnv):
    self.is_noise_trigger_condition_by_xml(env)


  def is_noise_trigger_condition_by_xml(
      self,
      env: interface.AsyncEnv,
  ) -> float:

    ui_elements = representation_utils.xml_dump_to_ui_elements(
        adb_utils.uiautomator_dump(env.controller)
    )
    target = self.params[sqlite_validators.ROW_OBJECTS][0]
    expense_name = target.name

    if self.numb_start_step == -1:
    # numb not triggered yet
      if self._expense_name_is_entered(expense_name, ui_elements):
        # numb just triggered
        self.numb_start_step = self.current_step
        self.numb_end_step = self.numb_start_step + self.numb_steps
        self.step_numb = True
    else:
      # numb already triggered
      if self.current_step >= self.numb_start_step and self.current_step < self.numb_end_step:
        self.step_numb = True
      else:
        self.step_numb = False



class NoiseResistExpenseAddSingleAPPCollapse(_ExpenseAddMultiple):
  """Task to add a single expense in the Expense Tracking App."""

  complexity = 3.0
  n_rows = 1
  n_rows_noise = 10


  # Kill app condition and program
  collapse_steps = [random.randint(4, 7)]

  def check_status(self, env: interface.AsyncEnv):
    self.is_collapse_condition(env)

  def is_collapse_condition(
      self,
      env: interface.AsyncEnv,
  ) -> float:
    if self.current_step in self.collapse_steps:
      adb_utils.close_app(_APP_NAME, env.controller)


class MultiRoundExpense1Add2Add3Delete(_ExpenseAddMultiple):
  """Task to add a single expense in the Expense Tracking App."""

  complexity = 5
  n_rows = 1
  n_rows_noise = 10

  round = 0
  max_round = 3
 

  @property
  def goal(self) -> str:
    if self.round == 0:
      text_repr = _get_expense_rows_as_text(
          self.params[sqlite_validators.ROW_OBJECTS],
          self.params[_TEXT_REPRESENTATION_TYPE],
      )
      return f'Add the following expenses into the {_APP_NAME}:\n{text_repr}'
    elif self.round == 1:
      text_repr = self.params["target_expense_2nd_round"].name
      return f'Continue adding another expense identical to the one above, except that its name is different—use {text_repr} as the name.'
    elif self.round == 2:
      return "Delete the first expense I asked you to add just now."
    

  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.Expense:
    return _generate_expense()

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for an add expense task."""
    target_rows = sqlite_schema_utils.get_random_items(
        cls.n_rows,
        cls._get_random_target_row,
        replacement=False,
    )
    noise_rows = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        cls._get_random_target_row,
        replacement=False,
        filter_fn=lambda r: all(r.name != t.name for t in target_rows),
    )

    first_old_expense = target_rows[0]
    # Select a different name for the 2nd round expense.

    category_name = sqlite_schema_utils.Expense.category_id_to_name[first_old_expense.category]
   
    candidates = _EXPENSE_NAMES[category_name]
    
    if not candidates:          
        raise ValueError("No alternative name available in this category")
    secondname = random.choice(candidates)
    target_expense_2nd_round = sqlite_schema_utils.Expense(
      name=secondname,
      amount=first_old_expense.amount,
      category=first_old_expense.category,
      note=first_old_expense.note,
      created_date=first_old_expense.created_date,
      modified_date=first_old_expense.modified_date,
    )
    
    
    return {
        sqlite_validators.ROW_OBJECTS: target_rows,
        sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
        'target_expense_2nd_round': target_expense_2nd_round,
        _TEXT_REPRESENTATION_TYPE: random.choice(['csv', 'text_block']),
    }
  

  def validate_deletion_integrity(
      self,
      before: list[sqlite_schema_utils.Expense],
      after: list[sqlite_schema_utils.Expense],
  ) -> bool:
    """Validates the integrity of the expense deletion."""
    target1 = self.params[sqlite_validators.ROW_OBJECTS][0]
    
    
    if len(before) != len(after) + 1:
      return False
    for row in after:
      if row not in before:
        return False
      if row.amount == target1.amount and row.name == target1.name:
        return False
    return True
    
    

  def is_successful(self, env: interface.AsyncEnv) -> float:
    """Determine if the row addition task was successful."""
    if self.round == 0:
      after = self.list_rows(env)
      row_addition_successful = self.validate_addition_integrity(
          self.before, after, self.params[sqlite_validators.ROW_OBJECTS]
      )
      
      if row_addition_successful:
        return 1.0 
      else: 
        return 0.0
    
    elif self.round == 1:
      after = self.list_rows(env)
      row_addition_successful = self.validate_addition_integrity(
          self.before, after,  self.params[sqlite_validators.ROW_OBJECTS] + [self.params['target_expense_2nd_round']]
      )
      if row_addition_successful:
        self.before = self.list_rows(env)
        return 1.0 
      else: 
        return 0.0
      
    elif self.round == 2:
      super().is_successful(env)
      # Get the state of the database after the deletion attempt
      after = self.list_rows(env)
      # Validate the integrity of the deletion
      deletion_successful = self.validate_deletion_integrity(self.before, after)
      return 1.0 if deletion_successful else 0.0


class MultiRoundExpenseAddMultipleCHS(MultiRoundExpense1Add2Add3Delete):
  @property
  def goal(self) -> str:
    if self.round == 0:
      text_repr = _get_expense_rows_as_text(
          self.params[sqlite_validators.ROW_OBJECTS],
          self.params[_TEXT_REPRESENTATION_TYPE],
      )
      return f'将以下支出添加到{_APP_NAME}中：\n{text_repr}'
    elif self.round == 1:
      text_repr = self.params["second_round_target_expense"]
      return f'继续添加一个与上方相同的支出，但名称不同——将{text_repr}用作名称。'
    elif self.round == 2:
      return '删除我刚才让你添加的第一个支出。'
 
class MultiRoundExpenseAddMultipleVariation(MultiRoundExpense1Add2Add3Delete):
  @property
  def goal(self) -> str:
    if self.round == 0:
      text_repr = _get_expense_rows_as_text(
          self.params[sqlite_validators.ROW_OBJECTS],
          self.params[_TEXT_REPRESENTATION_TYPE],
      )
      return f'Add the expenses listed below to {_APP_NAME}:\n{text_repr}'
    elif self.round == 1:
      text_repr = self.params["second_round_target_expense"]
      return f'Then add another expense identical to the one above, but give it a different name—name it {text_repr}.'
    elif self.round == 2:
      return 'Remove the first expense I just asked you to add.'

class ExpenseAddMultiple(_ExpenseAddMultiple):
  """Task to add multiple expenses in the Expense Tracking App."""

  complexity = 6
  n_rows = 3
  n_rows_noise = 10


class ExpenseAddMultipleFromMarkor(_ExpenseAddMultiple):
  """Task to add multiple expenses from Markor into the Expense Tracking app."""

  complexity = 6
  n_rows = 2
  n_rows_noise = 100

  @property
  def goal(self) -> str:
    return (
        'Go through the transactions in my_expenses.txt in Markor. Log the '
        f'reimbursable transactions in the {_APP_NAME}.'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    targets = [
        dataclasses.replace(row, note=row.note + '. ' + 'Reimbursable.')
        for row in self.params[sqlite_validators.ROW_OBJECTS]
    ]
    rows = targets + self.params[sqlite_validators.NOISE_ROW_OBJECTS]
    random.shuffle(rows)
    file_utils.clear_directory(device_constants.MARKOR_DATA, env.controller)
    user_data_generation.write_to_markor(
        _get_expense_rows_as_text(rows, 'csv'),
        'my_expenses.txt',
        env,
    )

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    file_utils.clear_directory(device_constants.MARKOR_DATA, env.controller)


class ExpenseAddMultipleFromGallery(_ExpenseAddMultiple):
  """Task to add multiple expenses from Gallery into Expense Tracking app."""

  complexity = 6
  n_rows = 3
  n_rows_noise = 10

  app_names = (_APP_NAME, 'simple gallery pro')

  @property
  def goal(self) -> str:
    return (
        'Add the expenses from expenses.jpg in Simple Gallery Pro to '
        f'{_APP_NAME}.'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    user_data_generation.clear_device_storage(env)
    data = _get_expense_rows_as_text(
        self.params[sqlite_validators.ROW_OBJECTS], 'text_block', wrap_width=60
    )
    user_data_generation.write_to_gallery(data, 'expenses.jpg', env)
    for i in range(10):
      data = _get_expense_rows_as_text(
          self.params[sqlite_validators.NOISE_ROW_OBJECTS],
          'text_block',
          wrap_width=60,
      )
      user_data_generation.write_to_gallery(data, f'old_expenses_{i}.jpg', env)

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    user_data_generation.clear_device_storage(env)


#### Generate expense data for tasks. ##########################################


def _generate_expense(
    expense_unix_time_s: Optional[int] = None,
    category_id: int | None = None,
) -> sqlite_schema_utils.Expense:
  """Generates a realistic expense entry.

  Args:
    expense_unix_time_s: The time the expense is entered into the app. This will
      be reflected in the UI.
    category_id: Optional value to override random generation.

  Returns:
      An Expense object with random realistic parameters.
  """
  if expense_unix_time_s is None:
    expense_unix_time_s = _get_random_timestamp()
  expense_unix_time_ms = expense_unix_time_s * 1000

  if category_id is None:
    category_id = random.choice(
        list(sqlite_schema_utils.Expense.category_id_to_name.keys())
    )
  name = random.choice(
      _EXPENSE_NAMES[
          sqlite_schema_utils.Expense.category_id_to_name[category_id]
      ]
  )
  amount = random.randint(
      1000, 50000
  )  # Amount in cents (e.g., $10.00 - $500.00)
  note = random.choice(_NOTES)
  return sqlite_schema_utils.Expense(
      name,
      amount,
      category_id,
      note,
      expense_unix_time_ms,
      expense_unix_time_ms,
  )



_EXPENSE_NAMES = {
    'Others': [
        'Emergency Repairs',
        'Pet Supplies',
        'Pet Care',
        'Household Items',
        'Stationery',
        'Unexpected Expenses',
        'Miscellaneous Gifts',
        'Subscriptions',
        'Membership Fees',
        'Legal Fees',
    ],
    'Income': [
        'Salary',
        'Freelance Payment',
        'Bonus',
        'Dividends',
        'Interest Income',
        'Rental Income',
        'Capital Gains',
        'Reimbursements',
        'Side Business',
        'Consulting Fees',
    ],
    'Food': [
        'Restaurant Meal',
        'Groceries',
        'Coffee',
        'Fast Food',
        'Fine Dining',
        'Bakery Items',
        'Snacks',
        'Food Delivery',
        'Specialty Foods',
        'Dining Out',
    ],
    'Housing': [
        'Rent Payment',
        'Mortgage',
        'Home Repairs',
        'Utilities',
        'Property Taxes',
        'Home Insurance',
        'Furnishing',
        'Cleaning Services',
        'Landscaping',
        'Pest Control',
    ],
    'Social': [
        'Dinner Party',
        'Gift for Friend',
        'Club Membership',
        'Wedding Gift',
        'Charity Donations',
        'Birthday Present',
        'Social Club Dues',
        'Event Tickets',
        'Night Out',
        'Party Supplies',
    ],
    'Entertainment': [
        'Concert Tickets',
        'Movie Night',
        'Theater Show',
        'Streaming Services',
        'Video Games',
        'Books',
        'Magazines',
        'Hobbies',
        'Museum Tickets',
        'Amusement Park',
    ],
    'Transportation': [
        'Taxi Fare',
        'Public Transit Pass',
        'Gas',
        'Parking Fees',
        'Car Maintenance',
        'Bike Repairs',
        'Car Insurance',
        'Public Transit',
        'Flight Tickets',
        'Ride-Sharing',
    ],
    'Clothes': [
        'New Jacket',
        'Shirt Purchase',
        'Shoes',
        'Dress',
        'Jeans',
        'Accessories',
        'Sportswear',
        'Undergarments',
        'Tailoring Services',
        'Laundry',
    ],
    'Health Care': [
        'Doctor Visits',
        'Medications',
        'Health Insurance',
        'Dental Care',
        'Eyecare',
        'Wellness Products',
        'Gym Membership',
        'Therapy Sessions',
        'Medical Tests',
    ],
    'Education': [
        'Tuition Fees',
        'School Supplies',
        'Textbooks',
        'Online Courses',
        'Seminars',
        'Workshops',
        'Educational Software',
        'Library Fees',
        'ProDev',
        'Tutoring Services',
    ],
    'Donation': [
        'Charity',
        'Fundraising Events',
        'Sponsorships',
        'Non-Profit Support',
        'Crowdfunding',
        'Religious',
        'Political',
        'Educational',
        'Medical Research',
        'Environmental',
    ],
}

_NOTES = [
    'Paid by card',
    'Urgent',
    'Monthly recurring',
    'Want to have',
    'A need',
    'Remember to transfer funds',
    'I may repeat this',
]


# New Tasks of Ant_UI_Bench

# _ExpenseDeleteMultipleConflict
class _ExpenseDeleteMultipleConflict(_Expense, sqlite_validators.DeleteMultipleRowsConflict):
  """Task to delete multiple expenses in an expense tracking app."""
  """Some expenses to be deleted not exist in the app"""
  complexity = 2
  n_rows = 3  # Default number of expenses to delete
  n_rows_noise = 0  # Default noise rows


  @property
  def goal(self) -> str:
    targets = self.params[sqlite_validators.ROW_OBJECTS]
    expense_names = [expense.name for expense in targets]
    expense_names_str = ', '.join(expense_names)
    return (
        f'Delete the following expenses from {_APP_NAME}: {expense_names_str}.'
    )

  def validate_deletion_integrity(
      self,
      before: list[sqlite_schema_utils.Expense],
      after: list[sqlite_schema_utils.Expense],
  ) -> bool:
    """Validates the integrity of the expense deletion."""
    return sqlite_validators.validate_rows_removal_integrity(
        before,
        after,
        [expense.expense_id for expense in self.rows_to_delete],
        self.db_key,
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a remove expense task."""
    """Add Conflicts """
    expenses = []
    while len(expenses) < cls.n_rows + cls.n_rows_noise:
      candidate = _generate_expense()
      if not any([candidate.name == expense.name for expense in expenses]):
        expenses.append(candidate)

    if cls.n_rows_noise > 0:
      target_rows = expenses[: cls.n_rows]
      noise_rows = expenses[cls.n_rows :]
      # some rows in target_rows will be added in db.
      # some rows in target_rows will not.
      return {
          sqlite_validators.ROW_OBJECTS: target_rows,
          sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
      }
    else:
      return {
          sqlite_validators.ROW_OBJECTS: expenses,
      }


class RefusalExpenseDeleteMultipleConflictSingle(_ExpenseDeleteMultipleConflict):
  """Harder task to delete multiple expenses in an expense tracking app."""
  """Conflict"""
  complexity = 3
  n_rows = 3
  n_true_rows = n_rows-1
  n_rows_noise = 20

  def is_successful(self, env: interface.AsyncEnv) -> float:
    parent_success = super().is_successful(env)

    # Get the state of the database after the deletion attempt
    after = self.list_rows(env)
    if parent_success == 0.0:
      return 0.0

    targets_no_exist = self.params[sqlite_validators.ROW_OBJECTS][self.n_true_rows:]

    missing_name = targets_no_exist[0].name
    answer = f"The expense '{missing_name}' is not available in {_APP_NAME}. Please check the name and try again."


    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    ply = VerifyPolicy({})

    result = ply.verify_textual_prediction(
            goal=self.goal,
            prediction=env.interaction_cache,
            gt_answer=answer,
            judgement_type = 'refusal'
        )
    return result


class RefusalExpenseDeleteMultipleConflictAll(_ExpenseDeleteMultipleConflict):
  """Harder task to delete multiple expenses in an expense tracking app."""
  """Conflict: all 3 expenses to be deleted"""
  complexity = 3
  n_rows = 3
  n_true_rows = 0
  n_rows_noise = 20

  def is_successful(self, env: interface.AsyncEnv) -> float:
    parent_success = super().is_successful(env)

    # Get the state of the database after the deletion attempt
    after = self.list_rows(env)
    if len(after) != len(self.before) or parent_success == 0.0:
      return 0.0

    targets = self.params[sqlite_validators.ROW_OBJECTS]
    expense_names = [expense.name for expense in targets]
    expense_names_str = ', '.join(expense_names)
    answer = f"None of the expenses named {expense_names_str} exist in {_APP_NAME}."

    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    ply = VerifyPolicy({})

    result = ply.verify_textual_prediction(
            goal=self.goal,
            prediction=env.interaction_cache,
            gt_answer=answer,
            judgement_type = 'refusal'
        )
    return result




class RefusalExpenseDeleteMultipleConflictMultiple(_ExpenseDeleteMultipleConflict):
  """Harder task to delete multiple expenses in an expense tracking app."""
  """Conflict"""
  complexity = 3
  n_rows = 5
  # rows to be deleted and exist
  n_true_rows = 3
  n_rows_noise = 20


  def is_successful(self, env: interface.AsyncEnv) -> float:
    
    parent_success = super().is_successful(env)

    # Get the state of the database after the deletion attempt
    after = self.list_rows(env)
    if parent_success == 0.0:
      return 0.0

    targets_no_exist = self.params[sqlite_validators.ROW_OBJECTS][self.n_true_rows:]
    expense_names = [expense.name for expense in targets_no_exist]
    expense_names_str = ', '.join(expense_names)

    answer = (
        f"The expenses {expense_names_str} are "
        f"not available in {_APP_NAME}. "
        f"Please check the names and try again."
    )

    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    ply = VerifyPolicy({})

    result = ply.verify_textual_prediction(
            goal=self.goal,
            prediction=env.interaction_cache,
            gt_answer=answer,
            judgement_type = 'refusal'
        )
    return result


# Explore expense
class FuncAssistExpenseExplainAllFunctionality(_Expense, sqlite_validators.AddMultipleRows):
  complexity = 4
  n_rows = 3
  n_rows_noise = 10
# _APP_NAME

  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.Expense:
    return _generate_expense()

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for an add expense task."""
    target_rows = sqlite_schema_utils.get_random_items(
        cls.n_rows,
        cls._get_random_target_row,
        replacement=False,
    )
    noise_rows = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        cls._get_random_target_row,
        replacement=False,
        filter_fn=lambda r: all(r.name != t.name for t in target_rows),
    )
    return {
        sqlite_validators.ROW_OBJECTS: target_rows,
        sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
        _TEXT_REPRESENTATION_TYPE: random.choice(['csv', 'text_block']),
    }
  def validate_addition_integrity(
      self,
      before: list[sqlite_schema_utils.Expense],
      after: list[sqlite_schema_utils.Expense],
      reference_rows: list[sqlite_schema_utils.Expense],
  ) -> bool:
    """Validates the integrity of the expense addition."""
    return sqlite_validators.validate_rows_addition_integrity(
        before,
        after,
        reference_rows,
        compare_fields=[
            'name',
            'amount',
            'category',
            'note',
        ],
        free_form_fields=[
            'name',
            'note',
        ],
    )
  @property
  def goal(self) -> str:
    return f'What can I do with {_APP_NAME}? Explore this APP and tell me all its functionalities.'

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
    gt_answer = 'Log expenses, income, and transfers quickly; organize by categories, tags, and accounts. Set budgets, track recurring bills, split transactions, and attach receipt photos. View summaries and charts by date, category, or account to spot trends. Support for multiple currencies and account balances keeps totals accurate. Back up or export your data to files.'
    prediction = env.interaction_cache
    
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
          'Error during verification for FuncAssistExpenseExplainAllFunctionality: %s', e
      )
      return 0.0



class FuncAssistExpenseExplainOneFunctionality1(_Expense, sqlite_validators.AddMultipleRows):
  complexity = 4
  n_rows = 3
  n_rows_noise = 10
# _APP_NAME
  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.Expense:
    return _generate_expense()
  
  def validate_addition_integrity(
      self,
      before: list[sqlite_schema_utils.Expense],
      after: list[sqlite_schema_utils.Expense],
      reference_rows: list[sqlite_schema_utils.Expense],
  ) -> bool:
    """Validates the integrity of the expense addition."""
    return sqlite_validators.validate_rows_addition_integrity(
        before,
        after,
        reference_rows,
        compare_fields=[
            'name',
            'amount',
            'category',
            'note',
        ],
        free_form_fields=[
            'name',
            'note',
        ],
    )
  

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for an add expense task."""
    target_rows = sqlite_schema_utils.get_random_items(
        cls.n_rows,
        cls._get_random_target_row,
        replacement=False,
    )
    noise_rows = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        cls._get_random_target_row,
        replacement=False,
        filter_fn=lambda r: all(r.name != t.name for t in target_rows),
    )
    return {
        sqlite_validators.ROW_OBJECTS: target_rows,
        sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
        _TEXT_REPRESENTATION_TYPE: random.choice(['csv', 'text_block']),
    }
  @property
  def goal(self) -> str:
    return f'''Explore {_APP_NAME} and tell me how to add a new expense.
Specifically, identify all available fields, options, and settings 
that can be configured when creating an expense.'''

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
    gt_answer = 'To add a new expense in Pro Expense, follow these steps: 1. Tap the plus (+) button to open the Expense Entry screen. 2. Enter the name of the expense in the \'Name\' field. 3. Enter the amount in the \'Amount\' field. 4. Optionally, add a note in the \'Note\' field. 5. Select a category for the expense. 6. Tap the \'SAVE\' button to save the new expense.'
    prediction = env.interaction_cache
    
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
          'Error during verification for FuncAssistExpenseExplainOneFunctionality1: %s', e
      )
      return 0.0


class FuncAssistExpenseExplainOneFunctionality1CHS(FuncAssistExpenseExplainOneFunctionality1):
  @property
  def goal(self) -> str:
    return f'''请深入探索 {_APP_NAME}，并尽可能详细地告诉我如何新增一笔支出。
请具体列出创建支出时可填写的所有字段、可配置选项与设置项。'''


class FuncAssistExpenseExplainOneFunctionality1Variation(FuncAssistExpenseExplainOneFunctionality1):
  @property
  def goal(self) -> str:
    return f'''Open {_APP_NAME} and walk me through adding a new expense in the most exhaustive way possible.
Enumerate every field, option, and setting that appears while creating an expense.'''


class FuncAssistExpenseLocateOneFunctionality1(_Expense, sqlite_validators.AddMultipleRows):
  complexity = 2
  n_rows = 3
  n_rows_noise = 10


  @property
  def goal(self) -> str:
    return f'Explore Pro Expense and show me the interface where I can add a new expense.'


  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.Expense:
    return _generate_expense()
  
  def validate_addition_integrity(
      self,
      before: list[sqlite_schema_utils.Expense],
      after: list[sqlite_schema_utils.Expense],
      reference_rows: list[sqlite_schema_utils.Expense],
  ) -> bool:
    """Validates the integrity of the expense addition."""
    return sqlite_validators.validate_rows_addition_integrity(
        before,
        after,
        reference_rows,
        compare_fields=[
            'name',
            'amount',
            'category',
            'note',
        ],
        free_form_fields=[
            'name',
            'note',
        ],
    )
  

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for an add expense task."""
    target_rows = sqlite_schema_utils.get_random_items(
        cls.n_rows,
        cls._get_random_target_row,
        replacement=False,
    )
    noise_rows = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        cls._get_random_target_row,
        replacement=False,
        filter_fn=lambda r: all(r.name != t.name for t in target_rows),
    )
    return {
        sqlite_validators.ROW_OBJECTS: target_rows,
        sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
        _TEXT_REPRESENTATION_TYPE: random.choice(['csv', 'text_block']),
    }
  

  def is_successful(self, env: interface.AsyncEnv) -> float:
    """Check if the task is successful by verifying the screenshot.
    
    Args:
      env: The environment interface.
      
    Returns:
      1.0 if successful (screenshot shows the correct interface), 0.0 otherwise.
    """
    super().is_successful(env)
    
   
    goal = self.goal
    task_key = 'FuncAssistExpenseLocateOneFunctionality1'
    
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
          'Error during verification for FuncAssistExpenseLocateOneFunctionality1: %s', e
      )
      return 0.0




class NoiseResistExpenseDeleteSingleWithOrientation(_ExpenseDeleteMultiple):
  """Task to delete a single expense in an expense tracking app."""

  complexity = 3
  n_rows = 1
  n_rows_noise = 0

  landscape_step = 5
  portrait_step = 10

  def check_status(self, env: interface.AsyncEnv):
    self.is_o_condition(env)

  def is_o_condition(
      self,
      env: interface.AsyncEnv,
  ) -> float:
    if self.current_step == self.landscape_step:
      adb_utils.change_orientation('landscape', env=env.controller)

    if self.current_step == self.portrait_step:
      adb_utils.change_orientation('portrait', env=env.controller)

    return ''

class NoiseResistExpenseDeleteSingleWithOrientationCHS(NoiseResistExpenseDeleteSingleWithOrientation):

  complexity = 1
  n_rows = 1
  n_rows_noise = 0
  landscape_step = 5
  portrait_step = 10

  @property
  def goal(self) -> str:
    targets = self.params[sqlite_validators.ROW_OBJECTS]
    expense_names = [expense.name for expense in targets]
    expense_names_str = ', '.join(expense_names)
    return (
        f'在{_APP_NAME}里删除以下支出记录：{expense_names_str}。'
    )
 
class NoiseResistExpenseDeleteSingleWithOrientationVariation(NoiseResistExpenseDeleteSingleWithOrientation):

  complexity = 1
  n_rows = 1
  n_rows_noise = 0
  landscape_step = 5
  portrait_step = 10

  @property
  def goal(self) -> str:
    targets = self.params[sqlite_validators.ROW_OBJECTS]
    expense_names = [expense.name for expense in targets]
    expense_names_str = ', '.join(expense_names)
    return (
        f'Remove the following expense entries from {_APP_NAME}: {expense_names_str}.'
    )

# To be deleted
class ExpenseAddSingleWithNotifications(_ExpenseAddMultiple):
  """Task to add a single expense in the Expense Tracking App."""

  complexity = 5.0
  n_rows = 1
  n_rows_noise = 10

  landscape_step = 2
  portrait_step = 4


  def check_status(self, env: interface.AsyncEnv):
    self.is_o_condition(env)

  def is_o_condition(
      self,
      env: interface.AsyncEnv,
  ) -> float:
    if self.current_step == self.landscape_step:
      adb_utils.send_android_intent(
          command='broadcast',
          action='com.example.TOAST',         
          env=env.controller,
          extras={'title': '标题', 'message': '这是一条弹窗'},
      )

    if self.current_step == self.portrait_step:
      # adb_utils.change_orientation('portrait', env=env.controller)
      adb_utils.show_system_notification(
          title="测试弹窗",
          message="Hello from ADB!",
          env=env.controller,              
      )
    return ''


class NoiseResistExpenseAddSingleWithCall(_ExpenseAddMultiple):
  """Task to add a single expense in the Expense Tracking App."""
  complexity = 3.0
  n_rows = 1
  n_rows_noise = 10

  # call_coming_steps = [5,10]
  call_coming_steps = sorted(random.sample(range(0, 31, 3), 3))

  def check_status(self, env: interface.AsyncEnv):
    self.is_call_condition(env)


class RefusalExpenseDeleteMultipleConflictMultipleCHS(_ExpenseDeleteMultipleConflict):
  """Harder task to delete multiple expenses in an expense tracking app."""
  """Conflict"""
  complexity = 3.4
  n_rows = 5
  n_true_rows = 3
  n_rows_noise = 20
  @property
  def goal(self) -> str:
    targets = self.params[sqlite_validators.ROW_OBJECTS]
    expense_names = [expense.name for expense in targets]
    expense_names_str = ', '.join(expense_names)
    return (
        f'在{_APP_NAME}里删除以下支出记录：{expense_names_str}。'
    )

class RefusalExpenseDeleteMultipleConflictMultipleVariation(_ExpenseDeleteMultipleConflict):
  """Harder task to delete multiple expenses in an expense tracking app."""
  """Conflict"""
  complexity = 3.4
  n_rows = 5
  n_true_rows = 3
  n_rows_noise = 20
  @property
  def goal(self) -> str:
    targets = self.params[sqlite_validators.ROW_OBJECTS]
    expense_names = [expense.name for expense in targets]
    expense_names_str = ', '.join(expense_names)
    return (
        f'Remove the following expenses from {_APP_NAME}: {expense_names_str}.'
    )


class VagueDailyExpenseRecordTask(_Expense, sqlite_validators.AddMultipleRows):
  """Task to record daily expenses with vague instructions.
  
  This task asks the user to record multiple expenses with amounts and categories
  that are randomly generated, making it a more open-ended task.
  """

  complexity = 4
  n_rows = 2
  n_rows_noise = 5

  @property
  def goal(self) -> str:
    """
    Instruction format: 
    "I spent $X on [category1] and $Y on [category2] today. 
    Please record it for me."
    """
    expenses = self.params[sqlite_validators.ROW_OBJECTS]
    expense_descriptions = []
    for expense in expenses:
      # Convert amount from cents to dollars
      dollars = expense.amount / 100
      category = sqlite_schema_utils.Expense.category_id_to_name[expense.category]
      expense_descriptions.append(f'${dollars:.2f} on {category.lower()}')
    
    expenses_str = ' and '.join(expense_descriptions)
    return f'I spent {expenses_str} today. Please record it for me.'

  def validate_addition_integrity(
      self,
      before: list[sqlite_schema_utils.Expense],
      after: list[sqlite_schema_utils.Expense],
      reference_rows: list[sqlite_schema_utils.Expense],
  ) -> bool:
    """Validates the integrity of the expense addition."""
    return sqlite_validators.validate_rows_addition_integrity(
        before,
        after,
        reference_rows,
        compare_fields=[
            'amount',
            'category',
        ],
        free_form_fields=[
            'name',
            'note',
        ],
    )

  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.Expense:
    return _generate_expense()

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a vague daily expense task."""
    target_rows = sqlite_schema_utils.get_random_items(
        cls.n_rows,
        cls._get_random_target_row,
        replacement=False,
    )
    noise_rows = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        cls._get_random_target_row,
        replacement=False,
        filter_fn=lambda r: all(r.name != t.name for t in target_rows),
    )
    return {
        sqlite_validators.ROW_OBJECTS: target_rows,
        sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
        _TEXT_REPRESENTATION_TYPE: random.choice(['csv', 'text_block']),
    }

  def is_successful(self, env: interface.AsyncEnv) -> float:
    # Use the parent class's is_successful which validates the addition integrity
    return super().is_successful(env)