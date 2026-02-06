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

"""Tasks for Simple Calendar Pro app."""
from android_world.policy.verification import VerifyPolicy

import dataclasses
import random
from android_world.env import interface

from android_world.env import adb_utils
from typing import Any, Callable, Optional
from android_world.env import device_constants
from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.single.calendar import calendar_evaluators
from android_world.task_evals.single.calendar import calendar_utils
from android_world.task_evals.single.calendar import events_generator
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.utils import datetime_utils

# Keys in generated parameters and used to populate goal template.
_YEAR = "year"
_MONTH = "month"
_DAY = "day"
_DAY_OF_WEEK = "day_of_week"
_HOUR = "hour"
EVENT_TITLE = "event_title"
_EVENT_DESCRIPTION = "event_description"
_DURATION_MINS = "duration_mins"
_REPEAT_INTERVAL = "repeat_rule"
_REPEAT_INTERVALS = {"daily": 60 * 60 * 24, "weekly": 60 * 60 * 24 * 7}


def generate_noise_events(
    target_events: list[sqlite_schema_utils.CalendarEvent],
    n: int,
    filter_fn: Optional[
        Callable[[sqlite_schema_utils.CalendarEvent], bool]
    ] = None,
) -> list[sqlite_schema_utils.CalendarEvent]:
  if filter_fn is None:
    target_titles = set(event.title for event in target_events)
    filter_fn = lambda candidate: candidate.title not in target_titles

  return sqlite_schema_utils.get_random_items(
      n,
      lambda: events_generator.generate_event(
          datetime_utils.create_random_october_2023_unix_ts(start_day=1)
      ),
      filter_fn=filter_fn,
  )


class _SimpleCalendar(sqlite_validators.SQLiteApp):
  """Base class for calendar tasks and evaluation logic.

                  October 2023
              Su Mo Tu We Th Fr Sa
              1  2  3  4  5  6  7
              8  9 10 11 12 13 14
              [15]16 17 18 19 20 21
              22 23 24 25 26 27 28
              29 30 31

  The current date on the emulator will be set as October 15, 2023.
  """

  app_name_with_db = "simple calendar pro"
  app_names = ("simple calendar pro",)
  schema = {}

  db_key = "id"
  db_path = calendar_utils.DB_PATH
  table_name = calendar_utils.EVENTS_TABLE
  row_type = sqlite_schema_utils.CalendarEvent

class SimpleCalendarAddOneEvent(
    sqlite_validators.AddMultipleRows, _SimpleCalendar
):
  """Task for creating a calendar event in Simple Calendar Pro.

  Uses the absolute date in the template.
  """

  n_rows = 1  # Unused, but required by base class.
  complexity = 3.4
  template = (
      "In Simple Calendar Pro, create a calendar event on {year}-{month}-{day}"
      " at {hour}h with the title '{event_title}' and the description"
      " '{event_description}'. The event should last for {duration_mins} mins."
  )

  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.CalendarEvent:
    """Generates a random calendar event."""
    return events_generator.generate_event(
        datetime_utils.create_random_october_2023_unix_ts()
    )

  def validate_addition_integrity(
      self,
      before: list[sqlite_schema_utils.CalendarEvent],
      after: list[sqlite_schema_utils.CalendarEvent],
      reference_rows: list[sqlite_schema_utils.CalendarEvent],
  ) -> bool:
    """Validates the integrity of the event addition."""
    return calendar_evaluators.validate_event_addition_integrity(
        before,
        after,
        reference_rows,
        extras_compare=["repeat_rule", "repeat_interval"],
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a new calendar event task."""
    event = cls._get_random_target_row()
    n_noise_events = random.randint(0, 20)
    return {
        _YEAR: device_constants.DT.year,
        _MONTH: device_constants.DT.month,
        _DAY: event.start_datetime.day,
        _HOUR: event.start_datetime.hour,
        _DURATION_MINS: event.duration_mins,
        EVENT_TITLE: event.title,
        _EVENT_DESCRIPTION: event.description,
        sqlite_validators.ROW_OBJECTS: [event],
        sqlite_validators.NOISE_ROW_OBJECTS: generate_noise_events(
            [event], n_noise_events
        ),
    }


class SimpleCalendarAddOneEventRelativeDay(SimpleCalendarAddOneEvent):
  """Task for creating a calendar event in Simple Calendar Pro.

  Uses the relative day of week in the template: from "this Monday" -> "this
  Sunday".
  """

  complexity = 3.4
  _DAY_RANGE = 6

  template = (
      "In Simple Calendar Pro, create a calendar event for this {day_of_week}"
      " at {hour}h with the title '{event_title}' and the description"
      " '{event_description}'. The event should last for {duration_mins} mins."
  )

  @property
  def goal(self) -> str:
    # Add day of week.
    dt: sqlite_schema_utils.CalendarEvent = self.params[
        sqlite_validators.ROW_OBJECTS
    ][0]
    day_of_week = dt.start_datetime.strftime("%A")
    self.params[_DAY_OF_WEEK] = day_of_week
    return self.template.format(**self.params)

  @classmethod
  def _get_random_target_row(cls):
    return events_generator.generate_event(
        datetime_utils.create_random_october_2023_unix_ts(
            # Monday, Oct 16 -> Saturday, Oct 21.
            start_day=device_constants.DT.day + 1,
            end_day=(
                device_constants.DT.day
                + SimpleCalendarAddOneEventRelativeDay._DAY_RANGE
            ),
        )
    )


class SimpleCalendarAddOneEventTomorrow(SimpleCalendarAddOneEvent):
  """Task for creating a calendar event in Simple Calendar Pro for tomorrow."""

  complexity = 2.6
  template = (
      "In Simple Calendar Pro, create a calendar event for tomorrow"
      " at {hour}h with the title '{event_title}' and the description"
      " '{event_description}'. "
      "The event should last for {duration_mins} mins."
  )
  answer = (
    "Sorry, I can't create the event. "
    "Simple Calendar Pro doesn't allow custom durations."
  )

  @classmethod
  def _get_random_target_row(cls):
    # Generate an event for tomorrow.
    return events_generator.generate_event(
        datetime_utils.create_random_october_2023_unix_ts(
            device_constants.DT.day + 1, device_constants.DT.day + 1
        )
    )

COLORS = [
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "indigo",
    "violet",
    "purple",
    "magenta",
    "pink",
    "brown",
    "maroon",
    "tan",
    "gray",
    "black",
    "white",
    "silver",
    "gold",
    "turquoise",
    "teal",
    "navy",
    "olive",
    "lime",
    "lavender",
    "crimson",
    "coral",
    "salmon",
    "khaki",
    "beige"
]



class RefusalSimpleCalendarAddOneEventInTwoWeeksConflict1(SimpleCalendarAddOneEvent):
  """Task for creating a calendar event in Simple Calendar Pro in two weeks from today."""

  complexity = 2
  template = (
      "In Simple Calendar Pro, create a calendar event in two weeks from today"
      " at {hour}h with the title '{event_title}' and the description"
      " '{event_description}'. "
      "The event should last for {duration_mins} mins. "
      "Set the Task color as {task_theme_color_str} and "
      "change the Title font color as {title_font_color_str}."
  )
  answer = (
    "Sorry, I can't add the event. "
    "Simple Calendar Pro doesn't let me set "
    "a custom title font color when creating a new event."
    )
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a new calendar event task."""
    event = cls._get_random_target_row()
    n_noise_events = random.randint(0, 20)
    return {
        _YEAR: device_constants.DT.year,
        _MONTH: device_constants.DT.month,
        _DAY: event.start_datetime.day,
        _HOUR: event.start_datetime.hour,
        _DURATION_MINS: event.duration_mins,
        EVENT_TITLE: event.title,
        _EVENT_DESCRIPTION: event.description,
        sqlite_validators.ROW_OBJECTS: [event],
        sqlite_validators.NOISE_ROW_OBJECTS: generate_noise_events(
            [event], n_noise_events
        ),
        'task_theme_color_str': random.choice(COLORS),
        'title_font_color_str': random.choice(COLORS),
    }

  @classmethod
  def _get_random_target_row(cls):
    return events_generator.generate_event(
        datetime_utils.create_random_october_2023_unix_ts(
            device_constants.DT.day + 14, device_constants.DT.day + 14
        )
    )
  
  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not env.interaction_cache:
      return 0.0
    
    # Adding the task is wrong because the instruction cannot be finished.
    after = self.list_rows(env) 
    if len(after) != len(self.before):
      return 0.0
    
    ply = VerifyPolicy({})
    
    result = ply.verify_textual_prediction(
            goal=self.goal,
            prediction=env.interaction_cache,
            gt_answer= self.answer,
            judgement_type = 'refusal'
        )
    # print(result)  # 期望 True  
    return result



# 

class SimpleCalendarAddRepeatingEvent(SimpleCalendarAddOneEvent):
  """Task for creating a repeating calendar event in Simple Calendar Pro."""

  complexity = 2.8
  template = (
      "In Simple Calendar Pro, create a recurring calendar event titled"
      " '{event_title}' starting on {year}-{month}-{day} at"
      " {hour}h. The event recurs {repeat_rule}, forever, and lasts for"
      " {duration_mins} minutes each occurrence. The event description should"
      " be '{event_description}'."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a new calendar event task."""
    template = cls._get_random_target_row()
    repeat_interval = random.choice(list(_REPEAT_INTERVALS))
    if repeat_interval == "weekly":
      repeat_rule = calendar_utils.generate_simple_calendar_weekly_repeat_rule(
          template.start_datetime.isoweekday()
      )
    else:
      repeat_rule = 0
    event = dataclasses.replace(
        template,
        repeat_interval=_REPEAT_INTERVALS[repeat_interval],
        repeat_rule=repeat_rule,
    )
    noise_events = generate_noise_events([event], random.randint(0, 20))
    return {
        _YEAR: device_constants.DT.year,
        _MONTH: device_constants.DT.month,
        _DAY: event.start_datetime.day,
        _HOUR: event.start_datetime.hour,
        _DURATION_MINS: event.duration_mins,
        EVENT_TITLE: event.title,
        _EVENT_DESCRIPTION: event.description,
        sqlite_validators.ROW_OBJECTS: [event],
        sqlite_validators.NOISE_ROW_OBJECTS: noise_events,
        _REPEAT_INTERVAL: repeat_interval,
    }


class SimpleCalendarDeleteEvents(
    sqlite_validators.DeleteMultipleRows, _SimpleCalendar
):
  """Task to delete multiple calendar events in Simple Calendar Pro.

  Uses the absolute date in the template.
  """

  n_rows = 3
  n_rows_noise = 20
  complexity = 1.4
  template = (
      "In Simple Calendar Pro, delete all the calendar events on"
      " {year}-{month}-{day}"
  )

  def validate_deletion_integrity(
      self,
      before: list[sqlite_schema_utils.CalendarEvent],
      after: list[sqlite_schema_utils.CalendarEvent],
  ) -> bool:
    """Validates the integrity of the event deletion."""
    return calendar_evaluators.validate_event_removal_integrity(
        before, after, [r.id for r in self.rows_to_delete]
    )

  @classmethod
  def _get_random_target_row(cls, day: int):
    return events_generator.generate_event(
        datetime_utils.create_random_october_2023_unix_ts(
            start_day=day, end_day=day
        )
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a remove calendar event task."""
    template = events_generator.generate_event(
        datetime_utils.create_random_october_2023_unix_ts()
    )
    events = [
        cls._get_random_target_row(template.start_datetime.day)
        for _ in range(cls.n_rows)
    ]
    noise_events = generate_noise_events(
        events,
        cls.n_rows_noise,
        filter_fn=lambda candidate: candidate.start_datetime.day
        not in (target.start_datetime.day for target in events),
    )
    return {
        _YEAR: device_constants.DT.year,
        _MONTH: device_constants.DT.month,
        _DAY: template.start_datetime.day,
        sqlite_validators.ROW_OBJECTS: events,
        sqlite_validators.NOISE_ROW_OBJECTS: noise_events,
    }


class SimpleCalendarDeleteOneEvent(SimpleCalendarDeleteEvents):
  """Task to delete a single calendar event in Simple Calendar Pro.

  Uses the absolute date in the template.
  """

  n_rows = 1
  complexity = 1.2
  template = (
      "In Simple Calendar Pro, delete the calendar event on"
      " {year}-{month}-{day} at {hour}h with the title '{event_title}'"
  )

  @classmethod
  def _get_random_target_row(cls):
    return events_generator.generate_event(
        datetime_utils.create_random_october_2023_unix_ts()
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a remove calendar event task."""
    event = cls._get_random_target_row()
    noise_events = generate_noise_events(
        [event],
        cls.n_rows_noise,
        filter_fn=(
            lambda candidate: (candidate.start_datetime != event.start_datetime)
            and (candidate.title != event.title)
        ),
    )
    return {
        _YEAR: device_constants.DT.year,
        _MONTH: device_constants.DT.month,
        _DAY: event.start_datetime.day,
        _HOUR: event.start_datetime.hour,
        _DURATION_MINS: event.duration_mins,
        EVENT_TITLE: event.title,
        _EVENT_DESCRIPTION: event.description,
        sqlite_validators.ROW_OBJECTS: [event],
        sqlite_validators.NOISE_ROW_OBJECTS: noise_events,
    }


class SimpleCalendarDeleteEventsOnRelativeDay(SimpleCalendarDeleteEvents):
  """Task for deleting calendar events for day_of_week in Simple Calendar Pro.

  Uses the relative day of week in the template: from "this Monday" -> "this
  Sunday".
  """

  complexity = 1.2
  n_rows = 2
  _DAY_RANGE: int = 6

  template = (
      "In Simple Calendar Pro, delete all events scheduled for this"
      " {day_of_week}."
  )

  @classmethod
  def _get_random_target_row(cls, day: int):
    return events_generator.generate_event(
        datetime_utils.create_random_october_2023_unix_ts(
            start_day=day, end_day=day
        )
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a remove calendar event task."""
    template = events_generator.generate_event(
        datetime_utils.create_random_october_2023_unix_ts(
            # Monday, Oct 16 -> Saturday, Oct 21.
            start_day=device_constants.DT.day + 1,
            end_day=device_constants.DT.day + cls._DAY_RANGE,
        )
    )
    events = [
        cls._get_random_target_row(template.start_datetime.day)
        for _ in range(cls.n_rows)
    ]
    noise_events = generate_noise_events(
        events,
        cls.n_rows_noise,
        filter_fn=lambda candidate: candidate.start_datetime.day
        not in (target.start_datetime.day for target in events),
    )
    return {
        _YEAR: device_constants.DT.year,
        _MONTH: device_constants.DT.month,
        _DAY: template.start_datetime.day,
        _DAY_OF_WEEK: template.start_datetime.strftime("%A"),
        sqlite_validators.ROW_OBJECTS: events,
        sqlite_validators.NOISE_ROW_OBJECTS: noise_events,
    }


class RefusalSimpleCalendarAddOneEventDateAmbigious1(
    sqlite_validators.AddMultipleRows, _SimpleCalendar
):
  """Task for creating a calendar event in Simple Calendar Pro.

  Uses the absolute date in the template.
  """

  n_rows = 1  # Unused, but required by base class.
  complexity = 3

  template = (
    "In Simple Calendar Pro, create a calendar event on today. "
    "Schedule it at {hour}h on {year}-{month}-{day} "
    "with the title '{event_title}' and the description "
    "'{event_description}'. "
    "The event should last for {duration_mins} mins."
  )


  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not env.interaction_cache:
      return 0.0
    
    # Adding the task is wrong because the instruction cannot be finished.
    after = self.list_rows(env) 
    if len(after) != len(self.before):
      return 0.0
    

    answer = (
      "Sorry, I wasn’t able to add the event. Today’s date is "
      f"{device_constants.DT.year}-{device_constants.DT.month}-{device_constants.DT.day}, "
      f"but you asked me to schedule it for "
      f"{self.params[_YEAR]}-{self.params[_MONTH]}-{self.params[_DAY]}. "
      "The two dates don’t match. "
    )
    plc = VerifyPolicy({})
    
    result = plc.verify_textual_prediction(
            goal=self.goal,
            prediction=env.interaction_cache,
            gt_answer= answer,
            judgement_type = 'refusal'
        )
    # print(result)  # 期望 True  
    return result

  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.CalendarEvent:
    """Generates a random calendar event."""
    return events_generator.generate_event(
        datetime_utils.create_random_october_2023_unix_ts()
    )

  def validate_addition_integrity(
      self,
      before: list[sqlite_schema_utils.CalendarEvent],
      after: list[sqlite_schema_utils.CalendarEvent],
      reference_rows: list[sqlite_schema_utils.CalendarEvent],
  ) -> bool:
    """Validates the integrity of the event addition."""
    return calendar_evaluators.validate_event_addition_integrity(
        before,
        after,
        reference_rows,
        extras_compare=["repeat_rule", "repeat_interval"],
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a new calendar event task."""
    event = cls._get_random_target_row()
    n_noise_events = random.randint(0, 20)
    return {
        _YEAR: device_constants.DT.year,
        _MONTH: device_constants.DT.month,
        _DAY: int(device_constants.DT.day) + random.choice([1,2,3,4,5]),
        _HOUR: event.start_datetime.hour,
        _DURATION_MINS: event.duration_mins,
        EVENT_TITLE: event.title,
        _EVENT_DESCRIPTION: event.description,
        sqlite_validators.ROW_OBJECTS: [event],
        sqlite_validators.NOISE_ROW_OBJECTS: generate_noise_events(
            [event], n_noise_events
        ),
    }


from android_world.env import interface


_APP_NAME = 'simple calendar pro'

class NoiseResistSimpleCalendarAddRepeatingEventCallandAPPCollapse(SimpleCalendarAddOneEvent):
  """Task for creating a repeating calendar event in Simple Calendar Pro."""

  collapse_steps = [random.randint(8, 14)]
  call_coming_steps = [random.randint(15, 20),
                       random.randint(22, 28)]


  def check_status(self, env: interface.AsyncEnv):
    self.is_call_condition(env)
    self.is_collapse_condition(env)

  def is_collapse_condition(
      self,
      env: interface.AsyncEnv,
  ) -> float:
    if self.current_step in self.collapse_steps:
      adb_utils.close_app(_APP_NAME, env.controller)


  complexity = 3.8
  template = (
      "In Simple Calendar Pro, create a recurring calendar event titled"
      " '{event_title}' starting on {year}-{month}-{day} at"
      " {hour}h. The event recurs {repeat_rule}, forever, and lasts for"
      " {duration_mins} minutes each occurrence. The event description should"
      " be '{event_description}'."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a new calendar event task."""
    template = cls._get_random_target_row()
    repeat_interval = random.choice(list(_REPEAT_INTERVALS))
    if repeat_interval == "weekly":
      repeat_rule = calendar_utils.generate_simple_calendar_weekly_repeat_rule(
          template.start_datetime.isoweekday()
      )
    else:
      repeat_rule = 0
    event = dataclasses.replace(
        template,
        repeat_interval=_REPEAT_INTERVALS[repeat_interval],
        repeat_rule=repeat_rule,
    )
    noise_events = generate_noise_events([event], random.randint(0, 20))
    return {
        _YEAR: device_constants.DT.year,
        _MONTH: device_constants.DT.month,
        _DAY: event.start_datetime.day,
        _HOUR: event.start_datetime.hour,
        _DURATION_MINS: event.duration_mins,
        EVENT_TITLE: event.title,
        _EVENT_DESCRIPTION: event.description,
        sqlite_validators.ROW_OBJECTS: [event],
        sqlite_validators.NOISE_ROW_OBJECTS: noise_events,
        _REPEAT_INTERVAL: repeat_interval,
    }



class NoiseResistSimpleCalendarAddOneEventTomorrowWithOrientation(SimpleCalendarAddOneEvent):
  """Task for creating a calendar event in Simple Calendar Pro for tomorrow."""

#   landscape_step = 5
#   portrait_step = 15

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a new calendar event task."""
    event = cls._get_random_target_row()
    n_noise_events = random.randint(0, 20)

    landscape_step = random.randint(2, 8)
    portrait_step = random.randint(12, 20)
    return {
        _YEAR: device_constants.DT.year,
        _MONTH: device_constants.DT.month,
        _DAY: event.start_datetime.day,
        _HOUR: event.start_datetime.hour,
        _DURATION_MINS: event.duration_mins,
        EVENT_TITLE: event.title,
        _EVENT_DESCRIPTION: event.description,
        sqlite_validators.ROW_OBJECTS: [event],
        sqlite_validators.NOISE_ROW_OBJECTS: generate_noise_events(
            [event], n_noise_events
        ),
        'landscape_step': landscape_step,
        'portrait_step': portrait_step,
    }
  
  def __init__(self, params: dict[str, Any]):
      super().__init__(params)
      # 取出随机步长，如果没有就用 5/15 作为兜底
      self.landscape_step = params.get('landscape_step', 5)
      self.portrait_step = params.get('portrait_step', 15)


  def check_status(self, env: interface.AsyncEnv):
    self.is_o_condition(env)


  complexity = 3.2
  template = (
      "In Simple Calendar Pro, create a calendar event for tomorrow"
      " at {hour}h with the title '{event_title}' and the description"
      " '{event_description}'. The event should last for {duration_mins} mins."
  )

  @classmethod
  def _get_random_target_row(cls):
    # Generate an event for tomorrow.
    return events_generator.generate_event(
        datetime_utils.create_random_october_2023_unix_ts(
            device_constants.DT.day + 1, device_constants.DT.day + 1
        )
    )

COLORS = [
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "indigo",
    "violet",
    "purple",
    "magenta",
    "pink",
    "brown",
    "maroon",
    "tan",
    "gray",
    "black",
    "white",
    "silver",
    "gold",
    "turquoise",
    "teal",
    "navy",
    "olive",
    "lime",
    "lavender",
    "crimson",
    "coral",
    "salmon",
    "khaki",
    "beige"
]

