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
# Changes: Modify _run_task function to support saving screenshot of last step in one episode.


"""Utilities for evaluating automation agents."""
import yaml
from PIL import Image
import io
import json
import numpy as np
import collections
import datetime
import hashlib
import logging
import os
import random
import time
import traceback
from typing import Any, Callable, Type, TypeVar

from android_env import env_interface
from android_world import checkpointer as checkpointer_lib
from android_world import constants
from android_world import episode_runner
from android_world.agents import base_agent
from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.miniwob import miniwob_base
from fuzzywuzzy import process
import numpy as np
import pandas as pd

# A fixed seed to use when use identical parameters but seed is not set.
_FIXED_SEED = 123
_TASK_TEMPLATE_COLUMN = 'task_template'
_TASK_PROMPT_COLUMN = 'task_prompt'
TaskEvalType = TypeVar('TaskEvalType', bound=task_eval.TaskEval)
current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(current_dir)

yaml_path = os.path.join(project_root, 'config', 'venus_benchmark_settings.yaml')


class Suite(dict[str, list[task_eval.TaskEval]]):
  """A suite of tasks.

  Each key is the task name as defined in registry.py and its value is a list
  of instantiated task objects. These instances differ from each other by their
  parameter initializations; i.e. each task will have different task parameters.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._suite_family = None

  @property
  def suite_family(self) -> str:
    """Getter for suite_family."""
    if self._suite_family is None:
      raise ValueError('Suite family is not set; please first set it.')
    return self._suite_family

  @suite_family.setter
  def suite_family(self, value: str):
    """Setter for suite_family."""
    self._suite_family = value


def _instantiate_task(
    task: Type[task_eval.TaskEval],
    params: dict[str, Any] | None = None,
    seed: int | None = None,
    env: interface.AsyncEnv | None = None,
) -> task_eval.TaskEval:
  """Creates an instance of a task with params.

  If params is not provided, it will use random params, controlled by a seed.

  Args:
    task: The task to instantiate.
    params: Params to use.
    seed: Seed for the random number generator.
    env: The environment.

  Returns:
    An instance of a task.
  """
  task.set_device_time(env)
  if params is None:
    if seed is not None:
      random.seed(seed)
    params = task.generate_random_params()
    # print(params)
    params[constants.EpisodeConstants.SEED] = seed
  return task(params)


def create_suite(
    task_registry: dict[str, Type[task_eval.TaskEval]],
    n_task_combinations: int = 1,
    seed: int | None = None,
    tasks: list[str] | None = None,
    use_identical_params: bool = False,
    env: interface.AsyncEnv | None = None
) -> Suite:
  """Creates task suite.

  A task suite is a set of tasks. Each task is instantiated
  `n_task_combinations` times using new parameters. For example a task suite
  could look like:

  ```python
  {
      'GoogleSearchTask': [
          GoogleSearchTask({'term': 'cute cats'}),
          GoogleSearchTask({'term': 'comfy pillows'}),
      ],
      'WifiDisable': [  # No params for WiFi task.
          WifiDisable({}),
          WifiDisable({}),
      ],
  }
  ```

  Args:
    task_registry: Maps task names to their TaskEvals.
    n_task_combinations: Number of instances to create per task. Each instance
      will have unique param combinations.
    seed: Seed for the random number generator. Setting the seed will result in
      the same sequence of params for task instantiation per each task.
    tasks: List of task types that should be in the suite. If value is `None`
      all task types and associated instances will be created.
    use_identical_params: If True, each instance of a task, for a total of
      `n_task_combinations`, will have the same params.
    env: The environment that will be run on.

  Returns:
    A mapping of task name to instances of the task.
  """

  def _get_instance_seed(name: str, i: int) -> int:
    unique_seed_str = f'{seed}_{name}_{i}'
    return int(hashlib.sha256(unique_seed_str.encode()).hexdigest(), 16) % (
        2**32
    )

  suite = {}
  for name, task_type in task_registry.items():
    current = []
    for i in range(n_task_combinations):
      if use_identical_params:
        instance_seed = (
            _get_instance_seed(name, 0) if seed is not None else _FIXED_SEED
        )
      elif seed is not None:
        instance_seed = _get_instance_seed(name, i)
      else:
        instance_seed = None
      current.append(_instantiate_task(task_type, seed=instance_seed, env=env))
    suite[name] = current
  
  # Retrieve tasks from suite.
  suite = _filter_tasks(suite, task_registry, tasks)

  # Sort suite alphabetically by task name.
  return Suite(sorted(suite.items()))


def _suggest_keyword(
    typo: str, keywords: list[str], threshold: int = 80
) -> str:
  """Suggests a keyword."""
  suggestion, score = process.extractOne(typo, keywords)
  if score >= threshold:
    return f" Did you mean '{suggestion}'?"
  else:
    return ''


def _filter_tasks(
    suite: dict[str, list[task_eval.TaskEval]],
    task_registry: dict[str, Type[task_eval.TaskEval]],
    tasks: list[str] | None = None,
) -> dict[str, list[task_eval.TaskEval]]:
  """Filters a suite by specific tasks.

  Args:
    suite: The suite to retrieve tasks from.
    task_registry: The task registry the suite is from.
    tasks: The tasks to retrieve. If None, just return entire suite.

  Returns:
    A "mini-suite" of tasks from suite.

  Raises:
    ValueError: If invalid task name.
  """
  if tasks is None:
    return suite
  subset = {}

  # Validate.
  for name in tasks:
    if name not in task_registry:
      raise ValueError(
          f'Task {name} not found in the task registry.'
          + _suggest_keyword(name, list(task_registry.keys()))
      )

  # Filter.
  for name, instances in suite.items():
    if name in tasks:
      subset[name] = instances
  return subset


def _calculate_total_output_chars(step_data: dict[str, Any]) -> int:
  """计算agent输出的所有字符数（仅统计LLM输出，不包括prompt和元数据）。
  
  Args:
    step_data: Episode中的step_data，包含agent在每一步的输出。
              格式: {key: [value1, value2, ...]}，其中list是多个step的结果
    
  Returns:
    总字符数
  """
  total_chars = 0
  
  output_keys = [
      'action_output',      # LLM action output
      'action_response',    # alternative action output (qwen3vl uses this)
      'summary',            # LLM summary output
      'action_reason',      # action reason
  ]
  
  try:
    #
    for key in output_keys:
      if key in step_data:
        outputs = step_data[key]
        if isinstance(outputs, list):
          for output in outputs:
            if output is not None and output != '':
              if isinstance(output, str):
                total_chars += len(output)
        elif outputs is not None and outputs != '' and isinstance(outputs, str):
          total_chars += len(outputs)
  except Exception:
    pass
  
  return total_chars


def _run_task(
    task: TaskEvalType,
    run_episode: Callable[[TaskEvalType], episode_runner.EpisodeResult],
    env: interface.AsyncEnv,
    demo_mode: bool,
    output_path: str,
    agent:  base_agent.EnvironmentInteractingAgent,
) -> dict[str, Any]:
  """Runs a task.

  Args:
    task: The task.
    run_episode: Runs the agent on the task.
    env: Environment that will be run on.
    demo_mode: Whether running in demo mode; will display success overlay if so.
    output_path: Path storing traj screenshots.

  Returns:
    Episode data and associated success signals.

  Raises:
    ValueError: If step data was not as expected.
  """
  start = time.time()
  try:

    task.initialize_task(env)
    print("MRO: ", type(task).mro())

    logging.info('Running task %s with goal "%s"', task.name, task.goal)
    print(f'Running task {task.name} with goal "{task.goal}"')

    interaction_results = run_episode(task)
    
    # step
    step_data = interaction_results.step_data
    agent_name = interaction_results.agent_name
    
    screenshot_key = 'raw_screenshot'
    screenshot_key = None

    print(f"agent_name: {agent_name}")
    print(f"agent_name: {agent.name}")
    if agent_name == 'eant1' or agent_name == 'eant2':
      screenshot_key = 'raw_screenshot'
      screenshot_task_end_state = interaction_results.step_data[screenshot_key][-1]

    elif agent_name == 'qwen3vl' or agent_name == 'gui_owl':
      screenshot_key = 'screenshot'
      screenshot_task_end_state = interaction_results.step_data[screenshot_key][-1]
      last_step_idx = len(interaction_results.step_data[screenshot_key])

    else:
      state = agent.get_post_transition_state()
      screenshot_task_end_state = state.pixels.copy()

    # Locating tasks need the screenshot in last step.

    with open(yaml_path, 'r', encoding='utf-8') as f:
      config = yaml.safe_load(f)
    locating_results_path = config.get('locating_results_path')

    task_output_dir = os.path.join(
      locating_results_path, 
      agent_name,
      task.name
      )
    last_screenshot_path = os.path.join(task_output_dir, f"screenshot_task_end_state.png")
    
    if not os.path.exists(task_output_dir):
        os.makedirs(task_output_dir)
    

    screenshot = Image.fromarray(screenshot_task_end_state)
    screenshot.save(last_screenshot_path)

    task.last_screenshot = screenshot_task_end_state
    task.last_screenshot_path = last_screenshot_path

    # Task result judgement.
    if interaction_results.done == False:
      # Agent does not indicate whether the task is finished.
      task_successful = 0.0
    else:
      print("interaction_results.done == True")
      print('Start evaluating the WHOLE TASK.')
      task_successful = task.is_successful(env)

  except Exception as e:  # pylint: disable=broad-exception-caught
    print('~' * 80 + '\n' + f'SKIPPING {task.name}.')
    logging.exception(
        'Logging exception and skipping task. Will keep running. Task: %s: %s',
        task.name,
        e,
    )
    traceback.print_exc()
    return _create_failed_result(
        task.name, task.goal, traceback.format_exc(), time.time() - start
    )
  else:
    # agent_successful = task_successful if interaction_results.done else 0.0
    agent_successful = task_successful
    print(
        f'{"Task Successful ✅" if agent_successful > 0.5 else "Task Failed ❌"};'
        f' {task.goal}'
    )

    demo_mode_val = demo_mode.value if hasattr(demo_mode, 'value') else demo_mode
    if demo_mode_val:
      _display_success_overlay(env.controller, agent_successful)


    print(interaction_results.step_data.keys())
    # mobile agent v3

    if agent_name == 'mobile_agent_v3':
      # total_output_chars = _calculate_total_output_chars(interaction_results.step_data)
      raw_response_list = interaction_results.step_data['raw_response_list']
      total_output_chars = 0
      
      if raw_response_list:
        last_step_responses = raw_response_list[-1]  # 获取最后一步的完整响应列表
        for response in last_step_responses:
          total_output_chars += len(response)
      print('total_output_chars: ', total_output_chars)

      raw_response_path = os.path.join(task_output_dir, 'raw_response_list.json')
      try:
          with open(raw_response_path, 'w', encoding='utf-8') as f:
              json.dump(raw_response_list, f, ensure_ascii=False, indent=2)
          print(f'raw_response_list saved to: {raw_response_path}')
      except Exception as e:
          print(f'Failed to save raw_response_list: {e}')
      
      interaction_results.step_data['raw_response_list'] = []
      print('raw_response_list cleared from step_data')

    else:
      total_output_chars = _calculate_total_output_chars(interaction_results.step_data)

    result = {
        constants.EpisodeConstants.GOAL: task.goal,
        constants.EpisodeConstants.TASK_TEMPLATE: task.name,
        constants.EpisodeConstants.EPISODE_DATA: interaction_results.step_data,
        constants.EpisodeConstants.IS_SUCCESSFUL: agent_successful,
        constants.EpisodeConstants.RUN_TIME: time.time() - start,
        constants.EpisodeConstants.FINISH_DTIME: datetime.datetime.now(),
        constants.EpisodeConstants.EPISODE_LENGTH: len(
            interaction_results.step_data[constants.STEP_NUMBER]
        ),
        constants.EpisodeConstants.AUX_DATA: interaction_results.aux_data,
        constants.EpisodeConstants.SCREEN_CONFIG: _get_screen_config(task),
        constants.EpisodeConstants.EXCEPTION_INFO: None,
        constants.EpisodeConstants.SEED: task.params[
            constants.EpisodeConstants.SEED
        ],
        'total_output_chars': total_output_chars,
    }
    # Whole task is evaluated. The result is constructed.
    task.tear_down(env)
    return result


def _get_task_info(
    episodes: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
  """Gets task info from episodes.

  Args:
    episodes: Episodes to get info from.

  Returns:
    A tuple of completed and failed task lookup tables.
  """

  completed = collections.defaultdict(list)
  failed = collections.defaultdict(list)
  for episode in episodes:
    instance_name = (
        episode[constants.EpisodeConstants.TASK_TEMPLATE]
        + checkpointer_lib.INSTANCE_SEPARATOR
        + str(episode[constants.EpisodeConstants.INSTANCE_ID])
    )
    if episode.get(constants.EpisodeConstants.EXCEPTION_INFO) is not None:
      failed[instance_name].append(episode)
    else:
      completed[instance_name].append(episode)
  return completed, failed


from android_world.env import tools

def initialize_chrome(env):
  print("Running additional chrome initialization...")
  # handle chrome initialization problem for browser tasks
  adb_utils.launch_app("chrome", env.controller)
  time.sleep(5)

  tool_controller = tools.AndroidToolController(env=env.controller)
  time.sleep(2)

  first_op = False
  try:
    print("try first variant...")
    tool_controller.click_element("Use without an account")
    time.sleep(5.0)
    first_op = True
  except:
    print("Failed to click 'Use without an account' button.")
    pass
  
  if not first_op:
    print("try second variant...")
    try:
      tool_controller.click_element("Accept & continue")
    except:
      pass
    time.sleep(3.0)
    try:
      tool_controller.click_element("No thanks")
    except:
      pass
    time.sleep(5.0)
    
  adb_utils.press_home_button(env.controller)
  time.sleep(2.0)
  print("Done additional chrome initialization")


def _run_task_suite(
    suite: Suite,
    run_episode: Callable[[task_eval.TaskEval], episode_runner.EpisodeResult],
    env: interface.AsyncEnv,
    agent: base_agent.EnvironmentInteractingAgent,
    checkpointer: checkpointer_lib.Checkpointer = checkpointer_lib.NullCheckpointer(),
    demo_mode: bool = False,
    agent_name: str = '',
    return_full_episode_data: bool = False,
    process_episodes_fn=None,
    check_episode_fn: Callable[[dict[str, Any]], bool] | None = None,

    
) -> list[dict[str, Any]]:
  """Runs e2e system on suite.

  Args:
    suite: The suite to run it on.
    run_episode: The e2e system. See run_suite.py for an example.
    env: The environment e2e system runs on.
    checkpointer: See docstring from `run`.
    demo_mode: Whether to display the scoreboard.
    agent_name: The name of the agent.
    return_full_episode_data: Whether to return full episode data instead of
      just metadata.
    process_episodes_fn: The function to process episode data. Usually to
      compute metrics. Deafaults to process_episodes from this file.
    check_episode_fn: The function to check episode data.

  Returns:
    Metadata for each episode, including the scripted reward.
  """

  output_path = agent.output_path

  metadata_fields = [
      constants.EpisodeConstants.GOAL,
      constants.EpisodeConstants.TASK_TEMPLATE,
      constants.EpisodeConstants.INSTANCE_ID,
      constants.EpisodeConstants.IS_SUCCESSFUL,
      constants.EpisodeConstants.EPISODE_LENGTH,
      constants.EpisodeConstants.RUN_TIME,
      constants.EpisodeConstants.EXCEPTION_INFO,
      constants.EpisodeConstants.AUX_DATA,
      'total_output_chars',
  ]
  completed_tasks, failed_tasks = _get_task_info(
      checkpointer.load(fields=metadata_fields)
  )
  if process_episodes_fn is None:
    process_episodes_fn = process_episodes

  if (completed_tasks or failed_tasks) and return_full_episode_data:
    raise ValueError(
        'Cannot return full episode data when resuming from a checkpoint.'
    )
  episodes_metadata: list[dict[str, Any]] = []
  full_episode_data = []
  correct, total = 0, 0

  # initialize_chrome(env=env)

  for name, instances in suite.items():
    msg = 'Running task: ' + name
    print(msg + '\n' + '=' * len(msg))

    for i, instance in enumerate(instances):
      instance_name = (
          instance.name + checkpointer_lib.INSTANCE_SEPARATOR + str(i)
      )
      # Transferring from old checkpoint.
      if instance_name in completed_tasks:
        completed_episodes: list[dict[str, Any]] = completed_tasks[
            instance_name
        ]
        episodes_metadata.extend(completed_episodes)
      if instance_name in failed_tasks:
        episodes_metadata.extend(failed_tasks[instance_name])
      already_processed = (
          instance_name in completed_tasks and instance_name not in failed_tasks
      )
      if already_processed:
        print(f'Skipping already processed task {instance_name}')
        continue

      episode = _run_task(instance, run_episode, env, demo_mode=demo_mode, output_path=output_path, agent=agent)
      # print(episode.keys())
      
      if (
          episode.get(constants.EpisodeConstants.EXCEPTION_INFO) is None
          and check_episode_fn is not None
      ):
        if not check_episode_fn(episode):
          continue
      episode[constants.EpisodeConstants.AGENT_NAME] = agent_name
      episode[constants.EpisodeConstants.INSTANCE_ID] = i
      checkpointer.save_episodes([episode], instance_name)

      if return_full_episode_data:
        full_episode_data.append(episode)

      
      episodes_metadata.append({k: episode.get(k) for k in metadata_fields})
      
      process_episodes_fn(episodes_metadata, print_summary=True)

      if episode[constants.EpisodeConstants.EXCEPTION_INFO] is not None:
        # Don't include episode in tally if execution/eval logic errored out.
        continue
      correct += episode[constants.EpisodeConstants.IS_SUCCESSFUL]
      total += 1
      demo_mode_val = demo_mode.value if hasattr(demo_mode, 'value') else demo_mode
      if demo_mode_val:
        _update_scoreboard(correct, total, env.controller)
    print()

  tagged_result_df = process_episodes_fn(episodes_metadata, print_summary=True)
  df_path = os.path.join(checkpointer.directory, 'result.csv')
  print(f'{tagged_result_df}\n, tagged_result_df type: {type(tagged_result_df)}')
  tagged_result_df.to_csv(df_path, index=True)
  return full_episode_data if return_full_episode_data else episodes_metadata


def run(
    suite: Suite,
    agent: base_agent.EnvironmentInteractingAgent,
    checkpointer: checkpointer_lib.Checkpointer = checkpointer_lib.NullCheckpointer(),
    demo_mode: bool = False,
    return_full_episode_data: bool = False,
    process_episodes_fn=None,
    check_episode_fn: Callable[[dict[str, Any]], bool] | None = None,
    dark_mode: str = 'off',
    pad_mode: str = 'off',

) -> list[dict[str, Any]]:
  """Create suite and runs eval suite.

  Args:
    suite: The suite of tasks to run on.
    agent: An agent that interacts on the environment.
    checkpointer: Checkpointer that loads from existing run and resumes from
      there. NOTE: It will resume from the last fully completed task template.
      Relatedly, data for a task template will not be saved until all instances
      are executed.
    demo_mode: Whether to run in demo mode, which displays a scoreboard and the
      task instruction as a notification.
    return_full_episode_data: Whether to return full episode data instead of
      just metadata.
    process_episodes_fn: The function to process episode data. Usually to
      compute metrics. Deafaults to process_episodes from this file.
    check_episode_fn: The function to check episode data.

  Returns:
    Step-by-step data from each episode.
  """

  def run_episode(task: task_eval.TaskEval) -> episode_runner.EpisodeResult:
    demo_mode_val = demo_mode.value if hasattr(demo_mode, 'value') else demo_mode
    if demo_mode_val:
      _display_goal(agent.env, task)

    return episode_runner.run_episode(
        task=task,
        goal=task.goal,
        agent=agent,
        max_n_steps=_allocate_step_budget(task.complexity),
        start_on_home_screen=task.start_on_home_screen,
        termination_fn=(
            miniwob_base.is_episode_terminated
            if task.name.lower().startswith('miniwob')
            else None
        ),
    )

  # Handle FlagHolder objects by accessing .value if present
  demo_mode_val = demo_mode.value if hasattr(demo_mode, 'value') else demo_mode
  dark_mode_val = dark_mode.value if hasattr(dark_mode, 'value') else dark_mode
  pad_mode_val = pad_mode.value if hasattr(pad_mode, 'value') else pad_mode
  
  if demo_mode_val:
    adb_utils.send_android_intent(
        'broadcast',
        'com.example.ACTION_UPDATE_SCOREBOARD',
        agent.env.controller,
        extras={'player_name': agent.name, 'scoreboard_value': '00/00'},
    )

  if dark_mode_val == 'on':
    res = adb_utils.issue_generic_request(
        ["shell", "cmd", "uimode", "night", "yes"],
        agent.env.controller,
    )
    
  else:
    res = adb_utils.issue_generic_request(
        ["shell", "cmd", "uimode", "night", "no"],
        agent.env.controller,
    )

  
  
  logging.info(f"Dard mode: {res.generic.output.decode().strip()}")
  
  results = _run_task_suite(
      suite=suite,
      run_episode=run_episode,
      env=agent.env,
      checkpointer=checkpointer,
      demo_mode=demo_mode_val,
      agent_name=agent.name,
      return_full_episode_data=return_full_episode_data,
      process_episodes_fn=process_episodes_fn,
      check_episode_fn=check_episode_fn,
      agent=agent,
  )

  return results


def _allocate_step_budget(task_complexity: float) -> int:
  """Allocates number of steps dynamically based on the complexity score.

  Args:
    task_complexity: Complexity score of the task.

  Returns:
    Allocated number of steps for the task.
  """
  if task_complexity is None:
    raise ValueError('Task complexity must be provided.')
  return int(10 * (task_complexity))


def _display_message(
    header: str, body: str, env: env_interface.AndroidEnvInterface
) -> None:
  adb_utils.send_android_intent(
      'broadcast',
      'com.example.ACTION_UPDATE_OVERLAY',
      env,
      extras={'task_type_string': header, 'goal_string': body},
  )


def _display_goal(env: interface.AsyncEnv, task: task_eval.TaskEval) -> None:
  """Displays the goal on the screen using Android World.

  Args:
    env: The environment.
    task: The current task.
  """
  adb_utils.launch_app('android world', env.controller)
  time.sleep(1.0)
  _display_message(task.goal, task.name, env.controller)
  time.sleep(6.0)
  adb_utils.press_home_button(env.controller)
  time.sleep(1.0)


def _get_screen_config(task: task_eval.TaskEval) -> dict[str, Any]:
  return {
      'width': task.width if hasattr(task, 'width') else 1080,
      'height': task.height if hasattr(task, 'height') else 2400,
      'orientation': (
          task.orientation if hasattr(task, 'orientation') else 'portrait'
      ),
      'config_name': (
          task.config_name if hasattr(task, 'config_name') else 'default'
      ),
  }


def _create_failed_result(
    name: str, goal: str, exception: str, run_time: float
) -> dict[str, Any]:
  """Creates empty result to use if the run fails for some reason."""
  return {
      constants.EpisodeConstants.GOAL: goal,
      constants.EpisodeConstants.TASK_TEMPLATE: name,
      constants.EpisodeConstants.EPISODE_DATA: np.nan,
      constants.EpisodeConstants.IS_SUCCESSFUL: np.nan,
      constants.EpisodeConstants.FINISH_DTIME: datetime.datetime.now(),
      constants.EpisodeConstants.RUN_TIME: run_time,
      constants.EpisodeConstants.EPISODE_LENGTH: np.nan,
      constants.EpisodeConstants.EXCEPTION_INFO: exception,
      constants.EpisodeConstants.AUX_DATA: None,
  }


def _display_success_overlay(
    env: env_interface.AndroidEnvInterface, success: float
) -> None:
  """Displays success overlay."""
  adb_utils.send_android_intent(
      'broadcast',
      'com.example.ACTION_UPDATE_OVERLAY',
      env,
      extras={'success_string': str(int(success))},
  )
  time.sleep(1.0)  # Let display linger.


def _update_scoreboard(
    n_correct: int, n: int, env: env_interface.AndroidEnvInterface
) -> None:
  """Updates the scoreboard."""
  percentage = (n_correct / n) * 100
  scoreboard_value = f'{n_correct}/{n} ({percentage:.1f}%)'

  adb_utils.send_android_intent(
      'broadcast',
      'com.example.ACTION_UPDATE_SCOREBOARD',
      env,
      extras={'scoreboard_value': scoreboard_value},
  )


def _extract_task_metadata() -> pd.DataFrame:
  """Extracts metadata from task_metadata.json."""
  name = 'task_metadata.json'
  filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
  df = pd.read_json(filepath)
  df.rename(columns={_TASK_TEMPLATE_COLUMN: _TASK_PROMPT_COLUMN}, inplace=True)
  df.rename(columns={'task_name': _TASK_TEMPLATE_COLUMN}, inplace=True)
  return df.set_index(_TASK_TEMPLATE_COLUMN)[
      ['difficulty', 'optimal_steps', 'tags']
  ]


def _print_results_by_tag(result_df: pd.DataFrame) -> None:
  exploded_df = result_df.explode('tags').reset_index()
  exploded_df.replace(regex={'tags': r''}, value='untagged', inplace=True)  # pytype: disable=wrong-arg-types
  return (
      exploded_df.groupby(['tags', 'difficulty'], as_index=False)
      .agg(
          num_tasks=(_TASK_TEMPLATE_COLUMN, 'count'),
          mean_success_rate=('mean_success_rate', 'mean'),
      )
      .pivot_table(
          index=['tags'],
          columns='difficulty',
          values=[
              'mean_success_rate',
          ],
      )
      .fillna('-')
      .reindex(columns=['easy', 'medium', 'hard'], level='difficulty')
  )


def process_episodes(
    episodes: list[dict[str, Any]], print_summary: bool = False
) -> pd.DataFrame:
  """Processes task suite results; i.e. the output from `run_task_suite`.

  results = run_task_suite(...)
  # Contents of results.
  results = [
    {
        'goal': 'Pause the stopwatch.',
        'task_template': 'ClockStopWatchPaused',
        'episode_data': ...,
        'is_successful': True
    },
    {
        'goal': 'Pause the stopwatch.',
        'task_template': 'ClockStopWatchPaused',
        'episode_data': ...,
        'is_successful': False
    },
    {
        'goal': 'Run the stopwatch.',
        'task_template': 'ClockStopWatchRunnin',
        'episode_data': ...,
        'is_successful': True
    },
    {
        'goal': 'Run the stopwatch.',
        'task_template': 'ClockStopWatchRunnin',
        'episode_data': ...,
        'is_successful': True
    }
  ]

  process_episodes(results)
  # Output:
  # | task_template               |   n_trials |   average_success_rate |
  # |:----------------------------|-----------:|-----------------------:|
  # | ClockStopWatchPausedVerify  |          2 |                   0.5  |
  # | ClockStopWatchRunning       |          2 |                   1    |
  # | ==========Average========== |          2 |                   0.75 |

  Args:
    episodes: Results from running `run_task_suite`.
    print_summary: Whether to print the dataframe with a summary row.

  Returns:
    A dataframe aggregating results of run.
  """

  df = pd.DataFrame(list(episodes))

  # Add exeception info for backwards compatibility.
  df = df.assign(**{
      constants.EpisodeConstants.EXCEPTION_INFO: df.get(
          constants.EpisodeConstants.EXCEPTION_INFO, np.nan
      )
  })

  df['total_output_chars'] = df.get('total_output_chars', 0)

  result_df = df.groupby(
      constants.EpisodeConstants.TASK_TEMPLATE, dropna=True
  ).agg({
      constants.EpisodeConstants.IS_SUCCESSFUL: ['count', 'mean'],
      constants.EpisodeConstants.EPISODE_LENGTH: 'mean',
      constants.EpisodeConstants.RUN_TIME: 'sum',
      constants.EpisodeConstants.EXCEPTION_INFO: [
          ('none_count', lambda x: x.notnull().sum())
      ],
      'total_output_chars': 'sum',  # 添加字符数求和
  })
  result_df = result_df.sort_index()
  result_df.columns = [
      'num_complete_trials',
      'mean_success_rate',
      'mean_episode_length',
      'total_runtime_s',
      'num_fail_trials',
      'total_output_chars',
  ]
  result_df['total_runtime_s'] = result_df['total_runtime_s'].map(
      lambda x: float('{:.1f}'.format(x))
  )
  
  result_df['avg_output_chars'] = result_df.apply(
      lambda row: row['total_output_chars'] / row['num_complete_trials'] if row['num_complete_trials'] > 0 else 0,
      axis=1
  )
  result_df['avg_output_chars'] = result_df['avg_output_chars'].map(
      lambda x: int(x) if pd.notna(x) else 0
  )

  # Extract metadata and merge with the results table.
  metadata_df = _extract_task_metadata()
  tagged_result_df = result_df.merge(
      metadata_df, on=[_TASK_TEMPLATE_COLUMN], how='left'
  )

  if print_summary:
    numeric_cols = ['num_complete_trials', 'mean_success_rate', 'mean_episode_length', 
                   'total_runtime_s', 'num_fail_trials', 'total_output_chars', 'avg_output_chars']
    
    avg = result_df[numeric_cols].mean(axis=0)
    avg.name = '========= Average ========='

    result = pd.concat([result_df, avg.to_frame().T])
    result.index.name = 'task'
    result.insert(0, 'task_num', list(range(len(result) - 1)) + [0])
    result.task_num = result.task_num.astype(int)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.width', 1200)
    print(f'\n\n{result}')

    # Add a chart that shows mean success rate by tag and difficulty.
    tags_df = _print_results_by_tag(tagged_result_df)
    pd.set_option('display.precision', 2)
    print(f'\n\n{tags_df}')

  return tagged_result_df

