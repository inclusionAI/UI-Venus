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
# Changes: Modify run_episode function to support multi round interaction.


"""Runs an agent on the environment."""

import dataclasses
from typing import Any, Callable, Optional
from android_world import constants
from android_world.agents import base_agent
from android_world.env import interface
import termcolor

from android_world.task_evals import task_eval

@dataclasses.dataclass()
class EpisodeResult:
  """Represents an episode of an agent interacting with the environment.

  Attributes:
    done: Whether the agent indicated the task is complete.
    step_data: Environment and agent data for each step.
    env_reward: Reward returned by environment, if applicable.
    aux_data: Additional data from the episode which may be used for metrics.
  """

  done: bool
  step_data: dict[str, Any]
  agent_name: str = ''
  env_reward: Optional[float] = None
  aux_data: Optional[dict[str, Any]] = None


def run_episode(
    task: task_eval.TaskEval,
    goal: str,
    agent: base_agent.EnvironmentInteractingAgent,
    max_n_steps: int = 10,
    start_on_home_screen: bool = False,
    termination_fn: Callable[[interface.AsyncEnv], float] | None = None,
    print_fn: Callable[[str], None] = print,
) -> EpisodeResult:
  """Runs an agent on goal, e.g., "turn off wifi".

  An agent will start from whatever state the provided environment is in and
  run until it determines a task is complete, if the max number of
  steps is reached, of if the termination_fn is True.

  Args:
    goal: The goal instruction for the agent.
    agent: The agent to run on the environment.
    max_n_steps: The max number of steps to allow an agent to run before ending
      an episode.
    start_on_home_screen: Whether to start episode from the home screen or just
      the current screen.
    termination_fn: If provided, a determines whether to terminate an episode.
      For example, for MiniWoB++ tasks, the episode should terminate if there is
      a nonzero reward.
    print_fn: A function to print log messages to the console or logger.

  Returns:
    Data collected during running agent on goal.
  """
  if max_n_steps == 0:
    return EpisodeResult(done=False, 
                         step_data={}, 
                         agent_name=agent.name)
  if termination_fn is None:
    termination_fn = lambda env: False

  agent.reset(task.start_on_home_screen)
  # agent.reset(start_on_home_screen)
  agent.set_max_steps(max_n_steps)

  output = []

  goal_history = {}
  # Episode running
  for step_n in range(max_n_steps):
    # agent current step of the task  update
    task.current_step = step_n
    # update agent status
    task.check_status(env=agent.env)

    raw_goal = task.goal
    current_round = task.round
    # 维护 history
    if current_round not in goal_history:
        goal_history[current_round] = raw_goal
    
    if task.max_round == 1:
        # generate goal of singleround task
        final_goal = raw_goal

    else:
      # generate multiround goal
      # current_round_goal = goal
      # if current_round == 0:
      #   final_goal = "Round 1 goal: " + raw_goal
      # else:
      #   parts = []
      #   for r in range(current_round + 1):
      #       prefix = f"Round {r + 1} goal:"
      #       parts.append(f"{prefix} {goal_history[r]}")
      #       if r < current_round:                  # 除最后一段外都追加标记
      #         parts.append(f"Round {r + 1} goal finished")
      #   final_goal = "\n\n".join(parts)

        # Example
        # Round 1 goal: Add the following expenses into the pro expense:
        # Expense: Shoes
        #  amount_dollars: $89.06
        #  category_name: Clothes
        #  note: I may repeat this

        # Round 1 goal finished

        # Round 2 goal: Continue adding another expense identical to the one above, except that its name is different—use h as the name.
        
        # Simply use raw_goal as multiround goals
        final_goal = raw_goal

    # agent step once
    result = agent.step(final_goal, task.step_numb)
    # print(result.data.keys())

    print_fn('Completed step {:d}.'.format(step_n + 1))
    assert constants.STEP_NUMBER not in result.data
    output.append(result.data | {constants.STEP_NUMBER: step_n})
    
    if termination_fn(agent.env):
      print_fn('Environment ends episode.')
      return EpisodeResult(
          done=True,
          step_data=_transpose_lod_to_dol(output),
          agent_name=agent.name,
      )
    elif result.done:
      
      print_fn(f'Agent indicates task in task.round {task.round} is done.')

      
      if task.round + 1 < task.max_round:
        # Move to the next round.

        # Current round is finished BUT the whole task is not finished.
        # So turn the 
        result.done = False
        # Evaluate the sub-task from round 0 to round max_round-2
        sub_task_success = task.is_successful(env=agent.env)
        # After evaluation, the whole sub-task is complete.
        # According to the evalution result of sub-task,
        # program continus to next sub-task or return failure.
        task.round = task.round + 1
        if sub_task_success == 1:
          continue
        else:
          # sub_task fails. Then just return EpisodeResult 
          # in which result.done = False.
          # Finally, the whole task will be judge failure due to result.done = False.
          return EpisodeResult(
            done=result.done,
            step_data=_transpose_lod_to_dol(output),
            agent_name=agent.name,
         )
      else:
        # Reach the max round. Episode ends.
        return EpisodeResult(
            done=result.done,
            step_data=_transpose_lod_to_dol(output),
            agent_name=agent.name,
        )

  # Do not reach the max round. Episode ends.
  # For example, max_round = 1. But result.done = False, task.round still is 0.
  print_fn(
      termcolor.colored(
          'Agent did not indicate task is done. Reached max number of steps.',
          'red',
      )
  )
  return EpisodeResult(
      done=result.done, 
      step_data=_transpose_lod_to_dol(output),  # pylint: disable=undefined-variable
      agent_name=agent.name,
  )


def _transpose_lod_to_dol(data: list[dict[str, Any]]) -> dict[str, list[Any]]:
  """Transposes a list of dictionaries to a dictionary of lists.

  Args:
    data: A list of dictionaries.

  Returns:
    A dictionary where each key is from the input dictionaries and each value is
    a list of values for that key.
  """
  result = {}
  for d in data:
    for key, value in d.items():
      if key not in result:
        result[key] = []
      result[key].append(value)
  return result


def transpose_dol_to_lod(data: dict[str, list[Any]]) -> list[dict[str, Any]]:
  """Converts a dictionary of lists to a list of dictionaries.

  Useful for post-processing of results; e.g., in colab.

  Args:
    data: A dictionary where each value is a list.

  Returns:
    A list of dictionaries.
  """
  return [dict(zip(data.keys(), values)) for values in zip(*data.values())]
