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

"""Agent for human playing."""

import sys

from android_world.agents import base_agent
from android_world.env import interface
from android_world.env import json_action


class HumanAgent(base_agent.EnvironmentInteractingAgent):
  """Human agent; wait for user to indicate they are done."""

  def __init__(self, env: interface.AsyncEnv, name: str = '', output_path: str = ''):
    """Initializes the HumanAgent.
    
    Args:
      env: The environment.
      name: The agent name.
      output_path: Path for output data.
    """
    super().__init__(env, name)
    self.output_path = output_path

  def step(self, goal: str, step_numb: int = 0) -> base_agent.AgentInteractionResult:
    """Executes one step of the agent.
    
    Args:
      goal: The goal/instruction for this step.
      step_numb: The step number (optional).
      
    Returns:
      AgentInteractionResult with the state after action execution.
    """
    print(f"goal: {goal}")
    del goal, step_numb
    is_done = False
    try:
      response = input(
          'Human playing! Hit enter when you are ready for evaluation (or q to'
          ' quit).'
      )
      # If user explicitly hits enter or types something, they're ready
      is_done = True
    except EOFError:
      # Handle non-interactive input (e.g., when running in batch mode)
      # In batch mode, don't mark as done - let the episode continue
      response = ''
      is_done = False
    
    if response == 'q':
      sys.exit()
    action_details = {'action_type': 'answer', 'text': response}
    self.env.execute_action(json_action.JSONAction(**action_details))

    state = self.get_post_transition_state()
    step_data = {
        'raw_screenshot': state.pixels,
        'ui_elements': state.ui_elements,
    }
    return base_agent.AgentInteractionResult(is_done, step_data)

  def get_post_transition_state(self) -> interface.State:
    return self.env.get_state()
