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

"""Tasks for general system tasks like interacting with settings."""

import dataclasses
import os
import random
from typing import Any

import yaml
from absl import logging
from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.utils import fuzzy_match_lib
from android_world.task_evals.single import contacts
from android_world.task_evals.single import expense
import datetime

import immutabledict

from android_world.policy.verification import VerifyPolicy

from pathlib import Path
import sys
# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from config import config

# Load local paths from config
_LOCAL_GUI_BROWSING_VIDEO = config.get('local_paths.gui_browsing_video')
_LOCAL_GUI_BROWSING_P1 = config.get('local_paths.gui_browsing_p1')
_LOCAL_GUI_BROWSING_P2 = config.get('local_paths.gui_browsing_p2')
_LOCAL_GUI_BROWSING_P3 = config.get('local_paths.gui_browsing_p3')
_LOCAL_GUI_BROWSING_P4 = config.get('local_paths.gui_browsing_p4')
_LOCAL_GUI_BROWSING_FIGURE1 = config.get('local_paths.gui_browsing_figure1')
_LOCAL_GUI_BROWSING_FIGURE2 = config.get('local_paths.gui_browsing_figure2')
_LOCAL_GUI_BROWSING_BESTBUY = config.get('local_paths.gui_browsing_bestbuy')

# Load remote paths from config
_REMOTE_GUI_BROWSING = config.get('remote_paths.gui_browsing')
_REMOTE_TEST = config.get('remote_paths.test')




class _SystemBrightnessToggle(task_eval.TaskEval):
  """Task for checking that the screen brightness has been set to {max_or_min}."""

  app_names = ('settings',)
  complexity = 1
  schema = {
      'type': 'object',
      'properties': {'max_or_min': {'type': 'string', 'enum': ['max', 'min']}},
      'required': ['max_or_min'],
  }
  template = 'Turn brightness to the {max_or_min} value.'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    res = adb_utils.issue_generic_request(
        ['shell', 'settings', 'get', 'system', 'screen_brightness'],
        env.controller,
    )
    brightness_level = int(res.generic.output.decode().strip())

    if self.params['max_or_min'] == 'max':
      return 1.0 if brightness_level == 255 else 0.0
    else:
      return 1.0 if brightness_level == 1 else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {'max_or_min': 'max' if random.choice([True, False]) else 'min'}


class SystemBrightnessMinVerify(_SystemBrightnessToggle):
  """Task for verifying that the screen brightness is already at minimum.

  Precondition: Screen brightness is at minimum.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.set_brightness('min', env.controller)

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'max_or_min': 'min'}


class SystemBrightnessMaxVerify(_SystemBrightnessToggle):
  """Task for verifying that the screen brightness is already at maximum.

  Precondition: Screen brightness is at maximum.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.set_brightness('max', env.controller)

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'max_or_min': 'max'}





class SystemBrightnessMin(_SystemBrightnessToggle):
  """Task for ensuring that the screen brightness is set to minimum.

  Precondition: Screen brightness is not at minimum.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.set_brightness('max', env.controller)

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'max_or_min': 'min'}


class SystemBrightnessMax(_SystemBrightnessToggle):
  """Task for ensuring that the screen brightness is set to maximum.

  Precondition: Screen brightness is not at maximum.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.set_brightness('min', env.controller)

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'max_or_min': 'max'}


class _SystemWifiToggle(task_eval.TaskEval):
  """Task for checking that WiFi has been turned {on_or_off}."""

  app_names = ('settings',)
  complexity = 1
  schema = {
      'type': 'object',
      'properties': {'on_or_off': {'type': 'string', 'enum': ['on', 'off']}},
      'required': ['on_or_off'],
  }
  template = 'Turn wifi {on_or_off}.'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    res = adb_utils.issue_generic_request(
        ['shell', 'settings', 'get', 'global', 'wifi_on'], env.controller
    )
    wifi_status = res.generic.output.decode().strip()

    if self.params['on_or_off'] == 'on':
      # WiFi is on when the value is either 1 or 2. If Airplane mode is on, and
      # WiFi is on, it will be "2".
      return 1.0 if wifi_status in ['1', '2'] else 0.0
    else:
      # WiFi is off when the value is 0.
      return 1.0 if wifi_status == '0' else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'on' if random.choice([True, False]) else 'off'}


class SystemWifiTurnOffVerify(_SystemWifiToggle):
  """Task for verifying that WiFi is already turned off.

  Precondition: WiFi is off.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_wifi(env.controller, 'off')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'off'}


class SystemWifiTurnOnVerify(_SystemWifiToggle):
  """Task for verifying that WiFi is already turned on.

  Precondition: WiFi is on.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_wifi(env.controller, 'on')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'on'}


class SystemWifiTurnOff(_SystemWifiToggle):
  """Task for ensuring that WiFi is turned off.

  Precondition: WiFi is on.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_wifi(env.controller, 'on')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'off'}


class SystemWifiTurnOn(_SystemWifiToggle):
  """Task for ensuring that WiFi is turned on.

  Precondition: WiFi is off.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_wifi(env.controller, 'off')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'on'}


class _SystemBluetoothToggle(task_eval.TaskEval):
  """Task for checking that Bluetooth has been turned {on_or_off}."""

  app_names = ('settings',)
  complexity = 1
  schema = {
      'type': 'object',
      'properties': {'on_or_off': {'type': 'string', 'enum': ['on', 'off']}},
      'required': ['on_or_off'],
  }
  template = 'Turn bluetooth {on_or_off}.'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    res = adb_utils.issue_generic_request(
        ['shell', 'settings', 'get', 'global', 'bluetooth_on'], env.controller
    )
    bluetooth_status = res.generic.output.decode().strip()
    expected_status = '1' if self.params['on_or_off'] == 'on' else '0'
    return 1.0 if bluetooth_status == expected_status else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {'on_or_off': 'on' if random.choice([True, False]) else 'off'}


class SystemBluetoothTurnOffVerify(_SystemBluetoothToggle):
  """Task for verifying that Bluetooth is already turned off.

  Precondition: Bluetooth is off.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_bluetooth(env.controller, 'off')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'off'}


class SystemBluetoothTurnOnVerify(_SystemBluetoothToggle):
  """Task for verifying that Bluetooth is already turned on.

  Precondition: Bluetooth is on.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_bluetooth(env.controller, 'on')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'on'}


class SystemBluetoothTurnOff(_SystemBluetoothToggle):
  """Task for ensuring that Bluetooth is turned off.

  Precondition: Bluetooth is on.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_bluetooth(env.controller, 'on')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'off'}


class SystemBluetoothTurnOn(_SystemBluetoothToggle):
  """Task for ensuring that Bluetooth is turned on.

  Precondition: Bluetooth is off.
  """

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_bluetooth(env.controller, 'off')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'on'}



class SystemCopyToClipboard(task_eval.TaskEval):
  """Task for verifying that the correct params are copied to the clipboard."""

  app_names = ('clipper',)
  complexity = 1
  schema = {
      'type': 'object',
      'properties': {
          'clipboard_content': {'type': 'string'},
      },
      'required': ['clipboard_content'],
  }

  template = 'Copy the following text to the clipboard: {clipboard_content}'

  def __init__(self, params: dict[str, Any]):
    """Initialize the task with given params."""
    super().__init__(params)
    self.clipboard_content = params['clipboard_content']

  def _clear_clipboard(self, env: interface.AsyncEnv) -> None:
    # Use a unique string to set the clipboard contents.
    adb_utils.set_clipboard_contents('~~~RESET~~~', env.controller)

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self._clear_clipboard(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    """Check if the clipboard content matches the expected content."""
    actual_clipboard_content = adb_utils.get_clipboard_contents(env.controller)
    return (
        1.0
        if fuzzy_match_lib.fuzzy_match(
            self.clipboard_content, actual_clipboard_content
        )
        else 0.0
    )

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self._clear_clipboard(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {
        'clipboard_content': random.choice([
            '1234 Elm St, Springfield, IL',
            'Acme Corp, Suite 200',
            'john.doe@example.com',
            "Jane's Flower Shop",
            'Call me at 555-1234',
            'Order No: A123456',
            'Reservation under: Jane',
            'Discount code: SAVE20',
            'Membership ID: XYZ789',
            'Invoice #98765',
            'Tracking #: 1Z204E2A',
            'Transaction ID: abc123',
            '9876 Pine Ave, Riverside, CA',
            'Global Tech, Floor 3',
            'jane.smith@example.com',
            "Mike's Grocery Store",
            'Text me at 555-6789',
            'Order No: B654321',
            'Reservation under: Mike',
            'Promo code: DEAL30',
            'Membership ID: ABC123',
            'Invoice #54321',
            'Tracking #: 3H488Y2B',
            'Transaction ID: def456',
            '2554 Oak Street, Boston, MA',
            'Innovate Inc, Room 10',
            'alex.jordan@example.net',
            "Sara's Bakery",
            'Reach out at 555-9101',
            'Order No: C987654',
            'Reservation under: Sara',
            'Coupon code: OFF50',
            'Membership ID: LMN456',
            'Invoice #32198',
            'Tracking #: 5K672F4C',
            'Transaction ID: ghi789',
        ])
    }


@dataclasses.dataclass(frozen=True)
class _ComponentName:
  """Android identifier for an application component.

  Identifier for an application component - i.e., an Activity, a Service, a
  BroadcastReceiver, or a Content Provider. Encapsulates two pieces of
  information used to identify the component - the package name of the app it
  exists in, and the class name of the object within that app.
  """

  package_name: str
  class_name: str


def _normalize_class_name(package_name: str, class_name: str) -> str:
  """Normalizes a fully qualified class name to be relative to the package.

  Class names are strings, which can be fully qualified or relative to the
  app's package. This function normalizes a fully qualified class name to be
  relative, to make it easy to test two class names for equality.

      normalized_class_name = _normalize_class_name(
          'com.android.settings',
          'com.android.settings.Settings'
      )
      assert normalized_class_name == '.Settings'

  Args:
    package_name: The package name of the app.
    class_name: The name of the class.

  Returns:
    The class name, normalized to be relative if fully qualified.
  """
  if class_name.startswith(package_name):
    return class_name[len(package_name) :]
  return class_name


def parse_component_name(component_name: str) -> _ComponentName:
  """Parses a ComponentName from a string.

  Args:
    component_name: The string representation of the component name, e.g.
      'com.android.settings/com.android.settings.Settings'.

  Returns:
    The parsed ComponentName.
  Raises:
    ValueError: If called with an invalid string representation of a
      ComponentName.
  """
  parts = component_name.split('/')
  if len(parts) != 2:
    raise ValueError(
        'Badly formed component name: the package and class names must be '
        'separated by a single slash'
    )
  return _ComponentName(
      package_name=parts[0],
      class_name=_normalize_class_name(
          package_name=parts[0], class_name=parts[1]
      ),
  )


_APP_NAME_TO_PACKAGE_NAME = immutabledict.immutabledict({
    'camera': 'com.android.camera2',
    'clock': 'com.google.android.deskclock',
    'contacts': 'com.google.android.contacts',
    'settings': 'com.android.settings',
    'dialer': 'com.google.android.dialer',
})


class OpenAppTaskEval(task_eval.TaskEval):
  """Task eval for opening an app."""

  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())

  complexity = 1

  schema = {
      'type': 'object',
      'properties': {
          'app_name': {'type': 'string'},
      },
      'required': ['app_name'],
  }

  template = (
      'Open the {app_name} app. Clear any pop-ups that may appear by granting'
      ' all permissions that are required.'
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    app_name = random.choice(list(_APP_NAME_TO_PACKAGE_NAME.keys()))
    return {'app_name': app_name}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    active_activity, _ = adb_utils.get_current_activity(env.controller)
    expected_package_name = _APP_NAME_TO_PACKAGE_NAME[self.params['app_name']]
    if (
        parse_component_name(active_activity).package_name
        == expected_package_name
    ):
      return 1.0
    else:
      logging.info(
          'Expected %s to be active app but saw %s',
          expected_package_name,
          active_activity,
      )
      return 0.0



class FuncAssistJoblinExplainOneFunctionality1(task_eval.TaskEval):
  """Task eval for opening an app."""

  app_names = ('joplin',)

  complexity = 4

  schema = {
      'type': 'object',
      'properties': {
          'app_name': {'type': 'string'},
      },
      'required': ['app_name'],
  }


  template = 'In the Joplin app, when a user opens a note, there is a settings button in the top-right corner. ' \
  'What functions are available after clicking it? Please describe them.'
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    app_name = random.choice(list(_APP_NAME_TO_PACKAGE_NAME.keys()))
    return {'app_name': app_name}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not env.interaction_cache:
      return 0.0

    goal = self.template
    prediction = env.interaction_cache
    gt_answer = (
        "Users can choose to add attachments (files or photos), draw pictures, "
        "share the note, use voice typing, add tags, convert it into a to-do, "
        "copy the Markdown link, view properties, and delete the note."
    )

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
          'Error during verification for FuncAssistJoblinExplainOneFunctionality1: %s', e
      )
      return 0.0

class FuncAssistLocateSystemInterfaceTaskEval2(task_eval.TaskEval):
  """Task eval for locating the System Interface with Bluetooth, Airplane mode, and Brightness controls."""

  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())

  complexity = 2

  schema = {}

  template = 'Find a phone screen that lets me toggle Bluetooth and Airplane mode, as well as adjust the brightness, all in one place. Navigate to that screen and show it to me.'
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    """Check if the task is successful by verifying the screenshot prediction.
    
    Args:
      env: The environment interface.
      
    Returns:
      1.0 if successful (screenshot shows the correct interface), 0.0 otherwise.
    """
    super().is_successful(env)

    
    goal = self.template
    task_key = 'FuncAssistLocateSystemInterfaceTaskEval2'
    
    try:
      ply = VerifyPolicy({})
      result = ply.verify_locating_interface(
          goal=goal,
          # last_screenshot_path=self.last_screenshot_path,
          task_key=task_key,
          last_screenshot=self.last_screenshot,
      )
      return 1.0 if result else 0.0
    except Exception as e:
      logging.warning(
          'Error during verification for FuncAssistLocateSystemInterfaceTaskEval2: %s', e
      )
      return 0.0



class RefusalSystemBluetoothTurnOnAlreadyOn(_SystemBluetoothToggle):
  """Task for ensuring that Bluetooth is turned on.

  Precondition: Bluetooth is on.
  """
  answer = 'The Bluetooth is already turned on so I did not toggle the button.'

  complexity = 2

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_bluetooth(env.controller, 'on')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'on'}


  def is_successful(self, env: interface.AsyncEnv) -> float:
    parent_score = super().is_successful(env)
    if parent_score != 1.0:
        return 0.0 
    
    if not env.interaction_cache:
      return 0.0
    ply = VerifyPolicy({})
    
    result = ply.verify_textual_prediction(
            goal=self.goal,
            prediction=env.interaction_cache,
            gt_answer= self.answer,
            judgement_type = 'refusal'
        )
    return result
  

class RefusalSystemBluetoothTurnOnAlreadyOnCHS(_SystemBluetoothToggle):
  """Task for ensuring that Bluetooth is turned on.

  Precondition: Bluetooth is on.
  """

  template = '打开或关闭蓝牙：{on_or_off}。'

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_bluetooth(env.controller, 'on')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'on'}
  

class RefusalSystemBluetoothTurnOnAlreadyOnVariation(_SystemBluetoothToggle):
  """Task for ensuring that Bluetooth is turned on.

  Precondition: Bluetooth is on.
  """

  template = 'Switch bluetooth {on_or_off}.'

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_bluetooth(env.controller, 'on')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'on'}
  


class RefusalSystemBluetoothTurnOffAlreadyOff(_SystemBluetoothToggle):
  """Task for ensuring that Bluetooth is turned off.

  Precondition: Bluetooth is off.
  """
  

  answer = 'The Bluetooth is already turned off so I did not toggle the button.'

  complexity = 2

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_bluetooth(env.controller, 'off')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'off'}
  
  def is_successful(self, env: interface.AsyncEnv) -> float:
    parent_score = super().is_successful(env)
    if parent_score != 1.0:
        return 0.0 
    
    if not env.interaction_cache:
      return 0.0
    ply = VerifyPolicy({})
    
    result = ply.verify_textual_prediction(
            goal=self.goal,
            prediction=env.interaction_cache,
            gt_answer= self.answer,
           judgement_type = 'refusal'
        )
    return result



class VagueBluetoothTurnOn2(_SystemBluetoothToggle):
  """Task for ensuring that Bluetooth is turned on.

  Precondition: Bluetooth is off.
  """
  template = (
    'My wireless earbuds won’t hook up to my phone—everything '
    'just blares out loud. '
    'Why is that, and can you fix it for me?'
              )
  
  complexity = 2
  
  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_bluetooth(env.controller, 'off')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'on'}

  

class BrowsecompSystemBluetoothTurnOffUIComp(_SystemBluetoothToggle):
  """Task for ensuring that Bluetooth is turned off.

  Precondition: Bluetooth is off.
  """
  template = 'Go to a screen that has 8 buttons (arranged in 4 rows and 2 columns), one slider, and the time displayed in the top-left corner. ' \
  'Hint: two of the buttons have an alarm clock icon and an airplane icon. On that screen, tap the second button in the first row.'
  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_bluetooth(env.controller, 'off')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'on'}


class VagueBluetoothTurnOn1(_SystemBluetoothToggle):
  """Task for ensuring that Bluetooth is turned off.

  Precondition: Bluetooth is off.
  """
  template = 'Make the phone discoverable and able to pair with ' \
  'nearby headphones, smart bands, and car systems.'

  complexity = 2

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.toggle_bluetooth(env.controller, 'off')

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'on'}


class VagueSystemBrightnessMax(_SystemBrightnessToggle):
  """Task for ensuring that the screen brightness is not minimum.

  Precondition: Screen brightness is at minimum.
  """
  template = 'The screen is too dark. I can see nothing.'

  complexity = 2

  brightness_threshold = 50
  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.set_brightness('min', env.controller)

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'max_or_min': 'max'}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    res = adb_utils.issue_generic_request(
        ['shell', 'settings', 'get', 'system', 'screen_brightness'],
        env.controller,
    )
    brightness_level = int(res.generic.output.decode().strip())

    if brightness_level < self.brightness_threshold:
      return 0
    else:
      return 1


class VagueSystemBrightnessMaxCHS(VagueSystemBrightnessMax):
  template = '屏幕太暗了，我什么都看不见。'


class VagueSystemBrightnessMaxVariation(VagueSystemBrightnessMax):
  template = 'It’s too dark on the screen—nothing is visible.'



class VagueFindPhoneTaskEvalUI(task_eval.TaskEval):
  """Task eval for identifying a phone model from a screenshot."""

  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())

  complexity = 5

  schema = {}

  template = (
      'I previously saw a smartphone on a website and took a screenshot, '
      'but I have forgotten which model it was. '
      'Please tell me the phone model.'
  )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not env.interaction_cache:
      return 0.0

    goal = self.template
    prediction = env.interaction_cache
    gt_answer = (
       "The phone model is iPhone 17 Pro."
    )

    ply = VerifyPolicy({})
    result = ply.verify_textual_prediction(
        goal=goal,
        prediction=prediction,
        gt_answer=gt_answer,
        judgement_type='guibrowsing'
      )
    return 1.0 if result else 0.0

  
  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env) 
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_BESTBUY, remote_path=_REMOTE_TEST, env=env.controller)
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    app_name = random.choice(list(_APP_NAME_TO_PACKAGE_NAME.keys()))
    return {'app_name': app_name}



class VagueZenModeTaskEvalUI(task_eval.TaskEval):
  """Task eval with a vague instruction to set up zen_mod."""

  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())

  complexity = 2

  schema = {}

  template = (
      'I\'m going to take a nap now. '
      'I really need some peace and '
      'quiet without my phone buzzing or ringing.'
  )
  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    adb_utils.issue_generic_request(
        ['shell', 'cmd', 'notification', 'set_dnd', 'off'],
        env.controller
    )
    
  def _get_zen_mode(self, env: interface.AsyncEnv) -> str | None:
    try:
      res = adb_utils.issue_generic_request(
          ['shell', 'settings', 'get', 'global', 'zen_mode'],
          env.controller
      )
      if res:
        return res.generic.output.decode().strip()
      return None
    except Exception as e:
      print(f"Error getting zen_mode: {e}")
      return None

  def is_successful(self, env: interface.AsyncEnv) -> float:
    current_zen_mode = self._get_zen_mode(env)
    print(f"Verifying: Current zen_mode value is: {current_zen_mode}")
    if current_zen_mode is not None and current_zen_mode != '0':
      return 1.0
    else:
      return 0.0
  
  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    print("Initializing: Forcing zen_mode (DND) to 0 (OFF) via ADB...")

    adb_utils.issue_generic_request(
      ['shell', 'cmd', 'notification', 'set_dnd', 'off'],
      env.controller
    )
    initial_zen_mode = self._get_zen_mode(env)
    print(f"Initialized zen_mode: {initial_zen_mode}")
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {"app_name": "Settings"}
    

class GUIBrowsingPaper1(task_eval.TaskEval):
  """
  GUI Browsing,
  Hard
  """
  template = 'Read the "Second Half" blog located at the folder GUIBrowsing/p1 within sdk_gphone_x86_64 storage area' \
  'and determine which of the other three papers in the same directory is mentioned in that blog. ' \
  'Tell me the name of the file including the file extension.'


  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())

  complexity = 10

  schema = {}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)    
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = '2406.12045.pdf'
    
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
          'Error during verification for GUIBrowsingPaper1: %s', e
      )
      return 0.0
  

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_P1, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)


  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}

class GUIBrowsingPaper1CHS(GUIBrowsingPaper1):
  template = '请阅读存储空间sdk_gphone_x86_64中 GUIBrowsing/p1 目录下的博客《Second Half》，同一目录下还有三篇论文，找出该博客里提到了其中哪一篇，告诉我对应文件的带格式后缀的完整文件名。'
class GUIBrowsingPaper1Variation(GUIBrowsingPaper1):
  template = 'Read the blog titled "Second Half" located at the folder GUIBrowsing/p1 within sdk_gphone_x86_64 storage area. Identify which of the three other papers in the same directory is mentioned in that blog, and report the exact filename including the file extension.'



# p1
class GUIBrowsingPaper2(task_eval.TaskEval):
  """
  GUI Browsing, Medium-Hard
  Among three paper PDFs stored in the sdk_gphone_x86_64 storage area
  of the Android filesystem, only one contains exclusively line charts
  (and no bar charts).
  """

  template = (
      'There are three research-paper PDFs located at the folder GUIBrowsing/p1 within sdk_gphone_x86_64 storage area. '
      'Exactly one of these three documents contains only line charts and no bar charts. '
      'Please open each file, inspect the figures, '
      'and report only the exact filename of the document including the file extension that satisfies this condition.'
  )

  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())
  complexity = 10
  schema = {}

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_P1, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)


  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)   
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = '2305.16291.pdf'
    
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
          'Error during verification for GUIBrowsingPaper2: %s', e
      )
      return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}

# p1
class GUIBrowsingPaper3(task_eval.TaskEval):
  """
  GUI Browsing, Medium-Hard
  Among three research-paper PDFs and one blog PDF stored in the
  sdk_gphone_x86_64 storage area, exactly one document discusses
  “ReAct” and “function calling” in its experiments.
  """

  template = (
      'There are three research-paper PDFs and one blog PDF located at the folder GUIBrowsing/p1 within sdk_gphone_x86_64 storage area. '
      'Exactly one of these four documents contains '
      'experiments that discuss both “ReAct” and “function calling”. '
      'Please Tell me the name of the file including the file extension.'
  )

  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())
  complexity = 10
  schema = {}

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_P1, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)
   
  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)   
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = '2406.12045.pdf'
    
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
          'Error during verification for GUIBrowsingPaper3: %s', e
      )
      return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}


class GUIBrowsingAnswer1(task_eval.TaskEval):

  template = (
     'There are four PDF files located at the folder GUIBrowsing/p2 within sdk_gphone_x86_64 storage area: student1.pdf, student2.pdf (student answers) and solution1.pdf, solution2.pdf (correct answers). '
     'Open each student file and compare it with its '
     'corresponding solution file to determine which '
     'numbered questions or exercises the student did not complete at all. '
     'Report your findings in the following format: '
     '<student_file including the file extension>: missing Exercise <the number of the exercise>. '
  )
  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())
  complexity = 9
  schema = {}

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)  
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_P2, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)
  

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)   
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = 'student1.pdf: missing Exercise 0.3.3'
    
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
          'Error during verification for GUIBrowsingAnswer1: %s', e
      )
      return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}


class GUIBrowsingAnswer2(task_eval.TaskEval):
  """
  GUI Browsing, Hard
  Four PDF files reside in the sdk_gphone_x86_64 storage area:
  answer1.pdf and answer2.pdf are student solutions,
  solution1.pdf and solution2.pdf are the corresponding ground-truth answers.
  Identify every place where the student answers deviate from the correct ones.
  """

  template = (
      'There are four PDF files located at the folder GUIBrowsing/p2 within sdk_gphone_x86_64 storage area: student1.pdf, student2.pdf (student answers) and solution1.pdf, solution2.pdf (correct answers).'
      'Open each student file and compare it with its corresponding solution file to determine which questions or exercises the student did incorrectly. Report your findings in the following format: <student_file including the file extension>: wrong Exercise <the number of the exercise>. '
  )

  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())
  complexity = 9
  schema = {}

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)  
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_P2, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)
   

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)   
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = 'student2.pdf: wrong Exercise 1.3.1'
    
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
          'Error during verification for GUIBrowsingAnswer2: %s', e
      )
      return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}
  

class GUIBrowsingPDF1(task_eval.TaskEval):
  """
  GUI Browsing, Medium-Hard
  Among four PDF documents (LearnAct, AndroidWorld, MVISU, UI-NEXUS) stored in
  the sdk_gphone_x86_64 storage area, determine which one contains the largest
  number of figures in its main body.
  """

  template = (
      'There are four PDF files located at the folder GUIBrowsing/p3 within sdk_gphone_x86_64 storage area. '
      'Open each document and count only the figures that appear in the main body (exclude tables and appendices). '
      'Report the exact filename including the file extension that contains the highest number of such figures.'
  )



  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())
  complexity = 9
  schema = {}

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_P3, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)
    
  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)   
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = 'LearnAct.pdf'
    
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
          'Error during verification for GUIBrowsingPDF1: %s', e
      )
      return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}



class GUIBrowsingPDF2(task_eval.TaskEval):
  """
  GUI Browsing, Hard
  Among four PDF documents (LearnAct, AndroidWorld, MVISU, UI-NEXUS) stored in
  the sdk_gphone_x86_64 storage area, identify which one states that its benchmark
  contains the largest number of **Chinese tasks**.
  """

  template = (
      'There are four PDF files located at the folder GUIBrowsing/p3 within sdk_gphone_x86_64 storage area. '
      'Open each file and locate any mention of benchmark composition. '
      'Determine which document explicitly reports the highest count of Chinese-language tasks within its benchmark suite.'
      ' Return the exact filename including the file extension of that document.'
  )

  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())
  complexity = 9
  schema = {}

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env) 
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_P3, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)
   

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)   
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = 'MVISU.pdf'
    
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
          'Error during verification for GUIBrowsingPDF2: %s', e
      )
      return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}


class GUIBrowsingFindImageInPaper1(task_eval.TaskEval):
  template = (
      'There are four PDF files located at the folder GUIBrowsing/p4 within sdk_gphone_x86_64 storage area. '
      'There is an image that contains the process of ordering food delivery with a smartphone. '
      'Tell me the file name including extension and the page number where the target image is located. '
      'Use the format: "<Filename>: Page <Number>".'
  )
  
  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())
  complexity = 10
  schema = {}

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_P4, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)   
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = '2410.15164.pdf: Page 4'
    
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
          'Error during verification for GUIBrowsingFindImageInPaper1: %s', e
      )
      return 0.0  
    
  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}


class GUIBrowsingFindImageInPaper2(task_eval.TaskEval):
  template = (
      'There are four PDF files located at the folder GUIBrowsing/p4 within sdk_gphone_x86_64 storage area. '
      'There is an image that contains one red robot and one blue robot.'
      'Tell me the file name including extension and the page number where the target image is located. '
      'Use the format: "<Filename>: Page <Number>".'
  )

  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())
  complexity = 10
  schema = {}

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_P4, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)   
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = '2510.08558.pdf: Page 5'
    
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
          'Error during verification for GUIBrowsingFindImageInPaper2: %s', e
      )
      return 0.0  
    
  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}
    
  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}


class GUIBrowsingOrder1(task_eval.TaskEval):


  template = (
      'There are some screenshot located at the folder GUIBrowsing/figure1 within sdk_gphone_x86_64 storage area. In the current screenshot of the Meituan food-delivery app (open it in Gallery), help me find the cheapest Americano that can be delivered within 30 minutes. Return the answer in Chinese in the following format: <merchant full name>, <the item full name>.'
  )
  answer = '肯悦咖啡(双榆树店), 浓萃美式(冰)'

  app_names = (_APP_NAME_TO_PACKAGE_NAME.keys())  
  complexity = 6
  schema = {}

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_FIGURE1, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not env.interaction_cache:
      return 0.0
    ply = VerifyPolicy({})
    
    result = ply.verify_textual_prediction(
            goal=self.template,
            prediction=env.interaction_cache,
            gt_answer= self.answer,
            judgement_type = 'guibrowsing'
        )
    return result

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}
  


class GUIBrowsingOrder2(task_eval.TaskEval):
  """
  GUI Browsing, Hard
  跨 App 比价：美团、淘宝、京东外卖
  """

  template = (
      'There are some screenshot located at the folder GUIBrowsing/figure1 '
      'within sdk_gphone_x86_64 storage area. '
      'In the current screenshots of the Meituan, JD, and Taobao '
      'food-delivery apps (open them in Gallery), '
      'help me find the cheapest Americano. '
      'Return the answer in Chinese in the following format: '
      '<the platform name>, <the merchant full name>, <the item full name>.'
  )
  answer = '淘宝, 玛莲朵X瑄品咖啡, 椰青美式'

  app_names = (_APP_NAME_TO_PACKAGE_NAME.keys())  
  complexity = 8
  schema = {}

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_FIGURE1, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not env.interaction_cache:
      return 0.0
    ply = VerifyPolicy({})
    
    result = ply.verify_textual_prediction(
            goal=self.template,
            prediction=env.interaction_cache,
            gt_answer= self.answer,
            judgement_type = 'guibrowsing'
        )
    return result

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}
  

class GUIBrowsingOrder2CHS(GUIBrowsingOrder2):
  template = '在我手机存储空间sdk_gphone_x86_64的 GUIBrowsing/figure1 目录下，有一些截图。请用Gallery打开美团、京东、淘宝三家外卖 App 的当前截图，帮我找出最便宜的美式咖啡。请用中文按格式返回：平台名称,商家完整名称,商品完整名称。'

class GUIBrowsingOrder2Variation(GUIBrowsingOrder2):
  template = 'Some screenshots are located at GUIBrowsing/figure1 in the Android file system. Open the current screenshots of Meituan, JD, and Taobao food-delivery apps in Gallery and locate the cheapest Americano. Return the answer in Chinese in the following format: <the platform name>, <the merchant full name>, <the item full name>.'


class GUIBrowsingOrder3(task_eval.TaskEval):
  """
  GUI Browsing, Hard
  在美团、京东、淘宝外卖 App 中搜索瑞幸美式咖啡，
  按配送时间最短优先，相同时间则选距离最近。
  """

  template = (
      'There are some screenshot located at the folder GUIBrowsing/figure1 within sdk_gphone_x86_64 storage area. '
      'I want a Luckin Coffee. In the current screenshots of the Meituan, JD, and Taobao food-delivery apps (open them in Gallery), '
      'find the option with the shortest delivery time. '
      'If multiple options have the same delivery time, '
      'choose the one with the shortest delivery distance. '
      'Return the answer in Chinese in the following format: '
      '<the platform name>, <the merchant full name>.'
  )
  answer = '京东, 瑞幸咖啡(理工科技大厦店)'

  app_names = (_APP_NAME_TO_PACKAGE_NAME.keys())  
  complexity = 9
  schema = {}

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_FIGURE1, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    
    if not env.interaction_cache:
      return 0.0
    ply = VerifyPolicy({})
    
    result = ply.verify_textual_prediction(
            goal=self.template,
            prediction=env.interaction_cache,
            gt_answer= self.answer,
            judgement_type = 'guibrowsing'
        )
    return result


  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}
  

class GUIBrowsingOnlineShopping1(task_eval.TaskEval):
  """
  GUI Browsing, Medium
  Locate the cheapest **wireless Bluetooth Beats headphones** visible in the
  current web-page screenshot and return its model name and price.
  """

  template = (
      'There are some screenshot located at the folder GUIBrowsing/figure2 within sdk_gphone_x86_64 storage area. '
      'Examine the current web-page screenshot and find all listed products. '
      'Select wireless Bluetooth Beats headphones with the lowest price and return the name of the figure including the extension that contains that product.'
  )
  answer = '2.png'

  app_names = (_APP_NAME_TO_PACKAGE_NAME.keys())  
  complexity = 8
  schema = {}

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_FIGURE2, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)   
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = '2.png'
    
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
          'Error during verification for GUIBrowsingOnlineShopping1: %s', e
      )
      return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}
  

class GUIBrowsingOnlineShopping2(task_eval.TaskEval):


  template = (
      'There are some screenshot located at the folder GUIBrowsing/figure1 within sdk_gphone_x86_64 storage area. '
      'Look at the current web-page screenshot and identify all listed products. '
      'Choose the black-colored earphones with the lowest price. Return return the name of the figure including the extension that contains that product.'
  )

  app_names = (_APP_NAME_TO_PACKAGE_NAME.keys())  
  complexity = 8
  schema = {}

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_FIGURE2, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)   
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = '7.png'
    
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
          'Error during verification for GUIBrowsingOnlineShopping2: %s', e
      )
      return 0.0  

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {}


class BrowsecompOpenAppTaskEvalUICompCreate1(task_eval.TaskEval):
  """Task eval for opening an app and creating a project."""

  app_names = ('files', 'joplin', 'tasks', 'recipe', 'expense', 'markor', 'calendar',)

  complexity = 8

  schema = {
      'type': 'object',
      'properties': {
          'app_name': {'type': 'string'},
      },
      'required': ['app_name'],
  }

  template = (
      'Among Files, Joplin, Tasks, Broccoli, Pro Expense, Markor and Calender, '
      'open an app whose icon contains blue color. '
      'After launching, the UI should not have any search bar or search button at the top. '
      'Find the APP and tell me its name. '
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    app_name = random.choice(list(_APP_NAME_TO_PACKAGE_NAME.keys()))
    return {'app_name': app_name}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)    
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = 'Pro Expense'
    
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
          'Error during verification for BrowsecompOpenAppTaskEvalUICompCreate1: %s', e
      )
      return 0.0


class BrowsecompOpenAppTaskEvalUICompCreate2(task_eval.TaskEval):
  """Task eval for opening an app and creating a project."""

  app_names = (
    "audio recorder",
    "markor", 
    "recipe", 
    "medilog", 
    "expense", 
    "vlc", 
    "joplin"
  )

  complexity = 9

  schema = {
      'type': 'object',
      'properties': {
          'app_name': {'type': 'string'},
      },
      'required': ['app_name'],
  }

  template = (
      'Among Audio Recorder, Markor, Broccoli, MediLog, Pro Expense, Vlc, and Joplin,'
      'open an app whose icon uses no more than two colors. '
      'After launching, the UI should display a "+" symbol '
      'at the bottom-right corner. '
      'Among such apps, select the one that '
      'requires the fewest input fields or rows after tapping "+". '
      "Find the APP and tell me its name. "
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    app_name = random.choice(list(_APP_NAME_TO_PACKAGE_NAME.keys()))
    return {'app_name': app_name}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)    
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = 'Pro Expense'
    
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
          'Error during verification for BrowsecompOpenAppTaskEvalUICompCreate2: %s', e
      )
      return 0.0


class BrowsecompFindAppTaskEvalUI1(task_eval.TaskEval):
  """Task eval for finding an app that supports the most languages."""

  app_names = ('audio recorder', 'markor', 'draw', 'recipe', 'medilog', 'vlc')

  complexity = 10

  schema = {
      'type': 'object',
      'properties': {
          'app_name': {'type': 'string'},
      },
      'required': ['app_name'],
  }

  template = (
      'Among Audio Recorder, Draw, Vlc, Pro Expense, '
      'Broccoli, MediLog and Markor,'
      'some apps support changing '
      'the interface language and the theme color.'
      'Among the apps that meet the requirements, '
      'find the one that offers more than 50 supported languages '
      'in its in-app settings. '
      'Tell me the name of the APP.'
  )


  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    app_name = random.choice(list(_APP_NAME_TO_PACKAGE_NAME.keys()))
    return {'app_name': app_name}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)    
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = 'Markor'
    
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
          'Error during verification for BrowsecompFindAppTaskEvalUI1: %s', e
      )
      return 0.0
    
class BrowsecompFindAppTaskEvalUI1CHS(BrowsecompFindAppTaskEvalUI1):
  template = '在 Audio Recorder、Draw、Vlc、Pro Expense、Broccoli、MediLog 和 Markor 之中，有些应用支持更改界面语言和主题颜色。请在满足条件的APP中找出在其应用内的设置中提供超过 50 种支持的语言的应用。告诉我该应用的名称。'

class BrowsecompFindAppTaskEvalUI1Variation(BrowsecompFindAppTaskEvalUI1):
  template = (
    'From the list of Audio Recorder, Draw, Vlc, Pro Expense, '
    'Broccoli, MediLog and Markor, '
    'certain applications allow users to modify both the UI language and the color scheme. '
    'Identify the application that provides support for over 50 languages within its settings or preferences menu. '
    'Report the name of that APP.'
  )


class BrowsecompFindAppTaskEvalUI2(task_eval.TaskEval):
  """Task eval for finding an app with the most theme color options and setting the last color."""

  app_names = ('audio recorder', 'gallery', 'photos', 'expense', 'markor', 'draw', 'recipe', 'vlc',)

  complexity = 10

  schema = {
      'type': 'object',
      'properties': {
          'app_name': {'type': 'string'},
      },
      'required': ['app_name'],
  }

  template = (
      'Among VLC, Gallery, Photos, Draw, Markor, '
      'Pro Expense, Audio Recorder and Broccoli, '
      'find an app that can play recordings or videos.'
      'Which one offers more than 6 theme color options in its in-app settings. '
      'Tell me the name of the APP.'
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    app_name = random.choice(list(_APP_NAME_TO_PACKAGE_NAME.keys()))
    return {'app_name': app_name}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)    
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = 'Audio Recorder'
    
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
          'Error during verification for BrowsecompFindAppTaskEvalUI2: %s', e
      )
      return 0.0



class BrowsecompFindAppTaskEvalUI3(task_eval.TaskEval):
  """Task eval for finding Recipe Broccoli"""
  app_names = ('joplin', 'markor', 'calendar', 'files', 'recipe', 'medilog', 'vlc')

  complexity = 8

  schema = {
      'type': 'object',
      'properties': {
          'app_name': {'type': 'string'},
      },
      'required': ['app_name'],
  }

  template = (
      'Among Joplin, Markor, Calendar, Files, Broccoli, MediLog and Vlc, '
      'there is an information-management app in which you can create individual information items. '
      'Each item can be favorited by tapping the heart-shaped button in the upper-right corner, and it can also be exported to PDF. '
      'Explore the device, identify this app, and tell me its name.'
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    app_name = random.choice(list(_APP_NAME_TO_PACKAGE_NAME.keys()))
    return {'app_name': app_name}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)    
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = 'Broccoli'
    
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
          'Error during verification for BrowsecompFindAppTaskEvalUI3: %s', e
      )
      return 0.0


class BrowsecompFindAppandAskInfo1(task_eval.TaskEval):
  """Task eval for finding markor and tell me the project team, which is Gregor Santner."""

  app_names = (
    "vlc", 
    "retro", 
    "joplin", 
    "expense", 
    "markor",
    "files"
  )

  complexity = 10

  schema = {
      'type': 'object',
      'properties': {
          'app_name': {'type': 'string'},
      },
      'required': ['app_name'],
  }
  

  template = (
  'I need to find an app among VLC, Retro Music, Joplin, Expense, Markor and Files. '
  'From its main screen, check the bottom navigation bar: it must have at least 4 buttons. '
  'Somewhere on the same main screen there is a magnifying-glass icon button. '
  'The app’s Theme settings offer more than 4 choices. '
  'Identify this app and tell me the name of its Project Team.'
  )
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    app_name = random.choice(list(_APP_NAME_TO_PACKAGE_NAME.keys()))
    return {'app_name': app_name}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)    
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = 'Gregor Santner'
    
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
          'Error during verification for BrowsecompFindAppandAskInfo1: %s', e
      )
      return 0.0


class BrowsecompFindAppandRelatedAppandAskInfo1(task_eval.TaskEval):
  """Task eval for finding joplin and markor and tell me functionality of one button in markor."""

  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())

  complexity = 10

  schema = {
      'type': 'object',
      'properties': {
          'app_name': {'type': 'string'},
      },
      'required': ['app_name'],
  }

  template = (
      'Among Joplin, Files, Broccoli, VLC, Markor, Calender and MediLog, '
      'there is an information-management app. '
      'When you create a new information item and '
      'start editing its content, a button (three horizontal dots) appears in the bottom button bar. '
      'Tapping it reveals additional editing tools. '
      'Explore the device to locate this app and tell me its name. '
  )
  
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    app_name = random.choice(list(_APP_NAME_TO_PACKAGE_NAME.keys()))
    return {'app_name': app_name}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)    
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = 'Markor'
    
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
          'Error during verification for BrowsecompFindAppandRelatedAppandAskInfo1: %s', e
      )
      return 0.0


class GUIBrowsingFindGameFromGalleryTaskEvalUI(task_eval.TaskEval):
  """Task eval for identifying a game from a photo in the Gallery."""

  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())

  complexity = 9

  schema = {
      'type': 'object',
      'properties': {
          'app_name': {'type': 'string'},
      },
      'required': ['app_name'],
  }

  template = (
      'In my photo gallery, there is a photo that contains a '
      'character from a Japanese video game. '
      'Locate that photo and tell me the name of the game.'
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env) 
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_BESTBUY, remote_path=_REMOTE_TEST, env=env.controller)


  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    app_name = random.choice(list(_APP_NAME_TO_PACKAGE_NAME.keys()))
    return {'app_name': app_name}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)   
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = 'Super Mario'
    
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
          'Error during verification for GUIBrowsingFindGameFromGalleryTaskEvalUI: %s', e
      )
      return 0.0




class GUIBrowsingFindPDF1(task_eval.TaskEval):
  """Task eval for opening an app and creating a project."""

  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())

  complexity = 6

  schema = {
      'type': 'object',
      'properties': {
          'app_name': {'type': 'string'},
      },
      'required': ['app_name'],
  }

  template = 'There are some PDFs located at the folder GUIBrowsing within sdk_gphone_x86_64 storatge area. ' \
  'I want to find a research paper whose author is Shunyu Yao. Tell me the name of the file including the file extension.'

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_P1, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_P2, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_P3, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_P4, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    app_name = random.choice(list(_APP_NAME_TO_PACKAGE_NAME.keys()))
    return {'app_name': app_name}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)   
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = 'The Second Half.pdf'
    
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
          'Error during verification for GUIBrowsingFindPDF1: %s', e
      )
      return 0.0  


class GUIBrowsingFindPDF2(task_eval.TaskEval):
  """Task eval for opening an app and creating a project."""

  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())

  complexity = 7

  schema = {
      'type': 'object',
      'properties': {
          'app_name': {'type': 'string'},
      },
      'required': ['app_name'],
  }

  template = 'There are some PDFs located at the folder GUIBrowsing within sdk_gphone_x86_64 storatge area.' \
  ' I want to find the research paper with the most authors. Tell me the name of the file including the file extension.'
  


  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_P1, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_P2, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_P3, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_P4, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    app_name = random.choice(list(_APP_NAME_TO_PACKAGE_NAME.keys()))
    return {'app_name': app_name}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)   
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = 'ANDROIDWORLD.pdf'
    
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
          'Error during verification for GUIBrowsingFindPDF2: %s', e
      )
      return 0.0  


# 
class VagueCalendarDateOffset(task_eval.TaskEval):

  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())

  complexity = 7

  schema = {
      'type': 'object',
      'properties': {
          'app_name': {'type': 'string'},
      },
      'required': ['app_name'],
  }

  
  template = (
    "Find the date and weekday after {offset_days} days starting from {reference_date}. "
    "Tell me the answer. "
    "Answer must be in the exact format: YYYY-MM-DD MMM (e.g., 2024-05-12 Sun)."

  )


  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
      """
      Randomly generate:
        - a start date (2020-01-01 ~ 2024-12-31)
        - an offset (1 ~ 365 days)
      Then compute the exact answer in required format.
      """
     
      start = datetime.date(
          year=random.randint(2023, 2024),
          month=random.randint(1, 12),
          day=random.randint(1, 28)  
      )

      offset_days = random.randint(70, 120)
      target = start + datetime.timedelta(days=offset_days)
      weekday_abbr = target.strftime("%a")  # 'Mon', 'Tue', ...
      expected = f"{target.strftime('%Y-%m-%d')} {weekday_abbr}"

      return {
          "reference_date": start.strftime("%Y-%m-%d"),
          "offset_days": offset_days,
          "expected_answer": expected,
      }


  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not env.interaction_cache:
      return 0.0 
    if env.interaction_cache.strip() == self.params['expected_answer']:
      return 1.0
    return 0.0


class BrowsecompFindImage(task_eval.TaskEval):

  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())

  complexity = 8

  schema = {
      'type': 'object',
      'properties': {
          'app_name': {'type': 'string'},
      },
      'required': ['app_name'],
  }

  template = 'There are some images located at sdk_gphone_x86_64. ' \
  'I want to find an image related to ball sports. ' \
  'Tell me the name of the ball sports.'

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env) 
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_BESTBUY, remote_path=_REMOTE_TEST, env=env.controller)
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    app_name = random.choice(list(_APP_NAME_TO_PACKAGE_NAME.keys()))
    return {'app_name': app_name}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)    
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
    prediction = env.interaction_cache
    gt_answer = 'Tennis'
    
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
          'Error during verification for BrowsecompFindImage: %s', e
      )
      return 0.0
    


class BrowsecompFindVideo(task_eval.TaskEval):
  """Task eval for opening an app and creating a project."""

  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())

  complexity = 8

  schema = {
      'type': 'object',
      'properties': {
          'app_name': {'type': 'string'},
      },
      'required': ['app_name'],
  }

  template = 'There are some videos located at ' \
  'sdk_gphone_x86_64.' \
  'I need to find a video that meets the following requirements: ' \
  '1. It contains animals. ' \
  '2. It includes at least two animals. ' \
  'Tell me the name of the video including the file extension.'


  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_VIDEO, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    app_name = random.choice(list(_APP_NAME_TO_PACKAGE_NAME.keys()))
    return {'app_name': app_name}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)    
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    
    goal = self.template
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
          'Error during verification for BrowsecompFindVideo: %s', e
      )
      return 0.0


class VagueSaveBatteryWithoutLosingInternet(task_eval.TaskEval):
    """
    Task: Turn off non-essential wireless services (Bluetooth, Location, Hotspot)
    to save battery, but without losing internet connectivity (Wi-Fi or Mobile Data).
    This task evaluates reasoning, domain knowledge, and handling complex constraints.
    """
    app_names = ("Settings",)
    complexity = 8
    schema = {}

    template = (
        "I want to save some battery, "
        "so please turn off unnecessary wireless services, "
        "but make sure I don't lose my internet connection."
    )

    def initialize_task(self, env: interface.AsyncEnv) -> None:
        """
        Initializes the environment by turning ON all relevant services:
        Wi-Fi, Mobile Data, Bluetooth, NFC, Location (High Accuracy), and WLAN Hotspot.
        This creates a complex, "messy" state for the agent to resolve.
        """
        super().initialize_task(env)
        logging.info(
            "Initializing state: Turning on Wi-Fi, Mobile Data, Bluetooth, NFC, Location, and Hotspot."
        )

        # Use 'put' to set the initial state for all services.
        adb_utils.issue_generic_request(
            ['shell', 'settings', 'put', 'global', 'wifi_on', '1'], env.controller
        )
        adb_utils.issue_generic_request(
            ['shell', 'settings', 'put', 'global', 'mobile_data', '1'], env.controller
        )
        adb_utils.issue_generic_request(
            ['shell', 'settings', 'put', 'global', 'bluetooth_on', '1'], env.controller
        )
        
        adb_utils.issue_generic_request(
            ['shell', 'settings', 'put', 'secure', 'location_mode', '3'], env.controller
        )
        logging.info("Device state initialized.")


    def is_successful(self, env: interface.AsyncEnv) -> float:
        """
        Checks if the task is successful. Success requires:
        1. Bluetooth, NFC, Location, and Hotspot are ALL OFF.
        2. At least one of Wi-Fi or Mobile Data is ON.
        """
        super().is_successful(env)

        try:
            # --- Fetch current state of all six services ---
            services_to_get = {
                'wifi': ['shell', 'settings', 'get', 'global', 'wifi_on'],
                'mobile_data': ['shell', 'settings', 'get', 'global', 'mobile_data'],
                'bluetooth': ['shell', 'settings', 'get', 'global', 'bluetooth_on'],
                # 'nfc': ['shell', 'settings', 'get', 'secure', 'nfc_on'],
                'location': ['shell', 'settings', 'get', 'secure', 'location_mode'],
                # 'hotspot': ['shell', 'settings', 'get', 'global', 'tether_on'],
            }
            
            states = {}
            for name, cmd in services_to_get.items():
                res = adb_utils.issue_generic_request(cmd, env.controller)
                states[name] = res.generic.output.decode().strip()

            print('setting states', states)
        except Exception as e:
            logging.error(f"Failed to get device status via ADB, cannot evaluate success: {e}")
            return 0.0

        # --- Evaluate the conditions for "unnecessary" services ---
        unnecessary_services_to_check = {
            'Bluetooth': states['bluetooth'],
            'Location': states['location'],
        }

        for name, status in unnecessary_services_to_check.items():
            if status != '0':
                logging.info(
                    f"Task failed: {name} was expected to be OFF, but its status is '{status}'."
                )
                return 0.0

        # --- Evaluate the condition for internet connectivity ---
        is_internet_on = (states['wifi'] not in ['0']) or (states['mobile_data'] == '1')
        if not is_internet_on:
            logging.info(
                f"Task failed: Internet connection was lost. "
                f"Wi-Fi status: {states['wifi']}, Mobile Data status: {states['mobile_data']}."
            )
            return 0.0
        if (states['wifi'] in ['1', '2']) and (states['mobile_data'] == '1'):
          logging.info("Should only keep one network service.")
          return 0.0

        return 1.0

    @classmethod
    def generate_random_params(cls) -> dict[str, Any]:
        """This task has a fixed goal and does not require random parameters."""
        return {"app_name": "Settings"}


class VagueWatchLocalVideo(task_eval.TaskEval):
  
  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())

  complexity = 5

  schema = {}

  template = (
      'I want to watch a local video about water sports. '
      'Do not exit after completing the task.'
  )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    # Get the last screenshot path from instance
    if not self.last_screenshot_path:
      print("self.last_screenshot_path not exist !")
      return 0.0
    
    goal = self.template
    task_key = 'VagueWatchLocalVideo'
    ply = VerifyPolicy({})
    result = ply.verify_vague_task(
        goal=goal,
        canvas_screenshot_path=self.last_screenshot_path,
        task_key=task_key,
    )
    return result

  
  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.push_file_to_emulator(_LOCAL_GUI_BROWSING_VIDEO, remote_path=_REMOTE_GUI_BROWSING, env=env.controller)
  
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    app_name = random.choice(list(_APP_NAME_TO_PACKAGE_NAME.keys()))
    return {'app_name': app_name}



class VagueSetAlarm(task_eval.TaskEval):
  
  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())

  complexity = 3

  schema = {}

  template = (
      'I want to take a nap and get up at 14:00.'
      'After finishing the setup, do not return to the home screen.'
  )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not self.last_screenshot_path:
      print("self.last_screenshot_path not exist !")
      return 0.0
    
    goal = self.template
    task_key = 'VagueSetAlarm'
    ply = VerifyPolicy({})
    result = ply.verify_vague_task(
        goal=goal,
        canvas_screenshot_path=self.last_screenshot_path,
        task_key=task_key,
    )
    return result

  
  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    app_name = random.choice(list(_APP_NAME_TO_PACKAGE_NAME.keys()))
    return {'app_name': app_name}


class VagueCurrencyExchange(task_eval.TaskEval):
  
  app_names = tuple(_APP_NAME_TO_PACKAGE_NAME.keys())

  complexity = 5

  schema = {}

  template = (
      'I can’t access the internet right now, and I’d like to know how many New Taiwan Dollars 1 euro can be exchanged for.'
      'Please just tell me the number directly (rounded to three decimal places).'
  )

  answer = '35.693'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not env.interaction_cache:
      return 0.0
    if self.answer in env.interaction_cache:
      return 1
    else:
      return 0

  
  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    app_name = random.choice(list(_APP_NAME_TO_PACKAGE_NAME.keys()))
    return {'app_name': app_name}
  

class FuncAssistMedilogExplainAllFunctionality(task_eval.TaskEval):
  
  app_names = ('MediLog',)

  complexity = 4

  schema = {
      'type': 'object',
      'properties': {
          'app_name': {'type': 'string'},
      },
      'required': ['app_name'],
  }

  template = 'What can I do with {app_name}? Explore this APP and tell me all its functionalities.'
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    app_name = 'Medilog'
    return {'app_name': app_name}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not env.interaction_cache:
      return 0.0

    goal = self.template
    prediction = env.interaction_cache
    gt_answer = (
        "Users can record their daily weight and blood pressure data, write diary entries to log their mood each day, and also add images and files to their records."
    )

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
          'Error during verification for FuncAssistMedilogExplainAllFunctionality: %s', e
      )
      return 0.0
    

class FuncAssistOpenTracksExplainOneFunctionality(task_eval.TaskEval):
  
  app_names = ('OpenTracks',)

  complexity = 4

  schema = {
      'type': 'object',
      'properties': {
          'app_name': {'type': 'string'},
      },
      'required': ['app_name'],
  }

  template = (
    'When I work out using OpenTracks, '
    'what workout data can it record for me?'
    )
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    app_name = random.choice(list(_APP_NAME_TO_PACKAGE_NAME.keys()))
    return {'app_name': app_name}

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not env.interaction_cache:
      return 0.0

    goal = self.template
    prediction = env.interaction_cache
    gt_answer = (
        "It can record data such as distance, "
        "total time, speed, "
        "moving time, average speed, max speed, "
        "average moving speed, and elevation."
    )

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
          'Error during verification for FuncAssistOpenTracksExplainOneFunctionality: %s', e
      )
      return 0.0