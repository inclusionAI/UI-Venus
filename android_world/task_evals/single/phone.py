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
# Changes: Implementing tasks in VenusBench using phone.


"""Tasks for making and receiving phone calls."""
import threading
from android_world.policy.verification import VerifyPolicy

import random
import time
from absl import logging

from typing import Any
from android_world.env import adb_utils
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import phone_validators
from android_world.task_evals.common_validators import sms_validators
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils


class MarkorCallApartment(phone_validators.MakeCall):
  """Represents a task that combines phone calling with a Markor note lookup.

  This task involves reading a phone number from a Markor note and making
  a phone call to the specified number. It checks whether the In-Call UI is
  displayed
  with options like 'Hold'.
  """

  app_names = ("markor",)
  complexity = 1
  schema = {
      "type": "object",
      "properties": {
          "name": {"type": "string"},
          "phone_number": {"type": "string"},
      },
      "required": ["phone_number"],
  }
  template = (
      "Call the number for the apartment name {name}. The number is in"
      " apartments.md file in Markor. Ensure the In-Call UI is displayed with"
      " options like 'Hold'."
  )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    content = user_data_generation.dict_to_notes(
        user_data_generation.generate_apartments()
    )
    file_utils.create_file(
        "apartments.md", device_constants.MARKOR_DATA, env.controller, content
    )
    self.phone_number = self.params["phone_number"]

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    candidates = user_data_generation.generate_apartments()
    name = random.choice(list(candidates.keys()))
    number = candidates[name][0]
    return {
        "name": name,
        "phone_number": number,
    }


class PhoneMakeCall(phone_validators.MakeCall):
  """Task to make a phone call."""

  template = (
      "Make a phone call to the number {phone_number} and ensure the In-Call UI"
      " is displayed with options like 'Hold'."
  )


class PhoneReturnMissedCall(phone_validators.MakeCall):
  """Task to return a missed phone call.

  This task involves making a call back to a number that was missed and
  verifying that the In-Call UI appears with options like 'Hold'.
  """

  template = (
      "Return the call I just missed and ensure the In-Call UI is displayed"
      " with options like 'Hold'."
  )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    adb_utils.call_emulator(env.controller, self.phone_number)
    time.sleep(5)
    adb_utils.end_call_if_active(env.controller)


class PhoneRedialNumber(phone_validators.MakeCall):
  """Task to re-dial the last dialed number.

  This task involves re-dialing the last dialed number and verifying that the
  In-Call UI appears with options like 'Hold'.
  """

  template = (
      "Re-dial the number I was just talking to and ensure the In-Call UI is "
      "displayed with options like 'Hold'."
  )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    adb_utils.call_phone_number(env.controller, self.phone_number)
    time.sleep(5)
    adb_utils.end_call_if_active(env.controller)


class PhoneCallTextSender(phone_validators.MakeCall):
  """Task to call the sender of the most recent text message.

  This task involves making a call back to the number that most recently sent a
  text message and verifying that the In-Call UI appears with options like
  'Hold'.
  """

  template = (
      "Call the number that just texted me and ensure the In-Call UI is "
      "displayed with options like 'Hold'."
  )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    sms_validators.clear_sms_and_threads(env.controller)
    print(self.phone_number)
    adb_utils.text_emulator(
        env.controller, self.phone_number, "Hey give me a call"
    )


class PhoneAnswerCall(phone_validators.MakeCall):
  """Task to answer an incoming phone call.

  This task involves answering an incoming call from a specified number and
  verifying that the In-Call UI appears with options like 'Hold'.
  """

  template = (
      "Answer the incoming phone call and ensure the In-Call UI is displayed"
      " with options like 'Hold'."
  )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    adb_utils.call_emulator(env.controller, self.phone_number)





class GSATrackingPhoneCall(phone_validators.MakeCall):
  """Task to answer an incoming phone call.

  This task involves answering an incoming call from a specified number and
  verifying that the In-Call UI appears with options like 'Hold'.
  """

  complexity = 4

  template = (
      "8 calls will be coming in—decline all of them. "
      "For any number that has called at least three times, "
      "append that number to the file ‘phonenumber.md’ in Markor. "
  )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    file_utils.clear_directory(device_constants.MARKOR_DATA, env.controller)

    self.call_thread = threading.Thread(
        target=self._make_multiple_calls,
        args=(env,),
        daemon=True
    )
    self.call_thread.start()
    self._stop_calling = False  

  def _make_multiple_calls(self, env: interface.AsyncEnv):
    """后台线程：每10秒打一次电话，共8个电话"""
    
    phone_numbers = self.params["phone_numbers"]

    time.sleep(10)

    for i, phone_number in enumerate(phone_numbers):
      if self._stop_calling:
        print("Stopping phone calls as requested")
        break
      self._wait_for_call_to_end(env, max_wait_time=15)
      

      if self._stop_calling:
        print("Stopping phone calls as requested")
        break

      success = False
      retry_count = 0
      max_retries = 3
      
      while not success and retry_count < max_retries:
        try:
          adb_utils.call_emulator(env.controller, phone_number)
          print(f"Call {i+1}/8: {phone_number}")
          success = True
        except Exception as e:
          print(f"Error making call {i+1} (attempt {retry_count+1}): {e}")
          retry_count += 1
          if retry_count < max_retries:
            time.sleep(2)  
      
      if i < len(phone_numbers) - 1:
        time.sleep(10)
    
  def _wait_for_call_to_end(self, env: interface.AsyncEnv, max_wait_time: int = 20):
    """等待当前电话结束"""
    
    wait_time = 0
    check_interval = 1
    
    while wait_time < max_wait_time:
      if self._stop_calling:
        return False
      
      try:
        call_state = adb_utils.get_call_state(env.controller)
        print(f"Call state check: {call_state}")
        # IDLE RINING OFFHOOK
        if "IDLE" in call_state:
          print("Previous call ended, ready for next call")
          return True
        
        time.sleep(check_interval)
        wait_time += check_interval
      except Exception as e:
        print(f"Error checking call state: {e}")
        time.sleep(check_interval)
        wait_time += check_interval
    
    print(f"Warning: Call did not end within {max_wait_time} seconds, proceeding anyway")
    return False

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    
    repeated_number = "555" + "".join(random.choices("0123456789", k=7))
    
    other_numbers = []
    for _ in range(5):
      phone_num = "555" + "".join(random.choices("0123456789", k=7))
      other_numbers.append(phone_num)
    
    all_numbers = [repeated_number] * 3 + other_numbers
    
    random.shuffle(all_numbers)
    # print(repeated_number)
    return {
        "phone_number": repeated_number, 
        "phone_numbers": all_numbers       
    }
  

  def is_successful(self, env: interface.AsyncEnv) -> float:
    self._stop_calling = True
    
    if hasattr(self, 'call_thread') and self.call_thread.is_alive():
      self.call_thread.join(timeout=5)  
      if self.call_thread.is_alive():
        print("Warning: Call thread did not stop within timeout")
    
    # super().is_successful(env)
    if not file_utils.check_file_or_folder_exists(
          'phonenumber.md',
          device_constants.MARKOR_DATA,
          env.controller,
      ):
      return 0.0
    
    if not file_utils.check_file_content(
          file_utils.convert_to_posix_path(
              device_constants.MARKOR_DATA, 'phonenumber.md'
          ),
          self.params['phone_number'],
          env.controller,
      ):
      return 0.0
    return 1.0


class GSATrackingPhoneCallCHS(GSATrackingPhoneCall):
  template = (
      "接下来会收到 8 通电话——全部拒接。"
"对于任何号码，只要它在本次测试中累计拨打过至少 3 次，就把该号码追加到 Markor 的 ‘phonenumber.md’ 文件里。"
  )


class GSATrackingPhoneCallENGVariation(GSATrackingPhoneCall):
  template = (
    "You will receive 8 incoming calls—reject all of them. "
    "For each number that has called at least 3 times, "
    "add that number to the 'phonenumber.md' file in Markor. "
)

class GSATrackingWhitelistCallAnswer(phone_validators.MakeCall):
  """Task to pick up calls from a whitelist only.
  Any number not in the whitelist should be declined immediately.
  """

  template = (
      "8 calls will be coming in. "
      "Decline all of them. "
      "After that, "
      "report me the numbers in the following {whitelist} "
      "that called you, "
      "in the order they called, separated by commas. "
      "For example: 5551234567,5559876543,5555555555"
  )

  complexity = 5

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    self.call_thread = threading.Thread(
        target=self._make_multiple_calls,
        args=(env,),
        daemon=True
    )
    self.call_thread.start()
    self._stop_calling = False  


  def _make_multiple_calls(self, env: interface.AsyncEnv):
    phone_numbers = self.params["call_numbers"]

    time.sleep(10)

    for i, phone_number in enumerate(phone_numbers):
      if self._stop_calling:
        print("Stopping phone calls as requested")
        break
      self._wait_for_call_to_end(env, max_wait_time=15)

      if self._stop_calling:
        print("Stopping phone calls as requested")
        break

      success = False
      retry_count = 0
      max_retries = 3
      
      while not success and retry_count < max_retries:
        try:
          adb_utils.call_emulator(env.controller, phone_number)
          print(f"Call {i+1}/8: {phone_number}")
          success = True
        except Exception as e:
          print(f"Error making call {i+1} (attempt {retry_count+1}): {e}")
          retry_count += 1
          if retry_count < max_retries:
            time.sleep(2)  
      
      if i < len(phone_numbers) - 1:
        time.sleep(10)

  def _wait_for_call_to_end(self, env: interface.AsyncEnv, max_wait_time: int = 20):
    
    wait_time = 0
    check_interval = 1
    
    while wait_time < max_wait_time:
      if self._stop_calling:
        return False
      
      try:
        call_state = adb_utils.get_call_state(env.controller)
        print(f"Call state check: {call_state}")
        # IDLE RINING OFFHOOK
        if "IDLE" in call_state:
          print("Previous call ended, ready for next call")
          return True
        
        time.sleep(check_interval)
        wait_time += check_interval
      except Exception as e:
        print(f"Error checking call state: {e}")
        time.sleep(check_interval)
        wait_time += check_interval
    
    print(f"Warning: Call did not end within {max_wait_time} seconds, proceeding anyway")
    return False


  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    
    whitelist_numbers = []
    for _ in range(10):
      phone_num = "555" + "".join(random.choices("0123456789", k=7))
      whitelist_numbers.append(phone_num)
    
    expected = random.sample(whitelist_numbers, 3)

    other_numbers = []
    for _ in range(5):
      phone_num = "555" + "".join(random.choices("0123456789", k=7))
      while phone_num in whitelist_numbers:
        phone_num = "555" + "".join(random.choices("0123456789", k=7))
      other_numbers.append(phone_num)

    call_numbers = expected + other_numbers
    random.shuffle(call_numbers)
    
    random.shuffle(whitelist_numbers)
    expected_in_order = [num for num in call_numbers if num in whitelist_numbers]


    return {
        "phone_number": '',
        "whitelist": whitelist_numbers,   
        "call_numbers": call_numbers,       
        'expected': expected_in_order,
    }
  
  def is_successful(self, env: interface.AsyncEnv) -> float:
   
    self._stop_calling = True
    
    if hasattr(self, 'call_thread') and self.call_thread.is_alive():
      self.call_thread.join(timeout=5)  
      if self.call_thread.is_alive():
        print("Warning: Call thread did not stop within timeout")
    

    # Verify answer
    if not env.interaction_cache:
      return 0.0
    
    gt_ans = ','.join(self.params['expected'])
    
    user_answer = env.interaction_cache.strip()
    
    if user_answer == gt_ans:
      return 1.0
    else:
      print(f"Expected: {gt_ans}")
      print(f"Got: {user_answer}")
      return 0.0


class GSATrackingBlacklistBlockAndSendSms(sms_validators.SimpleSMSSendSms):

  template = (
      "8 calls will be coming in. "
      "Monitor the incoming-call screen continuously. "
      "If the caller's number is in the blacklist {blacklist}, "
      "decline the call immediately. "
      "Otherwise, decline it and immediately follow up with an SMS "
      "that says: 'I am tied up right now—will get back to you later.'"
  )
  complexity = 5
  schema = {
      "type": "object",
      "properties": {
          "blacklist": {"type": "array"},
          "call_numbers": {"type": "array"},
          "non_blacklist_numbers": {"type": "array"},
      },
      "required": ["blacklist", "call_numbers", "non_blacklist_numbers"],
  }
  

  def initialize_task(self, env: interface.AsyncEnv):
    # Skip SimpleSMSSendSms.initialize_task and call TaskEval.initialize_task directly
    # to avoid checking for required 'message' parameter
    task_eval.TaskEval.initialize_task(self, env)
    adb_utils.toggle_airplane_mode("off", env.controller)
    sms_validators.clear_sms_and_threads(env.controller)
    
    self.call_thread = threading.Thread(
        target=self._make_multiple_calls,
        args=(env,),
        daemon=True
    )
    self.call_thread.start()
    self._stop_calling = False  



  def _make_multiple_calls(self, env: interface.AsyncEnv):
    phone_numbers = self.params["call_numbers"]

    time.sleep(10)

    for i, phone_number in enumerate(phone_numbers):
      if self._stop_calling:
        print("Stopping phone calls as requested")
        break
      self._wait_for_call_to_end(env, max_wait_time=15)

      if self._stop_calling:
        print("Stopping phone calls as requested")
        break

      success = False
      retry_count = 0
      max_retries = 3
      
      while not success and retry_count < max_retries:
        try:
          adb_utils.call_emulator(env.controller, phone_number)
          print(f"Call {i+1}/8: {phone_number}")
          success = True
        except Exception as e:
          print(f"Error making call {i+1} (attempt {retry_count+1}): {e}")
          retry_count += 1
          if retry_count < max_retries:
            time.sleep(2)  
      
      if i < len(phone_numbers) - 1:
        time.sleep(10)

  def _wait_for_call_to_end(self, env: interface.AsyncEnv, max_wait_time: int = 20):
    
    wait_time = 0
    check_interval = 1
    
    while wait_time < max_wait_time:
      if self._stop_calling:
        return False
      
      try:
        #  adb shell dumpsys telephony.registry | grep mCallState
        call_state = adb_utils.get_call_state(env.controller)
        print(f"Call state check: {call_state}")
        # IDLE RINING OFFHOOK
        if "IDLE" in call_state:
          print("Previous call ended, ready for next call")
          return True
        
        time.sleep(check_interval)
        wait_time += check_interval
      except Exception as e:
        print(f"Error checking call state: {e}")
        time.sleep(check_interval)
        wait_time += check_interval
    
    print(f"Warning: Call did not end within {max_wait_time} seconds, proceeding anyway")
    return False
  

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:

    
    blacklist = []
    for _ in range(6):
      phone_num = "555" + "".join(random.choices("0123456789", k=7))
      blacklist.append(phone_num)
    
    blacklisted_calls = blacklist.copy()  
    
    non_blacklist_calls = []
    for _ in range(2):
      phone_num = "555" + "".join(random.choices("0123456789", k=7))
      # 
      while phone_num in blacklist:
        phone_num = "555" + "".join(random.choices("0123456789", k=7))
      non_blacklist_calls.append(phone_num)
    
    # 
    call_numbers = blacklisted_calls + non_blacklist_calls
    
    # 
    random.shuffle(call_numbers)
    
    return {
        "phone": '',
        "number": '',
        "blacklist": blacklist,         
        "call_numbers": call_numbers,   
        "non_blacklist_numbers": non_blacklist_calls,  
    }
  
  def is_successful(self, env: interface.AsyncEnv) -> float:

    self._stop_calling = True
    
    if hasattr(self, 'call_thread') and self.call_thread.is_alive():
      self.call_thread.join(timeout=5)  
      if self.call_thread.is_alive():
        print("Warning: Call thread did not stop within timeout")
  
    expected_sms_dict = {
      number: "I am tied up right now—will get back to you later."
      for number in self.params["non_blacklist_numbers"]
    }
    return self.is_successful_multiple_sms(env, expected_sms_dict = expected_sms_dict)