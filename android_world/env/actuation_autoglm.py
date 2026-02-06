"""Actuation utilities for AutoGLM Agent in Android World framework.
This module provides action execution capabilities for AutoGLM actions,
extending Android World's standard action set.
"""
import base64
import copy
import logging
import time
from typing import Any
from android_env import env_interface
from android_world.env import adb_utils
from android_world.env import json_action_autoglm as json_action
# ADB Keyboard constants
ADB_KEYBOARD_IME = "com.android.adbkeyboard/.AdbIME"
ADB_INPUT_B64_ACTION = "ADB_INPUT_B64"
ADB_CLEAR_TEXT_ACTION = "ADB_CLEAR_TEXT"
# Timing delays (in seconds)
KEYBOARD_SWITCH_DELAY = 0.5
TEXT_CLEAR_DELAY = 0.3
TEXT_INPUT_DELAY = 0.3
KEYBOARD_RESTORE_DELAY = 0.3
def execute_autoglm_action(
    action: json_action.JSONAction,
    screen_size: tuple[int, int],
    env: env_interface.AndroidEnvInterface,
) -> None:
    """
    Execute an AutoGLM action using ADB commands.
    
    Args:
        action: AutoGLMJSONAction object containing the action to execute.
        screen_size: Tuple of (width, height) of the screen in pixels.
        env: The Android environment interface.
        
    Raises:
        ValueError: If action parameters are invalid.
        Exception: If action execution fails.
    """
    screen_width, screen_height = screen_size
    action_type = action.action_type
    
    if action_type == json_action.CLICK:
        _execute_tap(action, env)
    
    elif action_type == json_action.DOUBLE_TAP:
        _execute_double_tap(action, env)
    
    elif action_type == json_action.LONG_PRESS:
        _execute_long_press(action, env)
    
    elif action_type == json_action.TYPE:
        _execute_type(action, env)
    
    elif action_type == json_action.SWIPE:
        _execute_swipe(action, env)
    
    elif action_type == json_action.BACK:
        _execute_back(env)
    
    elif action_type == json_action.HOME:
        _execute_home(env)
    
    elif action_type == json_action.LAUNCH:
        _execute_launch(action, env)
    
    elif action_type == json_action.WAIT:
        _execute_wait(action)
    
    elif action_type == json_action.TAKE_OVER:
        logging.info(f"Takeover requested: {action.message or 'User intervention required'}")
    
    elif action_type == json_action.NOTE:
        logging.info(f"Note: {action.message or 'Note recorded'}")
    
    elif action_type == json_action.CALL_API:
        logging.info(f"API call: {action.message or 'API called'}")
    
    elif action_type == json_action.INTERACT:
        logging.info(f"Interaction requested: {action.message or 'User interaction required'}")
    
    elif action_type == json_action.FINISH:
        logging.info(f"Finish: {action.message or 'Task completed'}")
    
    else:
        raise ValueError(f"Unsupported action type: {action_type}")
def _execute_tap(
    action: json_action.JSONAction,
    env: env_interface.AndroidEnvInterface,
) -> None:
    """Execute tap action."""
    if action.x is None or action.y is None:
        raise ValueError("Tap action requires x and y coordinates")
    
    adb_utils.tap_screen(action.x, action.y, env)
def _execute_double_tap(
    action: json_action.JSONAction,
    env: env_interface.AndroidEnvInterface,
) -> None:
    """Execute double tap action."""
    if action.x is None or action.y is None:
        raise ValueError("Double tap action requires x and y coordinates")
    
    adb_utils.double_tap(action.x, action.y, env)
def _execute_long_press(
    action: json_action.JSONAction,
    env: env_interface.AndroidEnvInterface,
) -> None:
    """Execute long press action."""
    if action.x is None or action.y is None:
        raise ValueError("Long press action requires x and y coordinates")
    
    adb_utils.long_press(action.x, action.y, env)
def _execute_type(
    action: json_action.JSONAction,
    env: env_interface.AndroidEnvInterface,
) -> None:
    text = action.text
    if text:
    
      response = adb_utils.issue_generic_request(f'shell settings get secure default_input_method', env)
      current_ime = response.generic.output.decode().strip()
      if "com.android.adbkeyboard/.AdbIME" not in current_ime:
        adb_utils.issue_generic_request(f'shell ime set com.android.adbkeyboard/.AdbIME', env)
      _clear_text(env=env)
      adb_utils.type_text(text, env, timeout_sec=10)
      adb_utils.issue_generic_request(f'shell ime set {current_ime}', env)
      # adb_utils.press_enter_button(env)
      
    else:
      logging.warning(
          'Input_text action indicated, but no text provided. No '
          'action will be executed.'
      )
def _send_adb_keyboard_text(
    text: str, 
    env: env_interface.AndroidEnvInterface
) -> None:
    """
    Send text to the device using ADB Keyboard with base64 encoding.
    
    This method handles special characters and multiline text correctly.
    
    Args:
        text: The text to input
        env: Android environment interface
    """
    if not text:
        # Send empty broadcast to warm up keyboard
        adb_utils.issue_generic_request([
            "shell", "am", "broadcast",
            "-a", ADB_INPUT_B64_ACTION,
            "--es", "msg", ""
        ], env)
        return
    
    # Encode text in base64 to handle special characters
    encoded_text = base64.b64encode(text.encode("utf-8")).decode("utf-8")
    
    # Send broadcast to ADB keyboard
    adb_utils.issue_generic_request([
        "shell", "am", "broadcast",
        "-a", ADB_INPUT_B64_ACTION,
        "--es", "msg", encoded_text
    ], env)
def _clear_text(env: env_interface.AndroidEnvInterface) -> None:
    """
    Clear text in the focused input field using ADB keyboard.
    
    Args:
        env: Android environment interface
    """
    adb_utils.issue_generic_request([
        "shell", "am", "broadcast",
        "-a", ADB_CLEAR_TEXT_ACTION
    ], env)
def _get_current_ime(env: env_interface.AndroidEnvInterface) -> str:
    """
    Get the current input method editor (IME).
    
    Args:
        env: Android environment interface
        
    Returns:
        The current IME identifier, or empty string if not found
    """
    response = adb_utils.issue_generic_request([
        "shell", "settings", "get", "secure", "default_input_method"
    ], env)
    
    if response.generic.output:
        ime = response.generic.output.decode().strip()
        return ime
    return ""
def _set_adb_keyboard(env: env_interface.AndroidEnvInterface) -> None:
    """
    Set ADB keyboard as the current input method.
    
    Args:
        env: Android environment interface
    """
    adb_utils.issue_generic_request([
        "shell", "ime", "set", ADB_KEYBOARD_IME
    ], env)
def _restore_ime(ime: str, env: env_interface.AndroidEnvInterface) -> None:
    """
    Restore the specified input method editor.
    
    Args:
        ime: The IME identifier to restore
        env: Android environment interface
    """
    if ime and ime != ADB_KEYBOARD_IME:
        adb_utils.issue_generic_request([
            "shell", "ime", "set", ime
        ], env)
def _execute_swipe(
    action: json_action.JSONAction,
    env: env_interface.AndroidEnvInterface,
) -> None:
    """Execute swipe action."""
    if (action.x is None or action.y is None or 
        action.x2 is None or action.y2 is None):
        raise ValueError("Swipe action requires x, y, x2, y2 coordinates")
    
    duration_ms = action.duration_ms or 500
    
    command = adb_utils.generate_swipe_command(
        action.x, action.y, action.x2, action.y2, duration_ms
    )
    adb_utils.issue_generic_request(command, env)
def _execute_back(env: env_interface.AndroidEnvInterface) -> None:
    """Execute back button press."""
    adb_utils.press_back_button(env)
def _execute_home(env: env_interface.AndroidEnvInterface) -> None:
    """Execute home button press."""
    adb_utils.press_home_button(env)
def _execute_launch(
    action: json_action.JSONAction,
    env: env_interface.AndroidEnvInterface,
) -> None:
    """Execute app launch action."""
    app_name = action.app_name
    if app_name:
        try:
            adb_utils.launch_app(app_name, env)
        except Exception as e:
            logging.warning(f"Failed to launch app '{app_name}': {e}")
    else:
        raise ValueError('No app name provided for launch action')
def _execute_wait(action: json_action.JSONAction) -> None:
    """Execute wait action."""
    duration_str = action.duration or '1 seconds'
    try:
        # Parse duration string like "2 seconds" or "1.5 seconds"
        duration = float(duration_str.replace('seconds', '').replace('second', '').strip())
    except ValueError:
        logging.warning(f"Invalid duration format: {duration_str}, defaulting to 1 second")
        duration = 1.0
    
    time.sleep(duration)