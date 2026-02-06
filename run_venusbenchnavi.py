from collections.abc import Sequence
import os
import tempfile
tempfile.tempdir = os.path.join(os.getcwd(), "temp")

from absl import app
from absl import flags
from absl import logging
from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from android_world.agents import base_agent
from android_world.agents import human_agent
from android_world.agents import infer
from android_world.agents import infer_ma3

from android_world.agents import m3a
from android_world.agents import random_agent
from android_world.agents import seeact
from android_world.env import env_launcher
from android_world.env import interface
from android_world.agents import gui_owl
from android_world.agents import qwen3vl
from android_world.agents import mobile_agent_v3
from android_world.agents import autoglm

import yaml
from pathlib import Path

"""
from android_world.env import env_launcher
env = env_launcher.load_and_setup_env(console_port=5554, emulator_setup=False, adb_path='/root/android/android_sdk/platform-tools/adb', grpc_port=8554)
state = env.get_state()
from PIL import Image
Image.from_array(state.pixels).save('tmp.jpg')
"""

logging.set_verbosity(logging.WARNING)

os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Only show errors
os.environ['GRPC_TRACE'] = 'none'  # Disable tracing



script_dir = os.path.dirname(os.path.abspath(__file__))
# 
yaml_path = os.path.join(script_dir, './config/venus_benchmark_settings.yaml')
# 
yaml_path = os.path.abspath(yaml_path)

with open(yaml_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


splits = [['SystemBrightnessMin', 'MarkorChangeNoteContent', 'RecipeDeleteMultipleRecipesWithNoise', 'ClockStopWatchPausedVerify', 'FilesDeleteFile', 'TurnOffWifiAndTurnOnBluetooth', 'SimpleCalendarLocationOfEvent', 'OsmAndTrack', 'BrowserMaze', 'TasksDueOnDate', 'SystemBluetoothTurnOnVerify', 'NotesIsTodo', 'SimpleCalendarAddOneEventRelativeDay', 'TurnOnWifiAndOpenApp', 'MarkorEditNote', 'MarkorCreateNoteFromClipboard', 'SimpleCalendarEventsOnDate', 'NotesMeetingAttendeeCount', 'MarkorMoveNote', 'SimpleSmsReplyMostRecent', 'SimpleSmsResend', 'SimpleCalendarDeleteOneEvent', 'SimpleCalendarFirstEventAfterStartTime', 'SimpleCalendarAddOneEventTomorrow', 'RetroSavePlaylist', 'SystemWifiTurnOnVerify', 'MarkorMergeNotes', 'RecipeDeleteMultipleRecipes', 'SystemBrightnessMinVerify', 'ContactsAddContact', 'RecipeAddMultipleRecipes', 'OpenAppTaskEval', 'SimpleSmsSend', 'RetroPlayingQueue', 'TasksDueNextWeek', 'MarkorDeleteNote', 'MarkorAddNoteHeader', 'RecipeDeleteDuplicateRecipes2'],
['FilesMoveFile', 'ExpenseDeleteMultiple', 'ExpenseAddSingle', 'MarkorTranscribeReceipt', 'SportsTrackerTotalDistanceForCategoryOverInterval', 'SystemWifiTurnOff', 'SimpleSmsSendReceivedAddress', 'ClockTimerEntry', 'RecipeDeleteDuplicateRecipes3', 'SimpleCalendarAddOneEvent', 'SystemBrightnessMax', 'ExpenseAddMultipleFromGallery', 'AudioRecorderRecordAudioWithFileName', 'OsmAndFavorite', 'SimpleSmsReply', 'ExpenseDeleteSingle', 'RetroPlaylistDuration', 'SystemWifiTurnOn', 'MarkorCreateFolder', 'MarkorDeleteNewestNote', 'SimpleCalendarNextMeetingWithPerson', 'SimpleSmsSendClipboardContent', 'TasksCompletedTasksForDate', 'RecipeAddSingleRecipe', 'MarkorTranscribeVideo', 'CameraTakeVideo', 'SimpleCalendarDeleteEventsOnRelativeDay', 'RecipeDeleteSingleWithRecipeWithNoise', 'RecipeDeleteMultipleRecipesWithConstraint', 'RecipeAddMultipleRecipesFromMarkor2', 'BrowserDraw', 'BrowserMultiply', 'TasksHighPriorityTasksDueOnDate', 'SportsTrackerLongestDistanceActivity', 'SimpleCalendarNextEvent', 'MarkorDeleteAllNotes', 'SimpleCalendarAddRepeatingEvent', 'SimpleCalendarAddOneEventInTwoWeeks', 'SimpleCalendarAnyEventsOnDate'],
['RecipeDeleteDuplicateRecipes', 'SystemBrightnessMaxVerify', 'SportsTrackerActivitiesCountForWeek', 'SportsTrackerTotalDurationForCategoryThisWeek', 'AudioRecorderRecordAudio', 'SimpleCalendarEventOnDateAtTime', 'SportsTrackerActivityDuration', 'SystemWifiTurnOffVerify', 'VlcCreateTwoPlaylists', 'RecipeAddMultipleRecipesFromImage', 'NotesTodoItemCount', 'TasksIncompleteTasksOnDate', 'SimpleCalendarEventsInTimeRange', 'ExpenseAddMultipleFromMarkor', 'SystemCopyToClipboard', 'SystemBluetoothTurnOff', 'SportsTrackerActivitiesOnDate', 'SimpleCalendarEventsInNextWeek', 'ExpenseDeleteDuplicates', 'ExpenseDeleteMultiple2', 'SystemBluetoothTurnOffVerify', 'SimpleCalendarDeleteEvents', 'NotesRecipeIngredientCount', 'ClockStopWatchRunning', 'TasksHighPriorityTasks', 'OsmAndMarker', 'RecipeAddMultipleRecipesFromMarkor', 'RecipeDeleteSingleRecipe', 'MarkorCreateNote', 'SimpleDrawProCreateDrawing', 'ContactsNewContactDraft', 'ExpenseAddMultiple', 'SystemBluetoothTurnOn', 'RetroCreatePlaylist', 'SaveCopyOfReceiptTaskEval', 'MarkorCreateNoteAndSms', 'ExpenseDeleteDuplicates2', 'VlcCreatePlaylist', 'CameraTakePhoto'],]

_ADB_PATH = flags.DEFINE_string(
    'adb_path',
    'Path to adb. Set if not installed through SDK.',
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    'perform_emulator_setup',
    False,
    'Whether to perform emulator setup. This must be done once and only once'
    ' before running Android World. After an emulator is setup, this flag'
    ' should always be False.',
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port',
    5556,
    'The console port of the running Android device. This can usually be'
    ' retrieved by looking at the output of `adb devices`. In general, the'
    ' first connected device is port 5554, the second is 5556, and'
    ' so on.',
)
_DEVICE_GRPC_PORT = flags.DEFINE_integer(
    'grpc_port',
    #8572,
    8556,
    'grpc port',
)

_DEVICE_ADB_SERVER_PORT = flags.DEFINE_integer(
    'adb_server_port',
    5037,
    'adb server port',
)

_DEVICE_DARK_MODE = flags.DEFINE_string(
    'dark_mode',
    'off',
    'Enable dark mode',
)

_DEVICE_PAD_MODE = flags.DEFINE_string(
    'pad_mode',
    'off',
    'Enable pad mode',
)

_DEVICE_NAME = flags.DEFINE_string(
    'device_name',
    '',
    'device name',
)

_SUITE_FAMILY = flags.DEFINE_enum(
    'suite_family',
    registry.TaskRegistry.VENUS_FAMILY,
    [
        # Families from the paper.
        registry.TaskRegistry.ANDROID_WORLD_FAMILY,
        # registry.TaskRegistry.MINIWOB_FAMILY_SUBSET,
        # Other families for more testing.
        # registry.TaskRegistry.MINIWOB_FAMILY,
        registry.TaskRegistry.VENUS_FAMILY,
        registry.TaskRegistry.ANDROID_FAMILY,
        registry.TaskRegistry.INFORMATION_RETRIEVAL_FAMILY,
    ],
    'Suite family to run. See registry.py for more information.',
)
_TASK_RANDOM_SEED = flags.DEFINE_integer(
    'task_random_seed', 30, 'Random seed for task randomness.'
)

_SPLITS_ID = flags.DEFINE_integer(
    'splits_id',
    -1,
    'split index',
)

_HISTORY_LENGTH = flags.DEFINE_integer(
    'history_length',
    0,
    'history length',
)

_TASKS = flags.DEFINE_list(
    'tasks',
    None,
    'List of specific tasks to run in the given suite family. If None, run all'
    ' tasks in the suite family.',
)
_N_TASK_COMBINATIONS = flags.DEFINE_integer(
    'n_task_combinations',
    1,
    'Number of task instances to run for each task template.',
)

_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    '',
    'The directory to save checkpoints and resume evaluation from. If the'
    ' directory contains existing checkpoint files, evaluation will resume from'
    ' the latest checkpoint. If the directory is empty or does not exist, a new'
    ' directory will be created.',
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    # os.path.expanduser('~/android_world/runs'),
    './results',
    'The path to save results to if not resuming from a checkpoint is not'
    ' provided.',
)

_IMAGES_FIRST = flags.DEFINE_boolean(
    'images_first',
    True,
    'input image-text or text-image',
)
_SUMMARY_MODEL = flags.DEFINE_enum(
    'summary_model',
    'self',
    ['qwen', 'self'],
    'choices = qwen or self, qwen means qwen-2.5-vl-72b',
)

_ADD_ANSWER = flags.DEFINE_boolean(
    'add_answer',
    True,
    'Append a new action `answer` into the original action prompt.',
)
_TRAJ_OUTPUT_PATH = flags.DEFINE_string(
    'traj_output_path',
    '',
    'The path to save traj'
)
# Agent specific.
_AGENT_NAME = flags.DEFINE_string('agent_name', 'qwen_raw_7b', help='Agent name.')


_FIXED_TASK_SEED = flags.DEFINE_boolean(
    'fixed_task_seed',
    False,
    'Whether to use the same task seed when running multiple task combinations'
    ' (n_task_combinations > 1).',
)

_TV_OR_VT = flags.DEFINE_string(
    'tv_or_vt',
    'tv',
    'Text-vision or vision-text input.',
)

_NEED_CONCLUSION = flags.DEFINE_boolean(
    'need_conclusion',
    False,
    'Need summary the action in conclusion tag.',
)

_API_KEY = flags.DEFINE_string(
    'api_key',
    '',
    'Your api key'
)
_BASE_URL = flags.DEFINE_string(
    'base_url',
    '',
    'Your base url'
)
_MODEL = flags.DEFINE_string(
    'model',
    '',
    'Your model name.',
)

def _get_agent(
    env: interface.AsyncEnv,
    family: str | None = None,
) -> base_agent.EnvironmentInteractingAgent:
  """Gets agent."""
  print('Initializing agent...')
  agent = None
  if _AGENT_NAME.value == 'human_agent':
    agent = human_agent.HumanAgent(env, output_path=_TRAJ_OUTPUT_PATH.value)

  # Gemini via Proxy.
  elif _AGENT_NAME.value == 'm3a_gemini_gcp':
    base_url = config['agents']['gemini_proxy']['base_url']
    api_key = config['agents']['gemini_proxy']['api_key']

    agent = m3a.M3A(
        env, infer.GeminiProxyWrapper(
            model_name='gemini-2.5-pro',
            base_url=base_url,
            api_key=api_key,
        ),
        output_path=_TRAJ_OUTPUT_PATH.value
    )

  # Gemini 3 via Proxy
  elif _AGENT_NAME.value == 'm3a_gemini3':
    base_url = config['agents']['gemini3_proxy']['base_url']
    api_key = config['agents']['gemini3_proxy']['api_key']
    model_name = config['agents']['gemini3_proxy']['model']

    agent = m3a.M3A(
        env, infer.GeminiProxyWrapper(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
        ),
        output_path=_TRAJ_OUTPUT_PATH.value
    )

  # GPT-4o via Proxy 
  elif _AGENT_NAME.value == 'm3a_gpt4o_proxy':
    base_url = config['agents']['gpt4o_proxy']['base_url']
    api_key = config['agents']['gpt4o_proxy']['api_key']

    agent = m3a.M3A(
        env, infer.Gpt4Wrapper(
            model_name='gpt-4o-2024-11-20',
            base_url=base_url,
            api_key=api_key,
            use_stream=True
        ),
        output_path=_TRAJ_OUTPUT_PATH.value
    )

  # Claude via Proxy
  elif _AGENT_NAME.value == 'm3a_claude_proxy':
    base_url = config['agents']['claude_proxy']['base_url']
    api_key = config['agents']['claude_proxy']['api_key']
    agent = m3a.M3A(
        env, infer.ClaudeProxyWrapper(
            model_name='claude-sonnet-4-20250514',
            base_url=base_url,
            api_key=api_key,
            use_stream=True
        ),
        output_path=_TRAJ_OUTPUT_PATH.value
    )
  elif _AGENT_NAME.value == 'QwenGD_V':
    from android_world.agents.qwen_agent_gd_withgpt4 import QwenGD_V
    base_url = config['agents']['gemini3_proxy']['base_url']
    api_key = config['agents']['gemini3_proxy']['api_key']
    model_name = config['agents']['gemini3_proxy']['model']

    base_url_gd = config['models']['gd']['base_url']
    agent = QwenGD_V(env,
                     infer.GeminiProxyWrapper(
                        model_name=model_name,
                        base_url=base_url,
                        api_key=api_key,
                     ), 
                     base_url_gd=base_url_gd)
    agent.set_output_path(_TRAJ_OUTPUT_PATH.value)

  elif _AGENT_NAME.value == 'QwenGD_GPT':
    from android_world.agents.qwen_agent_gd_withgpt4 import QwenGD_V
    base_url = config['agents']['gpt5_proxy']['base_url']
    api_key = config['agents']['gpt5_proxy']['api_key']
    model_name = config['agents']['gpt5_proxy']['model']

    base_url_gd = config['models']['gd']['base_url']
    agent = QwenGD_V(env,
                     infer.Gpt4Wrapper(
                        model_name=model_name,
                        base_url=base_url,
                        api_key=api_key,
                     ), 
                     base_url_gd=base_url_gd)
    agent.set_output_path(_TRAJ_OUTPUT_PATH.value)

 
  elif _AGENT_NAME.value == 'qwen3vl':
    print('Agent: qwen3vl')
    base_url = config['agents']['qwen3vl']['base_url']
    agent = qwen3vl.Qwen3_VL(env, infer_ma3.Qwen3VLWrapper('empty', base_url, _MODEL.value),"abs_resized", api_key=None, url=None, output_path=(_TRAJ_OUTPUT_PATH.value))
  
  elif _AGENT_NAME.value == 'gui_owl':

    base_url = config['agents']['gui_owl']['base_url']
    agent = gui_owl.GUIOwl(env, 
                           infer_ma3.GUIOwlWrapper('empty', base_url, _MODEL.value), 
                           "abs_resized", 
                           api_key=None, 
                           url=None, 
                           output_path=(_TRAJ_OUTPUT_PATH.value))
    

  elif _AGENT_NAME.value == 'mobile_agent_v3':
    
    base_url = config['agents']['gui_owl']['base_url']
    agent = mobile_agent_v3.MobileAgentV3_M3A(env, 
    infer_ma3.GUIOwlWrapper('empty', base_url, _MODEL.value), 
    output_path=(_TRAJ_OUTPUT_PATH.value))
  

  elif _AGENT_NAME.value == 'autoglm':
    base_url = config['agents']['autoglm']['base_url']
    agent = autoglm.AutoGLMAgent(env, 
                                 model_base_url=base_url,
                                 output_path=(_TRAJ_OUTPUT_PATH.value))

  if not agent:
    raise ValueError(f'Unknown agent: {_AGENT_NAME.value}')
  agent.name = _AGENT_NAME.value
  return agent

import json


def _main() -> None:
  """Runs eval suite and gets rewards back."""
  from utils.logger import setup_logging
  # setup_logging()
  env = env_launcher.load_and_setup_env(
      console_port=_DEVICE_CONSOLE_PORT.value,
      emulator_setup=_EMULATOR_SETUP.value,
      adb_path=_ADB_PATH.value,
      grpc_port=_DEVICE_GRPC_PORT.value,
      adb_server_port=_DEVICE_ADB_SERVER_PORT.value,
      device_name=_DEVICE_NAME.value,
  )

  n_task_combinations = _N_TASK_COMBINATIONS.value
  task_registry = registry.TaskRegistry()
  suite = suite_utils.create_suite(
      task_registry.get_registry(family=_SUITE_FAMILY.value),
      n_task_combinations=n_task_combinations,
      seed=_TASK_RANDOM_SEED.value,
      tasks=_TASKS.value,
      use_identical_params=_FIXED_TASK_SEED.value,
  )
  suite.suite_family = _SUITE_FAMILY.value
      


  agent = _get_agent(env, _SUITE_FAMILY.value)
  agent.transition_pause = None

  if _CHECKPOINT_DIR.value:
    checkpoint_dir = _CHECKPOINT_DIR.value
  else:
    checkpoint_dir = _OUTPUT_PATH.value

  print(
      f'Starting eval with agent {_AGENT_NAME.value} and writing to'
      f' {checkpoint_dir}'
  )
  result = suite_utils.run(
      suite,
      agent,
      checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
      demo_mode=False,
      dark_mode=_DEVICE_DARK_MODE.value,
      pad_mode=_DEVICE_PAD_MODE.value,
  )
  print(
      f'Finished running agent {_AGENT_NAME.value} on {_SUITE_FAMILY.value}'
      f' family. Wrote to {checkpoint_dir}.'
  )
  env.close()


stability_subset = [
    # GSA 4
    "GSATimingAudioRecorderRecordAudioTime",
    "GSATrackingPhoneCall",

    # GUIM 8
    "GUIMDrawA",
    "GUIMEraseObject1",

    # Refusal 14
    "RefusalExpenseDeleteMultipleConflictMultiple",
    "RefusalSystemBluetoothTurnOnAlreadyOn",
    "RefusalMarkorDeleteMultipleNotesAmbigious1",
    
    # FuncAssist 18
    "FuncAssistExpenseExplainOneFunctionality1",
    "FuncAssistVlcLocateOneFunctionality1",
    
    # GUIBrowsing 24
    "GUIBrowsingBrowserRandomButtons2",
    "GUIBrowsingPaper1",
    "GUIBrowsingOrder2",
    
    # Browsecomp 28
    "BrowsecompFindAppTaskEvalUI1",
    "BrowsecompVlcFindVlcAPP1",
    
    # NoiseResist 32
    "NoiseResistExpenseDeleteSingleWithOrientation",
    "NoiseResistFilesMoveFileAPPCollapse",
    
    # MultiRound 36
    "MultiRoundMarkorAppendCopyRename",
    "MultiRoundRetroSavePlaylist",
    
    # Vague 40
    "VagueSystemBrightnessMax",
    "VagueMarkorDeleteNewestTwoNotes",

]

stability_subset_instruction_variations = [
    # GSA 4
    "GSATimingAudioRecorderRecordAudioTimeCHS",
    "GSATimingAudioRecorderRecordAudioTimeENGVariation",
    "GSATrackingPhoneCallCHS",
    "GSATrackingPhoneCallENGVariation",

    # GUIM 8
    "GUIMDrawACHS",
    "GUIMDrawAENGVariation",
    "GUIMEraseObject1CHS",
    "GUIMEraseObject1ENGVariation",

    # Refusal 14
    "RefusalExpenseDeleteMultipleConflictMultipleCHS",
    "RefusalExpenseDeleteMultipleConflictMultipleVariation",
    "RefusalSystemBluetoothTurnOnAlreadyOnCHS",
    "RefusalSystemBluetoothTurnOnAlreadyOnVariation",
    "RefusalMarkorDeleteMultipleNotesAmbigious1CHS",
    "RefusalMarkorDeleteMultipleNotesAmbigious1Variation",
    
    # FuncAssist 18
    "FuncAssistExpenseExplainOneFunctionality1CHS",
    "FuncAssistExpenseExplainOneFunctionality1Variation",
    "FuncAssistVlcLocateOneFunctionality1CHS",
    "FuncAssistVlcLocateOneFunctionality1Variation",
    
    # GUIBrowsing 24
    "GUIBrowsingBrowserRandomButtons2CHS",
    "GUIBrowsingBrowserRandomButtons2Variation",
    "GUIBrowsingPaper1CHS",
    "GUIBrowsingPaper1Variation",
    "GUIBrowsingOrder2CHS",
    "GUIBrowsingOrder2Variation",
    
    # Browsecomp 28
    "BrowsecompFindAppTaskEvalUI1CHS",
    "BrowsecompFindAppTaskEvalUI1Variation",
    "BrowsecompVlcFindVlcAPP1CHS",
    "BrowsecompVlcFindVlcAPP1Variation",
    
    # NoiseResist 32
    "NoiseResistExpenseDeleteSingleWithOrientationCHS",
    "NoiseResistExpenseDeleteSingleWithOrientationVariation",
    "NoiseResistFilesMoveFileAPPCollapseCHS",
    "NoiseResistFilesMoveFileAPPCollapseVariation",
    
    # MultiRound 36
    "MultiRoundMarkorAppendCopyRenameCHS",
    "MultiRoundMarkorAppendCopyRenameVariation",
    "MultiRoundRetroSavePlaylistCHS",
    "MultiRoundRetroSavePlaylistVariation",
    
    # Vague 40
    "VagueSystemBrightnessMaxCHS",
    "VagueSystemBrightnessMaxVariation",
    "VagueMarkorDeleteMultipleNotesCHS",
    "VagueMarkorDeleteMultipleNotesVariation",
]


def main(argv: Sequence[str]) -> None:
  if not _TASKS.value:  #
      splits_id = _SPLITS_ID.value
      if splits_id >= 0:
        task_split = splits[splits_id]
      else:
        task_split = None
      flags.FLAGS.tasks = task_split 
  elif _TASKS.value[0] == 'stability':
    flags.FLAGS.tasks = stability_subset
    
  del argv
  _main()


if __name__ == '__main__':
  app.run(main)
