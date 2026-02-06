from collections.abc import Sequence
import os
import tempfile
tempfile.tempdir = os.path.join(os.getcwd(), "temp")
import glob

from absl import app
from absl import flags
from absl import logging
from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from android_world.agents import base_agent
from android_world.agents import human_agent
from android_world.agents import infer
from android_world.agents import m3a
from android_world.agents import random_agent
from android_world.agents import seeact
from android_world.agents import t3a
from android_world.env import env_launcher
from android_world.env import interface
from android_world.env import json_action, adb_utils, actuation
from android_world.utils import file_utils
from config import config


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


splits = [['SystemBrightnessMin', 'MarkorChangeNoteContent', 'RecipeDeleteMultipleRecipesWithNoise', 'ClockStopWatchPausedVerify', 'FilesDeleteFile', 'TurnOffWifiAndTurnOnBluetooth', 'SimpleCalendarLocationOfEvent', 'OsmAndTrack', 'BrowserMaze', 'TasksDueOnDate', 'SystemBluetoothTurnOnVerify', 'NotesIsTodo', 'SimpleCalendarAddOneEventRelativeDay', 'TurnOnWifiAndOpenApp', 'MarkorEditNote', 'MarkorCreateNoteFromClipboard', 'SimpleCalendarEventsOnDate', 'NotesMeetingAttendeeCount', 'MarkorMoveNote', 'SimpleSmsReplyMostRecent', 'SimpleSmsResend', 'SimpleCalendarDeleteOneEvent', 'SimpleCalendarFirstEventAfterStartTime', 'SimpleCalendarAddOneEventTomorrow', 'RetroSavePlaylist', 'SystemWifiTurnOnVerify', 'MarkorMergeNotes', 'RecipeDeleteMultipleRecipes', 'SystemBrightnessMinVerify', 'ContactsAddContact', 'RecipeAddMultipleRecipes', 'OpenAppTaskEval', 'SimpleSmsSend', 'RetroPlayingQueue', 'TasksDueNextWeek', 'MarkorDeleteNote', 'MarkorAddNoteHeader', 'RecipeDeleteDuplicateRecipes2'],
['FilesMoveFile', 'ExpenseDeleteMultiple', 'ExpenseAddSingle', 'MarkorTranscribeReceipt', 'SportsTrackerTotalDistanceForCategoryOverInterval', 'SystemWifiTurnOff', 'SimpleSmsSendReceivedAddress', 'ClockTimerEntry', 'RecipeDeleteDuplicateRecipes3', 'SimpleCalendarAddOneEvent', 'SystemBrightnessMax', 'ExpenseAddMultipleFromGallery', 'AudioRecorderRecordAudioWithFileName', 'OsmAndFavorite', 'SimpleSmsReply', 'ExpenseDeleteSingle', 'RetroPlaylistDuration', 'SystemWifiTurnOn', 'MarkorCreateFolder', 'MarkorDeleteNewestNote', 'SimpleCalendarNextMeetingWithPerson', 'SimpleSmsSendClipboardContent', 'TasksCompletedTasksForDate', 'RecipeAddSingleRecipe', 'MarkorTranscribeVideo', 'CameraTakeVideo', 'SimpleCalendarDeleteEventsOnRelativeDay', 'RecipeDeleteSingleWithRecipeWithNoise', 'RecipeDeleteMultipleRecipesWithConstraint', 'RecipeAddMultipleRecipesFromMarkor2', 'BrowserDraw', 'BrowserMultiply', 'TasksHighPriorityTasksDueOnDate', 'SportsTrackerLongestDistanceActivity', 'SimpleCalendarNextEvent', 'MarkorDeleteAllNotes', 'SimpleCalendarAddRepeatingEvent', 'SimpleCalendarAddOneEventInTwoWeeks', 'SimpleCalendarAnyEventsOnDate'],
['RecipeDeleteDuplicateRecipes', 'SystemBrightnessMaxVerify', 'SportsTrackerActivitiesCountForWeek', 'SportsTrackerTotalDurationForCategoryThisWeek', 'AudioRecorderRecordAudio', 'SimpleCalendarEventOnDateAtTime', 'SportsTrackerActivityDuration', 'SystemWifiTurnOffVerify', 'VlcCreateTwoPlaylists', 'RecipeAddMultipleRecipesFromImage', 'NotesTodoItemCount', 'TasksIncompleteTasksOnDate', 'SimpleCalendarEventsInTimeRange', 'ExpenseAddMultipleFromMarkor', 'SystemCopyToClipboard', 'SystemBluetoothTurnOff', 'SportsTrackerActivitiesOnDate', 'SimpleCalendarEventsInNextWeek', 'ExpenseDeleteDuplicates', 'ExpenseDeleteMultiple2', 'SystemBluetoothTurnOffVerify', 'SimpleCalendarDeleteEvents', 'NotesRecipeIngredientCount', 'ClockStopWatchRunning', 'TasksHighPriorityTasks', 'OsmAndMarker', 'RecipeAddMultipleRecipesFromMarkor', 'RecipeDeleteSingleRecipe', 'MarkorCreateNote', 'SimpleDrawProCreateDrawing', 'ContactsNewContactDraft', 'ExpenseAddMultiple', 'SystemBluetoothTurnOn', 'RetroCreatePlaylist', 'SaveCopyOfReceiptTaskEval', 'MarkorCreateNoteAndSms', 'ExpenseDeleteDuplicates2', 'VlcCreatePlaylist', 'CameraTakePhoto'],]

_ADB_PATH = flags.DEFINE_string(
    'adb_path',
    
    '/usr/bin/adb',
    
    'Path to adb. Set if not installed through SDK.',
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    'perform_emulator_setup',
    # True,
    False,
    'Whether to perform emulator setup. This must be done once and only once'
    ' before running Android World. After an emulator is setup, this flag'
    ' should always be False.',
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port',
    5564,
    # 5556,
    'The console port of the running Android device. This can usually be'
    ' retrieved by looking at the output of `adb devices`. In general, the'
    ' first connected device is port 5554, the second is 5556, and'
    ' so on.',
)
_DEVICE_GRPC_PORT = flags.DEFINE_integer(
    'grpc_port',
    8564,
    # 8556,
    'grpc port',
)

_DEVICE_ADB_SERVER_PORT = flags.DEFINE_integer(
    'adb_server_port',
    5037,
    'adb server port',
)

_DEVICE_NAME = flags.DEFINE_string(
    'device_name',
    '',
    'device name',
)

_SUITE_FAMILY = flags.DEFINE_enum(
    'suite_family',
    registry.TaskRegistry.ANDROID_WORLD_FAMILY,
    [
        # Families from the paper.
        registry.TaskRegistry.ANDROID_WORLD_FAMILY,
        # registry.TaskRegistry.MINIWOB_FAMILY_SUBSET,
        # Other families for more testing.
        # registry.TaskRegistry.MINIWOB_FAMILY,
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
    # 3,
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


def _main() -> None:
  """Runs eval suite and gets rewards back."""
  from utils.logger import setup_logging
  # setup_logging()

  apk_root = config.apk_root
  adb_path = config.adb_path

  env = env_launcher.load_and_setup_env(
      console_port=_DEVICE_CONSOLE_PORT.value,
      emulator_setup=_EMULATOR_SETUP.value,
      adb_path=_ADB_PATH.value,
      grpc_port=_DEVICE_GRPC_PORT.value,
      adb_server_port=_DEVICE_ADB_SERVER_PORT.value,
      device_name=_DEVICE_NAME.value,
      freeze_datetime=True,
  )

  apk_dir = file_utils.convert_to_posix_path(
    f'{apk_root}/Calculator_9.0.apk',
    f'{apk_root}/com.google.android.apps.pdfviewer_2.19.381.03.80-193810380_minAPI16(x86_64).apk',
    f'{apk_root}/com.zell_mbc.medilog_5499.apk',
    f'{apk_root}/fitbook.apk',
    f'{apk_root}/net.youapps.calcyou_6.apk',
    f'{apk_root}/org.nsh07.pomodoro_10.apk',
    f'{apk_root}/zipxtract.apk',
    f'{apk_root}/ADBKeyboard.apk',
  )
  
  adb_utils.install_apk(apk_dir, env.controller)

  env.close()


def main(argv: Sequence[str]) -> None:
  if not _TASKS.value:  #
      splits_id = _SPLITS_ID.value
      if splits_id >= 0:
        task_split = splits[splits_id]
      else:
        # -1 for all tasks
        task_split = None
      flags.FLAGS.tasks = task_split  # 
  del argv
  _main()


if __name__ == '__main__':
  app.run(main)
