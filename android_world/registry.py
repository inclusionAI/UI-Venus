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
# Changes: Adding task classed in VENUS_FAMILY in registry.py

"""Registers the task classes."""

import types
from typing import Any, Final
from android_world.env import interface

from android_world.task_evals import task_eval
from android_world.task_evals.composite import markor_sms
from android_world.task_evals.composite import system as system_composite
from android_world.task_evals.information_retrieval import information_retrieval
from android_world.task_evals.information_retrieval import information_retrieval_registry
from android_world.task_evals.miniwob import miniwob_registry
from android_world.task_evals.single import audio_recorder
from android_world.task_evals.single import browser
from android_world.task_evals.single import camera
from android_world.task_evals.single import clock
from android_world.task_evals.single import contacts
from android_world.task_evals.single import expense
from android_world.task_evals.single import files
from android_world.task_evals.single import markor
from android_world.task_evals.single import osmand
from android_world.task_evals.single import recipe
from android_world.task_evals.single import retro_music
from android_world.task_evals.single import simple_draw_pro
from android_world.task_evals.single import simple_gallery_pro
from android_world.task_evals.single import sms
from android_world.task_evals.single import system
from android_world.task_evals.single import vlc
from android_world.task_evals.single import tasksapp
from android_world.task_evals.single import calcyou
from android_world.task_evals.single import tomato
from android_world.task_evals.single import joplin
from android_world.task_evals.single import phone
from android_world.task_evals.single import fitbook
from android_world.task_evals.single import zipxtract


from android_world.task_evals.single.calendar import calendar


def get_information_retrieval_task_path() -> None:
  return None


def get_information_retrieval_task_names() -> list[str] | None:
  """Get the list of information retrieval task names to load.
  
  Returns None to load all tasks, or a list to load only specified tasks.
  Can be specified via environment variable INFORMATION_RETRIEVAL_TASKS,
  with multiple tasks separated by commas.
  Example: export INFORMATION_RETRIEVAL_TASKS="SimpleCalendarEventsOnDate,SimpleCalendarNextEvent"
  """
  import os
  task_names = os.environ.get('INFORMATION_RETRIEVAL_TASKS')
  if task_names:
    return [name.strip() for name in task_names.split(',')]
  return None


def get_families() -> list[str]:
  return [
      TaskRegistry.VENUS_FAMILY,
      TaskRegistry.ANDROID_WORLD_FAMILY,
      TaskRegistry.ANDROID_FAMILY,
      TaskRegistry.MINIWOB_FAMILY,
      TaskRegistry.MINIWOB_FAMILY_SUBSET,
      TaskRegistry.INFORMATION_RETRIEVAL_FAMILY,
  ]

_INFORMATION_RETRIEVAL_TASK = [
  "SimpleCalendarEventsOnDate",
  "SimpleCalendarNextEvent",
  "SimpleCalendarEventOnDateAtTime",
  "SimpleCalendarAnyEventsOnDate",
  "SimpleCalendarNextMeetingWithPerson",
  "SimpleCalendarLocationOfEvent",
  "SimpleCalendarEventsInNextWeek",
  "SimpleCalendarFirstEventAfterStartTime",
  "SimpleCalendarEventsInTimeRange",
  "TasksDueOnDate",
  "TasksHighPriorityTasks",
  "TasksHighPriorityTasksDueOnDate",
  "TasksDueNextWeek",
  "TasksCompletedTasksForDate",
  "TasksIncompleteTasksOnDate",
  "SportsTrackerActivitiesOnDate",
  "SportsTrackerActivitiesCountForWeek",
  "SportsTrackerActivityDuration",
  "SportsTrackerLongestDistanceActivity",
  "SportsTrackerTotalDurationForCategoryThisWeek",
  "SportsTrackerTotalDistanceForCategoryOverInterval",
  "NotesRecipeIngredientCount",
  "NotesMeetingAttendeeCount",
  "NotesIsTodo",
  "NotesTodoItemCount",
]

class TaskRegistry:
  """Registry of tasks."""


  VENUS_FAMILY: Final[str] = 'venus'  # .

  # The AndroidWorld family.
  ANDROID_WORLD_FAMILY: Final[str] = 'android_world'  # Entire suite.
  ANDROID_FAMILY: Final[str] = 'android'  # Subset.
  INFORMATION_RETRIEVAL_FAMILY: Final[str] = 'information_retrieval'  # Subset.

  # The MiniWoB family.
  MINIWOB_FAMILY: Final[str] = 'miniwob'
  MINIWOB_FAMILY_SUBSET: Final[str] = 'miniwob_subset'

  ANDROID_TASK_REGISTRY = {}
  VENUS_TASK_REGISTRY = {}
  INFORMATION_RETRIEVAL_TASK_REGISTRY = (
      information_retrieval_registry.InformationRetrievalRegistry[
          information_retrieval.InformationRetrieval
      ](filename=get_information_retrieval_task_path(), 
        task_names=get_information_retrieval_task_names()).registry
  )

  MINIWOB_TASK_REGISTRY = miniwob_registry.TASK_REGISTRY

  def get_registry(self, family: str) -> Any:
    """Gets the task registry for the given family.

    Args:
      family: The family.

    Returns:
      Task registry.

    Raises:
      ValueError: If provided family doesn't exist.
    """

    if family == self.ANDROID_WORLD_FAMILY:
      return {
          **self.ANDROID_TASK_REGISTRY,
          **self.INFORMATION_RETRIEVAL_TASK_REGISTRY,
      }
    elif family == self.ANDROID_FAMILY:
      return self.ANDROID_TASK_REGISTRY
    
    elif family == self.VENUS_FAMILY:
      return self.VENUS_TASK_REGISTRY

    else:
      raise ValueError(f'Unsupported family: {family}')


  _TASKS_VENUS = (
      # GUI STATE AWARE (GSA) - Timing
      audio_recorder.GSATimingAudioRecorderRecordAudioTime,
      audio_recorder.GSATimingAudioRecorderPauseAudioRecordingTime,
      clock.GSAClockStopWatchPausedatCertainTime,
      tomato.GSATimingTomatoLongBreakTime,
      # GUI STATE AWARE (GSA) - Tracking
      phone.GSATrackingPhoneCall,
      phone.GSATrackingWhitelistCallAnswer,
      phone.GSATrackingBlacklistBlockAndSendSms,
      sms.GSATrackingSimpleSmsNotifyDelivery,
      sms.GSATrackingSMSHandleIncomingKeywords,
      sms.GSATrackingSimpleSmsHandleFiveIncomingHard,

      # GUI Manipulation - Drawing
      simple_draw_pro.GUIMDrawA,
      simple_draw_pro.GUIMDrawRectangleHard,
      joplin.GUIMJoplinDrawCircle,
      joplin.GUIMJoplinDrawCircleAndRectangle,
      # GUI Manipulation - Editing
      simple_draw_pro.GUIMEraseObject1,
      simple_draw_pro.GUIMEraseObject2,
      simple_draw_pro.GUIMCircleObject1,
      simple_draw_pro.GUIMCircleObject2,
      # GUI Manipulation - Gallery
      simple_gallery_pro.GUIMChangePicture,

      # MultiRound
      expense.MultiRoundExpense1Add2Add3Delete,
      markor.MultiRoundMarkorCreateNoteReverse,
      markor_sms.MultiRoundMarkorCreateNoteAndSms,
      markor.MultiRoundMarkorCreateTwoNote,
      markor.MultiRoundMarkorCreateNoteHeading,
      markor.MultiRoundMarkorAppendCopyRename,
      recipe.MultiRoundRecipe1AddImageRecipes2AddMarkor3DeleteImage,
      markor.MultiRoundMarkorAdd4NotesDeleteNote,
      retro_music.MultiRoundRetroSavePlaylist,
      vlc.MultiRoundVlcCreateTwoPlaylistsReverse,

      # Refusal
      audio_recorder.RefusalAudioRecorderRecordAudioWithFileNameConflict1,
      audio_recorder.RefusalAudioRecorderRecordAudioWithFileNameConflict2,
      camera.RefusalCameraTakeVideoConflict1,
      camera.RefusalCameraTakeVideoConflict2,
      calendar.RefusalSimpleCalendarAddOneEventDateAmbigious1,
      calendar.RefusalSimpleCalendarAddOneEventInTwoWeeksConflict1,
      expense.RefusalExpenseDeleteMultipleConflictAll,
      expense.RefusalExpenseDeleteMultipleConflictMultiple,
      expense.RefusalExpenseDeleteMultipleConflictSingle,
      files.RefusalFilesDeleteFileConflict1,
      markor.RefusalMarkorDeleteMultipleNotesAmbigious1,
      markor.RefusalMarkorDeleteMultipleNotesAmbigious2,
      markor.RefusalMarkorDeleteMultipleNotesConflict1,
      recipe.RefusalRecipeAddSingleRecipeConflict1,
      recipe.RefusalRecipeDeleteMultipleRecipesWithConstraintAmbigious1,
      retro_music.RefusalRetroPlayingQueueConflict1,
      system.RefusalSystemBluetoothTurnOffAlreadyOff,
      system.RefusalSystemBluetoothTurnOnAlreadyOn,
      tasksapp.RefusalTasksAddOneTaskAmbigious1,
      tomato.RefusalTomotoSettingConflict1,
      tomato.RefusalTomotoSettingConflict2,
      zipxtract.RefusalZipxtractCreateFile,

      # GUI Functional Assistance
      expense.FuncAssistExpenseExplainAllFunctionality,
      expense.FuncAssistExpenseExplainOneFunctionality1,
      expense.FuncAssistExpenseLocateOneFunctionality1,
      fitbook.FuncAssistFitbookLocateOneFunctionality,
      markor.FuncAssistMarkorLocateOneFunctionality1,
      markor.FuncAssistMarkorExplainOneFunctionality1,
      markor.FuncAssistMarkorExplainOneFunctionality2,
      markor.FuncAssistMarkorExplainOneFunctionality3,
      recipe.FuncAssistRecipeExplainOneFunctionality1,
      recipe.FuncAssistRecipeLocateOneFunctionality1,
      system.FuncAssistLocateSystemInterfaceTaskEval2,
      system.FuncAssistJoblinExplainOneFunctionality1,
      tasksapp.FuncAssistLocateTasksInterface1,
      tasksapp.FuncAssistTasksExplainOneFunctionality1,
      vlc.FuncAssistVlcLocateOneFunctionality1,
      vlc.FuncAssistVlcExplainOneFunctionality1,
      markor.FuncAssistMarkorJoplinTasks,
      system.FuncAssistMedilogExplainAllFunctionality,
      system.FuncAssistOpenTracksExplainOneFunctionality,
      markor.FuncAssistRefusalMarkorExplainOneFunctionality,
      recipe.FuncAssistVagueRecipeExplainOneFunctionality,
      zipxtract.FuncAssistZipxtractExplainOneFunctionality,

      # Noise Resistance
      audio_recorder.NoiseResistAudioRecorderRecordAudioWithFilenameCall,
      calendar.NoiseResistSimpleCalendarAddOneEventTomorrowWithOrientation,
      calendar.NoiseResistSimpleCalendarAddRepeatingEventCallandAPPCollapse,
      contacts.NoiseResistContactsAddContactWithCall,
      expense.NoiseResistExpenseAddSingleADs,
      expense.NoiseResistExpenseAddSingleAPPCollapse,
      expense.NoiseResistExpenseAddSingleAPPNumb,
      expense.NoiseResistExpenseAddSingleWithCall,
      expense.NoiseResistExpenseAddSingleWithOrientation,
      expense.NoiseResistExpenseDeleteSingleWithOrientation,
      files.NoiseResistFilesMoveFileAPPCollapse,
      markor.NoiseResistMarkorCreateNoteWithOrientation,
      osmand.NoiseResistOsmAndMarkerAPPCollapse,
      retro_music.NoiseResistRetroPlayingQueueAPPCollapse,
      vlc.NoiseResistVlcCreatePlaylistWithOrientation,
      vlc.NoiseResistVlcCreateTwoPlaylistsWithCallandCollapse,

      # Browsecomp-like
      system.BrowsecompOpenAppTaskEvalUICompCreate1,
      system.BrowsecompOpenAppTaskEvalUICompCreate2,
      system.BrowsecompFindAppTaskEvalUI1,
      system.BrowsecompFindAppTaskEvalUI2,
      system.BrowsecompFindAppTaskEvalUI3,
      system.BrowsecompFindAppandAskInfo1,
      system.BrowsecompFindAppandRelatedAppandAskInfo1,
      vlc.BrowsecompVlcFindVlcAPP1,
      system.BrowsecompFindImage,
      system.BrowsecompFindVideo,

      # GUI Browsing
      fitbook.GUIBrowsingFitbookCalories,
      markor.GUIBrowsingMarkorFindFilesPath,
      markor.GUIBrowsingMarkorFindFilesPathHard,
      markor.GUIBrowsingMarkorFindCommonPackage,
      vlc.GUIBrowsingWatchVideo1,
      vlc.GUIBrowsingWatchVideo2,
      vlc.GUIBrowsingWatchVideo3,
      vlc.GUIBrowsingWatchVideo4,
      vlc.GUIBrowsingWatchVideo5,
      vlc.GUIBrowsingWatchVideo6,
      system.GUIBrowsingPaper1,
      system.GUIBrowsingPaper2,
      system.GUIBrowsingPaper3,
      system.GUIBrowsingAnswer1,
      system.GUIBrowsingAnswer2,
      system.GUIBrowsingPDF1,
      system.GUIBrowsingPDF2,
      system.GUIBrowsingOrder1,
      system.GUIBrowsingOrder2,
      system.GUIBrowsingOrder3,
      system.GUIBrowsingOnlineShopping1,
      system.GUIBrowsingOnlineShopping2,
      system.GUIBrowsingFindGameFromGalleryTaskEvalUI,
      system.GUIBrowsingFindImageInPaper1,
      system.GUIBrowsingFindImageInPaper2,
      system.GUIBrowsingFindPDF1,
      system.GUIBrowsingFindPDF2,
      browser.GUIBrowsingBrowserRandomButtons1,
      browser.GUIBrowsingBrowserRandomButtons2,
      browser.GUIBrowsingBrowserRandomButtons3,
      browser.GUIBrowsingBrowserRandomButtons4,
      browser.GUIBrowsingBrowserRandomButtons5,
      vlc.GUIBrowsingRefusalVlcWatchVideo,
      vlc.GUIBrowsingVagueVlcWatchVideo,

      # Vague Instruction
      system.VagueBluetoothTurnOn1,
      system.VagueBluetoothTurnOn2,
      calcyou.VagueCalculator,
      system.VagueCalendarDateOffset,
      calcyou.VagueCurrencyExchange,
      expense.VagueDailyExpenseRecordTask,
      system.VagueFindPhoneTaskEvalUI,
      fitbook.VagueFoodCalorieCheck,
      calcyou.VagueGraphFunction,
      markor.VagueMarkorCreateNotewithCalculation,
      markor.VagueMarkorDeleteNewestTwoNotes,
      system.VagueSaveBatteryWithoutLosingInternet,
      system.VagueSystemBrightnessMax,
      system.VagueWatchLocalVideo,
      system.VagueZenModeTaskEvalUI,
      zipxtract.VagueZipxtractExtractFile,

      # Stability Tests - GSA
      audio_recorder.GSATimingAudioRecorderRecordAudioTimeCHS,
      audio_recorder.GSATimingAudioRecorderRecordAudioTimeENGVariation,
      phone.GSATrackingPhoneCallCHS,
      phone.GSATrackingPhoneCallENGVariation,
      # Stability Tests - GUIM
      simple_draw_pro.GUIMDrawACHS,
      simple_draw_pro.GUIMDrawAENGVariation,
      simple_draw_pro.GUIMEraseObject1CHS,
      simple_draw_pro.GUIMEraseObject1ENGVariation,
      # Stability Tests - Refusal
      expense.RefusalExpenseDeleteMultipleConflictMultipleCHS,
      expense.RefusalExpenseDeleteMultipleConflictMultipleVariation,
      system.RefusalSystemBluetoothTurnOnAlreadyOnCHS,
      system.RefusalSystemBluetoothTurnOnAlreadyOnVariation,
      markor.RefusalMarkorDeleteMultipleNotesAmbigious1CHS,
      markor.RefusalMarkorDeleteMultipleNotesAmbigious1Variation,
      # Stability Tests - FuncAssist
      expense.FuncAssistExpenseExplainOneFunctionality1CHS,
      expense.FuncAssistExpenseExplainOneFunctionality1Variation,
      vlc.FuncAssistVlcLocateOneFunctionality1CHS,
      vlc.FuncAssistVlcLocateOneFunctionality1Variation,
      # Stability Tests - GUIBrowsing
      browser.GUIBrowsingBrowserRandomButtons2CHS,
      browser.GUIBrowsingBrowserRandomButtons2Variation,
      system.GUIBrowsingPaper1CHS,
      system.GUIBrowsingPaper1Variation,
      system.GUIBrowsingOrder2CHS,
      system.GUIBrowsingOrder2Variation,
      # Stability Tests - Browsecomp
      system.BrowsecompFindAppTaskEvalUI1CHS,
      system.BrowsecompFindAppTaskEvalUI1Variation,
      vlc.BrowsecompVlcFindVlcAPP1CHS,
      vlc.BrowsecompVlcFindVlcAPP1Variation,
      # Stability Tests - NoiseResist
      expense.NoiseResistExpenseDeleteSingleWithOrientationCHS,
      expense.NoiseResistExpenseDeleteSingleWithOrientationVariation,
      files.NoiseResistFilesMoveFileAPPCollapseCHS,
      files.NoiseResistFilesMoveFileAPPCollapseVariation,
      # Stability Tests - MultiRound
      markor.MultiRoundMarkorAppendCopyRenameCHS,
      markor.MultiRoundMarkorAppendCopyRenameVariation,
      retro_music.MultiRoundRetroSavePlaylistCHS,
      retro_music.MultiRoundRetroSavePlaylistVariation,
      # Stability Tests - Vague
      system.VagueSystemBrightnessMaxCHS,
      system.VagueSystemBrightnessMaxVariation,
      markor.VagueMarkorDeleteNewestTwoNotesCHS,
      markor.VagueMarkorDeleteNewestTwoNotesVariation,
  )


  def register_task(
      self, task_registry: dict[Any, Any], task_class: type[task_eval.TaskEval]
  ) -> None:
    """Registers the task class.

    Args:
      task_registry: The registry to register the task in.
      task_class: The class to register.
    """
    task_registry[task_class.__name__] = task_class

  def __init__(self):
    for task in self._TASKS_VENUS:
      self.register_task(self.VENUS_TASK_REGISTRY, task)

  # Add names with "." notation for autocomplete in Colab.
  names = types.SimpleNamespace(**{
      k: k
      for k in {
          **VENUS_TASK_REGISTRY,
      }
  })
