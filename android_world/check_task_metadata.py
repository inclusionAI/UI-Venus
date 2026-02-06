import json

class_names = [
    "GSATimingAudioRecorderRecordAudioTime",
    "GSATimingAudioRecorderPauseAudioRecordingTime",
    "GSAClockStopWatchPausedatCertainTime",
    "GSATimingTomatoLongBreakTime",
    "GSATrackingPhoneCall",
    "GSATrackingWhitelistCallAnswer",
    "GSATrackingBlacklistBlockAndSendSms",
    "GSATrackingSimpleSmsNotifyDelivery",
    "GSATrackingSMSHandleIncomingKeywords",
    "GSATrackingSimpleSmsHandleFiveIncomingHard",
    "GUIMDrawA",
    "GUIMDrawRectangleHard",
    "GUIMJoplinDrawCircle",
    "GUIMJoplinDrawCircleAndRectangle",
    "GUIMEraseObject1",
    "GUIMEraseObject2",
    "GUIMCircleObject1",
    "GUIMCircleObject2",
    "GUIMChangePicture",
    "MultiRoundExpense1Add2Add3Delete",
    "MultiRoundMarkorCreateNoteReverse",
    "MultiRoundMarkorCreateNoteAndSms",
    "MultiRoundMarkorCreateTwoNote",
    "MultiRoundMarkorCreateNoteHeading",
    "MultiRoundMarkorAppendCopyRename",
    "MultiRoundRecipe1AddImageRecipes2AddMarkor3DeleteImage",
    "MultiRoundMarkorAdd4NotesDeleteNote",
    "MultiRoundRetroSavePlaylist",
    "MultiRoundVlcCreateTwoPlaylistsReverse",
    "RefusalAudioRecorderRecordAudioWithFileNameConflict1",
    "RefusalAudioRecorderRecordAudioWithFileNameConflict2",
    "RefusalCameraTakeVideoConflict1",
    "RefusalCameraTakeVideoConflict2",
    "RefusalSimpleCalendarAddOneEventDateAmbigious1",
    "RefusalSimpleCalendarAddOneEventInTwoWeeksConflict1",
    "RefusalExpenseDeleteMultipleConflictAll",
    "RefusalExpenseDeleteMultipleConflictMultiple",
    "RefusalExpenseDeleteMultipleConflictSingle",
    "RefusalFilesDeleteFileConflict1",
    "RefusalMarkorDeleteMultipleNotesAmbigious1",
    "RefusalMarkorDeleteMultipleNotesAmbigious2",
    "RefusalMarkorDeleteMultipleNotesConflict1",
    "RefusalRecipeAddSingleRecipeConflict1",
    "RefusalRecipeDeleteMultipleRecipesWithConstraintAmbigious1",
    "RefusalRetroPlayingQueueConflict1",
    "RefusalSystemBluetoothTurnOffAlreadyOff",
    "RefusalSystemBluetoothTurnOnAlreadyOn",
    "RefusalTasksAddOneTaskAmbigious1",
    "RefusalTomotoSettingConflict1",
    "RefusalTomotoSettingConflict2",
    "RefusalZipxtractCreateFile",
    "FuncAssistExpenseExplainAllFunctionality",
    "FuncAssistExpenseExplainOneFunctionality1",
    "FuncAssistExpenseLocateOneFunctionality1",
    "FuncAssistFitbookLocateOneFunctionality",
    "FuncAssistMarkorLocateOneFunctionality1",
    "FuncAssistMarkorExplainOneFunctionality1",
    "FuncAssistMarkorExplainOneFunctionality2",
    "FuncAssistMarkorExplainOneFunctionality3",
    "FuncAssistRecipeExplainOneFunctionality1",
    "FuncAssistRecipeLocateOneFunctionality1",
    "FuncAssistLocateSystemInterfaceTaskEval2",
    "FuncAssistJoblinExplainOneFunctionality1",
    "FuncAssistLocateTasksInterface1",
    "FuncAssistTasksExplainOneFunctionality1",
    "FuncAssistVlcLocateOneFunctionality1",
    "FuncAssistVlcExplainOneFunctionality1",
    "FuncAssistMarkorJoplinTasks",
    "FuncAssistMedilogExplainAllFunctionality",
    "FuncAssistOpenTracksExplainOneFunctionality",
    "FuncAssistRefusalMarkorExplainOneFunctionality",
    "FuncAssistVagueRecipeExplainOneFunctionality",
    "FuncAssistZipxtractExplainOneFunctionality",
    "NoiseResistAudioRecorderRecordAudioWithFilenameCall",
    "NoiseResistSimpleCalendarAddOneEventTomorrowWithOrientation",
    "NoiseResistSimpleCalendarAddRepeatingEventCallandAPPCollapse",
    "NoiseResistContactsAddContactWithCall",
    "NoiseResistExpenseAddSingleADs",
    "NoiseResistExpenseAddSingleAPPCollapse",
    "NoiseResistExpenseAddSingleAPPNumb",
    "NoiseResistExpenseAddSingleWithCall",
    "NoiseResistExpenseAddSingleWithOrientation",
    "NoiseResistExpenseDeleteSingleWithOrientation",
    "NoiseResistFilesMoveFileAPPCollapse",
    "NoiseResistMarkorCreateNoteWithOrientation",
    "NoiseResistOsmAndMarkerAPPCollapse",
    "NoiseResistRetroPlayingQueueAPPCollapse",
    "NoiseResistVlcCreatePlaylistWithOrientation",
    "NoiseResistVlcCreateTwoPlaylistsWithCallandCollapse",
    "BrowsecompOpenAppTaskEvalUICompCreate1",
    "BrowsecompOpenAppTaskEvalUICompCreate2",
    "BrowsecompFindAppTaskEvalUI1",
    "BrowsecompFindAppTaskEvalUI2",
    "BrowsecompFindAppTaskEvalUI3",
    "BrowsecompFindAppandAskInfo1",
    "BrowsecompFindAppandRelatedAppandAskInfo1",
    "BrowsecompVlcFindVlcAPP1",
    "BrowsecompFindImage",
    "BrowsecompFindVideo",
    "GUIBrowsingFitbookCalories",
    "GUIBrowsingMarkorFindFilesPath",
    "GUIBrowsingMarkorFindFilesPathHard",
    "GUIBrowsingMarkorFindCommonPackage",
    "GUIBrowsingWatchVideo1",
    "GUIBrowsingWatchVideo2",
    "GUIBrowsingWatchVideo3",
    "GUIBrowsingWatchVideo4",
    "GUIBrowsingWatchVideo5",
    "GUIBrowsingWatchVideo6",
    "GUIBrowsingPaper1",
    "GUIBrowsingPaper2",
    "GUIBrowsingPaper3",
    "GUIBrowsingAnswer1",
    "GUIBrowsingAnswer2",
    "GUIBrowsingPDF1",
    "GUIBrowsingPDF2",
    "GUIBrowsingOrder1",
    "GUIBrowsingOrder2",
    "GUIBrowsingOrder3",
    "GUIBrowsingOnlineShopping1",
    "GUIBrowsingOnlineShopping2",
    "GUIBrowsingFindGameFromGalleryTaskEvalUI",
    "GUIBrowsingFindImageInPaper1",
    "GUIBrowsingFindImageInPaper2",
    "GUIBrowsingFindPDF1",
    "GUIBrowsingFindPDF2",
    "GUIBrowsingBrowserRandomButtons1",
    "GUIBrowsingBrowserRandomButtons2",
    "GUIBrowsingBrowserRandomButtons3",
    "GUIBrowsingBrowserRandomButtons4",
    "GUIBrowsingBrowserRandomButtons5",
    "GUIBrowsingRefusalVlcWatchVideo",
    "GUIBrowsingVagueVlcWatchVideo",
    "VagueBluetoothTurnOn1",
    "VagueBluetoothTurnOn2",
    "VagueCalculator",
    "VagueCalendarDateOffset",
    "VagueCurrencyExchange",
    "VagueDailyExpenseRecordTask",
    "VagueFindPhoneTaskEvalUI",
    "VagueFoodCalorieCheck",
    "VagueGraphFunction",
    "VagueMarkorCreateNotewithCalculation",
    "VagueMarkorDeleteNewestTwoNotes",
    "VagueSaveBatteryWithoutLosingInternet",
    "VagueSystemBrightnessMax",
    "VagueWatchLocalVideo",
    "VagueZenModeTaskEvalUI",
    "VagueZipxtractExtractFile"
]

stability_class_names = [
    "GSATimingAudioRecorderRecordAudioTimeCHS",
    "GSATimingAudioRecorderRecordAudioTimeENGVariation",
    "GSATrackingPhoneCallCHS",
    "GSATrackingPhoneCallENGVariation",
    "GUIMDrawACHS",
    "GUIMDrawAENGVariation",
    "GUIMEraseObject1CHS",
    "GUIMEraseObject1ENGVariation",
    "RefusalExpenseDeleteMultipleConflictMultipleCHS",
    "RefusalExpenseDeleteMultipleConflictMultipleVariation",
    "RefusalSystemBluetoothTurnOnAlreadyOnCHS",
    "RefusalSystemBluetoothTurnOnAlreadyOnVariation",
    "RefusalMarkorDeleteMultipleNotesAmbigious1CHS",
    "RefusalMarkorDeleteMultipleNotesAmbigious1Variation",
    "FuncAssistExpenseExplainOneFunctionality1CHS",
    "FuncAssistExpenseExplainOneFunctionality1Variation",
    "FuncAssistVlcLocateOneFunctionality1CHS",
    "FuncAssistVlcLocateOneFunctionality1Variation",
    "GUIBrowsingBrowserRandomButtons2CHS",
    "GUIBrowsingBrowserRandomButtons2Variation",
    "GUIBrowsingPaper1CHS",
    "GUIBrowsingPaper1Variation",
    "GUIBrowsingOrder2CHS",
    "GUIBrowsingOrder2Variation",
    "BrowsecompFindAppTaskEvalUI1CHS",
    "BrowsecompFindAppTaskEvalUI1Variation",
    "BrowsecompVlcFindVlcAPP1CHS",
    "BrowsecompVlcFindVlcAPP1Variation",
    "NoiseResistExpenseDeleteSingleWithOrientationCHS",
    "NoiseResistExpenseDeleteSingleWithOrientationVariation",
    "NoiseResistFilesMoveFileAPPCollapseCHS",
    "NoiseResistFilesMoveFileAPPCollapseVariation",
    "MultiRoundMarkorAppendCopyRenameCHS",
    "MultiRoundMarkorAppendCopyRenameVariation",
    "MultiRoundRetroSavePlaylistCHS",
    "MultiRoundRetroSavePlaylistVariation",
    "VagueSystemBrightnessMaxCHS",
    "VagueSystemBrightnessMaxVariation",
    "VagueMarkorDeleteNewestTwoNotesCHS",
    "VagueMarkorDeleteNewestTwoNotesVariation"
]


# 合并所有需要检查的class names
all_class_names = class_names + stability_class_names

# 读取JSON文件
with open('', 'r', encoding='utf-8') as f:
    json_data = json.load(f)



# 提取JSON中的所有task_name
json_task_names_dict = {item['task_name']: item for item in json_data}
json_task_names_set = set(json_task_names_dict.keys())

# 转换class_names为set
class_names_set = set(all_class_names)

# 找出缺失和多余的
missing_in_json = class_names_set - json_task_names_set
extra_in_json = json_task_names_set - class_names_set

print(f"需要删除的多余项: {len(extra_in_json)}个")
print(f"需要添加的缺失项: {len(missing_in_json)}个")
print()

# 创建新的JSON数据
new_json_data = []

# 保留所有在class_names中的现有项
for class_name in all_class_names:
    if class_name in json_task_names_dict:
        # 如果已存在，保留原有数据
        new_json_data.append(json_task_names_dict[class_name])
    else:
        # 如果缺失，创建新的空白模板
        new_item = {
            "task_name": class_name,
            "app": [],
            "evaluation_method": "",
            "useExternalFile": False,
            "tags": [],
            "difficulty": "",
            "task_template": "",
            "optimal_steps": ""
        }
        new_json_data.append(new_item)
        print(f"✅ 添加缺失项: {class_name}")

# 打印被删除的项
if extra_in_json:
    print("\n❌ 删除的多余项:")
    for name in sorted(extra_in_json):
        print(f"  - {name}")

# 保存修改后的JSON
with open('updated_file.json', 'w', encoding='utf-8') as f:
    json.dump(new_json_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ 处理完成!")
print(f"原始数据: {len(json_data)}项")
print(f"新数据: {len(new_json_data)}项")
print(f"保存到: updated_file.json")