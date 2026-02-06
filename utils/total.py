import pandas as pd
import os
import sys
import json
from collections import defaultdict
from datetime import datetime


class TeeOutput:
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()


def load_chars_stats(data_path):
    chars_stats = {}
    
    if not os.path.exists(data_path):
        print(f"✗ Warning: {data_path} not found")
        return chars_stats
    
    dimensions = ['Browsecomp', 'FuncAssist', 'GSA', 'GUIBrowsing', 'GUIM', 
                  'MultiRound', 'NoiseResist', 'Refusal', 'Vague']
    
    for folder_name in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        chars_file = os.path.join(folder_path, "000000_chars_stats.json")
        if os.path.exists(chars_file):
            try:
                with open(chars_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                task_name = folder_name
                if '_' in task_name:
                    parts = task_name.rsplit('_', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        task_name = parts[0]
                
                dimension = next((d for d in dimensions if task_name.startswith(d)), 'Other')
                
                chars_stats[task_name] = {
                    'total_chars': data.get('total_chars', 0),
                    'dimension': dimension,
                    'folder_name': folder_name
                }
            except Exception as e:
                print(f"  Warning: Failed to load {chars_file}: {e}")
    
    return chars_stats


def analyze_chars_stats(chars_stats):
    if not chars_stats:
        return {}
    
    dimensions = ['Browsecomp', 'FuncAssist', 'GSA', 'GUIBrowsing', 'GUIM', 
                  'MultiRound', 'NoiseResist', 'Refusal', 'Vague']
    
    dimension_stats = {}
    for dim in dimensions:
        dim_tasks = {k: v for k, v in chars_stats.items() if v['dimension'] == dim}
        if dim_tasks:
            dim_chars = [v['total_chars'] for v in dim_tasks.values()]
            dim_avg = sum(dim_chars) / len(dim_chars)
            dimension_stats[dim] = dim_avg
    
    all_chars = [v['total_chars'] for v in chars_stats.values()]
    avg_chars = sum(all_chars) / len(all_chars) if all_chars else 0
    dimension_stats['Total'] = avg_chars
    
    return dimension_stats


def get_stability_subset():
    """Return stability subset list"""
    return [
        "GSATimingAudioRecorderRecordAudioTime",
        "GSATrackingPhoneCall",
        "GUIMDrawA",
        "GUIMEraseObject1",
        "RefusalExpenseDeleteMultipleConflictMultiple",
        "RefusalSystemBluetoothTurnOnAlreadyOn",
        "RefusalMarkorDeleteMultipleNotesAmbigious1",
        "FuncAssistExpenseExplainOneFunctionality1",
        "FuncAssistVlcLocateOneFunctionality1",
        "GUIBrowsingBrowserRandomButtons2",
        "GUIBrowsingPaper1",
        "GUIBrowsingOrder2",
        "BrowsecompFindAppTaskEvalUI1",
        "BrowsecompVlcFindVlcAPP1",
        "NoiseResistExpenseDeleteSingleWithOrientation",
        "NoiseResistFilesMoveFileAPPCollapse",
        "MultiRoundMarkorAppendCopyRename",
        "MultiRoundRetroSavePlaylist",
        "VagueSystemBrightnessMax",
        "VagueMarkorDeleteMultipleNotes",
    ]


def get_stability_variations():
    """Return stability variations list"""
    return [
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
        "VagueMarkorDeleteMultipleNotesCHS",
        "VagueMarkorDeleteMultipleNotesVariation",
    ]


def analyze_single_mode(csv_path, other_csv_paths, mode_name):
    
    stability_subset = get_stability_subset()
    
    stability_subset_instruction_variations = get_stability_variations()
    
    dimensions = ['Browsecomp', 'FuncAssist', 'GSA', 'GUIBrowsing', 'GUIM', 
                  'MultiRound', 'NoiseResist', 'Refusal', 'Vague']
    
    if not os.path.exists(csv_path):
        print(f"✗ Warning: {mode_name} - {csv_path} not found")
        return None
    
    df_current = pd.read_csv(csv_path)
    
    df_filtered = df_current[~df_current['task_template'].isin(stability_subset_instruction_variations)]
    
    df_filtered = df_filtered.copy()
    df_filtered['success'] = df_filtered['mean_success_rate'].apply(
        lambda x: 1 if (pd.notna(x) and x > 0) else 0
    )
    
    avg_success_rate_stats = {}
    total_success = 0
    total_tasks = 0
    for dim in dimensions:
        df_dim = df_filtered[df_filtered['task_template'].str.startswith(dim)]
        if len(df_dim) > 0:
            success_rate = df_dim['success'].sum() / len(df_dim)
            avg_success_rate_stats[dim] = success_rate
            total_success += df_dim['success'].sum()
            total_tasks += len(df_dim)
    
    overall_success_rate = total_success / total_tasks if total_tasks > 0 else 0
    avg_success_rate_stats['Overall'] = overall_success_rate
    
    return {
        'dimensions': dimensions,
        'avg_success_rate_stats': avg_success_rate_stats,
        'df': df_current  
    }


def load_and_analyze_results(output_path1, output_path2, output_path3):
    
    csv_paths = [
        os.path.join(output_path1, "result.csv"),
        os.path.join(output_path2, "result.csv"),
        os.path.join(output_path3, "result.csv"),
    ]
    
    mode_names = ["Normal", "DarkMode", "PadMode"]
    
    results_all = {}
    
    for csv_path, mode_name in zip(csv_paths, mode_names):
        result = analyze_single_mode(csv_path, csv_paths, mode_name)
        if result:
            results_all[mode_name] = result
    
    return results_all

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 2:
        base_path = sys.argv[1]
    else:
        base_path = ""
    
    output_path1 = base_path + "_normal"
    output_path2 = base_path + "_darkmode"
    output_path3 = base_path + "_padmode"
    
    chars_data_path1 = output_path1 + "_data"
    chars_data_path2 = output_path2 + "_data"
    chars_data_path3 = output_path3 + "_data"
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(os.path.dirname(base_path), f"stability_statistic_{timestamp}.log")
    
    tee = TeeOutput(log_file_path)
    original_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        print(f"Log file path: {log_file_path}")
        print(f"Execution time: {timestamp}")
        print(f"Base path: {base_path}")
        print("")
        
        results_all = load_and_analyze_results(output_path1, output_path2, output_path3)
        
        if results_all:
            csv_path1 = os.path.join(output_path1, "result.csv")
            csv_path2 = os.path.join(output_path2, "result.csv")
            csv_path3 = os.path.join(output_path3, "result.csv")
            
            dfs = {}
            for csv_path, mode_name in [(csv_path1, "Normal"), (csv_path2, "DarkMode"), (csv_path3, "PadMode")]:
                if os.path.exists(csv_path):
                    dfs[mode_name] = pd.read_csv(csv_path)
            
            stability_subset = get_stability_subset()
            stability_variations = get_stability_variations()
            dimensions = ['Browsecomp', 'FuncAssist', 'GSA', 'GUIBrowsing', 'GUIM', 
                         'MultiRound', 'NoiseResist', 'Refusal', 'Vague']
            
            stability_results_combined = defaultdict(lambda: {'results': [], 'dimension': None})
            
            for mode_name, df in dfs.items():
                for task in stability_subset:
                    if task in df['task_template'].values:
                        success = 1 if (pd.notna(df[df['task_template'] == task]['mean_success_rate'].iloc[0]) 
                                      and df[df['task_template'] == task]['mean_success_rate'].iloc[0] > 0) else 0
                        stability_results_combined[task]['results'].append(success)
                        if stability_results_combined[task]['dimension'] is None:
                            dim = next((d for d in dimensions if task.startswith(d)), 'Other')
                            stability_results_combined[task]['dimension'] = dim
                    
                    # variations
                    matching_variations = [v for v in stability_variations if task in v]
                    for var_task in matching_variations:
                        if var_task in df['task_template'].values:
                            success = 1 if (pd.notna(df[df['task_template'] == var_task]['mean_success_rate'].iloc[0])
                                          and df[df['task_template'] == var_task]['mean_success_rate'].iloc[0] > 0) else 0
                            stability_results_combined[task]['results'].append(success)
            
            stability_pass_rate_stats_combined = {}
            dimension_stability = defaultdict(lambda: {'total': 0, 'passed': 0})
            
            for task in stability_subset:
                if task in stability_results_combined:
                    results = stability_results_combined[task]['results']
                    dim = stability_results_combined[task]['dimension']
                    
                    total_attempts = len(results)
                    successful_attempts = sum(results)
                    
                    is_passed = (successful_attempts == total_attempts and total_attempts == 5)
                    
                    dimension_stability[dim]['total'] += 1
                    if is_passed:
                        dimension_stability[dim]['passed'] += 1
            
            for dim in dimensions:
                if dim in dimension_stability:
                    stats = dimension_stability[dim]
                    pass_rate = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
                    stability_pass_rate_stats_combined[dim] = pass_rate
                else:
                    stability_pass_rate_stats_combined[dim] = 0
            
            for mode_name in results_all:
                results_all[mode_name]['stability_pass_rate_stats'] = stability_pass_rate_stats_combined
            
            chars_stats_by_mode = {
                "Normal": analyze_chars_stats(load_chars_stats(chars_data_path1)),
                "DarkMode": analyze_chars_stats(load_chars_stats(chars_data_path2)),
                "PadMode": analyze_chars_stats(load_chars_stats(chars_data_path3))
            }
            
            
            mode_names = ["Normal", "DarkMode", "PadMode"]
            
            all_tables = {}
            
            for mode_name in mode_names:
                if mode_name not in results_all:
                    print(f"⊘ Skipping {mode_name}: Data not available")
                    continue
                
                results = results_all[mode_name]
                chars_stats_all = chars_stats_by_mode.get(mode_name, {})
                
                print("\n" + "="*100)
                print(f"Statistical Results Summary - {mode_name} Mode")
                print("="*100)
                
                table_data = []
                
                # Mean Success Rate row
                row1 = ["Mean Success Rate"]
                rates1 = []
                for dim in dimensions:
                    rate = results['avg_success_rate_stats'].get(dim, 0)
                    row1.append(f"{rate:.2%}")
                    rates1.append(rate)
                # Overall rate
                overall_rate = results['avg_success_rate_stats'].get('Overall', 0)
                row1.append(f"{overall_rate:.2%}")
                table_data.append(row1)
                
                # Stability Pass Rate row
                row2 = ["Stability Pass Rate"]
                rates2 = []
                for dim in dimensions:
                    rate = results['stability_pass_rate_stats'].get(dim, 0)
                    row2.append(f"{rate:.2%}")
                    rates2.append(rate)
                # Average rate
                avg_rate2 = sum(rates2) / len(rates2) if rates2 else 0
                row2.append(f"{avg_rate2:.2%}")
                table_data.append(row2)
                
                row3 = ["Mean Chars"]
                chars_list = []
                for dim in dimensions:
                    chars = chars_stats_all.get(dim, 0)
                    row3.append(f"{int(chars)}")
                    chars_list.append(chars)
                # Average chars
                avg_chars = chars_stats_all.get('Total', sum(chars_list) / len(chars_list) if chars_list else 0)
                row3.append(f"{int(avg_chars)}")
                table_data.append(row3)
                
                # Print table header
                print(f"\n{'Dimension':<15}", end="")
                for dim in dimensions:
                    print(f"{dim:<12}", end="")
                print(f"{'Total':<12}")
                print("-" * (15 + 12 * (len(dimensions) + 1)))
                
                for row in table_data:
                    print(f"{row[0]:<15}", end="")
                    for i in range(1, len(row)):
                        print(f"{row[i]:<12}", end="")
                    print()
                
                print("\n" + "="*100)
                
                all_tables[mode_name] = table_data
            
            # Save to CSV
            csv_file_path = os.path.join(os.path.dirname(base_path), f"stability_statistic_{timestamp}.csv")
            try:
                import csv
                with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    for idx, mode_name in enumerate(mode_names):
                        if mode_name not in all_tables:
                            continue
                        
                        if idx > 0:
                            writer.writerow([])
                        
                        writer.writerow([f"{mode_name} Mode"])
                        
                        header = ["Metric"] + dimensions + ["Total"]
                        writer.writerow(header)
                        
                        for row in all_tables[mode_name]:
                            writer.writerow(row)
                
                print(f"✓ All result tables saved to CSV file: {csv_file_path}")
            except Exception as e:
                print(f"✗ Failed to save CSV file: {e}")
        
    finally:
        sys.stdout = original_stdout
        tee.close()
        print(f"✓ Log: {log_file_path}")

