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

def load_and_analyze_results(output_path1, output_path2, output_path3, chars_data_path1=None, chars_data_path2=None, chars_data_path3=None):

    stability_subset = [
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
        "VagueMarkorDeleteNewestTwoNotes",
    ]
    
    stability_subset_instruction_variations = [
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
        "VagueMarkorDeleteNewestTwoNotesVariation",
    ]
    
    dimensions = ['Browsecomp', 'FuncAssist', 'GSA', 'GUIBrowsing', 'GUIM', 
                  'MultiRound', 'NoiseResist', 'Refusal', 'Vague']
    
    csv_paths = [
        os.path.join(output_path1, "result.csv"),
        os.path.join(output_path2, "result.csv"),
        os.path.join(output_path3, "result.csv"),
    ]
    
    path_names = ["Normal(base)", "DarkMode", "PadMode"]
    
    dfs = []
    for csv_path, name in zip(csv_paths, path_names):
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            dfs.append(df)
            print(f"✓ Loaded: {name} - {csv_path}")
        else:
            print(f"✗ Warning: {name} - {csv_path} not found")
    
    if not dfs:
        print("No CSV files found!")
        return
    
    print("\n" + "="*80)
    print("Part 1: Base Set Statistics (excluding instruction_variations)")
    print("="*80)
    
    df_base = dfs[0]
    
    df_base_filtered = df_base[~df_base['task_template'].isin(stability_subset_instruction_variations)]
    
    print(f"\nTotal number of tasks: {len(df_base_filtered)}")
    
    df_base_filtered = df_base_filtered.copy()
    df_base_filtered['success'] = df_base_filtered['mean_success_rate'].apply(
        lambda x: 1 if (pd.notna(x) and x > 0) else 0
    )
    
    overall_success_rate = df_base_filtered['success'].sum() / len(df_base_filtered) if len(df_base_filtered) > 0 else 0
    print(f"Overall success rate: {overall_success_rate:.2%}")
    
    print("\nStatistics by dimension:")
    dimension_stats = {}
    for dim in dimensions:
        df_dim = df_base_filtered[df_base_filtered['task_template'].str.startswith(dim)]
        if len(df_dim) > 0:
            success_rate = df_dim['success'].sum() / len(df_dim)
            dimension_stats[dim] = {
                'total': len(df_dim),
                'success': df_dim['success'].sum(),
                'success_rate': success_rate
            }
            print(f"  {dim:<20} {success_rate:>7.2%} ({int(df_dim['success'].sum())}/{len(df_dim)})")
    
    
    print("\n" + "="*80)
    print("Part 2: Stability Set Statistics (stability_subset - all 5 executions successful)")
    print("="*80)
    
    stability_results = defaultdict(lambda: {'results': [], 'dimension': None})
    
    for task in stability_subset:
        if task in df_base['task_template'].values:
            success = 1 if (pd.notna(df_base[df_base['task_template'] == task]['mean_success_rate'].iloc[0]) 
                          and df_base[df_base['task_template'] == task]['mean_success_rate'].iloc[0] > 0) else 0
            stability_results[task]['results'].append(('base', success))
            stability_results[task]['dimension'] = next((d for d in dimensions if task.startswith(d)), 'Other')
    
    for base_task in stability_subset:
        matching_variations = [v for v in stability_subset_instruction_variations if base_task in v]
        for var_task in matching_variations:
            if var_task in df_base['task_template'].values:
                success = 1 if (pd.notna(df_base[df_base['task_template'] == var_task]['mean_success_rate'].iloc[0])
                              and df_base[df_base['task_template'] == var_task]['mean_success_rate'].iloc[0] > 0) else 0
                stability_results[base_task]['results'].append((f'var_{var_task}', success))
    
    if len(dfs) > 1:
        for task in stability_subset:
            if task in dfs[1]['task_template'].values:
                success = 1 if (pd.notna(dfs[1][dfs[1]['task_template'] == task]['mean_success_rate'].iloc[0])
                              and dfs[1][dfs[1]['task_template'] == task]['mean_success_rate'].iloc[0] > 0) else 0
                stability_results[task]['results'].append(('darkmode', success))
    
    if len(dfs) > 2:
        for task in stability_subset:
            if task in dfs[2]['task_template'].values:
                success = 1 if (pd.notna(dfs[2][dfs[2]['task_template'] == task]['mean_success_rate'].iloc[0])
                              and dfs[2][dfs[2]['task_template'] == task]['mean_success_rate'].iloc[0] > 0) else 0
                stability_results[task]['results'].append(('padmode', success))
    
    all_passed = 0
    dimension_stability = defaultdict(lambda: {'total': 0, 'passed': 0})
    
    print(f"\n{'Task Name':<45} {'Attempts':<8} {'Successes':<8} {'Passed':<6}")
    print("-" * 75)
    
    for task in stability_subset:
        if task in stability_results:
            results = stability_results[task]['results']
            dim = stability_results[task]['dimension']
            
            total_attempts = len(results)
            successful_attempts = sum(1 for _, success in results if success == 1)
            all_success = successful_attempts == total_attempts and total_attempts >= 5
            
            if all_success:
                all_passed += 1
            
            dimension_stability[dim]['total'] += 1
            if all_success:
                dimension_stability[dim]['passed'] += 1
            
            status = "✓" if all_success else "✗"
            print(f"{task:<45} {total_attempts:<8} {successful_attempts:<8} {status:<6}")
    

    
    total_tasks = len(stability_subset)
    stability_pass_rate = all_passed / total_tasks if total_tasks > 0 else 0
    print("\n" + "="*80)
    print(f"Overall Stability Set Results: {all_passed}/{total_tasks} = {stability_pass_rate:.2%}")
    print("="*80)
    
    print("\n" + "="*80)
    print("Part 3: Character Count Statistics")
    print("="*80)
    
    chars_stats_by_mode = {}
    if chars_data_path1:
        chars_stats_by_mode["Normal(base)"] = analyze_chars_stats(load_chars_stats(chars_data_path1))
    
    for mode_name, chars_stats_all in chars_stats_by_mode.items():
        if chars_stats_all:
            print(f"\n{mode_name} mode character count statistics:")
            print(f"{'Dimension':<20} {'Average Chars':<15}")
            print("-" * 40)
            for dim in dimensions:
                chars = chars_stats_all.get(dim, 0)
                print(f"{dim:<20} {int(chars):<15}")
            total_chars = chars_stats_all.get('Total', 0)
            print(f"{'Total':<20} {int(total_chars):<15}")
    
    print("\n" + "="*80)
    print("Part 4: Results Summary")
    print("="*80)
    
    dimension_episode_length = {}
    for dim in dimensions:
        df_dim = df_base_filtered[df_base_filtered['task_template'].str.startswith(dim)]
        if len(df_dim) > 0:
            avg_episode_length = df_dim['mean_episode_length'].mean()
            dimension_episode_length[dim] = avg_episode_length
    
    summary_data = []
    
    header = ["Dimension", "Mean Success", "Mean Chars", "Chars Per Step"]
    
    for dim in dimensions:
        row = [dim]
        
        if dim in dimension_stats:
            part1_rate = dimension_stats[dim]['success_rate']
            row.append(f"{part1_rate:.2%}")
        else:
            row.append("N/A")
        
        if chars_stats_by_mode.get("Normal(base)"):
            part3_chars = chars_stats_by_mode["Normal(base)"].get(dim, 0)
            row.append(f"{int(part3_chars)}")
        else:
            row.append("N/A")
        
        if (chars_stats_by_mode.get("Normal(base)") and dim in dimension_episode_length 
            and dimension_episode_length[dim] > 0):
            part3_chars = chars_stats_by_mode["Normal(base)"].get(dim, 0)
            avg_chars_per_step = part3_chars / dimension_episode_length[dim]
            row.append(f"{avg_chars_per_step:.2f}")
        else:
            row.append("N/A")
        
        summary_data.append(row)
    
    total_row = ["Total"]
    total_row.append(f"{overall_success_rate:.2%}")
    
    if chars_stats_by_mode.get("Normal(base)"):
        total_chars = chars_stats_by_mode["Normal(base)"].get('Total', 0)
        total_row.append(f"{int(total_chars)}")
    else:
        total_row.append("N/A")
    
    if chars_stats_by_mode.get("Normal(base)"):
        total_chars = chars_stats_by_mode["Normal(base)"].get('Total', 0)
        total_avg_episode_length = df_base_filtered['mean_episode_length'].mean()
        if total_avg_episode_length > 0:
            total_avg_chars_per_step = total_chars / total_avg_episode_length
            total_row.append(f"{total_avg_chars_per_step:.2f}")
        else:
            total_row.append("N/A")
    else:
        total_row.append("N/A")
    
    summary_data.append(total_row)
    
    stability_row = ["Stability Pass"]
    stability_row.append(f"{stability_pass_rate:.2%}")
    stability_row.append("N/A")
    stability_row.append("N/A")
    summary_data.append(stability_row)
    
    stability_subset_accuracy = 0
    stability_subset_count = 0
    for task in stability_subset:
        if task in df_base['task_template'].values:
            success = 1 if (pd.notna(df_base[df_base['task_template'] == task]['mean_success_rate'].iloc[0])
                          and df_base[df_base['task_template'] == task]['mean_success_rate'].iloc[0] > 0) else 0
            stability_subset_accuracy += success
            stability_subset_count += 1
    
    if stability_subset_count > 0:
        stability_subset_accuracy_rate = stability_subset_accuracy / stability_subset_count
    else:
        stability_subset_accuracy_rate = 0
    
    stability_subset_row = ["Stability Subset Pass"]
    stability_subset_row.append(f"{stability_subset_accuracy_rate:.2%}")
    stability_subset_row.append("N/A")
    stability_subset_row.append("N/A")
    summary_data.append(stability_subset_row)
    
    print(f"\n{header[0]:<20} {header[1]:<20} {header[2]:<20} {header[3]:<20}")
    print("-" * 80)
    for row in summary_data:
        print(f"{row[0]:<20} {row[1]:<20} {row[2]:<20} {row[3]:<20}")
    
    return {
        'base_results': {
            'overall_success_rate': overall_success_rate,
            'dimension_stats': dimension_stats
        },
        'stability_results': {
            'total': total_tasks,
            'passed': all_passed,
            'pass_rate': stability_pass_rate,
            'dimension_stats': dict(dimension_stability)
        },
        'chars_stats': chars_stats_by_mode,
        'summary_data': summary_data
    }

if __name__ == "__main__":
    import sys
    import csv
    
    if len(sys.argv) >= 4:
        output_path1 = sys.argv[1]
        output_path2 = sys.argv[2]
        output_path3 = sys.argv[3]
    else:
        output_path1 = ""
        output_path2 = ""
        output_path3 = ""

    chars_data_path1 = output_path1 + "_data"
    chars_data_path2 = output_path2 + "_data"
    chars_data_path3 = output_path3 + "_data"
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(os.path.dirname(output_path1), f"stability_statistic_{timestamp}.log")
    
    tee = TeeOutput(log_file_path)
    original_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        print(f"Log file path: {log_file_path}")
        print(f"Execution time: {timestamp}")
        print("")
        
        results = load_and_analyze_results(output_path1, output_path2, output_path3, chars_data_path1, chars_data_path2, chars_data_path3)
        
        if results and 'summary_data' in results:
            csv_file_path = os.path.join(os.path.dirname(output_path1), f"stability_statistic_{timestamp}.csv")
            try:
                with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    header = ["Dimension", "Mean Success", "Mean Chars", "Chars Per Step"]
                    writer.writerow(header)
                    for row in results['summary_data']:
                        writer.writerow(row)
                
                print(f"\n✓ Results summary saved to CSV file: {csv_file_path}")
            except Exception as e:
                print(f"\n✗ Failed to save CSV file: {e}")
    
    finally:
        sys.stdout = original_stdout
        tee.close()
        print(f"✓ Log saved: {log_file_path}")
