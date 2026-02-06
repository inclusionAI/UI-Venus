#!/usr/bin/env python3


import os
import sys
import argparse
import pandas as pd
import gzip
import pickle
import json
from pathlib import Path
from collections import defaultdict


def count_text_chars(text):
    if text is None:
        return 0
    return len(str(text))


def extract_chars_from_pkl(pkl_file_path):
    try:
        with gzip.open(pkl_file_path, "rb") as f:
            data = pickle.load(f)
        
        episode_length = data[0].get('episode_length')
        if not isinstance(episode_length, int):
            return None
        
        episodes = data[0].get('episode_data', {})
        
        total_chars_stats = {
            'thinking_chars': 0,
            'tool_call_chars': 0,
            'conclusion_chars': 0,
            'other_chars': 0,
            'total_chars': 0,
        }
        
        output_keys = {
            'thinking': ['action_output', 'action_response'],
            'tool_call': ['action_output_json', 'dummy_action'],
            'conclusion': ['summary'],
        }
        
        for i in range(episode_length):
            for key in output_keys['thinking']:
                if key in episodes and i < len(episodes[key]):
                    thinking = episodes[key][i]
                    if thinking is not None:
                        total_chars_stats['thinking_chars'] += count_text_chars(thinking)
            
            for key in output_keys['tool_call']:
                if key in episodes and i < len(episodes[key]):
                    tool_call = episodes[key][i]
                    if tool_call is not None:
                        total_chars_stats['tool_call_chars'] += count_text_chars(tool_call)
            
            for key in output_keys['conclusion']:
                if key in episodes and i < len(episodes[key]):
                    conclusion = episodes[key][i]
                    if conclusion is not None:
                        total_chars_stats['conclusion_chars'] += count_text_chars(conclusion)
        
        total_chars_stats['total_chars'] = (
            total_chars_stats['thinking_chars'] + 
            total_chars_stats['tool_call_chars'] + 
            total_chars_stats['conclusion_chars'] + 
            total_chars_stats['other_chars']
        )
        
        return total_chars_stats
    except Exception as e:
        print(f"Error processing {pkl_file_path}: {e}")
        return None


def find_pkl_files(output_path):
    task_char_stats = defaultdict(list)
    
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith(".pkl.gz"):
                pkl_path = os.path.join(root, file)
                chars_stats = extract_chars_from_pkl(pkl_path)
                
                if chars_stats:
                    task_name = file.replace('.pkl.gz', '')
                    if '_' in task_name:
                        parts = task_name.rsplit('_', 1)
                        if len(parts) == 2:
                            task_name = parts[0]
                    
                    task_char_stats[task_name].append(chars_stats)
    
    return task_char_stats


def aggregate_char_stats(task_char_stats):
    aggregated = {}
    
    for task_name, stats_list in task_char_stats.items():
        if not stats_list:
            continue
        
        total_chars = sum(s['total_chars'] for s in stats_list)
        num_trials = len(stats_list)
        avg_chars = total_chars / num_trials if num_trials > 0 else 0
        
        aggregated[task_name] = {
            'total_output_chars': total_chars,
            'avg_output_chars': int(avg_chars),
            'num_trials': num_trials,
        }
    
    return aggregated


def main(args):
    output_paths = args.output_paths
    
    if not output_paths:
        print("Error: No output path specified")
        print("Usage: python merge_char_stats.py output_path1 output_path2 ...")
        sys.exit(1)
    
    all_result_dfs = []
    
    # Process each output path
    for output_path in output_paths:
        result_csv_path = os.path.join(output_path, 'result.csv')
        
        if not os.path.exists(result_csv_path):
            print(f"Warning: Cannot find {result_csv_path}")
            continue
        
        print(f"Processing {result_csv_path}...")
        
        # Read result.csv
        try:
            result_df = pd.read_csv(result_csv_path)
        except Exception as e:
            print(f"Error: Cannot read {result_csv_path}: {e}")
            continue
        
        task_char_stats = find_pkl_files(output_path)
        aggregated_stats = aggregate_char_stats(task_char_stats)
        
        if 'total_output_chars' not in result_df.columns:
            result_df['total_output_chars'] = 0
        if 'avg_output_chars' not in result_df.columns:
            result_df['avg_output_chars'] = 0
        
        for idx, row in result_df.iterrows():
            task_name = row.get('task_template')
            if task_name and task_name in aggregated_stats:
                stats = aggregated_stats[task_name]
                result_df.at[idx, 'total_output_chars'] = stats['total_output_chars']
                result_df.at[idx, 'avg_output_chars'] = stats['avg_output_chars']
        
        all_result_dfs.append(result_df)
    
    if not all_result_dfs:
        print("Error: No result.csv files found")
        sys.exit(1)
    
    merged_df = pd.concat(all_result_dfs, ignore_index=True)
    
    grouped = merged_df.groupby('task_template', as_index=False).agg({
        'num_complete_trials': 'sum',
        'mean_success_rate': 'mean',
        'mean_episode_length': 'mean',
        'total_runtime_s': 'sum',
        'num_fail_trials': 'sum',
        'total_output_chars': 'sum',
        'avg_output_chars': 'mean',
    })
    
    grouped.insert(0, 'task_num', range(len(grouped)))
    
    avg_row = pd.DataFrame([{
        'task_num': 0,
        'task_template': '========= Average =========',
        'num_complete_trials': grouped['num_complete_trials'].mean(),
        'mean_success_rate': grouped['mean_success_rate'].mean(),
        'mean_episode_length': grouped['mean_episode_length'].mean(),
        'total_runtime_s': grouped['total_runtime_s'].mean(),
        'num_fail_trials': grouped['num_fail_trials'].mean(),
        'total_output_chars': grouped['total_output_chars'].mean(),
        'avg_output_chars': grouped['avg_output_chars'].mean(),
    }])
    
    final_df = pd.concat([grouped, avg_row], ignore_index=True)
    
    output_csv_path = os.path.join(output_paths[0], 'result_with_chars.csv')
    final_df.to_csv(output_csv_path, index=False)
    
    print(f"\nMerge completed! Results saved to {output_csv_path}")
    print("\nStatistics:")
    print(final_df.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge character count statistics into result.csv')
    parser.add_argument('output_paths', nargs='+', 
                       help='Output paths containing result.csv and pkl.gz files')
    
    args = parser.parse_args()
    main(args)
