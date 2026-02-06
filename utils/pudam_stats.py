import json
import csv
import os
from collections import defaultdict

with open('', 'r') as f:
    metadata = json.load(f)

#  result.csv
input_file = ''
input_dir_name = os.path.basename(os.path.dirname(input_file))

results = {}
with open(input_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        task_template = row['task_template']
        success_rate = float(row['mean_success_rate'])
        results[task_template] = success_rate

task_abilities = {}
for task in metadata:
    task_name = task.get('task_name', '')
    ability = task.get('ability', {})
    if ability:
        task_abilities[task_name] = ability

# PUDAM: p, u, d, a, m
params = ['p', 'u', 'd', 'a', 'm']
param_names = {
    'p': 'Perception',
    'u': 'Understanding', 
    'd': 'Decision',
    'a': 'Action',
    'm': 'Memory'
}

stats = {p: {l: {'correct': 0, 'total': 0} for l in [1, 2, 3, 4]} for p in params}

for task_name, ability in task_abilities.items():
    if task_name in results:
        success_rate = results[task_name]
        
        for param in params:
            if param in ability:
                level = ability[param]
                stats[param][level]['total'] += 1
                if success_rate > 0:  # 成功
                    stats[param][level]['correct'] += 1

output_file = f''

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    writer.writerow(['parameter', 'parameter_name', 'level', 'correct', 'total', 'success_rate'])
    
    for param in params:
        for level in [1, 2, 3, 4]:
            correct = stats[param][level]['correct']
            total = stats[param][level]['total']
            
            if total > 0:
                rate = correct / total
            else:
                rate = ''
            
            writer.writerow([
                param,
                param_names[param],
                level,
                correct,
                total,
                rate
            ])
        
        correct_1_2 = stats[param][1]['correct'] + stats[param][2]['correct']
        total_1_2 = stats[param][1]['total'] + stats[param][2]['total']
        rate_1_2 = correct_1_2 / total_1_2 if total_1_2 > 0 else ''
        writer.writerow([
            param,
            param_names[param],
            '1+2',
            correct_1_2,
            total_1_2,
            rate_1_2
        ])
        
        correct_3_4 = stats[param][3]['correct'] + stats[param][4]['correct']
        total_3_4 = stats[param][3]['total'] + stats[param][4]['total']
        rate_3_4 = correct_3_4 / total_3_4 if total_3_4 > 0 else ''
        writer.writerow([
            param,
            param_names[param],
            '3+4',
            correct_3_4,
            total_3_4,
            rate_3_4
        ])

print(f"统计结果已保存到: {output_file}")
