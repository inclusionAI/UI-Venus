import os
import sys

import gzip
import pickle

from android_world.agents import base_agent
from android_world.env import interface
from android_world.agents import agent_utils
from android_world.agents import m3a_utils
from android_world.env import representation_utils
from android_world.env import json_action_qwen as json_action

from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', type=str, default='')
args = parser.parse_args()


folder_path = args.folder_path

result_files = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".pkl.gz"):
            file_path = os.path.join(root, file)
            result_files.append(file_path)
            print(f"找到文件: {file_path}")

for filename in result_files:
    with gzip.open(filename, "rb") as f:
        data = pickle.load(f)
    file = os.path.basename(filename)
    folder_name = file.replace('.pkl.gz', '')
    print(folder_name, )
    folder_name = os.path.join(f'{folder_path}_data', folder_name)
    #import pdb;pdb.set_trace()
    os.makedirs(folder_name, exist_ok=True)

    episode_length = data[0]['episode_length']
    if not isinstance(episode_length, int):
        continue

    with open(os.path.join(folder_name, "000000status.txt"), "w") as f:
        f.write(str(data[0]['is_successful'])+'\n')
        f.write(str(data[0]['task_template'])+'\n')
        f.write(str(data[0]['run_time'])+'\n')
    
    try:
        with open(os.path.join(folder_name, "000000goal.txt"), "w") as f:
            f.write(data[0]['goal'])
    except:
        pass
    episodes = data[0]['episode_data']
    for i in range(episode_length):
        print(i, ' / ', episode_length)

        raw_screenshot = episodes['raw_screenshot'][i]

        try:
            Image.fromarray(raw_screenshot).save(os.path.join(folder_name, "%03d_raw.jpg"%(i+1)))
        except:
            pass
        thinking_ = episodes['action_output'][i]
        thinking = thinking_.split('<think>')[1].split('</think>')[0]
        tool_call = episodes['action_output_json'][i]
        tool_call = str(tool_call)
        conclusion = thinking_.split('<conclusion>')[1].split('</conclusion>')[0]
        #import pdb;pdb.set_trace()
        try:
            with open(os.path.join(folder_name, "%03d_thinking.txt"%(i+1)), "w") as f:
                f.write(thinking)
        except:
            pass
        try:
            with open(os.path.join(folder_name, "%03d_tool_call.txt"%(i+1)), "w") as f:
                f.write(tool_call)
        except:
            pass
        try:
            with open(os.path.join(folder_name, "%03d_conclusion.txt"%(i+1)), "w") as f:
                f.write(conclusion)
        except:
            pass

