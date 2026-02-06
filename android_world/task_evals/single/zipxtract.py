import random
import logging
from typing import Any
from android_world.task_evals.information_retrieval import datetime_utils as datetime_utils_ir
from android_world.task_evals.information_retrieval import task_app_utils
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import file_validators
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils
from android_world.task_evals.information_retrieval.proto import task_pb2
from android_world.task_evals.information_retrieval import proto_utils
from android_world.task_evals.utils import sqlite_utils
from android_world.env import adb_utils
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.env import adb_utils
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals.information_retrieval import calendar_utils
from android_world.task_evals.information_retrieval import datetime_utils as datetime_utils_ir
from android_world.task_evals.information_retrieval import proto_utils
from android_world.task_evals.information_retrieval.proto import state_pb2
from android_world.task_evals.information_retrieval.proto import task_pb2
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import sqlite_utils
from android_world.utils import datetime_utils
from android_world.policy.verification import VerifyPolicy
import datetime
import random
import uuid
import yaml
import os

# # Load paths from venus_benchmark_settings.yaml
# _CONFIG_PATH = os.path.join(
#     os.path.dirname(__file__), '..', '..', '..', 'venus_benchmark_settings.yaml'
# )
# with open(_CONFIG_PATH, 'r') as f:
#   _BENCHMARK_CONFIG = yaml.safe_load(f)

# # Load local paths from config
# _LOCAL_ZIPXTRACT_ZIP = _BENCHMARK_CONFIG['local_paths']['zipxtract_zip']

# # Load remote paths from config
# _REMOTE_SDCARD = _BENCHMARK_CONFIG['remote_paths']['sdcard']

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from config import config

# Load local paths from config
_LOCAL_ZIPXTRACT_ZIP = config.get('local_paths.zipxtract_zip')

# Load remote paths from config
_REMOTE_SDCARD = config.get('remote_paths.sdcard')


class _Zipxtract(task_eval.TaskEval):
    app_names = ("zipxtract",)
    template = ''
    schema = {}
    complexity = 3.0
    
    def initialize_task(self, env: interface.AsyncEnv) -> None:
        super().initialize_task(env)
    
    def is_successful(self, env: interface.AsyncEnv) -> float:
        super().is_successful(env)
        return 0
    
    @classmethod
    def generate_random_params(cls) -> dict[str, Any]:
        return {}


class FuncAssistZipxtractExplainOneFunctionality(_Zipxtract):
    app_names = ("zipxtract",)
    template = 'Please tell me how to use ZipXtract to extract files.'
    complexity = 4
    schema = {}

    def is_successful(self, env: interface.AsyncEnv) -> float:
        super().is_successful(env)
        if not env.interaction_cache:
            return 0.0

        goal = self.template
        prediction = env.interaction_cache
        gt_answer = (
            "First, click the button below and select 'Extract'. "
            "Then click 'Pick File' to choose the folder path of the files you want to extract. "
            "Next, click 'Extract to' to select the destination path where the extracted folder should be saved. "
            "Finally, click 'Extract' to complete the extraction."
        )

        try:
            ply = VerifyPolicy({})
            result = ply.verify_textual_prediction(
                goal=goal,
                prediction=prediction,
                gt_answer=gt_answer,
                judgement_type='funcassist'
            )
            return 1.0 if result else 0.0
        except Exception as e:
            logging.warning(
                'Error during verification for FuncAssistZipxtractExplainOneFunctionality: %s', e
            )
            return 0.0
        

class VagueZipxtractExtractFile(_Zipxtract):
    app_names = ("zipxtract",)
    complexity = 10
    schema = {}
    
    # 关键词列表：(关键词, 对应的文件名)
    KEYWORD_FILES = [
        ('mycelium', '1.md'),
        ('buttress', '2.md'),
        ('exoplanet', '3.md'),
        ('liquidity', '4.md'),
        ('entropy', '5.md'),
    ]
    
    @property
    def template(self) -> str:
        keyword = self.params['keyword']
        return (
            "In the `zip` folder within sdk_gphone_x86_64 storage area, there is an archive named `MyFile.zip` that contains several markdown files. "
            f"Please directly tell me the filenames (including extensions) of the files whose main text content contains the keyword '{keyword}'(case-insensitive)."
        )
    
    @property
    def answer(self) -> str:
        return self.params['answer']
    
    def initialize_task(self, env: interface.AsyncEnv) -> None:
        super().initialize_task(env)
        adb_utils.push_file_to_emulator(_LOCAL_ZIPXTRACT_ZIP, remote_path=_REMOTE_SDCARD, env=env.controller)
    
    def is_successful(self, env: interface.AsyncEnv) -> float:
        super().is_successful(env)
        if not env.interaction_cache:
            return 0.0
        if self.answer in env.interaction_cache:
            return 1
        else:
            return 0
    
    @classmethod
    def generate_random_params(cls) -> dict[str, str]:
        keyword, filename = random.choice(cls.KEYWORD_FILES)
        return {
            'keyword': keyword,
            'answer': filename,
        }


class RefusalZipxtractCreateFile(_Zipxtract):
    app_names = ("zipxtract",)
    complexity = 3
    schema = {}
    
    # 图片文件列表
    IMAGE_FILES = [
        'flower.png',
        'fruit.png',
        'graphic.png',
        'table.png',   
    ]
    
    @property
    def template(self) -> str:
        filename = self.params['filename']
        return f"Use ZipXtract and compress the file {filename} in the GUIM folder within sdk_gphone_x86_64 storage area into {filename.split('.')[0]}.rar."
    
    @property
    def answer(self) -> str:
        filename = self.params['filename']
        return f"I can't compress {filename} into {filename.split('.')[0]}.rar because ZipXtract doesn't support compressing to the RAR format."

    def is_successful(self, env: interface.AsyncEnv) -> float:
        super().is_successful(env)
        ans = self.answer
        if not env.interaction_cache:
            print(f'Agent did not return answer')
            return 0.0
        ply = VerifyPolicy({})
        result = ply.verify_textual_prediction(
                goal=self.template,
                prediction=env.interaction_cache,
                gt_answer=ans,
                judgement_type='refusal'
            )
        return result
    
    @classmethod
    def generate_random_params(cls) -> dict[str, str]:
        filename = random.choice(cls.IMAGE_FILES)
        return {'filename': filename}