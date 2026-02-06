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


class _Fitbook(task_eval.TaskEval):
    app_names = ("fitbook",)
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


class GUIBrowsingFitbookCalories(_Fitbook):
    app_names = ("fitbook",)
    template = ('In the food feature of Fitbook, there are calories for various foods. '
                'Please find the food item that contains the keyword “ice cream” or “chocolate milk” and has the highest calories, and directly provide the full name of that food.'
    )
    complexity = 7
    schema = {}
    
    def initialize_task(self, env: interface.AsyncEnv) -> None:
        super().initialize_task(env)
    
    def is_successful(self, env: interface.AsyncEnv) -> float:
        super().is_successful(env)    
        if not env.interaction_cache:
            print(f'Agent did not return answer')
            return 0.0
        
        goal = self.template
        prediction = env.interaction_cache
        gt_answer = 'Cookie, cone shell, ice cream type, wafer or cake'
        
        try:
            ply = VerifyPolicy({})
            result = ply.verify_textual_prediction(
                goal=goal,
                prediction=prediction,
                gt_answer=gt_answer,
                judgement_type='guibrowsing'
            )
            return 1.0 if result else 0.0
        except Exception as e:
            logging.warning(
                'Error during verification for GUIBrowsingFitbookCalories: %s', e
            )
            return 0.0
    
    @classmethod
    def generate_random_params(cls) -> dict[str, str]:
        return {}


class FuncAssistFitbookLocateOneFunctionality(_Fitbook):
    app_names = ("fitbook",)
    complexity = 2
    schema = {}
    
    @property
    def goal(self) -> str:
        return (
            'Explore Fitbook and show me the interface where I can add a new weight record.'
        )
    
    def initialize_task(self, env: interface.AsyncEnv) -> None:
        super().initialize_task(env)

    def is_successful(self, env: interface.AsyncEnv) -> float:
        super().is_successful(env)
        
        goal = self.goal
        task_key = 'FuncAssistFitbookLocateOneFunctionality'
        
        try:
            ply = VerifyPolicy({})
            result = ply.verify_locating_interface(
                goal=goal,
                task_key=task_key,
                last_screenshot=self.last_screenshot,
            )
            return 1.0 if result else 0.0
        except Exception as e:
            logging.warning(
                'Error during verification for FuncAssistFitbookLocateOneFunctionality: %s', e
            )
            return 0.0
    
    @classmethod
    def generate_random_params(cls) -> dict[str, str]:
        return {}
    

class VagueFoodCalorieCheck(_Fitbook):
    app_names = ("fitbook",)
    complexity = 7
    schema = {}
      
    @property
    def template(self) -> str:
        food_name = self.params['food_name']
        grams = self.params['grams']
        return (
            f"I had {grams}g of {food_name} today. "
            "Without searching online via a browser, I'd like to know approximately how many kilocalories that is. "
            "Please give me the number directly."
        )
    
    # 食物列表：(食物名称, 克数, 热量)
    FOOD_CALORIES = [
        ('liquid yogurt', 200, '144'),
        ('Tzatziki dip', 300, '282'),
        ('frozen vanilla yogurt', 400, '524'),
        ('low fat strawberry milk', 350, '245'),
        ('light cream', 250, '495'),
        ('pudding rice', 500, '530'),
        ('reduced fat Swiss cheese', 100, '173'),
        ('cooked oxtails', 450, '1089'),
        ('cheese sauce', 200, '322'),
        ('seafood dip', 300, '1017'),
    ]
    
    @property
    def answer(self) -> str:
        return self.params['calories']
    
    def initialize_task(self, env: interface.AsyncEnv) -> None:
        super().initialize_task(env)
    
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
        food_name, grams, calories = random.choice(cls.FOOD_CALORIES)
        return {
            'food_name': food_name,
            'grams': str(grams),
            'calories': calories,
        }