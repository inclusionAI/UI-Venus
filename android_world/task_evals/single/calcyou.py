"""Tasks for general system tasks like interacting with settings."""

import dataclasses
import random
from typing import Any

import logging
from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.utils import fuzzy_match_lib
import immutabledict
from android_world.policy.verification import VerifyPolicy




class _Calcyou(task_eval.TaskEval):

  app_names = ('calcyou',)
  complexity = 1
  schema = {
  }
  template = ''

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}

class _Calculator(task_eval.TaskEval):

  app_names = ('calculator',)
  complexity = 1
  schema = {
  }
  template = ''

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}

class VagueCurrencyExchange(_Calcyou):
  
  app_names = ('CalcYou',)

  complexity = 7

  schema = {}

  CURRENCY_EXCHANGES = [
      ('Euro', 'New Taiwan Dollars', '35.693'),
      ('US Dollar', 'Thai Baht', '31.775'),
      ('Hong Kong Dollar', 'South Korean Won', '179.253'),
      ('Canadian Dollar', 'Japanese Yen', '106.775'),
      ('British Pound', 'Malaysian Ringgit', '5.678'),
  ]

  @property
  def goal(self) -> str:
    source_currency = self.params['source_currency']
    target_currency = self.params['target_currency']
    return (
        f"I can't access the internet right now, and I'd like to know how many {target_currency} "
        f"1 {source_currency} can be exchanged for. Please just tell me the number directly "
        f"(rounded to three decimal places)."
    )
  
  @property
  def answer(self) -> str:
    return self.params['answer']

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not env.interaction_cache:
      return 0.0
    if self.answer in env.interaction_cache:
      return 1.0
    else:
      return 0.0

  
  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    source_currency, target_currency, exchange_rate = random.choice(cls.CURRENCY_EXCHANGES)
    return {
        'app_name': 'CalcYou',
        'source_currency': source_currency,
        'target_currency': target_currency,
        'answer': exchange_rate,
    }



class VagueGraphFunction(_Calcyou):
  schema = {}

  complexity = 5
  
  template = 'I can’t access the internet right now, and I want to know the graph of the function f(x) = 3x^2 + 2x - 6.'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    
    
    goal = self.template
    task_key = 'VagueGraphFunction'
    
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
          'Error during verification for VagueGraphFunction: %s', e
      )
      return 0.0





class VagueCalculator(_Calculator):
  schema = {}

  complexity = 5
  
  template = 'Tell me the result of {formula}. Only return an integer.'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not env.interaction_cache:
      print(f'Agent did not return answer')
      return 0.0
    if self.params['text'] == env.interaction_cache:
      return 1
    else:
      return 0


  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:

    def _generate_log_part() -> tuple[str, int]:

      base = random.choice([2, 3, 5, 7, 13, 17, 19])

      # 根据底数动态调整指数范围，使结果的大数 N 足够大 (约 10^12 到 10^14)
      if base == 2:
          exponent = random.randint(38, 45) # 2^45 ≈ 3.5 x 10^13
      elif base == 3:
          exponent = random.randint(21, 26) # 3^29 ≈ 6.8 x 10^13
      elif base == 5:
          exponent = random.randint(15, 17) # 5^20 ≈ 9.5 x 10^13
      elif base <= 13:
          exponent = random.randint(10, 12) # 13^13 ≈ 3 x 10^14
      else: # base 17, 19
          exponent = random.randint(9, 10)

      big_num = base ** exponent
      
      func = random.choice(["ln", "log"])
      
      expr = f"{func}({big_num}) / {func}({base})"
      return expr, exponent

    def _generate_pyth_part() -> tuple[str, int]:
      """
      生成勾股数部分：√(a*a + b*b)。结果为 c。
      """
      m = random.randint(350, 550)
      n = random.randint(100, m - 1)

      a = m**2 - n**2
      b = 2 * m * n
      c = m**2 + n**2 

      expr = f"√({a} * {a} + {b} * {b})"
      
      return expr, c

    log_expr, log_ans = _generate_log_part()
    pyth_expr, pyth_ans = _generate_pyth_part()

    final_formula = f"{log_expr} + {pyth_expr}"
    
    final_answer = log_ans + pyth_ans

    logging.info(f"VagueCalculator final_answer: {final_answer}")
    return {"text": str(final_answer), 'formula': final_formula}

