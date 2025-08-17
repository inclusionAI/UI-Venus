from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from PIL import Image
import logging

from models.navigation.ui_venus_navi_vllm import NaviVLLM
from qwen_vl_utils import smart_resize
from .utils import parse_answer, USER_PROMPT


ACTION_MAPPING = {
    'click', 'drag', 'scroll', 'type', 'launch', 'wait', 'finished',
    'calluser', 'longpress', 'pressback', 'presshome', 'pressenter',
    'pressrecent', 'answer'
}


@dataclass
class StepData:
    image_path: str
    raw_screenshot: Image.Image
    query: str
    generated_text: str
    think: str
    action: str
    _conclusion: str
    action_output_json: Optional[Dict[str, Any]] = None
    status: str = 'success'

    def to_dict(self, include_screenshot: bool = False) -> dict:
        """
        Convert this step to a JSON-serializable dict.

        Args:
            include_screenshot (bool): Whether to include base64-encoded image.

        Returns:
            dict: Serializable step data.
        """
        data = asdict(self)
        data['raw_screenshot'] = None

        if include_screenshot and self.raw_screenshot is not None:
            import base64
            from io import BytesIO
            buffer = BytesIO()
            self.raw_screenshot.save(buffer, format="PNG")
            data['raw_screenshot_base64'] = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return data


class VenusNaviAgent:
    def __init__(self,
                 model_config,
                 logger: logging.Logger,
                 history_length: int = 0) -> None:
        self.model = NaviVLLM(model_config=model_config, logger=logger)
        self.max_pixels = model_config.max_pixels
        self.min_pixels = model_config.min_pixels
        self.logger = logger
        self.history: List[StepData] = []
        self.history_length = max(0, history_length)
    
    def reset(self):
        self.logger.info(f"Agent Reset")
        self.history = []

    def _build_query(self, goal: str) -> str:
        if len(self.history) == 0:
            history_str = ""
        else:
            recent_history = self.history[-self.history_length:]
            history_entries = [
                f"Step {i}: <think>{step.think}</think><action>{step.action}</action>" for i, step in enumerate(recent_history)
            ]
            history_str = "\n".join(history_entries)
        
        return USER_PROMPT.format(user_task=goal, previous_actions=history_str)

    def _rescale_coordinate(self, x: float, y: float, orig_size: Tuple[int, int], resized_size: Tuple[int, int]) -> Tuple[int, int]:
        o_w, o_h = orig_size
        r_w, r_h = resized_size
        x_scaled = int(x * o_w / r_w)
        y_scaled = int(y * o_h / r_h)
        return (
            max(0, min(x_scaled, o_w)),
            max(0, min(y_scaled, o_h))
        )

    def _convert_coordinate(self, action_json: dict, size_params: dict):        
        orig_size = (size_params['original_width'], size_params['original_height'])
        resized_size = (size_params['resized_width'], size_params['resized_height'])
        action_type = action_json['action'].lower()
        try:
            if action_type == 'click' or action_type == 'longpress':
                x, y = action_json['params']['box']
                action_json['params']['box'] = self._rescale_coordinate(x, y, orig_size, resized_size)
            elif action_type == 'drag':
                x1, y1 = action_json['params']['start']
                x2, y2 = action_json['params']['end']
                action_json['params']['start'] = self._rescale_coordinate(x1, y1, orig_size, resized_size)
                action_json['params']['end'] = self._rescale_coordinate(x2, y2, orig_size, resized_size)
            elif action_type == 'scroll':
                if 'start' in action_json['params'] and len(action_json['params']['start']) > 0:
                    x, y = action_json['params']['start']
                    action_json['params']['start'] = self._rescale_coordinate(x, y, orig_size, resized_size)
                if 'end' in action_json['params'] and len(action_json['params']['end']) > 0:
                    x, y = action_json['params']['end']
                    action_json['params']['end'] = self._rescale_coordinate(x, y, orig_size, resized_size)
        except (KeyError, ValueError, TypeError) as e:
            self.logger.warning(f"convert failed: {e}, action_json={action_json}")

        return action_json
    
    def step(self, goal: str, image_path: str):
        self.logger.info(f"----------step {len(self.history) + 1}")
        try:
            raw_screenshot = Image.open(image_path).convert('RGB')
        except Exception as e:
            self.logger.error(f"Can't load {image_path}: {e}")
            return None
        
        original_width, original_height  = raw_screenshot.size
        resized_height, resized_width = smart_resize(
            original_height, original_width,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels)
        
        size_params = {
            'original_width': original_width,
            'original_height': original_height,
            'resized_width': resized_width,
            'resized_height': resized_height,
        }
        
        user_query = self._build_query(goal)
        generated_text = self.model([(image_path, user_query)])[0][0]
        
        self.logger.info(f"Goal: {goal}")
        self.logger.info(f"USER Query: {repr(user_query)}")
        self.logger.info(f"ACTION text: {repr(str(generated_text))}")

        think_text = generated_text.split('<think>')[1].split('</think>')[0].strip('\n')
        answer_text = generated_text.split('<action>')[1].split('</action>')[0].strip('\n')
        conclusion_text = generated_text.split('<conclusion>')[1].split('</conclusion>')[0].strip('\n')
        
        self.logger.info(f"Think: {think_text}")
        self.logger.info(f"Answer: {answer_text}")
        
        try:
            action_name, action_params = parse_answer(answer_text)
            action_json = {'action': action_name, 'params': action_params}
            action_json = self._convert_coordinate(action_json, size_params)
        except Exception as e:
            self.logger.warning(f'Failed to parse_answer: {e}')
            step_data = StepData(
                image_path=image_path,
                raw_screenshot=raw_screenshot,
                query=user_query,
                generated_text=generated_text,
                think=think_text,
                action=answer_text,
                _conclusion=conclusion_text,
                status='failed'
            )
            self.history.append(step_data)
            return None

        step_data = StepData(
            image_path=image_path,
            raw_screenshot=raw_screenshot,
            query=user_query,
            generated_text=generated_text,
            think=think_text,
            action=answer_text,
            _conclusion=conclusion_text,
            action_output_json=action_json,
            status='success'
        )
        self.history.append(step_data)

        self.logger.info(f'Action: {repr(str(action_json))}')
        return action_json
    
    def export_history(self, include_screenshot=False):
        serialized_history = [
            step.to_dict(include_screenshot=include_screenshot)
            for step in self.history
        ]
        return serialized_history
    

        
        
        