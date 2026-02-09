import re
import base64
import json
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional

import requests
from PIL import Image
from policy.base_policy import BasePolicy


class UIVenusPolicy(BasePolicy):
    """UIVenus策略 - 基于大模型的决策策略"""
    
    def __init__(self, runtime_context, **kwargs):
        """初始化策略
        
        Args:
            runtime_context: 运行时上下文
            **kwargs: 模型配置参数(model_host, model_port, model_name, temperature)
        """
        self.logger = logging.getLogger(__name__)
        self.model_url = f'{kwargs.get("model_host", "localhost")}:{kwargs.get("model_port", 8000)}/v1/chat/completions'
        self.model_name = kwargs.get("model_name", "model")
        self.temperature = kwargs.get("temperature", 0.0)
        self.last_action = None
        self.runtime_context = runtime_context
    
    def get_next_action(self, state: Dict[str, Any]) -> tuple:
        """获取下一步操作
        
        Args:
            state: 当前状态(包含截图和用户查询)
            
        Returns:
            (action, result, parse_result, think, conclusion)元组
        """
        screenshot = state["screenshot_str"]
        conversations = state["user_query"]
        image_data = base64.b64decode(screenshot)
        image = Image.open(BytesIO(image_data))
        width, height = image.size

        response = self._call_model(screenshot, conversations)
        if not response:
            return None, None, None, None, None
            
        action, result, parse_result, think, conclusion = self._parse_response(response, width, height)
        self.last_action = action
        
        return action, result, parse_result, think, conclusion
            
    def report_result(self, success: bool) -> None:
        """报告操作结果"""
        if not success and self.last_action:
            self.logger.warning("操作失败: %s", self.last_action)
        self.last_action = None
        
    def _call_model(self, screenshot: str, conversations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """调用大语言模型"""
        input_text = conversations
        headers = {
            "Content-Type": "application/json"
        }

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": input_text},
                {"type": "image_url", "image_url": {"url": f"data:image;base64,{screenshot}"}}
            ]}
        ]

        body = {"model": self.model_name, "messages": messages, "temperature": self.temperature}
        try:
            response = requests.post(self.model_url, data=json.dumps(body), headers=headers, stream=False, timeout=30)
            response.raise_for_status()  
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error("模型调用异常: %s", str(e))
            return None
            
    def _parse_response(self, response: Dict[str, Any], width: int, height: int) -> tuple:
        """解析模型响应
        
        Args:
            response: 模型返回的响应
            width: 屏幕宽度
            height: 屏幕高度
            
        Returns:
            (action_output, result_new, parse_result, think, conclusion)元组
        """
        result = response['choices'][0]['message']['content']
        self.logger.info("模型输出: %s", result)
        parse_result = str(result)

        def extract_tag_content(tag_name: str, data: str) -> Optional[str]:
            pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
            match = re.search(pattern, data, re.DOTALL)
            return match.group(1).strip() if match else None

        action_str = extract_tag_content('action', result) or result.strip()
        action_type = "CallUser"
        space = {"content": "无法解析模型输出"}
        ori_action = action_str

        try:
            func_call_match = re.match(r'(\w+)\((.*)\)', action_str)
            if func_call_match:
                func_name = func_call_match.group(1)
                params_str = func_call_match.group(2)
                
                params = {}
                # 使用正则表达式来正确解析参数，避免括号内的逗号被错误分割
                # 匹配 key=value 模式，其中 value 可以是元组、字符串或数字
                param_pattern = r"(\w+)\s*=\s*(\([^)]+\)|'[^']*'|\"[^\"]*\"|[^,]+)"
                param_matches = re.findall(param_pattern, params_str)
                
                for key, value in param_matches:
                    key = key.strip()
                    value = value.strip()
                    if value.startswith("'") and value.endswith("'"):
                        params[key] = value[1:-1]
                    elif value.startswith('"') and value.endswith('"'):
                        params[key] = value[1:-1]
                    elif value.startswith('(') and value.endswith(')'):
                        # 解析元组，如 (868,487)
                        try:
                            inner = value[1:-1]
                            parts = [p.strip() for p in inner.split(',')]
                            params[key] = tuple(int(p) for p in parts)
                        except ValueError:
                            params[key] = value
                    elif value.replace('.', '', 1).replace('-', '', 1).isdigit():
                        params[key] = float(value) if '.' in value else int(value)
                    else:
                        params[key] = value
                
                action_type = func_name
                space = params
                
        except (ValueError, KeyError) as e:
            self.logger.warning("解析模型输出失败: %s", str(e))
            action_type = 'CallUser'
            space = {'content': f"解析失败: {str(e)}"}

        self.logger.info("动作类型: %s, 参数: %s", action_type, space)
        
        think = extract_tag_content('think', result) or ''
        conclusion = extract_tag_content('conclusion', result) or ''
        result_new = f"<think>{think}</think><action>{ori_action}</action>" if think else ori_action

        action_maps = {
            "Launch": "REOPEN",
            "PressBack": "BACK",
            "PressHome": "PressHome",
            "PressEnter": "PressEnter",
            "PressRecent": "PressMenu",
            "Wait": "WAIT",
            "Click": "CLK",
            "Type": "INPUT",
            "LongPress": "LongPress",
            "Finished": "SUCCESS",
            "CallUser": "CallUser",
        }

        mapped_action_type = action_maps.get(action_type, action_type)
        
        # Scroll 和 Drag 都使用 start/end 坐标，映射为 SWIPE
        if mapped_action_type in ['Scroll', 'Drag']:
            mapped_action_type = 'SWIPE'

        self.runtime_context.pred_action.append(result_new)

        action_output = {
            "action_type": None,
            "action_pos": [],
            "input": "",
            "duration": -1,
            "role": "SIPA",
            "timestamp": None,
            "extend": ""
        }

        # 归一化坐标转换函数（模型输出基于 1000x1000 坐标系）
        def normalize_to_screen(x, y):
            """将 1000x1000 归一化坐标转换为实际屏幕坐标"""
            actual_x = int(x * width / 1000)
            actual_y = int(y * height / 1000)
            return actual_x, actual_y

        if mapped_action_type in ['CLK', 'INPUT', 'LongPress']:
            if mapped_action_type == 'CLK':
                action_output["action_type"] = "CLK"
                if 'box' in space and isinstance(space['box'], tuple) and len(space['box']) == 2:
                    x, y = space['box']
                    actual_x, actual_y = normalize_to_screen(x, y)
                    self.logger.info("坐标转换: (%d,%d) -> (%d,%d) [屏幕: %dx%d]", x, y, actual_x, actual_y, width, height)
                    action_output["action_pos"] = [[actual_x, actual_y]]
            elif mapped_action_type == 'LongPress':
                action_output["action_type"] = "LongPress"
                if 'box' in space and isinstance(space['box'], tuple) and len(space['box']) == 2:
                    x, y = space['box']
                    actual_x, actual_y = normalize_to_screen(x, y)
                    self.logger.info("坐标转换: (%d,%d) -> (%d,%d) [屏幕: %dx%d]", x, y, actual_x, actual_y, width, height)
                    action_output["action_pos"] = [[actual_x, actual_y]]
            else:
                action_output["action_type"] = "INPUT"
                input_text = space.get('content', '')
                input_text = re.sub(r'\(', '（', input_text)
                input_text = re.sub(r'\)', '）', input_text)
                input_text = re.sub(r'\|', '｜', input_text)
                action_output["input"] = input_text

        elif 'SWIPE' in mapped_action_type:
            action_output["action_type"] = 'SWIPE'
            if 'start' in space and 'end' in space:
                # 模型给出的 start/end 也是 1000x1000 归一化坐标
                start_x, start_y = normalize_to_screen(*space['start'])
                end_x, end_y = normalize_to_screen(*space['end'])
                self.logger.info("Swipe 坐标转换: start %s -> (%d,%d), end %s -> (%d,%d)", 
                               space['start'], start_x, start_y, space['end'], end_x, end_y)
            else:
                # 使用预设的方向滑动（已经是实际屏幕坐标）
                drag_map_dict = {
                    'SWIPE_UP': [width * 3 / 4, height * 5 / 7, width * 3 / 4, height * 2 / 5],
                    'SWIPE_DOWN': [width / 2, height / 4, width / 2, height * 3 / 4],
                    'SWIPE_LEFT': [width / 2, height / 2, 0, height / 2],
                    'SWIPE_RIGHT': [width / 2, height / 2, width, height / 2]
                }
                start_x, start_y, end_x, end_y = [int(item) for item in drag_map_dict.get(mapped_action_type, drag_map_dict['SWIPE_UP'])]
            action_output["action_pos"] = [[int(start_x), int(start_y)], [int(end_x), int(end_y)]]

        elif mapped_action_type in ['BACK', 'WAIT', 'SUCCESS', 'PressHome', 'PressEnter', 'PressMenu']:
            action_output["action_type"] = mapped_action_type
            if mapped_action_type == 'WAIT':
                action_output["duration"] = 1000
        elif mapped_action_type == 'REOPEN':
            action_output["action_type"] = mapped_action_type
            action_output['app_name'] = space.get('app', '')
        elif mapped_action_type == 'CallUser':
            action_output["action_type"] = mapped_action_type
            action_output["input"] = space.get('content', '')

        # SUCCESS 也返回完整的 action，让 run_handler 能正确判断
        return action_output, result_new, parse_result, think, conclusion
