import os
import re
import logging
import time
import base64
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from app.runtime_context import RuntimeContext
from device.device_manager import DeviceManager
from policy.base_policy import BasePolicy
from utils.pickle_utils import gzip_pickle

# 策略类型 -> (Policy类, Processor类) 映射表
# 添加新策略只需在此注册
POLICY_REGISTRY = {
    "ui_venus": {
        "policy": "policy.ui_venus_policy.UIVenusPolicy",
        "processor": "processor.uivenus_processor.UIVenusProcessor",
    },
    # 示例：添加其他策略
    # "other_policy": {
    #     "policy": "policy.other_policy.OtherPolicy",
    #     "processor": "processor.other_processor.OtherProcessor",
    # },
}


class RunHandler:
    """运行处理器 - 协调设备控制、策略执行和轨迹保存"""
    
    def __init__(
        self,
        device_id: str,
        trace_dir: str,
        policy_type: str = "ui_venus",
        ep_config: Dict[str, Any] = None,
        app_mapping: Optional[Dict[str, str]] = None,
        **policy_kwargs
    ):
        """初始化运行处理器
        
        Args:
            device_id: 设备ID
            trace_dir: 轨迹保存目录
            policy_type: 策略类型（自动绑定对应的 processor）
            ep_config: 任务配置
            app_mapping: 应用名称映射
            **policy_kwargs: 策略额外参数
        """
        self.logger = logging.getLogger(__name__)
        self.runtime_context = RuntimeContext()
        self.step_limit = ep_config.get('step_limit')
        self.ep_config = ep_config
        self.purpose = ""
        self.app_mapping: Dict[str, str] = app_mapping or {}
        self.device_manager = DeviceManager()
        self.device_id = device_id
        
        if not self.device_manager.connect_device(device_id):
            raise ConnectionError(f"设备连接失败: {device_id}")

        # 根据策略类型创建对应的 policy 和 processor（自动绑定）
        self.policy, self.state_processor = self._create_policy_and_processor(policy_type, **policy_kwargs)
        self.trace_dir = trace_dir
        
    def _create_policy_and_processor(self, policy_type: str, **kwargs) -> tuple:
        """根据策略类型创建对应的 policy 和 processor
        
        策略和处理器通过 POLICY_REGISTRY 映射表绑定，
        添加新策略只需在映射表中注册即可
        """
        if policy_type not in POLICY_REGISTRY:
            available = ", ".join(POLICY_REGISTRY.keys())
            raise ValueError(f"不支持的策略类型: {policy_type}，可用类型: {available}")
        
        registry = POLICY_REGISTRY[policy_type]
        
        # 动态导入 Policy 类
        policy_module, policy_class = registry["policy"].rsplit(".", 1)
        PolicyClass = getattr(__import__(policy_module, fromlist=[policy_class]), policy_class)
        
        # 动态导入 Processor 类
        processor_module, processor_class = registry["processor"].rsplit(".", 1)
        ProcessorClass = getattr(__import__(processor_module, fromlist=[processor_class]), processor_class)
        
        policy = PolicyClass(self.runtime_context, **kwargs)
        processor = ProcessorClass()
        
        self.logger.info("已加载策略: %s, 处理器: %s", policy_class, processor_class)
        return policy, processor
            
    def run(self, purpose: str = '') -> bool:
        """执行任务主循环
        
        Args:
            purpose: 任务目标描述
            
        Returns:
            任务是否成功完成
        """
        self.purpose = purpose
        episode_start_time = time.time()
        
        # 生成轨迹目录名: 时间前缀 + 任务描述前10字 + 短UUID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 清理任务描述，移除特殊字符和标点，取前10个字符
        task_desc = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', purpose)[:10]
        short_id = str(uuid.uuid4())[:8]
        episode_id = f"{timestamp}_{task_desc}_{short_id}"
        
        # 创建本轮轨迹保存目录
        episode_dir = os.path.join(self.trace_dir, episode_id)
        screenshots_dir = os.path.join(episode_dir, "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        
        # 轨迹数据结构
        episode_data = []
        is_successful = False
        ep_end = False
        termination_reason = "max_steps"  # 默认原因：达到最大步数
        
        # 重复动作检测
        recent_actions = []  # 保存最近的动作用于检测重复
        max_repeat = 5  # 最大允许重复次数
        
        # 检测任务对应的 app
        mini_app_name = self.ep_config.get('mini_app_name', 'Android App')
        for app_name in self.app_mapping.keys():
            if app_name in purpose:
                mini_app_name = app_name
                break
        
        while self.runtime_context.step < self.step_limit:
            step_start_time = time.time()
            self.runtime_context.step += 1
            step_num = self.runtime_context.step
            self.logger.info("当前步数: %d", step_num)
            
            # 1. 获取截图
            screenshot_data = self.req_from_client()
            if screenshot_data.get('ep_end'):
                self.logger.info("截图获取失败，任务结束")
                ep_end = True
                termination_reason = "screenshot_failed"
                break
                
            screenshot_data.update({'purpose': self.purpose})
            
            # 2. 处理状态并获取动作
            state, action, pred_action, _, think, conclusion = self.handle_client_req(screenshot_data)
            
            # 打印模型思考过程
            if think:
                self.logger.info("模型思考: %s", think.strip())
            
            # 检测重复动作
            action_signature = self._get_action_signature(action)
            recent_actions.append(action_signature)
            if len(recent_actions) > max_repeat:
                recent_actions.pop(0)
            
            # 如果连续 N 次相同动作，认为陷入循环（排除 swipe 操作）
            action_type = action.get('action_type') if action else None
            is_swipe = action_type and 'swipe' in action_type.lower()
            is_repeat_loop = len(recent_actions) >= max_repeat and len(set(recent_actions)) == 1 and not is_swipe
            if is_repeat_loop:
                self.logger.warning("检测到连续 %d 次重复动作: %s，任务陷入循环，终止执行", max_repeat, action_signature)
            
            # 3. 保存截图
            screenshot_path = f"step_{step_num:03d}.png"
            self._save_screenshot(
                screenshot_data['screenshot_str'],
                os.path.join(screenshots_dir, screenshot_path)
            )
            
            # 4. 执行动作
            success = False
            
            if is_repeat_loop:
                # 陷入重复循环，不再执行，直接退出
                ep_end = True
                termination_reason = "repeat_loop"
            elif action is None or action_type in ['CallUser', 'SUCCESS']:
                self.logger.info("任务完成")
                is_successful = action_type == 'SUCCESS'
                ep_end = True
                termination_reason = "success" if is_successful else "call_user"
            else:
                success = self.rsp_to_client(action)
                time.sleep(2.5)
                
                # 更新历史
                history_item = {
                    'timestamp': time.time(),
                    'state': state,
                    'action': action,
                }
                self.runtime_context.history.append(history_item)
                self.runtime_context.update_action_description(state, action, pred_action)
                self.runtime_context.update_history('action_description', self.runtime_context.action_description[-1])
                self.policy.report_result(success)
            
            # 5. 记录步骤数据（不含原始截图数据，只保存路径）
            step_data = {
                'step': step_num,
                'screenshot_path': screenshot_path,
                'state': {k: v for k, v in state.items() if k != 'screenshot_str'},  # 排除大数据
                'action': action,
                'pred_action': pred_action,
                'think': think,
                'conclusion': conclusion,
                'success': success,
                'timestamp': time.time(),
                'step_time': time.time() - step_start_time
            }
            episode_data.append(step_data)
            
            if ep_end:
                break
        
        # 如果 ep_end 仍然是 False，说明是因为超过步数限制而退出循环
        if not ep_end:
            self.logger.warning("达到最大步数限制: %d", self.step_limit)
            termination_reason = "max_steps"
                
        # 计算总耗时
        run_time = time.time() - episode_start_time
        
        # 构建完整轨迹
        trajectory = {
            "goal": purpose,
            "episode_id": episode_id,
            "device_id": self.device_id,
            "mini_app_name": mini_app_name,
            "episode_data": episode_data,
            "episode_length": self.runtime_context.step,
            "run_time": run_time,
            "is_successful": is_successful,
            "ep_end": ep_end,
            "termination_reason": termination_reason,
        }
        
        # 保存轨迹到 pkl.gz
        save_path = os.path.join(episode_dir, "trajectory.pkl.gz")
        try:
            compressed = gzip_pickle(trajectory)
            with open(save_path, 'wb') as f:
                f.write(compressed)
            self.logger.info("轨迹已保存: %s", save_path)
        except (OSError, ValueError) as e:
            self.logger.error("保存轨迹失败: %s", str(e))

        self.logger.info(
            "Episode 完成: episode_id=%s, steps=%d, run_time=%.2fs, is_successful=%s",
            episode_id, self.runtime_context.step, run_time, is_successful
        )

        self.reset()
        return is_successful

    def _save_screenshot(self, screenshot_str: str, filepath: str):
        """保存截图
        
        Args:
            screenshot_str: base64编码的截图
            filepath: 保存路径
        """
        try:
            img_data = base64.b64decode(screenshot_str)
            with open(filepath, "wb") as f:
                f.write(img_data)
        except (OSError, ValueError) as e:
            self.logger.error("保存截图失败: %s", str(e))

    def _get_action_signature(self, action: Optional[Dict[str, Any]]) -> str:
        """生成动作签名用于重复检测
        
        Args:
            action: 动作字典
            
        Returns:
            动作的字符串签名
        """
        if action is None:
            return "None"
        
        action_type = action.get('action_type', '')
        action_pos = action.get('action_pos', [])
        action_input = action.get('input', '')
        app_name = action.get('app_name', '')
        
        # 对于点击动作，坐标相近也认为是重复（允许 50 像素误差）
        if action_type == 'CLK' and action_pos:
            x, y = action_pos[0]
            # 将坐标量化到 50 像素精度
            x_q, y_q = x // 50 * 50, y // 50 * 50
            return f"{action_type}:{x_q},{y_q}"
        elif action_type == 'INPUT':
            return f"{action_type}:{action_input}"
        elif action_type == 'REOPEN':
            return f"{action_type}:{app_name}"
        else:
            return action_type

    def req_from_client(self) -> Dict[str, Any]:
        """从设备获取截图"""
        device = self.device_manager.get_device(self.device_id)
        screenshot = device.screenshot()
        if not screenshot:
            if self.runtime_context.step == 1:
                raise RuntimeError("首次获取截图失败")
            return {'screenshot_str': None, 'ep_end': True}
        return {'screenshot_str': screenshot, 'ep_end': False}
    
    def handle_client_req(self, req_data: Dict[str, Any]) -> tuple:
        """处理请求并获取下一步操作"""
        self.logger.info("开始处理状态")
        state = self.state_processor.process(req_data, self.runtime_context.step, self.runtime_context.history)
        
        self.logger.info("请求策略决策")
        action, pred_action, parse_result, think, conclusion = self.policy.get_next_action(state)
        self.logger.info("策略输出: %s", action)
        return state, action, pred_action, parse_result, think, conclusion
    
    def rsp_to_client(self, rsp: Dict[str, Any]) -> bool:
        """执行操作并返回结果"""
        success = self._execute_action(rsp)
        if success:
            self.logger.info("操作成功: %s", rsp.get('action_type'))
        else:
            self.logger.error("操作失败: %s", rsp)
        return success
    
    def reset(self, go_home: bool = False):
        """重置运行环境
        
        Args:
            go_home: 是否返回主屏幕（默认不返回）
        """
        if go_home:
            device = self.device_manager.get_device(self.device_id)
            if device:
                device.presshome()
        self.runtime_context.reset()
        self.state_processor.reset()
     
    def _execute_action(self, action: Dict[str, Any]) -> bool:
        """执行具体动作
        
        Args:
            action: 动作字典
            
        Returns:
            执行是否成功
        """
        try:
            device = self.device_manager.get_device(self.device_id)
            if not device:
                return False

            action_type = action["action_type"]
            positions = action.get("action_pos", [])
            
            if action_type == "CLK":
                return device.tap(positions[0][0], positions[0][1])
            elif action_type == "SWIPE":
                return device.swipe(positions[0][0], positions[0][1], 
                                   positions[1][0], positions[1][1], 
                                   action.get("duration", 1000))
            elif action_type == "INPUT":
                return device.input_text(action.get("input", ""))
            elif action_type == "BACK":
                return device.pressback()
            elif action_type == "WAIT":
                time.sleep(action.get("duration", 1000) / 1000)
                return True
            elif action_type == "PressHome":
                return device.presshome()
            elif action_type == "PressMenu":
                return device.pressmenu()
            elif action_type == "PressEnter":
                return device.pressenter()
            elif action_type == "LongPress":
                return device.longpress(positions[0][0], positions[0][1])
            elif action_type == "REOPEN":
                target = self.app_mapping.get(action.get("app_name"), action.get("app_name"))
                return device.launch_app(target)
            elif action_type in ["SUCCESS", "FAIL", "CallUser"]:
                return True
            else:
                self.logger.error("不支持的操作类型: %s", action_type)
                return False
        except (IndexError, KeyError, TypeError) as e:
            self.logger.error("执行失败: %s", str(e))
            return False
            
