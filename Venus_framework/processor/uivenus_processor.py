from typing import Dict, Any, List
from .base_processor import BaseProcessor

import logging

USER_PROMPT = '''
**你是一个手机图形界面智能体代理**
你的任务是根据历史操作和当前设备状态去执行一系列操作来完成用户的任务。

###你可以用的操作以及对应功能如下
- Click(box=(x1,y1))
>>点击操作，点击屏幕上的指定位置。坐标区间从左上角(0,0)到右下角(999,999)。
- Drag(start=(x1,y1), end=(x2,y2))
>>拖动操作，从起始坐标长按数秒之后拖动到结束坐标。用于调整app布局，滑动滑块验证码等。
- Scroll(start=(x1,y1), end=(x2,y2))
>>滑动操作，从起始坐标拖动到结束坐标。用于滚动查找内容，切换选项卡，下拉通知栏等。坐标区间从左上角(0,0)到右下角(999,999)。
- Type(content='')
>>输入操作，在当前激活的输入框输入指定内容。
- Launch(app='')
>>启动目标app。当目标app在当前界面不可见时，可以使用该动作打开app。
- Wait()
>>等待页面加载。
- Finished(content='')
>>任务结束，退出设备接管。
- CallUser(content='')
>>回答用户的问题或者当前界面有多个符合要求的选项时需要用户接管。
- LongPress(box=(x1,y1))
>>长按操作，在指定位置长按一定的时间。该操作可以触发更多功能选项，例如复制、转发消息，删除等。坐标区间从左上角(0,0)到右下角(999,999)。
- PressBack()
>>返回上一个界面，一般用于错误回退或继续执行剩余任务。
- PressHome()
>>返回系统桌面，一般用于跨app任务中快速打开下一个app或遇到严重错误时回退到系统桌面。
- PressEnter()
>>回车操作，用于换行或者在搜索框中输入内容之后执行搜索操作。
- PressRecent()
>>打开系统后台界面。

###用户任务
{user_task}

###先前的动作和推理过程
{previous_actions}

###输出格式
<think>你的思考过程</think>
<action>执行的操作</action>
<conclusion>总结当前操作</conclusion>

###额外的提示
-输入内容之前，确保输入框已经被激活（出现键盘或者'ADB Keyboard {{ON}}'字样代表输入框已经激活）。
-在app内找不到任务要求的入口时，尝试使用搜索功能，或者如果当前页面上方存在选多个项卡，尝试使用Scroll操作查看。
-如果在执行任务的过程中进入到和任务无关的界面，使用PressBack进行回退。
-任务结束之前，确保已经完整准确地完成用户的任务，如果存在漏做、错做的内容，需要返回重新执行。
'''

class UIVenusProcessor(BaseProcessor):
    """UIVenus处理器 - 负责生成提示词"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.action_description = []
    
    def process(self, state: Dict[str, Any], step: int, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """处理当前状态，生成模型输入
        
        Args:
            state: 包含截图和任务的状态信息
            step: 当前步数
            history: 历史操作记录
            
        Returns:
            包含截图和提示词的字典
        """
        purpose = state['purpose']
        input_text = self._get_input_text(purpose, step, history)
        return {"screenshot_str": state["screenshot_str"], "user_query": input_text}
    
    def _get_input_text(self, purpose: str, step: int, history: List[Dict[str, Any]]) -> str:
        """生成模型输入文本
        
        Args:
            purpose: 任务目标
            step: 当前步数
            history: 历史操作记录
            
        Returns:
            格式化的提示词文本
        """
        if len(history) > 0:
            self.action_description.append(history[-1]['action_description'])
        
        formatted_list = [f"Step {i}: {item}" for i, item in enumerate(self.action_description, start=0)]
        action_list = "\n".join(formatted_list)
        
        return USER_PROMPT.format(user_task=purpose, previous_actions=action_list)
    
    def reset(self):
        """重置处理器状态"""
        self.action_description = []
