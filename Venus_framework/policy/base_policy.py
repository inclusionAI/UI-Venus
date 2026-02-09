from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class BasePolicy(ABC):
    """策略基类"""
    
    @abstractmethod
    def get_next_action(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """获取下一步操作
        
        Args:
            state: 当前状态信息
            
        Returns:
            操作信息，任务完成则返回None
        """
        pass
        
    @abstractmethod
    def report_result(self, success: bool) -> None:
        """报告操作结果
        
        Args:
            success: 操作是否成功
        """
        pass 