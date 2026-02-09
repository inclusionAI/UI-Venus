from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseProcessor(ABC):
    """状态处理器基类"""

    @abstractmethod
    def process(self, state: Dict[str, Any], step: int, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """处理当前状态并返回供 policy 使用的状态"""
        raise NotImplementedError

    def reset(self) -> None:
        """重置内部状态（可选）"""
        return None

