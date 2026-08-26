from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class BasePolicy(ABC):
    """Base policy interface."""
    
    @abstractmethod
    def get_next_action(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get the next action.
        
        Args:
            state: Current state.
            
        Returns:
            Action data, or None when the task is complete.
        """
        pass
        
    @abstractmethod
    def report_result(self, success: bool) -> None:
        """Report an action result.
        
        Args:
            success: Whether the action succeeded.
        """
        pass
