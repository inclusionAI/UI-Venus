from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseProcessor(ABC):
    """Base state processor interface."""

    @abstractmethod
    def process(self, state: Dict[str, Any], step: int, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process the current state and return input for the policy."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal state when needed."""
        return None
