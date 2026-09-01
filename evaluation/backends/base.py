"""Common interfaces for model inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Sequence


class BackendError(RuntimeError):
    """Inference failure with safe, structured diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        attempts: int = 0,
        retryable: bool = False,
        fatal: bool = False,
        error_type: str = "backend_error",
        cause_type: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.attempts = attempts
        self.retryable = retryable
        self.fatal = fatal
        self.error_type = error_type
        self.cause_type = cause_type

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return metadata safe to persist without request contents or secrets."""

        return {
            "error_type": self.error_type,
            "cause_type": self.cause_type,
            "status_code": self.status_code,
            "attempts": self.attempts,
            "retryable": self.retryable,
            "fatal": self.fatal,
        }


class BaseBackend(ABC):
    """Abstract backend used by the benchmark inference runner."""

    # The inference runner uses these declarations to choose how images are
    # embedded in messages and to enforce backend-specific safety limits.
    image_transport = "data-url"
    max_concurrency: Optional[int] = None
    # Backends that expose infer_batch() set this to a positive integer. The
    # runner then persists each completed batch without using request threads.
    batch_size: Optional[int] = None

    @abstractmethod
    def infer(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Run one chat inference request and return normalized response fields."""
