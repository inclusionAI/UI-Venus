"""Inference backend implementations for VenusBench-CAPTCHA."""

from .base import BackendError, BaseBackend
from .openai_compatible import OpenAICompatibleBackend
from .vllm_backend import VLLMBackend

BACKEND_REGISTRY = {
    "openai-compatible": OpenAICompatibleBackend,
    "vllm": VLLMBackend,
}


def build_backend(name, **kwargs):
    """Construct a registered backend by name."""

    try:
        backend_class = BACKEND_REGISTRY[name]
    except KeyError as error:
        raise ValueError(
            "unknown backend %r (available: %s)"
            % (name, ", ".join(sorted(BACKEND_REGISTRY)))
        ) from error
    return backend_class(**kwargs)


__all__ = [
    "BACKEND_REGISTRY",
    "BackendError",
    "BaseBackend",
    "OpenAICompatibleBackend",
    "VLLMBackend",
    "build_backend",
]
