"""Offline batched inference with the vLLM Python API."""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from .base import BackendError, BaseBackend


_SUPPORTED_GENERATION_PARAMETERS = frozenset(
    {
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "seed",
        "enable_thinking",
    }
)


class VLLMBackend(BaseBackend):
    """Load one vLLM engine and run multimodal conversations in batches."""

    image_transport = "data-url"
    max_concurrency = 1

    def __init__(
        self,
        model_name_or_path: str,
        generation_params: Optional[Mapping[str, Any]] = None,
        tensor_parallel_size: int = 1,
        batch_size: int = 32,
        max_model_len: int = 50000,
        gpu_memory_utilization: float = 0.95,
        dtype: str = "auto",
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        *,
        _vllm_module: Any = None,
    ) -> None:
        if not isinstance(model_name_or_path, str) or not model_name_or_path.strip():
            raise ValueError("model_name_or_path must be a non-empty string")
        if generation_params is not None and not isinstance(generation_params, Mapping):
            raise ValueError("generation_params must be a mapping")
        self._validate_positive_integer("tensor_parallel_size", tensor_parallel_size)
        self._validate_positive_integer("batch_size", batch_size)
        self._validate_positive_integer("max_model_len", max_model_len)
        if (
            isinstance(gpu_memory_utilization, bool)
            or not isinstance(gpu_memory_utilization, (int, float))
            or not math.isfinite(gpu_memory_utilization)
            or not 0 < gpu_memory_utilization <= 1
        ):
            raise ValueError("gpu_memory_utilization must be finite and in (0, 1]")
        if not isinstance(dtype, str) or not dtype.strip():
            raise ValueError("dtype must be a non-empty string")
        if revision is not None and (
            not isinstance(revision, str) or not revision.strip()
        ):
            raise ValueError("revision must be a non-empty string when provided")
        if not isinstance(trust_remote_code, bool):
            raise ValueError("trust_remote_code must be a boolean")
        self._validate_optional_positive_integer("min_pixels", min_pixels)
        self._validate_optional_positive_integer("max_pixels", max_pixels)
        if (
            min_pixels is not None
            and max_pixels is not None
            and min_pixels > max_pixels
        ):
            raise ValueError("min_pixels must not exceed max_pixels")

        self.model_name_or_path = model_name_or_path.strip()
        self.tensor_parallel_size = tensor_parallel_size
        self.batch_size = batch_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.dtype = dtype.strip()
        self.revision = revision.strip() if revision is not None else None
        self.trust_remote_code = trust_remote_code
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generation_params = self._validate_generation_params(
            generation_params or {}
        )

        self._vllm = self._load_dependency(_vllm_module)
        self._engine, self._sampling_params = self._load_engine()

    @staticmethod
    def _validate_positive_integer(name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def _validate_optional_positive_integer(
        cls, name: str, value: Optional[int]
    ) -> None:
        if value is not None:
            cls._validate_positive_integer(name, value)

    @staticmethod
    def _validate_generation_params(
        generation_params: Mapping[str, Any],
    ) -> dict[str, Any]:
        params = dict(generation_params)
        unsupported = sorted(set(params) - _SUPPORTED_GENERATION_PARAMETERS)
        if unsupported:
            raise ValueError(
                "unsupported vLLM generation parameter(s): "
                + ", ".join(unsupported)
            )

        max_tokens = params.get("max_tokens", 25600)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")

        temperature = params.get("temperature", 0.0)
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature < 0
        ):
            raise ValueError("temperature must be finite and non-negative")

        top_p = params.get("top_p", 0.95)
        if (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not math.isfinite(top_p)
            or not 0 < top_p <= 1
        ):
            raise ValueError("top_p must be finite and in (0, 1]")

        top_k = params.get("top_k", 20)
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k < -1
        ):
            raise ValueError("top_k must be an integer greater than or equal to -1")

        presence_penalty = params.get("presence_penalty", 0.0)
        if (
            isinstance(presence_penalty, bool)
            or not isinstance(presence_penalty, (int, float))
            or not math.isfinite(presence_penalty)
            or not -2 <= presence_penalty <= 2
        ):
            raise ValueError("presence_penalty must be finite and in [-2, 2]")

        seed = params.get("seed", 996)
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer when provided")

        enable_thinking = params.get("enable_thinking")
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise ValueError("enable_thinking must be a boolean when provided")

        return {
            "max_tokens": max_tokens,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": top_k,
            "presence_penalty": float(presence_penalty),
            "seed": seed,
            "enable_thinking": enable_thinking,
        }

    @staticmethod
    def _load_dependency(vllm_module: Any) -> Any:
        if vllm_module is not None:
            return vllm_module
        try:
            return importlib.import_module("vllm")
        except Exception as exc:
            raise BackendError(
                "local vLLM inference requires the vllm package",
                fatal=True,
                error_type="dependency_error",
                cause_type=type(exc).__name__,
            ) from None

    def _load_engine(self) -> tuple[Any, Any]:
        llm_class = getattr(self._vllm, "LLM", None)
        sampling_class = getattr(self._vllm, "SamplingParams", None)
        if not callable(llm_class) or not callable(sampling_class):
            raise BackendError(
                "installed vllm does not provide LLM and SamplingParams",
                fatal=True,
                error_type="dependency_error",
            )

        engine_kwargs: dict[str, Any] = {
            "model": self.model_name_or_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.revision is not None:
            engine_kwargs["revision"] = self.revision
        if self.generation_params["seed"] is not None:
            engine_kwargs["seed"] = self.generation_params["seed"]
        mm_processor_kwargs = {}
        if self.min_pixels is not None:
            mm_processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            mm_processor_kwargs["max_pixels"] = self.max_pixels
        if mm_processor_kwargs:
            engine_kwargs["mm_processor_kwargs"] = mm_processor_kwargs

        sampling_kwargs = {
            "max_tokens": self.generation_params["max_tokens"],
            "temperature": self.generation_params["temperature"],
            "top_p": self.generation_params["top_p"],
            "top_k": self.generation_params["top_k"],
            "presence_penalty": self.generation_params["presence_penalty"],
        }
        if self.generation_params["seed"] is not None:
            sampling_kwargs["seed"] = self.generation_params["seed"]

        try:
            engine = llm_class(**engine_kwargs)
            sampling_params = sampling_class(**sampling_kwargs)
        except Exception as exc:
            raise BackendError(
                "failed to initialize the local vLLM engine",
                fatal=True,
                error_type=(
                    "out_of_memory" if self._is_out_of_memory(exc) else "model_load_error"
                ),
                cause_type=type(exc).__name__,
            ) from None
        return engine, sampling_params

    def infer(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Run one conversation; dataset inference normally uses ``infer_batch``."""

        return self.infer_batch([messages])[0]

    def infer_batch(
        self,
        conversations: Sequence[Sequence[Mapping[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Run one vLLM batch and normalize every request output."""

        if isinstance(conversations, (str, bytes)) or not isinstance(
            conversations, Sequence
        ):
            raise BackendError(
                "conversations must be a sequence",
                fatal=True,
                error_type="invalid_request",
            )
        normalized = [list(messages) for messages in conversations]
        if not normalized:
            return []
        if any(
            not all(isinstance(message, Mapping) for message in messages)
            for messages in normalized
        ):
            raise BackendError(
                "every conversation must contain only mapping messages",
                fatal=True,
                error_type="invalid_request",
            )

        chat_kwargs: dict[str, Any] = {"use_tqdm": False}
        enable_thinking = self.generation_params["enable_thinking"]
        if enable_thinking is not None:
            chat_kwargs["chat_template_kwargs"] = {
                "enable_thinking": enable_thinking
            }

        started_at = time.perf_counter()
        try:
            outputs = self._engine.chat(
                normalized,
                sampling_params=self._sampling_params,
                **chat_kwargs,
            )
        except Exception as exc:
            raise BackendError(
                "local vLLM batch inference failed",
                attempts=1,
                fatal=True,
                error_type=(
                    "out_of_memory"
                    if self._is_out_of_memory(exc)
                    else "model_inference_error"
                ),
                cause_type=type(exc).__name__,
            ) from None

        if isinstance(outputs, (str, bytes)) or not isinstance(outputs, Sequence):
            raise BackendError(
                "local vLLM returned a non-sequence response",
                attempts=1,
                fatal=True,
                error_type="invalid_response",
            )
        if len(outputs) != len(normalized):
            raise BackendError(
                "local vLLM returned a different number of outputs than inputs",
                attempts=1,
                fatal=True,
                error_type="invalid_response",
            )

        latency = float(time.perf_counter() - started_at)
        return [self._normalize_output(output, latency) for output in outputs]

    def _normalize_output(self, request_output: Any, latency: float) -> dict[str, Any]:
        candidates = getattr(request_output, "outputs", None)
        if (
            isinstance(candidates, (str, bytes))
            or not isinstance(candidates, Sequence)
            or not candidates
        ):
            raise BackendError(
                "local vLLM response has no completion candidate",
                attempts=1,
                fatal=True,
                error_type="invalid_response",
            )
        candidate = candidates[0]
        text = str(getattr(candidate, "text", ""))
        prompt_tokens = self._token_count(
            getattr(request_output, "prompt_token_ids", None)
        )
        completion_tokens = self._token_count(getattr(candidate, "token_ids", None))
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason is None:
            finish_reason = (
                "length"
                if completion_tokens >= self.generation_params["max_tokens"]
                else "stop"
            )
        request_id = getattr(request_output, "request_id", None)
        return {
            "model_output": text,
            "reasoning_content": self._extract_reasoning(text),
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "finish_reason": finish_reason,
            "request_id": str(request_id) if request_id is not None else None,
            "latency": latency,
            "attempts": 1,
        }

    @staticmethod
    def _token_count(token_ids: Any) -> int:
        if token_ids is None:
            return 0
        if isinstance(token_ids, Sequence) and not isinstance(token_ids, (str, bytes)):
            return len(token_ids)
        return 0

    @staticmethod
    def _is_out_of_memory(error: BaseException) -> bool:
        return "out of memory" in str(error).lower()

    @staticmethod
    def _extract_reasoning(model_output: str) -> str:
        start_marker = "<think>"
        end_marker = "</think>"
        start = model_output.find(start_marker)
        if start < 0:
            return ""
        start += len(start_marker)
        end = model_output.find(end_marker, start)
        if end < 0:
            end = len(model_output)
        return model_output[start:end].strip()


__all__ = ["VLLMBackend"]
