"""Dependency-free client for OpenAI-compatible chat completion servers."""

from __future__ import annotations

import json
import math
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from .base import BackendError, BaseBackend


_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429})


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never forward POST bodies or credentials through HTTP redirects."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        status_code: int,
        message: str,
        headers: Mapping[str, str],
        new_url: str,
    ) -> None:
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


class _HTTPStatusFailure(Exception):
    """Internal HTTP failure that intentionally does not retain a response body."""

    def __init__(self, status_code: int, retry_after: Optional[str] = None) -> None:
        super().__init__(status_code)
        self.status_code = status_code
        self.retry_after = retry_after


class OpenAICompatibleBackend(BaseBackend):
    """Call an OpenAI-compatible ``/chat/completions`` endpoint with retries."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 300.0,
        max_retries: int = 3,
        generation_params: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        *,
        backoff_factor: float = 1.0,
        max_backoff: float = 30.0,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if (
            not math.isfinite(backoff_factor)
            or not math.isfinite(max_backoff)
            or backoff_factor < 0
            or max_backoff < 0
        ):
            raise ValueError("retry backoff values must be finite and non-negative")

        self._endpoint = self._build_endpoint(base_url)
        self.model = model
        self._api_key = api_key
        self.timeout = float(timeout)
        self.max_retries = max_retries
        self.generation_params = dict(generation_params or {})
        self.extra_body = dict(extra_body or {})
        self.backoff_factor = float(backoff_factor)
        self.max_backoff = float(max_backoff)

    @staticmethod
    def _build_endpoint(base_url: str) -> str:
        parsed = urllib.parse.urlsplit(base_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(
                "base_url must not contain userinfo; use --api-key or OPENAI_API_KEY"
            )
        path = parsed.path.rstrip("/")
        if not path.endswith("/chat/completions"):
            path += "/chat/completions"
        return urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment)
        )

    def infer(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Submit messages and normalize the first returned completion choice."""

        started_at = time.perf_counter()
        request_data = self._encode_payload(messages)
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key

        for attempt in range(1, self.max_retries + 2):
            try:
                response_data = self._request_once(request_data, headers)
            except _HTTPStatusFailure as exc:
                retryable = self._is_retryable_status(exc.status_code)
                if retryable and attempt <= self.max_retries:
                    self._sleep_before_retry(attempt, exc.retry_after)
                    continue
                raise BackendError(
                    "OpenAI-compatible request failed with HTTP "
                    f"{exc.status_code} after {attempt} attempt(s)",
                    status_code=exc.status_code,
                    attempts=attempt,
                    retryable=retryable,
                    error_type="http_error",
                ) from None
            except (
                urllib.error.URLError,
                TimeoutError,
                socket.timeout,
                ConnectionError,
                OSError,
            ) as exc:
                if attempt <= self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise BackendError(
                    "OpenAI-compatible request failed because of a network error "
                    f"after {attempt} attempt(s)",
                    attempts=attempt,
                    retryable=True,
                    error_type="network_error",
                    cause_type=type(exc).__name__,
                ) from None

            try:
                payload = self._decode_response(response_data, attempt)
                normalized = self._normalize_response(
                    payload,
                    attempts=attempt,
                    latency=time.perf_counter() - started_at,
                )
            except BackendError:
                if attempt <= self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise
            if not normalized["model_output"].strip():
                raise BackendError(
                    "OpenAI-compatible server returned an empty model response",
                    attempts=attempt,
                    retryable=False,
                    error_type="empty_response",
                )
            return normalized

        raise AssertionError("unreachable retry state")

    def _encode_payload(self, messages: Sequence[Mapping[str, Any]]) -> bytes:
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            raise BackendError(
                "messages must be a sequence of mappings",
                error_type="invalid_request",
            )
        normalized_messages = list(messages)
        if not all(isinstance(message, Mapping) for message in normalized_messages):
            raise BackendError(
                "every message must be a mapping",
                error_type="invalid_request",
            )

        payload: dict[str, Any] = {}
        payload.update(self.generation_params)
        payload.update(self.extra_body)
        payload["model"] = self.model
        payload["messages"] = normalized_messages
        payload["stream"] = False
        try:
            return json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise BackendError(
                "request payload is not JSON serializable",
                error_type="invalid_request",
                cause_type=type(exc).__name__,
            ) from None

    def _request_once(self, data: bytes, headers: Mapping[str, str]) -> bytes:
        request = urllib.request.Request(
            self._endpoint,
            data=data,
            headers=dict(headers),
            method="POST",
        )
        try:
            # urllib's default redirect handler can turn a POST into a GET and
            # forward Authorization across origins. Treat every 3xx as a
            # normal, non-retryable HTTP failure instead.
            response = _NO_REDIRECT_OPENER.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            status_code = int(exc.code)
            exc.close()
            raise _HTTPStatusFailure(status_code, retry_after) from None

        try:
            status_code = self._response_status(response)
            retry_after = (
                response.headers.get("Retry-After")
                if getattr(response, "headers", None) is not None
                else None
            )
            body = response.read()
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

        if not 200 <= status_code < 300:
            raise _HTTPStatusFailure(status_code, retry_after)
        return body

    @staticmethod
    def _response_status(response: Any) -> int:
        status = getattr(response, "status", None)
        if status is None:
            getcode = getattr(response, "getcode", None)
            status = getcode() if callable(getcode) else 200
        return int(status)

    @staticmethod
    def _decode_response(data: bytes, attempts: int) -> Mapping[str, Any]:
        try:
            payload = json.loads(data.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendError(
                "OpenAI-compatible server returned invalid JSON",
                attempts=attempts,
                retryable=True,
                error_type="invalid_response",
                cause_type=type(exc).__name__,
            ) from None
        if not isinstance(payload, Mapping):
            raise BackendError(
                "OpenAI-compatible server returned a non-object JSON response",
                attempts=attempts,
                retryable=True,
                error_type="invalid_response",
            )
        return payload

    @staticmethod
    def _normalize_response(
        payload: Mapping[str, Any], *, attempts: int, latency: float
    ) -> dict[str, Any]:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise BackendError(
                "OpenAI-compatible response has no valid completion choice",
                attempts=attempts,
                retryable=True,
                error_type="invalid_response",
            )
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise BackendError(
                "OpenAI-compatible response choice has no valid message",
                attempts=attempts,
                retryable=True,
                error_type="invalid_response",
            )

        reasoning = message.get("reasoning_content")
        if reasoning is None:
            reasoning = message.get("reasoning")
        if reasoning is None:
            reasoning = choice.get("reasoning_content")
        usage = payload.get("usage")
        if not isinstance(usage, Mapping):
            usage = {}
        return {
            "model_output": OpenAICompatibleBackend._extract_text(message.get("content")),
            "reasoning_content": OpenAICompatibleBackend._extract_text(reasoning),
            "usage": dict(usage),
            "finish_reason": choice.get("finish_reason"),
            "request_id": payload.get("id"),
            "latency": float(latency),
            "attempts": attempts,
        }

    @staticmethod
    def _extract_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, Mapping):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    elif isinstance(text, Mapping) and isinstance(text.get("value"), str):
                        parts.append(text["value"])
            return "".join(parts)
        return str(content)

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        return status_code in _RETRYABLE_STATUS_CODES or 500 <= status_code <= 599

    def _sleep_before_retry(
        self, failed_attempt: int, retry_after: Optional[str] = None
    ) -> None:
        delay = self.backoff_factor * (2 ** (failed_attempt - 1))
        if retry_after is not None:
            try:
                delay = max(delay, float(retry_after))
            except (TypeError, ValueError):
                pass
        delay = min(delay, self.max_backoff)
        if delay > 0:
            time.sleep(delay)
