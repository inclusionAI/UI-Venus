#!/usr/bin/env python3
"""Small loopback HTTP relay for the Venus Chrome extension.

The relay accepts OpenAI-compatible requests at http://127.0.0.1:8765/v1/* and
forwards them to one fixed upstream. Browser-only headers such as Origin are
intentionally not forwarded.

Usage:
  python3 venus_relay.py --upstream-base https://example.com/v1 \
    --allow-origin chrome-extension://<extension-id>
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


MAX_REQUEST_BYTES = 32 * 1024 * 1024
LOCAL_PREFIX = "/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local loopback relay for Venus")
    parser.add_argument(
        "--upstream-base",
        required=True,
        help="Fixed OpenAI-compatible upstream base URL",
    )
    parser.add_argument(
        "--upstream-api-key",
        default=None,
        help="Override the incoming Authorization header with this API key",
    )
    parser.add_argument(
        "--allow-origin",
        required=True,
        help="Exact extension origin, e.g. chrome-extension://<extension-id>",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def make_handler(config: argparse.Namespace) -> type[BaseHTTPRequestHandler]:
    upstream = config.upstream_base.rstrip("/")
    parsed_upstream = urlsplit(upstream)
    if parsed_upstream.scheme not in {"http", "https"} or not parsed_upstream.netloc:
        raise ValueError("--upstream-base must be an absolute HTTP(S) URL")

    class RelayHandler(BaseHTTPRequestHandler):
        server_version = "VenusRelay/1.0"

        def do_OPTIONS(self) -> None:  # noqa: N802
            if not self._path_allowed():
                self._send_json(404, {"error": "Not found"})
                return
            if not self._origin_allowed():
                self._send_json(403, {"error": "Origin is not allowed"})
                return

            self.send_response(204)
            self._send_cors_headers()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_json(200, {"ok": True, "upstream": upstream})
                return
            self._proxy("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._proxy("POST")

        def _proxy(self, method: str) -> None:
            if not self._path_allowed():
                self._send_json(404, {"error": "Only /v1 endpoints are available"})
                return
            if not self._origin_allowed():
                self._send_json(403, {"error": "Origin is not allowed"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "Invalid Content-Length"})
                return

            if content_length < 0 or content_length > MAX_REQUEST_BYTES:
                self._send_json(413, {"error": "Request body is too large"})
                return

            body = self.rfile.read(content_length) if content_length else None
            suffix = self.path[len(LOCAL_PREFIX) :]
            target = upstream + suffix

            headers = {
                "Accept": self.headers.get("Accept", "application/json"),
                "Content-Type": self.headers.get("Content-Type", "application/json"),
                "User-Agent": "venus-local-relay/1.0",
            }
            incoming_auth = self.headers.get("Authorization")
            if config.upstream_api_key is not None:
                headers["Authorization"] = f"Bearer {config.upstream_api_key}"
            elif incoming_auth:
                headers["Authorization"] = incoming_auth

            request = urllib.request.Request(target, data=body, headers=headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=config.timeout) as response:
                    response_body = response.read()
                    self._send_upstream_response(
                        response.status,
                        response.headers.get("Content-Type", "application/octet-stream"),
                        response_body,
                    )
            except urllib.error.HTTPError as error:
                self._send_upstream_response(
                    error.code,
                    error.headers.get("Content-Type", "application/octet-stream"),
                    error.read(),
                )
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                self._send_json(502, {"error": f"Upstream request failed: {error}"})

        def _path_allowed(self) -> bool:
            return self.path == LOCAL_PREFIX or self.path.startswith(f"{LOCAL_PREFIX}/")

        def _origin_allowed(self) -> bool:
            origin = self.headers.get("Origin")
            return origin == config.allow_origin

        def _send_cors_headers(self) -> None:
            origin = self.headers.get("Origin")
            if origin == config.allow_origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

        def _send_upstream_response(
            self, status: int, content_type: str, body: bytes
        ) -> None:
            self.send_response(status)
            self._send_cors_headers()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send_upstream_response(status, "application/json; charset=utf-8", body)

        def log_message(self, message: str, *args: object) -> None:
            # Never log request headers or bodies because they may contain keys.
            sys.stderr.write(f"[venus-relay] {self.address_string()} {message % args}\n")

    return RelayHandler


def main() -> None:
    config = parse_args()
    if not config.allow_origin.startswith("chrome-extension://"):
        raise SystemExit("--allow-origin must start with chrome-extension://")
    server = ThreadingHTTPServer((config.host, config.port), make_handler(config))

    print(f"Venus relay: http://{config.host}:{config.port}/v1")
    print(f"Upstream:    {config.upstream_base}")
    print(f"Allowed:     {config.allow_origin}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Venus relay.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
