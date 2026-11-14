"""OpenAI-compatible chat client with tool calls (T08 §2.1).

API keys come from environment variables only. They are sent in the Authorization header and are
never logged, stored or put into an error message.
"""

from __future__ import annotations

import functools
import json
import os
import ssl
import time
from typing import TypedDict

import httpx

BACKOFF_S = (1, 2, 4, 8, 16)  # deterministic: retry i sleeps BACKOFF_S[i]

_sleep = time.sleep  # tests monkeypatch this to record the backoff


class ModelError(RuntimeError):
    """The model endpoint failed for good. The runner records the episode as `status: "error"`."""


class RawToolCall(TypedDict):
    id: str
    name: str
    arguments_raw: str


class Usage(TypedDict):
    prompt_tokens: int
    completion_tokens: int


class ModelReply(TypedDict):
    tool_calls: list[RawToolCall]
    content: str | None
    usage: Usage


@functools.cache
def _ssl_context() -> ssl.SSLContext:
    """Loading the CA bundle takes seconds on Windows; do it once per process."""
    return httpx.create_ssl_context()


def _retryable(status: int) -> bool:
    return status == 429 or status >= 500


def parse_reply(data: dict) -> ModelReply:
    """Turn a chat-completions response body into a ModelReply."""
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelError(f"malformed response: no choices[0].message ({type(exc).__name__})") from None
    calls: list[RawToolCall] = []
    for i, tc in enumerate(message.get("tool_calls") or []):
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if args is None:
            args = ""
        elif not isinstance(args, str):  # some servers return an object; keep a raw string either way
            args = json.dumps(args)
        calls.append({"id": str(tc.get("id") or f"call_{i}"), "name": str(fn.get("name") or ""), "arguments_raw": args})
    usage = data.get("usage") or {}
    return {
        "tool_calls": calls,
        "content": message.get("content"),
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


class ModelClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str | None,
        temperature: float | None = None,
        max_tokens: int = 1024,
        timeout_s: float = 120,
        max_retries: int = 5,
    ):
        if max_retries > len(BACKOFF_S):
            raise ValueError(f"max_retries must be ≤ {len(BACKOFF_S)}")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        headers = {"Content-Type": "application/json"}
        if api_key_env:
            key = os.environ.get(api_key_env)
            if not key:
                raise ModelError(f"environment variable {api_key_env} is not set")
            headers["Authorization"] = f"Bearer {key}"
        self._http = httpx.Client(headers=headers, timeout=timeout_s, verify=_ssl_context())

    def __repr__(self) -> str:  # never shows headers
        return f"ModelClient(url={self.url!r}, model={self.model!r})"

    def close(self) -> None:
        self._http.close()

    def request_body(self, messages: list[dict], tools: list[dict]) -> dict:
        body = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens}
        if tools:  # an empty `tools` list is rejected by some servers, and so is tool_choice without tools
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return body

    def chat(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        body = self.request_body(messages, tools)
        attempt = 0
        while True:
            try:
                resp = self._http.post(self.url, json=body)
                status = resp.status_code
                detail = "(body withheld)" if status in (401, 403) else resp.text[:200]  # may echo a masked key
            except httpx.HTTPError as exc:  # connection refused, timeout, ...: treat like a 5xx
                status, detail = 599, type(exc).__name__
            if status == 200:
                try:
                    return parse_reply(resp.json())
                except ValueError:
                    raise ModelError("response is not JSON") from None
            if not _retryable(status):
                raise ModelError(f"HTTP {status}: {detail}")
            if attempt >= self.max_retries:
                raise ModelError(f"HTTP {status} after {attempt} retries: {detail}")
            _sleep(BACKOFF_S[attempt])
            attempt += 1
