"""OpenAI-compatible chat client with tool calls, reasoning and usage/cost tracking (T08 §2.1, Lane F).

API keys come from environment variables only. They are sent in request headers and are
never logged, stored or put into an error message.
"""

from __future__ import annotations

import functools
import json
import os
import re
import ssl
import time
from typing import NamedTuple, NotRequired, TypedDict

import httpx

BACKOFF_S = (1, 2, 4, 8, 16)  # deterministic: retry i sleeps BACKOFF_S[i]

_sleep = time.sleep  # tests monkeypatch this to record the backoff

_THINK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*(.*)$", re.DOTALL)


class ModelError(RuntimeError):
    """The model endpoint failed for good. The runner records the episode as `status: "error"`."""


class Price(NamedTuple):
    """USD per 1,000,000 tokens."""

    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0


# Mutable price table (USD per 1M tokens). Callers may update or add entries at runtime.
PRICE_TABLE: dict[str, Price] = {
    # Anthropic (https://platform.claude.com/docs/en/about-claude/pricing)
    "claude-fable-5-2": Price(input=8.0, output=40.0, cache_read=0.20, cache_write=10.0),
    "claude-mythos-5-2": Price(input=8.0, output=40.0, cache_read=0.20, cache_write=10.0),
    "claude-fable-5-1": Price(input=10.0, output=50.0, cache_read=0.25, cache_write=12.50),
    "claude-opus-5-5": Price(input=4.0, output=20.0, cache_read=0.20, cache_write=5.0),
    "claude-sonnet-5-5": Price(input=2.0, output=10.0, cache_read=0.10, cache_write=2.50),
    "claude-sonnet-5": Price(input=2.0, output=10.0, cache_read=0.20, cache_write=2.50),
    "claude-haiku-5-5": Price(input=0.10, output=0.50, cache_read=0.01, cache_write=0.125),
    # Google Gemini (https://ai.google.dev/gemini-api/docs/pricing, Nov 2026 standard tier)
    "gemini-4-carbon": Price(input=2.0, output=10.0, cache_read=0.10, cache_write=0.0),
    "gemini-4-argon": Price(input=2.0, output=10.0, cache_read=0.10, cache_write=0.0),
    "gemini-4-hydrogen": Price(input=1.50, output=7.50, cache_read=0.075, cache_write=0.0),
    "gemini-3.8-flash": Price(input=0.75, output=3.75, cache_read=0.075, cache_write=0.0),
    # NVIDIA Nemotron 3.5 Super VL (Verda / standard inference rate)
    "nemotron-3.5-super-vl": Price(input=0.15, output=0.50, cache_read=0.02, cache_write=0.0),
    "nemotron-3.5-super": Price(input=0.15, output=0.50, cache_read=0.02, cache_write=0.0),
    # Self-hosted Qwen 3.8 27B on Modal vLLM (billed by GPU time, $0 per API token)
    "qwen38": Price(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0),
    "qwen38-nothink": Price(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0),
    "Qwen/Qwen3.8-27B": Price(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0),
}


def lookup_price(model: str) -> Price:
    """Return the `Price` for `model` (with or without a `provider/` prefix), or zero if unlisted."""
    if model in PRICE_TABLE:
        return PRICE_TABLE[model]
    short = model.rsplit("/", 1)[-1]
    return PRICE_TABLE.get(short, Price(0.0, 0.0, 0.0, 0.0))


def compute_cost_usd(
    price: Price,
    *,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Compute turn cost in USD from uncached input, total billed output (incl. thinking), and cache tokens."""
    raw = (
        input_tokens * price.input
        + output_tokens * price.output
        + cached_tokens * price.cache_read
        + cache_write_tokens * price.cache_write
    ) / 1e6
    return round(raw, 8)


class RawToolCall(TypedDict):
    id: str
    name: str
    arguments_raw: str


class Usage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    cost_usd: float


class ModelReply(TypedDict):
    tool_calls: list[RawToolCall]
    content: str | None
    reasoning: str | None
    provider_blocks: list[dict] | None
    usage: Usage
    # Provider's raw stop reason ("tool_calls"/"length", "tool_use"/"max_tokens", "STOP"/"MAX_TOKENS"), so a
    # turn truncated by the output budget can be told apart from a model that chose not to call a tool.
    finish_reason: NotRequired[str | None]


@functools.cache
def _ssl_context() -> ssl.SSLContext:
    """Loading the CA bundle takes seconds on Windows; do it once per process."""
    return httpx.create_ssl_context()


def http_timeout(timeout_s: float) -> httpx.Timeout:
    """`timeout_s` bounds the wait for the (unstreamed) response; connecting should never take long."""
    return httpx.Timeout(timeout_s, connect=min(30.0, timeout_s))


def _retryable(status: int) -> bool:
    return status == 429 or status >= 500


def _extract_openai_reasoning(message: dict) -> tuple[str | None, str | None]:
    """Return (reasoning, content) from an OpenAI-compatible `choices[0].message` object.

    vLLM >= 0.20 (Qwen3 `--reasoning-parser qwen3`, Nemotron) populates `message["reasoning"]`.
    Older vLLM / OpenRouter / DeepSeek servers populate `message["reasoning_content"]` or `thought`.
    If neither field is present and `<think>...</think>` tags remain in `content`, split them cleanly.
    """
    content = message.get("content")
    for key in ("reasoning", "reasoning_content", "thought"):
        val = message.get(key)
        if isinstance(val, str) and val:
            return val, content
        if isinstance(val, list):
            parts = [
                str(b.get("text") or b.get("thinking") or b.get("summary") or "") for b in val if isinstance(b, dict)
            ]
            joined = "\n".join(p for p in parts if p)
            if joined:
                return joined, content
    if isinstance(content, str) and "<think>" in content and "</think>" in content:
        m = _THINK_RE.match(content)
        if m:
            thought = m.group(1).strip()
            rest = m.group(2).strip()
            return (thought or None), (rest or None)
    return None, content


def clean_openai_messages(messages: list[dict]) -> list[dict]:
    """Strip internal/provider-specific keys (e.g. `provider_blocks`) so OpenAI servers see pure messages."""
    cleaned: list[dict] = []
    for msg in messages:
        if "provider_blocks" not in msg and "reasoning" not in msg:
            cleaned.append(msg)
            continue
        copy_msg = {k: v for k, v in msg.items() if k not in ("provider_blocks", "reasoning")}
        cleaned.append(copy_msg)
    return cleaned


def parse_reply(data: dict, *, model: str | None = None, price: Price | None = None) -> ModelReply:
    """Turn a chat-completions response body into a ModelReply."""
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelError(f"malformed response: no choices[0].message ({type(exc).__name__})") from None
    raw_tcs = message.get("tool_calls") or []
    calls: list[RawToolCall] = []
    has_extra = False
    for i, tc in enumerate(raw_tcs):
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if args is None:
            args = ""
        elif not isinstance(args, str):  # some servers return an object; keep a raw string either way
            args = json.dumps(args)
        calls.append({"id": str(tc.get("id") or f"call_{i}"), "name": str(fn.get("name") or ""), "arguments_raw": args})
        if "extra_content" in tc or "thought_signature" in tc or "thoughtSignature" in tc:
            has_extra = True

    reasoning, content = _extract_openai_reasoning(message)

    usage = data.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    comp = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or 0)

    prompt_details = usage.get("prompt_tokens_details") or {}
    comp_details = usage.get("completion_tokens_details") or {}
    cached = int(prompt_details.get("cached_tokens") or usage.get("cached_tokens") or 0)
    reasoning_tok = int(comp_details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0)

    if reasoning_tok == 0 and total > prompt + comp:
        # Gemini's OpenAI shim counts thinking tokens in total_tokens - prompt_tokens - completion_tokens.
        reasoning_tok = total - prompt - comp
        comp = total - prompt

    uncached_in = max(0, prompt - cached)
    visible_out = max(0, comp - reasoning_tok) if comp >= reasoning_tok else comp

    resolved_model = model or str(data.get("model") or "")
    resolved_price = price if price is not None else lookup_price(resolved_model)
    cost_usd = compute_cost_usd(
        resolved_price,
        input_tokens=uncached_in,
        output_tokens=comp,
        cached_tokens=cached,
        cache_write_tokens=0,
    )

    provider_blocks = [dict(tc) for tc in raw_tcs] if has_extra else None

    finish = data["choices"][0].get("finish_reason")

    return {
        "tool_calls": calls,
        "content": content,
        "reasoning": reasoning,
        "provider_blocks": provider_blocks,
        "finish_reason": str(finish) if finish else None,
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": comp,
            "input_tokens": uncached_in,
            "output_tokens": visible_out,
            "reasoning_tokens": reasoning_tok,
            "cached_tokens": cached,
            "cache_write_tokens": 0,
            "cost_usd": cost_usd,
        },
    }


class ModelClient:
    """OpenAI-compatible `/chat/completions` client (works with OpenAI, vLLM Qwen, Verda Nemotron, etc.)."""

    provider: str = "openai"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str | None,
        temperature: float | None = None,
        max_tokens: int = 1024,
        timeout_s: float = 120,
        max_retries: int = 5,
        *,
        provider: str | None = None,
        reasoning_effort: str | None = None,
        extra_body: dict | None = None,
        price: Price | None = None,
    ):
        if max_retries > len(BACKOFF_S):
            raise ValueError(f"max_retries must be ≤ {len(BACKOFF_S)}")
        self.base_url = base_url.rstrip("/")
        self.url = self.base_url + "/chat/completions"
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self.extra_body = dict(extra_body) if extra_body else {}
        self.price = price if price is not None else lookup_price(model)
        if provider is not None:
            self.provider = provider
        elif ".modal.direct" in base_url or model.startswith("qwen"):
            self.provider = "vllm"
        elif "datacrunch.io" in base_url or "nemotron" in model.lower():
            self.provider = "verda"
        elif "generativelanguage.googleapis.com" in base_url:
            self.provider = "gemini-openai"

        headers = {"Content-Type": "application/json"}
        if api_key_env:
            key = os.environ.get(api_key_env)
            if not key:
                raise ModelError(f"environment variable {api_key_env} is not set")
            headers["Authorization"] = f"Bearer {key}"
        self._http = httpx.Client(headers=headers, timeout=http_timeout(timeout_s), verify=_ssl_context())

    def __repr__(self) -> str:  # never shows headers
        return f"ModelClient(url={self.url!r}, model={self.model!r}, provider={self.provider!r})"

    def settings(self) -> dict:
        out: dict = {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.reasoning_effort is not None:
            out["reasoning_effort"] = self.reasoning_effort
        if self.extra_body:
            out["extra_body"] = self.extra_body
        return out

    def close(self) -> None:
        self._http.close()

    def request_body(self, messages: list[dict], tools: list[dict]) -> dict:
        body: dict = {"model": self.model, "messages": clean_openai_messages(messages), "max_tokens": self.max_tokens}
        if tools:  # an empty `tools` list is rejected by some servers, and so is tool_choice without tools
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        if self.extra_body:
            body.update(self.extra_body)
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
                    payload = resp.json()
                except ValueError:
                    raise ModelError("response is not JSON") from None
                return parse_reply(payload, model=self.model, price=self.price)
            if not _retryable(status):
                raise ModelError(f"HTTP {status}: {detail}")
            if attempt >= self.max_retries:
                raise ModelError(f"HTTP {status} after {attempt} retries: {detail}")
            _sleep(BACKOFF_S[attempt])
            attempt += 1
