"""Unified provider clients for `railway/loop` (Lane F, docs/decisions.md #8).

Every client implements `chat(messages, tools) -> ModelReply` and exposes `.provider`, `.model`,
`.settings()`, and `.close()`.

Supported backends:
- `AnthropicClient`: native `/v1/messages` with adaptive/extended thinking (`display="summarized"`),
  ephemeral prefix caching on `MANUAL`, and round-tripping of signed `thinking` / `redacted_thinking` blocks.
- `GeminiClient`: native Gemini `generateContent` with `thinkingConfig` (`includeThoughts=True`,
  `thinkingLevel`) and round-tripping of `thought` parts and `thoughtSignature`.
- `OpenAICompatClient` (`ModelClient`): OpenAI-compatible `/chat/completions` with reasoning extraction
  for vLLM (`qwen38` `--reasoning-parser qwen3`), NVIDIA Nemotron (`verda` / vLLM), and OpenAI/OpenRouter.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Protocol

import httpx

from railway.loop import model_client as mc
from railway.loop.model_client import (
    BACKOFF_S,
    PRICE_TABLE,
    ModelClient,
    ModelError,
    ModelReply,
    Price,
    RawToolCall,
    Usage,
    _retryable,
    _ssl_context,
    compute_cost_usd,
    lookup_price,
    parse_reply,
)

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
QWEN_THINK_URL = "https://hoaianh517--railway-qwen-think.us-east.modal.direct/v1"
QWEN_NOTHINK_URL = "https://hoaianh517--railway-qwen-nothink.us-east.modal.direct/v1"
VERDA_NEMOTRON_URL = "https://containers.datacrunch.io/nemotron-3-5-super-vl-b300/v1"

OpenAICompatClient = ModelClient

# Canonical benchmark configurations (docs/decisions.md #8).
BENCHMARK_MODELS: dict[str, dict[str, Any]] = {
    "fable-high": {
        "provider": "anthropic",
        "base_url": ANTHROPIC_MESSAGES_URL,
        "model": "claude-fable-5-2",
        "api_key_env": "ANTHROPIC_API_KEY",
        "effort": "high",
        "max_tokens": 32768,
    },
    "qwen38-think": {
        "provider": "vllm",
        "base_url": QWEN_THINK_URL,
        "model": "qwen38",
        "api_key_env": "VLLM_API_KEY",
        "max_tokens": 32768,
    },
    "gemini-carbon-high": {
        "provider": "gemini",
        "base_url": GEMINI_BASE_URL,
        "model": "gemini-4-carbon",
        "api_key_env": "GEMINI_API_KEY",
        "effort": "high",
        "max_tokens": 32768,
    },
    "nemotron-super": {
        "provider": "verda",
        "base_url": VERDA_NEMOTRON_URL,
        "model": "nemotron-3.5-super-vl",
        "api_key_env": "VERDA_INFERENCE_KEY",
        "max_tokens": 32768,
    },
}


class ChatClient(Protocol):
    provider: str
    model: str

    def chat(self, messages: list[dict], tools: list[dict]) -> ModelReply: ...
    def close(self) -> None: ...


def _parse_tool_input(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw or not str(raw).strip():
        return {}
    try:
        parsed = json.loads(str(raw))
    except ValueError:
        return {"__raw__": str(raw)}
    return parsed if isinstance(parsed, dict) else {"__raw__": str(raw)}


def assistant_message_from_reply(reply: ModelReply) -> dict:
    """Build an OpenAI-shaped assistant message dict from a `ModelReply`, keeping `provider_blocks`."""
    msg: dict[str, Any] = {"role": "assistant", "content": reply.get("content")}
    tcs = reply.get("tool_calls") or []
    if tcs:
        msg["tool_calls"] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments_raw"]},
            }
            for tc in tcs
        ]
    if reply.get("provider_blocks") is not None:
        msg["provider_blocks"] = copy.deepcopy(reply["provider_blocks"])
    return msg


# --- Anthropic Messages API (/v1/messages) ---------------------------------------------------------

_ANTHROPIC_BLOCK_TYPES = frozenset(
    {"thinking", "redacted_thinking", "text", "tool_use", "server_tool_use", "compaction"}
)
_BINDING_BETA = "thinking-binding-controls-2026-08-01"


def _is_anthropic_blocks(blocks: Any) -> bool:
    return (
        isinstance(blocks, list)
        and len(blocks) > 0
        and isinstance(blocks[0], dict)
        and blocks[0].get("type") in _ANTHROPIC_BLOCK_TYPES
    )


def _normalize_anthropic_url(base_url: str | None) -> str:
    if not base_url:
        return ANTHROPIC_MESSAGES_URL
    url = base_url.rstrip("/")
    if url.endswith("/messages"):
        return url
    if url.endswith("/v1"):
        return url + "/messages"
    return url + "/v1/messages"


def parse_anthropic_reply(data: dict, *, model: str | None = None, price: Price | None = None) -> ModelReply:
    """Parse a native Anthropic `/v1/messages` JSON response into a `ModelReply`."""
    raw_blocks = data.get("content")
    if not isinstance(raw_blocks, list):
        raise ModelError("malformed Anthropic response: `content` is not a list")

    calls: list[RawToolCall] = []
    texts: list[str] = []
    thoughts: list[str] = []

    for i, blk in enumerate(raw_blocks):
        if not isinstance(blk, dict):
            continue
        btype = blk.get("type")
        if btype == "thinking":
            t_text = blk.get("thinking")
            if isinstance(t_text, str) and t_text:
                thoughts.append(t_text)
        elif btype == "tool_use":
            inp = blk.get("input")
            raw = json.dumps(inp, sort_keys=True) if isinstance(inp, dict) else str(inp or "")
            calls.append(
                {
                    "id": str(blk.get("id") or f"call_{i}"),
                    "name": str(blk.get("name") or ""),
                    "arguments_raw": raw,
                }
            )
        elif btype == "text":
            txt = blk.get("text")
            if isinstance(txt, str) and txt:
                texts.append(txt)

    u = data.get("usage") or {}
    in_tok = int(u.get("input_tokens") or 0)
    cw_tok = int(u.get("cache_creation_input_tokens") or 0)
    cr_tok = int(u.get("cache_read_input_tokens") or 0)
    out_tok = int(u.get("output_tokens") or 0)
    out_details = u.get("output_tokens_details") or {}
    reason_tok = int(
        out_details.get("thinking_tokens")
        or out_details.get("reasoning_tokens")
        or u.get("thinking_tokens")
        or u.get("reasoning_tokens")
        or 0
    )
    visible_out = max(0, out_tok - reason_tok) if out_tok >= reason_tok else out_tok

    resolved_model = model or str(data.get("model") or "")
    resolved_price = price if price is not None else lookup_price(resolved_model)
    cost_usd = compute_cost_usd(
        resolved_price,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cached_tokens=cr_tok,
        cache_write_tokens=cw_tok,
    )

    return {
        "tool_calls": calls,
        "content": "\n".join(texts) if texts else None,
        "reasoning": "\n\n".join(thoughts) if thoughts else None,
        "provider_blocks": copy.deepcopy(raw_blocks),
        "finish_reason": str(data["stop_reason"]) if data.get("stop_reason") else None,
        "usage": {
            "prompt_tokens": in_tok + cw_tok + cr_tok,
            "completion_tokens": out_tok,
            "input_tokens": in_tok,
            "output_tokens": visible_out,
            "reasoning_tokens": reason_tok,
            "cached_tokens": cr_tok,
            "cache_write_tokens": cw_tok,
            "cost_usd": cost_usd,
        },
    }


class AnthropicClient:
    """Native Anthropic `/v1/messages` client with adaptive/extended thinking and prompt caching."""

    provider: str = "anthropic"

    def __init__(
        self,
        base_url: str | None = ANTHROPIC_MESSAGES_URL,
        model: str = "claude-fable-5-2",
        api_key_env: str | None = "ANTHROPIC_API_KEY",
        *,
        effort: str | None = "high",
        thinking_mode: str = "adaptive",
        budget_tokens: int | None = None,
        display: str = "summarized",
        drop_mismatched_thinking: bool = True,
        temperature: float | None = None,
        max_tokens: int = 32768,
        timeout_s: float = 300,
        max_retries: int = 5,
        price: Price | None = None,
    ):
        if max_retries > len(BACKOFF_S):
            raise ValueError(f"max_retries must be ≤ {len(BACKOFF_S)}")
        if thinking_mode not in ("adaptive", "enabled", "disabled"):
            raise ValueError(f"thinking_mode must be 'adaptive', 'enabled' or 'disabled', not {thinking_mode!r}")
        self.url = _normalize_anthropic_url(base_url)
        self.model = model
        self.api_key_env = api_key_env
        self.effort = effort
        self.thinking_mode = thinking_mode
        self.budget_tokens = budget_tokens
        self.display = display
        self.drop_mismatched_thinking = drop_mismatched_thinking
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.price = price if price is not None else lookup_price(model)

        headers: dict[str, str] = {
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        if api_key_env:
            key = os.environ.get(api_key_env)
            if not key:
                raise ModelError(f"environment variable {api_key_env} is not set")
            headers["x-api-key"] = key
        self._http = httpx.Client(headers=headers, timeout=timeout_s, verify=_ssl_context())

    def __repr__(self) -> str:
        return f"AnthropicClient(url={self.url!r}, model={self.model!r}, effort={self.effort!r})"

    def settings(self) -> dict:
        out: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.url,
            "thinking_mode": self.thinking_mode,
            "display": self.display,
            "effort": self.effort,
            "max_tokens": self.max_tokens,
        }
        if self.budget_tokens is not None:
            out["budget_tokens"] = self.budget_tokens
        if self.temperature is not None and self.thinking_mode == "disabled":
            out["temperature"] = self.temperature
        return out

    def close(self) -> None:
        self._http.close()

    def request_body(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        use_block_binding: bool | None = None,
    ) -> dict:
        if use_block_binding is None:
            use_block_binding = self.thinking_mode == "adaptive" and self.drop_mismatched_thinking

        system_text: str | None = None
        anth_msgs: list[dict] = []
        cached_manual = False
        pending_tool_ids: list[str] = []

        def append_user_block(block: dict) -> None:
            if anth_msgs and anth_msgs[-1]["role"] == "user":
                anth_msgs[-1]["content"].append(block)
            else:
                anth_msgs.append({"role": "user", "content": [block]})

        def flush_unanswered_tools() -> None:
            while pending_tool_ids:
                extra_id = pending_tool_ids.pop(0)
                append_user_block(
                    {
                        "type": "tool_result",
                        "tool_use_id": extra_id,
                        "content": "ERROR: ignored; reply with exactly one tool call per turn.",
                        "is_error": True,
                    }
                )

        for msg in messages:
            role = msg["role"]
            if role != "tool" and pending_tool_ids:
                flush_unanswered_tools()

            if role == "system":
                system_text = msg["content"] if system_text is None else f"{system_text}\n\n{msg['content']}"
            elif role == "user":
                text = msg.get("content") or ""
                block: dict[str, Any] = {"type": "text", "text": text}
                if not cached_manual and text.startswith("MANUAL\n"):
                    block["cache_control"] = {"type": "ephemeral"}
                    cached_manual = True
                append_user_block(block)
            elif role == "assistant":
                pblocks = msg.get("provider_blocks")
                if _is_anthropic_blocks(pblocks):
                    # Pass Claude's `thinking`, `redacted_thinking`, `text`, and `tool_use` blocks back unchanged.
                    blocks = copy.deepcopy(pblocks)
                    pending_tool_ids = [str(b["id"]) for b in blocks if b.get("type") == "tool_use" and b.get("id")]
                    anth_msgs.append({"role": "assistant", "content": blocks})
                else:
                    blocks = []
                    if msg.get("content"):
                        blocks.append({"type": "text", "text": msg["content"]})
                    for tc in msg.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        tid = str(tc.get("id") or "call_0")
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": tid,
                                "name": str(fn.get("name") or ""),
                                "input": _parse_tool_input(fn.get("arguments") or ""),
                            }
                        )
                        pending_tool_ids.append(tid)
                    if blocks:
                        anth_msgs.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                tid = pending_tool_ids.pop(0) if pending_tool_ids else str(msg.get("tool_call_id") or "call_0")
                append_user_block(
                    {
                        "type": "tool_result",
                        "tool_use_id": tid,
                        "content": msg.get("content") or "",
                    }
                )

        if pending_tool_ids:
            flush_unanswered_tools()

        body: dict[str, Any] = {
            "model": self.model,
            "messages": anth_msgs,
            "max_tokens": self.max_tokens,
        }
        if system_text:
            body["system"] = system_text
        if tools:
            body["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"]["parameters"],
                }
                for t in tools
            ]
            body["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}

        if self.thinking_mode == "adaptive":
            thinking_cfg: dict[str, Any] = {"type": "adaptive", "display": self.display}
            if use_block_binding:
                thinking_cfg["block_binding"] = {"prefix_mismatch_behavior": "drop_block"}
            body["thinking"] = thinking_cfg
        elif self.thinking_mode == "enabled":
            budget = self.budget_tokens or max(1024, self.max_tokens // 2)
            body["thinking"] = {"type": "enabled", "budget_tokens": budget, "display": self.display}
        else:
            body["thinking"] = {"type": "disabled"}
            if self.temperature is not None:
                body["temperature"] = self.temperature

        if self.effort:
            body["output_config"] = {"effort": self.effort}
        return body

    def chat(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        use_binding = self.thinking_mode == "adaptive" and self.drop_mismatched_thinking
        attempt = 0
        while True:
            body = self.request_body(messages, tools, use_block_binding=use_binding)
            extra_headers = {"anthropic-beta": _BINDING_BETA} if use_binding else {}
            try:
                resp = self._http.post(self.url, json=body, headers=extra_headers)
                status = resp.status_code
                detail = "(body withheld)" if status in (401, 403) else resp.text[:300]
            except httpx.HTTPError as exc:
                status, detail = 599, type(exc).__name__
            if status == 200:
                try:
                    payload = resp.json()
                except ValueError:
                    raise ModelError("response is not JSON") from None
                return parse_anthropic_reply(payload, model=self.model, price=self.price)
            if status == 400 and use_binding and ("block_binding" in detail or _BINDING_BETA in detail):
                # Fallback for mock servers or proxies that don't yet support thinking-binding-controls.
                use_binding = False
                continue
            if not _retryable(status):
                raise ModelError(f"HTTP {status}: {detail}")
            if attempt >= self.max_retries:
                raise ModelError(f"HTTP {status} after {attempt} retries: {detail}")
            mc._sleep(BACKOFF_S[attempt])
            attempt += 1


# --- Native Google Gemini `generateContent` API ----------------------------------------------------


def _is_gemini_parts(blocks: Any) -> bool:
    return (
        isinstance(blocks, list)
        and len(blocks) > 0
        and isinstance(blocks[0], dict)
        and any(k in blocks[0] for k in ("thought", "functionCall", "text", "thoughtSignature", "thought_signature"))
    )


def _normalize_gemini_base(base_url: str | None) -> str:
    if not base_url:
        return GEMINI_BASE_URL
    url = base_url.rstrip("/")
    if url.endswith("/openai"):
        url = url[: -len("/openai")]
    return url


def _to_gemini_schema(schema: dict) -> dict:
    """Convert a Pydantic/OpenAI JSON Schema into Gemini's OpenAPI-compatible Schema object."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if isinstance(s, dict) and s.get("type") != "null"]
        has_null = any(isinstance(s, dict) and s.get("type") == "null" for s in schema["anyOf"])
        if len(non_null) == 1:
            merged = _to_gemini_schema(non_null[0])
            if has_null:
                merged["nullable"] = True
            if "description" in schema and "description" not in merged:
                merged["description"] = schema["description"]
            return merged

    out: dict[str, Any] = {}
    for k, v in schema.items():
        if k in ("additionalProperties", "title", "default", "$schema"):
            continue
        if k == "properties" and isinstance(v, dict):
            out["properties"] = {pk: _to_gemini_schema(pv) for pk, pv in v.items()}
        elif k == "items" and isinstance(v, dict):
            out["items"] = _to_gemini_schema(v)
        else:
            out[k] = copy.deepcopy(v)
    if out.get("type") == "object":
        out.setdefault("properties", {})
    return out


def parse_gemini_reply(data: dict, *, model: str | None = None, price: Price | None = None) -> ModelReply:
    """Parse a native Gemini `generateContent` JSON response into a `ModelReply`."""
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ModelError(f"malformed Gemini response: no candidates ({str(data)[:200]})")
    content_obj = candidates[0].get("content") or {}
    parts = content_obj.get("parts") or []
    if not isinstance(parts, list):
        raise ModelError("malformed Gemini response: candidates[0].content.parts is not a list")

    calls: list[RawToolCall] = []
    texts: list[str] = []
    thoughts: list[str] = []

    for i, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        if part.get("thought") is True:
            txt = part.get("text")
            if isinstance(txt, str) and txt:
                thoughts.append(txt)
        elif "functionCall" in part and isinstance(part["functionCall"], dict):
            fc = part["functionCall"]
            args = fc.get("args")
            raw = json.dumps(args, sort_keys=True) if isinstance(args, dict) else ("" if args is None else str(args))
            calls.append(
                {
                    "id": str(fc.get("id") or f"call_{i}"),
                    "name": str(fc.get("name") or ""),
                    "arguments_raw": raw,
                }
            )
        else:
            txt = part.get("text")
            if isinstance(txt, str) and txt:
                texts.append(txt)

    um = data.get("usageMetadata") or {}
    prompt_tok = int(um.get("promptTokenCount") or 0)
    cached_tok = int(um.get("cachedContentTokenCount") or 0)
    out_tok = int(um.get("candidatesTokenCount") or 0)
    thought_tok = int(um.get("thoughtsTokenCount") or 0)
    total_tok = int(um.get("totalTokenCount") or 0)
    if thought_tok == 0 and total_tok > prompt_tok + out_tok:
        thought_tok = total_tok - prompt_tok - out_tok

    uncached_in = max(0, prompt_tok - cached_tok)
    comp_tok = out_tok + thought_tok

    resolved_model = model or str(data.get("modelVersion") or "")
    resolved_price = price if price is not None else lookup_price(resolved_model)
    cost_usd = compute_cost_usd(
        resolved_price,
        input_tokens=uncached_in,
        output_tokens=comp_tok,
        cached_tokens=cached_tok,
        cache_write_tokens=0,
    )

    return {
        "tool_calls": calls,
        "content": "\n".join(texts) if texts else None,
        "reasoning": "\n\n".join(thoughts) if thoughts else None,
        "provider_blocks": copy.deepcopy(parts) if parts else None,
        "finish_reason": str(candidates[0]["finishReason"]) if candidates[0].get("finishReason") else None,
        "usage": {
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp_tok,
            "input_tokens": uncached_in,
            "output_tokens": out_tok,
            "reasoning_tokens": thought_tok,
            "cached_tokens": cached_tok,
            "cache_write_tokens": 0,
            "cost_usd": cost_usd,
        },
    }


class GeminiClient:
    """Native Google Gemini `generateContent` client with thought summaries and `thoughtSignature` preservation."""

    provider: str = "gemini"

    def __init__(
        self,
        base_url: str | None = GEMINI_BASE_URL,
        model: str = "gemini-4-carbon",
        api_key_env: str | None = "GEMINI_API_KEY",
        *,
        effort: str | None = "high",
        include_thoughts: bool = True,
        temperature: float | None = None,
        max_tokens: int = 32768,
        timeout_s: float = 300,
        max_retries: int = 5,
        price: Price | None = None,
    ):
        if max_retries > len(BACKOFF_S):
            raise ValueError(f"max_retries must be ≤ {len(BACKOFF_S)}")
        self.base_url = _normalize_gemini_base(base_url)
        self.model = model
        self.url = f"{self.base_url}/models/{model}:generateContent"
        self.api_key_env = api_key_env
        self.effort = effort
        self.include_thoughts = include_thoughts
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.price = price if price is not None else lookup_price(model)

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key_env:
            key = os.environ.get(api_key_env)
            if not key and api_key_env == "GEMINI_API_KEY":
                key = os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
            if not key:
                raise ModelError(f"environment variable {api_key_env} is not set")
            headers["x-goog-api-key"] = key
        self._http = httpx.Client(headers=headers, timeout=timeout_s, verify=_ssl_context())

    def __repr__(self) -> str:
        return f"GeminiClient(url={self.url!r}, model={self.model!r}, effort={self.effort!r})"

    def settings(self) -> dict:
        out: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "effort": self.effort,
            "include_thoughts": self.include_thoughts,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            out["temperature"] = self.temperature
        return out

    def close(self) -> None:
        self._http.close()

    def request_body(self, messages: list[dict], tools: list[dict]) -> dict:
        system_text: str | None = None
        contents: list[dict] = []
        pending_fn_names: list[str] = []

        def append_user_part(part: dict) -> None:
            if contents and contents[-1]["role"] == "user":
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": "user", "parts": [part]})

        def flush_unanswered_fns() -> None:
            while pending_fn_names:
                extra_name = pending_fn_names.pop(0)
                append_user_part(
                    {
                        "functionResponse": {
                            "name": extra_name,
                            "response": {"error": "ignored; reply with exactly one tool call per turn."},
                        }
                    }
                )

        for msg in messages:
            role = msg["role"]
            if role != "tool" and pending_fn_names:
                flush_unanswered_fns()

            if role == "system":
                system_text = msg["content"] if system_text is None else f"{system_text}\n\n{msg['content']}"
            elif role == "user":
                append_user_part({"text": msg.get("content") or ""})
            elif role == "assistant":
                pblocks = msg.get("provider_blocks")
                if _is_gemini_parts(pblocks):
                    parts = copy.deepcopy(pblocks)
                    pending_fn_names = [
                        str(p["functionCall"]["name"])
                        for p in parts
                        if isinstance(p.get("functionCall"), dict) and p["functionCall"].get("name")
                    ]
                    contents.append({"role": "model", "parts": parts})
                else:
                    parts = []
                    if msg.get("content"):
                        parts.append({"text": msg["content"]})
                    for idx, tc in enumerate(msg.get("tool_calls") or []):
                        fn = tc.get("function") or {}
                        name = str(fn.get("name") or "")
                        part: dict[str, Any] = {
                            "functionCall": {
                                "name": name,
                                "args": _parse_tool_input(fn.get("arguments") or ""),
                            }
                        }
                        if idx == 0:
                            part["thoughtSignature"] = "skip_thought_signature_validator"
                        parts.append(part)
                        pending_fn_names.append(name)
                    if parts:
                        contents.append({"role": "model", "parts": parts})
            elif role == "tool":
                fn_name = pending_fn_names.pop(0) if pending_fn_names else str(msg.get("name") or "tool")
                append_user_part(
                    {
                        "functionResponse": {
                            "name": fn_name,
                            "response": {"output": msg.get("content") or ""},
                        }
                    }
                )

        if pending_fn_names:
            flush_unanswered_fns()

        gen_cfg: dict[str, Any] = {"maxOutputTokens": self.max_tokens}
        if self.temperature is not None:
            gen_cfg["temperature"] = self.temperature
        thinking_cfg: dict[str, Any] = {"includeThoughts": self.include_thoughts}
        if self.effort:
            thinking_cfg["thinkingLevel"] = self.effort
        gen_cfg["thinkingConfig"] = thinking_cfg

        body: dict[str, Any] = {"contents": contents, "generationConfig": gen_cfg}
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        if tools:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t["function"]["name"],
                            "description": t["function"].get("description", ""),
                            "parameters": _to_gemini_schema(t["function"]["parameters"]),
                        }
                        for t in tools
                    ]
                }
            ]
        return body

    def chat(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        body = self.request_body(messages, tools)
        attempt = 0
        while True:
            try:
                resp = self._http.post(self.url, json=body)
                status = resp.status_code
                detail = "(body withheld)" if status in (401, 403) else resp.text[:300]
            except httpx.HTTPError as exc:
                status, detail = 599, type(exc).__name__
            if status == 200:
                try:
                    payload = resp.json()
                except ValueError:
                    raise ModelError("response is not JSON") from None
                return parse_gemini_reply(payload, model=self.model, price=self.price)
            if not _retryable(status):
                raise ModelError(f"HTTP {status}: {detail}")
            if attempt >= self.max_retries:
                raise ModelError(f"HTTP {status} after {attempt} retries: {detail}")
            mc._sleep(BACKOFF_S[attempt])
            attempt += 1


# --- Unified Factory & Metadata Helpers ------------------------------------------------------------


def infer_provider(base_url: str | None, model: str) -> str:
    url = (base_url or "").lower()
    m = model.lower()
    if "api.anthropic.com" in url or m.startswith("claude-"):
        return "anthropic"
    if "generativelanguage.googleapis.com" in url or m.startswith("gemini-"):
        return "gemini"
    if ".modal.direct" in url or m.startswith("qwen"):
        return "vllm"
    if "datacrunch.io" in url or "nemotron" in m:
        return "verda"
    return "openai"


def client_provider(client: Any) -> str:
    p = getattr(client, "provider", None)
    if isinstance(p, str) and p:
        return p
    return infer_provider(getattr(client, "url", None), str(getattr(client, "model", "")))


def client_settings(client: Any) -> dict:
    if hasattr(client, "settings") and callable(client.settings):
        return dict(client.settings())
    out: dict[str, Any] = {"provider": client_provider(client)}
    for k in ("model", "max_tokens", "temperature", "effort", "reasoning_effort"):
        val = getattr(client, k, None)
        if val is not None:
            out[k] = val
    return out


def make_client(
    base_url: str | None = None,
    model: str = "",
    api_key_env: str | None = None,
    *,
    provider: str | None = None,
    effort: str | None = None,
    thinking_mode: str = "adaptive",
    include_thoughts: bool = True,
    temperature: float | None = None,
    max_tokens: int = 32768,
    timeout_s: float = 300,
    max_retries: int = 5,
    extra_body: dict | None = None,
    price: Price | None = None,
) -> AnthropicClient | GeminiClient | OpenAICompatClient:
    """Instantiate the right provider client from `(provider, base_url, model)`."""
    resolved = provider or infer_provider(base_url, model)
    if resolved == "anthropic":
        return AnthropicClient(
            base_url=base_url or ANTHROPIC_MESSAGES_URL,
            model=model,
            api_key_env=api_key_env,
            effort=effort,
            thinking_mode=thinking_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            max_retries=max_retries,
            price=price,
        )
    if resolved == "gemini":
        return GeminiClient(
            base_url=base_url or GEMINI_BASE_URL,
            model=model,
            api_key_env=api_key_env,
            effort=effort,
            include_thoughts=include_thoughts,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            max_retries=max_retries,
            price=price,
        )
    if not base_url:
        raise ValueError(f"base_url is required for OpenAI-compatible provider {resolved!r}")
    return OpenAICompatClient(
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        max_retries=max_retries,
        provider=resolved,
        reasoning_effort=effort,
        extra_body=extra_body,
        price=price,
    )


def make_benchmark_client(preset_id: str, **overrides: Any) -> AnthropicClient | GeminiClient | OpenAICompatClient:
    """Instantiate one of the canonical `BENCHMARK_MODELS` (`fable-high`, `qwen38-think`, ...)."""
    if preset_id not in BENCHMARK_MODELS:
        raise KeyError(f"unknown benchmark model {preset_id!r}; expected one of {sorted(BENCHMARK_MODELS)}")
    cfg = {**BENCHMARK_MODELS[preset_id], **overrides}
    return make_client(**cfg)


__all__ = [
    "ANTHROPIC_MESSAGES_URL",
    "BENCHMARK_MODELS",
    "GEMINI_BASE_URL",
    "PRICE_TABLE",
    "QWEN_NOTHINK_URL",
    "QWEN_THINK_URL",
    "VERDA_NEMOTRON_URL",
    "AnthropicClient",
    "ChatClient",
    "GeminiClient",
    "ModelClient",
    "ModelError",
    "ModelReply",
    "OpenAICompatClient",
    "Price",
    "RawToolCall",
    "Usage",
    "assistant_message_from_reply",
    "client_provider",
    "client_settings",
    "compute_cost_usd",
    "infer_provider",
    "lookup_price",
    "make_benchmark_client",
    "make_client",
    "parse_anthropic_reply",
    "parse_gemini_reply",
    "parse_reply",
]
