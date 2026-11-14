"""Offline unit tests for Lane F: provider clients, reasoning capture, thinking-block roundtrip,
always-on episode transcript bundles, EventLog capture, and SFT training-sample reconstruction.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from railway.loop import model_client as mc
from railway.loop.model_client import PRICE_TABLE, ModelClient, Price, parse_reply
from railway.loop.providers import (
    AnthropicClient,
    GeminiClient,
    OpenAICompatClient,
    make_client,
    parse_anthropic_reply,
    parse_gemini_reply,
)
from railway.loop.runner import play_episode
from railway.loop.transcripts import (
    TrainingSample,
    load_training_samples,
    load_transcript,
)
from railway.net.events import Event, EventLog
from tests.t08.conftest import INSTRUCTION
from tests.t08.fakes import FakeSession

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "authorize",
            "description": "Grant movement authority.",
            "parameters": {
                "type": "object",
                "properties": {"train": {"type": "string"}, "station": {"type": "string"}, "track": {"type": "string"}},
                "required": ["train", "station", "track"],
                "additionalProperties": False,
            },
        },
    }
]


class RecordedEndpoint:
    """Local HTTP server that serves scripted JSON payloads on any POST path and records requests."""

    def __init__(self, responses: list[tuple[int, dict]]):
        self.queue = deque(responses)
        self.requests: list[dict] = []
        self.paths: list[str] = []
        self.headers: list[dict[str, str]] = []
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}"
                body = json.loads(raw)
                with outer._lock:
                    outer.paths.append(self.path)
                    outer.requests.append(body)
                    outer.headers.append({k.lower(): v for k, v in self.headers.items()})
                    status, payload = outer.queue.popleft() if outer.queue else (500, {"error": "exhausted"})
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> RecordedEndpoint:
        threading.Thread(target=self._server.serve_forever, args=(0.02,), daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def mini3() -> object:
    from railway.sim.model import Scenario

    raw = Path(__file__).resolve().parents[1] / "fixtures" / "mini3.json"
    return Scenario.model_validate_json(raw.read_text(encoding="utf-8"))


# --- 1. Recorded provider response shapes & pricing -----------------------------------------------


def test_parse_anthropic_fable_response_and_cost():
    payload = {
        "id": "msg_01Fable",
        "type": "message",
        "role": "assistant",
        "model": "claude-fable-5-2",
        "content": [
            {
                "type": "thinking",
                "thinking": "E1 enters at YW; S2/L1 is clear and avoids delaying F1.",
                "signature": "EqQBCgIYAhIM_SIGNED_BLOCK_1",
            },
            {
                "type": "redacted_thinking",
                "data": "ENCRYPTED_REDACTED_DATA",
            },
            {
                "type": "tool_use",
                "id": "toolu_01ABC",
                "name": "authorize",
                "input": {"train": "E1", "station": "S2", "track": "L1"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 500,
            "cache_creation_input_tokens": 1000,
            "cache_read_input_tokens": 6000,
            "output_tokens": 250,
        },
    }
    reply = parse_anthropic_reply(payload)
    assert reply["reasoning"] == "E1 enters at YW; S2/L1 is clear and avoids delaying F1."
    assert reply["content"] is None
    assert len(reply["tool_calls"]) == 1
    assert reply["tool_calls"][0]["id"] == "toolu_01ABC"
    assert reply["tool_calls"][0]["name"] == "authorize"
    assert json.loads(reply["tool_calls"][0]["arguments_raw"]) == {"train": "E1", "station": "S2", "track": "L1"}
    # Exact unmodified blocks preserved (including signature and redacted_thinking).
    assert reply["provider_blocks"] == payload["content"]
    u = reply["usage"]
    assert u["prompt_tokens"] == 7500
    assert u["input_tokens"] == 500
    assert u["cache_write_tokens"] == 1000
    assert u["cached_tokens"] == 6000
    assert u["output_tokens"] == 250
    assert u["completion_tokens"] == 250
    # Cost on claude-fable-5-2 ($8 in, $40 out, $0.20 cache_read, $10 cache_write per 1M):
    # (500*8 + 250*40 + 6000*0.20 + 1000*10) / 1e6 = (4000 + 10000 + 1200 + 10000) / 1e6 = 0.0252
    assert u["cost_usd"] == pytest.approx(0.0252)


def test_parse_vllm_qwen_reasoning_shapes():
    # vLLM >= 0.20 (`--reasoning-parser qwen3`) populates `message.reasoning`
    vllm_new = {
        "id": "chatcmpl-qwen-1",
        "model": "qwen38",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "reasoning": "Let's inspect the timetable first before authorizing F2.",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "chatcmpl-tool-01",
                            "type": "function",
                            "function": {"name": "timetable", "arguments": "{}"},
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 6800,
            "completion_tokens": 140,
            "total_tokens": 6940,
            "prompt_tokens_details": {"cached_tokens": 6200},
            "completion_tokens_details": {"reasoning_tokens": 115},
        },
    }
    r1 = parse_reply(vllm_new)
    assert r1["reasoning"] == "Let's inspect the timetable first before authorizing F2."
    assert r1["content"] is None
    assert r1["tool_calls"][0]["name"] == "timetable"
    assert r1["usage"]["cached_tokens"] == 6200
    assert r1["usage"]["input_tokens"] == 600
    assert r1["usage"]["reasoning_tokens"] == 115
    assert r1["usage"]["output_tokens"] == 25
    assert r1["usage"]["cost_usd"] == 0.0  # self-hosted Modal vLLM

    # Fallback when `<think>...</think>` tags remain inside `content`
    unparsed_think = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "<think>\nNeed to hold L1 at S2.\n</think>\n",
                    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "hold", "arguments": "{}"}}],
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 30},
    }
    r2 = parse_reply(unparsed_think)
    assert r2["reasoning"] == "Need to hold L1 at S2."
    assert r2["content"] is None


def test_parse_nemotron_reasoning_content_and_cost():
    # Verda / older vLLM Nemotron response using `reasoning_content`
    nemo_payload = {
        "id": "chatcmpl-nemo-1",
        "model": "nemotron-3.5-super-vl",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "Express E1 has weight 3.0; freight F2 should wait in loop S6/L2.",
                    "tool_calls": [
                        {
                            "id": "chatcmpl-tool-99",
                            "type": "function",
                            "function": {
                                "name": "authorize",
                                "arguments": '{"train": "F2", "station": "S6", "track": "L2"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 10000,
            "completion_tokens": 1000,
            "prompt_tokens_details": {"cached_tokens": 8000},
        },
    }
    reply = parse_reply(nemo_payload)
    assert reply["reasoning"] == "Express E1 has weight 3.0; freight F2 should wait in loop S6/L2."
    assert reply["tool_calls"][0]["name"] == "authorize"
    # Nemotron ($0.15 in, $0.50 out, $0.02 cache_read):
    # uncached_in=2000 -> 2000*0.15 + 8000*0.02 + 1000*0.50 = (300 + 160 + 500)/1e6 = 0.00096
    assert reply["usage"]["cost_usd"] == pytest.approx(0.00096)


def test_parse_gemini_carbon_native_response_and_cost():
    gemini_payload = {
        "modelVersion": "gemini-4-carbon",
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "thought": True,
                            "text": "**Dispatch Analysis**\n\nChecking conflict at S2 between E1 and F1.",
                        },
                        {
                            "functionCall": {
                                "name": "authorize",
                                "args": {"train": "E1", "station": "YE", "track": "main"},
                            },
                            "thoughtSignature": "EpoGCpcGAXLI2nx_GEMINI_SIG_1",
                        },
                    ],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 7000,
            "cachedContentTokenCount": 5000,
            "candidatesTokenCount": 40,
            "thoughtsTokenCount": 360,
            "totalTokenCount": 7400,
        },
    }
    reply = parse_gemini_reply(gemini_payload)
    assert reply["reasoning"] == "**Dispatch Analysis**\n\nChecking conflict at S2 between E1 and F1."
    assert reply["content"] is None
    assert len(reply["tool_calls"]) == 1
    assert reply["tool_calls"][0]["name"] == "authorize"
    assert json.loads(reply["tool_calls"][0]["arguments_raw"]) == {"train": "E1", "station": "YE", "track": "main"}
    assert reply["provider_blocks"][1]["thoughtSignature"] == "EpoGCpcGAXLI2nx_GEMINI_SIG_1"
    u = reply["usage"]
    assert u["prompt_tokens"] == 7000
    assert u["cached_tokens"] == 5000
    assert u["input_tokens"] == 2000
    assert u["output_tokens"] == 40
    assert u["reasoning_tokens"] == 360
    assert u["completion_tokens"] == 400
    # Gemini 4 Carbon ($2.00 in, $10.00 out incl. thoughts, $0.10 cache_read):
    # (2000*2.0 + 5000*0.10 + 400*10.0) / 1e6 = (4000 + 500 + 4000) / 1e6 = 0.0085
    assert u["cost_usd"] == pytest.approx(0.0085)


def test_price_table_is_mutable():
    custom_model = "custom-test-model-v1"
    PRICE_TABLE[custom_model] = Price(input=1.0, output=2.0, cache_read=0.5, cache_write=1.5)
    try:
        r = parse_reply(
            {
                "model": custom_model,
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
            }
        )
        assert r["usage"]["cost_usd"] == pytest.approx((1000 * 1.0 + 500 * 2.0) / 1e6)
    finally:
        PRICE_TABLE.pop(custom_model, None)


# --- 2. Multi-turn thinking block roundtrip (Anthropic & Gemini) ----------------------------------


def test_anthropic_thinking_blocks_passed_back_unchanged_in_episode(mini3, tmp_path, monkeypatch):
    secret = "sk-ant-DO-NOT-LEAK-999"
    monkeypatch.setenv("TEST_ANTH_KEY", secret)

    turn1_blocks = [
        {
            "type": "thinking",
            "thinking": "",
            "signature": "SIG_EMPTY_PREAMBLE_1",
        },
        {
            "type": "thinking",
            "thinking": "Turn 1 reasoning: write initial plan into notes.",
            "signature": "SIG_MAIN_1",
        },
        {
            "type": "redacted_thinking",
            "data": "REDACTED_OPAQUE_BLOB_1",
        },
        {
            "type": "tool_use",
            "id": "toolu_01TURN1",
            "name": "notes",
            "input": {"text": "plan v1"},
        },
    ]
    turn2_blocks = [
        {
            "type": "thinking",
            "thinking": "Turn 2 reasoning: notes saved, now end shift cleanly.",
            "signature": "SIG_MAIN_2",
        },
        {
            "type": "tool_use",
            "id": "toolu_01TURN2",
            "name": "end_shift",
            "input": {},
        },
    ]
    responses = [
        (
            200,
            {
                "id": "msg_1",
                "model": "claude-fable-5-2",
                "content": turn1_blocks,
                "usage": {"input_tokens": 200, "cache_creation_input_tokens": 1000, "output_tokens": 120},
            },
        ),
        (
            200,
            {
                "id": "msg_2",
                "model": "claude-fable-5-2",
                "content": turn2_blocks,
                "usage": {"input_tokens": 250, "cache_read_input_tokens": 1000, "output_tokens": 80},
            },
        ),
    ]

    with RecordedEndpoint(responses) as srv:
        client = AnthropicClient(
            base_url=srv.url,
            model="claude-fable-5-2",
            api_key_env="TEST_ANTH_KEY",
            effort="high",
            max_tokens=16384,
        )
        assert secret not in repr(client)
        session = FakeSession(mini3)
        rec = play_episode(
            mini3,
            client,
            "windowed",
            session=session,
            instruction=INSTRUCTION,
            transcript_root=tmp_path / "store",
            seed_set="dev",
            run="unit-fable",
            model="fable-high",
        )
        client.close()

    assert rec.status == "ok"
    assert len(srv.requests) == 2
    assert srv.paths == ["/v1/messages", "/v1/messages"]
    assert srv.headers[0]["x-api-key"] == secret

    # Request 1 asks for adaptive summarized thinking + high effort + disable_parallel_tool_use + ephemeral cache
    req1 = srv.requests[0]
    assert req1["thinking"]["type"] == "adaptive"
    assert req1["thinking"]["display"] == "summarized"
    assert req1["output_config"] == {"effort": "high"}
    assert req1["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert "temperature" not in req1
    assert req1["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    # Request 2 MUST pass Turn 1's assistant content blocks (thinking + empty thinking + redacted_thinking + tool_use)
    # back completely unmodified, and pair `tool_result.tool_use_id` with `"toolu_01TURN1"`.
    req2 = srv.requests[1]
    assistant_turn_1 = req2["messages"][1]
    assert assistant_turn_1["role"] == "assistant"
    assert assistant_turn_1["content"] == turn1_blocks

    user_after_turn_1 = req2["messages"][2]
    assert user_after_turn_1["role"] == "user"
    tool_res_block = user_after_turn_1["content"][0]
    assert tool_res_block["type"] == "tool_result"
    assert tool_res_block["tool_use_id"] == "toolu_01TURN1"
    assert "OK notes saved" in tool_res_block["content"]

    # Ensure API secret never leaked to any written file
    for p in (tmp_path / "store").rglob("*"):
        if p.is_file():
            assert secret not in p.read_text(encoding="utf-8")


def test_anthropic_double_tool_call_keeps_blocks_unmodified_and_flushes_error_result():
    """If Claude emits two tool_use blocks in one turn, runner executes only the first, and AnthropicClient
    synthesizes an error tool_result for the second so signed blocks remain 100% untouched.
    """
    double_blocks = [
        {"type": "thinking", "thinking": "Calling two tools.", "signature": "SIG_DBL"},
        {"type": "tool_use", "id": "toolu_A", "name": "notes", "input": {"text": "hi"}},
        {"type": "tool_use", "id": "toolu_B", "name": "kpis", "input": {}},
    ]
    client = AnthropicClient(base_url="http://127.0.0.1:9", model="claude-fable-5-2", api_key_env=None)
    try:
        msgs = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "MANUAL\nMAN"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "notes", "arguments": "{}"}}],
                "provider_blocks": double_blocks,
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "OK notes saved"},
            {"role": "user", "content": "CURRENT STATE"},
        ]
        body = client.request_body(msgs, TOOLS)
    finally:
        client.close()

    assert body["messages"][1]["content"] == double_blocks
    user_blocks = body["messages"][2]["content"]
    assert user_blocks[0] == {"type": "tool_result", "tool_use_id": "toolu_A", "content": "OK notes saved"}
    assert user_blocks[1]["type"] == "tool_result"
    assert user_blocks[1]["tool_use_id"] == "toolu_B"
    assert user_blocks[1]["is_error"] is True
    assert user_blocks[2] == {"type": "text", "text": "CURRENT STATE"}


def test_gemini_thought_signatures_passed_back_unchanged_in_episode(mini3, tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_GEM_KEY", "AIza-FAKE-GEMINI-SECRET")
    turn1_parts = [
        {"thought": True, "text": "Inspect KPIs first."},
        {"functionCall": {"name": "kpis", "args": {}}, "thoughtSignature": "GEM_SIG_TURN_1"},
    ]
    turn2_parts = [
        {"thought": True, "text": "KPIs checked; finish shift."},
        {"functionCall": {"name": "end_shift", "args": {}}, "thoughtSignature": "GEM_SIG_TURN_2"},
    ]
    responses = [
        (
            200,
            {
                "candidates": [{"content": {"role": "model", "parts": turn1_parts}}],
                "usageMetadata": {"promptTokenCount": 500, "candidatesTokenCount": 20, "thoughtsTokenCount": 90},
            },
        ),
        (
            200,
            {
                "candidates": [{"content": {"role": "model", "parts": turn2_parts}}],
                "usageMetadata": {"promptTokenCount": 600, "candidatesTokenCount": 15, "thoughtsTokenCount": 65},
            },
        ),
    ]
    with RecordedEndpoint(responses) as srv:
        client = make_client(
            srv.url,
            "gemini-4-carbon",
            "TEST_GEM_KEY",
            provider="gemini",
            effort="high",
        )
        assert isinstance(client, GeminiClient)
        rec = play_episode(
            mini3,
            client,
            "windowed",
            session=FakeSession(mini3),
            instruction=INSTRUCTION,
            transcript_root=tmp_path / "gem_store",
            seed_set="dev",
            run="unit-gemini",
            model="gemini-carbon-high",
        )
        client.close()

    assert rec.status == "ok"
    assert rec.tokens_reasoning == 155
    assert srv.paths[0].endswith("/models/gemini-4-carbon:generateContent")
    req1 = srv.requests[0]
    assert req1["generationConfig"]["thinkingConfig"] == {"includeThoughts": True, "thinkingLevel": "high"}
    req2 = srv.requests[1]
    model_turn_1 = req2["contents"][1]
    assert model_turn_1 == {"role": "model", "parts": turn1_parts}
    fn_resp_part = req2["contents"][2]["parts"][0]
    assert fn_resp_part["functionResponse"]["name"] == "kpis"


# --- 3. Transcript bundle always written (no transcript_path) + EventLog + SFT roundtrip ----------


class _FakeEngineWithLog:
    def __init__(self) -> None:
        self.log = EventLog()
        self.log.emit(0, "ENTER", train="E1", element="YW", booked=0)
        self.log.emit(60, "SHIFT_END")


def test_transcript_bundle_written_without_transcript_path_and_event_log_saved(mini3, tmp_path, monkeypatch):
    monkeypatch.setenv("RAILWAY_TRANSCRIPTS_DIR", str(tmp_path / "railway-transcripts"))

    vllm_turns = [
        (
            200,
            {
                "model": "qwen38",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "reasoning": f"Step {i} reasoning trace.",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"qcall_{i}",
                                    "type": "function",
                                    "function": {"name": tool_name, "arguments": json.dumps(tool_args)},
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 1000 + 100 * i,
                    "completion_tokens": 80,
                    "completion_tokens_details": {"reasoning_tokens": 60},
                },
            },
        )
        for i, (tool_name, tool_args) in enumerate(
            [
                ("notes", {"text": "step 1"}),
                ("notes", {"text": "step 2"}),
                ("notes", {"text": "step 3"}),
                ("notes", {"text": "step 4"}),
                ("notes", {"text": "step 5"}),
                ("notes", {"text": "step 6"}),  # exceeds WINDOW_TURNS=5 to test windowed prompt_ref!
                ("end_shift", {}),
            ],
            start=1,
        )
    ]

    session = FakeSession(mini3)
    session.engine = _FakeEngineWithLog()  # Lane B's engine.log EventLog contract

    with RecordedEndpoint(vllm_turns) as srv:
        client = OpenAICompatClient(srv.url + "/v1", "qwen38", None, provider="vllm", max_tokens=4096)
        # Note: NO `transcript_path` passed!
        rec = play_episode(
            mini3,
            client,
            "windowed",
            session=session,
            instruction=INSTRUCTION,
            seed_set="train",
            run="sft-collect",
            model="qwen38-think",
            rep=2,
            c_greedy=2000.0,
            c_ref=1000.0,
        )
        sent_requests = list(srv.requests)
        client.close()

    expected_dir = tmp_path / "railway-transcripts" / "sft-collect" / "qwen38-think" / "train" / "FIX-mini3" / "2"
    assert Path(rec.transcript_dir) == expected_dir
    assert (expected_dir / "transcript.jsonl").is_file()
    assert (expected_dir / "actions.json").is_file()
    assert (expected_dir / "events.jsonl").is_file()
    assert (expected_dir / "result.json").is_file()

    # Check EventLog (`events.jsonl`) round-trips back to railway.net.events.Event
    event_lines = (expected_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    parsed_events = [Event.model_validate_json(line) for line in event_lines]
    assert [e.kind for e in parsed_events] == ["ENTER", "SHIFT_END"]
    assert parsed_events[0].train == "E1"

    # Check `actions.json` and `result.json`
    actions = json.loads((expected_dir / "actions.json").read_text(encoding="utf-8"))
    assert len(actions["entries"]) == 7
    assert actions["entries"][-1]["tool"] == "end_shift"
    result_row = json.loads((expected_dir / "result.json").read_text(encoding="utf-8"))
    assert result_row["status"] == "ok"
    assert result_row["tokens_reasoning"] == 7 * 60

    # Check `transcript.jsonl` header, turns, and footer
    loaded = load_transcript(expected_dir)
    hdr = loaded.header
    assert hdr["run"] == "sft-collect"
    assert hdr["model"] == "qwen38-think"
    assert hdr["model_name"] == "qwen38"
    assert hdr["provider"] == "vllm"
    assert hdr["seed_set"] == "train"
    assert hdr["scenario_id"] == "FIX-mini3"
    assert hdr["scenario_hash"] == rec.scenario_hash
    assert hdr["rules_version"] == mini3.rules_version
    assert hdr["manual_hash"] and hdr["manual"] == "FAKE MANUAL v1"
    assert hdr["system_prompt"].startswith(INSTRUCTION)
    assert len(hdr["tools"]) == len(FakeSession.registry)

    assert len(loaded.turns) == 7
    ftr = loaded.footer
    assert ftr is not None
    assert ftr["status"] == "ok"
    assert ftr["cost_total"] == rec.cost_total
    assert ftr["reward"] == -rec.cost_total
    assert ftr["cost_by_component"]["total"] == rec.cost_total
    assert "budget" in ftr["kpis"]
    assert ftr["counts"]["unfinished"] == 3
    assert ftr["usage_total"]["reasoning_tokens"] == 420

    # Round-trip to (prompt, reasoning, completion) training samples and check byte-for-byte prompt identity!
    samples = load_training_samples(tmp_path / "railway-transcripts", beat_greedy_only=True)
    assert len(samples) == 7

    for idx, sample in enumerate(samples):
        assert isinstance(sample, TrainingSample)
        prompt, reasoning, completion = sample
        # Prompt messages reconstructed from `prompt_ref` MUST equal the exact wire `messages` sent on turn idx+1!
        assert prompt["messages"] == sent_requests[idx]["messages"]
        assert prompt["tools"] == sent_requests[idx]["tools"]
        assert reasoning == f"Step {idx + 1} reasoning trace."
        assert completion["role"] == "assistant"
        assert len(completion["tool_calls"]) == 1
        chat_msgs = sample.to_chat_messages()
        assert chat_msgs[-1]["reasoning_content"] == f"Step {idx + 1} reasoning trace."
        assert chat_msgs[:-1] == sent_requests[idx]["messages"]


def test_training_sample_filters_format_errors_and_greedy_threshold(mini3, tmp_path):
    responses = [
        # Turn 1: prose format error (0 tool calls)
        (
            200,
            {
                "model": "qwen38",
                "choices": [{"index": 0, "message": {"role": "assistant", "reasoning": "oops", "content": "prose"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        ),
        # Turn 2: valid tool call
        (
            200,
            {
                "model": "qwen38",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "reasoning": "now end shift",
                            "content": None,
                            "tool_calls": [
                                {"id": "c1", "type": "function", "function": {"name": "end_shift", "arguments": "{}"}}
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 120, "completion_tokens": 15},
            },
        ),
    ]
    with RecordedEndpoint(responses) as srv:
        client = ModelClient(srv.url + "/v1", "qwen38", None)
        rec = play_episode(
            mini3,
            client,
            "windowed",
            session=FakeSession(mini3),
            instruction=INSTRUCTION,
            transcript_root=tmp_path / "root",
            run="filter-test",
            model="qwen38",
            c_greedy=100.0,  # DO_NOTHING_MINI3 is 1485.0 > 100.0, so it does NOT beat greedy
            c_ref=50.0,
        )
        client.close()

    # Default skip_format_errors=True keeps only Turn 2
    clean_samples = load_training_samples(rec.transcript_dir)
    assert len(clean_samples) == 1
    assert clean_samples[0].completion["tool_calls"][0]["function"]["name"] == "end_shift"

    # skip_format_errors=False returns both turns
    all_samples = load_training_samples(rec.transcript_dir, skip_format_errors=False)
    assert len(all_samples) == 2

    # beat_greedy_only=True filters out episodes with cost_total >= c_greedy
    assert load_training_samples(rec.transcript_dir, beat_greedy_only=True) == []


def test_retries_work_across_all_provider_clients(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(mc, "_sleep", sleeps.append)

    # Anthropic retry on 503 then 200
    anth_payload = {
        "model": "claude-fable-5-2",
        "content": [{"type": "thinking", "thinking": "ok", "signature": "s"}, {"type": "text", "text": "done"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    with RecordedEndpoint([(503, {"error": "busy"}), (200, anth_payload)]) as srv:
        c = AnthropicClient(base_url=srv.url, model="claude-fable-5-2", api_key_env=None)
        r = c.chat([{"role": "user", "content": "hi"}], [])
        c.close()
    assert r["reasoning"] == "ok" and sleeps == [1]

    sleeps.clear()
    # Gemini retry on 429 then 200
    gem_payload = {
        "candidates": [{"content": {"role": "model", "parts": [{"thought": True, "text": "g-ok"}, {"text": "done"}]}}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "thoughtsTokenCount": 4},
    }
    with RecordedEndpoint([(429, {"error": "rate limit"}), (200, gem_payload)]) as srv:
        c2 = GeminiClient(base_url=srv.url, model="gemini-4-carbon", api_key_env=None)
        r2 = c2.chat([{"role": "user", "content": "hi"}], [])
        c2.close()
    assert r2["reasoning"] == "g-ok" and sleeps == [1]


def test_parse_anthropic_output_tokens_details_thinking_tokens():
    payload = {
        "model": "claude-fable-5-2",
        "content": [
            {"type": "thinking", "thinking": "Checking headway.", "signature": "sig1"},
            {"type": "tool_use", "id": "toolu_1", "name": "kpis", "input": {}},
        ],
        "usage": {
            "input_tokens": 400,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 6000,
            "output_tokens": 200,
            "output_tokens_details": {"thinking_tokens": 175},
        },
    }
    reply = parse_anthropic_reply(payload)
    assert reply["usage"]["completion_tokens"] == 200
    assert reply["usage"]["reasoning_tokens"] == 175
    assert reply["usage"]["output_tokens"] == 25
    assert reply["usage"]["cost_usd"] == pytest.approx((400 * 8.0 + 200 * 40.0 + 6000 * 0.20) / 1e6)


def test_finish_reason_recorded_for_every_provider_shape():
    """A turn that exhausts its output budget on thinking must be distinguishable from a declined tool call."""
    gemini = parse_gemini_reply(
        {
            "candidates": [{"content": {"role": "model", "parts": [{"text": ""}]}, "finishReason": "MAX_TOKENS"}],
            "usageMetadata": {"promptTokenCount": 8665, "candidatesTokenCount": 3, "thoughtsTokenCount": 32765},
        },
        model="gemini-4-carbon",
    )
    assert gemini["finish_reason"] == "MAX_TOKENS"
    assert gemini["tool_calls"] == [] and gemini["reasoning"] is None

    anth = parse_anthropic_reply(
        {"model": "claude-fable-5-2", "content": [{"type": "text", "text": "x"}], "stop_reason": "max_tokens"}
    )
    assert anth["finish_reason"] == "max_tokens"

    oai = parse_reply(
        {"choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": "length"}], "usage": {}},
        model="qwen38",
    )
    assert oai["finish_reason"] == "length"

    no_reason = parse_reply({"choices": [{"message": {"role": "assistant", "content": "hi"}}]}, model="qwen38")
    assert no_reason["finish_reason"] is None


def test_benchmark_clients_wait_long_enough_for_a_full_thinking_turn(monkeypatch):
    """Unstreamed: a 32768-token Qwen turn at ~40 tok/s takes ~14 min, so a 300 s timeout retries forever."""
    from railway.loop.providers import BENCHMARK_MODELS, BENCHMARK_TIMEOUT_S, make_benchmark_client

    for cfg in BENCHMARK_MODELS.values():
        monkeypatch.setenv(cfg["api_key_env"], "test-key")
    for preset_id, cfg in BENCHMARK_MODELS.items():
        client = make_benchmark_client(preset_id)
        try:
            timeout = client._http.timeout
            assert timeout.read == BENCHMARK_TIMEOUT_S >= cfg["max_tokens"] / 25, preset_id
            assert timeout.connect == 30.0, preset_id
        finally:
            client.close()


def test_net_timetable_and_lane_d_session_contract(tmp_path):
    """Verify `scenario_meta`, `extract_event_log`, and `extract_action_log` on v0.3 `Timetable` + Lane D session."""
    from railway.loop.runner import scenario_meta
    from railway.loop.transcripts import extract_action_log, extract_event_log
    from railway.net.model import Call, Move, Network, Node, Platform, Timetable, Track, TrainSpec, Trip
    from railway.util import content_hash

    net = Network(
        nodes=(Node(id="A", kind="station"), Node(id="B", kind="station")),
        tracks=(Track(id="AB", a="A", b="B", length_m=4000, line_speed_kmh=100, electrified=True, direction="both"),),
        platforms=(
            Platform(id="A1", node="A", length_m=300, kind="platform"),
            Platform(id="B1", node="B", length_m=300, kind="platform"),
        ),
        moves=(
            Move(id="mA", node="A", ends=("A1", "AB"), speed_kmh=60, side="b"),
            Move(id="mB", node="B", ends=("AB", "B1"), speed_kmh=60, side="a"),
        ),
    )
    trip = Trip(
        train=TrainSpec(id="IC1", length_m=200, electric=True, passenger=True),
        calls=(Call(platform="A1", depart=0), Call(platform="B1", arrive=10)),
    )
    tt = Timetable(net=net, trips=(trip,), shift_min=60, seed=5, tier="T1")

    meta = scenario_meta(tt)
    assert meta == {
        "scenario_id": "T1-5",
        "scenario_hash": content_hash(tt.model_dump(mode="json")),
        "seed": 5,
        "preset": "T1",
        "rules_version": "0.3-net",
    }

    class _LaneDSessionStub:
        registry = FakeSession.registry

        def __init__(self, timetable: Timetable) -> None:
            self.timetable = timetable
            self.scenario = timetable
            self.engine = _FakeEngineWithLog()
            self.engine.tick = 0
            self.closed = False
            self.used = 0
            self.log: list[dict] = []

        def peek(self, tool_name: str, args: dict) -> str:
            if tool_name == "manual":
                return "NET MANUAL v0.3"
            if tool_name == "status":
                return "t=0" if args.get("compact") else json.dumps({"tick": 0, "notes": "", "trains": []})
            if tool_name == "bulletins":
                return "[]"
            if tool_name == "kpis":
                return json.dumps({"budget": {"used": self.used, "left": 600 - self.used, "total": 600}, "counts": {}})
            raise ValueError(tool_name)

        def call(self, tool_name: str, args: dict) -> str:
            self.used += 1
            self.log.append(
                {"seq": len(self.log) + 1, "tick": 0, "tool": tool_name, "args": args, "status": "INFO", "rule": None}
            )
            if tool_name == "end_shift":
                self.close()
            return "SHIFT ENDED at t=60"

        def close(self) -> dict:
            self.closed = True
            self.engine.tick = 60
            return {
                "tick": 60,
                "cost": {"pax_lateness": 12.5, "total": 12.5},
                "counts": {
                    "gridlocks": 0,
                    "crew_timeouts": 0,
                    "reversals": 0,
                    "missed_calls": 0,
                    "refusals": 0,
                    "invalid_calls": 0,
                    "decision_points": 1,
                    "unfinished": 0,
                },
            }

        def episode_log(self) -> dict:
            self.close()
            return {
                "scenario_id": f"{self.timetable.tier}-{self.timetable.seed}",
                "scenario_hash": content_hash(self.timetable.model_dump(mode="json")),
                "rules_version": "0.3-net",
                "entries": self.log,
                "closed": True,
                "final_tick": 60,
                "final_cost": 12.5,
                "final_cost_by_component": {"pax_lateness": 12.5},
            }

    sess = _LaneDSessionStub(tt)
    ev_list = extract_event_log(sess)
    assert ev_list is not None and [e["kind"] for e in ev_list] == ["ENTER", "SHIFT_END"]

    with RecordedEndpoint(
        [
            (
                200,
                {
                    "model": "qwen38",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "reasoning": "finish shift",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "type": "function",
                                        "function": {"name": "end_shift", "arguments": "{}"},
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                },
            )
        ]
    ) as srv:
        client = OpenAICompatClient(srv.url + "/v1", "qwen38", None)
        rec = play_episode(
            tt,
            client,
            "windowed",
            session=sess,
            instruction=INSTRUCTION,
            transcript_root=tmp_path / "net_bundles",
            seed_set="dev",
            run="laneD-compat",
            model="qwen38-think",
        )
        client.close()

    act_log = extract_action_log(sess, rec, None)
    assert act_log["scenario_id"] == "T1-5" and act_log["rules_version"] == "0.3-net"

    assert rec.status == "ok"
    assert rec.scenario_id == "T1-5"
    assert rec.preset == "T1"
    assert rec.rules_version == "0.3-net"
    bdir = Path(rec.transcript_dir)
    assert bdir == tmp_path / "net_bundles" / "laneD-compat" / "qwen38-think" / "dev" / "T1-5" / "0"
    assert (bdir / "events.jsonl").is_file()
    assert json.loads((bdir / "actions.json").read_text(encoding="utf-8"))["scenario_id"] == "T1-5"
