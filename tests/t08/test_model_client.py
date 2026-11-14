"""T08-01..04: model client against the fake OpenAI-compatible server."""

from __future__ import annotations

import json

import pytest

from railway.eval.manifest import Manifest
from railway.eval.run import LoopDeps, run_manifest
from railway.loop import model_client as mc
from railway.loop.fake_model import FakeModelServer, flaky, scripted
from railway.loop.model_client import ModelClient, ModelError

from .conftest import INSTRUCTION, make_dataset
from .fakes import FakeSession, fake_reward_from

AUTH = {"name": "authorize", "arguments": {"train": "E1", "station": "S2", "track": "L1"}}
MSGS = [{"role": "user", "content": "hi"}]
TOOLS = [
    {
        "type": "function",
        "function": {"name": "authorize", "description": "x", "parameters": {"type": "object", "properties": {}}},
    }
]


@pytest.fixture
def sleeps(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(mc, "_sleep", calls.append)
    return calls


def test_T08_01_tool_call_roundtrip():
    with FakeModelServer(scripted([AUTH])) as srv:
        reply = ModelClient(srv.base_url, "fake-model", None).chat(MSGS, tools=TOOLS)
        body = srv.requests[-1]
    assert len(reply["tool_calls"]) == 1
    call = reply["tool_calls"][0]
    assert call["name"] == "authorize"
    assert json.loads(call["arguments_raw"]) == {"train": "E1", "station": "S2", "track": "L1"}
    assert call["id"]
    assert reply["usage"]["prompt_tokens"] > 0 and reply["usage"]["completion_tokens"] > 0
    assert body["tool_choice"] == "auto" and body["model"] == "fake-model" and body["tools"] == TOOLS


def test_T08_02_temperature_omitted():
    with FakeModelServer(scripted([AUTH, AUTH])) as srv:
        ModelClient(srv.base_url, "m", None, temperature=None).chat(MSGS, tools=[])
        assert "temperature" not in srv.requests[-1]
        ModelClient(srv.base_url, "m", None, temperature=0.7).chat(MSGS, tools=[])
        assert srv.requests[-1]["temperature"] == 0.7
        assert srv.requests[-1]["max_tokens"] == 1024


def test_T08_03_retries(sleeps):
    with FakeModelServer(flaky(2, [AUTH])) as srv:
        reply = ModelClient(srv.base_url, "m", None).chat(MSGS, tools=[])
        assert reply["tool_calls"][0]["name"] == "authorize"
        assert sleeps == [1, 2]
        assert len(srv.requests) == 3
    sleeps.clear()
    with FakeModelServer(flaky(10, [AUTH])) as srv:
        with pytest.raises(ModelError):
            ModelClient(srv.base_url, "m", None).chat(MSGS, tools=[])
        assert sleeps == [1, 2, 4, 8, 16]
        assert len(srv.requests) == 6  # 1 try + 5 retries


def test_T08_03_retries_429_and_4xx(sleeps):
    with FakeModelServer(flaky(1, [AUTH], status=429)) as srv:
        ModelClient(srv.base_url, "m", None).chat(MSGS, tools=[])
        assert sleeps == [1]
    sleeps.clear()
    with FakeModelServer(flaky(1, [AUTH], status=400)) as srv:
        with pytest.raises(ModelError):
            ModelClient(srv.base_url, "m", None).chat(MSGS, tools=[])
        assert sleeps == [] and len(srv.requests) == 1


def test_T08_04_no_key_leak(tmp_path, monkeypatch, sleeps):
    secret = "sk-test-SECRET"
    monkeypatch.setenv("T08_TEST_KEY", secret)
    dataset = make_dataset(tmp_path / "ds", 1)
    calls = [AUTH, {"name": "notes", "arguments": {"text": "plan"}}]
    with FakeModelServer(scripted(calls)) as srv:
        manifest = Manifest.model_validate(
            {
                "name": "leak",
                "harness": "loop",
                "mode": "windowed",
                "model": {"id": "fake", "name": "fake-model", "base_url": srv.base_url, "api_key_env": "T08_TEST_KEY"},
                "dataset": str(dataset),
                "reps": 1,
                "concurrency": 1,
            }
        )
        deps = LoopDeps(session_cls=FakeSession, score_fn=fake_reward_from, instruction=INSTRUCTION)
        run_manifest(manifest, results_dir=tmp_path / "results", deps=deps)
        assert srv.headers[-1].get("authorization") == f"Bearer {secret}"  # the key was really used
    files = [p for p in (tmp_path / "results").rglob("*") if p.is_file()]
    assert any(p.suffix == ".jsonl" and "transcripts" in p.parts for p in files)
    rows = [json.loads(line) for line in (tmp_path / "results" / "leak.jsonl").read_text().splitlines()]
    assert rows and rows[0]["status"] == "ok"
    for p in files:
        assert "SECRET" not in p.read_text(encoding="utf-8"), p


def test_T08_04_missing_key_env_is_model_error(monkeypatch):
    monkeypatch.delenv("T08_UNSET_KEY", raising=False)
    with pytest.raises(ModelError) as err:
        ModelClient("http://127.0.0.1:9/v1", "m", "T08_UNSET_KEY")
    assert "T08_UNSET_KEY" in str(err.value)
