"""T08-05..09: prompt builder, CURRENT block, peek, golden prompt, tool schemas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from railway.loop.fake_model import FakeModelServer, scripted
from railway.loop.model_client import ModelClient
from railway.loop.prompts import SYSTEM_SUFFIX, build_messages, render_current, system_message
from railway.loop.runner import observe_current, play_episode
from railway.sim.constants import TOOL_BUDGET
from railway.tools.schemas import tool_schemas

from .conftest import INSTRUCTION, real_session_cls
from .fakes import FAKE_REGISTRY, FakeSession

HERE = Path(__file__).parent
REPO_FIXTURES = HERE.parent / "fixtures"


def _turn(i: int):
    call_id = f"call_{i}"
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "kpis", "arguments": "{}"}}],
    }
    tool = {"role": "tool", "tool_call_id": call_id, "content": f"out {i}"}
    return assistant, tool


def test_T08_05_windowed_shape():
    turns = [_turn(i) for i in range(1, 9)]
    win = build_messages("SYS", "MAN", turns, "CUR", "windowed")
    full = build_messages("SYS", "MAN", turns, "CUR", "full")
    assert len(win) == 13 and len(full) == 19
    assert win[0] == {"role": "system", "content": "SYS"}
    assert win[1] == {"role": "user", "content": "MANUAL\nMAN"}
    assert win[-1] == {"role": "user", "content": "CUR"}
    assert win[2] == turns[3][0] and win[-2] == turns[7][1]  # the last 5 turns, in order
    assert full[2] == turns[0][0]
    short = build_messages("SYS", "MAN", turns[:2], "CUR", "windowed")
    assert len(short) == 2 + 4 + 1
    with pytest.raises(ValueError):
        build_messages("SYS", "MAN", turns, "CUR", "sideways")


def test_T08_05_system_message():
    assert SYSTEM_SUFFIX == "Reply with exactly one tool call per message. Do not write prose outside tool calls."
    assert system_message("INSTR") == "INSTR\n\n" + SYSTEM_SUFFIX


def test_T08_06_render_current():
    cur = render_current("t=3 | gridlock: none", "[]", "my notes", 7, 400)
    assert cur == (
        "CURRENT STATE\nt=3 | gridlock: none\nBULLETINS\n[]\nNOTES\nmy notes\n"
        "Budget used: 7/400. Call exactly one tool."
    )
    err = render_current("S", "B", "N", 1, 400, error=True)
    assert err.startswith("ERROR: reply with exactly one tool call.\nCURRENT STATE\n")


def test_T08_06_current_block(session_cls, mini3):
    session = session_cls(mini3)
    session.call("notes", {"text": "hello notes"})
    current = observe_current(session)
    assert f"Budget used: 1/{TOOL_BUDGET}. Call exactly one tool." in current
    # What the tools return (calling them costs budget, which CURRENT doesn't show for these lines).
    compact = session.call("status", {"compact": True})
    bulletins = session.call("bulletins", {})
    assert f"CURRENT STATE\n{compact}\nBULLETINS\n{bulletins}\nNOTES\nhello notes\n" in current


def test_T08_07_peek_free(mini3):
    session = real_session_cls()(mini3)
    before_kpis = json.loads(session.peek("kpis", {}))
    log_len = len(session.log)
    for _ in range(50):
        session.peek("status", {})
    after = json.loads(session.peek("kpis", {}))
    assert after["budget"]["used"] == before_kpis["budget"]["used"]
    assert len(session.log) == log_len
    with pytest.raises(Exception):  # noqa: B017 - the ticket only says "raises"
        session.peek("authorize", {"train": "E1", "station": "S2", "track": "L1"})


def _two_turn_prompt(scenario, session):
    calls = [{"name": "notes", "arguments": {"text": "plan A"}}, {"name": "notes", "arguments": {"text": "plan B"}}]
    with FakeModelServer(scripted(calls)) as srv:
        play_episode(
            scenario, ModelClient(srv.base_url, "fake", None), "windowed", session=session, instruction=INSTRUCTION
        )
        return srv.requests[2]["messages"]


def test_T08_08_golden_prompt_fake(mini3):
    """The prompt builder pinned on a scripted 2-turn episode (fake session outputs, hand-written golden)."""
    got = _two_turn_prompt(mini3, FakeSession(mini3))
    golden = json.loads((HERE / "fixtures" / "golden_prompt_fake.json").read_text(encoding="utf-8"))
    assert got == golden


def test_T08_08_golden_prompt(mini3):
    """Real T05 formats on mini3; golden frozen at tests/fixtures/golden_prompt_mini3.json."""
    session = real_session_cls()(mini3)
    path = REPO_FIXTURES / "golden_prompt_mini3.json"
    if not path.exists():
        pytest.fail(f"T05 has landed: hand-write and freeze {path} (see the T08 gate record)")
    got = _two_turn_prompt(mini3, session)
    assert got == json.loads(path.read_text(encoding="utf-8"))


def test_T08_09_tool_schemas_shape():
    specs = tool_schemas(registry=FAKE_REGISTRY)
    assert [s["function"]["name"] for s in specs] == list(FAKE_REGISTRY)
    auth = next(s for s in specs if s["function"]["name"] == "authorize")
    assert auth["type"] == "function"
    params = auth["function"]["parameters"]
    assert params["type"] == "object" and set(params["required"]) == {"train", "station", "track"}
    assert params.get("additionalProperties") is False
    assert auth["function"]["description"] == "Grant a movement authority. See manual()."
    assert "\n" not in auth["function"]["description"]
    names = [s["function"]["name"] for s in tool_schemas(disabled=["stringline"], registry=FAKE_REGISTRY)]
    assert "stringline" not in names and len(names) == len(FAKE_REGISTRY) - 1


def test_T08_09_tool_schemas(mini3):
    session_cls = real_session_cls()
    from railway.tools import session as session_mod

    specs = tool_schemas()
    assert sorted(s["function"]["name"] for s in specs) == sorted(session_mod.TOOLS)
    assert "stringline" not in [s["function"]["name"] for s in tool_schemas(disabled=["stringline"])]
    session = session_cls(mini3, disabled=("stringline",))
    out = session.call("stringline", {"from_tick": 0, "to_tick": 10})
    assert out.startswith("REFUSED R-BAD-ARGS") and "tool disabled" in out
