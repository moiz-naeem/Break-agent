"""T08-10..16: the episode loop (strict format, windowed/full, record, transcript)."""

from __future__ import annotations

import json

import pytest

from railway.loop import model_client as mc
from railway.loop.fake_model import FakeModelServer, double, flaky, policy, prose, scripted
from railway.loop.model_client import ModelClient
from railway.loop.runner import play_episode
from railway.sim.engine import Simulator

from .conftest import INSTRUCTION, real_session_cls
from .fakes import FakeSession, fake_decide, fake_static

# PRD §7.10, copied here on purpose (the test must not trust the module's own list).
RESULT_FIELDS = [
    "run",
    "harness",
    "mode",
    "model",
    "rep",
    "scenario_id",
    "scenario_hash",
    "seed",
    "preset",
    "rules_version",
    "cost_total",
    "cost_by_component",
    "c_greedy",
    "c_best",
    "score_clipped",
    "score_unclipped",
    "cost_ratio",
    "gridlocks",
    "crew_timeouts",
    "misroutes",
    "unfinished",
    "refusals",
    "invalid_calls",
    "tool_calls",
    "decision_points",
    "tokens_in",
    "tokens_out",
    "wall_s",
    "status",
    "error",
]
DO_NOTHING_MINI3 = 4310.75  # v0.3: every train unfinished, every passenger stop a missed call


def _entry(e) -> dict:
    return e if isinstance(e, dict) else e.model_dump(mode="json")


def _play(scenario, session, behaviour, mode="windowed", **kw):
    with FakeModelServer(behaviour) as srv:
        rec = play_episode(
            scenario, ModelClient(srv.base_url, "fake", None), mode, session=session, instruction=INSTRUCTION, **kw
        )
        return rec, list(srv.requests)


def _play_policy(scenario, session, decide, mode="windowed"):
    static = Simulator.load(scenario).static_info()
    with FakeModelServer(policy(decide, session, static)) as srv:
        return play_episode(
            scenario, ModelClient(srv.base_url, "fake", None), mode, session=session, instruction=INSTRUCTION
        )


def _greedy_cls():
    real_session_cls()
    greedy = pytest.importorskip("railway.policies.baselines", reason="waits for T05 policies")
    if not hasattr(greedy.Greedy, "decide"):
        pytest.skip("waits for Greedy.decide (T08 refactor of T05's greedy.py)")
    return greedy.Greedy


def _t1_dev(count):
    from railway.eval.run import episodes_from_scenarios

    pytest.importorskip("railway.gen.generator", reason="waits for T04 generator")
    pytest.importorskip("railway.ref.seed_filter", reason="waits for T06 accepted lists")
    return episodes_from_scenarios("T1", "dev", count)


def test_T08_10_fake_greedy_equals_greedy(mini3):
    Greedy = _greedy_cls()
    session_cls = real_session_cls()
    from railway.tools.client import InProcClient

    scenarios = [mini3] + [ep.scenario for ep in _t1_dev(2)]
    for sc in scenarios:
        rec = _play_policy(sc, session_cls(sc), Greedy().decide)
        direct = session_cls(sc)
        Greedy().run(InProcClient(direct))
        assert rec.status == "ok"
        assert rec.cost_total == direct.close().cost.total
    assert _play_policy(mini3, session_cls(mini3), Greedy().decide).cost_total == 6.5


def test_T08_11_fake_reference_equals_reference():
    session_cls = real_session_cls()
    ref = pytest.importorskip("railway.ref.reference", reason="waits for T06 reference")
    for ep in _t1_dev(2):
        rec = _play_policy(ep.scenario, session_cls(ep.scenario), ref.Reference().decide)
        assert rec.status == "ok"
        assert rec.cost_total == ep.c_best


def test_T08_12_prose_three_strikes(session_cls, mini3):
    session = session_cls(mini3)
    rec, requests = _play(mini3, session, prose())
    assert len(requests) == 3
    assert rec.format_errors == 3 and rec.tool_calls == 0
    assert session.closed
    assert rec.status == "ok" and rec.cost_total == DO_NOTHING_MINI3
    # strikes 2 and 3 see the error line at the top of CURRENT
    assert requests[1]["messages"][-1]["content"].startswith("ERROR: reply with exactly one tool call.\n")
    assert not requests[0]["messages"][-1]["content"].startswith("ERROR")


def test_T08_12_prose_strike_resets(session_cls, mini3):
    """Two prose replies, then a tool call: the count resets, so the episode goes on."""
    session = session_cls(mini3)
    behaviour = scripted(
        [None, None, {"name": "kpis", "arguments": {}}, None, None, {"name": "end_shift", "arguments": {}}]
    )
    rec, requests = _play(mini3, session, behaviour)
    assert len(requests) == 6 and rec.format_errors == 4 and rec.tool_calls == 2


def test_T08_12_max_turns(session_cls, mini3):
    session = session_cls(mini3)
    rec, requests = _play(mini3, session, scripted([{"name": "kpis", "arguments": {}}] * 10), max_turns=4)
    assert len(requests) == 4 and rec.tool_calls == 4 and session.closed
    assert rec.cost_total == DO_NOTHING_MINI3


def test_T08_13_double_calls(session_cls, mini3):
    session = session_cls(mini3)
    calls = [{"name": "notes", "arguments": {"text": "a"}}, {"name": "notes", "arguments": {"text": "b"}}]
    rec, requests = _play(mini3, session, double(calls))
    assert len(requests) == 3  # notes, notes, end_shift (each paired with a second call)
    assert rec.format_errors == 3
    assert [_entry(e)["tool"] for e in session.log] == ["notes", "notes", "end_shift"]
    assert rec.tool_calls == 3


def test_T08_14_bad_json_args(session_cls, mini3):
    session = session_cls(mini3)
    rec, _ = _play(mini3, session, scripted([{"name": "authorize", "arguments_raw": "{oops"}]))
    first = _entry(session.log[0])
    assert first["tool"] == "authorize" and first["status"] == "INVALID"
    assert (first["rule"] or "").startswith("R-BAD-ARGS")
    assert rec.tool_calls == 2  # the bad call counted, then end_shift
    assert rec.invalid_calls == 1


def test_T08_15_record_fields(session_cls, mini3, tmp_path):
    session = session_cls(mini3)
    path = tmp_path / "transcripts" / "FIX-mini3-r0.jsonl"
    calls = [{"name": "notes", "arguments": {"text": "x"}}, {"name": "kpis", "arguments": {}}]
    rec, requests = _play(mini3, session, scripted(calls), transcript_path=path, run="r", model="fake", rep=0)
    row = rec.to_row()
    for f in RESULT_FIELDS + ["format_errors"]:
        assert f in row, f
    assert row["harness"] == "loop" and row["mode"] == "windowed" and row["status"] == "ok"
    assert row["scenario_id"] == "FIX-mini3" and row["rules_version"] == mini3.rules_version
    assert row["tool_calls"] == 3 and row["decision_points"] is not None
    assert row["tokens_in"] > 0 and row["tokens_out"] > 0
    assert row["cost_by_component"]["total"] == row["cost_total"] == DO_NOTHING_MINI3
    assert row["unfinished"] == 3
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(requests) == 3
    turns = [json.loads(line) for line in lines]
    assert [t["executed"]["tool"] for t in turns] == ["notes", "kpis", "end_shift"]


def test_T08_15_model_error_is_error_record(mini3, monkeypatch):
    monkeypatch.setattr(mc, "_sleep", lambda s: None)
    session = FakeSession(mini3)
    rec, _ = _play(mini3, session, flaky(10, []))
    assert rec.status == "error" and "ModelError" in rec.error
    assert session.closed and rec.to_row()["status"] == "error"


def test_T08_16_windowed_vs_full_same_actions(mini3):
    results = {}
    for mode in ("windowed", "full"):
        session = FakeSession(mini3)
        rec = _play_policy(mini3, session, fake_decide, mode)
        results[mode] = ([(_entry(e)["tool"], _entry(e)["args"]) for e in session.log], rec.cost_total)
    assert results["windowed"] == results["full"]
    assert len(results["windowed"][0]) > 3
    assert fake_static(mini3).shift_ticks == 60


def test_T08_16_windowed_vs_full_greedy(mini3):
    Greedy = _greedy_cls()
    session_cls = real_session_cls()
    results = {}
    for mode in ("windowed", "full"):
        session = session_cls(mini3)
        rec = _play_policy(mini3, session, Greedy().decide, mode)
        results[mode] = ([(_entry(e)["tool"], _entry(e)["args"]) for e in session.log], rec.cost_total)
    assert results["windowed"] == results["full"]
