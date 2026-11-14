"""Tests for `railway/net/session.py`, `formats.py`, `replay.py` and `railway/loop` integration."""

from __future__ import annotations

import json
import re
import sys
import types

import pytest

from railway.loop.fake_model import FakeModelServer, scripted
from railway.loop.model_client import ModelClient
from railway.loop.runner import play_episode
from railway.net.events import EventLog
from railway.net.formats import format_key
from railway.net.manual import INSTRUCTION
from railway.net.model import Call, Connection, Timetable, Trip
from railway.net.replay import ReplayMismatch, replay
from railway.net.rules import NetRule
from railway.net.session import (
    LANE_B_TOOL_SPECS,
    PEEKABLE,
    TOOLS,
    CostBreakdown,
    Session,
    describe,
)
from railway.net.trains import TYPES
from railway.tools.schemas import tool_schemas

from .conftest import NET

EXPECTED_TOOLS = {
    "manual",
    "network",
    "status",
    "conflicts",
    "timetable",
    "bulletins",
    "kpis",
    "stringline",
    "notes",
    "order",
    "hold",
    "platform",
    "advance_to_next_event",
    "end_shift",
}


def test_tool_registration_and_openai_schemas():
    assert set(TOOLS) == EXPECTED_TOOLS
    for name in EXPECTED_TOOLS:
        doc = describe(name)
        assert doc and "\n" not in doc
    specs = tool_schemas(registry=TOOLS)
    assert [s["function"]["name"] for s in specs] == list(TOOLS)
    for spec in specs:
        assert spec["type"] == "function"
        assert spec["function"]["parameters"]["additionalProperties"] is False
    # Disabled tool is omitted from schemas and billed/refused as R-BAD-ARGS if called.
    trimmed = tool_schemas(disabled=("stringline",), registry=TOOLS)
    assert "stringline" not in {s["function"]["name"] for s in trimmed}
    # Lane B extension specs are ready with Args models and status labels, not yet exposed in TOOLS.
    assert set(LANE_B_TOOL_SPECS) == {"route", "skip", "short_turn", "cancel", "single_line", "relieve_crew"}
    assert not (set(LANE_B_TOOL_SPECS) & set(TOOLS))


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        ("teleport", {}),
        ("order", {"resource": "A-JA.u:0"}),
        ("order", {"resource": "A-JA.u:0", "trains": []}),
        ("order", {"resource": "A-JA.u:0", "trains": ["IC1", "IC1"]}),
        ("hold", {"train": "IC1", "until": "ten"}),
        ("status", {"compact": "yes"}),
        ("notes", {"text": "hi", "extra": 1}),
    ],
)
def test_bad_args_and_unknown_tools_are_invalid(sample_timetable, tool_name, args):
    s = Session(sample_timetable)
    before = s.engine.state()
    out = s.call(tool_name, args)
    assert out.startswith("REFUSED R-BAD-ARGS")
    assert (s.log[-1]["status"], s.log[-1]["rule"]) == ("INVALID", "R-BAD-ARGS")
    assert s.used == 1
    assert s.counts["invalid_calls"] == 1
    assert s.engine.state() == before


def test_disabled_tool_is_refused_and_billed(sample_timetable):
    s = Session(sample_timetable, disabled=("stringline",))
    out = s.call("stringline", {"corridor": "PW-C", "from_tick": 0, "to_tick": 20})
    assert out.startswith("REFUSED R-BAD-ARGS")
    assert (s.log[-1]["status"], s.log[-1]["rule"]) == ("INVALID", "R-BAD-ARGS")
    assert s.used == 1


def test_read_tools_happy_paths_and_area_scoping(sample_timetable):
    s = Session(sample_timetable)

    man = s.call("manual", {})
    assert man.startswith("# Network dispatcher manual")
    assert s.log[-1]["status"] == "INFO"

    full_net = json.loads(s.call("network", {}))
    ja_net = json.loads(s.call("network", {"area": "JA"}))
    corridor_net = json.loads(s.call("network", {"area": "JA-PN"}))
    assert len(ja_net["moves"]) < len(full_net["moves"])
    assert {m["id"] for m in ja_net["moves"]} == {"JA.m1", "JA.m2", "JA.m3", "JA.m4"}
    assert {"JA", "N", "PN"} <= {n["id"] for n in corridor_net["nodes"]}

    bad_net = s.call("network", {"area": "NOPE"})
    assert bad_net.startswith("REFUSED R-UNKNOWN-ELEMENT")
    assert s.log[-1]["status"] == "INVALID"

    tt_all = json.loads(s.call("timetable", {}))
    tt_one = json.loads(s.call("timetable", {"train": "IC1"}))
    assert len(tt_all["trains"]) == 3 and len(tt_one["trains"]) == 1
    assert tt_one["trains"][0]["id"] == "IC1" and tt_one["trains"][0]["type"] == "IC"
    bad_tt = s.call("timetable", {"train": "GHOST"})
    assert bad_tt.startswith("REFUSED R-UNKNOWN-TRAIN") and s.log[-1]["status"] == "INVALID"

    st_full = json.loads(s.call("status", {}))
    assert st_full["tick"] == 0 and len(st_full["trains"]) == 3
    st_area = json.loads(s.call("status", {"area": "PORT"}))
    assert st_area["trains"] == []  # F3 starts at B3, not PORT
    st_bad = s.call("status", {"area": "ZZZ", "compact": True})
    assert st_bad.startswith("REFUSED R-UNKNOWN-ELEMENT") and s.log[-1]["status"] == "INVALID"


def test_notes_and_bulletins_and_kpis_guarded_cost_import(sample_timetable, monkeypatch):
    s = Session(sample_timetable)
    assert s.call("notes", {"text": "x" * 1500}).startswith("OK")
    assert s.log[-1]["status"] == "OK"
    too_long = s.call("notes", {"text": "x" * 1501})
    assert too_long.startswith("REFUSED R-NOTES-TOO-LONG")
    assert (s.log[-1]["status"], s.log[-1]["rule"]) == ("REFUSED", "R-NOTES-TOO-LONG")

    s.call("notes", {"text": "hold L4 at A2"})
    assert json.loads(s.call("status", {}))["notes"] == "hold L4 at A2"
    assert 'notes:"hold L4 at A2"' in s.call("status", {"compact": True})

    assert json.loads(s.call("bulletins", {})) == []
    # When Lane B attaches an EventLog with disruption/crew events, bulletins() surfaces them automatically.
    ev_log = EventLog()
    ev_log.emit(0, "DISRUPTION_START", element="JA-N", detail={"id": "D1", "kind": "possession", "train": None})
    s.engine.log = ev_log  # type: ignore[attr-defined]
    shown = json.loads(s.call("bulletins", {}))
    assert len(shown) == 1 and shown[0]["kind"] == "DISRUPTION_START"

    # Without railway.net.cost installed, kpis() works cleanly with cost=None.
    monkeypatch.delitem(sys.modules, "railway.net.cost", raising=False)
    kpi_without = json.loads(s.call("kpis", {}))
    assert kpi_without["cost"] is None
    assert kpi_without["budget"]["used"] == s.used
    assert "on_time_pct" in kpi_without["kpis"]

    # With railway.net.cost present (simulating Lane C after merge), kpis() and close() surface its cost.
    fake_cost = types.ModuleType("railway.net.cost")
    fake_cost.cost = lambda tt, log: CostBreakdown(pax_lateness=12.5, operations=2.0, total=14.5)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "railway.net.cost", fake_cost)
    kpi_with = json.loads(s.call("kpis", {}))
    assert kpi_with["cost"]["total"] == 14.5 and kpi_with["cost"]["pax_lateness"] == 12.5
    final = s.close()
    assert final.cost.total == 14.5
    assert final.cost.model_dump()["operations"] == 2.0


def test_conflicts_order_hold_platform_and_changes_only(sample_timetable, stranded_timetable):
    s = Session(sample_timetable)
    adv1 = s.call("advance_to_next_event", {})
    assert "t=7 stop: CONFLICT IC1, L4" in adv1
    assert "C1 A-JA.u:0" in adv1 and "C2 A.e1/A.e3" in adv1
    assert "IC1 arrived A1" in adv1

    conf_before = s.call("conflicts", {})
    assert "2 conflicts" in conf_before

    # Refusals on order, hold and platform.
    assert s.call("order", {"resource": "NOPE", "trains": ["IC1"]}).startswith("REFUSED R-UNKNOWN-ELEMENT")
    assert s.log[-1]["status"] == "INVALID"
    assert s.call("order", {"resource": "A-JA.u:0", "trains": ["NOPE"]}).startswith("REFUSED R-UNKNOWN-TRAIN")
    assert s.log[-1]["status"] == "INVALID"
    assert s.call("hold", {"train": "NOPE", "until": 20}).startswith("REFUSED R-UNKNOWN-TRAIN")
    assert s.call("hold", {"train": "L4", "until": 7}).startswith("REFUSED R-HOLD-PAST")
    assert s.log[-1]["status"] == "REFUSED"
    assert s.call("platform", {"train": "NOPE", "station": "C", "n": 1}).startswith("REFUSED R-UNKNOWN-TRAIN")
    assert s.call("platform", {"train": "IC1", "station": "ZZ", "n": 1}).startswith("REFUSED R-UNKNOWN-ELEMENT")
    assert s.call("platform", {"train": "IC1", "station": "C", "n": 9}).startswith("REFUSED R-NO-ROUTE")
    assert s.call("platform", {"train": "IC1", "station": "N", "n": 1}).startswith("REFUSED R-NO-ROUTE")

    # Happy paths on order (exact block + reversed move-pair), hold and platform (int and Port alpha-prefix).
    assert s.call("order", {"resource": "A-JA.u:0", "trains": ["IC1", "L4"]}).startswith("GRANTED")
    assert s.log[-1]["status"] == "GRANTED"
    # Halfway: 1 conflict cleared, 1 left.
    assert "1 conflicts" in s.call("conflicts", {})
    assert s.call("order", {"resource": "A.e3/A.e1", "trains": ["IC1", "L4"]}).startswith("GRANTED")
    assert "0 conflicts" in s.call("conflicts", {})

    assert s.call("hold", {"train": "L4", "until": 18}).startswith("OK")
    assert s.call("platform", {"train": "IC1", "station": "C", "n": 2}).startswith("OK")
    assert s.call("platform", {"train": "F3", "station": "PORT", "n": 2}).startswith("OK")

    adv2 = s.call("advance_to_next_event", {})
    # Changes-only: IC1 arrived A1 (from t=7) must NOT be repeated in the next advance.
    assert "IC1 arrived A1" not in adv2
    assert "IC1 departed A1" in adv2
    assert "SHIFT ENDED at t=180" in adv2
    final = s.close()
    assert ("IC1", "done", "C2") in [r[:3] for r in final.state]
    assert ("F3", "done", "P2") in [r[:3] for r in final.state]

    # Once a train is running or done (while shift is still open), hold and platform return R-ALREADY-RUNNING.
    mid_trips = (
        *sample_timetable.trips,
        Trip(train=TYPES["S"].spec("L9"), calls=(Call(platform="N2", depart=120), Call(platform="A2", arrive=160))),
    )
    s_mid = Session(Timetable(net=NET, trips=mid_trips, shift_min=180))
    s_mid.call("advance_to_next_event", {})  # t=7 CONFLICT IC1, L4
    s_mid.call("order", {"resource": "A-JA.u:0", "trains": ["IC1", "L4"]})
    s_mid.call("order", {"resource": "A.e1/A.e3", "trains": ["IC1", "L4"]})
    s_mid.call("advance_to_next_event", {})  # stops at IDLE while L9 is still due at 120; IC1 is done/running
    assert not s_mid.closed
    assert s_mid.engine.trains["IC1"].phase in {"running", "done"}
    assert s_mid.call("hold", {"train": "IC1", "until": 150}).startswith("REFUSED R-ALREADY-RUNNING")
    assert s_mid.log[-1]["status"] == "REFUSED"
    assert s_mid.call("platform", {"train": "IC1", "station": "C", "n": 1}).startswith("REFUSED R-ALREADY-RUNNING")
    assert s_mid.log[-1]["status"] == "REFUSED"

    # Rescue an UNROUTABLE train with platform().
    u = Session(stranded_timetable)
    u_adv = u.call("advance_to_next_event", {})
    assert "UNROUTABLE IC9" in u_adv
    assert u.call("platform", {"train": "IC9", "station": "C", "n": "C1"}).startswith("OK")
    while not u.closed:
        u.call("advance_to_next_event", {})
        for c in u.engine._conflicts(u.engine._wanting(15)):
            u.call("order", {"resource": format_key(c.key), "trains": sorted(c.trains)})
    assert ("IC9", "done", "C1") in [r[:3] for r in u.close().state]


def test_connection_at_risk_shown_in_conflicts():
    trips = (
        Trip(
            train=TYPES["S"].spec("S1"),
            calls=(Call(platform="N1", depart=0), Call(platform="A2", arrive=24, alight=100)),
        ),
        Trip(
            train=TYPES["IC"].spec("IC2"),
            calls=(Call(platform="A1", depart=28, board=100), Call(platform="C1", arrive=65)),
        ),
    )
    conn = Connection(feeder="S1", connector="IC2", node="A", transfers=25, min_transfer_min=5)
    tt = Timetable(net=NET, trips=trips, connections=(conn,), shift_min=100)
    s = Session(tt)
    # Slack is 28 - 24 - 5 = -1 < 0, so the connection is already at risk at t=0.
    text = s.call("conflicts", {})
    assert "CONN A: S1->IC2" in text and "25p" in text


def test_stringline_corridors_projection_and_limits(t1_timetable):
    s = Session(t1_timetable)
    for bad_win in ((0, 61), (12, 8), (-1, 10), (0, 999)):
        out = s.call("stringline", {"corridor": "PW-C", "from_tick": bad_win[0], "to_tick": bad_win[1]})
        assert out.startswith("REFUSED R-WINDOW")
        assert s.log[-1]["status"] == "REFUSED"

    bad_corr = s.call("stringline", {"corridor": "PW-MARS", "from_tick": 0, "to_tick": 20})
    assert bad_corr.startswith("REFUSED R-UNKNOWN-ELEMENT") and s.log[-1]["status"] == "INVALID"

    # Advance a couple of steps on T1, then inspect trunk, branch and train-id corridors.
    for _ in range(3):
        s.call("advance_to_next_event", {})
        for c in s.engine._conflicts(s.engine._wanting(15)):
            s.call("order", {"resource": format_key(c.key), "trains": sorted(c.trains)})

    now = s.engine.tick
    out_trunk = s.call("stringline", {"corridor": "PW-C", "from_tick": 0, "to_tick": min(180, now + 30)})
    assert out_trunk.startswith("corridor PW A JA B JB C\n")
    assert len(out_trunk) <= 2000
    assert any(line.startswith(f">t{now}") for line in out_trunk.splitlines())

    out_branch = s.call("stringline", {"corridor": "JA-PN", "from_tick": 0, "to_tick": 60})
    assert out_branch.startswith("corridor JA N PN\n") and len(out_branch) <= 2000

    first_train = t1_timetable.trips[0].train.id
    out_train = s.call("stringline", {"corridor": first_train, "from_tick": 0, "to_tick": 60})
    assert out_train.startswith("corridor ") and len(out_train) <= 2000


def test_budget_exhaustion_and_close_equals_end_shift(sample_timetable):
    s = Session(sample_timetable, budget=4)
    outs = [s.call("kpis", {}) for _ in range(3)] + [s.call("advance_to_next_event", {})]
    assert f"BUDGET EXHAUSTED ({NetRule.BUDGET.value})" in outs[-1]
    assert s.closed and s.used == 4
    after = s.call("status", {})
    assert after.startswith("REFUSED R-SHIFT-ENDED")
    assert s.used == 4  # post-close calls do not increment budget used
    assert s.log[-1]["status"] == "REFUSED" and s.log[-1]["rule"] == "R-SHIFT-ENDED"

    # Stopping early (close()) produces the exact same final result as calling end_shift().
    a, b = Session(sample_timetable), Session(sample_timetable)
    for sess in (a, b):
        sess.call("advance_to_next_event", {})
        sess.call("order", {"resource": "A-JA.u:0", "trains": ["IC1", "L4"]})
    a.call("end_shift", {})
    b_final = b.close()
    assert a.final == b_final
    assert b.close() is b_final  # idempotent


def test_peek_is_free_and_unlogged(sample_timetable):
    s = Session(sample_timetable)
    for name in PEEKABLE:
        out = s.peek(name, {})
        assert out
    assert s.used == 0 and s.log == []
    with pytest.raises(ValueError, match="read-only"):
        s.peek("order", {"resource": "A-JA.u:0", "trains": ["IC1"]})


def test_replay_determinism_and_tamper_detection(t1_timetable):
    live = Session(t1_timetable)
    while not live.closed:
        live.call("advance_to_next_event", {})
        if live.closed:
            break
        for c in live.engine._conflicts(live.engine._wanting(15)):
            ordered = sorted(c.trains, key=lambda t: (live.engine.trains[t].booked(), t))
            live.call("order", {"resource": format_key(c.key), "trains": ordered})
    log = json.loads(json.dumps(live.episode_log()))
    replayed = replay(t1_timetable, log)
    assert replayed == live.final
    assert replayed.tick == 180
    assert all(phase == "done" for _, phase, _, _ in replayed.state)

    # Tampered status, args or scenario_hash must raise ReplayMismatch.
    bad_status = json.loads(json.dumps(log))
    bad_status["entries"][0]["status"] = "REFUSED"
    with pytest.raises(ReplayMismatch):
        replay(t1_timetable, bad_status)

    bad_args = json.loads(json.dumps(log))
    order_idx = next(i for i, e in enumerate(bad_args["entries"]) if e["tool"] == "order")
    bad_args["entries"][order_idx]["args"]["trains"] = ["NO_SUCH_TRAIN"]
    with pytest.raises(ReplayMismatch):
        replay(t1_timetable, bad_args)

    bad_hash = json.loads(json.dumps(log)) | {"scenario_hash": "0" * 64}
    with pytest.raises(ReplayMismatch):
        replay(t1_timetable, bad_hash)


def test_compact_status_and_advance_output_sizes_on_t1(t1_timetable):
    s = Session(t1_timetable)
    compact_lengths = []
    full_lengths = []
    advance_lengths = []
    while not s.closed:
        adv = s.call("advance_to_next_event", {})
        advance_lengths.append(len(adv))
        if s.closed:
            break
        compact_lengths.append(len(s.peek("status", {"compact": True})))
        full_lengths.append(len(s.peek("status", {"compact": False})))
        for c in s.engine._conflicts(s.engine._wanting(15)):
            ordered = sorted(c.trains, key=lambda t: (s.engine.trains[t].booked(), t))
            s.call("order", {"resource": format_key(c.key), "trains": ordered})

    avg_compact = sum(compact_lengths) / len(compact_lengths)
    avg_full = sum(full_lengths) / len(full_lengths)
    avg_adv = sum(advance_lengths) / len(advance_lengths)
    assert avg_compact < 900 and max(compact_lengths) < 1200
    assert avg_compact < avg_full / 4  # compact status is a small fraction of full JSON state
    assert avg_adv < 700 and max(advance_lengths) < 1200


class _NetTimetableOrderBehaviour:
    """FakeModelServer behaviour that reads CURRENT STATE from the prompt and settles conflicts in book order."""

    def __init__(self, timetable: Timetable):
        self.booked_entry = {t.train.id: t.entry_tick for t in timetable.trips}

    def respond(self, body: dict, n: int) -> tuple[int, dict]:
        from railway.loop.fake_model import _completion

        messages = body["messages"]
        current = messages[-1]["content"]
        # Look for an unresolved conflict line `C1 <resource> (<kind>): <t1> ... | <t2> ...` in CURRENT STATE.
        m = re.search(r"^C\d+\s+(\S+)\s+\([^)]+\):\s+(.+)$", current, re.M)
        if m:
            resource = m.group(1)
            trains = [part.strip().split()[0] for part in m.group(2).split("|")]
            trains.sort(key=lambda t: (self.booked_entry.get(t, 0), t))
            call = {"name": "order", "arguments": {"resource": resource, "trains": trains}}
        else:
            call = {"name": "advance_to_next_event", "arguments": {}}
        return 200, _completion(body, [call], None, n)


def test_loop_play_episode_on_t1_with_fake_model(t1_timetable):
    """End-to-end test: `railway.loop.runner.play_episode` drives `railway.net.session.Session` on T1."""
    # 1. Scripted calls (notes -> kpis -> advance -> end_shift) through FakeModelServer.
    calls = [
        {"name": "notes", "arguments": {"text": "T1 shift start"}},
        {"name": "conflicts", "arguments": {}},
        {"name": "advance_to_next_event", "arguments": {}},
        {"name": "end_shift", "arguments": {}},
    ]
    sess_short = Session(t1_timetable)
    with FakeModelServer(scripted(calls)) as srv:
        rec_short = play_episode(
            t1_timetable,
            ModelClient(srv.base_url, "fake-net", None),
            "windowed",
            session=sess_short,
            instruction=INSTRUCTION,
        )
    assert rec_short.status == "ok"
    assert rec_short.tool_calls == 4
    assert sess_short.closed
    final_short = sess_short.close()
    assert final_short.tick == 180
    assert isinstance(final_short.cost.total, float)
    assert isinstance(final_short.cost.model_dump(), dict)
    assert isinstance(final_short.counts, dict)
    assert isinstance(final_short.kpis, dict)

    # 2. Full T1 episode played to completion via `play_episode` (`session_cls=Session`) in windowed mode.
    with FakeModelServer(_NetTimetableOrderBehaviour(t1_timetable)) as srv:
        rec_full = play_episode(
            t1_timetable,
            ModelClient(srv.base_url, "fake-net-greedy", None),
            "windowed",
            session_cls=Session,
            instruction=INSTRUCTION,
        )
        seen_tools = {t["function"]["name"] for t in srv.requests[0]["tools"]}
    assert seen_tools == EXPECTED_TOOLS
    assert rec_full.status == "ok"
    assert rec_full.unfinished == 0
    assert rec_full.decision_points >= 15
    assert rec_full.invalid_calls == 0 and rec_full.refusals == 0
    assert rec_full.tokens_in > 0 and rec_full.tokens_out > 0
