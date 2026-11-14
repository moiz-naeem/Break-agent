import pytest

from railway.tools.session import TOOLS
from railway.util import canonical_json
from tests.t05.common import EPISODE_A, kpis, play, session

PRD_TOOLS = {
    "manual", "network", "timetable", "status", "bulletins", "kpis", "stringline", "notes", "authorize",
    "cancel_authority", "hold", "relieve_crew", "reverse", "advance_to_next_event", "end_shift",
}  # fmt: skip


def test_T05_01_all_tools_registered():
    assert set(TOOLS) == PRD_TOOLS


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("teleport", {}),
        ("authorize", {"train": "E1", "station": "S2"}),
        ("authorize", {"train": "E1", "station": "S2", "track": "L1", "speed": 9}),
        ("hold", {"train": "E1", "until_tick": "x"}),
    ],
)
def test_T05_02_bad_args(tool, args):
    s = session()
    before = canonical_json(s.sim.state().model_dump(mode="json"))
    out = s.call(tool, args)
    assert out.startswith("REFUSED R-BAD-ARGS")
    assert (s.log[-1]["status"], s.log[-1]["rule"]) == ("INVALID", "R-BAD-ARGS")
    assert s.used == 1
    assert canonical_json(s.sim.state().model_dump(mode="json")) == before


def test_T05_03_status_classification():
    s = session()
    play(s, EPISODE_A[:-1])
    assert [e["status"] for e in s.log] == [status for *_, status in EPISODE_A[:-1]]
    counts = kpis(s)["counts"]
    assert (counts["refusals"], counts["invalid_calls"]) == (1, 0)
    s.call("end_shift", {})
    assert s.final.cost.total == 769.0


def test_T05_04_budget_exhaustion():
    s = session(budget=5)
    outs = [s.call("kpis", {}) for _ in range(4)] + [s.call("advance_to_next_event", {})]
    assert "BUDGET EXHAUSTED (R-BUDGET)" in outs[-1] and "Final cost 1485.0" in outs[-1]
    assert s.closed
    sixth = s.call("status", {})
    assert sixth.startswith("REFUSED R-SHIFT-ENDED") and s.used == 5
    assert s.log[-1]["status"] == "REFUSED"
    assert s.final.cost.total == 1485.0


def test_T05_05_close_equals_end_shift():
    a, b = session(), session()
    play(a, EPISODE_A[:3])
    play(b, EPISODE_A[:3])
    a.call("end_shift", {})
    final = b.close()
    assert (a.final.tick, a.final.cost) == (final.tick, final.cost)
    assert b.close() is final  # idempotent


def test_T05_06_notes():
    s = session()
    assert s.call("notes", {"text": "x" * 1500}).startswith("OK")
    assert s.call("notes", {"text": "x" * 1501}).startswith("REFUSED R-NOTES-TOO-LONG")
    s.call("notes", {"text": "a"})
    s.call("notes", {"text": "b"})
    assert '"notes":"b"' in s.call("status", {})
    assert s.call("status", {"compact": True}).splitlines()[0].endswith('notes: "b"')


def test_T05_07_after_close():
    s = session()
    final = s.close()
    for tool, args in [("status", {}), ("authorize", {"train": "E1", "station": "S2", "track": "L1"}), ("kpis", {})]:
        out = s.call(tool, args)
        assert out.startswith("REFUSED R-SHIFT-ENDED") and f"Final cost {final.cost.total}" in out
