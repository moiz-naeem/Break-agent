"""T06-01..08: the reference dispatcher's decide / evaluate / rollout units."""

import json
import os
import subprocess
import sys

import pytest

from railway.policies.runner import run_policy
from railway.ref.candidates import WAIT, candidates
from railway.ref.reference import Reference, ToolCall
from railway.ref.rollout import terminal_estimate
from railway.sim.engine import Simulator
from railway.sim.model import Authorize
from tests.t06.common import at_decision_point, dev, fixture

PURITY_SCRIPT = """
import json
from tests.t06.common import at_decision_point, dev
from railway.ref.reference import Reference
static, state = at_decision_point(dev("T2", 1)[0], 5)
print(json.dumps([[c.tool, c.args] for c in Reference().decide(static, state)]))
"""


@pytest.mark.slow
def test_T06_01_decide_is_pure():
    static, state = at_decision_point(dev("T2", 1)[0], 5)
    first, second = Reference().decide(static, state), Reference().decide(static, state)
    assert first == second
    outs = set()
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        proc = subprocess.run(
            [sys.executable, "-c", PURITY_SCRIPT], capture_output=True, text=True, env=env, check=True
        )
        outs.add(proc.stdout.strip())
    assert outs == {json.dumps([[c.tool, c.args] for c in first])}


def test_T06_02_decide_ends_with_advance():
    sim = Simulator.load(fixture("mini3"))
    for static, state in [(sim.static_info(), sim.state())]:
        calls = Reference().decide(static, state)
        assert calls[-1] == ToolCall("advance_to_next_event", {})
        assert calls[:-1] and all(c.tool in {"relieve_crew", "authorize"} for c in calls[:-1])


@pytest.mark.slow
def test_T06_02b_decide_ends_with_advance_t2():
    static, state = at_decision_point(dev("T2", 1)[0], 5)
    calls = Reference().decide(static, state)
    assert calls[-1] == ToolCall("advance_to_next_event", {})
    assert all(c.tool in {"relieve_crew", "authorize"} for c in calls[:-1])


@pytest.mark.slow
def test_T06_03_never_refused():
    result = run_policy(Reference(), dev("T2", 1)[0])
    assert result.final.counts["refusals"] == 0 and result.final.counts["invalid_calls"] == 0


def test_T06_04_mini3_beats_greedy():
    result = run_policy(Reference(), fixture("mini3"))
    assert result.final.cost.total < 26.0  # greedy, T05-26
    assert result.unfinished == 0


@pytest.mark.parametrize(("crew_used", "relieves"), [(225, True), (200, False)])
def test_T06_05_crew_rule(crew_used, relieves):
    data = fixture("mini3").model_dump(mode="json")
    data["factors"] = sorted({*data["factors"], "F10"})
    data["trains"][0]["crew_used_at_entry"] = crew_used  # E1
    sim = Simulator.load(type(fixture("mini3")).model_validate(data))
    assert [s.id for s in sim._scenario.network.stations if s.crew_station] == ["S2"]
    assert sim.apply(Authorize(train="E1", station="S2", track="L1")).ok
    while sim.tick < 6:
        sim.advance()
    e1 = next(t for t in sim.state().trains if t.id == "E1")
    assert (sim.tick, e1.station, e1.track) == (6, "S2", "L1")
    assert e1.crew_remaining == 240 - crew_used - 6
    calls = Reference().decide(sim.static_info(), sim.state())
    assert (calls[0] == ToolCall("relieve_crew", {"train": "E1"})) is relieves
    assert any(c.tool == "relieve_crew" for c in calls) is relieves


def test_T06_06_deadlock_safety():
    scenario = fixture("mini3_noloops")
    sim = Simulator.load(scenario)
    assert sim.apply(Authorize(train="L2", station="S3", track="main")).ok
    assert candidates(sim, "L1") == [WAIT]
    result = run_policy(Reference(), scenario)
    assert result.final.counts["gridlocks"] == 0 and result.unfinished == 0


def test_T06_07_terminal_estimate():
    sim = Simulator.load(fixture("mini3"))
    assert terminal_estimate(sim) == 0.0
    assert sim.advance(max_tick=20).tick == 2
    assert sim.advance(max_tick=20).tick == 20
    assert terminal_estimate(sim) == 423.0  # E1 324 + L5 90 + F21 9 (ticket T06-07)


def test_T06_08_evaluate_deterministic_and_ordered():
    sim = Simulator.load(fixture("mini3"))
    static, state, ref = sim.static_info(), sim.state(), Reference()
    go = ToolCall("authorize", {"train": "E1", "station": "S2", "track": "L1"})
    hold = ToolCall("hold", {"train": "E1", "until_tick": 30})
    assert ref.evaluate(static, state, go) == ref.evaluate(static, state, go)
    assert ref.evaluate(static, state, go) < ref.evaluate(static, state, hold)
    # Refused, read-only and unknown calls are no-ops.
    noop = ref.evaluate(static, state, ToolCall("advance_to_next_event", {}))
    assert ref.evaluate(static, state, ToolCall("authorize", {"train": "E1", "station": "S2", "track": "L9"})) == (
        ref.evaluate(static, state, ToolCall("status", {}))
    )
    assert isinstance(noop, float)
