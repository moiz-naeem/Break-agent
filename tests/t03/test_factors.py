import pytest
from pydantic import ValidationError

from railway.sim.engine import Simulator
from railway.sim.model import Authorize, RelieveCrew, Scenario
from tests.helpers import load_fixture


def scenario(*factors, events=None):
    data = load_fixture("mini3")
    data["factors"] = sorted(set(data["factors"]) | set(factors))
    data["events"] = events or []
    return Scenario.model_validate(data)


def test_T03_01_freight_restart_and_f5_off():
    with_restart = Simulator.load(scenario("F5"))
    with_restart.advance()
    assert with_restart.apply(Authorize(train="F21", station="S2", track="L2")).ok
    assert with_restart.advance().tick == 14
    f = next(t for t in with_restart.state().trains if t.id == "F21")
    assert f.restart_pending
    assert with_restart.apply(Authorize(train="F21", station="YE", track="main")).message.endswith("arrive t=30)")
    with_restart.end_shift()
    assert with_restart.cost().freight_lateness == 2.0

    without = Simulator.load(scenario())
    without.advance()
    without.apply(Authorize(train="F21", station="S2", track="L2"))
    without.advance()
    assert without.apply(Authorize(train="F21", station="YE", track="main")).message.endswith("arrive t=26)")


def test_T03_02_extension_avoids_restart_penalty():
    sim = Simulator.load(scenario("F5"))
    sim.advance()
    sim.apply(Authorize(train="F21", station="S2", track="L2"))
    sim.advance(max_tick=8)
    result = sim.apply(Authorize(train="F21", station="YE", track="main"))
    assert result.message.endswith("arrive t=26)")
    assert not next(t for t in sim.state().trains if t.id == "F21").restart_pending


def test_T03_03_express_unscheduled_stop_adds_one_tick():
    sim = Simulator.load(scenario("F5"))
    sim.apply(Authorize(train="E1", station="S1", track="L1"))
    assert sim.advance().tick == 2  # F21 entry is an earlier decision point.
    assert sim.advance().tick == 3
    assert sim.apply(Authorize(train="E1", station="S2", track="L1")).message.endswith("arrive t=7)")


def test_T03_04_crew_relief_refusal_and_action():
    sim = Simulator.load(scenario("F10"))
    assert sim.apply(RelieveCrew(train="E1")).rule.value == "R-NOT-CREW-STATION"


def test_T03_05_rotation_waits_for_predecessor_and_turnaround():
    data = load_fixture("mini3")
    l6 = dict(data["trains"][2])
    l6.update(
        id="L6",
        direction="east",
        entry_tick=20,
        stops=[
            {"station": "S1", "arr": 24, "dep": 25, "min_dwell": 1},
            {"station": "S2", "arr": 29, "dep": 30, "min_dwell": 1},
            {"station": "S3", "arr": 34, "dep": 35, "min_dwell": 1},
            {"station": "YE", "arr": 39, "dep": None, "min_dwell": 0},
        ],
    )
    l5 = next(t for t in data["trains"] if t["id"] == "L5")
    l5["next_service"] = "L6"
    data["trains"].append(l6)
    data["trains"].sort(key=lambda train: train["id"])
    data["factors"] = sorted(set(data["factors"]) | {"F6"})
    sim = Simulator.load(Scenario.model_validate(data))
    l6_state = next(t for t in sim.state().trains if t.id == "L6")
    assert l6_state.waits_for == "L5"
    sim.apply(Authorize(train="L5", station="YW", track="main"))
    ready = None
    while sim.tick < 29:
        ready = sim.advance()
    l6_state = next(t for t in sim.state().trains if t.id == "L6")
    assert l6_state.phase == "at_station"
    assert l6_state.entry_tick == 29
    assert ready.tick == 29
    assert any(r.kind == "READY" and r.train == "L6" for r in ready.reasons)


def test_T03_18_initial_crew_warning_fires_at_entry():
    data = load_fixture("mini3")
    e1 = next(t for t in data["trains"] if t["id"] == "E1")
    e1["crew_used_at_entry"] = 215
    data["factors"] = sorted(set(data["factors"]) | {"F10"})
    sim = Simulator.load(Scenario.model_validate(data))
    result = sim.advance()
    warning = next(r for r in result.reasons if r.kind == "CREW_WARNING" and r.train == "E1")
    assert warning.detail == "crew_remaining=25"


def test_T03_20_crew_relief_resets_counter_and_delays_departure():
    data = load_fixture("mini3")
    e1 = next(t for t in data["trains"] if t["id"] == "E1")
    e1["crew_used_at_entry"] = 200
    data["factors"] = sorted(set(data["factors"]) | {"F10"})
    sim = Simulator.load(Scenario.model_validate(data))
    sim.apply(Authorize(train="E1", station="S2", track="L1"))
    assert sim.advance().tick == 2
    assert sim.advance().tick == 6
    result = sim.apply(RelieveCrew(train="E1"))
    assert result.ok
    train = next(t for t in sim.state().trains if t.id == "E1")
    assert train.crew_used == 0
    assert train.relief_until == 14
    assert sim.apply(Authorize(train="E1", station="YE", track="main")).message.endswith("arrive t=20)")


@pytest.mark.parametrize(
    "bad_event",
    [
        {"id": "EV1", "kind": "possession", "reveal_tick": 3, "section": "S2-S3", "start": 2, "end": 4},
        {"id": "EV1", "kind": "possession", "reveal_tick": 0, "section": "S2-S3", "start": 4, "end": 4},
        {
            "id": "EV1",
            "kind": "speed_restriction",
            "reveal_tick": 0,
            "sections": ["S1-S2"],
            "speed_kmh": 0,
            "start": 2,
            "end": 4,
        },
        {"id": "EV1", "kind": "late_entry", "reveal_tick": 0, "train": "F21", "new_entry_tick": 2},
        {"id": "EV1", "kind": "breakdown", "reveal_tick": 1, "train": "E1", "tick": 2, "duration": 4},
    ],
)
def test_T03_31_invalid_event_is_rejected(bad_event):
    data = load_fixture("mini3")
    data["factors"].append(
        {"possession": "F7", "speed_restriction": "F8", "late_entry": "F16", "breakdown": "F15"}[bad_event["kind"]]
    )
    data["factors"].sort()
    data["events"] = [bad_event]
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)
