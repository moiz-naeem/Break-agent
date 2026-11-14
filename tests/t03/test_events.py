import pytest

from railway.sim.engine import Simulator
from railway.sim.model import Authorize, Scenario
from tests.helpers import load_fixture


def make(event, factor, **patches):
    data = load_fixture("mini3")
    for path, value in patches.items():
        *parents, key = path.split(".")
        node = data
        for part in parents:
            node = node[int(part)] if isinstance(node, list) else node[part]
        if isinstance(node, list):
            node[int(key)] = value
        else:
            node[key] = value
    data["factors"].append(factor)
    data["factors"].sort()
    data["events"] = [event]
    return Simulator.load(Scenario.model_validate(data))


def test_T03_09_possession_is_visible_after_reveal_and_checks_overlap():
    event = {"id": "EV1", "kind": "possession", "reveal_tick": 0, "section": "S2-S3", "start": 20, "end": 40}
    sim = make(event, "F7")
    assert len(sim.state().possessions) == 1
    sim.advance()
    result = sim.apply(Authorize(train="F21", station="YE", track="main"))
    assert result.ok  # The scheduled occupancy exits the section at tick 20.


def test_T03_10_possession_overlap_refuses_without_state_change():
    event = {"id": "EV1", "kind": "possession", "reveal_tick": 0, "section": "S2-S3", "start": 20, "end": 40}
    sim = make(event, "F7", **{"trains.1.entry_tick": 3, "trains.1.stops.0.arr": 27})
    assert sim.advance().tick == 3
    before = sim.state().model_dump(mode="json")
    result = sim.apply(Authorize(train="F21", station="YE", track="main"))
    assert result.rule.value == "R-POSSESSION"
    assert sim.state().model_dump(mode="json") == before


def test_T03_14_speed_restriction_applies_at_section_entry():
    event = {
        "id": "EV1",
        "kind": "speed_restriction",
        "reveal_tick": 0,
        "sections": ["S1-S2"],
        "speed_kmh": 40,
        "start": 3,
        "end": 60,
    }
    sim = make(event, "F8")
    result = sim.apply(Authorize(train="E1", station="S2", track="L1"))
    assert result.message.endswith("arrive t=12)")
    assert next(t for t in sim.state().trains if t.id == "E1").delay == 6
    sim.advance()
    sim.advance()
    train = next(t for t in sim.state().trains if t.id == "E1")
    assert train.station == "S2"
    assert train.arrivals[-1].actual == 12


def test_T03_24_breakdown_extends_in_flight_section_exit():
    event = {"id": "EV1", "kind": "breakdown", "reveal_tick": 4, "train": "E1", "tick": 4, "duration": 10}
    sim = make(event, "F15")
    sim.apply(Authorize(train="E1", station="S2", track="L1"))
    assert sim.advance().tick == 2
    result = sim.advance()
    assert result.tick == 4 and result.reasons[0].kind == "BREAKDOWN"
    assert next(t for t in sim.state().trains if t.id == "E1").section_exit == 16


def test_T03_29_late_entry_reveals_and_updates_effective_entry():
    event = {"id": "EV1", "kind": "late_entry", "reveal_tick": 10, "train": "F22", "new_entry_tick": 45}
    # Add F22 through a copied freight service with a compatible unique id.
    data = load_fixture("mini3")
    f22 = dict(data["trains"][1])
    f22.update(id="F22", entry_tick=30, stops=[{"station": "YE", "arr": 54, "dep": None, "min_dwell": 0}])
    data["trains"].append(f22)
    data["trains"].sort(key=lambda t: t["id"])
    data["factors"].append("F16")
    data["factors"].sort()
    data["events"] = [event]
    sim = Simulator.load(Scenario.model_validate(data))
    assert next(t for t in sim.state().trains if t.id == "F22").entry_tick == 30
    assert sim.advance().tick == 2
    result = sim.advance()
    assert result.tick == 10 and result.reasons[0].kind == "BULLETIN"
    assert next(t for t in sim.state().trains if t.id == "F22").entry_tick == 45


def test_T03_11_revealed_possession_cuts_existing_authority():
    event = {"id": "EV1", "kind": "possession", "reveal_tick": 5, "section": "S2-S3", "start": 12, "end": 30}
    sim = make(event, "F7")
    sim.advance()
    assert sim.apply(Authorize(train="F21", station="YE", track="main")).ok
    result = sim.advance()
    assert result.tick == 5
    assert any(change.text == "AUTHORITY_CUT F21 -> S2/main: possession S2-S3 12-30" for change in result.changes)
    f21 = next(t for t in sim.state().trains if t.id == "F21")
    assert f21.authority.dest_station == "S2"
    assert f21.authority.dest_track == "main"


@pytest.mark.parametrize(
    ("event", "factor"),
    [
        ({"id": "EV1", "kind": "possession", "reveal_tick": 5, "section": "S2-S3", "start": 10, "end": 20}, "F7"),
        (
            {
                "id": "EV1",
                "kind": "speed_restriction",
                "reveal_tick": 5,
                "sections": ["S1-S2"],
                "speed_kmh": 40,
                "start": 5,
                "end": 20,
            },
            "F8",
        ),
        ({"id": "EV1", "kind": "late_entry", "reveal_tick": 5, "train": "F22", "new_entry_tick": 45}, "F16"),
        ({"id": "EV1", "kind": "breakdown", "reveal_tick": 5, "train": "E1", "tick": 5, "duration": 5}, "F15"),
    ],
)
def test_T03_36_unrevealed_events_do_not_leak(event, factor):
    with_event = load_fixture("mini3")
    without_event = load_fixture("mini3")
    if factor == "F16":
        f22 = dict(with_event["trains"][1])
        f22.update(id="F22", entry_tick=30, stops=[{"station": "YE", "arr": 54, "dep": None, "min_dwell": 0}])
        with_event["trains"].append(f22)
        without_event["trains"].append(dict(f22))
        with_event["trains"].sort(key=lambda train: train["id"])
        without_event["trains"].sort(key=lambda train: train["id"])
    with_event["factors"] = sorted(set(with_event["factors"]) | {factor})
    without_event["factors"] = list(with_event["factors"])
    with_event["events"] = [event]
    left = Simulator.load(Scenario.model_validate(with_event))
    right = Simulator.load(Scenario.model_validate(without_event))
    action = Authorize(train="E1", station="S2", track="L1")
    assert left.apply(action).model_dump(mode="json") == right.apply(action).model_dump(mode="json")
    assert left.state().model_dump(mode="json") == right.state().model_dump(mode="json")
    while left.tick < event["reveal_tick"]:
        l_result, r_result = left.advance(), right.advance()
        if left.tick < event["reveal_tick"]:
            assert l_result.model_dump(mode="json") == r_result.model_dump(mode="json")
        if left.tick < event["reveal_tick"]:
            assert left.state().model_dump(mode="json") == right.state().model_dump(mode="json")
