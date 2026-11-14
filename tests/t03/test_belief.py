from railway.sim.engine import Simulator
from railway.sim.model import Authorize, Scenario
from tests.helpers import load_fixture


def test_T03_32_belief_roundtrip_uses_only_public_observation():
    data = load_fixture("mini3")
    sim = Simulator.load(Scenario.model_validate(data))
    clone = Simulator.from_observation(sim.static_info(), sim.state())
    assert clone.state().model_dump(mode="json") == sim.state().model_dump(mode="json")
    assert "events" not in clone.static_info().model_dump(mode="json")


def test_T03_33_belief_continues_exactly_at_a_decision_point():
    data = load_fixture("mini3")
    data["factors"] = sorted(set(data["factors"]) | {"F5", "F7", "F8", "F10"})
    data["events"] = [
        {"id": "EV1", "kind": "possession", "reveal_tick": 0, "section": "S2-S3", "start": 40, "end": 50},
        {
            "id": "EV2",
            "kind": "speed_restriction",
            "reveal_tick": 0,
            "sections": ["S3-YE"],
            "speed_kmh": 40,
            "start": 10,
            "end": 30,
        },
    ]
    real = Simulator.load(Scenario.model_validate(data))
    real.advance()
    real.apply(Authorize(train="F21", station="S2", track="L2"))
    real.advance()
    belief = Simulator.from_observation(real.static_info(), real.state())
    assert belief.state().model_dump(mode="json") == real.state().model_dump(mode="json")
    assert belief.apply(Authorize(train="F21", station="YE", track="main")).model_dump(mode="json") == real.apply(
        Authorize(train="F21", station="YE", track="main")
    ).model_dump(mode="json")
    assert belief.advance().model_dump(mode="json") == real.advance().model_dump(mode="json")
    assert belief.cost().model_dump(mode="json") == real.cost().model_dump(mode="json")


def test_T03_34_unrevealed_breakdown_is_absent_from_belief():
    data = load_fixture("mini3")
    data["factors"] = sorted(set(data["factors"]) | {"F15"})
    data["events"] = [{"id": "EV1", "kind": "breakdown", "reveal_tick": 20, "train": "E1", "tick": 20, "duration": 10}]
    real = Simulator.load(Scenario.model_validate(data))
    real.apply(Authorize(train="E1", station="S2", track="L1"))
    real.advance()
    belief = Simulator.from_observation(real.static_info(), real.state())
    assert "EV1" not in str(belief.state().model_dump(mode="json"))
    assert "events" not in str(belief.static_info().model_dump(mode="json")).lower()
