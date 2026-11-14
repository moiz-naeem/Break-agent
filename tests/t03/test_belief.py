from hypothesis import example, given, settings

from railway.sim.engine import Simulator
from railway.sim.model import Authorize
from railway.util import canonical_json
from tests.t03.common import L6, dump, load, run_step, steps, vf

EVENTS = [
    {"id": "EV1", "kind": "possession", "reveal_tick": 0, "section": "S2-S3", "start": 15, "end": 45},
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


def scenario():
    """mini3, F1-F8 + F10, L6 rotation from L5, a possession and a restriction revealed at 0."""
    return vf("F5", "F6", "F7", "F8", "F10", extra=(L6,), L5={"next_service": "L6"}, events=EVENTS)


def to_decision_point(plan):
    sim = load(scenario())
    for acts, _ in plan:
        if sim.done:
            break
        run_step(sim, acts, None)  # jump=None: stop only at decision points
    return sim


def belief(sim):
    return Simulator.from_observation(sim.static_info(), sim.state())


@settings(max_examples=60)
@given(steps)
def test_T03_32_roundtrip(plan):
    sim = to_decision_point(plan)
    assert canonical_json(dump(belief(sim).state())) == canonical_json(dump(sim.state()))


@settings(max_examples=60)
@given(steps, steps)
# Random actions rarely route a train through the possession window; pin one that does (F21, S2-S3 at 14-20).
@example([([], None)], [([Authorize(train="F21", station="YE", track="main")], None), ([], None)])
def test_T03_33_exact_when_nothing_hidden(prefix, future):
    real = to_decision_point(prefix)
    twin = belief(real)
    for acts, jump in future:
        if real.done:
            break
        r_results, r_adv = run_step(real, acts, jump)
        b_results, b_adv = run_step(twin, acts, jump)
        assert [dump(x) for x in r_results] == [dump(x) for x in b_results]
        assert (r_adv.tick, dump(r_adv)["reasons"]) == (b_adv.tick, dump(b_adv)["reasons"])
    real.end_shift()
    twin.end_shift()
    assert dump(real.cost()) == dump(twin.cost())


def _visible(model) -> str:
    """Observation text without the always-present (null) `breakdown_until` field name."""
    data = dump(model)
    for t in data.get("trains", []):
        assert t.pop("breakdown_until", None) is None
    return canonical_json(data).lower()


def test_T03_34_no_hidden_future():
    event = {"id": "EV1", "kind": "breakdown", "reveal_tick": 20, "train": "E1", "tick": 20, "duration": 10}
    real = load(vf("F15", events=[event]))
    real.apply(Authorize(train="E1", station="S2", track="L1"))
    real.advance()
    assert real.advance().tick == 6  # E1 waits at S2/L1, still in the network at t=20
    twin = belief(real)
    for sim in (real, twin):
        while sim.tick < 20:
            assert "breakdown" not in _visible(sim.state())
            assert "breakdown" not in _visible(sim.static_info())
            sim.advance(max_tick=sim.tick + 1)
        while sim.tick < 21:  # the real run also stops at the t=20 breakdown
            sim.advance(max_tick=21)
        sim.apply(Authorize(train="E1", station="YE", track="main"))
        sim.end_shift()
    assert real.cost().total != twin.cost().total


def test_T03_35_static_info_has_no_events():
    breakdown = {"id": "EV9", "kind": "breakdown", "reveal_tick": 5, "train": "E1", "tick": 5, "duration": 2}
    text = canonical_json(dump(load(vf("F7", "F15", events=[EVENTS[0], breakdown])).static_info()))
    for key in ('"events"', '"seed"', '"scenario_id"', '"reveal_tick"'):
        assert key not in text
