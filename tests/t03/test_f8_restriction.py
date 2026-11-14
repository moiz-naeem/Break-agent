from railway.sim.model import Authorize
from tests.t03.common import load, tr, vf


def restriction(sections, speed, start, end, reveal=0):
    event = {
        "id": "EV1",
        "kind": "speed_restriction",
        "reveal_tick": reveal,
        "sections": sections,
        "speed_kmh": speed,
        "start": start,
        "end": end,
    }
    return load(vf("F8", events=[event]))


def test_T03_14_restriction_applies_on_entry():
    s = restriction(["S1-S2"], 40, 3, 60)
    assert s.apply(Authorize(train="E1", station="S2", track="L1")).message.endswith("(arrive t=12)")
    assert tr(s, "E1").delay == 6
    s.end_shift()
    assert [a.actual for a in tr(s, "E1").arrivals if a.station == "S2"] == [12]
    assert s.cost().pax_lateness == 24.0  # 3.0 x 2 x (6 - 2); no other train reaches a stop


def test_T03_15_boundary_not_applied():
    s = restriction(["S1-S2"], 40, 4, 60)
    assert s.apply(Authorize(train="E1", station="S2", track="L1")).message.endswith("(arrive t=6)")


def test_T03_16_revealed_after_grant():
    s = restriction(["S2-S3"], 40, 5, 60, reveal=1)
    assert s.apply(Authorize(train="E1", station="YE", track="main")).message.endswith("(arrive t=13)")
    s.end_shift()
    assert [a.actual for a in tr(s, "E1").arrivals if a.station == "YE"] == [19]


def test_T03_17_state_lists_restriction():
    s = restriction(["S1-S2"], 40, 10, 30, reveal=5)
    while s.tick < 8:
        expected = [] if s.tick < 5 else [{"id": "EV1", "sections": ["S1-S2"], "speed_kmh": 40, "start": 10, "end": 30}]
        assert s.state().restrictions == expected, s.tick
        s.advance(max_tick=s.tick + 1)
