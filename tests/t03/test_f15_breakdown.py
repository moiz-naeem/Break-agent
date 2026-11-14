from railway.sim.model import Authorize
from tests.t03.common import dump, kinds, load, tr, vf


def breakdown(train, tick, duration):
    return {"id": "EV1", "kind": "breakdown", "reveal_tick": tick, "train": train, "tick": tick, "duration": duration}


def test_T03_24_mid_section():
    s = load(vf("F15", events=[breakdown("E1", 4, 10)]))
    s.apply(Authorize(train="E1", station="S2", track="L1"))
    assert s.advance().tick == 2
    a = s.advance()
    assert a.tick == 4
    reason = next(r for r in a.reasons if r.kind == "BREAKDOWN")
    assert (reason.train, reason.detail) == ("E1", "until t=14")
    assert tr(s, "E1").section_exit == 16
    s.end_shift()
    assert [x.actual for x in tr(s, "E1").arrivals if x.station == "S2"] == [16]
    assert s.cost().pax_lateness == 12.0  # v0.3: 1.5 x 100/100 x f(10) = 1.5 x 8


def test_T03_25_not_in_network_skipped():
    with_event = load(vf("F15", events=[breakdown("F21", 1, 5)]))
    without = load(vf("F15"))
    for s in (with_event, without):
        s.apply(Authorize(train="E1", station="S2", track="L1"))
    while not with_event.done:
        a, b = with_event.advance(max_tick=with_event.tick + 1), without.advance(max_tick=without.tick + 1)
        assert "BREAKDOWN" not in [k for k, _ in kinds(a)]
        assert dump(a) == dump(b)
        assert dump(with_event.state()) == dump(without.state())


def test_T03_26_at_station():
    s = load(vf("F15", events=[breakdown("E1", 7, 5)]))
    s.apply(Authorize(train="E1", station="S2", track="L1"))
    s.advance()
    assert s.advance().tick == 6
    assert (tr(s, "E1").station, tr(s, "E1").track) == ("S2", "L1")
    assert s.apply(Authorize(train="E1", station="YE", track="main")).ok
    while tr(s, "E1").section != "S2-S3":
        assert not s.done, "E1 never entered S2-S3"
        s.advance(max_tick=s.tick + 1)
    assert tr(s, "E1").section_enter == 12


def test_T03_27_with_restart():
    s = load(vf("F5", "F15", events=[breakdown("E1", 4, 10)]))
    s.apply(Authorize(train="E1", station="S2", track="L1"))
    s.advance()
    assert s.advance().tick == 4
    assert tr(s, "E1").restart_pending  # F5: the penalty lands on the next movement (t=14)
    s.advance(max_tick=14)
    assert tr(s, "E1").section_exit == 17
    s.end_shift()
    assert [x.actual for x in tr(s, "E1").arrivals if x.station == "S2"] == [17]


def test_T03_28_possession_deferred():
    possession = {"id": "EV2", "kind": "possession", "reveal_tick": 0, "section": "S1-S2", "start": 10, "end": 20}
    s = load(vf("F7", "F15", events=[possession, breakdown("E1", 4, 10)]))
    assert s.apply(Authorize(train="E1", station="S2", track="L1")).ok  # projection [3,6) is clear
    while s.tick < 16:
        s.advance(max_tick=s.tick + 1)
    assert tr(s, "E1").station == "S2"
    assert [x.actual for x in tr(s, "E1").arrivals if x.station == "S2"] == [16]
    while s.tick < 20:
        p = s.state().possessions[0]
        assert (p["start"], p["end"]) == (16, 20), s.tick
        s.advance(max_tick=s.tick + 1)
