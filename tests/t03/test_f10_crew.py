from railway.sim.model import Authorize, RelieveCrew
from tests.t03.common import L6, kinds, load, tr, vf


def crew(used, *factors, **patches):
    return load(vf("F10", *factors, E1={"crew_used_at_entry": used}, **patches))


def test_T03_18_timeout_mid_section():
    s = crew(235)
    assert s.apply(Authorize(train="E1", station="S2", track="L1")).ok
    a = s.advance()  # remaining 5 <= 30 at entry: the warning fires at the entry tick (ticket 2.1, F10)
    assert (a.tick, kinds(a)) == (0, [("CREW_WARNING", "E1")])
    assert s.advance().tick == 2
    a = s.advance()
    assert a.tick == 5 and ("CREW_TIMEOUT", "E1") in kinds(a)
    assert s.cost().crew_timeout == 40.0  # v0.3: TIMEOUT_COST
    e1 = tr(s, "E1")
    assert (e1.phase, e1.section, e1.section_exit, e1.blocked_until) == ("in_section", "S1-S2", 51, 50)
    assert s._holders["S1-S2"] == "E1"
    s.advance(max_tick=50)
    assert s._holders["S1-S2"] == "E1"
    while s.tick < 51:
        s.advance(max_tick=51)
    assert s._holders.get("S1-S2") != "E1" and tr(s, "E1").station == "S2"


def test_T03_19_warning():
    s = crew(200)
    s.apply(Authorize(train="E1", station="S2", track="L1"))
    expected = [(2, ("READY", "F21")), (6, ("AUTHORITY_END", "E1")), (7, ("READY", "E1")), (10, ("CREW_WARNING", "E1"))]
    for tick, reason in expected:
        a = s.advance()
        assert a.tick == tick and reason in kinds(a), (tick, kinds(a))
    assert tr(s, "E1").crew_remaining == 30


def test_T03_19b_warning_fires_at_entry_when_already_low():
    s = crew(215)
    warning = next(r for r in s.advance().reasons if r.kind == "CREW_WARNING" and r.train == "E1")
    assert warning.detail == "crew_remaining=25"


def test_T03_20_relief():
    s = crew(200)
    s.apply(Authorize(train="E1", station="S2", track="L1"))
    reasons = []
    assert s.advance().tick == 2
    a = s.advance()
    reasons += kinds(a)
    assert a.tick == 6
    assert s.apply(RelieveCrew(train="E1")).ok
    e1 = tr(s, "E1")
    assert (e1.crew_used, e1.relief_until) == (0, 14)
    assert s.apply(Authorize(train="E1", station="YE", track="main")).message.endswith("(arrive t=20)")
    while not s.done:
        reasons += kinds(s.advance())
    assert ("CREW_WARNING", "E1") not in reasons
    e1 = tr(s, "E1")
    assert [a.actual for a in e1.arrivals if a.station == "YE"] == [20]
    assert s.cost().pax_lateness == 7.5  # v0.3 YE: 1.5 x 100/100 x f(7) = 1.5 x 5; S2 was on time


def test_T03_21_relief_refusals():
    s = crew(0)
    assert s.apply(RelieveCrew(train="E1")).rule.value == "R-NOT-CREW-STATION"  # at the origin yard
    s.apply(Authorize(train="E1", station="S1", track="L1"))
    assert s.advance().tick == 2
    assert tr(s, "E1").phase == "in_section"
    assert s.apply(RelieveCrew(train="E1")).rule.value == "R-NOT-STATIONARY"
    assert s.advance().tick == 3
    assert s.apply(RelieveCrew(train="E1")).rule.value == "R-NOT-CREW-STATION"  # S1 has no crew depot


def test_T03_22_crew_carries_over_rotation():
    s = load(vf("F6", "F10", extra=(L6,), L5={"next_service": "L6", "crew_used_at_entry": 100}))
    s.apply(Authorize(train="L5", station="YW", track="main"))
    while s.tick < 29:
        s.advance()
    assert s.tick == 29
    assert tr(s, "L6").crew_used == 129


def test_T03_23_f10_off_no_crew():
    s = load(vf(E1={"crew_used_at_entry": 235}))
    s.apply(Authorize(train="E1", station="S2", track="L1"))
    reasons = []
    while not s.done:
        assert all(t.crew_used is None and t.crew_remaining is None for t in s.state().trains)
        reasons += kinds(s.advance())
    assert s.tick == 60
    assert not [k for k, _ in reasons if k.startswith("CREW")]
