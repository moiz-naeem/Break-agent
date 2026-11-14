import pytest

from railway.sim.model import Authorize
from tests.t03.common import kinds, load, tr, vf


@pytest.mark.parametrize(("factors", "arrive", "late", "pending"), [(["F5"], 30, 2.0, True), ([], 26, 0.0, False)])
def test_T03_01_freight_restart(factors, arrive, late, pending):
    s = load(vf(*factors))
    assert s.advance().tick == 2
    assert s.apply(Authorize(train="F21", station="S2", track="L2")).ok
    a = s.advance()
    assert a.tick == 14 and ("AUTHORITY_END", "F21") in kinds(a)
    assert tr(s, "F21").restart_pending is pending
    assert s.apply(Authorize(train="F21", station="YE", track="main")).message.endswith(f"(arrive t={arrive})")
    s.end_shift()
    assert s.cost().freight_lateness == late


def test_T03_02_extension_avoids_restart():
    s = load(vf("F5"))
    s.advance()
    s.apply(Authorize(train="F21", station="S2", track="L2"))
    s.advance(max_tick=8)
    assert s.apply(Authorize(train="F21", station="YE", track="main")).ok
    while not s.done:
        assert not tr(s, "F21").restart_pending
        s.advance(max_tick=s.tick + 1)
    assert [a.actual for a in tr(s, "F21").arrivals if a.station == "YE"] == [26]
    assert s.cost().freight_lateness == 0.0


def test_T03_03_express_restart_one_tick():
    s = load(vf("F5"))
    assert s.apply(Authorize(train="E1", station="S1", track="L1")).message.endswith("(arrive t=3)")
    assert s.advance().tick == 2
    assert s.advance().tick == 3
    assert tr(s, "E1").restart_pending
    assert s.apply(Authorize(train="E1", station="S2", track="L1")).message.endswith("(arrive t=7)")


def test_T03_04_no_penalty_at_timed_stop_or_origin():
    """Episode A of T02 (test_T02_26) with F5 on: every stop is a timed stop or the origin."""
    s = load(vf("F5"))
    assert s.apply(Authorize(train="E1", station="S2", track="L1")).ok
    assert s.advance().tick == 2
    assert s.apply(Authorize(train="L5", station="S2", track="main")).message.endswith("(arrive t=11)")
    assert s.advance().tick == 6
    assert not s.apply(Authorize(train="E1", station="YE", track="main")).ok
    assert s.advance().tick == 7
    assert s.advance().tick == 11
    assert s.apply(Authorize(train="E1", station="YE", track="main")).ok
    s.end_shift()
    assert s.cost().total == 718.0
