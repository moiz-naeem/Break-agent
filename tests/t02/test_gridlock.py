from railway.sim.constants import GRIDLOCK_COST
from railway.sim.cost import freight_cost, pax_cost
from railway.sim.engine import Simulator
from railway.sim.model import Authorize, CancelAuthority, Scenario
from tests.helpers import load_fixture


def load(name):
    return Simulator.load(Scenario.model_validate(load_fixture(name)))


def test_T02_39_gridlock_appears_while_dwell_is_running():
    s = load("mini2_gridlock")
    assert s.apply(Authorize(train="L1", station="S1", track="main")).ok
    assert s.apply(Authorize(train="L2", station="S2", track="main")).ok
    result = s.advance()
    assert result.tick == 4
    assert [x.kind for x in result.reasons] == ["AUTHORITY_END", "AUTHORITY_END", "GRIDLOCK"]
    assert s.state().gridlock == ["L1", "L2"]
    assert s.cost().gridlock == GRIDLOCK_COST  # v0.3: charged when it forms


def test_T02_38_no_false_positive_when_a_loop_remains_free():
    s = load("mini3")
    # No movement authorities means each train can still choose an unoccupied route.
    assert s.state().gridlock == []


def test_T02_27_episode_B_gridlock_is_permanent_and_charged_once():
    """v0.3 has no reversal: once L1 and L2 face each other with no way past, nothing frees them."""
    s = load("mini2_gridlock")
    s.apply(Authorize(train="L1", station="S1", track="main"))
    s.apply(Authorize(train="L2", station="S2", track="main"))
    assert s.advance().tick == 4
    assert s.apply(Authorize(train="L1", station="S2", track="main")).rule.value == "R-MAIN-BLOCKED"
    assert s.apply(Authorize(train="L1", station="YW", track="main")).rule.value == "R-WRONG-DIRECTION"
    assert s.apply(CancelAuthority(train="L1")).rule.value == "R-NO-AUTHORITY"
    s.end_shift()
    assert s.state().gridlock == ["L1", "L2"]
    assert s.cost().gridlock == GRIDLOCK_COST
    assert {t.id: t.outcome for t in s.state().trains} == {"L1": None, "L2": None}
    assert pax_cost("express", 200, 7) == 15
    assert pax_cost("local", 100, 2) == 0
    assert freight_cost(-3) == 0


def test_T02_28_episode_C_gridlock_without_loops():
    s = load("mini3_noloops")
    s.apply(Authorize(train="L1", station="S2", track="main"))
    s.apply(Authorize(train="L2", station="S3", track="main"))
    assert s.advance().tick == 4
    assert s.advance().tick == 5
    assert s.advance().tick == 9
    assert s.state().gridlock == ["L1", "L2"]
    assert s.cost().gridlock == GRIDLOCK_COST


def test_T02_16_opposing_trains_can_pass_when_a_loop_is_free():
    s = load("mini3")
    s.apply(Authorize(train="E1", station="S2", track="L1"))
    s.apply(Authorize(train="L5", station="S2", track="main"))
    # E1 occupies S2/L1 while L5 approaches; the main at S2 remains a feasible destination.
    assert s.advance().tick == 2
    assert s.advance().tick == 6
    assert s.state().gridlock == []


def test_T02_40_train_queued_behind_a_gridlock_is_stuck_too_but_charged_once():
    """v0.3: every train that can never move again is in the gridlock, including one queued behind the
    deadlocked pair; it joins the same deadlock, so the charge is not repeated."""
    d = load_fixture("mini3_noloops")
    train = dict(d["trains"][0])
    train["id"] = "L3"
    d["trains"].append(train)
    d["trains"].sort(key=lambda item: item["id"])
    s = Simulator.load(Scenario.model_validate(d))
    s.apply(Authorize(train="L1", station="S2", track="main"))
    s.apply(Authorize(train="L2", station="S3", track="main"))
    assert s.advance().tick == 4
    assert s.advance().tick == 5
    assert s.apply(Authorize(train="L3", station="S1", track="main")).ok
    assert s.advance().tick == 9
    assert s.state().gridlock == ["L1", "L2", "L3"]
    s.end_shift()
    assert s.state().gridlock == ["L1", "L2", "L3"]
    assert s.cost().gridlock == GRIDLOCK_COST
