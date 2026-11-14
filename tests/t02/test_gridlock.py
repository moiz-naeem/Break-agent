from railway.sim.cost import freight_cost, pax_cost
from railway.sim.engine import Simulator
from railway.sim.model import Authorize, Scenario
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


def test_T02_38_no_false_positive_when_a_loop_remains_free():
    s = load("mini3")
    # No movement authorities means each train can still choose an unoccupied route.
    assert s.state().gridlock == []


def test_T02_27_episode_B_gridlock_is_permanent():
    """v0.3: there is no reversal. A gridlock lasts to shift end and its trains are charged as unfinished."""
    s = load("mini2_gridlock")
    s.apply(Authorize(train="L1", station="S1", track="main"))
    s.apply(Authorize(train="L2", station="S2", track="main"))
    assert s.advance().tick == 4
    assert s.state().gridlock == ["L1", "L2"]
    assert not s.apply(Authorize(train="L1", station="S2", track="main")).ok  # S2 main is L2's
    assert not s.apply(Authorize(train="L2", station="S1", track="main")).ok
    later = []
    while not s.done:
        later += [r.kind for r in s.advance().reasons]
    assert "GRIDLOCK" not in later  # an edge trigger: reported once, at tick 4
    assert s.state().gridlock == ["L1", "L2"]
    assert [t.station for t in s.state().trains] == ["S1", "S2"]
    # each local: 200 + 1.0 x 1 x (60 - 14 - 2)
    assert s.cost().unfinished == 488.0 and s.cost().total == 488.0
    assert pax_cost("express", 200, 7) == 30
    assert pax_cost("local", 100, 2) == 0
    assert freight_cost(-3) == 0


def test_T02_28_gridlock_with_free_track_elsewhere_is_still_permanent():
    """mini3_noloops: the trains outside the jam are unaffected, the jammed pair never moves again."""
    s = load("mini3_noloops")
    s.apply(Authorize(train="L1", station="S2", track="main"))
    s.apply(Authorize(train="L2", station="S3", track="main"))
    while s.state().gridlock != ["L1", "L2"]:
        assert not s.advance().done
    stuck = {t.id: t.station for t in s.state().trains if t.id in {"L1", "L2"}}
    s.end_shift()
    assert {t.id: t.station for t in s.state().trains if t.id in {"L1", "L2"}} == stuck
    assert s.state().gridlock == ["L1", "L2"]


def test_T02_16_opposing_trains_can_pass_when_a_loop_is_free():
    s = load("mini3")
    s.apply(Authorize(train="E1", station="S2", track="L1"))
    s.apply(Authorize(train="L5", station="S2", track="main"))
    # E1 occupies S2/L1 while L5 approaches; the main at S2 remains a feasible destination.
    assert s.advance().tick == 2
    assert s.advance().tick == 6
    assert s.state().gridlock == []


def test_T02_40_scc_excludes_train_waiting_on_the_cycle():
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
    assert s.state().gridlock == ["L1", "L2"]
