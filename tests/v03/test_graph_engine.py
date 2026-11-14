"""v0.3 engine on a graph network: routing, misroutes, terminals, the new cost terms, gridlock."""

import pytest

from railway.sim.constants import GRIDLOCK_COST, MISSED_CALL_FLAT, TIMEOUT_COST, UNFINISHED_COST
from railway.sim.cost import freight_cost, lateness_curve, missed_call_cost, pax_cost
from railway.sim.engine import Simulator
from railway.sim.model import Authorize, CancelAuthority, Hold
from railway.sim.observe import StaticInfo

from .graphs import MAIN, express_ta_tb, line_scenario, scenario, stop, train


def run(sim, until=None):
    while not sim.done and (until is None or sim.tick < until):
        sim.advance()
    return sim


def auth(sim, tid, station, track="main", via=()):
    return sim.apply(Authorize(train=tid, station=station, track=track, via=list(via)))


# -- cost formulas -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("late", "expected"),
    [(-5, 0), (0, 0), (2, 0), (3, 1), (14, 12), (15, 18), (29, 32), (30, 43), (59, 72), (60, 93), (100, 133)],
)
def test_lateness_curve(late, expected):
    assert lateness_curve(late) == expected


def test_cost_formulas():
    assert pax_cost("express", 200, 15) == 1.5 * 2 * 18
    assert pax_cost("local", 100, 2) == 0
    assert missed_call_cost("express", 100, 50, 60) == MISSED_CALL_FLAT + 1.5 * 1.5 * (93 + 10)
    assert missed_call_cost("local", 0, 0, None) == MISSED_CALL_FLAT
    assert freight_cost(30) == 15 and freight_cost(31) == 15.5 + 50 and freight_cost(-4) == 0


# -- routing -------------------------------------------------------------------------------------


def test_on_time_run_costs_nothing():
    sim = Simulator.load(scenario([express_ta_tb()]))
    assert auth(sim, "E1", "J1").ok
    run(sim, 8)
    assert auth(sim, "E1", "M2", via=["Y1"]).ok
    run(sim, 20)
    assert auth(sim, "E1", "TB", "P1").ok
    run(sim)
    t = sim.state().trains[0]
    assert (t.outcome, t.finished_tick, [a.actual for a in t.arrivals]) == ("arrived", 29, [6, 17, 29])
    assert sim.cost().total == 0


def test_two_equal_routes_need_via():
    sim = Simulator.load(scenario([express_ta_tb()]))
    out = auth(sim, "E1", "M2")
    assert (out.ok, out.rule) == (False, "R-ROUTE-AMBIGUOUS")
    assert auth(sim, "E1", "M2", via=["X1"]).ok
    assert "X1-M2" in sim.state().trains[0].authority.path


def test_no_u_turn():
    """D1 sits on a dead-end branch: a train there that came from J2 can only go on to TD."""
    tr = train("F6", "freight", ["TB", "B1", "J2", "D1", "TD"], [stop("TD", 40)])
    sim = Simulator.load(scenario([tr]))
    assert auth(sim, "F6", "D1", "L2").ok
    run(sim, 10)
    out = auth(sim, "F6", "J2")
    assert (out.ok, out.rule) == (False, "R-NO-ROUTE")


def test_a_balloon_loop_turns_a_train_round():
    """Round the M1-Y1-M2-X1 loop a train can come back the way it came: legal, and either way round."""
    sim = Simulator.load(scenario([express_ta_tb()]))
    assert auth(sim, "E1", "J1").ok
    run(sim, 8)
    assert auth(sim, "E1", "A1").rule == "R-ROUTE-AMBIGUOUS"
    assert auth(sim, "E1", "A1", via=["Y1", "M2", "X1", "M1"]).ok


def test_misroute_warns_and_counts_once():
    sim = Simulator.load(scenario([express_ta_tb()]))
    assert auth(sim, "E1", "J1").ok
    run(sim, 8)
    out = auth(sim, "E1", "C1")  # the branch to TC: TB is out of reach
    assert out.ok and "R-OFF-ROUTE" in out.message and "can no longer reach TB" in out.message
    assert "misses scheduled stop M2" in out.message
    assert sim.counters["misroutes"] == 1
    run(sim, 20)
    assert auth(sim, "E1", "TC", "P1").ok
    assert sim.counters["misroutes"] == 1


def test_wrong_terminal_is_unfinished_with_missed_calls():
    sim = Simulator.load(scenario([express_ta_tb()]))
    assert auth(sim, "E1", "TC", "P1").ok  # J1 is passed on the way; M2 and TB are lost
    run(sim)
    t = sim.state().trains[0]
    assert t.outcome == "wrong_terminal" and sim.counters["wrong_terminals"] == 1
    cost = sim.cost()
    assert cost.unfinished == UNFINISHED_COST
    expected = missed_call_cost("express", 100, 30, None) + missed_call_cost("express", 200, 0, None)
    assert cost.missed_calls == pytest.approx(expected)
    assert sim.counters["missed_calls"] == 2


def test_skipping_a_stop_by_the_bypass_is_a_missed_call():
    tr = train("L1", "local", MAIN, [stop("Y1", 10, 11), stop("TB", 40, alight=100)], pax=100)
    sim = Simulator.load(scenario([tr]))
    out = auth(sim, "L1", "M2", via=["X1"])
    assert out.ok and "R-OFF-ROUTE" not in out.message  # from M2 the train could still turn back to Y1
    out = auth(sim, "L1", "J2")  # past M2 towards TB, Y1 is out of reach
    assert out.ok and "misses scheduled stop Y1" in out.message and "can no longer reach" not in out.message
    run(sim, 15)
    assert auth(sim, "L1", "TB", "P1").ok
    run(sim)
    assert sim.state().trains[0].outcome == "arrived"
    alight, board = sim._train_defs["L1"].flow(0)
    assert sim.cost().missed_calls == missed_call_cost("local", alight, board, None)
    assert sim.cost().unfinished == 0
    assert sim.counters["misroutes"] == 0  # a skipped stop is not a misroute


def test_never_run_train_pays_flat_and_every_stop():
    sim = Simulator.load(scenario([express_ta_tb()]))
    run(sim)
    cost = sim.cost()
    assert cost.unfinished == UNFINISHED_COST
    assert cost.missed_calls == pytest.approx(
        sum(missed_call_cost("express", a, b, None) for a, b in [(50, 20), (100, 30), (200, 0)])
    )


# -- terminals -----------------------------------------------------------------------------------


def test_full_terminal_refuses_and_frees_after_turnaround():
    blocker = train("L2", "local", ["TC", "C1", "J1", "A1", "TA"], [stop("TA", 12, alight=100)], pax=100)
    second = train("L3", "local", ["TB", "B1", "J2", "M2", "Y1", "M1", "J1", "C1", "TC"], [stop("TC", 40)], pax=50)
    sim = Simulator.load(scenario([blocker, second], terminal_tracks={"TC": [{"id": "P1", "length_m": 300, "kind": "platform"}]}))
    assert sim.state().terminal_holds == []
    assert sim._holders["TC/P1"] == "L2"  # L2 entered on TC's only track
    assert auth(sim, "L3", "J1", "L1", via=["Y1"]).ok
    out = auth(sim, "L3", "TC", "P1")  # extending the authority on to TC
    assert (out.ok, out.rule) == (False, "R-TERMINAL-FULL")
    assert auth(sim, "L2", "TA", "P1").ok
    run(sim, 1)
    assert "TC/P1" not in sim._holders  # L2 left
    assert auth(sim, "L3", "TC", "P1").ok


def test_entry_waits_for_a_free_terminal_track():
    a = train("L2", "local", ["TC", "C1", "J1", "A1", "TA"], [stop("TA", 30)], pax=10)
    b = train("L4", "local", ["TC", "C1", "J1", "A1", "TA"], [stop("TA", 40)], pax=10, entry=5)
    sim = Simulator.load(scenario([a, b]))
    r = sim.advance()
    assert (r.tick, [(x.kind, x.train) for x in r.reasons]) == (5, [("ENTRY_BLOCKED", "L4")])
    assert {t.id: t.phase for t in sim.state().trains}["L4"] == "not_entered"
    assert auth(sim, "L2", "C1", "L1").ok
    r = sim.advance()
    assert (r.tick, [(x.kind, x.train) for x in r.reasons]) == (6, [("READY", "L4")])
    assert sim._holders["TC/P1"] == "L4"


# -- gridlock, crew, freight, operations ----------------------------------------------------------------


def _line(extra=(), s2_loops=()):
    f1 = train("F1", "freight", ["T1", "S1", "S2", "T2"], [stop("T2", 60)])
    f2 = train("F2", "freight", ["T2", "S2", "S1", "T1"], [stop("T1", 60)])
    return Simulator.load(line_scenario([f1, f2, *extra], [[], list(s2_loops)]))


def test_head_on_gridlock_is_charged_once_and_is_permanent():
    sim = _line()
    assert auth(sim, "F1", "S1").ok and auth(sim, "F2", "S2").ok
    r = sim.advance()
    assert r.tick == 5 and ("GRIDLOCK", None) in [(x.kind, x.train) for x in r.reasons]
    assert sim.state().gridlock == ["F1", "F2"] and sim.cost().gridlock == GRIDLOCK_COST
    run(sim)
    assert sim.cost().gridlock == GRIDLOCK_COST and sim.state().gridlock == ["F1", "F2"]


def test_no_gridlock_while_a_blocker_can_still_leave():
    """F1 and F2 wait on each other, but one of F1's ways in (S2/L1) is held by L3, which can still run on
    to T2. Once it does, F1 takes the loop and F2 gets past: not a deadlock."""
    l3 = train("L3", "local", ["T1", "S1", "S2", "T2"], [stop("T2", 90)], pax=10)
    sim = _line([l3], [{"id": "L1", "length_m": 700, "kind": "platform"}])
    assert auth(sim, "L3", "S2", "L1").ok
    run(sim, 6)
    assert auth(sim, "F1", "S1").ok and auth(sim, "F2", "S2").ok
    run(sim, 11)
    assert sim.state().gridlock == [] and sim.cost().gridlock == 0
    assert auth(sim, "L3", "T2", "P2").ok
    run(sim, 12)
    assert auth(sim, "F1", "S2", "L1").ok
    run(sim, 17)
    assert auth(sim, "F2", "T1", "P1").ok and auth(sim, "F1", "T2", "P1").ok
    run(sim)
    assert {t.id: t.outcome for t in sim.state().trains} == {"F1": "arrived", "F2": "arrived", "L3": "arrived"}
    assert sim.cost().gridlock == 0


def test_crew_timeout_costs_300():
    tr = train("F3", "freight", ["TA", "A1", "J1", "C1", "TC"], [stop("TC", 30)], crew=235)
    sim = Simulator.load(scenario([tr], factors=("F1", "F2", "F3", "F4", "F10"), terminal_tracks={"TC": [{"id": "P1", "length_m": 700, "kind": "goods"}]}))
    assert auth(sim, "F3", "TC", "P1").ok
    run(sim)
    assert sim.cost().crew_timeout == TIMEOUT_COST


def test_freight_late_step():
    tr = train("F4", "freight", ["TA", "A1", "J1", "C1", "TC"], [stop("TC", 10)])
    sim = Simulator.load(scenario([tr], terminal_tracks={"TC": [{"id": "P1", "length_m": 700, "kind": "goods"}]}))
    assert sim.apply(Hold(train="F4", until_tick=30)).ok
    run(sim, 30)
    assert auth(sim, "F4", "TC", "P1").ok
    run(sim)
    late = sim.state().trains[0].arrivals[-1].actual - 10
    assert late > 30 and sim.cost().freight_lateness == freight_cost(late)


def test_platform_change_and_unplanned_stop():
    tr = train("E2", "express", MAIN, [stop("J1", 7, 8, alight=40, board=60, platform="L1"), stop("TB", 40, alight=100)])
    sim = Simulator.load(scenario([tr], factors=("F1", "F2", "F3", "F4", "F5")))
    assert auth(sim, "E2", "A1", "L1").ok  # unplanned stop at A1
    run(sim, 3)
    assert sim.cost().operations == 1
    assert auth(sim, "E2", "J1", "main").ok  # not the published L1
    run(sim)
    assert sim.cost().operations == 1 + 1.5 * 100 / 100 * 2


def test_missed_connection():
    feeder = train("L7", "local", ["TC", "C1", "J1", "A1", "TA"], [stop("J1", 20, 21), stop("TA", 40)], pax=80)
    conn = express_ta_tb("E3")
    connections = [{"feeder": "L7", "connector": "E3", "station": "J1", "pax": 30, "min_transfer": 3}]
    sim = Simulator.load(scenario([feeder, conn], factors=("F1", "F2", "F3", "F4", "F11"), connections=connections))
    assert auth(sim, "E3", "J1", "L1").ok
    run(sim, 8)
    assert auth(sim, "E3", "M2", via=["Y1"]).ok  # leaves J1 at t=8, before the feeder arrives
    run(sim)
    assert sim.counters["missed_connections"] == 1


# -- observation round trip and determinism ---------------------------------------------------------


def test_belief_from_observation_matches_the_real_state():
    tr2 = train("F5", "freight", ["TD", "D1", "J2", "M2", "X1", "M1", "J1", "C1", "TC"], [stop("TC", 90)])
    sim = Simulator.load(scenario([express_ta_tb(), tr2], terminal_tracks={"TC": [{"id": "P1", "length_m": 700, "kind": "goods"}]}))
    assert auth(sim, "E1", "J1").ok and auth(sim, "F5", "M2", "L2").ok
    run(sim, 9)
    static = StaticInfo.model_validate(sim.static_info().model_dump())
    belief = Simulator.from_observation(static, sim.state())
    assert belief.state().model_dump() == sim.state().model_dump()
    assert belief._holders == sim._holders


def test_same_actions_same_result():
    def play():
        sim = Simulator.load(scenario([express_ta_tb()]))
        auth(sim, "E1", "M2", via=["X1"])
        sim.apply(CancelAuthority(train="E1"))
        auth(sim, "E1", "TB", "P1", via=["Y1"])
        return run(sim).state().model_dump()

    assert play() == play()
