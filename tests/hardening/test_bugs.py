"""Minimal reproductions of simulator defects found by the hardening suite.

Each test asserts the behaviour PRD §7 requires and is a strict xfail until `railway/sim/` is fixed
(the hardening task may not edit the simulator). Run with `--runxfail` to see the current failures.
"""

from __future__ import annotations

import copy

import pytest

from railway.sim.engine import Simulator
from railway.sim.model import Authorize, CancelAuthority, Hold, Reverse
from tests.hardening.scenarios import F22, L6, load, mini3, run_to, train, ts
from tests.hardening.simgen import check_resources, occupancy
from tests.helpers import load_fixture


def xbug(reason: str):
    return pytest.mark.xfail(strict=True, reason=reason)


def two_trains_never_share(sim, until: int = 60):
    """Advance step by step and assert no resource is ever occupied by two trains."""
    while not sim.done and sim.tick < until:
        sim.advance(max_tick=sim.tick + 1)
        doubled = {res: tids for res, tids in occupancy(sim).items() if len(tids) > 1}
        assert not doubled, f"t={sim.tick}: {doubled}"


# --- cancel_authority (PRD §7.5 rule 12) --------------------------------------------------------


@xbug("BUG-1: cancel_authority on a train inside its last section releases the section it occupies")
def test_bug_cancel_in_section_releases_the_occupied_section():
    sim = load(load_fixture("mini3"))
    assert sim.apply(Authorize(train="E1", station="S2", track="L1")).ok
    sim.advance()  # t=2 READY F21
    sim.advance(max_tick=4)
    assert ts(sim, "E1").section == "S1-S2"
    assert sim.apply(CancelAuthority(train="E1")).ok
    # E1 is still inside S1-S2, so a westbound authority through S1-S2 must be refused.
    result = sim.apply(Authorize(train="L5", station="S1", track="main"))
    assert not result.ok and result.rule.value == "R-SECTION-RESERVED", result.message


@xbug("BUG-1b: after BUG-1 the authority has an empty path and a later extension raises IndexError")
def test_bug_extension_after_in_section_cancel_does_not_crash():
    sim = load(load_fixture("mini3"))
    sim.apply(Authorize(train="E1", station="S2", track="L1"))
    sim.advance()
    sim.advance(max_tick=4)
    sim.apply(CancelAuthority(train="E1"))
    sim.apply(Authorize(train="E1", station="YE", track="main"))  # IndexError in engine._authorize
    check_resources(sim)


@xbug("BUG-2: cancel_authority while dwelling at a timed stop inside the authority frees the main it stands on")
def test_bug_cancel_while_dwelling_frees_the_occupied_main():
    sim = load(load_fixture("mini3"))
    sim.apply(Authorize(train="E1", station="YE", track="main"))
    run_to(sim, 6)
    e1 = ts(sim, "E1")
    assert (e1.phase, e1.station, e1.track) == ("at_station", "S2", "main")
    assert sim.apply(CancelAuthority(train="E1")).ok
    result = sim.apply(Authorize(train="L5", station="S2", track="main"))
    assert not result.ok, f"{result.message}: E1 is standing on S2/main"


@xbug("BUG-2 consequence: two trains stand on S2/main after the cancel in BUG-2")
def test_bug_cancel_while_dwelling_leads_to_two_trains_on_one_track():
    sim = load(load_fixture("mini3"))
    sim.apply(Authorize(train="E1", station="YE", track="main"))
    run_to(sim, 6)
    sim.apply(CancelAuthority(train="E1"))
    sim.apply(Authorize(train="L5", station="S2", track="main"))
    two_trains_never_share(sim)


# --- authority extension (PRD §7.5 rule 4) ------------------------------------------------------


@xbug("BUG-3: an extension takes the old end station's main without checking it is free (rule 4)")
def test_bug_extension_takes_an_occupied_main():
    sim = load(load_fixture("mini3"))
    assert sim.apply(Authorize(train="E1", station="S2", track="main")).ok  # E1 stops on S2/main at 6
    assert sim.apply(Authorize(train="L5", station="S2", track="L1")).ok  # L5 to the S2 loop
    run_to(sim, 6)
    assert (ts(sim, "E1").station, ts(sim, "E1").track) == ("S2", "main")
    assert ts(sim, "L5").section == "S2-S3"
    result = sim.apply(Authorize(train="L5", station="S1", track="main"))  # extends L5 through S2/main
    assert not result.ok and result.rule.value == "R-MAIN-BLOCKED", result.message


@xbug("BUG-3 consequence: L5 stops for its timed stop on S2/main while E1 stands there")
def test_bug_extension_leads_to_two_trains_on_one_main():
    sim = load(load_fixture("mini3"))
    sim.apply(Authorize(train="E1", station="S2", track="main"))
    sim.apply(Authorize(train="L5", station="S2", track="L1"))
    run_to(sim, 6)
    sim.apply(Authorize(train="L5", station="S1", track="main"))
    two_trains_never_share(sim)


# --- reverse (PRD §7.5 rule 11) -----------------------------------------------------------------


def episode_c():
    sim = load(load_fixture("mini3_noloops"))
    sim.apply(Authorize(train="L1", station="S2", track="main"))
    sim.apply(Authorize(train="L2", station="S3", track="main"))
    run_to(sim, 9)
    assert sim.state().gridlock == ["L1", "L2"]
    return sim


@xbug("BUG-4: a refused reverse (R-NO-REVERSE-PATH) still drops the train's track reservation")
def test_bug_refused_reverse_drops_the_reservation():
    sim = episode_c()
    refused = sim.apply(Reverse(train="L1"))
    assert refused.rule.value == "R-NO-REVERSE-PATH"
    # L1 still stands on S2/main, so L2 must not get an authority onto it.
    result = sim.apply(Authorize(train="L2", station="S2", track="main"))
    assert not result.ok, result.message


def noloop_with_s1_loop():
    data = load_fixture("mini3_noloops")
    data["network"]["stations"][0]["loops"] = [{"id": "L1", "length_m": 300, "kind": "platform"}]
    l3 = copy.deepcopy(train(data, "L1"))
    l3["id"], l3["entry_tick"] = "L3", 2
    for stop in l3["stops"]:
        stop["arr"] += 2
        stop["dep"] = None if stop["dep"] is None else stop["dep"] + 2
    data["trains"].append(l3)
    data["trains"].sort(key=lambda t: t["id"])
    return data


@xbug("BUG-5: reverse does not reserve its target loop; another train is granted it during the reversal")
def test_bug_reverse_target_loop_is_not_reserved():
    sim = load(noloop_with_s1_loop())
    sim.apply(Authorize(train="L1", station="S2", track="main"))
    sim.apply(Authorize(train="L2", station="S3", track="main"))
    run_to(sim, 9)
    assert sim.apply(Reverse(train="L1")).message == "OK L1 reversing to S1/L1"
    result = sim.apply(Authorize(train="L3", station="S1", track="L1"))
    assert not result.ok, f"{result.message}: S1/L1 is the destination of L1's reversal"


@xbug("BUG-5 consequence: L1 (reversing) and L3 both end up on S1/L1")
def test_bug_reverse_target_loop_double_occupied():
    sim = load(noloop_with_s1_loop())
    sim.apply(Authorize(train="L1", station="S2", track="main"))
    sim.apply(Authorize(train="L2", station="S3", track="main"))
    run_to(sim, 9)
    sim.apply(Reverse(train="L1"))
    sim.apply(Authorize(train="L3", station="S1", track="L1"))
    two_trains_never_share(sim)


def reversing_l1():
    sim = load(load_fixture("mini2_gridlock"))
    sim.apply(Authorize(train="L1", station="S1", track="main"))
    sim.apply(Authorize(train="L2", station="S2", track="main"))
    sim.advance()
    assert sim.apply(Reverse(train="L1")).ok  # L1 reverses into YW, arriving t=19
    assert sim.apply(Authorize(train="L2", station="S1", track="main")).ok
    run_to(sim, 10)  # L2 now stands on S1/main
    return sim


@xbug("BUG-6: a reversing train accepts a new authority planned from the station it reversed away from")
def test_bug_reversing_train_accepts_authority():
    sim = reversing_l1()
    assert ts(sim, "L1").phase == "reversing"
    result = sim.apply(Authorize(train="L1", station="S2", track="main"))
    assert not result.ok, result.message


@xbug("BUG-6 consequence: L1 leaves YW straight into S1-S2, skipping YW-S1 and S1 (where L2 stands)")
def test_bug_reversing_train_teleports():
    sim = reversing_l1()
    sim.apply(Authorize(train="L1", station="S2", track="main"))
    while not sim.done and ts(sim, "L1").section != "S1-S2":
        before = ts(sim, "L1")
        sim.advance(max_tick=sim.tick + 1)
        after = ts(sim, "L1")
        if after.section == "S1-S2":
            assert before.station == "S1", f"entered S1-S2 at t={sim.tick} from {before.station or before.section}"


@xbug("BUG-7: the reversal arrival reports READY twice for the same train in one AdvanceResult")
def test_bug_reverse_arrival_reports_ready_once():
    sim = load(load_fixture("mini2_gridlock"))
    sim.apply(Authorize(train="L1", station="S1", track="main"))
    sim.apply(Authorize(train="L2", station="S2", track="main"))
    sim.advance()
    sim.apply(Reverse(train="L1"))
    while sim.tick < 19:
        result = sim.advance()
    kinds = [(r.kind, r.train) for r in result.reasons]
    assert kinds.count(("READY", "L1")) == 1, kinds


# --- hold (PRD §7.5 rules 6 and 13) --------------------------------------------------------------


@xbug("CONTRACT-1: rule 6 says a train departs only when 'not held'; the engine ignores holds once authorized")
def test_contract_held_train_with_authority_does_not_depart():
    sim = load(load_fixture("mini3"))
    assert sim.apply(Authorize(train="E1", station="S2", track="L1")).ok
    assert sim.apply(Hold(train="E1", until_tick=5)).ok
    assert ts(sim, "E1").authority is not None  # rule 13: the authority is kept
    sim.advance()
    e1 = ts(sim, "E1")
    assert e1.phase == "at_station" or e1.section_enter >= 5, f"E1 left at t={e1.section_enter}"


# --- F6 rotations ---------------------------------------------------------------------------------


@xbug("BUG-8: an F6 successor with entry_tick 0 is placed in the network at load and departs before its predecessor")
def test_bug_rotation_successor_with_entry_zero_waits():
    data = mini3("F6")
    train(data, "E1")["next_service"] = "L5"  # E1 (east, ends at YE) forms L5 (west, from YE)
    sim = load(data)
    l5 = ts(sim, "L5")
    assert l5.phase == "not_entered" and l5.waits_for == "E1", (l5.phase, l5.waits_for)


@xbug("BUG-8 consequence: the waiting successor is granted an authority and runs at t=0")
def test_bug_rotation_successor_with_entry_zero_cannot_run():
    data = mini3("F6")
    train(data, "E1")["next_service"] = "L5"
    sim = load(data)
    result = sim.apply(Authorize(train="L5", station="S3", track="main"))
    assert not result.ok and result.rule.value == "R-NOT-IN-NETWORK", result.message


@xbug("CONTRACT-2: a successor still waiting for its predecessor (not in the network) can be authorized")
def test_contract_waiting_successor_is_not_in_network():
    data = mini3("F6", extra_trains=[L6])
    train(data, "L5")["next_service"] = "L6"
    sim = load(data)
    run_to(sim, 20)  # L6's timetable entry; L5 has not finished
    assert ts(sim, "L6").waits_for == "L5"
    result = sim.apply(Authorize(train="L6", station="S1", track="main"))
    assert not result.ok and result.rule.value == "R-NOT-IN-NETWORK", result.message


# --- F10 crew -------------------------------------------------------------------------------------


@xbug("BUG-9: with F16 a late-entered train's crew duty counts from its original entry, not its effective entry")
def test_bug_late_entry_crew_counts_from_effective_entry():
    late = {"id": "EV1", "kind": "late_entry", "reveal_tick": 10, "train": "F22", "new_entry_tick": 45}
    sim = load(mini3("F10", "F16", events=[late], extra_trains=[F22], shift_ticks=90))
    run_to(sim, 50)
    f22 = ts(sim, "F22")
    assert f22.entry_tick == 45
    assert f22.crew_used == 5, f"crew_used={f22.crew_used} (T03 §2.1: crew_used_at_entry + (t - entry))"


# --- belief simulator (T03 §2.4) ------------------------------------------------------------------


@xbug("BUG-10: from_observation restarts the crew counter of not-yet-entered trains at the build tick")
def test_bug_belief_crew_of_not_entered_train():
    data = mini3("F10")
    train(data, "F21")["crew_used_at_entry"] = 100
    real = load(data)
    belief = Simulator.from_observation(real.static_info(), real.state())
    assert real.advance().tick == belief.advance().tick == 2
    assert ts(belief, "F21").crew_used == ts(real, "F21").crew_used == 100


@xbug("BUG-11: from_observation during a reversal builds a simulator whose state() raises TypeError")
def test_bug_belief_during_reversal():
    sim = load(load_fixture("mini2_gridlock"))
    sim.apply(Authorize(train="L1", station="S1", track="main"))
    sim.apply(Authorize(train="L2", station="S2", track="main"))
    sim.advance()
    sim.apply(Reverse(train="L1"))
    sim.advance()  # t=5 READY L2, a decision point with L1 reversing
    belief = Simulator.from_observation(sim.static_info(), sim.state())
    assert belief.state().model_dump(mode="json") == sim.state().model_dump(mode="json")


# --- gridlock definition (PRD §7.5 rule 10) -------------------------------------------------------


def false_gridlock_scenario() -> dict:
    """F1 (east, S1/main) and F2 (west, S2/main) wait on each other's main; F1 could use S2/L1 once
    E1 (west, on S2/L1) moves on to the free S1/L1. Nothing is deadlocked."""
    data = load_fixture("mini2_gridlock")
    data["network"]["stations"][0]["loops"] = [{"id": "L1", "length_m": 300, "kind": "platform"}]
    data["network"]["stations"][1]["loops"] = [{"id": "L1", "length_m": 700, "kind": "platform"}]
    data["trains"] = [
        {
            "id": "E1",
            "type": "express",
            "direction": "west",
            "pax": 100,
            "entry_tick": 0,
            "stops": [{"station": "YW", "arr": 9, "dep": None, "min_dwell": 0}],
            "next_service": None,
            "crew_used_at_entry": 0,
        },
        {
            "id": "F1",
            "type": "freight",
            "direction": "east",
            "pax": 0,
            "entry_tick": 0,
            "stops": [{"station": "YE", "arr": 18, "dep": None, "min_dwell": 0}],
            "next_service": None,
            "crew_used_at_entry": 0,
        },
        {
            "id": "F2",
            "type": "freight",
            "direction": "west",
            "pax": 0,
            "entry_tick": 0,
            "stops": [{"station": "YW", "arr": 18, "dep": None, "min_dwell": 0}],
            "next_service": None,
            "crew_used_at_entry": 0,
        },
    ]
    return data


@xbug("SPEC-1: the SCC rule flags F1/F2 as a gridlock although E1 can move and free S2/L1 for F1")
def test_spec_gridlock_only_when_no_move_can_resolve_it():
    sim = load(false_gridlock_scenario())
    assert sim.apply(Authorize(train="E1", station="S2", track="L1")).ok
    assert sim.apply(Authorize(train="F1", station="S1", track="main")).ok
    run_to(sim, 3)  # E1 on S2/L1
    assert sim.apply(Authorize(train="F2", station="S2", track="main")).ok
    while sim.tick < 12:
        sim.advance()
    assert ts(sim, "F2").station == "S2" and ts(sim, "F1").station == "S1"
    assert sim.state().gridlock == []
