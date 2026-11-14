"""T03 ticket tests missing from tests/t03 (docs/gates/T03.md), written from the T03 ticket §5.

Present before this file (by content): T03-01, 02, 03, 05, 09, 10, 11, 14, 20, 24, 29, part of 31, 32,
33, 34, 36. The randomised T03-12, 32, 33, 36 and 37 live in test_belief_noleak.py and
test_properties.py.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from pydantic import ValidationError

from railway.sim.model import Authorize, RelieveCrew, Scenario
from railway.util import canonical_json
from tests.hardening.scenarios import L6, load, mini3, run_to, train, ts
from tests.helpers import REPO


def reasons(result):
    return [(r.kind, r.train) for r in result.reasons]


def episode_a(sim):
    assert sim.apply(Authorize(train="E1", station="S2", track="L1")).ok
    assert sim.advance().tick == 2
    assert sim.apply(Authorize(train="L5", station="S2", track="main")).ok
    assert sim.advance().tick == 6
    assert not sim.apply(Authorize(train="E1", station="YE", track="main")).ok
    assert sim.advance().tick == 7
    assert sim.advance().tick == 11
    assert sim.apply(Authorize(train="E1", station="YE", track="main")).ok
    sim.end_shift()


def test_T03_04_no_penalty_at_timed_stop_or_origin():
    sim = load(mini3("F5"))
    episode_a(sim)
    assert sim.cost().total == 519.0


def rotation(*factors, **kw):
    data = mini3(*factors, extra_trains=[L6], **kw)
    train(data, "L5")["next_service"] = "L6"
    return data


def test_T03_06_f6_off():
    sim = load(rotation())
    seen = []
    while not sim.done and sim.tick < 30:
        result = sim.advance()
        seen.append((result.tick, reasons(result)))
    assert (20, [("READY", "L6")]) in seen, seen


def test_T03_07_predecessor_unfinished():
    sim = load(rotation("F6"))
    sim.end_shift()
    assert ts(sim, "L6").phase == "not_entered"
    # E1 740 + F21 217 + L5 278 + L6 200 + 1.0×1×(60−39−2)×2 = 238
    assert sim.cost().unfinished == 740.0 + 217.0 + 278.0 + 238.0


@pytest.mark.parametrize("target", ["F21", "Z9"])
def test_T03_08_validation(target):
    data = mini3("F6")
    train(data, "E1")["next_service"] = target  # F21 has the same direction as E1; Z9 does not exist
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_T03_13_unrevealed_not_checked():
    event = {"id": "EV1", "kind": "possession", "reveal_tick": 15, "section": "S2-S3", "start": 20, "end": 40}
    data = mini3("F7", events=[event])
    train(data, "F21").update(entry_tick=3, stops=[{"station": "YE", "arr": 27, "dep": None, "min_dwell": 0}])
    sim = load(data)
    run_to(sim, 3)
    assert sim.apply(Authorize(train="F21", station="YE", track="main")).ok  # no leak through refusals
    result = sim.advance(max_tick=15)
    assert result.tick == 15 and ("BULLETIN", None) in reasons(result)
    assert "AUTHORITY_CUT F21 -> S2/main: possession S2-S3 20-40" in [c.text for c in result.changes]
    f21 = ts(sim, "F21")  # it reaches S2 in the same tick and stops on main
    assert (f21.station, f21.track, f21.authority) == ("S2", "main", None)


def restriction(sections, start, end, reveal=0, speed=40):
    return {
        "id": "EV1",
        "kind": "speed_restriction",
        "reveal_tick": reveal,
        "sections": sections,
        "speed_kmh": speed,
        "start": start,
        "end": end,
    }


def test_T03_15_boundary_not_applied():
    sim = load(mini3("F8", events=[restriction(["S1-S2"], 4, 60)]))
    assert sim.apply(Authorize(train="E1", station="S2", track="L1")).message.endswith("(arrive t=6)")


def test_T03_16_revealed_after_grant():
    sim = load(mini3("F8", events=[restriction(["S2-S3"], 5, 60, reveal=1)]))
    assert sim.apply(Authorize(train="E1", station="YE", track="main")).message.endswith("(arrive t=13)")
    sim.end_shift()
    assert ts(sim, "E1").finished_tick == 19


def test_T03_17_state_lists_restriction():
    sim = load(mini3("F8", events=[restriction(["S2-S3"], 30, 60, reveal=10)]))
    assert sim.state().restrictions == []
    run_to(sim, 9)
    assert sim.state().restrictions == []
    run_to(sim, 10)
    assert [r["id"] for r in sim.state().restrictions] == ["EV1"]


def crew(**used):
    data = mini3("F10")
    for tid, value in used.items():
        train(data, tid)["crew_used_at_entry"] = value
    return data


@pytest.mark.xfail(
    strict=True,
    reason="CONTRACT-3: PRD §7.7 says tick-0 triggers are consumed by load(); the engine reports E1's "
    "entry CREW_WARNING from the first advance() at t=0 (tests/t03 asserts the engine's behaviour)",
)
def test_T03_18_first_advance_skips_tick0_triggers():
    sim = load(crew(E1=235))
    assert sim.apply(Authorize(train="E1", station="S2", track="L1")).ok
    assert sim.advance().tick == 2


def test_T03_18_timeout_mid_section():
    sim = load(crew(E1=235))
    assert sim.apply(Authorize(train="E1", station="S2", track="L1")).ok
    result = sim.advance()
    while result.tick < 5:
        result = sim.advance()
    assert result.tick == 5 and ("CREW_TIMEOUT", "E1") in reasons(result)
    assert sim.cost().crew_timeout == 300.0
    e1 = ts(sim, "E1")
    assert (e1.phase, e1.section, e1.section_exit, e1.blocked_until) == ("in_section", "S1-S2", 51, 50)
    run_to(sim, 50)
    assert sim._holders.get("S1-S2") == "E1"


def test_T03_19_warning():
    sim = load(crew(E1=200))
    sim.apply(Authorize(train="E1", station="S2", track="L1"))
    got = [(r.tick, reasons(r)) for r in (sim.advance() for _ in range(4))]
    assert got == [
        (2, [("READY", "F21")]),
        (6, [("AUTHORITY_END", "E1")]),
        (7, [("READY", "E1")]),
        (10, [("CREW_WARNING", "E1")]),
    ]
    assert ts(sim, "E1").crew_remaining == 30


def test_T03_21_relief_refusals():
    sim = load(crew())
    assert sim.apply(RelieveCrew(train="E1")).rule.value == "R-NOT-CREW-STATION"  # at the yard
    sim.apply(Authorize(train="E1", station="S1", track="L1"))
    sim.advance()  # t=2, E1 in YW-S1
    assert sim.apply(RelieveCrew(train="E1")).rule.value == "R-NOT-STATIONARY"
    run_to(sim, 3)
    assert ts(sim, "E1").station == "S1"
    assert sim.apply(RelieveCrew(train="E1")).rule.value == "R-NOT-CREW-STATION"


def test_T03_22_crew_carries_over_rotation():
    data = rotation("F6", "F10")
    train(data, "L5")["crew_used_at_entry"] = 100
    sim = load(data)
    sim.apply(Authorize(train="L5", station="YW", track="main"))
    run_to(sim, 29)
    l6 = ts(sim, "L6")
    assert l6.phase == "at_station" and l6.crew_used == 129


def test_T03_23_f10_off_no_crew():
    sim = load(mini3())
    sim.apply(Authorize(train="E1", station="YE", track="main"))
    kinds = []
    while not sim.done:
        kinds += [r.kind for r in sim.advance().reasons]
        assert all(t.crew_used is None and t.crew_remaining is None for t in sim.state().trains)
    assert not {"CREW_WARNING", "CREW_TIMEOUT"} & set(kinds)


def breakdown(tid, tick, duration):
    return {"id": "EV1", "kind": "breakdown", "reveal_tick": tick, "train": tid, "tick": tick, "duration": duration}


def test_T03_25_not_in_network_skipped():
    sim = load(mini3("F15", events=[breakdown("F21", 1, 10)]))
    while not sim.done:
        result = sim.advance(max_tick=sim.tick + 1)
        assert "BREAKDOWN" not in [r.kind for r in result.reasons]
        assert all(t.breakdown_until is None for t in sim.state().trains)
        assert "EV1" not in canonical_json(sim.state().model_dump(mode="json"))


def test_T03_26_at_station():
    sim = load(mini3("F15", events=[breakdown("E1", 7, 5)]))
    sim.apply(Authorize(train="E1", station="S2", track="L1"))
    run_to(sim, 6)
    assert (ts(sim, "E1").station, ts(sim, "E1").track) == ("S2", "L1")
    assert sim.apply(Authorize(train="E1", station="YE", track="main")).ok
    run_to(sim, 13)
    assert ts(sim, "E1").section_enter == 12


def test_T03_27_with_restart():
    sim = load(mini3("F5", "F15", events=[breakdown("E1", 4, 10)]))
    sim.apply(Authorize(train="E1", station="S2", track="L1"))
    assert sim.advance().tick == 2
    result = sim.advance()
    assert result.tick == 4 and ("BREAKDOWN", "E1") in reasons(result)
    run_to(sim, 14)  # the restart penalty is added when the breakdown ends
    assert ts(sim, "E1").section_exit == 17


def test_T03_28_possession_deferred():
    possession = {"id": "EV2", "kind": "possession", "reveal_tick": 0, "section": "S1-S2", "start": 10, "end": 20}
    sim = load(mini3("F7", "F15", events=[possession, {**breakdown("E1", 4, 10), "id": "EV1"}]))
    assert sim.apply(Authorize(train="E1", station="S2", track="L1")).ok
    run_to(sim, 16)
    assert ts(sim, "E1").station == "S2"
    for _ in range(3):
        assert sim.state().possessions[0]["start"] == 16
        assert sim.state().possessions[0]["end"] == 20
        sim.advance(max_tick=sim.tick + 1)


def test_T03_30_late_entry_with_rotation():
    late = {"id": "EV1", "kind": "late_entry", "reveal_tick": 0, "train": "L6", "new_entry_tick": 25}
    data = rotation("F6", "F16", events=[late])
    sim = load(data)
    sim.apply(Authorize(train="L5", station="YW", track="main"))
    run_to(sim, 28)
    assert ts(sim, "L6").phase == "not_entered"
    run_to(sim, 29)
    assert ts(sim, "L6").phase == "at_station"


@pytest.mark.parametrize(
    "bad",
    [
        {"id": "EV1", "kind": "possession", "reveal_tick": 0, "section": "S7-S8", "start": 2, "end": 4},
        {
            "id": "EV1",
            "kind": "speed_restriction",
            "reveal_tick": 0,
            "sections": ["S1-S2", "S1-S2"],
            "speed_kmh": 40,
            "start": 2,
            "end": 4,
        },
        {"id": "EV1", "kind": "late_entry", "reveal_tick": 2, "train": "F21", "new_entry_tick": 9},
        {"id": "EV1", "kind": "possession", "reveal_tick": 0, "section": "S1-S2", "start": 2, "end": 61},
    ],
    ids=["unknown_section", "duplicate_sections", "late_entry_reveal_not_before_entry", "possession_past_shift"],
)
def test_T03_31_event_validation_remaining(bad):
    factor = {"possession": "F7", "speed_restriction": "F8", "late_entry": "F16"}[bad["kind"]]
    with pytest.raises(ValidationError):
        Scenario.model_validate(mini3(factor, events=[bad]))


def test_T03_31_event_kind_with_factor_off_and_unknown_factor():
    event = {"id": "EV1", "kind": "possession", "reveal_tick": 0, "section": "S1-S2", "start": 2, "end": 4}
    with pytest.raises(ValidationError):
        Scenario.model_validate(mini3(events=[event]))  # F7 off
    with pytest.raises(ValidationError):
        Scenario.model_validate(mini3("F9"))


def test_T03_35_static_info_has_no_events():
    event = {"id": "EV1", "kind": "possession", "reveal_tick": 20, "section": "S1-S2", "start": 30, "end": 40}
    sim = load(mini3("F7", events=[event]))
    text = canonical_json(sim.static_info().model_dump(mode="json"))
    for key in ('"events"', '"seed"', '"scenario_id"', '"reveal_tick"'):
        assert key not in text


ALL_FACTOR_SCRIPT = r"""
import sys
sys.path.insert(0, sys.argv[1])
from tests.hardening.simgen import RandChooser, gen_scenario, run_episode, SAFE_KINDS
from railway.util import canonical_json
out = []
for seed in range(8):
    c = RandChooser(seed)
    sc = gen_scenario(c, factors=("F5", "F6", "F7", "F8", "F10", "F15", "F16"))
    trace = run_episode(c, sc, n_steps=30, checks=set(), kinds=SAFE_KINDS + ("reverse", "hold"))
    out.append([trace.results, trace.states[-1], trace.final.cost().model_dump()])
print(canonical_json(out))
"""


def test_T03_38_cross_process_determinism_all_factors():
    outs = []
    for hashseed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        proc = subprocess.run(
            [sys.executable, "-c", ALL_FACTOR_SCRIPT, str(REPO)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO,
        )
        outs.append(proc.stdout)
    assert outs[0] == outs[1]
    assert len(outs[0]) > 1000
