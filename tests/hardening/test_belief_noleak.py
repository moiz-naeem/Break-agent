"""Belief simulator and information-leak properties (T03-12, T03-32, T03-33, T03-36), randomised.

- Belief: at every decision point `Simulator.from_observation(static, state)` must reproduce `state()`
  and then evolve exactly like the real simulator under the same actions until the next reveal.
- No leak: a scenario with an event and the same scenario without it must look identical (every
  `state()` and every action result) until the event's reveal tick.
- Possession window: no train ever enters a possessed section inside its (effective) window.
"""

from __future__ import annotations

import copy

import pytest
from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from railway.sim.engine import Simulator
from railway.sim.model import Scenario
from tests.hardening.scenarios import mini3
from tests.hardening.simgen import (
    ALL_KINDS,
    EXTRA_FACTORS,
    SAFE_KINDS,
    RandChooser,
    dump,
    gen_scenario,
    gen_step,
    public,
    record_departures,
    run_episode,
    run_step,
)

HC = [HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much]
NO_SHRINK = [Phase.explicit, Phase.reuse, Phase.generate]
SEEDS = st.integers(0, 2**31 - 1)
NO_CREW = tuple(f for f in EXTRA_FACTORS if f != "F10")


def belief_episode(seed, *, kinds, factors):
    c = RandChooser(seed)
    bias = c.bool(0.5)
    fs = tuple(f for f in factors if c.bool(0.7))
    scenario = gen_scenario(c, factors=fs, gridlock_bias=bias)
    return run_episode(c, scenario, n_steps=40, checks={"belief"}, kinds=kinds, gridlock_bias=bias)


@settings(max_examples=60, deadline=None, suppress_health_check=HC, phases=NO_SHRINK)
@given(SEEDS)
def test_T03_32_33_belief_matches_real_until_next_reveal(seed):
    """Round trip at every decision point, then identical results, states and cost until the next
    reveal. Factors F5-F8, F15, F16; SAFE actions (see the xfail below for F10 and reversals)."""
    trace = belief_episode(seed, kinds=SAFE_KINDS, factors=NO_CREW)
    assert trace.belief_checks >= 0


@pytest.mark.xfail(strict=True, reason="BUG-10/BUG-11: crew of not-entered trains; reversing trains crash")
@settings(max_examples=60, deadline=None, suppress_health_check=HC, phases=NO_SHRINK)
@given(SEEDS)
def test_T03_32_33_belief_matches_real_all_factors_full_actions(seed):
    belief_episode(seed, kinds=ALL_KINDS, factors=EXTRA_FACTORS)


def test_belief_comparisons_actually_happen():
    """Guard against a vacuous belief test: the generator must reach many compared steps."""
    total = sum(belief_episode(seed, kinds=SAFE_KINDS, factors=NO_CREW).belief_checks for seed in range(40))
    assert total >= 200, total


# --- T03-36: no leak through state() or action results -------------------------------------------


@settings(max_examples=80, deadline=None, suppress_health_check=HC, phases=NO_SHRINK)
@given(SEEDS)
def test_T03_36_unrevealed_event_is_invisible_random(seed):
    c = RandChooser(seed)
    with_event = gen_scenario(c, factors=("F5", "F6", "F7", "F8", "F10", "F15", "F16"), gridlock_bias=c.bool(0.3))
    hidden = [e for e in with_event["events"] if e["reveal_tick"] > 0]
    if not hidden:
        return
    event = c.choice(hidden)
    without = copy.deepcopy(with_event)
    without["events"] = [e for e in without["events"] if e["id"] != event["id"]]
    x = Simulator.load(Scenario.model_validate(with_event))
    y = Simulator.load(Scenario.model_validate(without))
    assert public(x) == public(y)
    kinds = tuple(k for k in ALL_KINDS if k != "cancel")  # BUG-1b crash path; not about leaks
    for _ in range(40):
        step = gen_step(c, x, kinds)
        rx, ry = run_step(x, step), run_step(y, step)
        if x.tick >= event["reveal_tick"] or y.tick >= event["reveal_tick"]:
            break
        assert dump(rx) == dump(ry), f"{event['kind']} {event['id']} leaks through {step}"
        assert public(x) == public(y), f"{event['kind']} {event['id']} leaks into state() after {step}"
        if x.done:
            break


# --- T03-12: possessions are never entered during their window -----------------------------------


def _check_possession_entries(log, sim):
    windows = {(p["section"], p["start"], p["end"]) for p in sim._possessions}
    for d in log:
        for section, start, end in windows:
            if d["section"] == section and start <= d["tick"] < end:
                raise AssertionError(f"{d['train']} entered {section} at t={d['tick']} inside [{start},{end})")


@settings(max_examples=60, deadline=None, suppress_health_check=HC, phases=NO_SHRINK)
@given(SEEDS)
def test_T03_12_no_entry_during_possession_window(seed):
    c = RandChooser(seed)
    if c.bool(0.5):  # the T03-11 scenario
        event = {"id": "EV1", "kind": "possession", "reveal_tick": 5, "section": "S2-S3", "start": 12, "end": 30}
        scenario = mini3("F7", events=[event])
    else:
        scenario = gen_scenario(c, factors=("F7", "F15", "F8", "F5"))
    sim = Simulator.load(Scenario.model_validate(scenario))
    kinds = tuple(k for k in ALL_KINDS if k not in {"cancel", "reverse"})  # BUG-1/BUG-6 move trains anyway
    log: list = []
    with record_departures(log):
        for _ in range(40):
            run_step(sim, gen_step(c, sim, kinds))
            _check_possession_entries([d for d in log if d["sim"] is sim], sim)
            if sim.done:
                break
