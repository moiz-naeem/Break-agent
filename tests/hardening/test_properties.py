"""Randomised property tests for the simulator (T02-41..48, T03-37 and more), PRD §7.5-§7.7, §7.12.

Every generated case is a random small scenario (1-3 stations, 0-3 loops, opposing trains, any of
F5-F8, F10, F15, F16 with random events) plus a random, state-aware action sequence; the invariants
in `simgen.py` are checked after every step. Half of the cases are biased toward head-on meetings on
lines with few or no loops so GRIDLOCK and `reverse` are exercised often.

Hypothesis draws a seed and the generators run on `random.Random(seed)`: Hypothesis's own byte-level
draws explore weighted choices poorly, and a failing seed is reproducible as-is
(`run_episode(RandChooser(seed), ...)`). Minimal hand-written reproductions live in `test_bugs.py`.

Two families:
- `SAFE`: action kinds not affected by the known defects (no cancel, reverse, hold or authority
  extension). Must pass; this keeps regression power while the defects are open.
- `FULL`: every action kind, one test per property. A property broken by a known defect is a strict
  xfail naming it.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from tests.hardening.simgen import (
    ALL_CHECKS,
    ALL_KINDS,
    SAFE_KINDS,
    STATS,
    RandChooser,
    gen_scenario,
    gen_step,
    internal,
    public,
    replay,
    run_episode,
    run_step,
)

HC = [HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much]
NO_SHRINK = [Phase.explicit, Phase.reuse, Phase.generate]  # a seed does not shrink meaningfully
SEEDS = st.integers(0, 2**31 - 1)
SAFE_CHECKS = ALL_CHECKS - {"belief", "rotation", "gridlock_sound", "hold"}


def episode(seed, *, checks, kinds, n_steps=40, factors=None, finish=True):
    c = RandChooser(seed)
    bias = c.bool(0.5)
    scenario = gen_scenario(c, factors=factors, gridlock_bias=bias)
    return run_episode(c, scenario, n_steps=n_steps, checks=checks, kinds=kinds, gridlock_bias=bias, finish=finish)


# --- SAFE action set: every check that is not broken by a known defect --------------------------


@settings(max_examples=60, deadline=None, suppress_health_check=HC, phases=NO_SHRINK)
@given(SEEDS)
def test_invariants_hold_for_safe_actions(seed):
    """One train per resource, reservation integrity, no movement without authority, contiguous moves,
    no early departure, loop length and platform, cost (non-negative, monotone, total == sum, equal to
    the PRD formulas), refusal is a no-op, check == apply, no unrevealed event visible, decision points,
    gridlock completeness. All factors, events included."""
    episode(seed, checks=SAFE_CHECKS, kinds=SAFE_KINDS)


@pytest.mark.slow
@settings(max_examples=600, deadline=None, suppress_health_check=HC, phases=NO_SHRINK)
@given(SEEDS)
def test_invariants_hold_for_safe_actions_deep(seed):
    episode(seed, checks=SAFE_CHECKS, kinds=SAFE_KINDS, n_steps=60)


# --- FULL action set: one test per property ------------------------------------------------------

# name: (checks, examples, known defect or None)
FULL = {
    "resources": ({"resources"}, 60, "BUG-1/2/3/5: cancel, extension and reverse put two trains on one resource"),
    "movement": ({"movement", "departures"}, 60, "BUG-6: a reversing train accepts an authority and teleports"),
    "hold": ({"departures", "hold"}, 60, "CONTRACT-1: a held train with an authority departs"),
    "rotation": ({"rotation"}, 60, "BUG-8: an F6 successor with entry 0 runs before its predecessor"),
    "refusal_noop": ({"refusal_noop"}, 100, "BUG-4: a refused reverse drops the train's reservation"),
    "loops": ({"loops"}, 60, None),
    "cost": ({"cost"}, 60, None),
    "check_equals_apply": ({"check_equals_apply"}, 60, None),
    "no_unrevealed": ({"no_unrevealed"}, 60, None),
    "decision_points": ({"decision_points"}, 60, None),
    "gridlock_complete": (
        {"gridlock_complete"},
        60,
        "BUG-4 consequence: after a refused reverse drops a reservation the engine misses a real gridlock",
    ),
}


# `hold` leaves out `reverse` so the first violation found is the hold one, not BUG-6's teleport.
KINDS = {"hold": tuple(k for k in ALL_KINDS if k != "reverse")}


def _make(name, checks, examples, defect):
    kinds = KINDS.get(name, ALL_KINDS)

    @settings(max_examples=examples, deadline=None, suppress_health_check=HC, phases=NO_SHRINK)
    @given(SEEDS)
    def test(seed):
        episode(seed, checks=checks, kinds=kinds)

    test.__name__ = f"test_full_actions_{name}"
    test.__doc__ = f"Property `{name}` with every action kind ({', '.join(sorted(checks))})."
    if defect:
        test = pytest.mark.xfail(strict=True, reason=defect)(test)
    return test


for _name, (_checks, _n, _defect) in FULL.items():
    globals()[f"test_full_actions_{_name}"] = _make(_name, _checks, _n, _defect)


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="SPEC-1: the SCC rule reports resolvable waits as gridlock")
@settings(max_examples=400, deadline=None, suppress_health_check=HC, phases=NO_SHRINK)
@given(SEEDS)
def test_full_actions_gridlock_sound(seed):
    """Every train the simulator reports in a gridlock is really deadlocked (rare case: slow)."""
    episode(seed, checks={"gridlock_sound"}, kinds=ALL_KINDS)


# --- replay and clone (T02-47, T02-48) -----------------------------------------------------------


@settings(max_examples=60, deadline=None, suppress_health_check=HC, phases=NO_SHRINK)
@given(SEEDS)
def test_T02_47_replay_identical_random(seed):
    """Replaying the same action list on a fresh load gives identical results, states and cost."""
    trace = episode(seed, checks=set(), kinds=SAFE_KINDS + ("reverse", "hold"))
    results, states, sim = replay(trace.scenario, trace.steps)
    assert results == trace.results
    assert states == trace.states
    assert sim.cost() == trace.final.cost()


@settings(max_examples=60, deadline=None, suppress_health_check=HC, phases=NO_SHRINK)
@given(SEEDS)
def test_T02_48_clone_independent_random(seed):
    """Mutating a clone never changes the original; clone and original then evolve identically."""
    c = RandChooser(seed)
    bias = c.bool(0.5)
    scenario = gen_scenario(c, gridlock_bias=bias)
    kinds = SAFE_KINDS + ("reverse", "hold")
    trace = run_episode(c, scenario, n_steps=c.int(0, 25), checks=set(), kinds=kinds, finish=False)
    sim = trace.final
    before_internal, before_public = internal(sim), public(sim)
    clone = sim.clone()
    steps = []
    for _ in range(c.int(1, 15)):
        step = gen_step(c, clone, kinds, bias)
        steps.append(step)
        run_step(clone, step)
        assert internal(sim) == before_internal
        assert public(sim) == before_public
    for step in steps:
        run_step(sim, step)
    assert public(sim) == public(clone)
    assert internal(sim) == internal(clone)


# --- how often the generators exercise gridlock and reversal ------------------------------------


def gridlock_statistics(n: int = 200, *, biased: bool = True) -> dict:
    """Run `n` generated episodes (all action kinds) and count GRIDLOCK, reversals and crashes."""
    STATS.clear()
    for seed in range(n):
        c = RandChooser(seed)
        try:
            run_episode(c, gen_scenario(c, gridlock_bias=biased), n_steps=40, checks=set(), gridlock_bias=biased)
        except IndexError:  # BUG-1b crash; counted here, reproduced in test_bugs.py
            STATS["crashes"] += 1
    return dict(STATS)


def test_generators_exercise_gridlock_and_reversal():
    """Frequency guard: the biased generator must hit GRIDLOCK and a granted reverse often."""
    stats = gridlock_statistics(200)
    gridlock = stats.get("episodes_with_gridlock", 0) / stats["episodes"]
    reversal = stats.get("episodes_with_reversal", 0) / stats["episodes"]
    print(f"gridlock in {gridlock:.0%} of episodes, reversal in {reversal:.0%}; {stats}")
    assert gridlock >= 0.15
    assert reversal >= 0.10
