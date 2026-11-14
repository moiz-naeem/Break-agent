"""Tests for railway/ref/offline_search.py on the three fixtures.

Hand-derived optima (full knowledge, honest play), each reproduced by the exhaustive search:

- mini2_gridlock = 48.0. No loops, so the trains cannot cross: one waits at its yard until the other
  has left the line. L1 runs on time and arrives at YE at 14; L2 leaves YE at 14 and is 14 late at
  S2, S1 and YW: (14-2) + (14-2) + 2*(14-2) = 48.
- mini3_noloops = 85.0. Same idea, L1 clears S3-YE at 19, L2 is 19 late at its four stops:
  3*17 + 2*17 = 85.
- mini3 = 3.0. E1 waits on S2/L1 for L5 (S2-S3 frees at 9): YE at 15, late 2 = grace, cost 0. L5
  runs on time. F21 (600 m) fits only S2/L2, cannot reach S2 before L5 needs S1-S2 at 10, and L5 may
  not slip more than 2 ticks for free; so F21 waits on S1/main while L5 takes S1/L1, enters S1-S2 at
  14 and reaches YE at 32: 0.5 * 6 = 3.0. Letting F21 go first costs L5 4 ticks = 2 + 4 = 6.
"""

from __future__ import annotations

import pytest

from railway.ref.offline_search import (
    exhaustive_search,
    fast_clone,
    lower_bound,
    offline_search,
    plan_from_log,
    replay_plan,
)
from railway.sim.engine import Simulator
from railway.sim.model import Authorize, Scenario
from tests.hardening.simgen import SAFE_KINDS, RandChooser, gen_step, internal, public, run_step
from tests.helpers import load_fixture

OPTIMUM = {"mini2_gridlock": 48.0, "mini3": 3.0, "mini3_noloops": 85.0}
DO_NOTHING = {"mini2_gridlock": 576.0, "mini3": 1235.0, "mini3_noloops": 556.0}


def A(train, station, track="main"):
    return Authorize(train=train, station=station, track=track)


# The optimal plans the exhaustive search returned (kept so the default run need not search again).
OPTIMAL_PLANS = {
    "mini2_gridlock": [(0, [A("L1", "YE")]), (14, [A("L2", "YW")])],
    "mini3_noloops": [(0, [A("L1", "YE")]), (19, [A("L2", "YW")])],
    "mini3": [
        (0, [A("E1", "S2"), A("L5", "S2", "L1")]),
        (3, [A("F21", "S1")]),
        (9, [A("E1", "YE")]),
        (10, [A("L5", "S1", "L1")]),
        (14, [A("F21", "S3")]),
        (15, [A("L5", "YW")]),
        (25, [A("F21", "YE")]),
    ],
}


def do_nothing_cost(name):
    sim = Simulator.load(Scenario.model_validate(load_fixture(name)))
    sim.end_shift()
    return sim.cost().total


@pytest.mark.parametrize("name", sorted(OPTIMUM))
def test_fixture_reference_costs(name):
    assert do_nothing_cost(name) == DO_NOTHING[name]
    assert replay_plan(load_fixture(name), OPTIMAL_PLANS[name]).cost().total == OPTIMUM[name]


def test_exhaustive_mini2_gridlock():
    result = exhaustive_search(load_fixture("mini2_gridlock"))
    assert result.proven
    assert result.cost == OPTIMUM["mini2_gridlock"] <= DO_NOTHING["mini2_gridlock"]
    assert replay_plan(load_fixture("mini2_gridlock"), result.plan).cost().total == result.cost
    assert replay_plan(load_fixture("mini2_gridlock"), plan_from_log(result.log)).cost().total == result.cost


@pytest.mark.slow
@pytest.mark.parametrize("name", ["mini3", "mini3_noloops"])
def test_exhaustive_slow_fixtures(name):
    result = exhaustive_search(load_fixture(name))
    assert result.proven
    assert result.cost == OPTIMUM[name]
    assert result.cost <= DO_NOTHING[name]
    assert replay_plan(load_fixture(name), result.plan).cost().total == result.cost


def test_beam_search_is_never_worse_than_its_seed():
    cost, log = offline_search(load_fixture("mini3"), seed_plan=OPTIMAL_PLANS["mini3"])
    assert cost == OPTIMUM["mini3"]
    assert replay_plan(load_fixture("mini3"), plan_from_log(log)).cost().total == cost
    cost, _ = offline_search(load_fixture("mini2_gridlock"))
    assert OPTIMUM["mini2_gridlock"] <= cost <= DO_NOTHING["mini2_gridlock"]


@pytest.mark.parametrize("name", sorted(OPTIMUM))
def test_beam_search_never_costs_more_than_doing_nothing(name):
    cost, _ = offline_search(load_fixture(name), beam=4, per_train=3)
    assert OPTIMUM[name] <= cost <= DO_NOTHING[name]


def test_never_move_freight_on_mini3():
    freight = frozenset({"F21"})
    result = exhaustive_search(load_fixture("mini3"), never_move=freight)
    # E1 and L5 cross at S2 for free; F21 is charged 200 + 0.5 * (60 - 26).
    assert result.cost == 217.0


@pytest.mark.parametrize("name", sorted(OPTIMUM))
def test_fast_clone_behaves_like_clone(name):
    sim = Simulator.load(Scenario.model_validate(load_fixture(name)))
    c = RandChooser(7)
    for _ in range(10):
        run_step(sim, gen_step(c, sim, SAFE_KINDS))
    a, b = sim.clone(), fast_clone(sim)
    assert internal(a) == internal(b)
    for _ in range(20):
        step = gen_step(c, a, SAFE_KINDS)
        run_step(a, step)
        run_step(b, step)
        assert public(a) == public(b)
    assert public(sim) != public(a) or sim.tick == a.tick  # the original is untouched by both


@pytest.mark.parametrize("seed", range(30))
def test_lower_bound_is_admissible(seed):
    """lower_bound(state) never exceeds the cost of any continuation (here: random ones)."""
    name = sorted(OPTIMUM)[seed % 3]
    sim = Simulator.load(Scenario.model_validate(load_fixture(name)))
    c = RandChooser(seed)
    for _ in range(c.int(0, 15)):
        run_step(sim, gen_step(c, sim, SAFE_KINDS))
    bound = lower_bound(sim)
    for k in range(3):
        cont = fast_clone(sim)
        cc = RandChooser(seed * 10 + k)
        while not cont.done:
            run_step(cont, gen_step(cc, cont, SAFE_KINDS))
        assert bound <= cont.cost().total + 1e-9
    idle = fast_clone(sim)
    idle.end_shift()
    assert bound <= idle.cost().total + 1e-9
    assert bound >= 0
