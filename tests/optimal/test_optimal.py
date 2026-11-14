"""The exact solver (railway/ref/optimal.py): its bound, its plan and the simulator must agree."""

import json
from pathlib import Path

import pytest

pytest.importorskip("ortools")

from railway.gen.generator import generate  # noqa: E402
from railway.gen.presets import PRESETS  # noqa: E402
from railway.policies.baselines import Greedy  # noqa: E402
from railway.policies.runner import run_policy  # noqa: E402
from railway.ref.optimal import dispatch, solve  # noqa: E402
from railway.ref.reference import Reference  # noqa: E402
from railway.sim.model import Scenario  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def fixture(name: str) -> Scenario:
    return Scenario.model_validate(json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8")))


# The expected costs come from an exhaustive search of the real simulator (every legal plan tried).
@pytest.mark.parametrize(("name", "optimum"), [("mini3", 3.0), ("mini2_gridlock", 48.0), ("mini3_noloops", 85.0)])
def test_matches_exhaustive_search_on_small_scenarios(name, optimum):
    scenario = fixture(name)
    bound, plan = solve(scenario, 60, bound_only=True), solve(scenario, 60)
    cost, _, undelivered = dispatch(scenario, plan["plan"])
    assert (bound["status"], plan["status"]) == ("OPTIMAL", "OPTIMAL")
    assert bound["cost"] == plan["cost"] == cost == optimum
    assert not undelivered


@pytest.mark.slow
@pytest.mark.parametrize("seed", [5, 10, 56])
def test_t1_plan_is_proven_and_playable_within_budget(seed):
    scenario = generate(PRESETS["T1"], seed)
    bound, plan = solve(scenario, 120, bound_only=True), solve(scenario, 120)
    cost, calls, undelivered = dispatch(scenario, plan["plan"])
    assert (bound["status"], plan["status"]) == ("OPTIMAL", "OPTIMAL")
    assert bound["cost"] == plan["cost"] == cost
    assert not undelivered and calls <= 600


@pytest.mark.slow
def test_no_dispatcher_beats_the_bound():
    scenario = generate(PRESETS["T1"], 5)
    bound = solve(scenario, 120, bound_only=True)["cost"]
    assert bound == 4.2
    for policy in (Greedy(), Reference()):
        assert run_policy(policy, scenario).final.cost.total >= bound


def test_scenarios_with_events_are_refused():
    with pytest.raises(ValueError, match="F1-F4"):
        solve(generate(PRESETS["T2"], 0))
