"""Red-team panel regression tests. Marked slow: each case plays whole episodes."""

import json

import pytest

from railway.gen.generator import generate
from railway.gen.presets import PRESETS
from railway.policies.baselines import Greedy
from railway.redteam import strategic, tamper
from railway.redteam.strategic import (
    ATTACKERS,
    best_plan,
    plan_attackers,
    plan_neutral,
    play,
    prefix_costs,
)

PRESET = "T1"


@pytest.fixture(scope="module")
def entry():
    return strategic.accepted(PRESET, "dev")[0]


@pytest.fixture(scope="module")
def scenario(entry):
    return generate(PRESETS[PRESET], entry["seed"])


@pytest.fixture(scope="module")
def plan(entry):
    calls = best_plan(PRESET, entry)
    if calls is None:
        pytest.skip("this seed stores no best plan and no exported task carries one")
    return calls


def _cost(policy, scenario):
    return play(scenario, policy)[0]["cost"]


@pytest.mark.slow
def test_no_attacker_beats_greedy(scenario, entry):
    """Every attacker in the battery costs at least what greedy costs, so its official reward is 0.

    Greedy is the reward's bottom anchor (v0.3: `max(0, (C_greedy - C_agent) / (C_greedy - C_best))`),
    so this is the exact condition for "no non-dispatching policy scores".
    """
    greedy = _cost(Greedy(), scenario)
    assert greedy == pytest.approx(entry["c_greedy"], rel=1e-6), "the seed list's C_greedy is stale"
    for name, make in ATTACKERS.items():
        assert _cost(make(), scenario) >= greedy - 1e-6, name


@pytest.mark.slow
def test_the_stored_plan_still_scores_one(scenario, entry, plan):
    """The 1.0 trajectory has to replay to C_best, or every reward on this seed is measured wrong."""
    replay = strategic.Replay(plan)
    assert _cost(replay, scenario) == pytest.approx(entry["c_best"], rel=1e-6)


@pytest.mark.slow
def test_cancel_churn_is_cost_neutral(scenario, entry, plan):
    """Cancelling and re-granting the plan's own authorities changes nothing: no farmable free action."""
    churn = plan_neutral(plan)["PlanCancelChurn"]
    assert _cost(churn, scenario) == pytest.approx(entry["c_best"], rel=1e-6)


@pytest.mark.slow
def test_no_plan_mutation_is_cheaper_than_the_plan(scenario, entry, plan):
    """There is no clip at the top in v0.3, so a mutation cheaper than C_best is a real finding."""
    for name, policy in plan_attackers(plan).items():
        assert _cost(policy, scenario) >= entry["c_best"] - 1e-6, name


@pytest.mark.slow
def test_prefix_curve_is_monotone_and_ends_at_the_plan(scenario, entry, plan):
    """Partial play must not earn partial score: only the whole plan reaches C_best."""
    prefixes = prefix_costs(scenario, plan)
    assert len(prefixes) > 5
    assert prefixes[0][3] > entry["c_best"]  # stopping at once is far worse
    assert prefixes[-1][3] == pytest.approx(entry["c_best"], rel=1e-6)
    assert [p[2] for p in prefixes] == sorted(p[2] for p in prefixes)  # order share only grows
    partial = [p for p in prefixes if p[2] < 0.9]
    assert all(p[3] >= entry["c_greedy"] for p in partial), "a partly played shift must not beat greedy"


@pytest.mark.slow
def test_tamper_suite_has_no_agent_reachable_win(tmp_path):
    """The scoring path holds against every case an agent in a shipped task could actually run."""
    code = tamper.main(["--preset", PRESET, "--seed", "0", "--out", str(tmp_path)])
    rows = json.loads((tmp_path / f"tamper-{PRESET}-0.json").read_text(encoding="utf-8"))
    reachable = [r for r in rows if r["reachable"]]
    assert reachable, "the suite must cover at least one attack an agent can really run"
    assert [r for r in rows if r["verdict"] == "WIN" and r["reachable"]] == []
    assert code == 0


@pytest.mark.slow
def test_only_patching_the_verifier_itself_wins(tmp_path):
    """Documented in docs/redteam-playbook.md: `tests/` integrity is the whole trust boundary.

    v0.3 closed the v0.2 `patched_norm` win, because the verifier re-derives both anchors from the
    frozen scenario. What remains is the trivial one: rewrite `tests/verify.py` and it writes what you
    tell it. If this fails because the win list changed, the playbook's findings section is stale.
    """
    tamper.main(["--preset", PRESET, "--seed", "0", "--out", str(tmp_path)])
    rows = json.loads((tmp_path / f"tamper-{PRESET}-0.json").read_text(encoding="utf-8"))
    wins = sorted(r["case"] for r in rows if r["verdict"] == "WIN")
    assert wins == ["patched_verifier"], wins
