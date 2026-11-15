"""Red-team panel regression tests. Marked slow: each case plays whole episodes."""

import json

import pytest

from railway.gen.generator import generate
from railway.gen.presets import PRESETS
from railway.policies.baselines import Greedy
from railway.redteam import strategic, tamper
from railway.redteam.strategic import ATTACKERS, NEUTRAL, prefix_costs, reference_calls
from railway.tools.client import InProcClient
from railway.tools.session import Session

PRESET = "T1"


@pytest.fixture(scope="module")
def entry():
    return strategic.accepted(PRESET, "dev")[0]


@pytest.fixture(scope="module")
def scenario(entry):
    return generate(PRESETS[PRESET], entry["seed"])


def _cost(policy, scenario):
    session = Session(scenario)
    policy.run(InProcClient(session))
    return session.close().cost.total


@pytest.mark.slow
def test_no_attacker_beats_greedy(scenario, entry):
    """Every attacker in the battery costs at least what greedy costs, so its official reward is 0."""
    greedy = _cost(Greedy(), scenario)
    for name, make in ATTACKERS.items():
        assert _cost(make(), scenario) >= greedy - 1e-6, name


@pytest.mark.slow
def test_cancel_churn_is_cost_neutral(scenario):
    """Cancelling and re-granting the reference's own authorities changes nothing: no farmable action."""
    _, c_ref = reference_calls(scenario)
    assert _cost(NEUTRAL["CancelChurn"](), scenario) == pytest.approx(c_ref)


@pytest.mark.slow
def test_prefix_curve_is_monotone_and_ends_at_the_reference(scenario):
    """The cut-off sweep ends at the full reference cost and never rewards a cut above the full run."""
    calls, c_ref = reference_calls(scenario)
    prefixes = prefix_costs(scenario, calls)
    assert len(prefixes) > 5
    assert prefixes[0][3] > c_ref  # stopping at once is far worse
    assert prefixes[-1][3] == pytest.approx(c_ref)
    assert [p[2] for p in prefixes] == sorted(p[2] for p in prefixes)  # order share only grows
    partial = [p for p in prefixes if p[2] < 0.5]
    assert all(p[3] > c_ref for p in partial), "half-played episodes must cost more than the reference"


@pytest.mark.slow
def test_tamper_suite_has_no_agent_reachable_win(tmp_path):
    """The scoring path holds against every case an agent in a shipped task could actually run."""
    code = tamper.main(["--preset", PRESET, "--seed", "0", "--out", str(tmp_path)])
    rows = json.loads((tmp_path / "tamper.json").read_text(encoding="utf-8"))
    reachable = [r for r in rows if r["reachable"]]
    assert reachable, "the suite must cover at least one attack an agent can really run"
    assert [r for r in rows if r["verdict"] == "WIN" and r["reachable"]] == []
    assert code == 0


@pytest.mark.slow
def test_patched_norm_is_still_the_only_known_win(tmp_path):
    """Documented in docs/redteam-playbook.md: norm.json's numbers are trusted, nothing else is.

    If this fails because the win is gone, the playbook's 'known finding' section is stale.
    """
    tamper.main(["--preset", PRESET, "--seed", "0", "--out", str(tmp_path)])
    rows = json.loads((tmp_path / "tamper.json").read_text(encoding="utf-8"))
    wins = sorted(r["case"] for r in rows if r["verdict"] == "WIN")
    assert wins == ["patched_norm"], wins
