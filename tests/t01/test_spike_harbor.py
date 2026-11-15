"""Harbor spike runs (DESIGN.md §8.4). Need Docker + the Harbor CLI; model tests need a model."""
import os

import pytest

from tests.helpers import REPO, harbor_run

pytestmark = pytest.mark.harbor
SPIKE = REPO / "archive" / "spike"


def test_T01_H1_oracle_reaches_sim():
    trials = harbor_run(SPIKE / "toy-task", "oracle")
    assert len(trials) == 1 and not trials[0].errored
    assert trials[0].reward == 1.0


def test_T01_H2_nop_scores_zero():
    trials = harbor_run(SPIKE / "toy-task", os.environ.get("HARBOR_NOP_AGENT", "nop"))
    assert not trials[0].errored and trials[0].reward == 0.0


def test_T01_H3_custom_agent_loads():
    trials = harbor_run(SPIKE / "toy-task", "railway.harbor.spike_agents:NoOpAgent")
    assert not trials[0].errored and trials[0].reward == 0.0


def test_T01_H4_sim_down_is_trial_error():
    trials = harbor_run(SPIKE / "toy-task-nosim", "oracle")
    assert trials[0].errored  # errored already covers a missing reward


def _model(var: str = "SPIKE_MODEL") -> str:
    m = os.environ.get(var)
    if not m:
        pytest.fail(f"set {var}=<provider/model> (key via HARBOR_ENV_FILE/HARBOR_API_KEY_NAME) for model tests")
    return m


@pytest.mark.model
def test_T01_H5_allowlist_opencode():
    trials = harbor_run(SPIKE / "toy-task-allowlist", "opencode", model=_model())
    assert not trials[0].errored and trials[0].reward == 1.0


@pytest.mark.model
def test_T01_H6_nonroot_opencode():
    trials = harbor_run(SPIKE / "toy-task-nonroot", "opencode", model=_model())
    assert not trials[0].errored and trials[0].reward == 1.0
    texts = [c["text"] for c in trials[0].verifier_details["calls"]]
    assert texts and all(t.strip() != "0" for t in texts)


@pytest.mark.model
def test_T01_H7_small_model():
    trials = harbor_run(SPIKE / "toy-task", "opencode", model=_model("SPIKE_SMALL_MODEL"))
    assert not trials[0].errored and trials[0].reward == 1.0
