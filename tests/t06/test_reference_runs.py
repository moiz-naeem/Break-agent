"""T06-09..13 (slow): the reference against greedy, its gridlocks, speed, budget and MCP parity."""

import time

import pytest

from railway.policies.baselines import Greedy
from railway.policies.runner import run_policy
from railway.ref.reference import Reference
from railway.sim.constants import TOOL_BUDGET
from tests.t06.common import cost, dev

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def t1_runs():
    return [(cost(Greedy(), s), run_policy(Reference(), s)) for s in dev("T1", 10)]


def test_T06_09_beats_greedy_t1(t1_runs):
    """First 10 T1 dev *candidate* seeds (unfiltered, so not circular)."""
    greedy = [g for g, _ in t1_runs]
    ref = [r.final.cost.total for _, r in t1_runs]
    print("T1 greedy", greedy, "reference", ref)
    assert sum(r <= g for g, r in zip(greedy, ref, strict=True)) >= 9
    assert sum(ref) / sum(greedy) <= 0.8


def test_T06_10_never_gridlocks(t1_runs):
    t2 = [run_policy(Reference(), s) for s in dev("T2", 5)]
    assert all(r.final.counts["gridlocks"] == 0 for _, r in t1_runs)
    assert all(r.final.counts["gridlocks"] == 0 for r in t2)


def test_T06_11_speed_t2():
    scenario = dev("T2", 1)[0]
    start = time.perf_counter()
    run_policy(Reference(), scenario)
    elapsed = time.perf_counter() - start
    print(f"reference T2 episode: {elapsed:.1f} s")
    assert elapsed < 120


def test_T06_12_budget_t2():
    calls = [run_policy(Reference(), s).tool_calls for s in dev("T2", 5)]
    print(f"reference T2 tool calls: {calls} / {TOOL_BUDGET}")
    assert max(calls) <= 0.7 * TOOL_BUDGET


def test_T06_13_mcp_parity():
    for scenario in [*dev("T1", 2), *dev("T2", 1)]:
        live = run_policy(Reference(), scenario, transport="mcp")
        local = run_policy(Reference(), scenario)
        assert [(e["tool"], e["args"], e["status"]) for e in live.log] == [
            (e["tool"], e["args"], e["status"]) for e in local.log
        ]
        assert live.final.cost == local.final.cost
