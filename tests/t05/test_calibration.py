"""T05-32..35 (slow): greedy on T2 dev seeds stays inside the DESIGN targets."""

import statistics

import pytest

from railway.gen.generator import generate
from railway.gen.presets import PRESETS
from railway.gen.seeds import candidate_seeds
from railway.policies.baselines import Greedy
from railway.policies.runner import run_policy
from railway.sim.constants import TOOL_BUDGET

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def runs():
    return [run_policy(Greedy(), generate(PRESETS["T2"], s)) for s in candidate_seeds("dev", count=10)]


def test_T05_32_t2_decision_points(runs):
    mean = statistics.mean(r.decision_points for r in runs)
    print(f"T2 greedy mean decision points: {mean:.1f}")
    assert 30 <= mean <= 120


def test_T05_33_t2_output_size(runs):
    mean = statistics.mean(r.output_chars for r in runs)
    print(f"T2 greedy mean advance+compact-status characters: {mean:.0f}")
    assert mean <= 250_000


def test_T05_34_t2_greedy_speed(runs):
    scenario = generate(PRESETS["T2"], next(iter(candidate_seeds("dev", count=1))))
    mcp = run_policy(Greedy(), scenario, transport="mcp")
    print(f"T2 greedy wall time: in-process max {max(r.wall_s for r in runs):.2f}s, MCP {mcp.wall_s:.2f}s")
    assert max(r.wall_s for r in runs) < 3
    assert mcp.wall_s < 30


def test_T05_35_t2_greedy_budget(runs):
    worst = max(r.tool_calls for r in runs)
    print(f"T2 greedy max tool calls: {worst} / {TOOL_BUDGET}")
    assert worst < 0.7 * TOOL_BUDGET
