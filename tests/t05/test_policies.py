"""T05-26..31: baseline policies, played through a Client only."""

import ast
import pathlib

import pytest

from railway.gen.generator import generate
from railway.gen.presets import PRESETS
from railway.gen.seeds import candidate_seeds
from railway.policies.baselines import DoNothing, Greedy, RandomLegal, StopEarly
from railway.policies.runner import run_policy
from tests.t05.common import session


def mini3():
    return session().scenario


def t2_dev(n):
    return [generate(PRESETS["T2"], seed) for seed in candidate_seeds("dev", count=n)]


def test_T05_26_greedy_mini3_exact():
    result = run_policy(Greedy(), mini3())
    assert result.final.cost.total == 6.5  # L5 8 late at S2, S1, YW: 3 x 0.25 x f(8); F21 4 late: 2
    assert result.final.cost.unfinished == 0
    assert result.final.counts["refusals"] == result.final.counts["invalid_calls"] == 0


def test_T05_27_do_nothing():
    result = run_policy(DoNothing(), mini3())
    assert result.tool_calls == 0 and result.log == []
    assert result.final.cost.total == 5810.75


def test_T05_28_stop_early():
    result = run_policy(StopEarly(), t2_dev(1)[0])
    assert result.log[-1]["tool"] == "end_shift"
    advances = [e for e in result.log if e["tool"] == "advance_to_next_event"]
    ticks = [e["tick"] for e in result.log]
    assert all(t < 30 for t in ticks[:-2] if t is not None)  # decided before 30 ...
    assert result.log[-1]["tick"] >= 30 and advances  # ... and ended at the first decision tick >= 30


def test_T05_29_random_legal():
    scenario = t2_dev(1)[0]
    a, b, c = (run_policy(RandomLegal(seed), scenario) for seed in (1, 1, 2))
    assert a.log == b.log
    assert a.log != c.log
    for r in (a, c):
        assert r.final.counts["refusals"] == r.final.counts["invalid_calls"] == 0


@pytest.mark.parametrize("make", [DoNothing, Greedy, StopEarly, lambda: RandomLegal(3)])
def test_T05_30_policies_over_mcp(make):
    for scenario in [mini3(), *t2_dev(2)]:
        live = run_policy(make(), scenario, transport="mcp")
        local = run_policy(make(), scenario)
        assert live.final == local.final
        assert [(e["tool"], e["args"], e["status"]) for e in live.log] == [
            (e["tool"], e["args"], e["status"]) for e in local.log
        ]


def test_T05_31_policies_import_hygiene():
    banned = {"railway.tools.session", "railway.gen"}
    for path in pathlib.Path("railway/policies").glob("*.py"):
        if path.name == "runner.py":  # the runner owns the session; policies never see it
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not any(module == b or module.startswith(b + ".") for b in banned), (path, module)
                if module == "railway.sim.model":
                    assert "Scenario" not in {a.name for a in node.names}, path
            elif isinstance(node, ast.Import):
                assert not any(a.name.startswith(tuple(banned)) for a in node.names), path
