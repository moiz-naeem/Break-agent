"""Shared T07 fixtures.

Two export sources:
- "fixture": tasks exported from `tests/fixtures/` scenarios with test-made anchors (runs today);
- "t1": the real `export_dataset("T1", "dev", ...)` path; skipped until T04 (generator) and T06
  (accepted seed lists) are merged, then it switches on by itself.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from railway.sim.engine import Simulator
from railway.sim.model import Scenario
from tests.helpers import load_fixture
from tests.t07.fake_replay import play
from tests.t07.fakes import norm_entry, scripted_plan

REAL_EXPORT_MODULES = ("railway.gen.generator", "railway.ref.seed_filter")


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def require_modules(*names: str) -> None:
    for name in names:
        pytest.importorskip(name, reason=f"{name} not merged yet (T04/T05/T06)")


def do_nothing_cost(scenario: dict) -> float:
    sim = Simulator.load(Scenario.model_validate(scenario))
    sim.end_shift()
    return sim.cost().total


def fixture_anchors(scenario: dict) -> dict:
    """Real anchors, as the verifier re-derives them: C_greedy from greedy, C_best from the best plan."""
    from railway.policies.baselines import Greedy
    from railway.policies.runner import run_policy
    from railway.ref.best import best_plan
    from railway.sim.model import Scenario

    sc = Scenario.model_validate(scenario)
    best = best_plan(sc)
    entry = norm_entry(scenario, c_greedy=run_policy(Greedy(), sc).final.cost.total, c_ref=best.cost)
    entry.update(best_source=best.source, best_proven=best.proven, plan=best.plan)
    return entry


@pytest.fixture(scope="session")
def fixture_items() -> list[tuple[dict, dict]]:
    items = []
    for name in ("mini3", "graph_mini"):  # the other fixtures have no headroom: greedy is already optimal
        scenario = load_fixture(name)
        items.append((scenario, fixture_anchors(scenario)))
    return items


@pytest.fixture(params=["fixture", "t1"])
def exporter(request, fixture_items):
    """A callable `(out, **option_overrides) -> manifest` exporting 2 tasks into `out`."""
    from railway.harbor.export import ExportOptions, export_dataset, export_scenarios

    if request.param == "fixture":
        return lambda out, **kw: export_scenarios(
            fixture_items, Path(out), ExportOptions(**kw), meta={"preset": "fixture", "set": "fixture"}
        )
    require_modules(*REAL_EXPORT_MODULES)
    return lambda out, **kw: export_dataset("T1", "dev", 2, Path(out), ExportOptions(**kw))


def task_dirs(out: Path) -> list[Path]:
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    return [out / t["task"] for t in manifest["tasks"]]


def run_verify(task_dir: Path, logs: Path, sim_url: str, *extra: str, agent_logs: Path | None = None):
    """Run the task's vendored `tests/verify.py` as Harbor would (its own folder first on sys.path)."""
    cmd = [
        sys.executable,
        str(task_dir / "tests" / "verify.py"),
        "--logs-dir",
        str(logs),
        "--sim-url",
        sim_url,
        "--retries",
        "2",
        "--retry-wait",
        "0",
        "--agent-logs-dir",
        str(agent_logs or (logs.parent / "agent-logs-absent")),
        *extra,
    ]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run(cmd, capture_output=True, text=True, cwd=logs.parent, env=env, timeout=300)
