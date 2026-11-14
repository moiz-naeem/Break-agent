"""T07 §5.3: real Harbor runs (marker `harbor`; T07-H8 also `model`). Need Docker + the Harbor CLI.

Fallback tests (§5.4, T07-F1..F3) are not written: the spike's `Decision: SIDECAR` means the single-container
export is not built (ticket §2.8).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.helpers import REPO, harbor_run
from tests.t07.conftest import REAL_EXPORT_MODULES, has_module, require_modules, task_dirs

pytestmark = pytest.mark.harbor

MINI = REPO / "datasets" / "t1-dev-v03"


def _ok(trials, expect: str) -> None:
    """The reward is −cost. `expect`: "best" (the oracle replays the best plan: reward == −c_best) or
    "below_greedy" (do-nothing, stop-early: reward < −c_greedy)."""
    assert trials
    for t in trials:
        assert not t.errored, t.raw_dir
        d = t.verifier_details
        assert t.reward == pytest.approx(-d["cost_total"], abs=1e-6), t.raw_dir
        if expect == "best":
            assert t.reward == pytest.approx(-d["c_best"], abs=1e-6), t.raw_dir
        else:
            assert t.reward < -d["c_greedy"], t.raw_dir


def _export(out: Path, preset: str = "T1", count: int = 1, **opts) -> list[Path]:
    require_modules(*REAL_EXPORT_MODULES)
    from railway.harbor.export import ExportOptions, export_dataset

    export_dataset(preset, "dev", count, out, ExportOptions(**opts))
    return task_dirs(out)


def test_T07_H1_oracle_all_tasks():
    assert MINI.is_dir(), "datasets/t1-dev-v03 is exported from seeds/T1-dev.json"
    trials = harbor_run(MINI, "oracle")
    assert len(trials) == 5
    _ok(trials, "best")


def test_T07_H2_nop_zero():
    _ok(harbor_run(MINI, "nop"), "below_greedy")  # built-in no-op name from the spike (F-NOP)


def test_T07_H3_stop_early_zero():
    # F-CUSTOM == PASS in the spike, so the agent must exist. Harbor imports it in its own tool env (which has
    # the `harbor` package); this venv doesn't, so check the module is there without importing it.
    assert has_module("railway.harbor.agents")
    _ok(harbor_run(MINI, "railway.harbor.agents:StopEarlyAgent"), "below_greedy")


def test_T07_H4_probe_isolation(tmp_path):
    _export(tmp_path / "ds", probe=True)
    probe_task = next(p for p in (tmp_path / "ds").iterdir() if p.name.endswith("-probe"))
    (trial,) = harbor_run(probe_task, "oracle")
    assert not trial.errored and trial.reward == pytest.approx(-trial.verifier_details["c_best"], abs=1e-6)
    probe = trial.verifier_details["probe"]
    assert probe["uid"] != "0"
    assert probe["github"] == "unreachable"
    assert probe["sim_health"] == "ok"
    assert probe["episode_log_no_token"] == "403" and probe["episode_log_wrong_token"] == "403"
    assert probe["scenario_files"] == ""
    assert probe["import_railway"] == "fails"


def test_T07_H5_image_contents(tmp_path):
    (task,) = _export(tmp_path / "ds")
    tag = f"railway-t07-main-{tmp_path.name.lower()}"
    subprocess.run(["docker", "build", "-t", tag, str(task / "environment")], check=True)
    try:
        script = "find / -xdev \\( -name 'scenario*.json' -o -name token \\) 2>/dev/null; python3 -c 'import railway'"
        res = subprocess.run(["docker", "run", "--rm", tag, "sh", "-c", script], capture_output=True, text=True)
        assert res.stdout.strip() == ""
        assert "ModuleNotFoundError" in res.stderr
        grep = f"grep -rl {task.name} / --exclude-dir=proc --exclude-dir=sys --exclude-dir=dev 2>/dev/null || true"
        res = subprocess.run(["docker", "run", "--rm", tag, "sh", "-c", grep], capture_output=True, text=True)
        assert res.stdout.strip() == ""
    finally:
        subprocess.run(["docker", "rmi", "-f", tag], check=False)


def test_T07_H6_sim_down_is_error(tmp_path):
    (task,) = _export(tmp_path / "ds")
    compose_path = task / "environment" / "docker-compose.yaml"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    compose["services"].pop("sim")
    compose["services"]["main"].pop("depends_on", None)
    if not compose["services"]["main"]:
        compose["services"]["main"] = {"environment": {"RAILWAY_T07_H6": "1"}}
    compose_path.write_text(yaml.safe_dump(compose), encoding="utf-8")
    shutil.rmtree(task / "environment" / "sim")
    for trial in harbor_run(task, "oracle"):
        assert trial.errored or trial.reward is None
        assert trial.reward is None


def test_T07_H7_t2_oracle(tmp_path):
    (task,) = _export(tmp_path / "ds", preset="T2")
    _ok(harbor_run(task, "oracle"), "best")


@pytest.mark.model
def test_T07_H8_opencode_smoke(tmp_path):
    model = os.environ.get("T07_SMOKE_MODEL", "anthropic/claude-haiku-4-5")
    (task,) = _export(tmp_path / "ds")
    (trial,) = harbor_run(task, "opencode", model=model)
    assert not trial.errored
    assert trial.reward == pytest.approx(-trial.verifier_details["cost_total"], abs=1e-6)
    assert trial.verifier_details["tool_calls"] > 0
