"""The v0.3 engine on a chain scenario must move trains exactly as the v0.2 engine did (only cost changed).

Replays random action scripts (tests/v03/make_scripts.py) through the v0.2 engine, taken from git at
V02_COMMIT, and through this checkout's engine, and compares the full trajectories with cost left out.
"""

import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V02_COMMIT = "c7cce53"  # last commit with the v0.2 engine


def _git_archive(dest: Path) -> bool:
    if shutil.which("git") is None:
        return False
    archive = dest / "old.tar"
    done = subprocess.run(
        ["git", "archive", "-o", str(archive), V02_COMMIT, "railway"], cwd=ROOT, capture_output=True
    )
    if done.returncode != 0:
        return False
    with tarfile.open(archive) as tar:
        tar.extractall(dest / "old", filter="data")
    return True


def _drive(pythonpath: Path, scenario: Path, script: Path) -> list:
    out = subprocess.run(
        [sys.executable, str(HERE / "drive_engine.py"), str(scenario), str(script)],
        env={"PYTHONPATH": str(pythonpath), "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


@pytest.mark.slow
def test_v03_chain_trajectories_match_v02(tmp_path):
    if not _git_archive(tmp_path):
        pytest.skip(f"git or commit {V02_COMMIT} not available")
    subprocess.run(
        [sys.executable, str(HERE / "make_scripts.py"), str(tmp_path / "scripts"), "6"],
        env={"PYTHONPATH": str(ROOT), "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")},
        check=True,
    )
    scenarios = sorted((tmp_path / "scripts").glob("*.scenario.json"))
    assert len(scenarios) == 12
    for scenario in scenarios:
        script = scenario.with_name(scenario.name.replace(".scenario.", ".script."))
        old = _drive(tmp_path / "old", scenario, script)
        new = _drive(ROOT, scenario, script)
        assert old == new, scenario.name
