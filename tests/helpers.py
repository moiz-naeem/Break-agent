"""Shared test helpers (PRD §5.2)."""
from __future__ import annotations

import contextlib
import copy
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def variant(base: dict, **patches) -> dict:
    """Deep copy of `base` with dotted-path patches, e.g. variant(d, **{"trains.0.pax": 7})."""
    out = copy.deepcopy(base)
    for path, value in patches.items():
        *parents, last = path.split(".")
        node = out
        for key in parents:
            node = node[int(key)] if isinstance(node, list) else node[key]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value
    return out


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def start_server(args: list[str], timeout: float = 30.0):
    """Run `python <args> --host 127.0.0.1 --port <free>`; yield the base URL once /health is ok."""
    port = free_port()
    # A file, not a pipe: an unread pipe fills up with access logs and blocks the server.
    log = tempfile.TemporaryFile()
    proc = subprocess.Popen([sys.executable, *args, "--host", "127.0.0.1", "--port", str(port)],
                            stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + timeout
        while True:
            if proc.poll() is not None:
                log.seek(0)
                raise RuntimeError(f"server exited: {log.read().decode(errors='replace')}")
            try:
                if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("server did not become healthy")
            time.sleep(0.2)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


@dataclass
class TrialResult:
    reward: float | None
    errored: bool
    verifier_details: dict | None
    raw_dir: Path


def harbor_run(task_path, agent: str, model: str | None = None, extra_args=()) -> list[TrialResult]:
    """Run `harbor run` on a task or dataset and parse each trial's result.

    Trial layout (F-LAYOUT in docs/spike-report.md): <jobs>/<job>/<trial>/{config.json,result.json,
    verifier/reward.txt}. Extra hosts for allowlist runs come from $SPIKE_ALLOW_HOSTS.
    """
    if shutil.which("harbor") is None:
        raise RuntimeError("harbor CLI not on PATH")
    (REPO / "jobs").mkdir(exist_ok=True)  # git-ignored; one place to inspect or clean up
    jobs = Path(tempfile.mkdtemp(prefix="harbor-", dir=REPO / "jobs"))
    cmd = ["harbor", "run", "-y", "-p", str(task_path), "-a", agent, "-o", str(jobs), *extra_args]
    if model:
        cmd += ["-m", model]
        # Keys stay in .env; Harbor loads the file and maps the named var to the provider's key.
        if os.environ.get("HARBOR_ENV_FILE"):
            cmd += ["--env-file", os.environ["HARBOR_ENV_FILE"]]
        if os.environ.get("HARBOR_API_KEY_NAME"):
            cmd += ["--api-key-name", os.environ["HARBOR_API_KEY_NAME"]]
    for host in os.environ.get("SPIKE_ALLOW_HOSTS", "").split():
        cmd += ["--allow-environment-host", host]
    # Harbor is a separate uv tool; PYTHONPATH lets it import custom agents from this repo.
    # UTF-8: Harbor prints Unicode tables; on a Windows cp1252 pipe it crashes before any trial starts.
    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    subprocess.run(cmd, check=False, env=env)
    results = []
    for trial in sorted(p.parent for p in jobs.glob("*/*/result.json")):  # <jobs>/<job>/<trial>/
        verifier = trial / "verifier"
        reward_file = verifier / "reward.txt"
        details_file = verifier / "details.json"
        reward = float(reward_file.read_text()) if reward_file.exists() else None
        details = json.loads(details_file.read_text()) if details_file.exists() else None
        res = json.loads((trial / "result.json").read_text())
        errored = reward is None or bool(res.get("exception_info"))
        results.append(TrialResult(reward, errored, details, trial))
    if not results:
        raise RuntimeError(f"no trial results found under {jobs}")
    return results
