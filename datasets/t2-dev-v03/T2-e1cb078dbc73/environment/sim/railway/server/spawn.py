"""Start `railway-sim` as a subprocess on a free local port (tests and the MCP runner)."""

from __future__ import annotations

import contextlib
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from railway.sim.model import Scenario
from railway.util import canonical_json


@dataclass
class Spawned:
    base: str
    token: str
    proc: subprocess.Popen
    log_path: Path
    captured: str | None = None  # the output, kept once the temp dir is gone

    @property
    def url(self) -> str:
        return self.base + "/mcp"

    def output(self) -> str:
        if self.captured is not None:
            return self.captured
        return self.log_path.read_text(encoding="utf-8", errors="replace")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stop(proc: subprocess.Popen, timeout: float):
    if sys.platform == "win32":
        # A venv python.exe is a launcher with the real interpreter as its child: kill the whole tree.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.terminate()
    proc.wait(timeout)


@contextlib.contextmanager
def spawn(scenario: Scenario, timeout: float = 60):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "scenario.json").write_text(canonical_json(scenario.model_dump(mode="json")), encoding="utf-8")
        token = secrets.token_hex(16)
        (tmp / "token").write_text(token, encoding="utf-8")
        port = _free_port()
        cmd = [sys.executable, "-m", "railway.server", "--scenario", str(tmp / "scenario.json")]
        cmd += ["--token-file", str(tmp / "token"), "--host", "127.0.0.1", "--port", str(port)]
        # Output goes to a file, not a pipe: a full pipe buffer would block the server.
        with open(tmp / "output.log", "wb") as out:
            proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
            spawned = Spawned(f"http://127.0.0.1:{port}", token, proc, tmp / "output.log")
            try:
                deadline = time.monotonic() + timeout
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"railway-sim exited with {proc.returncode}: {spawned.output()}")
                    with contextlib.suppress(httpx.HTTPError):
                        if httpx.get(spawned.base + "/health", timeout=1).text == "ok":
                            break
                    if time.monotonic() > deadline:
                        raise TimeoutError("railway-sim did not become healthy")
                    time.sleep(0.1)
                yield spawned
            finally:
                _stop(proc, timeout)
        spawned.captured = spawned.output()
