"""Test-only stand-ins for T07 while T04/T05/T06 are not merged.

- `FakeSim`: a stdlib HTTP server with `/health` and a token-protected `/episode_log` (the T05 routes the
  verifier uses), serving whatever body a test gives it.
- `scripted_plan`: a simple deterministic plan played directly on the real Simulator, used as the
  "reference" anchor of fixture tasks (NOT the T06 reference).
- `fixture_task`: exports one task from a `tests/fixtures/` scenario with `railway.harbor.export`, and,
  if the real `railway.tools.replay` is missing, drops `fake_replay.py` into the task's vendored tests copy.
"""

from __future__ import annotations

import contextlib
import hmac
import importlib.util
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from railway.sim.engine import Simulator
from railway.sim.model import Authorize, Scenario
from railway.util import content_hash

HERE = Path(__file__).resolve().parent


class FakeSim:
    """`/health` → ok; `/episode_log` → 403 without the right bearer token, else `body` (bytes)."""

    def __init__(self, token: str, body: bytes):
        self.token = token
        self.body = body
        self.log_requests = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 (stdlib name)
                if self.path == "/health":
                    return self._send(200, b"ok")
                if self.path == "/episode_log":
                    outer.log_requests += 1
                    auth = self.headers.get("Authorization", "")
                    if not hmac.compare_digest(auth.encode(), f"Bearer {outer.token}".encode()):
                        return self._send(403, b"forbidden")
                    return self._send(200, outer.body)
                return self._send(404, b"not found")

            def _send(self, code: int, body: bytes):
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> FakeSim:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()


@contextlib.contextmanager
def fake_sim(token: str, log: dict | bytes):
    body = log if isinstance(log, bytes) else json.dumps(log).encode()
    with FakeSim(token, body) as server:
        yield server


def scripted_plan(scenario: dict) -> list[tuple[str, dict]]:
    """Greedy-like plan on the true simulator: at each decision point authorize each waiting train to the
    farthest station ahead (yard: main; else a loop, then main) that `check()` grants; then advance."""
    sim = Simulator.load(Scenario.model_validate(scenario))
    chain = ["YW", *[s["id"] for s in scenario["network"]["stations"]], "YE"]
    loops = {s["id"]: [lp["id"] for lp in s["loops"]] for s in scenario["network"]["stations"]}
    calls: list[tuple[str, dict]] = []
    while not sim.done:
        for t in sorted(sim.state().trains, key=lambda t: t.id):
            if t.phase != "at_station" or t.authority is not None or t.station is None:
                continue
            i = chain.index(t.station)
            ahead = chain[i + 1 :] if t.direction == "east" else list(reversed(chain[:i]))
            for station in reversed(ahead):
                tracks = ["main"] if station in ("YW", "YE") else [*loops[station], "main"]
                granted = next(
                    (tr for tr in tracks if sim.check(Authorize(train=t.id, station=station, track=tr)).ok), None
                )
                if granted:
                    args = {"train": t.id, "station": station, "track": granted}
                    sim.apply(Authorize(**args))
                    calls.append(("authorize", args))
                    break
        sim.advance()
        calls.append(("advance_to_next_event", {}))
    return calls


def norm_entry(scenario: dict, c_greedy: float, c_best: float) -> dict:
    """An accepted-list entry whose top anchor is `c_best` (labelled best-known; the reference costs the same)."""
    return {
        "seed": scenario["seed"],
        "scenario_id": scenario["scenario_id"],
        "scenario_hash": content_hash(scenario),
        "c_greedy": c_greedy,
        "c_best": c_best,
        "c_best_kind": "best-known",
        "c_ref": c_best,
    }


def real_replay_available() -> bool:
    try:
        return importlib.util.find_spec("railway.tools.replay") is not None
    except ModuleNotFoundError:
        return False


def use_fake_replay(task_dir: Path) -> None:
    """Install the test fake as the task's vendored replay module, only if T05's real one is absent."""
    if not real_replay_available():
        shutil.copyfile(HERE / "fake_replay.py", task_dir / "tests" / "railway" / "tools" / "replay.py")
