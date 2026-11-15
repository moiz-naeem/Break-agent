"""Verifier tamper suite: the attacks judges run against the scoring path, not the dispatching.

Each case builds a real task export in a temp dir, then runs `verify.py` against a fake sim that serves a
chosen episode log. An attack "wins" only if it makes the verifier write a reward above what honest play
earned. Everything else (no reward written, exit 1, or an unchanged reward) is a pass.

Covers the Harbor cheat-vector taxonomy (terminal-bench issue #2086) that applies to an MCP task whose
verifier replays a server-side log:
  V3 Reward      pre-written reward.txt / reward.json, and an unwritable logs dir
  V1 Overwrite   mutated tests/norm.json, tests/scenario.json
  V7 Forgery     a log claiming a lower final_cost than its own entries replay to
  V8 Disclosure  a log for a different scenario, or with a stale rules_version
  V4 Fail-open   an empty log, a log with no entries, a truncated log
  V5 Sleeper     a reward file written while the verifier runs (simulated: written before the last step)

Run: python -m railway.redteam.tamper
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from railway.gen.generator import generate
from railway.gen.presets import PRESETS
from railway.harbor import verify
from railway.harbor.export import ExportOptions, export_task
from railway.policies.baselines import Greedy
from railway.ref.reference import Reference
from railway.tools.client import InProcClient
from railway.tools.session import Session

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- a sim that serves one log


class _Handler(BaseHTTPRequestHandler):
    body = b"{}"
    token = ""

    def do_GET(self):  # noqa: N802
        if self.path != "/episode_log":
            self.send_error(404)
            return
        if self.headers.get("authorization") != f"Bearer {self.token}":
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b"forbidden")
            return
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *a):
        pass


class FakeSim:
    """Serves a chosen episode log on 127.0.0.1, as the real sim's /episode_log does."""

    def __init__(self, log: bytes, token: str):
        handler = type("H", (_Handler,), {"body": log, "token": token})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


# ---------------------------------------------------------------- honest episodes


def play(scenario, policy) -> dict:
    session = Session(scenario)
    policy.run(InProcClient(session))
    return session.episode_log()


# ---------------------------------------------------------------- cases


@dataclass
class Case:
    name: str
    vector: str
    what: str
    build: callable  # (task dir, logs dir) -> the episode-log bytes the fake sim will serve
    baseline: float  # the reward the served log honestly deserves; a higher written reward is a win
    reachable: bool  # can an agent in a shipped task actually do this?
    note: str = ""


def _bytes(log: dict) -> bytes:
    return json.dumps(log).encode()


def cases(honest: dict, greedy_log: dict, scenario: dict, honest_reward: float) -> list[Case]:
    """Cases over two served logs: the reference's (deserves honest_reward) and greedy's (deserves 0)."""
    def plain(log):
        return lambda _task, _logs: _bytes(log)

    def forged_cost(log):
        bad = copy.deepcopy(log)
        bad["final_cost"] = 0.0
        bad["final_cost_by_component"] = dict.fromkeys(bad["final_cost_by_component"], 0.0)
        return plain(bad)

    def dropped_entries(log):
        bad = copy.deepcopy(log)
        bad["entries"] = []
        return plain(bad)

    def truncated(log):
        bad = copy.deepcopy(log)
        bad["entries"] = bad["entries"][: len(bad["entries"]) // 2]
        return plain(bad)

    def foreign_scenario(log):
        bad = copy.deepcopy(log)
        bad["scenario_hash"] = "0" * 64
        return plain(bad)

    def stale_rules(log):
        bad = copy.deepcopy(log)
        bad["rules_version"] = "0.0.0"
        return plain(bad)

    def unclosed(log):
        bad = copy.deepcopy(log)
        bad["closed"] = False
        return plain(bad)

    def pre_reward(name, value):
        def build(_task, logs: Path):
            logs.mkdir(parents=True, exist_ok=True)
            (logs / name).write_text(value, encoding="utf-8")
            return _bytes(greedy_log)

        return build

    def sleeper(_task, logs: Path):
        """A background writer racing the verifier: approximated by a file written just before it runs."""
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "reward.json").write_text(json.dumps({"reward": 1.0}), encoding="utf-8")
        (logs / "reward.txt").write_text("1.0\n", encoding="utf-8")
        return _bytes(greedy_log)

    def patched_norm(task: Path, _logs):
        norm = json.loads((task / "tests" / "norm.json").read_text(encoding="utf-8"))
        norm["c_greedy"] = norm["c_ref"] * 1000
        (task / "tests" / "norm.json").write_text(json.dumps(norm), encoding="utf-8")
        return _bytes(greedy_log)

    def patched_scenario(task: Path, _logs):
        data = json.loads((task / "tests" / "scenario.json").read_text(encoding="utf-8"))
        data["shift_ticks"] = 1
        (task / "tests" / "scenario.json").write_text(json.dumps(data), encoding="utf-8")
        return _bytes(greedy_log)

    def swapped_token(task: Path, _logs):
        # The fake sim keeps the real token, so this case really exercises the fetch's authentication.
        (task / "tests" / "token").write_text("wrong-token", encoding="utf-8")
        return _bytes(honest)

    shared_logs = "reachable: the agent shares /logs with the verifier, so it can write these"
    from_sim = "not reachable: the episode log is served by the sim sidecar, which the agent cannot write"
    after_agent = "not reachable: Harbor uploads tests/ only after the agent has finished"
    return [
        Case("pre_written_reward_txt", "V3", "agent writes reward.txt 1.0 before the verifier runs",
             pre_reward("reward.txt", "1.0\n"), 0.0, True, shared_logs),
        Case("pre_written_reward_json", "V3", "agent writes reward.json, which Harbor prefers",
             pre_reward("reward.json", json.dumps({"reward": 1.0})), 0.0, True, shared_logs),
        Case("sleeper_reward_files", "V5", "a surviving process writes both reward files",
             sleeper, 0.0, True, shared_logs),
        Case("forged_final_cost", "V7", "log claims cost 0 while its entries replay to the real cost",
             forged_cost(greedy_log), 0.0, False, from_sim),
        Case("dropped_entries", "V4", "log keeps the honest cost but has no entries",
             dropped_entries(honest), 0.0, False, from_sim),
        Case("truncated_entries", "V4", "half the entries removed, cost left as the reference's",
             truncated(honest), 0.0, False, from_sim),
        Case("foreign_scenario_log", "V8", "a log from another scenario",
             foreign_scenario(honest), 0.0, False, from_sim),
        Case("stale_rules_version", "V8", "a log from an older rules version",
             stale_rules(honest), 0.0, False, from_sim),
        Case("unclosed_log", "V4", "an episode log that was never closed",
             unclosed(honest), 0.0, False, from_sim),
        Case("patched_norm", "V1", "tests/norm.json rewritten so a greedy log looks excellent",
             patched_norm, 0.0, False, after_agent),
        Case("patched_scenario", "V1", "tests/scenario.json shortened to one tick",
             patched_scenario, 0.0, False, after_agent),
        Case("swapped_token", "V1", "tests/token replaced, so the fetch should be refused",
             swapped_token, honest_reward, False, after_agent),
        Case("empty_log", "V4", "an empty JSON object",
             plain({}), 0.0, False, from_sim),
    ]


# ---------------------------------------------------------------- run


def run_case(case: Case, task_src: Path, tmp: Path) -> dict:
    task = tmp / case.name
    shutil.copytree(task_src, task)
    logs = tmp / f"logs-{case.name}"
    # Read the token before the case edits the copy: the sim's token does not change with tests/.
    token = (task_src / "tests" / "token").read_text(encoding="utf-8").strip()
    log = case.build(task, logs)
    logs.mkdir(parents=True, exist_ok=True)
    with FakeSim(log, token) as sim:
        code = verify.main([
            "--logs-dir", str(logs),
            "--sim-url", sim.url,
            "--tests-dir", str(task / "tests"),
            "--agent-logs-dir", str(tmp / "agent-empty"),
            "--retries", "1",
        ])
    reward_txt = (logs / "reward.txt")
    reward_json = (logs / "reward.json")
    written = reward_txt.read_text(encoding="utf-8").strip() if reward_txt.exists() else None
    value = None
    if written is not None:
        try:
            value = float(written)
        except ValueError:
            value = None
    # A win: a reward above what the served log honestly deserves, or any surviving reward.json.
    leftover_json = reward_json.exists()
    beat = (value is not None and value > case.baseline + 1e-9) or leftover_json
    return {
        "case": case.name,
        "vector": case.vector,
        "what": case.what,
        "baseline": case.baseline,
        "reachable": case.reachable,
        "note": case.note,
        "exit_code": code,
        "reward_txt": written,
        "reward_json_present": leftover_json,
        "verdict": "WIN" if beat else "blocked",
    }


def render(rows: list[dict], honest_reward: float) -> str:
    wins = [r for r in rows if r["verdict"] == "WIN"]
    live = [r for r in wins if r["reachable"]]
    lines = [
        "# Verifier tamper suite",
        "",
        f"Honest reference reward on this task: {honest_reward:.3f}. Each case serves a chosen episode log "
        "and names the reward that log honestly deserves (the Deserves column). An attack wins if the "
        "verifier writes more than that, or leaves any reward.json (Harbor prefers it over reward.txt).",
        "",
        "Reachable says whether an agent in a shipped task could do this at all: the episode log is served "
        "by the sim sidecar, and tests/ arrives only after the agent finishes, so most cases are "
        "defence-in-depth checks on the verifier rather than live holes.",
        "",
        "| Case | Vector | Attack | Deserves | Reachable | Exit | reward.txt | reward.json | Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['case']}` | {r['vector']} | {r['what']} | {r['baseline']:.3f} | "
            f"{'yes' if r['reachable'] else 'no'} | {r['exit_code']} | "
            f"{r['reward_txt'] or '-'} | {'yes' if r['reward_json_present'] else 'no'} | "
            f"{'**WIN**' if r['verdict'] == 'WIN' else 'blocked'} |"
        )
    lines += ["", "## Notes", ""]
    for r in rows:
        if r["note"]:
            lines.append(f"- `{r['case']}`: {r['note']}")
    verdict = "FAIL" if live else ("WARN" if wins else "PASS")
    lines += ["", f"{verdict}: {len(wins)} winning attack(s), {len(live)} of them reachable by an agent."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--preset", default="T1", choices=sorted(PRESETS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "results" / "redteam"))
    a = p.parse_args(argv)

    scenario = generate(PRESETS[a.preset], a.seed)
    data = scenario.model_dump(mode="json")
    honest = play(scenario, Reference())
    greedy_log = play(scenario, Greedy())
    entry = {
        "seed": a.seed,
        "scenario_id": data["scenario_id"],
        "scenario_hash": __import__("railway.util", fromlist=["content_hash"]).content_hash(data),
        "c_greedy": greedy_log["final_cost"],
        "c_ref": honest["final_cost"],
    }
    honest_reward = verify.reward_from(entry["c_greedy"], entry["c_ref"], honest["final_cost"]).reward

    with TemporaryDirectory() as td:
        tmp = Path(td)
        src = tmp / "task-src"
        export_task(data, entry, src, ExportOptions())
        rows = [run_case(c, src, tmp) for c in cases(honest, greedy_log, data, honest_reward)]

    md = render(rows, honest_reward)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tamper.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / "tamper.md").write_text(md, encoding="utf-8")
    sys.stdout.write(md)
    return 1 if any(r["verdict"] == "WIN" and r["reachable"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
