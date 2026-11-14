"""Harbor verifier for one railway task (ticket T07 §2.6; DESIGN.md §6.2, §8.3).

Vendored as `<task>/tests/verify.py` next to `tests/railway/` (util, sim, tools), `scenario.json`, `norm.json`
and `token`; Harbor runs it from `tests/test.sh` in the main container after the agent has finished.

Outcomes:
- clean replay → writes `details.json`, `action_log.json`, then `reward.txt` LAST; exit 0;
- anything else → prints `INFRA: ...` or `REPLAY MISMATCH: ...`, writes no reward, exits 1, so Harbor
  records a trial error (to rerun), never a 0.

MCP startup failure (docs/spike-report.md, T07 risk): an agent whose MCP client never connected makes zero
tool calls, which would look exactly like a do-nothing agent. The rule that tells them apart:
- zero budgeted tool calls in the episode log, AND
- the agent's own logs (`/logs/agent`) contain a line saying the `railway` MCP server failed/was unavailable
→ trial error. Zero calls without such a line (MCP healthy, or no agent logs at all) is a genuine
do-nothing and scores normally (0). Episodes with at least one tool call are always scored: the server was
reachable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from railway.sim.constants import RULES_VERSION
from railway.util import canonical_json, content_hash

STATUSES = {"GRANTED", "REFUSED", "OK", "INVALID", "INFO"}
COMPONENTS = ("pax_lateness", "freight_lateness", "crew_timeout", "unfinished")
COUNT_KEYS = ("gridlocks", "crew_timeouts", "refusals", "invalid_calls", "decision_points")
TOLERANCE = 1e-6
SHIFT_ENDED = "R-SHIFT-ENDED"  # calls after the shift ended are logged but use no budget (T05 §2.1)

# Agent-log scan limits: enough for OpenCode's logs, bounded so a huge log can't stall the verifier.
MAX_LOG_FILES = 2000
MAX_LOG_BYTES = 20_000_000
FAILURE_RE = re.compile(r"\b(unavailable|failed|failure|error|refused|timeout|timed out|unreachable)\b")


class VerifyError(Exception):
    """Any non-clean outcome. `str(e)` is the line printed to stdout; no reward is written."""


class Infra(VerifyError):
    def __init__(self, what: str):
        super().__init__(f"INFRA: {what}")


class Mismatch(VerifyError):
    def __init__(self, what: str):
        super().__init__(f"REPLAY MISMATCH: {what}")


@dataclass(frozen=True)
class Score:
    reward: float
    score_unclipped: float
    above_reference: bool


def reward_from(c_greedy: float, c_ref: float, c_agent: float) -> Score:
    """u = (C_greedy − C_agent) / (C_greedy − C_ref); reward = clip(u, 0, 1)."""
    gap = c_greedy - c_ref
    if not (gap > 0 and math.isfinite(gap)):
        raise ValueError(f"need c_greedy > c_ref, got {c_greedy} and {c_ref}")
    u = (c_greedy - c_agent) / gap
    return Score(reward=min(1.0, max(0.0, u)), score_unclipped=u, above_reference=u > 1.0)


# ---------------------------------------------------------------- fetch


def fetch_log(sim_url: str, token: str, retries: int, retry_wait: float) -> bytes:
    """GET {sim}/episode_log with the bearer token. Connection problems and 5xx are retried; 4xx is not."""
    request = urllib.request.Request(f"{sim_url.rstrip('/')}/episode_log", headers={"Authorization": f"Bearer {token}"})
    for attempt in range(max(1, retries)):
        if attempt:
            time.sleep(retry_wait)
        try:
            with urllib.request.urlopen(request, timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code < 500:
                raise Infra(f"episode_log refused (HTTP {e.code})") from None
        except (urllib.error.URLError, OSError):
            pass
    raise Infra("sim unreachable")


# ---------------------------------------------------------------- validate


def _is_number(x) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)


def parse_log(raw: bytes, scenario: dict) -> dict:
    """Parse the PRD §7.9 episode log and check it belongs to this task and to this code."""
    corrupt = Infra("corrupt or foreign episode log")
    try:
        log = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise corrupt from None
    if not isinstance(log, dict):
        raise corrupt
    ok = (
        log.get("scenario_hash") == content_hash(scenario)
        and log.get("scenario_id") == scenario.get("scenario_id")
        and log.get("rules_version") == RULES_VERSION
        and log.get("closed") is True
        and isinstance(log.get("final_tick"), int)
        and _is_number(log.get("final_cost"))
        and isinstance(log.get("final_cost_by_component"), dict)
        and all(_is_number(v) for v in log["final_cost_by_component"].values())
        and isinstance(log.get("entries"), list)
        and all(_valid_entry(e) for e in log["entries"])
    )
    if not ok:
        raise corrupt
    return log


def _valid_entry(e) -> bool:
    return (
        isinstance(e, dict)
        and isinstance(e.get("seq"), int)
        and isinstance(e.get("tick"), int)
        and isinstance(e.get("tool"), str)
        and isinstance(e.get("args"), dict)
        and e.get("status") in STATUSES
        and (e.get("rule") is None or isinstance(e.get("rule"), str))
    )


def load_norm(path: Path, scenario: dict) -> dict:
    try:
        norm = json.loads(path.read_text(encoding="utf-8"))
        reward_from(norm["c_greedy"], norm["c_ref"], norm["c_ref"])
        ok = norm["scenario_hash"] == content_hash(scenario) and norm["rules_version"] == RULES_VERSION
    except (OSError, ValueError, KeyError, TypeError):
        ok = False
    if not ok:
        raise Infra("bad norm.json")
    return norm


def tool_calls(log: dict) -> int:
    """Budgeted calls: every logged call except those refused because the shift had already ended."""
    return sum(1 for e in log["entries"] if e["rule"] != SHIFT_ENDED)


# ---------------------------------------------------------------- MCP startup failure rule


def _mcp_failure_line(line: str) -> bool:
    """OpenCode's spike line was `server unavailable key=<server>`; also accept other MCP-failure phrasings
    that name our server. Lines that don't name the `railway` MCP server (e.g. a probe's
    `No module named 'railway'`, or another server's failure) never match."""
    low = line.lower()
    names_us = "key=railway" in low or "server=railway" in low or ("mcp" in low and "railway" in low)
    return names_us and FAILURE_RE.search(low) is not None


def mcp_startup_failure(agent_logs: Path) -> str | None:
    """The first agent-log line saying our `railway` MCP server failed, or None. Never raises."""
    try:
        files = sorted(p for p in agent_logs.rglob("*") if p.is_file())[:MAX_LOG_FILES]
    except OSError:
        return None
    for path in files:
        try:
            with path.open("rb") as f:
                text = f.read(MAX_LOG_BYTES).decode("utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if _mcp_failure_line(line):
                return line.strip()[:300]
    return None


# ---------------------------------------------------------------- replay and compare


def _field(obj, name: str):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def _as_dict(obj) -> dict:
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(vars(obj))


def replay_log(scenario: dict, log: dict):
    """Replay through the vendored T05 `railway.tools.replay`. Imported here so infra checks never need it."""
    from railway.sim.model import Scenario
    from railway.tools.replay import ReplayMismatch, replay

    try:
        return replay(Scenario.model_validate(scenario), log)
    except ReplayMismatch as e:
        raise Mismatch(str(e) or "status") from None


def compare(final, log: dict) -> tuple[dict, float]:
    """Replayed cost and tick must equal the live server's (abs tolerance 1e-6)."""
    cost = _as_dict(_field(final, "cost"))
    total = cost.pop("total")
    logged = dict(log["final_cost_by_component"])
    logged_total = logged.pop("total", log["final_cost"])
    same = (
        set(cost) == set(logged) == set(COMPONENTS)
        and abs(total - log["final_cost"]) <= TOLERANCE
        and abs(logged_total - log["final_cost"]) <= TOLERANCE
        and all(abs(cost[k] - logged[k]) <= TOLERANCE for k in COMPONENTS)
    )
    if not same:
        raise Mismatch("cost")
    if _field(final, "tick") != log["final_tick"]:
        raise Mismatch("final_tick")
    return cost, total


# ---------------------------------------------------------------- outputs


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def clear_rewards(logs_dir: Path) -> None:
    """Remove reward files nobody but us may write (an agent could pre-write them; reward.json wins in Harbor)."""
    for name in ("reward.txt", "reward.json"):
        try:
            (logs_dir / name).unlink(missing_ok=True)
        except OSError:
            raise Infra(f"cannot remove a pre-existing {name}") from None


def verify(args: argparse.Namespace) -> Score:
    tests = Path(args.tests_dir)
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    clear_rewards(logs_dir)

    scenario = json.loads((tests / "scenario.json").read_text(encoding="utf-8"))
    norm = load_norm(tests / "norm.json", scenario)
    token = (tests / "token").read_text(encoding="utf-8").strip()
    if not token:
        raise Infra("empty token")

    log = parse_log(fetch_log(args.sim_url, token, args.retries, args.retry_wait), scenario)
    calls = tool_calls(log)
    if calls == 0:
        line = mcp_startup_failure(Path(args.agent_logs_dir))
        if line is not None:
            raise Infra(f"agent started without MCP tools (railway server unavailable): {line}")

    final = replay_log(scenario, log)
    cost, total = compare(final, log)
    score = reward_from(norm["c_greedy"], norm["c_ref"], total)
    counts = _as_dict(_field(final, "counts"))
    details = {
        "scenario_id": scenario["scenario_id"],
        "rules_version": RULES_VERSION,
        "cost_total": total,
        "cost_by_component": cost,
        "c_greedy": norm["c_greedy"],
        "c_ref": norm["c_ref"],
        "score_unclipped": score.score_unclipped,
        "reward": score.reward,
        "above_reference": score.above_reference,
        "counts": {k: counts[k] for k in COUNT_KEYS},
        "tool_calls": calls,
        "final_tick": log["final_tick"],
    }
    if args.probe:
        details["probe"] = _read_probe(Path(args.agent_logs_dir) / "probe.json")

    _write(logs_dir / "action_log.json", canonical_json(log))
    _write(logs_dir / "details.json", json.dumps(details, indent=2, sort_keys=True))
    _write(logs_dir / "reward.txt", f"{score.reward:.6f}\n")  # last action: only a clean replay gets here
    return score


def _read_probe(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score one railway episode by replaying its log.")
    p.add_argument("--logs-dir", required=True)
    p.add_argument("--sim-url", default="http://sim:8000")
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--retry-wait", type=float, default=2.0)
    p.add_argument("--tests-dir", default=str(Path(__file__).resolve().parent))
    p.add_argument("--agent-logs-dir", default="/logs/agent")
    p.add_argument("--probe", action="store_true", help="copy <agent-logs-dir>/probe.json into details.json")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        score = verify(args)
    except VerifyError as e:
        print(e)
        return 1
    except Exception as e:  # a crash is a trial error, never a 0; keep the message free of secrets
        print(f"INFRA: verifier error ({type(e).__name__})")
        return 1
    print(f"reward {score.reward:.6f} (unclipped {score.score_unclipped:.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
