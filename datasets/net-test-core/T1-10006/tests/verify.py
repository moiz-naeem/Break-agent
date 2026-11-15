"""Harbor verifier for one v0.3 railway network task (decisions.md #2, #4, #10).

Vendored as `<task>/tests/verify.py` next to `tests/railway/` (`__init__.py`, `util.py`, `net/`),
`timetable.json`, and `token`; Harbor runs it from `tests/test.sh` in the main container after the
agent has finished.

Reward = -(C1 + C2 + C3 + C4 + C5) = -cost_total. Greedy and reference anchors are completely removed:
no `norm.json`, no `c_greedy`, no `c_ref`, no `c_best`, and no normalised score.

Outcomes:
- clean replay -> writes `details.json`, `action_log.json`, `events.jsonl`, then `reward.txt` LAST; exit 0;
- anything else -> prints `INFRA: ...` or `REPLAY MISMATCH: ...`, writes no reward, exits 1, so Harbor
  records a trial error (to rerun), never a false score.
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

from railway.net.cost import COMPONENTS, CostBreakdown, cost
from railway.net.events import Event, EventLog
from railway.net.formats import NET_RULES_VERSION
from railway.net.model import Timetable
from railway.net.session import COUNT_KEYS, FinalResult, Session
from railway.util import canonical_json, content_hash

STATUSES = {"GRANTED", "REFUSED", "OK", "INVALID", "INFO"}
TOLERANCE = 1e-6
SHIFT_ENDED = "R-SHIFT-ENDED"

# Agent-log scan limits: enough for OpenCode's logs, bounded so a huge log cannot stall the verifier.
MAX_LOG_FILES = 2000
MAX_LOG_BYTES = 20_000_000
FAILURE_RE = re.compile(r"\b(unavailable|failed|failure|error|refused|timeout|timed out|unreachable)\b")


class VerifyError(Exception):
    """Any non-clean outcome. `str(e)` is printed to stdout; no reward is written."""


class Infra(VerifyError):
    def __init__(self, what: str):
        super().__init__(f"INFRA: {what}")


class Mismatch(VerifyError):
    def __init__(self, what: str):
        super().__init__(f"REPLAY MISMATCH: {what}")


@dataclass(frozen=True)
class VerifiedResult:
    reward: float
    cost_total: float
    details: dict


# ---------------------------------------------------------------- fetch


def fetch_log(sim_url: str, token: str, retries: int, retry_wait: float) -> bytes:
    """GET {sim}/episode_log with the bearer token. Connection problems and 5xx are retried; 4xx is not."""
    request = urllib.request.Request(
        f"{sim_url.rstrip('/')}/episode_log",
        headers={"Authorization": f"Bearer {token}"},
    )
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


def _is_int(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x: object) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)


def _valid_entry(e: object) -> bool:
    return (
        isinstance(e, dict)
        and _is_int(e.get("seq"))
        and _is_int(e.get("tick"))
        and isinstance(e.get("tool"), str)
        and isinstance(e.get("args"), dict)
        and e.get("status") in STATUSES
        and (e.get("rule") is None or isinstance(e.get("rule"), str))
    )


def load_timetable(path: Path) -> tuple[Timetable, str, str]:
    """Load `timetable.json` and derive `(timetable, scenario_id, scenario_hash)`."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Infra("missing or corrupt timetable.json") from exc
    if not isinstance(raw, dict):
        raise Infra("corrupt timetable.json")
    if "rules_version" in raw and raw.get("rules_version") != NET_RULES_VERSION:
        raise Infra("timetable rules_version mismatch")
    clean = {k: v for k, v in raw.items() if k != "rules_version"}
    try:
        timetable = Timetable.model_validate(clean)
    except Exception as exc:
        raise Infra("invalid timetable.json") from exc
    scenario_id = f"{timetable.tier}-{timetable.seed}"
    scenario_hash = content_hash(timetable.model_dump(mode="json"))
    return timetable, scenario_id, scenario_hash


def parse_log(raw: bytes, expected_id: str, expected_hash: str) -> dict:
    """Parse the episode log and verify it belongs to this timetable and rules version."""
    corrupt = Infra("corrupt or foreign episode log")
    try:
        log = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise corrupt from None
    if not isinstance(log, dict):
        raise corrupt
    entries = log.get("entries")
    ok = (
        log.get("scenario_hash") == expected_hash
        and log.get("scenario_id") == expected_id
        and log.get("rules_version") == NET_RULES_VERSION
        and log.get("closed") is True
        and _is_int(log.get("final_tick"))
        and _is_number(log.get("final_cost"))
        and isinstance(log.get("final_cost_by_component"), dict)
        and all(_is_number(v) for v in log["final_cost_by_component"].values())
        and isinstance(entries, list)
        and all(_valid_entry(e) for e in entries)
        and [e["seq"] for e in entries] == list(range(1, len(entries) + 1))
    )
    if not ok:
        raise corrupt
    return log


def tool_calls(log: dict) -> int:
    """Budgeted calls: every logged call except those refused because the shift had already ended."""
    return sum(1 for e in log["entries"] if e["rule"] != SHIFT_ENDED)


# ---------------------------------------------------------------- MCP startup failure rule


def _mcp_failure_line(line: str) -> bool:
    """Match agent-log lines indicating our `railway` MCP server failed or was unavailable."""
    low = line.lower()
    names_us = "key=railway" in low or "server=railway" in low or ("mcp" in low and "railway" in low)
    return names_us and FAILURE_RE.search(low) is not None


def mcp_startup_failure(agent_logs: Path) -> str | None:
    """Return the first agent-log line saying our `railway` MCP server failed, or None."""
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


# ---------------------------------------------------------------- replay and cost


def replay_episode(timetable: Timetable, log: dict) -> tuple[FinalResult, Session]:
    """Replay deterministically through `railway.net.replay.replay` and capture the replayed Session."""
    from railway.net import replay as net_replay

    captured: list[Session] = []
    orig_session = net_replay.Session

    def _capturing_session(*a, **kw) -> Session:
        s = orig_session(*a, **kw)
        captured.append(s)
        return s

    net_replay.Session = _capturing_session  # type: ignore[assignment]
    try:
        final = net_replay.replay(timetable, log)
    except net_replay.ReplayMismatch as e:
        raise Mismatch(str(e) or "status") from None
    finally:
        net_replay.Session = orig_session  # type: ignore[assignment]

    if captured:
        return final, captured[-1]
    fallback = Session(timetable)
    for entry in log["entries"]:
        fallback.call(entry["tool"], entry["args"])
    fallback.close()
    return final, fallback


def compare(breakdown: CostBreakdown, final: FinalResult, log: dict) -> dict[str, float]:
    """Replayed cost, components, tick, counts, and KPIs must match the live server's log."""
    comp = breakdown.model_dump(mode="json", exclude={"total"})
    final_comp = final.cost.model_dump(mode="json", exclude={"total"})
    logged = dict(log["final_cost_by_component"])
    logged_total = logged.pop("total", log["final_cost"])
    same_cost = (
        set(comp) == set(final_comp) == set(logged) == set(COMPONENTS)
        and abs(breakdown.total - final.cost.total) <= TOLERANCE
        and abs(breakdown.total - log["final_cost"]) <= TOLERANCE
        and abs(logged_total - log["final_cost"]) <= TOLERANCE
        and all(abs(comp[k] - logged[k]) <= TOLERANCE and abs(comp[k] - final_comp[k]) <= TOLERANCE for k in COMPONENTS)
    )
    if not same_cost:
        raise Mismatch("cost")
    if final.tick != log["final_tick"]:
        raise Mismatch("final_tick")
    if "counts" in log and log["counts"] != final.counts:
        raise Mismatch("counts")
    if "kpis" in log and log["kpis"] != final.kpis:
        raise Mismatch("kpis")
    return comp


def format_events_jsonl(events: list[Event]) -> str:
    if not events:
        return ""
    return "".join(canonical_json(ev.model_dump(mode="json")) + "\n" for ev in events)


# ---------------------------------------------------------------- outputs


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def clear_rewards(logs_dir: Path) -> None:
    """Remove any pre-existing reward files before running verification."""
    for name in ("reward.txt", "reward.json"):
        try:
            (logs_dir / name).unlink(missing_ok=True)
        except OSError:
            raise Infra(f"cannot remove a pre-existing {name}") from None


def _read_probe(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def verify(args: argparse.Namespace) -> VerifiedResult:
    tests = Path(args.tests_dir)
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    clear_rewards(logs_dir)

    timetable, scenario_id, scenario_hash = load_timetable(tests / "timetable.json")
    try:
        token = (tests / "token").read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise Infra("missing token") from exc
    if not token:
        raise Infra("empty token")

    raw_log = fetch_log(args.sim_url, token, args.retries, args.retry_wait)
    log = parse_log(raw_log, scenario_id, scenario_hash)
    calls = tool_calls(log)
    if calls == 0:
        line = mcp_startup_failure(Path(args.agent_logs_dir))
        if line is not None:
            raise Infra(f"agent started without MCP tools (railway server unavailable): {line}")

    final, replayed_session = replay_episode(timetable, log)
    engine_log = getattr(replayed_session.engine, "log", None)
    event_log = engine_log if isinstance(engine_log, EventLog) else EventLog()
    breakdown = cost(timetable, event_log)
    comp = compare(breakdown, final, log)

    reward = 0.0 if breakdown.total == 0 else round(-breakdown.total, 6) + 0.0
    counts = {k: int(final.counts[k]) for k in COUNT_KEYS}
    details: dict = {
        "scenario_id": scenario_id,
        "scenario_hash": scenario_hash,
        "tier": timetable.tier,
        "seed": timetable.seed,
        "rules_version": NET_RULES_VERSION,
        "reward": reward,
        "cost_total": breakdown.total,
        "cost_by_component": comp,
        "kpis": dict(final.kpis),
        "counts": counts,
        "tool_calls": calls,
        "final_tick": log["final_tick"],
    }
    if args.probe:
        details["probe"] = _read_probe(Path(args.agent_logs_dir) / "probe.json")

    _write(logs_dir / "details.json", json.dumps(details, indent=2, sort_keys=True) + "\n")
    _write(logs_dir / "action_log.json", canonical_json(log))
    _write(logs_dir / "events.jsonl", format_events_jsonl(event_log.events))
    _write(logs_dir / "reward.txt", f"{reward}\n")  # written LAST: only a clean replay reaches here
    return VerifiedResult(reward=reward, cost_total=breakdown.total, details=details)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score one v0.3 railway network episode by replaying its log.")
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
        result = verify(args)
    except VerifyError as e:
        print(e)
        return 1
    except Exception as e:  # a crash is a trial error, never a false score; keep free of secrets
        print(f"INFRA: verifier error ({type(e).__name__})")
        return 1
    print(f"reward {result.reward} (cost_total {result.cost_total})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
