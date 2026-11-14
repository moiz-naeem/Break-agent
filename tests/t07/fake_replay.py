"""TEST FAKE of `railway.tools.replay` (T05), for the T07 verifier tests while T05 is not merged.

It is copied into a temporary task's `tests/railway/tools/replay.py`, never into the repo's `railway/`.
It follows the T05 contract (`replay(scenario, log) -> FinalResult`, `ReplayMismatch`) with a minimal
session: mutating tools go through the real `railway.sim` Simulator, read-only tools are `INFO`, and
after the shift ends every call is `REFUSED R-SHIFT-ENDED`. No budget, no output strings.

`play(scenario, calls)` produces a PRD §7.9 episode log the same way, standing in for the live server.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from railway.sim.cost import CostBreakdown
from railway.sim.engine import Simulator
from railway.sim.model import ActionAdapter, Scenario
from railway.util import content_hash

MUTATING = {"authorize", "cancel_authority", "hold", "relieve_crew"}
READ_ONLY = {"manual", "network", "timetable", "status", "bulletins", "kpis", "stringline", "notes"}
INVALID_RULES = {"R-BAD-ARGS", "R-UNKNOWN-TRAIN", "R-UNKNOWN-STATION", "R-UNKNOWN-TRACK"}


class ReplayMismatch(Exception):
    pass


@dataclass
class FinalResult:
    tick: int
    cost: CostBreakdown
    counts: dict = field(default_factory=dict)


class _FakeSession:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.sim = Simulator.load(scenario)
        self.closed = False
        self.counts = dict.fromkeys(
            ("gridlocks", "crew_timeouts", "refusals", "invalid_calls", "decision_points"), 0
        )

    def call(self, tool: str, args: dict) -> tuple[str, str | None]:
        if self.closed:
            return "REFUSED", "R-SHIFT-ENDED"
        if tool in MUTATING:
            try:
                action = ActionAdapter.validate_python({"kind": tool, **args})
            except Exception:
                self.counts["invalid_calls"] += 1
                return "INVALID", "R-BAD-ARGS"
            res = self.sim.apply(action)
            if res.ok:
                return ("GRANTED" if tool == "authorize" else "OK"), None
            rule = str(getattr(res.rule, "value", res.rule))
            if rule in INVALID_RULES:
                self.counts["invalid_calls"] += 1
                return "INVALID", rule
            self.counts["refusals"] += 1
            return "REFUSED", rule
        if tool == "advance_to_next_event":
            self._note(self.sim.advance())
            self.counts["decision_points"] += 1
            if self.sim.done:
                self.close()
            return "INFO", None
        if tool == "end_shift":
            self.close()
            return "INFO", None
        if tool in READ_ONLY:
            return "INFO", None
        self.counts["invalid_calls"] += 1
        return "INVALID", "R-BAD-ARGS"

    def _note(self, result) -> None:
        for r in result.reasons:
            if r.kind == "GRIDLOCK":
                self.counts["gridlocks"] += 1
            elif r.kind == "CREW_TIMEOUT":
                self.counts["crew_timeouts"] += 1

    def close(self) -> FinalResult:
        if not self.closed:
            self.sim.end_shift()
            self.closed = True
        return FinalResult(tick=self.sim.tick, cost=self.sim.cost(), counts=dict(self.counts))


def _as_scenario(scenario) -> Scenario:
    return scenario if isinstance(scenario, Scenario) else Scenario.model_validate(scenario)


def replay(scenario, log) -> FinalResult:
    session = _FakeSession(_as_scenario(scenario))
    for entry in log["entries"]:
        status, rule = session.call(entry["tool"], entry["args"])
        if status != entry["status"] or rule != entry["rule"]:
            logged = f"{entry['status']} {entry['rule']}"
            raise ReplayMismatch(f"seq {entry['seq']}: logged {logged}, replayed {status} {rule}")
    return session.close()


def play(scenario: dict, calls: list[tuple[str, dict]]) -> dict:
    """Play `calls` and return the PRD §7.9 episode log, as the live server's `/episode_log` would."""
    session = _FakeSession(_as_scenario(scenario))
    entries = []
    for seq, (tool, args) in enumerate(calls, start=1):
        tick = session.sim.tick
        status, rule = session.call(tool, args)
        entries.append({"seq": seq, "tick": tick, "tool": tool, "args": args, "status": status, "rule": rule})
    final = session.close()
    cost = final.cost.model_dump()
    total = cost.pop("total")
    return {
        "scenario_id": scenario["scenario_id"],
        "scenario_hash": content_hash(scenario),
        "rules_version": scenario["rules_version"],
        "entries": entries,
        "closed": True,
        "final_tick": final.tick,
        "final_cost": total,
        "final_cost_by_component": cost,
    }
