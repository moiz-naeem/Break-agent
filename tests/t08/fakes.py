"""Test-only stand-ins for modules other tickets own.

`FakeSession` stands in for the T05 `Session` (with the T08 `peek` addition) on top of the real
T02/T03 simulator, so costs are real but output formats are simplified. `fake_reward_from` stands
in for T07's `reward_from`. Tests that need the real modules use `pytest.importorskip`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, ValidationError

from railway.sim.constants import TOOL_BUDGET
from railway.sim.cost import CostBreakdown
from railway.sim.engine import Simulator
from railway.sim.model import Authorize, Scenario

FAKE_MANUAL = "FAKE MANUAL v1"
READ_ONLY = ("status", "bulletins", "kpis", "manual")


class _NoArgs(BaseModel):
    """Takes no arguments."""

    model_config = ConfigDict(extra="forbid")


class _StatusArgs(_NoArgs):
    compact: bool = False


class _NotesArgs(_NoArgs):
    text: str


class _AuthorizeArgs(_NoArgs):
    train: str
    station: str
    track: str


class _StringlineArgs(_NoArgs):
    from_tick: int
    to_tick: int


def _h(doc):
    def handler():  # pragma: no cover - only the docstring matters
        return None

    handler.__doc__ = doc
    return handler


# name -> (args model, handler); mirrors the shape T05 §2.1 describes.
FAKE_REGISTRY = {
    "manual": (_NoArgs, _h("Rules text. See manual().")),
    "status": (_StatusArgs, _h("Visible state. See manual().")),
    "bulletins": (_NoArgs, _h("Revealed events. See manual().")),
    "kpis": (_NoArgs, _h("Cost and counts. See manual().")),
    "notes": (_NotesArgs, _h("Replace the scratchpad. See manual().")),
    "stringline": (_StringlineArgs, _h("Time-distance diagram. See manual().")),
    "authorize": (_AuthorizeArgs, _h("Grant a movement authority. See manual().")),
    "advance_to_next_event": (_NoArgs, _h("Advance time. See manual().")),
    "end_shift": (_NoArgs, _h("End the shift. See manual().")),
}


@dataclass
class FakeFinal:
    tick: int
    cost: CostBreakdown
    counts: dict


@dataclass
class FakeToolCall:
    tool: str
    args: dict = field(default_factory=dict)


class FakeSession:
    """Minimal Session: budget, log, INVALID/REFUSED statuses, peek, close. Not T05's formats."""

    registry = FAKE_REGISTRY

    def __init__(self, scenario: Scenario, *, budget: int = TOOL_BUDGET, disabled=()):
        self.scenario = scenario
        self.sim = Simulator.load(scenario)
        self.budget = budget
        self.used = 0
        self.log: list[dict] = []
        self.closed = False
        self.final: FakeFinal | None = None
        self.notes = ""
        self.disabled = set(disabled)
        self.counts = {
            "gridlocks": 0,
            "crew_timeouts": 0,
            "misroutes": 0,
            "refusals": 0,
            "invalid_calls": 0,
            "decision_points": 0,
        }

    # -- public API ---------------------------------------------------------------------------------
    def call(self, tool: str, args: dict) -> str:
        if self.closed:
            self.log.append(self._entry(tool, args, "REFUSED", "R-SHIFT-ENDED"))
            return f"REFUSED R-SHIFT-ENDED: shift is over. Final cost {self.final.cost.total}"
        self.used += 1
        out, status, rule = self._dispatch(tool, args)
        self.log.append(self._entry(tool, args, status, rule))
        if self.used >= self.budget and not self.closed:
            self.close()
            out += f"\nBUDGET EXHAUSTED (R-BUDGET): shift ended. Final cost {self.final.cost.total}"
        return out

    def peek(self, tool: str, args: dict) -> str:
        if tool not in READ_ONLY:
            raise ValueError(f"peek is read-only; {tool} is not allowed")
        model, _ = FAKE_REGISTRY[tool]
        parsed = model.model_validate(args)
        return getattr(self, f"_t_{tool}")(parsed)[0]

    def close(self) -> FakeFinal:
        if not self.closed:
            if not self.sim.done:
                self.sim.end_shift()
            self.closed = True
            self.final = FakeFinal(self.sim.tick, self.sim.cost(), dict(self.counts))
        return self.final

    # -- internals ----------------------------------------------------------------------------------
    def _entry(self, tool, args, status, rule):
        return {
            "seq": len(self.log) + 1,
            "tick": self.sim.tick,
            "tool": tool,
            "args": args,
            "status": status,
            "rule": rule,
        }

    def _dispatch(self, tool, args):
        if tool not in FAKE_REGISTRY:
            self.counts["invalid_calls"] += 1
            return f"REFUSED R-BAD-ARGS: unknown tool {tool}", "INVALID", "R-BAD-ARGS"
        if tool in self.disabled:
            self.counts["invalid_calls"] += 1
            return "REFUSED R-BAD-ARGS: tool disabled", "INVALID", "R-BAD-ARGS"
        model, _ = FAKE_REGISTRY[tool]
        try:
            parsed = model.model_validate(args)
        except ValidationError as exc:
            self.counts["invalid_calls"] += 1
            return f"REFUSED R-BAD-ARGS: {exc.error_count()} error(s)", "INVALID", "R-BAD-ARGS"
        return getattr(self, f"_t_{tool}")(parsed)

    def _t_manual(self, _):
        return FAKE_MANUAL, "INFO", None

    def _t_status(self, a):
        state = self.sim.state()
        if a.compact:
            return f't={state.tick} | gridlock: none | notes: "{self.notes}"\nrestrictions: none', "INFO", None
        state.notes = self.notes
        return state.model_dump_json(), "INFO", None

    def _t_bulletins(self, _):
        return "[]", "INFO", None

    def _t_kpis(self, _):
        cost = self.sim.cost().model_dump()
        out = {
            "tick": self.sim.tick,
            "cost": cost,
            "counts": self.counts,
            "budget": {"used": self.used, "left": self.budget - self.used, "total": self.budget},
        }
        return json.dumps(out, sort_keys=True, separators=(",", ":")), "INFO", None

    def _t_notes(self, a):
        if len(a.text) > 1500:
            self.counts["refusals"] += 1
            return "REFUSED R-NOTES-TOO-LONG: max 1500", "REFUSED", "R-NOTES-TOO-LONG"
        self.notes = a.text
        return f"OK notes saved ({len(a.text)} chars)", "OK", None

    def _t_stringline(self, _):
        return "(fake stringline)", "INFO", None

    def _t_authorize(self, a):
        res = self.sim.apply(Authorize(train=a.train, station=a.station, track=a.track))
        if res.ok:
            return res.message, "GRANTED", None
        self.counts["refusals"] += 1
        rule = res.rule.value if hasattr(res.rule, "value") else res.rule
        return res.message, "REFUSED", rule

    def _t_advance_to_next_event(self, _):
        res = self.sim.advance()
        self.counts["decision_points"] += 1
        stops = ", ".join(f"{r.kind} {r.train or ''}".strip() for r in res.reasons)
        out = f"t={res.tick} stop: {stops}\nbudget: {self.used}/{self.budget}"
        if res.done:
            self.close()
        return out, "INFO", None

    def _t_end_shift(self, _):
        self.close()
        return f"SHIFT ENDED at t={self.sim.tick}. Final cost {self.final.cost.total}", "INFO", None


def fake_static(scenario: Scenario):
    return Simulator.load(scenario).static_info()


def fake_decide(static, state):
    """Pure (static, state) -> calls: send every idle train in the network to its destination yard."""
    calls = []
    for t in state.trains:
        if t.phase == "at_station" and t.authority is None and t.finished_tick is None:
            yard = "YE" if t.direction == "east" else "YW"
            calls.append(FakeToolCall("authorize", {"train": t.id, "station": yard, "track": "main"}))
    calls.append(FakeToolCall("advance_to_next_event", {}))
    return calls


def fake_reward_from(c_greedy: float, c_best: float, c_agent: float) -> dict:
    u = (c_greedy - c_agent) / (c_greedy - c_best)
    return {"reward": min(1.0, max(0.0, u)), "score_unclipped": u, "above_best": u > 1.0}
