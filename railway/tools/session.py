"""The one code path for every tool call (T05 §2.1): MCP server, in-process client and replay all use it."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, ValidationError

from railway.sim.constants import NOTES_MAX, TOOL_BUDGET
from railway.sim.cost import CostBreakdown
from railway.sim.engine import Simulator
from railway.sim.manual import render_manual
from railway.sim.model import Authorize, CancelAuthority, Hold, RelieveCrew, Scenario
from railway.sim.observe import Change
from railway.sim.rules import RuleId
from railway.tools import formats
from railway.util import canonical_json, content_hash


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class NoArgs(Args):
    pass


class TrainArg(Args):
    train: str


class AuthorizeArgs(Args):
    train: str
    station: str
    track: str


class HoldArgs(Args):
    train: str
    until_tick: int


class StatusArgs(Args):
    compact: bool = False


class TimetableArgs(Args):
    train: str | None = None


class StringlineArgs(Args):
    from_tick: int
    to_tick: int


class NotesArgs(Args):
    text: str


class FinalResult(BaseModel):
    tick: int
    cost: CostBreakdown
    counts: dict[str, int]


# Each handler returns (output, status, rule); status is GRANTED | OK | REFUSED | INVALID | INFO.
Handler = Callable[["Session", BaseModel], tuple[str, str, str | None]]
TOOLS: dict[str, tuple[type[Args], Handler]] = {}


def describe(name: str) -> str:
    """A tool's one-line description, shared by the MCP server and the loop's tool schemas (T08 §2.2)."""
    return TOOLS[name][1].__doc__


PEEKABLE = frozenset({"status", "bulletins", "kpis", "manual"})


def tool(name: str, args: type[Args] = NoArgs):
    def register(fn: Handler) -> Handler:
        TOOLS[name] = (args, fn)
        return fn

    return register


INVALID_RULES = {RuleId.BAD_ARGS, RuleId.UNKNOWN_TRAIN, RuleId.UNKNOWN_STATION, RuleId.UNKNOWN_TRACK}


def _refused(rule: RuleId, detail: str) -> tuple[str, str, str]:
    status = "INVALID" if rule in INVALID_RULES else "REFUSED"
    return f"REFUSED {rule.value}: {detail}", status, rule.value


class Session:
    def __init__(
        self,
        scenario: Scenario,
        *,
        budget: int = TOOL_BUDGET,
        diagnostics: bool = False,
        disabled: tuple[str, ...] = (),
    ):
        self.scenario = scenario
        self.sim = Simulator.load(scenario)
        self.budget = budget
        self.diagnostics = diagnostics  # conflicts / what_if arrive in T09
        self.disabled = frozenset(disabled)  # ablations (T08): refused like an unknown tool, and billed
        self.used = 0
        self.notes = ""
        self.log: list[dict] = []
        self.closed = False
        self.final: FinalResult | None = None
        self.counts = dict.fromkeys(
            ("gridlocks", "crew_timeouts", "refusals", "invalid_calls", "decision_points"), 0
        )
        # Movement changes, for stringline. Trains already in the network at load never emit ENTRY, so seed them.
        self.history: list = [
            Change(tick=0, kind="ENTRY", train=t.id, text=f"{t.id} enters at {t.station}")
            for t in self.sim.state().trains
            if t.phase == "at_station"
        ]
        self.breakdowns_seen: set[tuple[str, int]] = set()  # (train, tick) of breakdowns that happened
        self._last_advance = 0

    # -- the only entry point -------------------------------------------------
    def call(self, tool_name: str, args: dict) -> str:
        tick = self.sim.tick
        if self.closed:
            out, status, rule = self._ended()
            self._log(tick, tool_name, args, status, rule)
            return out
        self.used += 1
        out, status, rule = self._dispatch(tool_name, args)
        if status == "REFUSED":
            self.counts["refusals"] += 1
        elif status == "INVALID":
            self.counts["invalid_calls"] += 1
        self._log(tick, tool_name, args, status, rule)
        if self.used >= self.budget and not self.closed:
            final = self.close()
            out += f"\nBUDGET EXHAUSTED (R-BUDGET): shift ended. Final cost {final.cost.total}"
        return out

    def peek(self, tool_name: str, args: dict) -> str:
        """Read-only output for our own loop's CURRENT block (T08 §2.3): no budget, no log entry."""
        if tool_name not in PEEKABLE:
            raise ValueError(f"peek is read-only; {tool_name!r} is not allowed")
        model, handler = TOOLS[tool_name]
        return handler(self, model.model_validate(args))[0]

    def close(self) -> FinalResult:
        if self.final is None:
            if not self.sim.done:
                self._record(self.sim.end_shift())
            self.final = FinalResult(tick=self.sim.tick, cost=self.sim.cost(), counts=dict(self.counts))
        self.closed = True
        return self.final

    # -- internals ---------------------------------------------------------------
    def _dispatch(self, tool_name: str, args: dict) -> tuple[str, str, str | None]:
        if tool_name not in TOOLS:
            return _refused(RuleId.BAD_ARGS, f"unknown tool {tool_name!r}")
        if tool_name in self.disabled:
            return _refused(RuleId.BAD_ARGS, "tool disabled")
        model, handler = TOOLS[tool_name]
        try:
            parsed = model.model_validate(args if isinstance(args, dict) else {})
        except ValidationError as e:
            err = e.errors()[0]
            where = ".".join(str(x) for x in err["loc"]) or tool_name
            return _refused(RuleId.BAD_ARGS, f"{where}: {err['msg']}")
        return handler(self, parsed)

    def _log(self, tick, tool_name, args, status, rule):
        self.log.append(
            {"seq": len(self.log) + 1, "tick": tick, "tool": tool_name, "args": args, "status": status, "rule": rule}
        )

    def _ended(self) -> tuple[str, str, str]:
        final = self.close()
        return (
            f"REFUSED {RuleId.SHIFT_ENDED.value}: shift is over. Final cost {final.cost.total}",
            "REFUSED",
            (RuleId.SHIFT_ENDED.value),
        )

    def _record(self, result):
        self.history.extend(result.changes)
        for r in result.reasons:
            if r.kind == "GRIDLOCK":
                self.counts["gridlocks"] += 1
            elif r.kind == "CREW_TIMEOUT":
                self.counts["crew_timeouts"] += 1
            elif r.kind == "BREAKDOWN":
                self.breakdowns_seen.add((r.train, result.tick))
        return result

    def _action(self, action, granted: str = "OK"):
        result = self.sim.apply(action)
        if result.ok:
            return result.message, granted, None
        return _refused(result.rule, result.message.split(": ", 1)[-1])

    def episode_log(self) -> dict:
        """PRD §7.9. Closes the episode first, so every later call is refused with R-SHIFT-ENDED."""
        final = self.close()
        return {
            "scenario_id": self.scenario.scenario_id,
            "scenario_hash": content_hash(self.scenario.model_dump(mode="json")),
            "rules_version": self.scenario.rules_version,
            "entries": self.log,
            "closed": True,
            "final_tick": final.tick,
            "final_cost": final.cost.total,
            "final_cost_by_component": final.cost.model_dump(mode="json", exclude={"total"}),
        }

    def state(self):
        return self.sim.state().model_copy(update={"notes": self.notes})

    def revealed_events(self) -> list[dict]:
        out = []
        for event in self.scenario.events:
            if event.id not in self.sim._revealed_events:
                continue
            if event.kind == "breakdown" and (event.train, event.tick) not in self.breakdowns_seen:
                continue  # skipped (train not in the network): never shown
            out.append(event.model_dump(mode="json"))
        return sorted(out, key=lambda e: (e["reveal_tick"], e["id"]))


# -- read-only tools ---------------------------------------------------------------
@tool("manual")
def _manual(s: Session, a):
    """The dispatcher manual: rules, costs, tools. See manual()."""
    return render_manual(), "INFO", None


@tool("network")
def _network(s: Session, a):
    """Static network as JSON. See manual()."""
    return formats.network(s.scenario), "INFO", None


@tool("timetable", TimetableArgs)
def _timetable(s: Session, a: TimetableArgs):
    """Timetable as JSON, for all trains or one. See manual()."""
    trains = s.scenario.trains
    if a.train is not None:
        trains = [t for t in trains if t.id == a.train]
        if not trains:
            return _refused(RuleId.UNKNOWN_TRAIN, f"no train {a.train}")
    return formats.timetable(s.scenario, trains), "INFO", None


@tool("status", StatusArgs)
def _status(s: Session, a: StatusArgs):
    """Current state, as JSON or as compact text. See manual()."""
    state = s.state()
    return (formats.compact_status(state, s.scenario) if a.compact else formats.full_status(state)), "INFO", None


@tool("bulletins")
def _bulletins(s: Session, a):
    """Revealed events as JSON. See manual()."""
    return canonical_json(s.revealed_events()), "INFO", None


@tool("kpis")
def _kpis(s: Session, a):
    """Cost so far, counters and budget as JSON. See manual()."""
    cost = s.sim.cost().model_dump(mode="json")
    budget = {"used": s.used, "left": s.budget - s.used, "total": s.budget}
    return canonical_json({"tick": s.sim.tick, "cost": cost, "counts": s.counts, "budget": budget}), "INFO", None


@tool("stringline", StringlineArgs)
def _stringline(s: Session, a: StringlineArgs):
    """ASCII time-distance diagram for a tick window. See manual()."""
    from railway.tools.stringline import render

    return render(s, a.from_tick, a.to_tick)


@tool("notes", NotesArgs)
def _notes(s: Session, a: NotesArgs):
    """Replace your notes scratchpad. See manual()."""
    if len(a.text) > NOTES_MAX:
        return _refused(RuleId.NOTES_TOO_LONG, f"{len(a.text)} characters, max {NOTES_MAX}")
    s.notes = a.text
    return f"OK notes saved ({len(a.text)} characters)", "OK", None


# -- actions -------------------------------------------------------------------------
@tool("authorize", AuthorizeArgs)
def _authorize(s: Session, a: AuthorizeArgs):
    """Grant a train movement authority to a station track. See manual()."""
    return s._action(Authorize(train=a.train, station=a.station, track=a.track), granted="GRANTED")


@tool("cancel_authority", TrainArg)
def _cancel(s: Session, a: TrainArg):
    """Cancel or truncate a train's authority. See manual()."""
    return s._action(CancelAuthority(train=a.train))


@tool("hold", HoldArgs)
def _hold(s: Session, a: HoldArgs):
    """Hold a train at its station until a tick. See manual()."""
    return s._action(Hold(train=a.train, until_tick=a.until_tick))


@tool("relieve_crew", TrainArg)
def _relieve(s: Session, a: TrainArg):
    """Start a crew relief at a crew station. See manual()."""
    return s._action(RelieveCrew(train=a.train))


@tool("advance_to_next_event")
def _advance(s: Session, a):
    """Advance time to the next decision point. See manual()."""
    since = s._last_advance
    result = s._record(s.sim.advance())
    s.counts["decision_points"] += 1
    s._last_advance = result.tick
    final = None
    if result.done:
        done = s.close()
        final = formats.final_line(done.tick, done.cost)
    return formats.advance(result, since, s.used, s.budget, final), "INFO", None


@tool("end_shift")
def _end_shift(s: Session, a):
    """Run to the end of the shift and report the final cost. See manual()."""
    final = s.close()
    return formats.final_line(final.tick, final.cost), "INFO", None
