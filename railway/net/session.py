"""One Session per network scenario (Timetable): single entry point for loop, MCP server and replay."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import cached_property

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from railway.util import canonical_json, content_hash

from . import formats
from .engine import LOOKAHEAD_MIN, Engine, Row
from .events import EventLog
from .interlocking import Refusal
from .manual import NOTES_MAX, TOOL_BUDGET, render_manual
from .model import Platform, Timetable
from .rules import NetRule

ON_TIME_MINUTES = 5


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class NoArgs(Args):
    pass


class NetworkArgs(Args):
    area: str | None = None


class StatusArgs(Args):
    area: str | None = None
    compact: bool = False


class TimetableArgs(Args):
    train: str | None = None


class StringlineArgs(Args):
    corridor: str
    from_tick: int
    to_tick: int


class NotesArgs(Args):
    text: str


class OrderArgs(Args):
    resource: str
    trains: list[str]


class HoldArgs(Args):
    train: str
    until: int


class PlatformArgs(Args):
    train: str
    station: str
    n: int | str


# -- Extension-point argument models for Lane B service & crew actions -------------


class TrainArg(Args):
    train: str = Field(description="Train ID (e.g. 'IC1').")


class RouteArgs(Args):
    train: str = Field(description="Train ID to divert.")
    via: list[str] = Field(description="Ordered track or platform/road IDs to route through (e.g. ['JA-B.d', 'B1']).")


class StationActionArgs(Args):
    train: str = Field(description="Train ID.")
    station: str = Field(description="Station node ID (e.g. 'B').")


class SingleLineArgs(Args):
    segment: str = Field(description="Reversible track ID (e.g. 'JA-B.d').")
    on: bool = Field(description="True to open bidirectional working, False to restore normal direction.")


class CostBreakdown(BaseModel):
    """Fallback v0.3 §6.6 cost breakdown shape when `railway.net.cost` (lane C) is not installed."""

    model_config = ConfigDict(extra="forbid")

    pax_lateness: float = 0.0
    missed_calls: float = 0.0
    unfinished: float = 0.0
    freight_lateness: float = 0.0
    crew_timeout: float = 0.0
    gridlock: float = 0.0
    operations: float = 0.0
    total: float = 0.0


class FinalResult(BaseModel):
    tick: int
    cost: CostBreakdown | BaseModel
    counts: dict[str, int]
    kpis: dict[str, float | int | bool] = {}
    state: list[tuple[str, str, str, int]] = []


# Each handler returns (output, status, rule); status is GRANTED | OK | REFUSED | INVALID | INFO.
Handler = Callable[["Session", BaseModel], tuple[str, str, str | None]]
TOOLS: dict[str, tuple[type[Args], Handler]] = {}
PEEKABLE = frozenset({"status", "bulletins", "kpis", "manual"})
INVALID_RULES = frozenset({NetRule.BAD_ARGS, NetRule.UNKNOWN_TRAIN, NetRule.UNKNOWN_ELEMENT})

COUNT_KEYS = (
    "gridlocks",
    "crew_timeouts",
    "reversals",
    "missed_calls",
    "refusals",
    "invalid_calls",
    "decision_points",
    "unfinished",
)


def describe(name: str) -> str:
    """A tool's one-line description, shared by the MCP server and `railway/tools/schemas.py`."""
    doc = (TOOLS[name][1].__doc__ or "").strip()
    return doc.splitlines()[0].strip()


def tool(name: str, args: type[Args] = NoArgs):
    def register(fn: Handler) -> Handler:
        TOOLS[name] = (args, fn)
        return fn

    return register


def _refused(rule: NetRule | str, detail: str) -> tuple[str, str, str]:
    code = rule.value if isinstance(rule, NetRule) else str(rule)
    is_invalid = (rule in INVALID_RULES) if isinstance(rule, NetRule) else (code in {r.value for r in INVALID_RULES})
    status = "INVALID" if is_invalid else "REFUSED"
    return f"REFUSED {code}: {detail}", status, code


def _synthesize_fallback_log(timetable: Timetable, engine: Engine) -> EventLog:
    """Build a canonical EventLog from `engine.trains` when `engine.log` is not attached yet."""
    net = timetable.net
    raw: list[tuple[int, int, str, dict]] = []
    seq = 0

    for trip in sorted(timetable.trips, key=lambda t: t.train.id):
        tid = trip.train.id
        mv = engine.trains[tid]
        starts_berthed = trip.entry_node is None
        visit_offset = 1 if starts_berthed else 0
        call_cursor = 1 if starts_berthed else 0
        matched_calls: set[int] = {0} if (starts_berthed and mv.visits) else set()

        if starts_berthed and mv.visits:
            v0 = mv.visits[0]
            el0 = net.element(v0.platform)
            node0 = el0.node if isinstance(el0, Platform) else None
            seq += 1
            raw.append(
                (v0.arrived, seq, "ENTER", {"train": tid, "element": v0.platform, "call": 0, "booked": trip.entry_tick})
            )
            if v0.departed is not None:
                seq += 1
                raw.append(
                    (
                        v0.departed,
                        seq,
                        "DEPART",
                        {
                            "train": tid,
                            "element": v0.platform,
                            "node": node0,
                            "call": 0,
                            "booked": trip.calls[0].depart,
                        },
                    )
                )

        last_tick = trip.entry_tick
        for v in mv.visits[visit_offset:]:
            el = net.element(v.platform)
            node = el.node if isinstance(el, Platform) else None
            matched_idx: int | None = None
            for idx in range(call_cursor, len(trip.calls)):
                c_el = net.element(trip.calls[idx].platform)
                if isinstance(c_el, Platform) and c_el.node == node:
                    matched_idx = idx
                    call_cursor = idx + 1
                    matched_calls.add(idx)
                    break
            booked_call = trip.calls[matched_idx] if matched_idx is not None else None
            detail = (
                {"booked_platform": booked_call.platform}
                if (booked_call is not None and booked_call.platform != v.platform)
                else {}
            )
            seq += 1
            raw.append(
                (
                    v.arrived,
                    seq,
                    "ARRIVE",
                    {
                        "train": tid,
                        "element": v.platform,
                        "node": node,
                        "call": matched_idx,
                        "booked": booked_call.arrive if booked_call is not None else None,
                        "detail": detail,
                    },
                )
            )
            last_tick = max(last_tick, v.arrived)
            if v.departed is not None:
                seq += 1
                raw.append(
                    (
                        v.departed,
                        seq,
                        "DEPART",
                        {
                            "train": tid,
                            "element": v.platform,
                            "node": node,
                            "call": matched_idx,
                            "booked": booked_call.depart if booked_call is not None else None,
                        },
                    )
                )
                last_tick = max(last_tick, v.departed)

        if mv.phase == "done" and trip.exit_node is not None:
            seq += 1
            exit_tick = min(engine.tick, max(last_tick, engine.tick))
            raw.append((exit_tick, seq, "EXIT", {"train": tid, "node": trip.exit_node}))

        if engine.tick >= timetable.shift_min:
            for c_idx, call in enumerate(trip.calls):
                if c_idx not in matched_calls:
                    c_el = net.element(call.platform)
                    c_node = c_el.node if isinstance(c_el, Platform) else None
                    seq += 1
                    raw.append(
                        (
                            engine.tick,
                            seq,
                            "CALL_MISSED",
                            {
                                "train": tid,
                                "element": call.platform,
                                "node": c_node,
                                "call": c_idx,
                                "booked": call.arrive,
                                "detail": {"why": "shift_end"},
                            },
                        )
                    )
            if mv.phase != "done":
                seq += 1
                raw.append(
                    (
                        engine.tick,
                        seq,
                        "UNFINISHED",
                        {"train": tid, "detail": {"freight": not trip.train.passenger}},
                    )
                )

    if engine.tick >= timetable.shift_min:
        raw.append((engine.tick, 0, "SHIFT_END", {}))

    raw.sort(key=lambda item: (item[0], 0 if item[2] == "SHIFT_END" and item[0] == engine.tick else 1, item[1]))
    # Keep SHIFT_END right before shift-end CALL_MISSED/UNFINISHED while preserving earlier events.
    ordered: list[tuple[int, int, str, dict]] = []
    shift_end_tail: list[tuple[int, int, str, dict]] = []
    for item in sorted(raw, key=lambda x: (x[0], x[1])):
        if (
            engine.tick >= timetable.shift_min
            and item[0] == engine.tick
            and item[2]
            in {
                "SHIFT_END",
                "CALL_MISSED",
                "UNFINISHED",
            }
        ):
            shift_end_tail.append(item)
        else:
            ordered.append(item)
    shift_end_tail.sort(key=lambda x: (0 if x[2] == "SHIFT_END" else 1, x[1]))

    log = EventLog()
    for tick, _, kind, fields in [*ordered, *shift_end_tail]:
        log.emit(tick, kind, **fields)  # type: ignore[arg-type]
    return log


def _try_lane_c_cost(timetable: Timetable, engine: Engine) -> tuple[BaseModel | None, bool]:
    """Compute cost through `railway.net.cost` (lane C) if present; `(None, False)` otherwise."""
    try:
        import importlib

        net_cost = importlib.import_module("railway.net.cost")
    except ImportError:
        return None, False

    log = getattr(engine, "log", None)
    if log is None:
        log = _synthesize_fallback_log(timetable, engine)

    fn = None
    for candidate in ("cost", "compute_cost", "evaluate", "score_events"):
        maybe = getattr(net_cost, candidate, None)
        if callable(maybe) and not isinstance(maybe, type):
            fn = maybe
            break
    if fn is None:
        return None, True

    params = list(inspect.signature(fn).parameters)
    events_arg = log.events if hasattr(log, "events") else log
    first_name = params[0].lower() if params else ""
    if first_name.startswith(("event", "log")):
        first = log if "log" in first_name else events_arg
        try:
            raw = fn(first, timetable)
        except TypeError:
            raw = fn(events_arg, timetable)
    else:
        second_name = params[1].lower() if len(params) > 1 else ""
        second = log if "log" in second_name else events_arg
        try:
            raw = fn(timetable, second)
        except TypeError:
            raw = fn(timetable, log)

    if isinstance(raw, BaseModel):
        return raw, True
    if isinstance(raw, dict):
        return CostBreakdown(**{k: float(raw.get(k, 0.0)) for k in CostBreakdown.model_fields}), True
    if isinstance(raw, int | float):
        return CostBreakdown(total=float(raw)), True
    return None, True


class Session:
    """Drop-in network session compatible with `railway/loop/runner.py`, MCP server and replay."""

    registry = TOOLS

    def __init__(
        self,
        timetable: Timetable,
        *,
        budget: int = TOOL_BUDGET,
        diagnostics: bool = False,
        disabled: tuple[str, ...] = (),
    ):
        self.timetable = timetable
        self.scenario = timetable  # compatibility alias for callers expecting `session.scenario`
        if hasattr(Engine, "from_timetable") and callable(Engine.from_timetable):
            self.engine = Engine.from_timetable(timetable)
        else:
            self.engine = Engine(timetable.net, list(timetable.trips), timetable.shift_min)
        self._applied_actions: list[tuple[int, str, tuple]] = []
        self.budget = budget
        self.diagnostics = diagnostics
        self.disabled = frozenset(disabled)
        self.used = 0
        self.notes = ""
        self.log: list[dict] = []
        self.closed = False
        self.final: FinalResult | None = None
        self.counts: dict[str, int] = dict.fromkeys(COUNT_KEYS, 0)
        self._last_advance = 0
        self._last_decision = None
        self._log_cursor = len(getattr(getattr(self.engine, "log", None), "events", ()))

    @cached_property
    def _initial_engine(self) -> Engine:
        if hasattr(Engine, "from_timetable") and callable(Engine.from_timetable):
            return Engine.from_timetable(self.timetable)
        return Engine(self.timetable.net, list(self.timetable.trips), self.timetable.shift_min)

    # -- public API -----------------------------------------------------------------

    def call(self, tool_name: str, args: dict) -> str:
        tick = self.engine.tick
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
            out += f"\nBUDGET EXHAUSTED ({NetRule.BUDGET.value}): shift ended. Final cost {final.cost.total}"
        return out

    def peek(self, tool_name: str, args: dict) -> str:
        """Read-only CURRENT block inspection: zero budget cost and no log entry."""
        if tool_name not in PEEKABLE:
            raise ValueError(f"peek is read-only; {tool_name!r} is not allowed")
        model, handler = TOOLS[tool_name]
        return handler(self, model.model_validate(args))[0]

    def close(self) -> FinalResult:
        if self.final is None:
            self._finish_shift()
            self._sync_log_counts()
            counts = self._current_counts()
            cost_obj, _ = _try_lane_c_cost(self.timetable, self.engine)
            if cost_obj is None:
                cost_obj = CostBreakdown()
            ops = self._ops_kpis(counts)
            rows: list[Row] = self.engine.state()
            self.final = FinalResult(
                tick=self.engine.tick,
                cost=cost_obj,
                counts=counts,
                kpis=ops,
                state=[tuple(r) for r in rows],
            )
        self.closed = True
        return self.final

    def episode_log(self) -> dict:
        """Close the episode and return the canonical verifiable episode log."""
        final = self.close()
        return {
            "scenario_id": f"{self.timetable.tier}-{self.timetable.seed}",
            "scenario_hash": content_hash(self.timetable.model_dump(mode="json")),
            "rules_version": formats.NET_RULES_VERSION,
            "entries": self.log,
            "closed": True,
            "final_tick": final.tick,
            "final_cost": final.cost.total,
            "final_cost_by_component": final.cost.model_dump(mode="json", exclude={"total"}),
            "final_state": [list(r) for r in final.state],
            "counts": final.counts,
            "kpis": final.kpis,
        }

    def revealed_bulletins(self) -> list[dict]:
        """Revealed disruptions and crew warnings; wired automatically if lane B populates engine bulletins/log."""
        if hasattr(self.engine, "revealed_bulletins") and callable(self.engine.revealed_bulletins):
            return list(self.engine.revealed_bulletins())
        items: list[dict] = []
        seen_disruption_ids: set[str] = set()
        if hasattr(self.engine, "bulletins") and callable(self.engine.bulletins):
            active_ids = set(getattr(self.engine, "_active", ()))
            for d in self.engine.bulletins():
                dumped = d.model_dump(mode="json") if hasattr(d, "model_dump") else dict(d)
                did = str(dumped.get("id", ""))
                if did:
                    seen_disruption_ids.add(did)
                dumped.setdefault("active", did in active_ids)
                items.append(dumped)
        log = getattr(self.engine, "log", None)
        if log is not None and hasattr(log, "events"):
            shown = {"DISRUPTION_START", "DISRUPTION_END", "CREW_WARNING", "CREW_TIMEOUT"}
            for e in log.events:
                if e.kind not in shown or e.tick > self.engine.tick:
                    continue
                if e.kind == "DISRUPTION_START" and e.detail.get("id") in seen_disruption_ids:
                    continue
                items.append(e.model_dump(mode="json"))
        return items

    # -- internals ------------------------------------------------------------------

    def _dispatch(self, tool_name: str, args: dict) -> tuple[str, str, str | None]:
        if tool_name not in TOOLS:
            return _refused(NetRule.BAD_ARGS, f"unknown tool {tool_name!r}")
        if tool_name in self.disabled:
            return _refused(NetRule.BAD_ARGS, "tool disabled")
        model, handler = TOOLS[tool_name]
        if not isinstance(args, dict):
            return _refused(NetRule.BAD_ARGS, f"{tool_name}: arguments must be an object")
        try:
            parsed = model.model_validate(args)
        except ValidationError as exc:
            err = exc.errors()[0]
            where = ".".join(str(x) for x in err["loc"]) or tool_name
            return _refused(NetRule.BAD_ARGS, f"{where}: {err['msg']}")
        return handler(self, parsed)

    def _log(self, tick: int, tool_name: str, args: dict, status: str, rule: str | None) -> None:
        self.log.append(
            {"seq": len(self.log) + 1, "tick": tick, "tool": tool_name, "args": args, "status": status, "rule": rule}
        )

    def _ended(self) -> tuple[str, str, str]:
        final = self.close()
        return (
            f"REFUSED {NetRule.SHIFT_ENDED.value}: shift is over. Final cost {final.cost.total}",
            "REFUSED",
            NetRule.SHIFT_ENDED.value,
        )

    def _engine_action(
        self, method: str, *call_args, granted: str = "OK", message: str = ""
    ) -> tuple[str, str, str | None]:
        """Shared dispatcher for Engine actions (including Lane B's service/crew actions once merged)."""
        fn = getattr(self.engine, method, None)
        if fn is None:
            return _refused(NetRule.BAD_ARGS, f"engine action {method!r} is not available")
        refusal = fn(*call_args)
        if isinstance(refusal, Refusal):
            return _refused(refusal.rule, refusal.detail)
        self._applied_actions.append((self.engine.tick, method, call_args))
        return (message or f"{granted} {method}"), granted, None

    def _snapshot_trains(self) -> dict[str, tuple[str, str, bool, list]]:
        return {tid: (mv.phase, mv.position, mv.waiting, list(mv.visits)) for tid, mv in self.engine.trains.items()}

    def _sync_log_counts(self) -> list[tuple[int, str]]:
        log = getattr(self.engine, "log", None)
        if log is None or not hasattr(log, "events"):
            return []
        new_events = log.events[self._log_cursor :]
        self._log_cursor = len(log.events)
        changes: list[tuple[int, str]] = []
        for ev in new_events:
            if ev.kind == "GRIDLOCK":
                self.counts["gridlocks"] += 1
            elif ev.kind == "CREW_TIMEOUT":
                self.counts["crew_timeouts"] += 1
            elif ev.kind == "CALL_MISSED":
                self.counts["missed_calls"] += 1
            if ev.kind in {"ACTION", "SHIFT_END"}:
                continue
            target = ev.element or ev.node or ""
            who = f"{ev.train} " if ev.train else ""
            changes.append((ev.tick, f"{who}{ev.kind.lower()} {target}".strip()))
        return changes

    def _collect_changes(self, before: dict[str, tuple[str, str, bool, list]], now_tick: int) -> list[tuple[int, str]]:
        log = getattr(self.engine, "log", None)
        if log is not None and hasattr(log, "events"):
            return self._sync_log_counts()

        timed: list[tuple[int, str, str]] = []
        for tid in sorted(self.engine.trains):
            prev_phase, prev_pos, prev_wait, prev_visits = before[tid]
            mv = self.engine.trains[tid]
            if prev_visits and prev_visits[-1].departed is None:
                updated = mv.visits[len(prev_visits) - 1]
                if updated.departed is not None:
                    timed.append((updated.departed, tid, f"{tid} departed {updated.platform}"))
            if prev_phase == "offstage" and mv.phase != "offstage":
                timed.append(
                    (max(self._last_advance, mv.trip.entry_tick), tid, f"{tid} entered from {mv.trip.entry_node}")
                )
            for visit in mv.visits[len(prev_visits) :]:
                timed.append((visit.arrived, tid, f"{tid} arrived {visit.platform}"))
                if visit.departed is not None:
                    timed.append((visit.departed, tid, f"{tid} departed {visit.platform}"))
            if mv.phase == "done" and prev_phase != "done" and mv.trip.exit_node and mv.position == mv.trip.exit_node:
                timed.append((now_tick, tid, f"{tid} exited at {mv.position}"))
            elif mv.phase == "running" and (mv.position != prev_pos or mv.waiting != prev_wait):
                if mv.waiting:
                    timed.append((now_tick, tid, f"{tid} signal stop at {mv.position}"))
                elif mv.position != prev_pos:
                    timed.append((now_tick, tid, f"{tid} on {mv.position}"))
        timed.sort(key=lambda item: (item[0], item[1]))
        return [(t, msg) for t, _, msg in timed]

    def _finish_shift(self) -> None:
        if self.engine.tick >= self.engine.shift_min:
            if hasattr(self.engine, "_finish") and not getattr(self.engine, "ended", True):
                self.engine._finish()
            return
        if hasattr(self.engine, "end_shift") and callable(self.engine.end_shift):
            self.engine.end_shift()
            return
        # Run uncontested traffic to shift_min; contested trains without agent orders remain held at signals/berths.
        eng = self.engine
        has_clock = hasattr(eng, "_clock")
        has_in_place = hasattr(eng, "_in_place")
        has_gridlock = hasattr(eng, "_gridlock")
        has_flush = hasattr(eng, "_flush_stops")
        safety_cap = eng.shift_min * (len(eng.trains) + 2) + 200
        for _ in range(safety_cap):
            if eng.tick >= eng.shift_min:
                break
            if has_clock:
                eng._clock()
            eng._resume()
            if has_in_place:
                eng._in_place()
            soon = eng._wanting(LOOKAHEAD_MIN)
            conflicts = eng._conflicts(soon)
            contested = {t for c in conflicts for t in c.trains}
            now = {
                t: s
                for t, s in soon.items()
                if t not in contested
                and (
                    ((mv := eng.trains[t]).phase == "running" and mv.waiting)
                    or ((due := mv.ready_at()) is not None and due <= eng.tick)
                )
            }
            for tid in eng._priority(now):
                eng._extend(eng.trains[tid], now[tid])
            eng.orders = {
                k: o for k, o in eng.orders.items() if any(eng.trains[t].working for t in o if t in eng.trains)
            }
            if has_gridlock:
                eng._gridlock()
            times = [mv.next_tick for mv in eng.trains.values() if mv.next_tick is not None]
            for mv in eng.trains.values():
                if mv.train.id not in contested and (t := mv.ready_at()) is not None and t > eng.tick:
                    times.append(t)
                for attr in ("stopped_until", "crew_back"):
                    val = getattr(mv, attr, None)
                    if val is not None and val > eng.tick:
                        times.append(val)
                crew_since = getattr(mv, "crew_since", None)
                if crew_since is not None:
                    due = crew_since + 240 - getattr(mv, "crew_base", 0)
                    if due > eng.tick:
                        times.append(due)
            sched = getattr(eng, "_schedule", None)
            next_sched = getattr(eng, "_next_sched", 0)
            if sched is not None and next_sched < len(sched) and sched[next_sched][0] > eng.tick:
                times.append(sched[next_sched][0])
            times = [t for t in times if t <= eng.shift_min]
            if not times:
                if not has_gridlock and (blocked := eng._blocked()):
                    if blocked.kind == "BLOCKED":
                        self.counts["gridlocks"] += 1
                break
            if has_flush:
                eng._flush_stops()
            eng.tick = min(min(times), eng.shift_min)
            if has_clock:
                eng._clock()
            eng._run_transitions()
        if has_flush:
            eng._flush_stops()
        eng.tick = eng.shift_min
        if has_clock:
            eng._clock()
        if hasattr(eng, "_finish"):
            eng._finish()

    def _current_counts(self) -> dict[str, int]:
        counts = dict(self.counts)
        counts["unfinished"] = sum(
            1 for mv in self.engine.trains.values() if mv.phase != "done" or getattr(mv, "cancelled", False)
        )
        if getattr(self.engine, "log", None) is None and self.engine.tick >= self.engine.shift_min:
            fallback_log = _synthesize_fallback_log(self.timetable, self.engine)
            counts["missed_calls"] = len(fallback_log.of("CALL_MISSED"))
        return counts

    def _ops_kpis(self, counts: dict[str, int] | None = None) -> dict[str, float | int | bool]:
        counts = counts if counts is not None else self._current_counts()
        net = self.timetable.net
        stops = on_time = 0
        delay_minutes = 0.0
        freight = delivered = 0
        for trip in self.timetable.trips:
            mv = self.engine.trains[trip.train.id]
            if not trip.train.passenger:
                freight += 1
                delivered += int(mv.phase == "done")
                continue
            visits_by_node: dict[str, list[int]] = {}
            for v in mv.visits:
                el = net.element(v.platform)
                if el is not None:
                    visits_by_node.setdefault(el.node, []).append(v.arrived)
            used_idx: dict[str, int] = {}
            for call in trip.calls:
                if call.arrive is None:
                    continue
                stops += 1
                node = net.element(call.platform).node  # type: ignore[union-attr]
                arrivals = visits_by_node.get(node, ())
                idx = used_idx.get(node, 0)
                if idx < len(arrivals):
                    used_idx[node] = idx + 1
                    late = max(0, arrivals[idx] - call.arrive)
                    if late <= ON_TIME_MINUTES:
                        on_time += 1
                    delay_minutes += call.alight * late
        unfinished = counts.get("unfinished", 0) if self.engine.tick >= self.engine.shift_min else 0
        red_flags = counts["gridlocks"] + counts["crew_timeouts"] + counts["missed_calls"] + unfinished
        return {
            "on_time_pct": round(100 * on_time / stops, 2) if stops else 100.0,
            "passenger_delay_minutes": round(delay_minutes, 2),
            "freight_delivered_pct": round(100 * delivered / freight, 2) if freight else 100.0,
            "red_flags": red_flags,
            "clean_shift": red_flags == 0,
        }


# -- read-only tools ----------------------------------------------------------------


@tool("manual")
def _manual(s: Session, a: NoArgs) -> tuple[str, str, str | None]:
    """The network dispatcher manual: infrastructure, ARS, rules, tools and costs."""
    return render_manual(), "INFO", None


@tool("network", NetworkArgs)
def _network(s: Session, a: NetworkArgs) -> tuple[str, str, str | None]:
    """Static network layout as JSON, optionally scoped to `area` (node, track or corridor)."""
    out = formats.network(s.timetable, a.area)
    if isinstance(out, Refusal):
        return _refused(out.rule, out.detail)
    return out, "INFO", None


@tool("status", StatusArgs)
def _status(s: Session, a: StatusArgs) -> tuple[str, str, str | None]:
    """Current network and train state (JSON or `compact=True` text), optionally scoped to `area`."""
    out = formats.compact_status(s, a.area) if a.compact else formats.full_status(s, a.area)
    if isinstance(out, Refusal):
        return _refused(out.rule, out.detail)
    return out, "INFO", None


@tool("conflicts")
def _conflicts(s: Session, a: NoArgs) -> tuple[str, str, str | None]:
    """Open resource conflicts and at-risk passenger connections, bundled together."""
    return formats.conflicts_text(s), "INFO", None


@tool("timetable", TimetableArgs)
def _timetable(s: Session, a: TimetableArgs) -> tuple[str, str, str | None]:
    """Published timetable and passenger connections as JSON, for all trains or one `train`."""
    trips = list(s.timetable.trips)
    if a.train is not None:
        trips = [t for t in trips if t.train.id == a.train]
        if not trips:
            return _refused(NetRule.UNKNOWN_TRAIN, f"no train {a.train}")
    return formats.timetable_json(s.timetable, trips), "INFO", None


@tool("bulletins")
def _bulletins(s: Session, a: NoArgs) -> tuple[str, str, str | None]:
    """Revealed disruptions and warnings as JSON."""
    return canonical_json(s.revealed_bulletins()), "INFO", None


@tool("kpis")
def _kpis(s: Session, a: NoArgs) -> tuple[str, str, str | None]:
    """Cost (when `railway.net.cost` is present), counters, operational KPIs and tool budget as JSON."""
    counts = s._current_counts()
    cost_obj, has_cost = _try_lane_c_cost(s.timetable, s.engine)
    budget = {"used": s.used, "left": s.budget - s.used, "total": s.budget}
    payload = {
        "tick": s.engine.tick,
        "cost": cost_obj.model_dump(mode="json") if (has_cost and cost_obj is not None) else None,
        "counts": counts,
        "kpis": s._ops_kpis(counts),
        "budget": budget,
    }
    return canonical_json(payload), "INFO", None


@tool("stringline", StringlineArgs)
def _stringline(s: Session, a: StringlineArgs) -> tuple[str, str, str | None]:
    """ASCII time-distance diagram along `corridor` for `[from_tick, to_tick]`."""
    return formats.render_stringline(s, a.corridor, a.from_tick, a.to_tick)


@tool("notes", NotesArgs)
def _notes(s: Session, a: NotesArgs) -> tuple[str, str, str | None]:
    """Replace your notes scratchpad (up to 1500 characters)."""
    if len(a.text) > NOTES_MAX:
        return _refused(NetRule.NOTES_TOO_LONG, f"{len(a.text)} characters, max {NOTES_MAX}")
    s.notes = a.text
    return f"OK notes saved ({len(a.text)} characters)", "OK", None


# -- actions ------------------------------------------------------------------------


@tool("order", OrderArgs)
def _order(s: Session, a: OrderArgs) -> tuple[str, str, str | None]:
    """Set train priority order across a contested `resource` (key, track or node)."""
    if s.engine.tick >= s.engine.shift_min:
        return _refused(NetRule.SHIFT_ENDED, "the shift is over")
    if not a.trains:
        return _refused(NetRule.BAD_ARGS, "trains must not be empty")
    if len(set(a.trains)) != len(a.trains):
        return _refused(NetRule.BAD_ARGS, "duplicate train in order")
    keys = formats.resolve_order_keys(s.timetable.net, a.resource)
    if isinstance(keys, Refusal):
        return _refused(keys.rule, keys.detail)
    for key in keys:
        refusal = s.engine.order(key, a.trains)
        if isinstance(refusal, Refusal):
            return _refused(refusal.rule, refusal.detail)
        s._applied_actions.append((s.engine.tick, "order", (key, list(a.trains))))
    return f"GRANTED order on {a.resource}: {', '.join(a.trains)}", "GRANTED", None


@tool("hold", HoldArgs)
def _hold(s: Session, a: HoldArgs) -> tuple[str, str, str | None]:
    """Hold an offstage or berthed `train` until tick `until`."""
    return s._engine_action("hold", a.train, a.until, granted="OK", message=f"OK {a.train} held until t={a.until}")


@tool("platform", PlatformArgs)
def _platform(s: Session, a: PlatformArgs) -> tuple[str, str, str | None]:
    """Re-platform `train`'s next booked call to platform `n` at `station`."""
    if s.engine.tick >= s.engine.shift_min:
        return _refused(NetRule.SHIFT_ENDED, "the shift is over")
    if a.train not in s.engine.trains:
        return _refused(NetRule.UNKNOWN_TRAIN, f"no train {a.train}")
    plat_id = formats.resolve_platform(s.timetable.net, a.station, a.n)
    if isinstance(plat_id, Refusal):
        return _refused(plat_id.rule, plat_id.detail)
    return s._engine_action(
        "platform", a.train, plat_id, granted="OK", message=f"OK {a.train} re-platformed to {plat_id}"
    )


# -- Lane B service & crew actions --------------------------------------------------


def _resolve_station_arg(s: Session, station: str) -> str:
    """Normalize a platform ID (e.g. 'B1') to its station node ID ('B') when `Engine` expects a node ID."""
    if s.timetable.net.node(station) is None:
        el = s.timetable.net.element(station)
        if isinstance(el, Platform):
            return el.node
    return station


LANE_B_TOOL_SPECS: dict[str, tuple[type[Args], str, str]] = {
    "route": (RouteArgs, "GRANTED", "Divert `train`'s next leg via ordered track or platform IDs."),
    "skip": (StationActionArgs, "OK", "Skip `train`'s next scheduled call at `station`."),
    "short_turn": (
        StationActionArgs,
        "OK",
        "Turn `train` back early at `station` and start its return working there.",
    ),
    "cancel": (TrainArg, "OK", "Cancel `train` for the remainder of the shift."),
    "single_line": (SingleLineArgs, "OK", "Enable or disable bidirectional single-line working on `segment`."),
    "relieve_crew": (
        TrainArg,
        "OK",
        "Relieve `train`'s crew at its current crew-relief platform, or at the next one it calls at.",
    ),
}


@tool("route", RouteArgs)
def _route(s: Session, a: RouteArgs) -> tuple[str, str, str | None]:
    """Divert `train`'s next leg via ordered track or platform IDs (e.g. `["JA-B.d", "B1"]`)."""
    via = list(a.via)
    return s._engine_action(
        "route",
        a.train,
        via,
        granted="GRANTED",
        message=f"GRANTED route for {a.train} via {' -> '.join(via)}",
    )


@tool("skip", StationActionArgs)
def _skip(s: Session, a: StationActionArgs) -> tuple[str, str, str | None]:
    """Skip `train`'s next scheduled call at `station` (node ID, e.g. `"B"`)."""
    station = _resolve_station_arg(s, a.station)
    return s._engine_action("skip", a.train, station, granted="OK", message=f"OK {a.train} skipping {station}")


@tool("short_turn", StationActionArgs)
def _short_turn(s: Session, a: StationActionArgs) -> tuple[str, str, str | None]:
    """Turn `train` back early at `station` (node ID) and start its return working there."""
    station = _resolve_station_arg(s, a.station)
    return s._engine_action(
        "short_turn",
        a.train,
        station,
        granted="OK",
        message=f"OK {a.train} short-turned at {station}",
    )


@tool("cancel", TrainArg)
def _cancel(s: Session, a: TrainArg) -> tuple[str, str, str | None]:
    """Cancel `train` for the remainder of the shift."""
    return s._engine_action("cancel", a.train, granted="OK", message=f"OK {a.train} cancelled")


@tool("single_line", SingleLineArgs)
def _single_line(s: Session, a: SingleLineArgs) -> tuple[str, str, str | None]:
    """Enable or disable bidirectional single-line working on reversible track `segment` (e.g. `"JA-B.d"`)."""
    state_word = "on" if a.on else "off"
    return s._engine_action(
        "single_line",
        a.segment,
        a.on,
        granted="OK",
        message=f"OK single-line working {state_word} on {a.segment}",
    )


@tool("relieve_crew", TrainArg)
def _relieve_crew(s: Session, a: TrainArg) -> tuple[str, str, str | None]:
    """Relieve `train`'s crew at its current crew-relief platform, or at the next one it calls at."""
    return s._engine_action("relieve_crew", a.train, granted="OK", message=f"OK crew relief set for {a.train}")


@tool("advance_to_next_event")
def _advance(s: Session, a: NoArgs) -> tuple[str, str, str | None]:
    """Advance time to the next decision point and report changes since the previous advance."""
    since = s._last_advance
    before = s._snapshot_trains()
    decision = s.engine.advance()
    s._last_decision = decision
    s.counts["decision_points"] += 1
    if decision.kind == "BLOCKED" and getattr(s.engine, "log", None) is None:
        s.counts["gridlocks"] += 1
    changes = s._collect_changes(before, decision.tick)
    s._last_advance = decision.tick
    final_str = None
    if decision.kind == "SHIFT_END":
        done = s.close()
        _, has_cost = _try_lane_c_cost(s.timetable, s.engine)
        final_str = formats.final_line(done.tick, done.cost, has_cost_module=has_cost)  # type: ignore[arg-type]
    return formats.advance_text(s, decision, since, changes, final_str), "INFO", None


@tool("end_shift")
def _end_shift(s: Session, a: NoArgs) -> tuple[str, str, str | None]:
    """Run the uncontested remainder of the shift and report the final cost."""
    final = s.close()
    _, has_cost = _try_lane_c_cost(s.timetable, s.engine)
    return formats.final_line(final.tick, final.cost, has_cost_module=has_cost), "INFO", None  # type: ignore[arg-type]
