from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict

from .cost import CostBreakdown
from .rules import RuleId


class LooseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Authority(LooseModel):
    dest_station: str
    dest_track: str
    path: list[str]


class Arrival(LooseModel):
    station: str
    sched: int
    actual: int


class StopView(LooseModel):
    station: str
    arr: int
    dep: int | None


class TrainState(LooseModel):
    id: str
    type: str
    direction: str | None = None
    origin: str
    destination: str
    pax: int
    phase: str
    station: str | None = None
    track: str | None = None
    section: str | None = None
    section_from: str | None = None
    came_by: str | None = None
    section_enter: int | None = None
    section_exit: int | None = None
    authority: Authority | None = None
    entry_tick: int
    ready_tick: int | None = None
    hold_until: int | None = None
    dwell_until: int | None = None
    next_stop: StopView | None = None
    arrivals: list[Arrival] = []
    delay: int = 0
    finished_tick: int | None = None
    outcome: str | None = None  # "arrived" or "wrong_terminal" once finished
    off_route: bool = False  # a grant made the destination unreachable
    restart_pending: bool = False
    crew_used: int | None = None
    crew_remaining: int | None = None
    relief_until: int | None = None
    breakdown_until: int | None = None
    blocked_until: int | None = None
    waits_for: str | None = None


class TerminalHold(LooseModel):
    """A terminal track still held by a train that finished there (turnaround, or waiting for its next
    service)."""

    resource: str
    train: str
    until: int | None = None


class State(LooseModel):
    tick: int
    shift_ticks: int
    done: bool
    gridlock: list[str]
    gridlock_charged: list[str] = []
    trains: list[TrainState]
    restrictions: list[dict] = []
    possessions: list[dict] = []
    terminal_holds: list[TerminalHold] = []
    counters: dict[str, int] = {}
    cost: CostBreakdown
    notes: str = ""
    last_decision_tick: int = 0


class StaticInfo(LooseModel):
    network: dict
    trains: list[dict]
    shift_ticks: int
    factors: list[str]
    rules_version: str
    connections: list[dict] = []

    @classmethod
    def from_tool_json(cls, network_json: str, timetable_json: str) -> StaticInfo:
        """Rebuild exactly what `Simulator.static_info()` returns from the `network` + `timetable` tool outputs."""
        net, table = json.loads(network_json), json.loads(timetable_json)
        trains = [
            {k: v for k, v in t.items() if k not in {"length_m", "max_kmh"}}
            | {"crew_used_at_entry": t.get("crew_used_at_entry", 0)}  # only shown when F10 is on
            for t in table["trains"]
        ]
        return cls(
            network={k: net[k] for k in ("yards", "terminals", "stations", "sections") if k in net},
            trains=trains,
            shift_ticks=net["shift_ticks"],
            factors=net["factors"],
            rules_version=net["rules_version"],
            connections=table.get("connections", []),
        )


class ActionResult(LooseModel):
    ok: bool
    rule: RuleId | None = None
    message: str


class Reason(LooseModel):
    kind: str
    train: str | None = None
    detail: str = ""


class Change(LooseModel):
    tick: int
    kind: str
    train: str | None = None
    text: str


class AdvanceResult(LooseModel):
    tick: int
    reasons: list[Reason]
    changes: list[Change]
    done: bool
