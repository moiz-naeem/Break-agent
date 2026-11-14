from __future__ import annotations

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
    direction: str
    pax: int
    phase: str
    station: str | None = None
    track: str | None = None
    section: str | None = None
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


class State(LooseModel):
    tick: int
    shift_ticks: int
    done: bool
    gridlock: list[str]
    trains: list[TrainState]
    restrictions: list[dict] = []
    possessions: list[dict] = []
    cost: CostBreakdown
    notes: str = ""


class StaticInfo(LooseModel):
    network: dict
    trains: list[dict]
    shift_ticks: int
    factors: list[str]
    rules_version: str


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
