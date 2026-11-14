"""The engine's event log: the one record of what happened in a shift (v0.3 network, lanes B and C).

The engine (lane B) appends an `Event` for everything below as it happens; nothing else is written. The cost
(lane C, DESIGN.md §6.6) is a pure function of the event log plus the timetable, so it can be recomputed from a
replay, audited event by event, and never drifts from what the engine did. Tools show slices of it.

Rules for emitters:
- `tick` is the minute the thing happened. Events of one tick keep the order they happened in; ties between
  trains follow train id, as the engine already resolves them.
- `call` is the index into the trip's `calls` (the booked timetable, never the edited plan), or None if the
  event is not about a booked call. `booked` is the booked time the event is measured against, if any.
- One fact, one event. A train that arrives at a platform it calls at emits ARRIVE with `call` set, not ARRIVE
  plus something else. Derived charges (lateness, a missed call's passengers) are the cost's job, not the log's.
- Every shift ends with exactly one SHIFT_END; after it, CALL_MISSED for every booked call not made and
  UNFINISHED for every train whose work is not done are emitted at the shift-end tick, in train-id order.

Kinds and their fields (`train` is set for every kind except SHIFT_END, GRIDLOCK and DISRUPTION_*):

| kind | when | element | node | call | booked | detail |
|---|---|---|---|---|---|---|
| ENTER | enters through a portal, or starts standing in its first platform | track or platform | portal node or None | 0 if it starts at call 0 | entry tick | |
| ARRIVE | stops in a platform | platform | its node | booked call or None (unbooked stop) | booked arrive | {"booked_platform": id} if the call was booked at another platform of that node |
| DEPART | leaves a platform | platform | its node | booked call or None | booked depart | |
| PASS | runs through a platform without stopping | platform | its node | booked call if it should have stopped there | booked arrive | |
| SIGNAL_STOP | stops at a signal with no authority ahead (an unplanned stop: restart penalty) | element it stands on | | | | {"restart_min": n} |
| EXIT | runs out of the area through a portal | track | portal node | | booked exit time, if the trip has one | |
| CALL_MISSED | a booked call can no longer be made (skip, cancel, short turn, route, or not reached by shift end) | booked platform | its node | call | booked arrive | {"why": "skip"/"cancel"/"short_turn"/"route"/"shift_end"} |
| CONNECTION | the connector leaves a connection node (made or not) | | node | | | {"feeder": id, "connector": id, "transfers": n, "made": bool} |
| UNFINISHED | at shift end, its work is not done | | | | | {"freight": bool} |
| CREW_WARNING | crew within the warning margin of its duty limit | | | | | {"remaining_min": n} |
| CREW_TIMEOUT | crew reaches its duty limit; the train stops where it is | element | | | | {"blocked_min": n} |
| CREW_RELIEF | a relief starts | platform | node | | | {"until": tick} |
| DISRUPTION_START / DISRUPTION_END | a breakdown, possession, speed restriction or signal failure begins or ends | element, if one | node, if one | | | {"id": id, "kind": kind, "train": id or None} |
| GRIDLOCK | a new set of trains can never move again | | | | | {"trains": [ids]} |
| ACTION | the agent changed the plan (order, hold, platform, route, skip, short_turn, cancel, single_line, ...) | | | | | {"action": name, "args": {...}} |
| SHIFT_END | the shift is over | | | | | |
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EventKind = Literal[
    "ENTER",
    "ARRIVE",
    "DEPART",
    "PASS",
    "SIGNAL_STOP",
    "EXIT",
    "CALL_MISSED",
    "CONNECTION",
    "UNFINISHED",
    "CREW_WARNING",
    "CREW_TIMEOUT",
    "CREW_RELIEF",
    "DISRUPTION_START",
    "DISRUPTION_END",
    "GRIDLOCK",
    "ACTION",
    "SHIFT_END",
]
KINDS: tuple[str, ...] = EventKind.__args__  # type: ignore[attr-defined]


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tick: int = Field(ge=0)
    kind: EventKind
    train: str | None = None
    element: str | None = None
    node: str | None = None
    call: int | None = Field(default=None, ge=0)
    booked: int | None = None
    detail: dict = Field(default_factory=dict)


class EventLog:
    """Append-only. `events` is the whole shift so far, in the order things happened."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, tick: int, kind: EventKind, **fields) -> Event:
        event = Event(tick=tick, kind=kind, **fields)
        if self.events and tick < self.events[-1].tick:
            raise ValueError(f"event at t={tick} after one at t={self.events[-1].tick}: the log must be in time order")
        self.events.append(event)
        return event

    def of(self, *kinds: EventKind) -> list[Event]:
        return [e for e in self.events if e.kind in kinds]

    def since(self, tick: int) -> list[Event]:
        return [e for e in self.events if e.tick >= tick]
