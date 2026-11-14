from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_serializer, model_validator

from .constants import RULES_VERSION


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Optional fields listed here are left out of dumps while unset, so a scenario that doesn't use them
    # keeps the same JSON (and content hash) as before the field existed.
    _omit_unset: ClassVar[tuple[str, ...]] = ()

    @model_serializer(mode="wrap")
    def _dump(self, handler):
        data = handler(self)
        for name in self._omit_unset:
            if data.get(name) in (None, [], {}):
                data.pop(name, None)
        return data


class Loop(StrictModel):
    id: str
    length_m: Literal[150, 300, 700]
    kind: Literal["platform", "goods"]


class Station(StrictModel):
    id: str
    major: bool
    crew_station: bool
    loops: list[Loop] = Field(max_length=3)

    @model_validator(mode="after")
    def unique_loops(self):
        if len({x.id for x in self.loops}) != len(self.loops):
            raise ValueError("duplicate loop id")
        if any(x.id not in {"L1", "L2", "L3"} for x in self.loops):
            raise ValueError("invalid loop id")
        return self


class Terminal(StrictModel):
    """An end point of the network with a fixed number of tracks (a legacy yard has unlimited room)."""

    id: str
    tracks: list[Loop] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def valid_tracks(self):
        ids = [x.id for x in self.tracks]
        if len(set(ids)) != len(ids) or any(x not in {"P1", "P2", "P3", "P4"} for x in ids):
            raise ValueError("terminal track ids must be distinct and in P1..P4")
        return self


class Section(StrictModel):
    """`west` and `east` are the two ends. On a graph network they carry no compass meaning."""

    id: str
    west: str
    east: str
    length_m: int = Field(gt=0)
    line_speed_kmh: int = Field(gt=0)


class Network(StrictModel):
    """Either a legacy chain (`yards` YW/YE around S1..SN) or a graph (`terminals`, any layout)."""

    _omit_unset: ClassVar[tuple[str, ...]] = ("yards", "terminals")

    yards: dict[str, str] | None = None
    terminals: list[Terminal] = []
    stations: list[Station]
    sections: list[Section]

    @property
    def terminal_ids(self) -> list[str]:
        return [*(self.yards or {}).values(), *[t.id for t in self.terminals]]

    @model_validator(mode="after")
    def valid_layout(self):
        if (self.yards is None) == (not self.terminals):
            raise ValueError("a network has either yards (chain) or terminals (graph)")
        return self._ordered_chain() if self.yards is not None else self._graph()

    def _ordered_chain(self):
        ids = ["YW", *[s.id for s in self.stations], "YE"]
        if self.yards != {"west": "YW", "east": "YE"}:
            raise ValueError("yards must be YW and YE")
        if [s.id for s in self.stations] != [f"S{i}" for i in range(1, len(self.stations) + 1)]:
            raise ValueError("stations must be ordered S1..SN")
        expected = list(zip(ids, ids[1:], strict=False))
        actual = [(s.west, s.east) for s in self.sections]
        if actual != expected or any(s.id != f"{w}-{e}" for s, (w, e) in zip(self.sections, expected, strict=False)):
            raise ValueError("sections must form the ordered contiguous chain")
        return self

    def _graph(self):
        stations, terminals = [s.id for s in self.stations], [t.id for t in self.terminals]
        nodes = set(stations) | set(terminals)
        if len(nodes) != len(stations) + len(terminals):
            raise ValueError("station and terminal ids must be distinct")
        if any("/" in n or "-" in n for n in nodes):
            raise ValueError("node ids must not contain '/' or '-'")
        degree = dict.fromkeys(nodes, 0)
        pairs = set()
        for s in self.sections:
            if s.west not in nodes or s.east not in nodes or s.west == s.east:
                raise ValueError(f"section {s.id} must join two different nodes")
            if s.id != f"{s.west}-{s.east}":
                raise ValueError(f"section id must be '{s.west}-{s.east}'")
            if frozenset((s.west, s.east)) in pairs:
                raise ValueError("at most one section between two nodes")
            pairs.add(frozenset((s.west, s.east)))
            degree[s.west] += 1
            degree[s.east] += 1
        if any(degree[t] != 1 for t in terminals):
            raise ValueError("a terminal has exactly one section")
        if any(not 2 <= degree[s] <= 3 for s in stations):
            raise ValueError("a station has two or three sections")
        seen, todo = set(), [sorted(nodes)[0]]
        while todo:
            node = todo.pop()
            if node in seen:
                continue
            seen.add(node)
            todo.extend(s.east if s.west == node else s.west for s in self.sections if node in (s.west, s.east))
        if seen != nodes:
            raise ValueError("the network must be connected")
        return self


class Stop(StrictModel):
    """`alight`/`board`: passengers getting off and on here; `headway`: scheduled wait for the next service
    here; `platform`: the published track at a hub (a different track costs a platform change)."""

    _omit_unset: ClassVar[tuple[str, ...]] = ("alight", "board", "headway", "platform")

    station: str
    arr: int = Field(ge=0)
    dep: int | None
    min_dwell: int = Field(ge=0, le=2)
    alight: int | None = Field(default=None, ge=0)
    board: int | None = Field(default=None, ge=0)
    headway: int | None = Field(default=None, gt=0)
    platform: str | None = None

    @model_validator(mode="after")
    def valid_times(self):
        if self.dep is not None and self.arr > self.dep:
            raise ValueError("arr must not exceed dep")
        if self.dep is None and self.min_dwell != 0:
            raise ValueError("destination dwell must be 0")
        return self


class Train(StrictModel):
    """A chain train has a `direction`. A graph train has `origin`, `destination` and the timetabled
    `route` (every node from origin to destination); the dispatcher may send it another way."""

    _omit_unset: ClassVar[tuple[str, ...]] = ("direction", "origin", "destination", "route")

    id: str
    type: Literal["express", "local", "freight", "engineering"]
    direction: Literal["east", "west"] | None = None
    origin: str | None = None
    destination: str | None = None
    route: list[str] | None = None
    pax: int = Field(ge=0)
    entry_tick: int = Field(ge=0)
    stops: list[Stop] = Field(min_length=1)
    next_service: str | None
    crew_used_at_entry: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_train(self):
        expected = {"express": "E", "local": "L", "freight": "F", "engineering": "N"}[self.type]
        if not self.id.startswith(expected) or not self.id[len(expected) :].isdigit():
            raise ValueError("train id does not match type")
        if self.type in {"freight", "engineering"} and self.pax != 0:
            raise ValueError("non-passenger train pax must be 0")
        graph_fields = (self.origin, self.destination, self.route)
        if self.direction is not None and any(x is not None for x in graph_fields):
            raise ValueError("a train has either a direction, or origin, destination and route")
        if self.direction is None and any(x is None for x in graph_fields):
            raise ValueError("a train has either a direction, or origin, destination and route")
        if self.route is not None and (
            len(self.route) < 2 or self.route[0] != self.origin or self.route[-1] != self.destination
        ):
            raise ValueError("route must run from origin to destination")
        if self.stops[-1].station != self.end:
            raise ValueError("last stop must be destination yard")
        if self.stops[-1].dep is not None or self.stops[-1].min_dwell != 0:
            raise ValueError("destination stop must have null dep and zero dwell")
        if any(not 1 <= stop.min_dwell <= 2 for stop in self.stops[:-1]):
            raise ValueError("intermediate stop dwell must be 1 or 2")
        if any(s.dep is None for s in self.stops[:-1]):
            raise ValueError("only final stop may have null dep")
        if self.type == "freight" and len(self.stops) != 1:
            raise ValueError("freight has no intermediate stops")
        return self

    def flow(self, index: int) -> tuple[int, int]:
        """(alight, board) at stop `index`. Without explicit counts (legacy scenarios), the train's pax get off
        evenly over its stops and the same number get on at every stop but the last."""
        stop = self.stops[index]
        if stop.alight is not None:
            return stop.alight, stop.board or 0
        n = len(self.stops)
        share = self.pax // n + (1 if index < self.pax % n else 0)
        return share, (share if index < n - 1 else 0)

    @property
    def start(self) -> str:
        return self.origin or ("YW" if self.direction == "east" else "YE")

    @property
    def end(self) -> str:
        return self.destination or ("YE" if self.direction == "east" else "YW")


class EventBase(StrictModel):
    id: str
    reveal_tick: int


class PossessionEvent(EventBase):
    kind: Literal["possession"]
    section: str
    start: int
    end: int


class SpeedRestrictionEvent(EventBase):
    kind: Literal["speed_restriction"]
    sections: list[str]
    speed_kmh: int
    start: int
    end: int


class LateEntryEvent(EventBase):
    kind: Literal["late_entry"]
    train: str
    new_entry_tick: int


class BreakdownEvent(EventBase):
    kind: Literal["breakdown"]
    train: str
    tick: int
    duration: int


class Connection(StrictModel):
    """F11: `pax` passengers change from `feeder` to `connector` at `station`, needing `min_transfer` ticks."""

    feeder: str
    connector: str
    station: str
    pax: int = Field(gt=0)
    min_transfer: int = Field(ge=0)


Event = Annotated[
    PossessionEvent | SpeedRestrictionEvent | LateEntryEvent | BreakdownEvent,
    Field(discriminator="kind"),
]


class Scenario(StrictModel):
    schema_version: Literal[1, 2]
    rules_version: str
    scenario_id: str
    seed: int
    preset: str
    shift_ticks: int = Field(gt=0)
    factors: list[str]
    network: Network
    trains: list[Train]
    events: list[Event]
    connections: list[Connection] = []

    _omit_unset: ClassVar[tuple[str, ...]] = ("connections",)

    @model_validator(mode="after")
    def valid_scenario(self):
        if self.rules_version != RULES_VERSION:
            raise ValueError("rules_version mismatch")
        if self.factors != sorted(set(self.factors)):
            raise ValueError("factors must be unique and sorted")
        supported = {"F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F10", "F11", "F15", "F16"}
        if not set(self.factors) <= supported:
            raise ValueError("unsupported factor")
        if [t.id for t in self.trains] != sorted(t.id for t in self.trains):
            raise ValueError("trains must be sorted by id")
        ids = {t.id for t in self.trains}
        if len(ids) != len(self.trains):
            raise ValueError("duplicate train id")
        chain = self.network.yards is not None
        joined = {frozenset((s.west, s.east)) for s in self.network.sections}
        for t in self.trains:
            if t.next_service and t.next_service not in ids:
                raise ValueError("next_service points to missing train")
            if chain != (t.direction is not None):
                raise ValueError("chain trains have a direction, graph trains an origin, destination and route")
            if chain:
                route = ["YW", *[s.id for s in self.network.stations], "YE"]
                if t.direction == "west":
                    route.reverse()
            else:
                route = t.route
                if t.origin not in self.network.terminal_ids or t.destination not in self.network.terminal_ids:
                    raise ValueError("origin and destination must be terminals")
                if len(set(route)) != len(route) or any(
                    frozenset(pair) not in joined for pair in zip(route, route[1:], strict=False)
                ):
                    raise ValueError("route must follow sections and visit no node twice")
            indices = [route.index(stop.station) if stop.station in route else -1 for stop in t.stops]
            if -1 in indices or indices != sorted(indices) or len(indices) != len(set(indices)):
                raise ValueError("train stops must be in route order")
            if t.next_service:
                successor = next(x for x in self.trains if x.id == t.next_service)
                if successor.start != t.end or (chain and successor.direction == t.direction):
                    raise ValueError("next_service must be opposite direction")
        by_id = {t.id: t for t in self.trains}
        if self.connections and "F11" not in self.factors:
            raise ValueError("connections require F11")
        for c in self.connections:
            feeder, connector = by_id.get(c.feeder), by_id.get(c.connector)
            if feeder is None or connector is None or c.feeder == c.connector:
                raise ValueError("connection trains must exist and differ")
            if c.station not in {s.station for s in feeder.stops} or c.station not in {
                s.station for s in connector.stops[:-1]
            }:
                raise ValueError("a connection's station must be a stop of both trains, not the connector's last")
        event_ids = [event.id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("duplicate event id")
        if [(e.reveal_tick, e.id) for e in self.events] != sorted((e.reveal_tick, e.id) for e in self.events):
            raise ValueError("events must be sorted by reveal_tick and id")
        section_ids = {section.id for section in self.network.sections}
        factor_for = {"possession": "F7", "speed_restriction": "F8", "late_entry": "F16", "breakdown": "F15"}
        for event in self.events:
            if factor_for[event.kind] not in self.factors:
                raise ValueError(f"event {event.id} requires {factor_for[event.kind]}")
            if isinstance(event, PossessionEvent):
                if event.section not in section_ids:
                    raise ValueError("possession section does not exist")
                if not event.reveal_tick <= event.start < event.end <= self.shift_ticks:
                    raise ValueError("invalid possession window")
            elif isinstance(event, SpeedRestrictionEvent):
                if not event.reveal_tick <= event.start < event.end:
                    raise ValueError("invalid restriction window")
                if not 0 < event.speed_kmh < 140:
                    raise ValueError("invalid restriction speed")
                if not event.sections or len(event.sections) != len(set(event.sections)):
                    raise ValueError("restriction sections must be nonempty and distinct")
                if not set(event.sections) <= section_ids:
                    raise ValueError("restriction section does not exist")
            elif isinstance(event, LateEntryEvent):
                train = next((x for x in self.trains if x.id == event.train), None)
                if train is None:
                    raise ValueError("late entry train does not exist")
                if event.new_entry_tick <= train.entry_tick or event.reveal_tick >= train.entry_tick:
                    raise ValueError("invalid late entry timing")
            elif isinstance(event, BreakdownEvent):
                if event.reveal_tick != event.tick or event.duration < 1:
                    raise ValueError("invalid breakdown timing")
        return self


class Authorize(StrictModel):
    kind: Literal["authorize"] = "authorize"
    train: str
    station: str
    track: str
    via: list[str] = []  # stations to pass through, in order, when two routes lead to `station`


class CancelAuthority(StrictModel):
    kind: Literal["cancel_authority"] = "cancel_authority"
    train: str


class Hold(StrictModel):
    kind: Literal["hold"] = "hold"
    train: str
    until_tick: int


class RelieveCrew(StrictModel):
    kind: Literal["relieve_crew"] = "relieve_crew"
    train: str


class Reverse(StrictModel):
    kind: Literal["reverse"] = "reverse"
    train: str


Action = Annotated[Authorize | CancelAuthority | Hold | RelieveCrew | Reverse, Field(discriminator="kind")]
ActionAdapter = TypeAdapter(Action)
