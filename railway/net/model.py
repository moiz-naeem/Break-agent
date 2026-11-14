"""v0.3 network infrastructure: a graph of nodes joined by tracks, with moves inside nodes (docs/network-v0.3.md §2)."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

Travel = Literal["ab", "ba"]
Side = Literal["a", "b"]


class NetModel(BaseModel):
    """Frozen, so a validated network cannot be changed into an invalid one; the generator builds new ones."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Node(NetModel):
    id: str
    kind: Literal["station", "junction", "terminal", "yard", "freight_terminal", "portal"]


class Track(NetModel):
    """One physical track between two nodes. Double track is two Tracks, one per direction."""

    id: str
    a: str
    b: str
    length_m: int = Field(gt=0)
    line_speed_kmh: int = Field(gt=0)
    electrified: bool
    direction: Travel | Literal["both"]
    block_m: int = Field(default=2000, ge=500)
    # Signalled for running the wrong way too: single_line() can open it to both directions during a blockage.
    reversible: bool = False

    @model_validator(mode="after")
    def valid_track(self):
        if self.reversible and self.direction == "both":
            raise ValueError(f"track {self.id}: only a one-way track can be reversible; it already runs both ways")
        return self

    @property
    def blocks(self) -> int:
        return -(-self.length_m // self.block_m)

    def other(self, node: str) -> str:
        return self.b if node == self.a else self.a

    def travel_from(self, node: str) -> Travel:
        return "ab" if node == self.a else "ba"


class Platform(NetModel):
    """Where a train stands inside a node: a platform face, a through road or a siding (loop, goods, yard road)."""

    id: str
    node: str
    length_m: int = Field(gt=0)
    kind: Literal["platform", "through", "siding"]
    electrified: bool = True
    crew_relief: bool = False  # a crew can be relieved here (F10)


class Move(NetModel):
    """A path over points inside a node linking two of its tracks or platforms, usable either way.

    `side` says which end of the platform the move serves, when one end is a platform. Moves that share a track end
    or a platform side share points and conflict; crossing paths are listed in Network.crossings.
    """

    id: str
    node: str
    ends: tuple[str, str]
    speed_kmh: int = Field(gt=0)
    side: Side | None = None

    def other(self, end: str) -> str:
        return self.ends[1] if end == self.ends[0] else self.ends[0]


class TrainSpec(NetModel):
    """A train as the interlocking and the engine see it. The generator (step 3) fills these from §3's type table."""

    id: str
    length_m: int = Field(gt=0)
    electric: bool
    passenger: bool
    max_speed_kmh: int = Field(default=120, gt=0)
    restart_min: int = Field(default=1, ge=0)  # extra minutes to get going after an unplanned stop


class Call(NetModel):
    """A scheduled stop. `arrive` is None at the origin, `depart` at the destination.

    `alight` and `board` are the passengers this call serves; the reward charges both when the call is missed
    (DESIGN.md §6.6 C2). A dwell of TURNAROUND_MIN or more is a turnaround: the train may leave the way it came.
    """

    platform: str
    arrive: int | None = None
    depart: int | None = None
    min_dwell: int = Field(default=1, ge=0)
    alight: int = Field(default=0, ge=0)
    board: int = Field(default=0, ge=0)


class Trip(NetModel):
    """One train's work for the shift: where it starts, the calls it must make, and where it leaves the area."""

    train: TrainSpec
    calls: tuple[Call, ...] = Field(min_length=1)
    entry_node: str | None = None  # portal it enters from; None means it starts standing at calls[0]
    entry_tick: int = Field(default=0, ge=0)
    exit_node: str | None = None  # portal it runs out to after its last call; None means it stays in the platform
    crew_min: int = Field(default=0, ge=0)  # minutes its crew has already worked when the trip starts (F10)

    @model_validator(mode="after")
    def valid_trip(self):
        """A call the train leaves needs a departure; a call it arrives at needs an arrival. One call can be both."""
        leaves = len(self.calls) - (0 if self.exit_node else 1)
        arrives = 0 if self.entry_node else 1
        for i, call in enumerate(self.calls):
            if i < leaves and call.depart is None:
                raise ValueError(f"trip {self.train.id}: it leaves {call.platform}, so that call needs a departure")
            if i >= arrives and call.arrive is None:
                raise ValueError(f"trip {self.train.id}: it arrives at {call.platform}, so that call needs an arrival")
        return self


class ServiceGroup(NetModel):
    """One clockface line, repeated every `every_min` from `first_min` while a whole run still fits in the shift."""

    code: str
    type: str
    entry_node: str
    stops: tuple[str, ...] = Field(min_length=1)
    dwell: tuple[int, ...] = Field(min_length=1)
    exit_node: str | None = None
    first_min: int = Field(default=0, ge=0)
    every_min: int = Field(default=60, gt=0)
    slack_pct: int = Field(default=10, ge=0)

    @model_validator(mode="after")
    def valid_group(self):
        if len(self.dwell) != len(self.stops):
            raise ValueError(f"group {self.code}: one dwell per stop, got {len(self.dwell)} for {len(self.stops)}")
        if not self.code or self.code[-1].isdigit():
            raise ValueError(f"group {self.code!r}: a code may not end in a digit, or train ids would not say the line")
        return self


class Connection(NetModel):
    """Passengers changing from one service to another at a node (§4.1). Missing it is charged by §6.6 C2."""

    feeder: str
    connector: str
    node: str
    transfers: int = Field(gt=0)
    min_transfer_min: int = Field(default=5, gt=0)

    @model_validator(mode="after")
    def valid_connection(self):
        if self.feeder == self.connector:
            raise ValueError(f"connection at {self.node}: {self.feeder} cannot feed itself")
        return self


DisruptionKind = Literal[
    "breakdown", "possession", "speed_restriction", "signal_failure", "points_failure", "platform_closed", "late_entry"
]
UNANNOUNCED: frozenset[str] = frozenset({"breakdown", "signal_failure", "points_failure"})  # revealed as they happen
ON_TRACKS: frozenset[str] = frozenset({"possession", "speed_restriction", "signal_failure"})


class Disruption(NetModel):
    """One thing that breaks the plan (docs/network-v0.3.md §4.2), fixed by the scenario seed.

    `start` to `end` is when it is in force. `reveal` is when the agent learns of it: at `start` for the unannounced
    kinds, ahead of it for the rest. A breakdown stops `train` wherever it is at `start`, if it is in the area then.
    A late entry moves `train`'s entry from `start` (the booked minute) to `end`. The other kinds close or slow
    `elements`: tracks for a possession, a speed restriction or a signal failure, one move for a points failure,
    one platform for a platform out of use.
    """

    id: str
    kind: DisruptionKind
    start: int = Field(ge=0)
    end: int
    reveal: int = Field(ge=0)
    train: str | None = None
    elements: tuple[str, ...] = ()
    speed_kmh: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def valid_disruption(self):
        if self.end <= self.start:
            raise ValueError(f"disruption {self.id}: it must end after it starts")
        if self.reveal > self.start or (self.kind in UNANNOUNCED and self.reveal != self.start):
            when = "at" if self.kind in UNANNOUNCED else "by"
            raise ValueError(f"disruption {self.id}: a {self.kind} is revealed {when} its start")
        about_train = self.kind in {"breakdown", "late_entry"}
        if about_train != (self.train is not None) or about_train == bool(self.elements):
            named = "a train" if about_train else "elements"
            raise ValueError(f"disruption {self.id}: a {self.kind} names {named} only")
        if self.kind in {"points_failure", "platform_closed"} and len(self.elements) != 1:
            raise ValueError(f"disruption {self.id}: a {self.kind} is on exactly one element")
        if (self.kind == "speed_restriction") != (self.speed_kmh is not None):
            raise ValueError(f"disruption {self.id}: a speed is given exactly for a speed restriction")
        return self


class Network(NetModel):
    nodes: tuple[Node, ...] = Field(min_length=1)
    tracks: tuple[Track, ...]
    platforms: tuple[Platform, ...]
    moves: tuple[Move, ...]
    crossings: tuple[tuple[str, str], ...] = ()  # move pairs whose paths cross (diamonds) without sharing points
    _index: dict[str, Track | Platform | Move] = PrivateAttr(default_factory=dict)
    _nodes: dict[str, Node] = PrivateAttr(default_factory=dict)
    _conflicts: dict[str, frozenset[str]] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def valid_graph(self):
        counts = Counter(x.id for group in (self.nodes, self.tracks, self.platforms, self.moves) for x in group)
        if dup := sorted(i for i, n in counts.items() if n > 1):
            raise ValueError(f"duplicate ids: {dup}")
        nodes = self._nodes = {n.id: n for n in self.nodes}
        self._index = {x.id: x for x in (*self.tracks, *self.platforms, *self.moves)}
        for t in self.tracks:
            if t.a not in nodes or t.b not in nodes or t.a == t.b:
                raise ValueError(f"track {t.id} must join two different known nodes")
        for p in self.platforms:
            if p.node not in nodes or nodes[p.node].kind in {"junction", "portal"}:
                raise ValueError(f"platform {p.id} must be in a station, terminal, yard or freight terminal")
        for m in self.moves:
            self._check_move(m, nodes)
        self._conflicts = self._derive_conflicts()
        used = {(end, m.node) for m in self.moves for end in m.ends}
        for t in self.tracks:
            for end in (t.a, t.b):
                if nodes[end].kind != "portal" and (t.id, end) not in used:
                    raise ValueError(f"track {t.id} dead-ends at {end}: no move uses it")
        for p in self.platforms:
            if (p.id, p.node) not in used:
                raise ValueError(f"platform {p.id} has no move")
        for n in self.nodes:  # a portal is where one line leaves the network: nothing inside it, no link between lines
            if n.kind == "portal" and (
                any(m.node == n.id for m in self.moves)
                or len({t.other(n.id) for t in self.tracks if n.id in (t.a, t.b)}) != 1
            ):
                raise ValueError(f"portal {n.id} must have no moves and lead to exactly one node")
        reached, todo = set(), [self.nodes[0].id]
        while todo:
            n = todo.pop()
            if n not in reached:
                reached.add(n)
                if nodes[n].kind != "portal" or n == self.nodes[0].id:
                    todo += [t.other(n) for t in self.tracks if n in (t.a, t.b)]
        if missing := sorted(set(nodes) - reached):
            raise ValueError(f"network is not connected: {missing} unreachable")
        return self

    def _check_move(self, m: Move, nodes: dict[str, Node]) -> None:
        if m.node not in nodes or m.ends[0] == m.ends[1]:
            raise ValueError(f"move {m.id} must link two different elements at a known node")
        platform_ends = 0
        for end in m.ends:
            el = self._index.get(end)
            if isinstance(el, Track) and m.node in (el.a, el.b):
                continue
            if not (isinstance(el, Platform) and el.node == m.node):
                raise ValueError(f"move {m.id}: {end} is not a track or platform at {m.node}")
            platform_ends += 1
        if platform_ends > 1 or (platform_ends == 1) != (m.side is not None):
            raise ValueError(f"move {m.id}: needs a side exactly when one end is a platform")

    def _derive_conflicts(self) -> dict[str, frozenset[str]]:
        pairs = set()
        for x, y in combinations(self.moves, 2):
            if x.node != y.node:
                continue
            shared = set(x.ends) & set(y.ends)
            if any(isinstance(self._index[e], Track) for e in shared) or (shared and x.side == y.side):
                pairs.add((x.id, y.id))
        for x, y in self.crossings:
            mx, my = self._index.get(x), self._index.get(y)
            if x == y or not isinstance(mx, Move) or not isinstance(my, Move) or mx.node != my.node:
                raise ValueError(f"crossing ({x}, {y}) must be two different moves at one node")
            pairs.add((x, y))
        conflicts: dict[str, set[str]] = {m.id: set() for m in self.moves}
        for x, y in pairs:
            conflicts[x].add(y)
            conflicts[y].add(x)
        return {k: frozenset(v) for k, v in conflicts.items()}

    def element(self, element_id: str) -> Track | Platform | Move | None:
        return self._index.get(element_id)

    def node(self, node_id: str) -> Node | None:
        return self._nodes.get(node_id)

    def conflicts(self, move_id: str) -> list[str]:
        return sorted(self._conflicts.get(move_id, ()))


class Timetable(NetModel):
    """A generated scenario: the layout, the shift's trips, and the connections passengers rely on."""

    net: Network
    trips: tuple[Trip, ...] = Field(min_length=1)
    connections: tuple[Connection, ...] = ()
    shift_min: int = Field(gt=0)
    seed: int = 0
    tier: str = "T1"
    disruptions: tuple[Disruption, ...] = ()

    @model_validator(mode="after")
    def valid_timetable(self):
        trips = {t.train.id for t in self.trips}
        if len(trips) != len(self.trips):
            raise ValueError("two trips share a train id")
        for trip in self.trips:
            for call in trip.calls:
                if not isinstance(self.net.element(call.platform), Platform):
                    raise ValueError(f"trip {trip.train.id} calls at {call.platform}, which is not a platform")
            for portal in (trip.entry_node, trip.exit_node):
                node = self.net.node(portal) if portal else None
                if portal is not None and (node is None or node.kind != "portal"):
                    raise ValueError(f"trip {trip.train.id}: {portal} is not a portal")
        for link in self.connections:
            if missing := sorted({link.feeder, link.connector} - trips):
                raise ValueError(f"connection at {link.node} names unknown train {missing[0]}")
            if self.net.node(link.node) is None:
                raise ValueError(f"connection {link.feeder}->{link.connector} is at unknown node {link.node}")
        self._check_disruptions({t.train.id: t for t in self.trips})
        return self

    def _check_disruptions(self, trips: dict[str, Trip]) -> None:
        if len({d.id for d in self.disruptions}) != len(self.disruptions):
            raise ValueError("two disruptions share an id")
        wanted = {"points_failure": Move, "platform_closed": Platform}
        for d in self.disruptions:
            if d.train is not None and d.train not in trips:
                raise ValueError(f"disruption {d.id} names unknown train {d.train}")
            if d.kind == "late_entry" and (trips[d.train].entry_node is None or trips[d.train].entry_tick != d.start):
                raise ValueError(f"disruption {d.id}: a late entry starts at the booked entry of a train from a portal")
            kind = wanted.get(d.kind, Track)
            for element in d.elements:
                if not isinstance(self.net.element(element), kind):
                    raise ValueError(f"disruption {d.id}: {element} is not a {kind.__name__.lower()}")
