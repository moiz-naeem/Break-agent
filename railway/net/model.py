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
    """A scheduled stop. `arrive` is None at the origin, `depart` at the destination."""

    platform: str
    arrive: int | None = None
    depart: int | None = None
    min_dwell: int = Field(default=1, ge=0)


class Trip(NetModel):
    """One train's work for the shift: where it starts and the calls it must make, in order."""

    train: TrainSpec
    calls: tuple[Call, ...] = Field(min_length=2)
    entry_node: str | None = None  # portal it enters from; None means it starts standing at calls[0]
    entry_tick: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def valid_trip(self):
        if self.calls[0].depart is None or self.calls[-1].arrive is None:
            raise ValueError(f"trip {self.train.id}: the origin needs a departure and the destination an arrival")
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
