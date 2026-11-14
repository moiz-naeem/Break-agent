"""Interlocking for the v0.3 network: route locking with conflict tables, direction locks and block occupancy.

A route is a list of element ids walked from `start_node`. elements[0] is where the train is now:
- a platform: `start_node` is that platform's node;
- a track: `start_node` is the track end the train came from (a portal when it is entering the network).
Moves and platforms are taken whole. A track the route starts on or ends on is taken only by its entry block,
after which the train runs on under automatic block signalling; a track crossed in the middle of one route is
taken whole. The train's own position is checked, and taken only when entering.
"""

from __future__ import annotations

from typing import NamedTuple

from .model import Move, Network, Platform, Track, TrainSpec, Travel
from .rules import NetRule

Resource = str | tuple[str, int]  # a move or platform id, or (track id, block index)


class Refusal(NamedTuple):
    rule: NetRule
    detail: str

    def __str__(self) -> str:
        return f"REFUSED {self.rule.value}: {self.detail}"


class Step(NamedTuple):
    element: Track | Platform | Move
    travel: Travel | None  # direction of travel on tracks, None elsewhere


def walk(net: Network, start_node: str, elements: list[str]) -> list[Step] | Refusal:
    """Check that the elements form one connected path from start_node without reversing or repeating."""
    if len(elements) < 2:
        return Refusal(NetRule.BAD_ROUTE, "a route needs the train's position and at least one element")
    if dup := sorted({e for e in elements if elements.count(e) > 1}):
        return Refusal(NetRule.BAD_ROUTE, f"route visits {dup[0]} twice")
    node, steps = start_node, []
    for i, eid in enumerate(elements):
        el = net.element(eid)
        if el is None:
            return Refusal(NetRule.UNKNOWN_ELEMENT, f"no element {eid!r}")
        prev = steps[-1].element if steps else None
        if prev is not None and not isinstance(el, Move) and not isinstance(prev, Move):
            return Refusal(NetRule.NOT_CONNECTED, f"{prev.id} and {eid} need a move between them")
        if isinstance(el, Track):
            if node not in (el.a, el.b):
                return Refusal(NetRule.NOT_CONNECTED, f"{eid} does not touch {node}")
            travel = el.travel_from(node)
            if el.direction not in (travel, "both"):
                one_way = f"{el.a}->{el.b}" if el.direction == "ab" else f"{el.b}->{el.a}"
                return Refusal(NetRule.WRONG_DIRECTION, f"{eid} runs {one_way} only")
            steps.append(Step(el, travel))
            node = el.other(node)
            continue
        if el.node != node:
            return Refusal(NetRule.NOT_CONNECTED, f"{eid} is not at {node}")
        nxt = elements[i + 1] if i + 1 < len(elements) else None
        if isinstance(el, Move):
            if prev is None or prev.id not in el.ends or nxt != el.other(prev.id):
                return Refusal(NetRule.NOT_CONNECTED, f"{eid} links {el.ends[0]} and {el.ends[1]}")
        elif prev is not None and isinstance(out := net.element(nxt or ""), Move) and out.side == prev.side:
            return Refusal(NetRule.BAD_ROUTE, f"{eid}: a route may not reverse in a platform")
        steps.append(Step(el, None))
    return steps


def admissible(train: TrainSpec, steps: list[Step]) -> Refusal | None:
    """Checks that no amount of waiting can fix, so a caller never retries a route that can never be granted."""
    for i, (el, _) in enumerate(steps):
        if train.electric and not isinstance(el, Move) and not el.electrified:
            return Refusal(NetRule.NOT_ELECTRIFIED, f"{el.id} has no overhead line for electric {train.id}")
        # A train must fit every road it runs onto, not only the one it stops at. Never steps[0]: that is where
        # the train already stands, and refusing it there would strand the train instead of keeping it out.
        if i and isinstance(el, Platform) and el.length_m < train.length_m:
            return Refusal(NetRule.PLATFORM_LEN, f"{el.id} is {el.length_m} m, {train.id} is {train.length_m} m")
    last = steps[-1].element
    if isinstance(last, Platform) and train.passenger and last.kind == "siding":
        return Refusal(NetRule.PLATFORM_KIND, f"{last.id} is a siding; passenger trains need a platform")
    return None


def claims(steps: list[Step], entering: bool) -> list[tuple[Resource, Step]]:
    """The resources a route takes. The single source of truth for check() and lock()."""
    last = len(steps) - 1
    out: list[tuple[Resource, Step]] = []
    for i, step in enumerate(steps):
        if i == 0 and not entering:
            continue  # the train already stands there
        el = step.element
        if isinstance(el, Track):
            order = list(range(el.blocks)) if step.travel == "ab" else list(range(el.blocks - 1, -1, -1))
            # Only the entry block, at both ends: a train entering the area occupies one block like any other,
            # and runs on under automatic block signalling. Taking the whole track would cap a portal at one
            # train per transit, which is several times the real entry headway.
            whole = order if 0 < i < last else order[:1]
            out += [((el.id, idx), step) for idx in whole]
        else:
            out.append((el.id, step))
    return out


def _name(resource: Resource) -> str:
    return resource if isinstance(resource, str) else f"{resource[0]} block {resource[1]}"


class Interlocking:
    """Who holds what. Every lock is all-or-nothing: a refused route changes nothing."""

    def __init__(self, net: Network):
        self.net = net
        self.holder: dict[Resource, str] = {}
        self.directions: dict[str, dict[str, Travel]] = {}  # track id -> {train id: travel}

    def entering(self, start_node: str, steps: list[Step]) -> bool:
        # An entering train takes its whole entry track; step_block frees each block behind it as it runs on.
        node = self.net.node(start_node)
        return isinstance(steps[0].element, Track) and node is not None and node.kind == "portal"

    def _obstacles(self, train: TrainSpec, start_node: str, steps: list[Step]):
        """Yield (rule, detail, holder) for everything held by others that the route needs, in refusal order."""
        first = steps[0].element
        if isinstance(first, Platform) and self.holder.get(first.id) not in (None, train.id):
            yield NetRule.PLATFORM_OCCUPIED, f"{first.id} is held by {self.holder[first.id]}", self.holder[first.id]
        for el, travel in steps:
            if isinstance(el, Track):
                for t in sorted(t for t, d in self.directions.get(el.id, {}).items() if d != travel):
                    yield NetRule.DIRECTION_LOCKED, f"{el.id} is set the other way for {t}", t
            elif isinstance(el, Move):
                for c in self.net.conflicts(el.id):
                    if c in self.holder:
                        holder = self.holder[c]
                        yield NetRule.MOVE_CONFLICT, f"{el.id} conflicts with {c}, set for {holder}", holder
        for resource, (el, _) in claims(steps, self.entering(start_node, steps)):
            other = self.holder.get(resource)
            if other not in (None, train.id):
                if isinstance(el, Track):
                    rule = NetRule.BLOCK_OCCUPIED
                else:
                    rule = NetRule.MOVE_LOCKED if isinstance(el, Move) else NetRule.PLATFORM_OCCUPIED
                yield rule, f"{_name(resource)} is held by {other}", other

    def check(self, train: TrainSpec, start_node: str, elements: list[str]) -> list[Step] | Refusal:
        steps = walk(self.net, start_node, elements)
        if isinstance(steps, Refusal):
            return steps
        if refusal := admissible(train, steps):
            return refusal
        for rule, detail, _ in self._obstacles(train, start_node, steps):
            return Refusal(rule, detail)
        return steps

    def blockers(self, train: TrainSpec, start_node: str, elements: list[str]) -> set[str]:
        """Every other train holding something this route needs: who it is waiting for, if it cannot be set."""
        steps = walk(self.net, start_node, elements)
        if isinstance(steps, Refusal):
            return set()
        return {holder for _, _, holder in self._obstacles(train, start_node, steps)} - {train.id}

    def step_block(self, train_id: str, track_id: str, travel: Travel, done: int) -> str | None:
        """Move a train from its done-th block (in travel order) into the next one; return the blocker, if any."""
        track = self.net.element(track_id)
        order = list(range(track.blocks)) if travel == "ab" else list(range(track.blocks - 1, -1, -1))
        nxt, cur = (track_id, order[done + 1]), (track_id, order[done])
        if self.holder.get(nxt) not in (None, train_id):
            return self.holder[nxt]
        self.holder[nxt] = train_id
        if self.holder.get(cur) == train_id:
            del self.holder[cur]
        return None

    def occupy(self, train_id: str, platform_id: str) -> str | None:
        """Place a train that starts its trip standing in a platform; return the holder if it is taken."""
        other = self.holder.get(platform_id)
        if other not in (None, train_id):
            return other
        self.holder[platform_id] = train_id
        return None

    def lock(self, train: TrainSpec, start_node: str, elements: list[str]) -> Refusal | None:
        steps = self.check(train, start_node, elements)
        if isinstance(steps, Refusal):
            return steps
        for resource, (el, travel) in claims(steps, self.entering(start_node, steps)):
            self.holder[resource] = train.id
            if isinstance(el, Track):
                self.directions.setdefault(el.id, {})[train.id] = travel
        return None

    def release(self, train_id: str, element_id: str) -> None:
        """Sectional release: free one element behind a train (a whole track frees all its blocks there)."""
        mine = [
            r
            for r, t in self.holder.items()
            if t == train_id and (r == element_id or (isinstance(r, tuple) and r[0] == element_id))
        ]
        for resource in mine:
            del self.holder[resource]
        holders = self.directions.get(element_id, {})
        holders.pop(train_id, None)
        if not holders:
            self.directions.pop(element_id, None)

    def release_train(self, train_id: str) -> None:
        self.holder = {r: t for r, t in self.holder.items() if t != train_id}
        for track in list(self.directions):
            self.release(train_id, track)
