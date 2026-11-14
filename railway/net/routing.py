"""Shortest route between two places on the graph, ignoring who holds what (docs/network-v0.3.md §5.1).

The interlocking decides whether a route can be taken now; this only decides where a train would go. It obeys the
rules no amount of waiting can change: one-way tracks, overhead line, and no reversing inside a platform.
"""

from __future__ import annotations

from collections import deque

from .model import Network, Platform, Track, TrainSpec

State = tuple[str, str, str | None]  # element, node reached, side it entered a platform by


def _usable(element: Track | Platform, train: TrainSpec) -> bool:
    return element.electrified or not train.electric


def find_route(
    net: Network, start: str, facing: str, target: str, train: TrainSpec, side_in: str | None = None
) -> list[str] | None:
    """Element path from `start` (the train's position, facing `facing`) to platform `target`, or None.

    `facing` is the node the train is heading towards: for a platform, its own node. `side_in` is the side the
    train entered its current platform by, so it cannot leave the way it came in, not even across a stop.
    """
    goal = net.element(target)
    if start == target:
        return None  # already there: there is no route to walk, and callers must not be handed a one-element path
    if not isinstance(goal, Platform) or goal.length_m < train.length_m or (train.passenger and goal.kind == "siding"):
        return None
    begin: State = (start, facing, side_in)
    paths: dict[State, list[str]] = {begin: [start]}
    queue = deque([begin])
    while queue:
        state = queue.popleft()
        element, node, side_in = state
        path = paths[state]
        if element == target:
            return path
        for move in sorted((m for m in net.moves if m.node == node and element in m.ends), key=lambda m: m.id):
            if move.side is not None and move.side == side_in:
                continue  # a route may not reverse in a platform
            nxt = net.element(move.other(element))
            if nxt.id in path or not _usable(nxt, train):
                continue
            if isinstance(nxt, Track):
                if nxt.direction not in (nxt.travel_from(node), "both"):
                    continue
                ahead: State = (nxt.id, nxt.other(node), None)
            else:
                ahead = (nxt.id, node, move.side)
            if ahead not in paths:
                paths[ahead] = [*path, move.id, nxt.id]
                queue.append(ahead)
    return None
