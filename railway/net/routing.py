"""Shortest route between two places on the graph, ignoring who holds what (docs/network-v0.3.md §5.1).

The interlocking decides whether a route can be taken now; this only decides where a train would go. It obeys the
rules no amount of waiting can change: one-way tracks, overhead line, and no reversing inside a platform.
"""

from __future__ import annotations

from collections import deque

from .model import Network, Platform, Side, Track, TrainSpec

State = tuple[str, str, Side | None]  # element, node reached, side it entered a platform by


def _usable(element: Track | Platform, train: TrainSpec) -> bool:
    """Whether a train can be on this element at all: overhead line, and a road it fits on even to pass through."""
    if train.electric and not element.electrified:
        return False
    return not isinstance(element, Platform) or element.length_m >= train.length_m


def portal_track(net: Network, node: str, inbound: bool) -> str | None:
    """The track at a portal a train can run in on, or out on. Double track has one of each.

    None for a node that is not a portal: only a portal is a place a train can come from or disappear to.
    """
    at = net.node(node)
    if at is None or at.kind != "portal":
        return None
    for track in sorted(net.tracks, key=lambda t: t.id):
        if node not in (track.a, track.b):
            continue
        if track.direction in (track.travel_from(node if inbound else track.other(node)), "both"):
            return track.id
    return None


def find_route(
    net: Network, start: str, facing: str, target: str, train: TrainSpec, side_in: Side | None = None
) -> list[str] | None:
    """Element path from `start` (the train's position, facing `facing`) to platform `target`, or None.

    `facing` is the node the train is heading towards: for a platform, its own node. `side_in` is the side the
    train entered its current platform by, so it cannot leave the way it came in, unless the caller clears it for
    a turnaround. The target is a platform, or a track when the train is running out of the area through a portal.
    """
    goal = net.element(target)
    if start == target:
        return None  # already there: there is no route to walk, and callers must not be handed a one-element path
    if isinstance(goal, Platform) and train.passenger and goal.kind == "siding":
        return None  # length is checked for every element by _usable, including this one
    if not isinstance(goal, Platform | Track):
        return None  # a platform to call at, or the track a train runs out of the area on
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
