"""Shortest route between two places on the graph, ignoring who holds what (docs/network-v0.3.md §5.1).

The interlocking decides whether a route can be taken now; this only decides where a train would go. It obeys the
rules no amount of waiting can change: one-way tracks, overhead line, and no reversing inside a platform.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .model import Move, Network, Platform, Side, Track, TrainSpec

State = tuple[str, str, Side | None]  # element, node reached, side it entered a platform by
RouteKey = tuple[str, str, str, Side | None, int, bool, bool]


@dataclass
class _NetCache:
    tracks: tuple[Track, ...]
    platforms: tuple[Platform, ...]
    moves: tuple[Move, ...]
    moves_at: dict[tuple[str, str], tuple[Move, ...]]
    portals: dict[tuple[str, bool], str | None] = field(default_factory=dict)
    routes: dict[RouteKey, tuple[str, ...] | None] = field(default_factory=dict)


def _net_cache(net: Network) -> _NetCache:
    priv = net.__pydantic_private__
    cache: _NetCache | None = priv.get("_rcache")
    if (
        cache is not None
        and cache.tracks is net.tracks
        and cache.platforms is net.platforms
        and cache.moves is net.moves
    ):
        return cache
    grouped: dict[tuple[str, str], list[Move]] = {}
    for m in sorted(net.moves, key=lambda x: x.id):
        for end in m.ends:
            grouped.setdefault((m.node, end), []).append(m)
    cache = _NetCache(
        tracks=net.tracks,
        platforms=net.platforms,
        moves=net.moves,
        moves_at={k: tuple(v) for k, v in grouped.items()},
    )
    priv["_rcache"] = cache
    return cache


def _usable(element: Track | Platform, train: TrainSpec) -> bool:
    """Whether a train can be on this element at all: overhead line, and a road it fits on even to pass through."""
    if train.electric and not element.electrified:
        return False
    return not isinstance(element, Platform) or element.length_m >= train.length_m


def portal_track(net: Network, node: str, inbound: bool) -> str | None:
    """The track at a portal a train can run in on, or out on. Double track has one of each.

    None for a node that is not a portal: only a portal is a place a train can come from or disappear to.
    """
    cache = _net_cache(net)
    pkey = (node, inbound)
    if pkey in cache.portals:
        return cache.portals[pkey]
    at = net.node(node)
    if at is None or at.kind != "portal":
        cache.portals[pkey] = None
        return None
    found: str | None = None
    for track in sorted(net.tracks, key=lambda t: t.id):
        if node not in (track.a, track.b):
            continue
        if track.direction in (track.travel_from(node if inbound else track.other(node)), "both"):
            found = track.id
            break
    cache.portals[pkey] = found
    return found


def find_route(
    net: Network, start: str, facing: str, target: str, train: TrainSpec, side_in: Side | None = None
) -> list[str] | None:
    """Element path from `start` (the train's position, facing `facing`) to platform `target`, or None.

    `facing` is the node the train is heading towards: for a platform, its own node. `side_in` is the side the
    train entered its current platform by, so it cannot leave the way it came in, unless the caller clears it for
    a turnaround. The target is a platform, or a track when the train is running out of the area through a portal.
    """
    if start == target:
        return None  # already there: there is no route to walk, and callers must not be handed a one-element path
    cache = _net_cache(net)
    rkey: RouteKey = (start, facing, target, side_in, train.length_m, train.electric, train.passenger)
    if rkey in cache.routes:
        hit = cache.routes[rkey]
        return list(hit) if hit is not None else None
    goal = net.element(target)
    if isinstance(goal, Platform) and train.passenger and goal.kind == "siding":
        cache.routes[rkey] = None
        return None  # length is checked for every element by _usable, including this one
    if not isinstance(goal, Platform | Track):
        cache.routes[rkey] = None
        return None  # a platform to call at, or the track a train runs out of the area on
    moves_at = cache.moves_at
    begin: State = (start, facing, side_in)
    paths: dict[State, list[str]] = {begin: [start]}
    queue = deque([begin])
    while queue:
        state = queue.popleft()
        element, node, cur_side = state
        path = paths[state]
        if element == target:
            cache.routes[rkey] = tuple(path)
            return list(path)
        for move in moves_at.get((node, element), ()):
            if move.side is not None and move.side == cur_side:
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
    cache.routes[rkey] = None
    return None
