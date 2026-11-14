"""Legal, useful authority candidates for scripted dispatch policies, along each train's timetabled route."""

from __future__ import annotations

from dataclasses import dataclass

from railway.policies.routes import ahead, authorize, is_terminal, sim_route, sim_tracks, via
from railway.sim.constants import TRAIN_TYPES


@dataclass(frozen=True)
class _Wait:
    """Sentinel for keeping a train at its current decision point."""


WAIT = _Wait()


def _graph(sim) -> bool:
    return not sim._graph.chain


def _section_between(sim, a: str, b: str) -> str | None:
    return next((section for section, nxt in sim._graph.adj.get(a, []) if nxt == b), None)


def _allowed_tracks(sim, train, station: str) -> list[str]:
    if is_terminal(sim, station):
        return sim_tracks(sim, station)
    tracks = sim_tracks(sim, station, loops_only=True)
    if any(stop.station == station for stop in sim._train_defs[train["id"]].stops):
        tracks = [*tracks, "main"]
    return sorted(set(tracks))


def _grant(sim, train_id: str, start: str, station: str, track: str):
    return authorize(train_id, station, track, via(sim_route(sim, train_id), start, station, _graph(sim)))


def _deadlock_safe(sim, train_id: str, destination: str) -> bool:
    """Reject a destination from which opposing traffic has no safe place to clear."""
    if is_terminal(sim, destination):
        return True
    train = sim._trains[train_id]
    forward = ahead(sim_route(sim, train_id), destination)
    for station in forward[:3]:
        if is_terminal(sim, station):
            if not _opposing_resource_on_path(sim, train_id, destination, station):
                return True
            continue
        fitting = [
            loop
            for loop in sim._stations[station].loops
            if TRAIN_TYPES[train["type"]]["length_m"] <= loop.length_m
            and (loop.kind == "platform" or train["type"] not in {"express", "local"})
            and sim._holders.get(f"{station}/{loop.id}") in {None, train_id}
        ]
        if fitting and not _opposing_resource_on_path(sim, train_id, destination, station):
            return True
    return False


def _opposes(sim, holder: str, a: str, b: str) -> bool:
    """Whether `holder` runs from b towards a (the other way) on its own timetabled route."""
    route = sim_route(sim, holder)
    return a in route and b in route and route.index(b) < route.index(a)


def _opposing_resource_on_path(sim, train_id: str, start: str, end: str) -> bool:
    route = sim_route(sim, train_id)
    if start not in route or end not in route:
        return False
    nodes = route[route.index(start) : route.index(end) + 1]
    for a, b in zip(nodes, nodes[1:], strict=False):
        resources = [_section_between(sim, a, b)]
        if b not in {start, end} and not is_terminal(sim, b):
            resources.append(f"{b}/main")
        for resource in resources:
            holder = sim._holders.get(resource)
            if holder and holder != train_id and _opposes(sim, holder, a, b):
                return True
    return False


def candidates(sim, train_id: str, *, lookahead_stations: int = 4, max_candidates: int = 5, deadlock_safe: bool = True):
    """Return legal authority choices, ordered by farthest progress, then track id.

    ``sim`` is the belief simulator.  The function inspects only that simulator's
    currently known scenario and reservations.
    """
    if train_id not in sim._trains:
        return [WAIT]
    train = sim._trains[train_id]
    if train["phase"] == "finished" or train["phase"] == "not_entered":
        return [WAIT]
    start = train["auth"]["dest_station"] if train["auth"] else train["station"]
    if not sim._is_place(start):
        return [WAIT]
    nodes = ahead(sim_route(sim, train_id), start)
    if not nodes:
        return [WAIT]
    destination = sim._train_defs[train_id].end
    reachable = nodes[:lookahead_stations]
    if destination in nodes and destination not in reachable:
        reachable.append(destination)
    options = []
    for station in reachable:
        distance = nodes.index(station) + 1
        if deadlock_safe and not _deadlock_safe(sim, train_id, station):
            continue
        for track in _allowed_tracks(sim, train, station):
            action = _grant(sim, train_id, start, station, track)
            if sim.check(action).ok:
                options.append((distance, track, action))
    options.sort(key=lambda item: (-item[0], item[1]))
    chosen = [item[2] for item in options[:max_candidates]]
    # Greedy's own move is always evaluated (rollouts penalise any gridlock it leads to), so the reference
    # can't lose to greedy just because the farthest-first cut dropped it.
    nearest = greedy_choice(sim, train_id)
    if nearest is not None and nearest not in chosen:
        chosen.append(nearest)
    return chosen + [WAIT]


def greedy_choice(sim, train_id: str):
    """T05 greedy's move for a waiting train: the nearest station ahead on its route with a grantable loop,
    lowest loop id first; the destination terminal counts; never a station's main. None if nothing is granted."""
    train = sim._trains[train_id]
    if train["phase"] != "at_station" or train["auth"]:
        return None
    for station in ahead(sim_route(sim, train_id), train["station"]):
        terminal = is_terminal(sim, station)
        for track in sim_tracks(sim, station, loops_only=not terminal):
            action = _grant(sim, train_id, train["station"], station, track)
            if sim.check(action).ok:
                return action
    return None
