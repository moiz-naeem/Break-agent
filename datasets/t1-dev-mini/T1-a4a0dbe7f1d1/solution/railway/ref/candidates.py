"""Legal, useful authority candidates for scripted dispatch policies."""

from __future__ import annotations

from dataclasses import dataclass

from railway.sim.constants import TRAIN_TYPES
from railway.sim.model import Authorize


@dataclass(frozen=True)
class _Wait:
    """Sentinel for keeping a train at its current decision point."""


WAIT = _Wait()


def _route_stations(sim, start: str, direction: str) -> list[str]:
    stations = ["YW", *[s.id for s in sim._scenario.network.stations], "YE"]
    if direction == "west":
        stations.reverse()
    try:
        return stations[stations.index(start) + 1 :]
    except ValueError:
        return []


def _allowed_tracks(sim, train, station: str) -> list[str]:
    if station in {"YW", "YE"}:
        return ["main"]
    definition = sim._stations[station]
    tracks = [loop.id for loop in definition.loops]
    if any(stop.station == station for stop in sim._train_defs[train["id"]].stops):
        tracks.append("main")
    return sorted(set(tracks))


def _deadlock_safe(sim, train_id: str, destination: str) -> bool:
    """Reject a destination from which opposing traffic has no safe place to clear."""
    if destination in {"YW", "YE"}:
        return True
    train = sim._trains[train_id]
    forward = _route_stations(sim, destination, train["direction"])
    for station in forward[:3]:
        if station in {"YW", "YE"}:
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


def _opposing_resource_on_path(sim, train_id: str, start: str, end: str) -> bool:
    train = sim._trains[train_id]
    stations = ["YW", *[s.id for s in sim._scenario.network.stations], "YE"]
    first, last = stations.index(start), stations.index(end)
    step = 1 if last > first else -1
    path = []
    for i in range(first, last, step):
        a, b = stations[i], stations[i + step]
        section = sim._section_by_pair.get((a, b)) or sim._section_by_pair.get((b, a))
        if section:
            path.append(section.id)
        if b not in {start, end, "YW", "YE"}:
            path.append(f"{b}/main")
    for resource in path:
        holder = sim._holders.get(resource)
        if holder and holder != train_id and sim._trains[holder]["direction"] != train["direction"]:
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
    if start not in sim._station_idx:
        return [WAIT]
    ahead = _route_stations(sim, start, train["direction"])
    if not ahead:
        return [WAIT]
    destination = sim._train_defs[train_id].stops[-1].station
    reachable = ahead[:lookahead_stations]
    if destination in ahead and destination not in reachable:
        reachable.append(destination)
    options = []
    start_index = sim._station_idx[start]
    for station in reachable:
        distance = abs(sim._station_idx[station] - start_index)
        if deadlock_safe and not _deadlock_safe(sim, train_id, station):
            continue
        for track in _allowed_tracks(sim, train, station):
            action = Authorize(train=train_id, station=station, track=track)
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
    """T05 greedy's move for a waiting train: the nearest station ahead (to its destination) with a grantable
    loop, lowest loop id first; the destination yard counts; never a station's main. None if nothing is granted."""
    train = sim._trains[train_id]
    if train["phase"] != "at_station" or train["auth"]:
        return None
    ahead = _route_stations(sim, train["station"], train["direction"])
    destination = sim._train_defs[train_id].stops[-1].station
    for station in ahead[: ahead.index(destination) + 1] if destination in ahead else ahead:
        tracks = ["main"] if station in {"YW", "YE"} else sorted(loop.id for loop in sim._stations[station].loops)
        for track in tracks:
            action = Authorize(train=train_id, station=station, track=track)
            if sim.check(action).ok:
                return action
    return None
