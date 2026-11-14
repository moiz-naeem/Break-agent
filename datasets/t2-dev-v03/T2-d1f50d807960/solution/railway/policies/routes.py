"""Timetabled routes for scripted policies, on a chain (by direction) or a graph (the train's `route`).

Scripted policies follow each train's timetabled route. On a graph they name the route's intermediate stations
in `via`, so a grant can never take another way; on a chain `via` stays empty and grants are as in v0.2.
"""

from __future__ import annotations

from railway.sim.model import Authorize

YARDS = ("YW", "YE")


def _chain(network_stations, direction: str) -> list[str]:
    places = ["YW", *network_stations, "YE"]
    return places if direction == "east" else places[::-1]


def route_of(static, train_id: str) -> list[str]:
    """From what an agent sees: `static.trains` dicts and `static.network`."""
    train = next(t for t in static.trains if t["id"] == train_id)
    if train.get("route"):
        return list(train["route"])
    return _chain([s["id"] for s in static.network["stations"]], train["direction"])


def sim_route(sim, train_id: str) -> list[str]:
    """From a (belief) simulator."""
    definition = sim._train_defs[train_id]
    if definition.route:
        return list(definition.route)
    return _chain([s.id for s in sim._scenario.network.stations], definition.direction)


def ahead(route: list[str], here: str | None) -> list[str]:
    """Nodes after `here` on the route, up to the destination; empty if `here` is not on it."""
    if here not in route:
        return []
    return route[route.index(here) + 1 :]


def via(route: list[str], start: str, dest: str, graph: bool) -> list[str]:
    if not graph or start not in route or dest not in route:
        return []
    return route[route.index(start) + 1 : route.index(dest)]


def is_graph_static(static) -> bool:
    return "terminals" in static.network and bool(static.network["terminals"])


def static_tracks(static, node: str, *, loops_only: bool = False) -> list[str]:
    """Tracks a policy may ask for at `node`: a station's loops (and main unless `loops_only`), a terminal's
    tracks, or a yard's main."""
    for s in static.network["stations"]:
        if s["id"] == node:
            loops = sorted(loop["id"] for loop in s["loops"])
            return loops if loops_only else ["main", *loops]
    for t in static.network.get("terminals", []):
        if t["id"] == node:
            return sorted(track["id"] for track in t["tracks"])
    return ["main"]


def sim_tracks(sim, node: str, *, loops_only: bool = False) -> list[str]:
    if node in sim._stations:
        loops = sorted(loop.id for loop in sim._stations[node].loops)
        return loops if loops_only else ["main", *loops]
    if node in sim._graph.terminals and not sim._graph.unlimited(node):
        return sorted(sim._graph.tracks[node])
    return ["main"]


def is_terminal(sim_or_static, node: str) -> bool:
    if hasattr(sim_or_static, "_graph"):
        return node in sim_or_static._graph.terminals
    return node in YARDS or any(t["id"] == node for t in sim_or_static.network.get("terminals", []))


def authorize(train: str, station: str, track: str, via_nodes: list[str]) -> Authorize:
    return Authorize(train=train, station=station, track=track, via=list(via_nodes))


def call_args(action: Authorize) -> dict:
    """Tool-call arguments for a grant; `via` only when there is one (chain logs stay as in v0.2)."""
    args = {"train": action.train, "station": action.station, "track": action.track}
    if action.via:
        args["via"] = list(action.via)
    return args
