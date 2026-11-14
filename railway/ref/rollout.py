"""Short-horizon simulation used by the reference dispatcher."""

from __future__ import annotations

from railway.policies.routes import ahead, sim_route, via
from railway.sim.constants import CREW_LIMIT, TRAIN_TYPES, TURNAROUND, UNFINISHED_COST
from railway.sim.cost import freight_cost, missed_call_cost, pax_cost, unfinished_cost
from railway.sim.model import RelieveCrew
from railway.sim.timing import travel_ticks

from .candidates import _section_between, greedy_choice


def _missed(definition, index: int) -> float:
    alight, board = definition.flow(index)
    return missed_call_cost(definition.type, alight, board, definition.stops[index].headway)


def terminal_estimate(sim) -> float:
    """What the rest of the shift will still cost under free running: lateness at the remaining stops, stops
    that fall after the shift end (missed calls), trains that won't finish, and stops already skipped."""
    estimate = 0.0
    shift = sim._scenario.shift_ticks
    for tid, train in sim._trains.items():
        definition = sim._train_defs[tid]
        pax = definition.type in {"express", "local"}
        if pax:
            estimate += sum(_missed(definition, i) for i in range(train["stop_idx"]) if i not in train["served"])
        if train["phase"] == "finished":
            continue
        tick = max(sim.tick, train["entry_tick"])
        if train["phase"] == "in_section":
            tick = max(tick, train["section_exit"] or tick)
            current = sim._graph.other_end(train["section"], train["section_from"])
        elif train["phase"] == "not_entered":
            current = definition.start
        else:
            current = train["station"] or train["position"]
            tick = max(tick, train.get("dwell_until") or 0, train.get("ready_tick") or 0)
        remaining = list(range(train["stop_idx"], len(definition.stops)))
        route = sim_route(sim, tid)
        if current not in route:  # off its route: assume what is left is lost
            if pax:
                estimate += sum(_missed(definition, i) for i in remaining)
            estimate += unfinished_cost(definition, shift)
            continue
        cursor = route.index(current)
        for i in remaining:
            stop = definition.stops[i]
            target = route.index(stop.station) if stop.station in route else cursor
            while cursor < target:
                a, b = route[cursor], route[cursor + 1]
                section = sim._section_by_id[_section_between(sim, a, b)]
                tick += travel_ticks(section.length_m, section.line_speed_kmh, TRAIN_TYPES[definition.type]["max_speed_kmh"])
                cursor += 1
            late = tick - stop.arr
            final = i == len(definition.stops) - 1
            if pax:
                alight, _ = definition.flow(i)
                estimate += _missed(definition, i) if tick > shift else pax_cost(definition.type, alight, late)
            elif definition.type == "freight" and final:
                estimate += unfinished_cost(definition, shift) if tick > shift else freight_cost(late)
            if final and tick > shift and definition.type != "freight":
                estimate += UNFINISHED_COST
            if stop.dep is not None:
                tick = max(tick + stop.min_dwell, stop.dep)
    return round(estimate, 6)


def _base_policy_step(sim) -> None:
    """Apply greedy grants (plus the crew rule) at the current decision point."""
    _apply_crew_rule(sim)
    trains = sorted(
        (
            t
            for t in sim._trains.values()
            if t["phase"] == "at_station" and not t["auth"] and (t["hold_until"] is None or sim.tick >= t["hold_until"])
        ),
        key=lambda t: (t["entry_tick"], t["id"]),
    )
    for train in trains:
        # Exactly T05 greedy (no deadlock filter): the rollout policy then can't do worse than greedy by design.
        action = greedy_choice(sim, train["id"])
        if action is not None:
            sim.apply(action)


def crew_target_arrival(sim, train_id: str, station: str) -> int | None:
    """Projected arrival at the next crew station on the route (or the destination), from `station`."""
    crew_stations = {s.id for s in sim._scenario.network.stations if s.crew_station}
    route = sim_route(sim, train_id)
    nodes = ahead(route, station)
    if not nodes:
        return None
    target = next((s for s in nodes if s in crew_stations), nodes[-1])
    train = sim._trains[train_id]
    path = sim._route(station, train["came_by"], target, "main", via(route, station, target, not sim._graph.chain), strict=False)
    if not path or isinstance(path, str):
        return None
    return sim._project_arrival(train_id, path)


def _apply_crew_rule(sim) -> None:
    if "F10" not in sim._scenario.factors:
        return
    crew_stations = {station.id for station in sim._scenario.network.stations if station.crew_station}
    for train in sim._trains.values():
        if train["phase"] != "at_station" or train["station"] not in crew_stations:
            continue
        if train.get("relief_until") is not None and sim.tick < train["relief_until"]:
            continue
        used = sim._crew_used_at(train)
        if used is None:
            continue
        remaining = max(0, CREW_LIMIT - used)
        arrival = crew_target_arrival(sim, train["id"], train["station"])
        if arrival is not None and sim.tick + remaining < arrival + TURNAROUND:
            sim.apply(RelieveCrew(train=train["id"]))


def rollout(sim, horizon: int) -> float:
    """Advance a cloned belief simulator with safe greedy for ``horizon`` ticks."""
    start = sim.tick
    limit = min(start + horizon, sim._scenario.shift_ticks)
    saw_gridlock = bool(sim._gridlock)
    while not sim.done and sim.tick < limit:
        _base_policy_step(sim)
        result = sim.advance(max_tick=limit)
        saw_gridlock = saw_gridlock or any(r.kind == "GRIDLOCK" for r in result.reasons)
        if not result.reasons and result.tick >= limit:
            break
    return round(sim.cost().total + terminal_estimate(sim) + (10_000.0 if saw_gridlock else 0.0), 6)
