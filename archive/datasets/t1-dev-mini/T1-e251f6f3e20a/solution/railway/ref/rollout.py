"""Short-horizon simulation used by the reference dispatcher."""

from __future__ import annotations

from railway.sim.constants import CREW_LIMIT, TRAIN_TYPES, TURNAROUND
from railway.sim.cost import freight_cost, pax_cost
from railway.sim.model import RelieveCrew
from railway.sim.timing import travel_ticks

from .candidates import greedy_choice


def terminal_estimate(sim) -> float:
    """Estimate lateness at all remaining scheduled stops under free running."""
    estimate = 0.0
    stations = sim._station_ids
    for tid, train in sim._trains.items():
        if train["phase"] == "finished":
            continue
        definition = sim._train_defs[tid]
        tick = max(sim.tick, definition.entry_tick)
        if train["phase"] == "in_section":
            tick = max(tick, train["section_exit"] or tick)
            current = stations[sim._station_idx[train["position"]] + (1 if train["direction"] == "east" else -1)]
        elif train["phase"] == "reversing" and train.get("reversing_to"):
            tick = max(tick, train["section_exit"] or tick)
            current = train["reversing_to"][0]
        else:
            current = train["station"] or train["position"]
            tick = max(tick, train.get("dwell_until") or 0, train.get("ready_tick") or 0)

        remaining_stops = definition.stops[train["stop_idx"] :]
        if current not in sim._station_idx:
            continue
        cursor = sim._station_idx[current]
        for stop in remaining_stops:
            target = sim._station_idx[stop.station]
            step = 1 if target > cursor else -1
            while cursor != target:
                a, b = stations[cursor], stations[cursor + step]
                section = sim._section_by_pair.get((a, b)) or sim._section_by_pair.get((b, a))
                if section:
                    tick += travel_ticks(
                        section.length_m,
                        section.line_speed_kmh,
                        TRAIN_TYPES[definition.type]["max_speed_kmh"],
                    )
                cursor += step
                station = stations[cursor]
                if station != stop.station:
                    timed = next((x for x in remaining_stops if x.station == station), None)
                    if timed:
                        tick = max(tick + timed.min_dwell, timed.dep or tick)
            if definition.type in {"express", "local"}:
                estimate += pax_cost(definition.type, definition.pax, tick - stop.arr, stop.station in {"YW", "YE"})
            elif definition.type == "freight" and stop.station in {"YW", "YE"}:
                estimate += freight_cost(tick - stop.arr)
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


def _apply_crew_rule(sim) -> None:
    if "F10" not in sim._scenario.factors:
        return
    crew_stations = {station.id for station in sim._scenario.network.stations if station.crew_station}
    route = sim._station_ids
    for train in sim._trains.values():
        if train["phase"] != "at_station" or train["station"] not in crew_stations:
            continue
        if train.get("relief_until") is not None and sim.tick < train["relief_until"]:
            continue
        used = sim._crew_used_at(train)
        if used is None:
            continue
        remaining = max(0, CREW_LIMIT - used)
        direction = 1 if train["direction"] == "east" else -1
        index = sim._station_idx[train["station"]]
        ahead = route[index + direction :: direction]
        target = next((station for station in ahead if station in crew_stations), None)
        if target is None:
            target = sim._train_defs[train["id"]].stops[-1].station
        path = sim._route(train["station"], target, "main")
        arrival = sim._project_arrival(train["id"], path)
        if sim.tick + remaining < arrival + TURNAROUND:
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
