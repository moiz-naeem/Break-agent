from __future__ import annotations

from .constants import RESTART_TICKS, TRAIN_TYPES


def travel_ticks(length_m: int, line_kmh: int, train_kmh: int, restriction_kmh: int | None = None) -> int:
    speed = min(line_kmh, train_kmh, restriction_kmh) if restriction_kmh is not None else min(line_kmh, train_kmh)
    return max(1, -(-(length_m * 60) // (speed * 1000)))


def project(sim_view, train_id: str, path: list[str], start_tick: int):
    """Project section occupancy using only factors visible to this simulator."""
    train = sim_view._train_defs[train_id]
    state = sim_view._trains[train_id]
    sections = sim_view._section_by_id
    result = []
    tick = max(
        start_tick,
        state["entry_tick"],
        state["dwell_until"] or 0,
        state["relief_until"] or 0,
        state["breakdown_until"] or 0,
        state["blocked_until"] or 0,
        state["hold_until"] or 0,
    )
    for pos, resource in enumerate(path):
        if resource in sections:
            section = sections[resource]
            if state["section"] == resource and state["section_exit"] is not None:
                duration = max(0, state["section_exit"] - sim_view.tick)
            else:
                restriction = sim_view._restriction_at(resource, tick)
                duration = travel_ticks(
                    section.length_m, section.line_speed_kmh, TRAIN_TYPES[train.type]["max_speed_kmh"], restriction
                )
                if pos == 0 and state["restart_pending"] and "F5" in sim_view._scenario.factors:
                    duration += RESTART_TICKS[train.type]
            result.append((resource, tick, tick + duration))
            tick += duration
        elif pos < len(path) - 1 and "/" in resource:
            st = resource.split("/", 1)[0]
            stop = next((s for s in train.stops if s.station == st), None)
            if stop:
                tick = max(tick + stop.min_dwell, stop.dep or tick)
    return result
