from __future__ import annotations

from .constants import TRAIN_TYPES


def travel_ticks(length_m: int, line_kmh: int, train_kmh: int, restriction_kmh: int | None = None) -> int:
    speed = min(line_kmh, train_kmh, restriction_kmh) if restriction_kmh is not None else min(line_kmh, train_kmh)
    return max(1, -(-(length_m * 60) // (speed * 1000)))


def project(sim_view, train_id: str, path: list[str], start_tick: int):
    """Project section occupancy and destination arrival without conflicts."""
    raw = sim_view._scenario
    train = next(t for t in raw.trains if t.id == train_id)
    sections = {s.id: s for s in raw.network.sections}
    result = []
    tick = start_tick
    for pos, resource in enumerate(path):
        if resource in sections:
            section = sections[resource]
            duration = travel_ticks(section.length_m, section.line_speed_kmh, TRAIN_TYPES[train.type]["max_speed_kmh"])
            result.append((resource, tick, tick + duration))
            tick += duration
        elif pos < len(path) - 1 and "/" in resource:
            st = resource.split("/", 1)[0]
            stop = next((s for s in train.stops if s.station == st), None)
            if stop and stop.dep is not None:
                tick = max(tick + stop.min_dwell, stop.dep)
    return result
