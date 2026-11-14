"""Operations KPIs reported beside the score (docs/design-changes-v0.3.md §2.5). Never part of the reward."""

from __future__ import annotations

ON_TIME_MINUTES = 5  # a passenger stop reached at most this late counts as on time
RED_FLAGS = ("gridlocks", "crew_timeouts", "misroutes", "wrong_terminals", "missed_calls", "unfinished_trains")


def kpis(sim, counts: dict[str, int]) -> dict:
    """From a finished simulator and the episode's counts (session counts merged with the simulator's)."""
    stops = on_time = 0
    delay_minutes = 0.0
    freight = delivered = 0
    for definition in sim._scenario.trains:
        t = sim._trains[definition.id]
        if definition.type == "freight":
            freight += 1
            delivered += t["outcome"] == "arrived"
            continue
        if definition.type not in {"express", "local"}:
            continue
        arrived = {(a["station"], a["sched"]): a["actual"] for a in t["arrivals"]}
        for i, stop in enumerate(definition.stops):
            stops += 1
            actual = arrived.get((stop.station, stop.arr))
            if actual is None:
                continue
            late = max(0, actual - stop.arr)
            on_time += late <= ON_TIME_MINUTES
            delay_minutes += definition.flow(i)[0] * late
    flags = {k: counts.get(k, 0) for k in RED_FLAGS}
    return {
        "on_time_pct": round(100 * on_time / stops, 2) if stops else 100.0,
        "passenger_delay_minutes": round(delay_minutes, 2),
        "freight_delivered_pct": round(100 * delivered / freight, 2) if freight else 100.0,
        "red_flags": sum(flags.values()),
        "clean_shift": not any(flags.values()),
    }
