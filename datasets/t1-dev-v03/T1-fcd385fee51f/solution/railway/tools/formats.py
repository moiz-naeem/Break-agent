"""Tool output formats (T05 §2.2). Pure functions of visible data; JSON outputs are canonical."""

from __future__ import annotations

from railway.sim.constants import TRAIN_TYPES
from railway.sim.cost import CostBreakdown
from railway.sim.model import Scenario
from railway.sim.observe import AdvanceResult, State
from railway.util import canonical_json

def network(scenario: Scenario) -> str:
    net = scenario.network.model_dump(mode="json")
    return canonical_json(
        {
            "rules_version": scenario.rules_version,
            "shift_ticks": scenario.shift_ticks,
            "factors": list(scenario.factors),
            **net,
            "crew_stations": [s["id"] for s in net["stations"] if s["crew_station"]],
        }
    )


def timetable_entry(scenario: Scenario, train) -> dict:
    out = train.model_dump(mode="json")
    out["length_m"] = TRAIN_TYPES[train.type]["length_m"]
    out["max_kmh"] = TRAIN_TYPES[train.type]["max_speed_kmh"]
    if "F10" not in scenario.factors:
        del out["crew_used_at_entry"]
    return out


def timetable(scenario: Scenario, trains) -> str:
    out = {"trains": [timetable_entry(scenario, t) for t in trains]}
    if scenario.connections:
        ids = {t.id for t in trains}
        out["connections"] = [
            c.model_dump(mode="json") for c in scenario.connections if c.feeder in ids or c.connector in ids
        ]
    return canonical_json(out)


def full_status(state: State) -> str:
    return canonical_json(state.model_dump(mode="json"))


def _position(t, loop_kinds: dict[tuple[str, str], str]) -> str:
    if t.phase == "at_station":
        if t.track == "main" and t.station in {"YW", "YE"}:
            return f"at {t.station}"
        if t.track == "main":
            return f"at {t.station}/main"
        return f"at {t.station}/{t.track}-{loop_kinds[(t.station, t.track)]}"
    if t.phase == "in_section":
        if t.direction is not None:
            return f"{t.section} {'eastbound' if t.direction == 'east' else 'westbound'}"
        west, east = t.section.split("-", 1)
        return f"{t.section} to {east if t.section_from == west else west}"
    origin = t.origin
    return f"waiting {origin} after {t.waits_for}" if t.waits_for else f"waiting {origin} entry@{t.entry_tick}"


def compact_status(state: State, scenario: Scenario) -> str:
    loop_kinds = {(s.id, loop.id): loop.kind for s in scenario.network.stations for loop in s.loops}
    loop_kinds |= {(t.id, track.id): track.kind for t in scenario.network.terminals for track in t.tracks}
    lines = [f't={state.tick} | gridlock: {",".join(state.gridlock) or "none"} | notes: "{state.notes}"']
    for t in sorted(state.trains, key=lambda x: x.id):
        if t.phase == "finished":
            continue
        line = f"{t.id:<4} {t.type:<9} {_position(t, loop_kinds)}"
        if t.phase != "not_entered":
            auth = f"{t.authority.dest_station}/{t.authority.dest_track}" if t.authority else "-"
            line += f" auth:{auth} delay:{f'+{t.delay}' if t.delay > 0 else '0'}"
            if t.crew_remaining is not None:
                line += f" crew:{t.crew_remaining}"
            if t.next_stop is not None:
                line += f" next:{t.next_stop.station}@{t.next_stop.arr}"
            if t.off_route:
                line += f" OFF-ROUTE(cannot reach {t.destination})"
        lines.append(line)
    parts = [f"possession {p['section']} {p['start']}-{p['end']}" for p in state.possessions]
    parts += [
        f"speed {','.join(r['sections'])} {r['speed_kmh']}km/h {r['start']}-{r['end']}" for r in state.restrictions
    ]
    lines.append(f"restrictions: {'; '.join(parts) or 'none'}")
    return "\n".join(lines)


def final_line(tick: int, cost: CostBreakdown) -> str:
    return (
        f"SHIFT ENDED at t={tick}. Final cost {cost.total} (pax {cost.pax_lateness}, missed calls "
        f"{cost.missed_calls}, unfinished {cost.unfinished}, freight {cost.freight_lateness}, crew "
        f"{cost.crew_timeout}, gridlock {cost.gridlock}, operations {cost.operations})"
    )


def advance(result: AdvanceResult, since: int, used: int, budget: int, final: str | None) -> str:
    stops = ", ".join(
        " ".join(x for x in (r.kind, r.train, f"({r.detail})" if r.detail else None) if x) for r in result.reasons
    )
    lines = [f"t={result.tick} stop: {stops}"]
    if final:
        lines.append(final)
    lines.append(f"changes since t={since}:")
    lines += [f"  t={c.tick} {c.text}" for c in result.changes]
    lines.append(f"budget: {used}/{budget}")
    return "\n".join(lines)
