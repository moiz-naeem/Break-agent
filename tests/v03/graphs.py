"""A small graph network for v0.3 tests.

    TA --- A1 --- J1 --- M1 --- Y1 --- M2 --- J2 --- B1 --- TB
                  |       \\           /      |
    TC --- C1 ----+        +--- X1 ---+      +--- D1 --- TD

J1 and J2 are junctions. M1 -> M2 has two routes of equal length (via Y1 or via the bypass X1), so an
authority across it must name one with `via`. Every section is 5 km at 100 km/h: 3 ticks for an express or
a local, 5 for freight.
"""

from __future__ import annotations

from railway.sim.constants import RULES_VERSION
from railway.sim.model import Scenario

EDGES = [
    ("TA", "A1"), ("A1", "J1"), ("J1", "M1"), ("M1", "Y1"), ("Y1", "M2"), ("M2", "J2"), ("J2", "B1"),
    ("B1", "TB"), ("TC", "C1"), ("C1", "J1"), ("M1", "X1"), ("X1", "M2"), ("J2", "D1"), ("D1", "TD"),
]  # fmt: skip
STATIONS = ["A1", "J1", "M1", "Y1", "X1", "M2", "J2", "B1", "C1", "D1"]
MAIN = ["TA", "A1", "J1", "M1", "Y1", "M2", "J2", "B1", "TB"]


def loops(station: str) -> list[dict]:
    if station in {"Y1", "X1"}:
        return [{"id": "L1", "length_m": 700, "kind": "goods"}]
    return [{"id": "L1", "length_m": 300, "kind": "platform"}, {"id": "L2", "length_m": 700, "kind": "goods"}]


def network(terminal_tracks: dict | None = None) -> dict:
    tracks = {
        "TA": [{"id": "P1", "length_m": 300, "kind": "platform"}, {"id": "P2", "length_m": 700, "kind": "goods"}],
        "TB": [{"id": "P1", "length_m": 300, "kind": "platform"}, {"id": "P2", "length_m": 700, "kind": "goods"}],
        "TC": [{"id": "P1", "length_m": 300, "kind": "platform"}],
        "TD": [{"id": "P1", "length_m": 700, "kind": "goods"}],
    } | (terminal_tracks or {})
    return {
        "terminals": [{"id": t, "tracks": tracks[t]} for t in ("TA", "TB", "TC", "TD")],
        "stations": [
            {"id": s, "major": True, "crew_station": s in {"J1", "M2"}, "loops": loops(s)} for s in STATIONS
        ],
        "sections": [
            {"id": f"{a}-{b}", "west": a, "east": b, "length_m": 5000, "line_speed_kmh": 100} for a, b in EDGES
        ],
    }


def train(tid, kind, route, stops, entry=0, pax=None, next_service=None, crew=0):
    pax = {"express": 300, "local": 100}.get(kind, 0) if pax is None else pax
    return {
        "id": tid,
        "type": kind,
        "origin": route[0],
        "destination": route[-1],
        "route": route,
        "pax": pax,
        "entry_tick": entry,
        "stops": stops,
        "next_service": next_service,
        "crew_used_at_entry": crew,
    }


def stop(station, arr, dep=None, dwell=1, **extra):
    return {"station": station, "arr": arr, "dep": dep, "min_dwell": dwell if dep is not None else 0, **extra}


def scenario(trains, factors=("F1", "F2", "F3", "F4"), shift=200, events=(), connections=(), terminal_tracks=None):
    return Scenario.model_validate(
        {
            "schema_version": 2,
            "rules_version": RULES_VERSION,
            "scenario_id": "G-test",
            "seed": 0,
            "preset": "fixture",
            "shift_ticks": shift,
            "factors": sorted(factors),
            "network": network(terminal_tracks),
            "trains": sorted(trains, key=lambda t: t["id"]),
            "events": list(events),
            "connections": list(connections),
        }
    )


def express_ta_tb(tid="E1", entry=0, **kw):
    """TA -> TB along the main line, calling at J1 and M2, due at TB at t=30."""
    return train(
        tid, "express", MAIN,
        [stop("J1", 7, 8, alight=50, board=20), stop("M2", 19, 20, alight=100, board=30), stop("TB", 30, alight=200)],
        entry=entry, **kw,
    )  # fmt: skip


def line_scenario(trains, station_loops, factors=("F1", "F2", "F3", "F4"), shift=200):
    """A plain line T1 - S1 - ... - SN - T2 in graph form. `station_loops`: one list of loops per station."""
    ids = [f"S{i + 1}" for i in range(len(station_loops))]
    nodes = ["T1", *ids, "T2"]
    goods = [{"id": "P1", "length_m": 700, "kind": "goods"}, {"id": "P2", "length_m": 300, "kind": "platform"}]
    return Scenario.model_validate(
        {
            "schema_version": 2,
            "rules_version": RULES_VERSION,
            "scenario_id": "L-test",
            "seed": 0,
            "preset": "fixture",
            "shift_ticks": shift,
            "factors": sorted(factors),
            "network": {
                "terminals": [{"id": "T1", "tracks": goods}, {"id": "T2", "tracks": goods}],
                "stations": [
                    {"id": s, "major": True, "crew_station": False, "loops": lp}
                    for s, lp in zip(ids, station_loops, strict=True)
                ],
                "sections": [
                    {"id": f"{a}-{b}", "west": a, "east": b, "length_m": 5000, "line_speed_kmh": 100}
                    for a, b in zip(nodes, nodes[1:], strict=False)
                ],
            },
            "trains": sorted(trains, key=lambda t: t["id"]),
            "events": [],
        }
    )
