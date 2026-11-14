"""Small scenario builders shared by the hardening tests (mini3 variants from the T03 ticket)."""

from __future__ import annotations

import copy

from railway.sim.engine import Simulator
from railway.sim.model import Scenario
from tests.helpers import load_fixture

L6 = {
    "id": "L6",
    "type": "local",
    "direction": "east",
    "pax": 100,
    "entry_tick": 20,
    "stops": [
        {"station": "S1", "arr": 24, "dep": 25, "min_dwell": 1},
        {"station": "S2", "arr": 29, "dep": 30, "min_dwell": 1},
        {"station": "S3", "arr": 34, "dep": 35, "min_dwell": 1},
        {"station": "YE", "arr": 39, "dep": None, "min_dwell": 0},
    ],
    "next_service": None,
    "crew_used_at_entry": 0,
}
F22 = {
    "id": "F22",
    "type": "freight",
    "direction": "east",
    "pax": 0,
    "entry_tick": 30,
    "stops": [{"station": "YE", "arr": 54, "dep": None, "min_dwell": 0}],
    "next_service": None,
    "crew_used_at_entry": 0,
}


def mini3(*factors, events=(), extra_trains=(), **top) -> dict:
    data = load_fixture("mini3")
    data["factors"] = sorted(set(data["factors"]) | set(factors))
    data["events"] = sorted((copy.deepcopy(e) for e in events), key=lambda e: (e["reveal_tick"], e["id"]))
    data["trains"] = sorted(data["trains"] + [copy.deepcopy(t) for t in extra_trains], key=lambda t: t["id"])
    data.update(top)
    return data


def train(data: dict, tid: str) -> dict:
    return next(t for t in data["trains"] if t["id"] == tid)


def load(data: dict) -> Simulator:
    return Simulator.load(Scenario.model_validate(data))


def ts(sim: Simulator, tid: str):
    return next(t for t in sim.state().trains if t.id == tid)


def run_to(sim: Simulator, tick: int):
    """Advance (stopping at every decision point) until `tick` is reached."""
    while sim.tick < tick and not sim.done:
        sim.advance(max_tick=tick)
    assert sim.tick == tick
