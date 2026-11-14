"""v0.3 graph generator invariants (docs/design-changes-v0.3.md §4)."""

import pytest

from railway.gen.generator import generate
from railway.gen.graph_gen import _quickest, _terminal_peak
from railway.gen.presets import PRESETS
from railway.sim.engine import Simulator

GRAPH = ["T1", "T2", "T3", "T2-X"]


@pytest.mark.parametrize("preset", GRAPH)
@pytest.mark.parametrize("seed", range(6))
def test_graph_scenario_invariants(preset, seed):
    config = PRESETS[preset]
    sc = generate(config, seed)
    assert sc == generate(config, seed)  # deterministic
    Simulator.load(sc)
    net = sc.network
    assert len(net.stations) == config.stations
    assert len(net.terminals) == 2 + config.branches
    degree = {}
    for s in net.sections:
        degree[s.west] = degree.get(s.west, 0) + 1
        degree[s.east] = degree.get(s.east, 0) + 1
    junctions = [s for s in net.stations if degree[s.id] == 3]
    assert len(junctions) == config.branches + 2 * config.bypasses
    assert all(s.loops and s.major for s in junctions)
    ports = [t for t in net.terminals if all(track.kind == "goods" for track in t.tracks)]
    assert len(ports) == (1 if config.port else 0)
    raw = sc.model_dump(mode="json")
    trunk_middle = None
    trunk = sorted((s.id for s in net.stations if s.id.startswith("S")), key=lambda x: int(x[1:]))
    trunk_middle = {trunk[len(trunk) // 2 - 1], trunk[len(trunk) // 2]}
    crossing = 0
    for t in raw["trains"]:
        assert t["route"] == _quickest(raw["network"], t["type"], t["origin"], t["destination"])
        crossing += trunk_middle <= set(t["route"])
        if any(p.id in (t["origin"], t["destination"]) for p in ports):
            assert t["type"] == "freight"
        if t["type"] == "freight":
            continue
        # every passenger gets off exactly once
        assert sum(s["alight"] for s in t["stops"]) == t["pax"] + sum(s["board"] for s in t["stops"])
        assert all(s["headway"] > 0 for s in t["stops"])
    assert crossing >= 0.5 * len(raw["trains"])
    for term in net.terminals:
        for kind in ("platform", "goods"):
            n = sum(track.kind == kind for track in term.tracks)
            if n:
                assert _terminal_peak(raw["trains"], term.id, kind) <= max(1, n - 1)
    for s in net.stations:  # passenger branches have a passing place at every station
        branch_end = f"T{s.id[0]}"
        if s.id[0] in "ABCD" and branch_end not in {p.id for p in ports}:
            assert any(loop.kind == "platform" and loop.length_m >= 300 for loop in s.loops)
    by_id = {t.id: t for t in sc.trains}
    for t in sc.trains:
        if t.next_service:
            nxt = by_id[t.next_service]
            assert nxt.type == t.type and nxt.origin == t.destination
    for c in sc.connections:
        assert c.station in {j.id for j in junctions}


def test_presets_sizes():
    sizes = {name: (PRESETS[name].stations, PRESETS[name].trains, PRESETS[name].shift_ticks) for name in GRAPH}
    assert sizes == {"T1": (10, 10, 240), "T2": (16, 14, 720), "T3": (28, 28, 720), "T2-X": (24, 16, 720)}
