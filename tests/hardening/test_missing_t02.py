"""T02 ticket tests that the gate record lists as never written (T02-05, 06, 09, 10, 12, 15, 19, 25).

Written verbatim from docs/tickets/T02-simulator-core.md §5.
"""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from railway.sim.engine import Simulator
from railway.sim.model import Authorize, Scenario
from railway.sim.timing import project, travel_ticks
from tests.hardening.scenarios import load, run_to, ts
from tests.helpers import load_fixture


def mini3() -> dict:
    return load_fixture("mini3")


def _set(data, tid, **fields):
    t = next(x for x in data["trains"] if x["id"] == tid)
    t.update(fields)
    return data


def _t05_cases():
    def bad_id(d):
        d["trains"][0]["id"] = "X1"

    def type_mismatch(d):
        d["trains"][0]["type"] = "local"  # E1 with type local

    def stops_out_of_order(d):
        e1 = d["trains"][0]
        e1["stops"] = [
            {"station": "S3", "arr": 2, "dep": 3, "min_dwell": 1},
            {"station": "S2", "arr": 6, "dep": 7, "min_dwell": 1},
            {"station": "YE", "arr": 13, "dep": None, "min_dwell": 0},
        ]

    def last_not_destination(d):
        d["trains"][0]["stops"][-1]["station"] = "YW"

    def arr_after_dep(d):
        d["trains"][0]["stops"][0].update(arr=8, dep=7)

    def dep_null_mid(d):
        d["trains"][0]["stops"][0]["dep"] = None

    def negative_entry(d):
        d["trains"][0]["entry_tick"] = -1

    def freight_mid_stop(d):
        f = next(x for x in d["trains"] if x["id"] == "F21")
        f["stops"].insert(0, {"station": "S2", "arr": 14, "dep": 15, "min_dwell": 1})

    def duplicate_id(d):
        d["trains"].insert(1, copy.deepcopy(d["trains"][0]))

    return [
        bad_id,
        type_mismatch,
        stops_out_of_order,
        last_not_destination,
        arr_after_dep,
        dep_null_mid,
        negative_entry,
        freight_mid_stop,
        duplicate_id,
    ]


@pytest.mark.parametrize("mutate", _t05_cases(), ids=lambda f: f.__name__)
def test_T02_05_trains(mutate):
    data = mini3()
    mutate(data)
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_T02_06_rules_version():
    data = mini3()
    data["rules_version"] = "0.0-old"
    with pytest.raises(ValidationError, match="rules_version"):
        Scenario.model_validate(data)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((6000, 120, 140), 3),
        ((6000, 120, 100), 4),
        ((6000, 120, 70), 6),
        ((6000, 120, 140, 40), 9),
        ((1, 140, 140), 1),
        ((4500, 80, 100), 4),
        ((8000, 140, 140), 4),
        ((9500, 140, 70), 9),
    ],
)
def test_T02_09_travel_ticks(args, expected):
    result = travel_ticks(*args)
    assert result == expected
    assert type(result) is int


def test_T02_10_free_run_matches_fixture_timetable():
    """Free-run projection of the full route from entry gives exactly the mini3 timetable."""
    sim = load(mini3())
    expected = {
        "E1": {"S2": (6, 7), "YE": (13, None)},
        "L5": {"S3": (4, 5), "S2": (9, 10), "S1": (14, 15), "YW": (19, None)},
        "F21": {"YE": (26, None)},
    }
    for tid, stops in expected.items():
        d = sim._train_defs[tid]
        origin = "YW" if d.direction == "east" else "YE"
        dest = d.stops[-1].station
        path = sim._route(origin, dest, "main")
        intervals = project(sim, tid, path, d.entry_tick)
        # Arrival at each station = exit of the section before it; departure = entry of the next one.
        order = sim._station_ids if d.direction == "east" else list(reversed(sim._station_ids))
        arrivals = {}
        departures = {}
        for res, enter, exit_ in intervals:
            sec = sim._section_by_id[res]
            a, b = (sec.west, sec.east) if d.direction == "east" else (sec.east, sec.west)
            arrivals[b] = exit_
            departures[a] = enter
        for station, (arr, dep) in stops.items():
            assert arrivals[station] == arr, (tid, station, arrivals)
            if dep is not None:
                assert departures[station] == dep, (tid, station, departures)
        assert order[0] == origin


def test_T02_12_unknown_args():
    sim = load(mini3())
    cases = [
        (Authorize(train="X9", station="S2", track="L1"), "R-UNKNOWN-TRAIN"),
        (Authorize(train="E1", station="S9", track="main"), "R-UNKNOWN-STATION"),
        (Authorize(train="E1", station="S2", track="L7"), "R-UNKNOWN-TRACK"),
        (Authorize(train="E1", station="YE", track="L1"), "R-UNKNOWN-TRACK"),
    ]
    for action, rule in cases:
        result = sim.apply(action)
        assert not result.ok and result.rule.value == rule, (action, result.message)
        assert result.message.startswith(f"REFUSED {rule}: ")


def test_T02_15_loop_len():
    sim = load(mini3())
    result = sim.apply(Authorize(train="E1", station="S3", track="L1"))
    assert result.rule.value == "R-LOOP-LEN"
    assert result.message == "REFUSED R-LOOP-LEN: E1 is 200 m, S3/L1 is 150 m"
    assert sim.advance().tick == 2
    assert sim.apply(Authorize(train="F21", station="S1", track="L1")).rule.value == "R-LOOP-LEN"
    assert sim.apply(Authorize(train="F21", station="S2", track="L2")).ok


def test_T02_19_track_occupied():
    sim = load(mini3())
    assert sim.apply(Authorize(train="E1", station="S2", track="L1")).ok
    result = sim.apply(Authorize(train="L5", station="S2", track="L1"))
    assert result.message == "REFUSED R-TRACK-OCCUPIED: S2/L1 reserved by E1"


def test_T02_25_extension():
    sim = load(mini3())
    assert sim.apply(Authorize(train="E1", station="S2", track="L1")).ok
    first = sim.advance()
    assert first.tick == 2 and [(r.kind, r.train) for r in first.reasons] == [("READY", "F21")]
    sim.advance(max_tick=4)
    assert ts(sim, "E1").section == "S1-S2"
    # Not strictly beyond the current end.
    assert sim.apply(Authorize(train="E1", station="S2", track="main")).rule.value == "R-WRONG-DIRECTION"
    result = sim.apply(Authorize(train="E1", station="YE", track="main"))
    assert result.ok, result.message
    assert "S2/L1" not in sim._holders  # the old destination track is released at the extension
    assert "S2/L1" not in ts(sim, "E1").authority.path
    seen_l1 = False
    s2_arrival = None
    while not sim.done and ts(sim, "E1").phase != "finished":
        sim.advance(max_tick=sim.tick + 1)
        e1 = ts(sim, "E1")
        seen_l1 |= e1.station == "S2" and e1.track == "L1"
        if e1.station == "S2":
            assert e1.track == "main"
        if s2_arrival is None and any(a.station == "S2" for a in e1.arrivals):
            s2_arrival = next(a.actual for a in e1.arrivals if a.station == "S2")
    e1 = ts(sim, "E1")
    assert not seen_l1
    assert s2_arrival == 6
    assert e1.finished_tick == 13
    assert [(a.station, a.sched, a.actual) for a in e1.arrivals] == [("S2", 6, 6), ("YE", 13, 13)]


def test_T02_25_extension_departs_s2_at_dep_not_before():
    sim = load(mini3())
    sim.apply(Authorize(train="E1", station="S2", track="L1"))
    sim.advance()
    sim.advance(max_tick=4)
    sim.apply(Authorize(train="E1", station="YE", track="main"))
    run_to(sim, 7)
    e1 = ts(sim, "E1")
    assert e1.section == "S2-S3" and e1.section_enter == 7


def test_T02_unknown_station_in_simulator_is_refused_not_raised():
    """Robustness: every refusal path returns an ActionResult (no exception)."""
    sim = Simulator.load(Scenario.model_validate(mini3()))
    for action in ({"kind": "authorize", "train": "E1"}, {"kind": "hold", "train": "E1", "until_tick": "x"}):
        result = sim.apply(action)
        assert result.rule.value == "R-BAD-ARGS"
