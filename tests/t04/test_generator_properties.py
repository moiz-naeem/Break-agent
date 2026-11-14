import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from railway.gen.generator import generate
from railway.gen.presets import PRESETS
from railway.sim.constants import TURNAROUND
from railway.sim.timing import travel_ticks


@pytest.mark.parametrize("preset", ["T1", "T2", "T2-S14", "T3"])
@settings(max_examples=50, deadline=None)
@given(seed=st.integers(min_value=0, max_value=999_999))
def test_T04_16_to_24_generated_invariants(preset, seed):
    config = PRESETS[preset]
    scenario = generate(config, seed)
    stations = scenario.network.stations
    assert len(stations) == config.stations
    assert len(scenario.trains) == config.trains
    assert len(scenario.network.sections) == config.stations + 1
    assert all(len(station.loops) <= 3 for station in stations)
    assert sum(any(loop.length_m == 700 for loop in station.loops) for station in stations) >= math.ceil(
        config.stations / 3
    )
    assert all(stations[i].loops or stations[i + 1].loops for i in range(len(stations) - 1))
    for third in range(3):
        lo = third * len(stations) // 3
        hi = max(lo + 1, (third + 1) * len(stations) // 3)
        assert any(any(loop.length_m == 700 for loop in stations[i].loops) for i in range(lo, hi))
    assert all(train.type != "engineering" for train in scenario.trains)
    for kind, share in config.mix.items():
        assert abs(sum(train.type == kind for train in scenario.trains) - share * config.trains) <= 1
    assert (
        abs(
            sum(train.direction == "east" for train in scenario.trains)
            - sum(train.direction == "west" for train in scenario.trains)
        )
        <= 1
    )

    ids = {train.id for train in scenario.trains}
    successors = []
    for train in scenario.trains:
        assert train.stops[-1].arr + train.entry_tick <= config.shift_ticks - 15
        assert train.stops[-1].station == ("YE" if train.direction == "east" else "YW")
        if train.next_service:
            successor = next(other for other in scenario.trains if other.id == train.next_service)
            successors.append(successor.id)
            assert successor.direction != train.direction
            assert successor.entry_tick >= train.entry_tick + train.stops[-1].arr + TURNAROUND
            assert successor.id in ids
        if "F6" not in config.factors:
            assert train.next_service is None
        if "F10" not in config.factors:
            assert train.crew_used_at_entry == 0
        if train.type == "express":
            assert 150 <= train.pax <= 400 and train.pax % 10 == 0
            assert all(
                stop.station in {station.id for station in stations if station.major} for stop in train.stops[:-1]
            )
        elif train.type == "local":
            assert 50 <= train.pax <= 200 and train.pax % 10 == 0
            assert [stop.station for stop in train.stops[:-1]] == [
                station.id for station in (stations if train.direction == "east" else list(reversed(stations)))
            ]
        else:
            assert train.pax == 0
        free = sum(
            travel_ticks(
                section.length_m, section.line_speed_kmh, {"express": 140, "local": 100, "freight": 70}[train.type]
            )
            for section in scenario.network.sections
        )
        assert train.stops[-1].arr >= free

    assert len(successors) == len(set(successors))
    if "F10" in config.factors:
        crew_stations = [station for station in stations if station.crew_station]
        assert 2 <= len(crew_stations) <= 3
        assert all(station.loops for station in crew_stations)
    trains_by_id = {train.id: train for train in scenario.trains}
    kinds_by_factor = {"F7": "possession", "F8": "speed_restriction", "F15": "breakdown", "F16": "late_entry"}
    for factor, kind in kinds_by_factor.items():
        of_kind = [event for event in scenario.events if event.kind == kind]
        if factor not in config.factors:
            assert not of_kind
        else:
            assert of_kind
    for event in scenario.events:
        if event.kind == "possession":
            lookahead = event.start - event.reveal_tick
            assert 0 <= event.reveal_tick <= event.start < event.end <= config.shift_ticks
            assert lookahead == 0 or 45 <= lookahead <= 60
        elif event.kind == "speed_restriction":
            lookahead = event.start - event.reveal_tick
            assert event.reveal_tick <= event.start < event.end
            assert 0 <= event.speed_kmh < 140
            assert lookahead == 0 or 30 <= lookahead <= 60
            assert 30 <= event.end - event.start <= 90
        elif event.kind == "late_entry":
            target = trains_by_id[event.train]
            assert target.entry_tick >= 20
            assert event.reveal_tick == target.entry_tick - 20
            assert event.new_entry_tick > target.entry_tick
        elif event.kind == "breakdown":
            target = trains_by_id[event.train]
            assert target.entry_tick + 5 <= event.tick <= target.entry_tick + target.stops[-1].arr - 5


def test_T04_20_factor_off_has_no_crew_station_or_events():
    scenario = generate(PRESETS["T1"], 5)
    assert not any(station.crew_station for station in scenario.network.stations)
    assert not scenario.events
