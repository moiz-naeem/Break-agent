"""Shared cached fixtures for `tests/net` (T1 generation takes ~7 s, so build once per process)."""

from __future__ import annotations

from functools import cache

import pytest

from railway.net.model import Call, Timetable, Trip
from railway.net.templates import sample_trips, t1_services, trunk_and_branches
from railway.net.timetable import generate
from railway.net.trains import TYPES

NET = trunk_and_branches()


@cache
def build_t1(seed: int = 1, shift_min: int = 180) -> Timetable:
    core, fill = t1_services()
    return generate(NET, core, fill, shift_min, seed, "T1")


@cache
def build_sample_timetable(shift_min: int = 180) -> Timetable:
    return Timetable(net=NET, trips=tuple(sample_trips()), shift_min=shift_min, seed=0, tier="T1")


@cache
def build_stranded_timetable() -> Timetable:
    """IC9 is booked into 200 m platform N1 (too short for 260 m IC), raising UNROUTABLE until re-platformed."""
    trips = (
        Trip(
            train=TYPES["IC"].spec("IC9"),
            calls=(Call(platform="A1", depart=5), Call(platform="N1", arrive=30)),
        ),
        Trip(
            train=TYPES["S"].spec("L8"),
            calls=(Call(platform="A2", depart=5), Call(platform="B1", arrive=30)),
        ),
    )
    return Timetable(net=NET, trips=trips, shift_min=120, seed=0, tier="T1")


@pytest.fixture(scope="session")
def t1_timetable() -> Timetable:
    return build_t1(seed=1, shift_min=180)


@pytest.fixture(scope="session")
def sample_timetable() -> Timetable:
    return build_sample_timetable()


@pytest.fixture(scope="session")
def stranded_timetable() -> Timetable:
    return build_stranded_timetable()
