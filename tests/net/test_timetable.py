from functools import cache

import pytest

from railway.net.engine import TURNAROUND_MIN, Engine
from railway.net.interlocking import Interlocking, admissible, walk
from railway.net.model import Call, Connection, Timetable, Trip
from railway.net.templates import t1_services, trunk_and_branches
from railway.net.timetable import (
    HIGH,
    ServiceGroup,
    _dwells,
    generate,
    infeasible,
    leg_minutes,
    legs,
    peak_utilisation,
    plan_run,
    utilisation,
)
from railway.net.trains import TYPES

NET = trunk_and_branches()
SHIFT = 180


@cache
def t1(seed=1):
    core, fill = t1_services()
    return generate(NET, core, fill, SHIFT, seed, "T1")


def trip(train_id, stops, type_code="S", **kw):
    calls = tuple(Call(platform=p, arrive=a, depart=d, min_dwell=w) for p, a, d, w in stops)
    return Trip(train=TYPES[type_code].spec(train_id), calls=calls, **kw)


def entering(train_id, stops, type_code="S", **kw):
    return trip(train_id, stops, type_code, entry_node="PW", **kw)


# ---------------------------------------------------------------- the generated shift


def test_the_generated_shift_runs_to_its_own_book():
    """The point of the feasibility check: with nothing in the way, every train makes every call on time."""
    timetable = t1()
    assert infeasible(timetable) == []
    assert 12 <= len(timetable.trips) <= 35
    assert all(mv.phase == "done" for mv in plan_run(NET, list(timetable.trips), SHIFT).trains.values())


def test_the_shift_is_busy_enough_to_need_an_agent():
    """Utilisation aims at §1.2's band; T1's single-track branch caps it, so the test pins the bottleneck."""
    timetable = t1()
    peak = peak_utilisation(NET, timetable.trips, SHIFT)
    assert 0.55 <= peak <= HIGH
    assert utilisation(NET, timetable.trips, SHIFT)["JA-N"] == peak  # the branch is what the service is built round
    engine = Engine(NET, list(timetable.trips), SHIFT)
    raised = 0
    for _ in range(200):
        decision = engine.advance()
        raised += len(decision.conflicts)
        if decision.kind == "SHIFT_END":
            break
        for conflict in decision.conflicts:
            engine.order(conflict.key, sorted(conflict.trains))
    assert raised >= 25  # real contention before a single disruption has been added


def test_every_kind_of_traffic_runs():
    codes = {t.train.id[:2] for t in t1().trips}
    assert {"IC", "SE", "SW"} <= codes
    assert any(not t.train.passenger for t in t1().trips), "no freight path survived"


def test_the_same_seed_gives_the_same_shift_and_another_seed_gives_a_different_one():
    assert generate(NET, *t1_services(), SHIFT, 3, "T1") == t1(3)  # a fresh run, not the cached object
    assert t1(3) != t1(4)
    assert [t.entry_tick for t in t1(3).trips] != [t.entry_tick for t in t1(4).trips]


# ---------------------------------------------------------------- rotations and connections


def test_the_intercity_turns_at_the_terminal_and_the_return_is_the_same_unit():
    """A rotation (§4.1): one trip with a turnaround dwell, so a late arrival carries into the return service."""
    ic = next(t for t in t1().trips if t.train.id.startswith("IC"))
    assert [c.platform for c in ic.calls] == ["A1", "C1", "B2"]  # out on the up line, back on the down
    assert next(c for c in ic.calls if c.platform == "C1").min_dwell >= TURNAROUND_MIN
    assert ic.exit_node == "PW"


def test_connections_name_two_real_services_with_a_workable_gap():
    timetable = t1()
    assert timetable.connections
    trips = {t.train.id: t for t in timetable.trips}
    for link in timetable.connections:
        feeder, connector = trips[link.feeder], trips[link.connector]
        arrive = next(c.arrive for c in feeder.calls if NET.element(c.platform).node == link.node)
        depart = next(c.depart for c in connector.calls if NET.element(c.platform).node == link.node)
        assert depart is not None and depart > arrive and link.transfers > 0


# ---------------------------------------------------------------- passenger counts


def test_passengers_fill_at_the_origin_and_are_all_off_at_the_destination():
    for service in t1().trips:
        counts = [(c.alight, c.board) for c in service.calls]
        if not service.train.passenger:
            assert counts == [(0, 0)] * len(counts)
            continue
        if len(counts) == 1:  # a service that only calls once books the turnover at that call
            assert counts[0][0] == counts[0][1] > 0
            continue
        assert counts[0][0] == 0 and counts[0][1] > 0
        assert counts[-1][1] == 0 and counts[-1][0] > 0
        assert counts[0][1] == counts[-1][0]  # it neither gains nor loses passengers over the journey
        low, high = TYPES[{"IC": "IC", "RE": "RE", "SE": "S", "SW": "S"}[service.train.id[:2]]].load
        assert low <= counts[0][1] <= high


# ---------------------------------------------------------------- utilisation


def test_a_single_track_counts_its_whole_transit_and_a_one_way_track_only_its_critical_block():
    loads = utilisation(NET, (entering("L1", [("A1", 9, 10, 2), ("N1", 40, None, 1)]),), 100)
    assert loads["JA-N"] == (6 + 2) / 100  # bidirectional: the whole 10 km transit is locked out
    assert loads["A-JA.u"] == (1 + 2) / 100  # one-way: followers come up block by block


def test_conflicting_moves_share_the_time_one_of_them_is_set():
    loads = utilisation(NET, (entering("L1", [("A1", 9, 10, 2), ("N1", 40, None, 1)]),), 100)
    assert loads["JA.m3"] == loads["JA.m2"]  # the flat junction crossing is unavailable just as long
    assert loads["A.e3"] == loads["A.e1"]  # and so is its neighbour in the A throat


def test_a_terminating_train_holds_its_platform_for_the_rest_of_the_shift():
    freight = Trip(
        train=TYPES["FB"].spec("F1"),
        calls=(Call(platform="B3", arrive=20, depart=24, min_dwell=4), Call(platform="P1", arrive=60)),
        entry_node="PW",
    )
    assert utilisation(NET, (freight,), 120)["P1"] == pytest.approx((1 + 120 - 60) / 120)  # dwell, then parked


# ---------------------------------------------------------------- run times and the exit run


def test_leg_minutes_matches_what_the_engine_really_takes():
    service = entering("L1", [("A1", 9, 10, 2), ("N1", 40, 42, 2)], exit_node="PN")
    paths = legs(NET, service)
    assert [p[0] for p in paths] == ["PW-A.u", "A1", "N1"]
    movement = plan_run(NET, [service], 200).trains["L1"]
    assert movement.visits[0].arrived == leg_minutes(NET, paths[0], service.train)
    assert (movement.phase, movement.position) == ("done", "PN")


def test_a_train_that_leaves_the_area_gives_everything_back():
    first = entering("L1", [("A1", 9, 10, 2), ("N1", 40, 42, 2)], exit_node="PN")
    second = entering("L2", [("A1", 29, 30, 2), ("N1", 60, None, 1)], entry_tick=20)
    engine = plan_run(NET, [first, second], 200)
    assert [engine.trains[t].position for t in ("L1", "L2")] == ["PN", "N1"]
    assert engine.il.holder == {"N1": "L2"} and engine.il.directions == {}  # L1 kept nothing, L2 stands in N1


# ---------------------------------------------------------------- the rules step 3 leans on


def test_a_turnaround_dwell_lets_a_train_leave_the_way_it_came_and_a_short_one_does_not():
    stops = [("A1", 9, 11, 2), ("C1", 45, 45 + TURNAROUND_MIN, TURNAROUND_MIN), ("B2", 99, 100, 1)]
    turned = plan_run(NET, [entering("T1", stops, "RE", exit_node="PW")], 220).trains["T1"]
    assert (turned.phase, turned.position) == ("done", "PW")
    short = [(p, a, d, 1 if w == TURNAROUND_MIN else w) for p, a, d, w in stops]
    stuck = plan_run(NET, [entering("T2", short, "RE", exit_node="PW")], 220).trains["T2"]
    assert (stuck.phase, stuck.position) == ("berthed", "C1")


def test_a_train_held_by_the_one_ahead_does_not_take_the_junction_in_front_of_it():
    """Without this the pair deadlocks: the follower locks points its leader still needs to clear the section."""
    slow = Trip(
        train=TYPES["FB"].spec("F1"),
        calls=(Call(platform="A3", arrive=11, depart=12, min_dwell=1), Call(platform="N2", arrive=99)),
        entry_node="PW",
    )
    fast = entering("IC1", [("A1", 9, 13, 2), ("C1", 99, None, 1)], "IC", entry_tick=2)
    engine = plan_run(NET, [slow, fast], 220)
    assert [engine.trains[t].position for t in ("F1", "IC1")] == ["N2", "C1"]
    assert [engine.trains[t].phase for t in ("F1", "IC1")] == ["done", "done"]


def test_a_train_may_not_run_through_a_platform_it_does_not_fit():
    """A 740 m freight cannot pass A1, so it is routed over the through road A3 instead."""
    freight = Trip(train=TYPES["FB"].spec("F1"), calls=(Call(platform="B3", arrive=40, depart=44),), entry_node="PW")
    first = legs(NET, freight)[0]
    assert "A3" in first and "A1" not in first


# ---------------------------------------------------------------- refusals


def test_a_timetable_that_cannot_be_run_says_why():
    too_tight = Timetable(net=NET, trips=(entering("L1", [("A1", 3, 4, 1), ("N1", 8, None, 1)]),), shift_min=120)
    assert infeasible(too_tight) == ["L1 runs 17 min late with nothing in its way"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stops": ("A1", "N1"), "dwell": (2,)}, "one dwell per stop"),
        ({"code": "S1"}, "may not end in a digit"),
    ],
)
def test_a_broken_service_group_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ServiceGroup(**{"code": "X", "type": "S", "entry_node": "PW", "stops": ("A1",), "dwell": (2,), **kwargs})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"type": "nope"}, "no train type"),
        ({"stops": ("C1",)}, "cannot run"),  # a 160 m regional express cannot reach C1 from the branch portal
        ({"entry_node": "A"}, "no track leads into the network from A"),
        ({"exit_node": "B"}, "no track leads out of the network to B"),
    ],
)
def test_a_service_group_the_network_cannot_run_says_which_group(kwargs, message):
    group = ServiceGroup(
        **{"code": "X", "type": "RE", "entry_node": "PN", "stops": ("N1",), "dwell": (2,), "exit_node": "PN", **kwargs}
    )
    with pytest.raises(ValueError, match=message):
        generate(NET, [group], [], SHIFT, 1, "T1")


@pytest.mark.parametrize(
    ("platform", "links", "message"),
    [
        ("zz", (), "not a platform"),
        ("A1", (("L1", "L9"),), "unknown train L9"),
        ("A1", (("L1", "L1"),), "cannot feed itself"),
    ],
)
def test_a_broken_timetable_is_refused(platform, links, message):
    with pytest.raises(ValueError, match=message):
        Timetable(
            net=NET,
            trips=(entering("L1", [(platform, 9, 10, 1), ("N1", 40, None, 1)]),),
            connections=tuple(Connection(feeder=f, connector=c, node="N", transfers=5) for f, c in links),
            shift_min=120,
        )


# ---------------------------------------------------------------- what review found


def test_a_long_shift_is_booked_to_a_fixed_point():
    """Publishing new departures reorders the next run, so the generator rebooks until the book holds (§4.1)."""
    core, fill = t1_services()
    for shift in (240, 360):
        timetable = generate(NET, core, fill, shift, 1, "T1")
        assert infeasible(timetable) == [], f"shift {shift}"


def test_a_filler_path_never_costs_the_service_a_passenger_train():
    """§4.1: a real timetable gives up its freight paths first, not the commuters."""
    core, fill = t1_services()
    without = {t.train.id for t in generate(NET, core, [], SHIFT, 1, "T1").trips}
    with_fill = {t.train.id for t in t1().trips}
    assert without <= with_fill
    assert any(not t.train.passenger for t in t1().trips)


def test_a_train_cannot_start_in_a_platform_it_does_not_fit():
    """It would stand there for the shift with no route the interlocking could ever grant, and never be reported."""
    freight = Trip(
        train=TYPES["FB"].spec("F1"),  # 740 m
        calls=(Call(platform="A1", depart=5), Call(platform="B3", arrive=40)),
    )
    with pytest.raises(ValueError, match="cannot stand in A1"):
        Engine(NET, [freight], 120)


def test_a_train_may_always_leave_the_platform_it_stands_in():
    """The length rule is about roads a train runs onto; applying it to its own position would strand it."""
    long_freight = TYPES["FB"].spec("F1")
    standing = Interlocking(NET).lock(long_freight, "B", ["B3", "B.e2", "B-JB.u"])
    assert standing is None
    assert admissible(long_freight, walk(NET, "B", ["B3", "B.e2", "B-JB.u"])) is None


def test_a_booked_dwell_is_charged_once_however_many_legs_touch_the_platform():
    freight = Trip(
        train=TYPES["FB"].spec("F1"),
        calls=(Call(platform="B3", arrive=20, depart=24, min_dwell=4), Call(platform="P1", arrive=60)),
        entry_node="PW",
    )
    assert utilisation(NET, (freight,), 120)["B3"] == pytest.approx(4 / 120)  # the dwell, not twice the dwell


def test_a_trip_that_cannot_be_routed_is_not_measured_as_using_nothing():
    nowhere = Trip(train=TYPES["FB"].spec("Fz"), calls=(Call(platform="C1", arrive=10, depart=12),), entry_node="PW")
    with pytest.raises(ValueError, match="cannot be routed"):
        utilisation(NET, (nowhere,), 100)


def test_the_peak_adds_a_minute_but_never_turns_a_stop_into_a_turnaround():
    """A 7-minute stop bumped to 8 would silently let the peak service reverse where the off-peak one cannot."""
    group = ServiceGroup(
        code="X", type="RE", entry_node="PW", exit_node="PW", stops=("A1", "C3", "B2"), dwell=(2, TURNAROUND_MIN - 1, 2)
    )
    assert _dwells(group, in_peak=False) == [2, TURNAROUND_MIN - 1, 2]
    assert _dwells(group, in_peak=True) == [3, TURNAROUND_MIN - 1, 3]


@pytest.mark.parametrize(
    ("stops", "kwargs", "message"),
    [
        ((("A1", 9, None, 1), ("N1", 40, None, 1)), {"entry_node": "PW"}, "it leaves A1"),
        ((("A1", None, 9, 1), ("N1", None, None, 1)), {}, "it arrives at N1"),
        ((("N2", 9, None, 1),), {"entry_node": "PN", "exit_node": "PW"}, "it leaves N2"),
        ((("N2", None, 9, 1),), {"entry_node": "PN"}, "it arrives at N2"),
    ],
)
def test_a_trip_needs_a_time_at_every_call_it_uses(stops, kwargs, message):
    """A one-call service that runs through the area needs both times at that call; a terminating one does not."""
    calls = tuple(Call(platform=p, arrive=a, depart=d, min_dwell=w) for p, a, d, w in stops)
    with pytest.raises(ValueError, match=message):
        Trip(train=TYPES["S"].spec("L1"), calls=calls, **kwargs)


def test_a_one_call_service_that_runs_straight_through_is_accepted():
    through = Trip(
        train=TYPES["S"].spec("L1"),
        calls=(Call(platform="N2", arrive=9, depart=11, min_dwell=2),),
        entry_node="PN",
        exit_node="PW",
    )
    assert plan_run(NET, [through], 120).trains["L1"].phase == "done"
