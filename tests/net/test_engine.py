import pytest

from railway.net.engine import Engine
from railway.net.model import Call, Trip
from railway.net.rules import NetRule
from railway.net.templates import sample_trips, trunk_and_branches
from railway.net.trains import TYPES

NET = trunk_and_branches()


def engine(trips=None, shift=180):
    return Engine(NET, trips or sample_trips(), shift)


def trip(train_id, calls, code="S", **kw):
    return Trip(train=TYPES[code].spec(train_id), calls=calls, **kw)


def run(eng, first=None, answer=True, limit=40):
    """Advance to the end, optionally answering every conflict with `first` ahead of the rest."""
    seen = []
    for _ in range(limit):
        decision = eng.advance()
        seen.append(decision)
        if decision.kind == "SHIFT_END":
            return seen
        if decision.kind == "CONFLICT" and answer:
            for conflict in decision.conflicts:
                assert eng.order(conflict.key, [first, *(t for t in conflict.trains if t != first)]) is None
    raise AssertionError(f"did not finish: {[(d.tick, d.kind) for d in seen]}")


def delays(eng):
    return {row.train: row.delay for row in eng.state()}


def test_a_train_runs_to_its_booked_times():
    eng = engine()
    run(eng, first="IC1")
    assert [row[:3] for row in eng.state()] == [
        ("F3", "done", "P1"),
        ("IC1", "done", "C1"),
        ("L4", "done", "N1"),
    ]


def test_the_flat_junction_raises_one_conflict_naming_both_trains():
    eng = engine()
    decision = eng.advance()
    assert (decision.kind, decision.trains) == ("CONFLICT", ("IC1", "L4"))
    assert decision.tick <= 10  # raised before they are due away, inside the 15-minute lookahead
    # the shared up-line block, and the two throat moves that cross at A
    assert {c.key for c in decision.conflicts} == {("A-JA.u", 0), ("A.e1", "A.e3")}
    assert all(c.trains == ("IC1", "L4") for c in decision.conflicts)


def test_crossing_moves_conflict_even_though_they_share_no_block():
    """An up train turning onto the branch (JA.m3) crosses a down train (JA.m2); they claim nothing in common."""
    trips = [
        trip("UP", (Call(platform="A1", depart=7), Call(platform="N1", arrive=60))),
        trip("DOWN", (Call(platform="B2", depart=0), Call(platform="A2", arrive=60))),
    ]
    eng = engine(trips, shift=200)
    keys = []
    for decision in run(eng, first="UP"):
        keys += [c.key for c in decision.conflicts]
        if decision.kind == "CONFLICT":
            assert decision.trains == ("DOWN", "UP")
    assert ("JA.m2", "JA.m3") in keys  # the agent is asked, even with no shared block
    assert [row.phase for row in eng.state()] == ["done", "done"]


def test_a_following_train_keeps_station_instead_of_waiting_for_the_far_end():
    """Block signalling: S2 follows S1 up the trunk and only loses a couple of minutes."""
    trips = [
        trip("S1", (Call(platform="A1", depart=0), Call(platform="C1", arrive=40))),
        trip("S2", (Call(platform="A2", depart=3), Call(platform="C2", arrive=43))),
    ]
    eng = engine(trips)
    run(eng, first="S1")
    assert delays(eng)["S2"] <= 5
    assert [row.phase for row in eng.state()] == ["done", "done"]


def test_a_train_holds_only_the_section_it_is_in():
    eng = engine()
    eng.advance()
    eng.order(("A-JA.u", 0), ["IC1", "L4"])
    eng.order(("A.e1", "A.e3"), ["IC1", "L4"])
    eng.advance()
    held = [train for train in eng.il.holder.values() if train == "IC1"]
    assert len(held) <= 4  # a section, not the whole 33 km journey


@pytest.mark.parametrize("first", ["IC1", "L4"])
def test_order_decides_who_waits(first):
    eng = engine()
    run(eng, first=first)
    expected = {"IC1": 2, "L4": 0} if first == "IC1" else {"IC1": 6, "L4": -4}
    assert {k: v for k, v in delays(eng).items() if k != "F3"} == expected


def test_a_partial_order_is_enough_to_settle_a_conflict():
    """order() promises that trains left out queue behind those listed, so naming one must not loop."""
    eng = engine()
    seen = []
    for _ in range(10):
        decision = eng.advance()
        seen.append(decision.kind)
        if decision.kind == "SHIFT_END":
            break
        if decision.kind == "CONFLICT":
            for conflict in decision.conflicts:
                eng.order(conflict.key, ["IC1"])  # only one of the two trains
    assert seen.count("CONFLICT") == 1 and seen[-1] == "SHIFT_END"
    assert delays(eng)["IC1"] == 2


def test_holding_a_train_delays_it_and_is_refused_once_it_runs():
    eng = engine()
    eng.advance()
    assert eng.hold("L4", 40) is None
    assert eng.hold("L4", 5).rule == NetRule.HOLD_PAST
    assert eng.hold("nope", 40).rule == NetRule.UNKNOWN_TRAIN
    run(eng, first="IC1")
    assert delays(eng)["L4"] > 20  # it left at 40 instead of 10
    assert eng.hold("L4", 200).rule == NetRule.SHIFT_ENDED


def test_platform_sends_a_train_to_another_face():
    eng = engine()
    eng.advance()
    assert eng.platform("IC1", "C2") is None
    run(eng, first="IC1")
    assert ("IC1", "done", "C2") in [row[:3] for row in eng.state()]


@pytest.mark.parametrize(
    ("train", "target", "rule"),
    [
        ("IC1", "C9", NetRule.NO_ROUTE),
        ("IC1", "A-JA.u", NetRule.NO_ROUTE),
        ("IC1", "N1", NetRule.NO_ROUTE),  # too short for a 260 m intercity
        ("L4", "A2", NetRule.NO_ROUTE),  # already standing there
        ("nope", "C1", NetRule.UNKNOWN_TRAIN),
    ],
)
def test_platform_refusals(train, target, rule):
    eng = engine()
    eng.advance()
    assert eng.platform(train, target).rule == rule


def test_actions_are_refused_once_the_shift_is_over():
    eng = engine()
    run(eng, first="IC1")
    assert eng.order(("A-JA.u", 0), ["IC1"]).rule == NetRule.SHIFT_ENDED
    assert eng.platform("IC1", "C2").rule == NetRule.SHIFT_ENDED


def stranded_trips():
    """IC9 is booked into N1, which is too short for a 260 m intercity."""
    return [
        trip("IC9", (Call(platform="A1", depart=5), Call(platform="N1", arrive=30)), code="IC"),
        trip("L8", (Call(platform="A2", depart=5), Call(platform="B1", arrive=30))),
    ]


def test_a_train_with_nowhere_to_go_is_reported_once():
    eng = engine(stranded_trips())
    first = eng.advance()
    assert (first.kind, first.trains) == ("UNROUTABLE", ("IC9",))
    assert [d.kind for d in run(eng, first="L8")].count("UNROUTABLE") == 0
    assert ("IC9", "berthed", "A1") in [row[:3] for row in eng.state()]


def test_re_platforming_frees_a_stranded_train():
    eng = engine(stranded_trips())
    assert eng.advance().kind == "UNROUTABLE"
    assert eng.platform("IC9", "C1").rule == NetRule.NO_ROUTE
    assert eng.platform("IC9", "N2") is None
    run(eng, first="IC9")
    assert ("IC9", "done", "N2") in [row[:3] for row in eng.state()]


def test_a_deadlock_is_reported_once_and_then_the_shift_runs_out():
    eng = engine([trip("L1", (Call(platform="A1", depart=5), Call(platform="A1", arrive=30)))], shift=120)
    assert [(d.tick, d.kind) for d in run(eng)] == [(5, "UNROUTABLE"), (5, "BLOCKED"), (120, "SHIFT_END")]


def test_a_train_may_not_turn_back_through_a_platform_it_called_at():
    """It enters A2 from the east, so it cannot leave east again towards B."""
    calls = (Call(platform="N1", depart=0), Call(platform="A2", arrive=25, depart=26), Call(platform="B1", arrive=70))
    eng = engine([trip("L9", calls)], shift=200)
    assert [d.kind for d in run(eng)][:2] == ["UNROUTABLE", "BLOCKED"]
    assert ("L9", "berthed", "A2") in [row[:3] for row in eng.state()]


def test_a_quiet_shift_still_looks_in_and_never_passes_the_end():
    eng = engine([trip("L7", (Call(platform="A2", depart=300), Call(platform="N1", arrive=330)))], shift=100)
    assert [(d.tick, d.kind) for d in run(eng)] == [(30, "IDLE"), (60, "IDLE"), (90, "IDLE"), (100, "SHIFT_END")]


@pytest.mark.parametrize(
    ("trips", "message"),
    [
        ([trip("Z1", (Call(platform="Z9", depart=5), Call(platform="N1", arrive=30)))], "not a platform"),
        (
            [
                trip("R1", (Call(platform="A2", depart=5), Call(platform="N1", arrive=30))),
                trip("R1", (Call(platform="N1", depart=40), Call(platform="A2", arrive=70))),
            ],
            "a unit's whole day is one trip",
        ),
        (
            [
                trip("D1", (Call(platform="A1", depart=5), Call(platform="C1", arrive=40))),
                trip("D2", (Call(platform="A1", depart=6), Call(platform="C2", arrive=40))),
            ],
            "both start in A1",
        ),
    ],
)
def test_a_broken_timetable_is_refused_at_once(trips, message):
    with pytest.raises(ValueError, match=message):
        engine(trips)


def test_nothing_is_left_locked_except_where_the_trains_stand():
    eng = engine()
    run(eng, first="IC1")
    assert eng.il.holder == {"P1": "F3", "C1": "IC1", "N1": "L4"}
    assert eng.il.directions == {}


def test_two_identical_runs_agree():
    one, two = engine(), engine()
    assert [(d.tick, d.kind, d.conflicts) for d in run(one, first="IC1")] == [
        (d.tick, d.kind, d.conflicts) for d in run(two, first="IC1")
    ]
    assert one.state() == two.state()
