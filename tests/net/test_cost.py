"""The v0.3 network cost on hand-built event logs. Expected numbers are worked by hand in the comments.

f(L) = max(0, L-2) + 5[L>=15] + 10[L>=30] + 20[L>=60]: f(9)=7, f(10)=8, f(20)=23, f(60)=93, f(120)=153.
"""

import pytest

from railway.net.cost import (
    COMPONENTS,
    CostBreakdown,
    _Scenario,
    classify,
    cost,
    itemise,
    lateness_curve,
)
from railway.net.events import Event, EventLog
from railway.net.model import Call, Connection, Timetable, TrainSpec, Trip
from railway.net.templates import trunk_and_branches
from railway.net.trains import TYPES

NET = trunk_and_branches()  # PW - A - JA - B - JB - C, with JA - N - PN and JB - PORT


def trip(train_id, code, calls, **kw):
    return Trip(train=TYPES[code].spec(train_id), calls=calls, **kw)


# Express, v = 1.5. Heads PW -> A -> B -> C (terminates).
IC01 = trip("IC01", "IC", [
    Call(platform="A1", arrive=10, depart=12, alight=100, board=50),
    Call(platform="B1", arrive=30, depart=32, alight=200, board=40),
    Call(platform="C1", arrive=50, alight=300),
], entry_node="PW")
# Local, v = 1.0, the same way 9-10 minutes behind IC01.
S01 = trip("S01", "S", [
    Call(platform="A2", arrive=20, depart=21, alight=40, board=60),
    Call(platform="B2", arrive=40, depart=41, alight=30, board=20),
    Call(platform="C2", arrive=60, alight=80),
], entry_node="PW")
S02 = trip("S02", "S", [Call(platform="B2", arrive=60, depart=61, board=10), Call(platform="C2", arrive=75, alight=10)],
           entry_node="PW")
S03 = trip(
    "S03", "S", [Call(platform="A2", arrive=249, depart=250, board=100), Call(platform="B2", arrive=270, alight=100)],
    entry_node="PW",
)
# The other way: starts standing at C, runs C -> B -> A -> out at PW.
S99 = trip("S99", "S", [
    Call(platform="C3", depart=25, board=50),
    Call(platform="B1", arrive=45, depart=46, alight=10, board=10),
    Call(platform="A1", arrive=60, depart=61, alight=50),
], exit_node="PW")
FB01 = trip("FB01", "FB", [Call(platform="A3", arrive=15, depart=16), Call(platform="P1", arrive=70)], entry_node="PW")
FI01 = trip("FI01", "FI", [Call(platform="A3", arrive=30, depart=31)], entry_node="PW", exit_node="PN")
E01 = trip("E01", "S", [Call(platform="A2", arrive=100, depart=101, alight=50), Call(platform="B2", arrive=120)],
           entry_node="PW")  # empty stock, by `types`

TT = Timetable(
    net=NET, trips=(IC01, S01, S02, S03, S99, FB01, FI01, E01), shift_min=300,
    connections=(Connection(feeder="IC01", connector="S01", node="B", transfers=50),),
)
TYPES_OVERRIDE = {"E01": "ECS"}


def ev(tick, kind, **kw):
    return Event(tick=tick, kind=kind, **kw)


def bill(events, tt=TT, **kw):
    return cost(tt, events, types=TYPES_OVERRIDE if tt is TT else None, **kw)


def only(breakdown: CostBreakdown, component: str, amount: float):
    assert getattr(breakdown, component) == pytest.approx(amount)
    assert breakdown.total == pytest.approx(amount)
    assert all(getattr(breakdown, c) == 0 for c in COMPONENTS if c != component)


@pytest.mark.parametrize(
    ("late", "f"),
    [(-5, 0), (0, 0), (2, 0), (3, 1), (14, 12), (15, 18), (29, 32), (30, 43), (59, 72), (60, 93), (61, 94)],
)
def test_lateness_curve_steps(late, f):
    # f(14) = 12; f(15) = 13 + 5; f(29) = 27 + 5; f(30) = 28 + 15; f(59) = 57 + 15; f(60) = 58 + 35
    assert lateness_curve(late) == f


def test_classes_come_from_the_spec_whatever_the_id():
    assert classify(trip("SE01", "S", S01.calls, entry_node="PW")).reward == "local"
    assert classify(IC01).reward == "express" and classify(IC01).value == 1.5
    assert classify(trip("RE01", "RE", S01.calls, entry_node="PW")).value == 1.0
    fn = classify(trip("FN01", "FI", FI01.calls, entry_node="PW", exit_node="PN"))
    assert (fn.code, fn.reward, fn.window) == ("FI", "freight", 30)
    assert classify(trip("FW01", "FB", FB01.calls, entry_node="PW")).window == 60
    assert classify(E01, "ECS").reward == "none" and classify(E01, "ENG").reward == "none"
    with pytest.raises(ValueError, match="unknown train type"):
        classify(E01, "XX")


def test_c1_express_and_local_lateness():
    # IC01 at B 20 late: 1.5 x 200/100 x f(20)=23 -> 69.0
    only(bill([ev(50, "ARRIVE", train="IC01", element="B1", node="B", call=1, booked=30)]), "pax_lateness", 69.0)
    # S01 at A 3 late: 1.0 x 40/100 x f(3)=1 -> 0.4
    only(bill([ev(23, "ARRIVE", train="S01", element="A2", node="A", call=0, booked=20)]), "pax_lateness", 0.4)
    # within the grace: free; an unbooked stop (call None): free
    assert bill([ev(32, "ARRIVE", train="IC01", element="B1", node="B", call=1, booked=30)]).total == 0
    assert bill([ev(32, "ARRIVE", train="IC01", element="B1", node="B")]).total == 0


def test_booked_time_falls_back_to_the_timetable():
    # no `booked` on the event: IC01's call 2 is booked to arrive at 50; 80 is 30 late: 1.5 x 3 x 43 = 193.5
    only(bill([ev(80, "ARRIVE", train="IC01", element="C1", node="C", call=2)]), "pax_lateness", 193.5)


def test_headway_is_derived_from_the_timetable():
    sc = _Scenario(TT, TYPES_OVERRIDE, None)
    assert sc.headway("IC01", 0) == (9, True)  # S01 leaves A at 21, IC01 at 12, both for B
    assert sc.headway("IC01", 1) == (9, True)  # S01 leaves B at 41
    assert sc.headway("IC01", 2) == (10, True)  # terminating at C: S01 arrives from B at 60
    assert sc.headway("S01", 1) == (20, True)  # S02 leaves B for C at 61
    assert sc.headway("S01", 0) == (120, True)  # S03 leaves A for B at 250: 229, capped at 120
    assert sc.headway("S03", 0) == (60, False)  # nothing after 250: the default
    # S99 at A heads out to PW; S03 at A (250) heads to B, the wrong way, so it does not count
    assert sc.headway("S99", 2) == (60, False)
    assert sc.headway("S99", 1) == (60, False)  # S99 at B heads to A; nothing later goes that way


def test_headway_handles_rotations_and_express_skipping_intermediate_stops():
    # IC_ROT runs PW -> A1 (dep 12) -> C1 (arr 45, dep 57, turnaround 12) -> B2 (arr 75, dep 77) -> PW.
    ic_rot = trip("IC10", "IC", [
        Call(platform="A1", arrive=10, depart=12, min_dwell=2, board=200),
        Call(platform="C1", arrive=45, depart=57, min_dwell=12, alight=200, board=200),
        Call(platform="B2", arrive=75, depart=77, min_dwell=2, alight=100),
    ], entry_node="PW", exit_node="PW")
    # RE10 runs PW -> A1 (dep 27) -> B1 -> C3 (arr 65, dep 75, turnaround 10) -> B2 (dep 92) -> PW.
    re_rot = trip("RE10", "RE", [
        Call(platform="A1", arrive=25, depart=27, min_dwell=2, board=150),
        Call(platform="B1", arrive=42, depart=44, min_dwell=2, alight=50, board=50),
        Call(platform="C3", arrive=65, depart=75, min_dwell=10, alight=150, board=150),
        Call(platform="B2", arrive=90, depart=92, min_dwell=2, alight=60),
    ], entry_node="PW", exit_node="PW")
    # S_WEST runs westbound A2 (dep 20) -> PW: must NOT count for eastbound IC10 at A1 (dep 12).
    s_west = trip("SW10", "S", [Call(platform="A2", arrive=19, depart=20, board=40)],
                  entry_node="PN", exit_node="PW")
    tt = Timetable(net=NET, trips=(ic_rot, re_rot, s_west), shift_min=180)
    sc = _Scenario(tt, None, None)
    # Eastbound at A: SW10 at 20 heads to PW (opposite way), RE10 at 27 shares onward C -> 27 - 12 = 15
    assert sc.headway("IC10", 0) == (15, True)
    # Turnaround at C: IC10 departs west at 57, RE10 departs west at 75 -> 75 - 57 = 18
    assert sc.headway("IC10", 1) == (18, True)
    # Westbound at B: IC10 departs for PW at 77, RE10 at 92 -> 92 - 77 = 15
    assert sc.headway("IC10", 2) == (15, True)


def test_c2_missed_calls():
    def missed(train, call, node):
        return [ev(70, "CALL_MISSED", train=train, call=call, node=node, detail={"why": "skip"})]

    # IC01 at A: 500 + 1.5 x 150/100 x (f(9)=7 + 10) = 500 + 38.25
    only(bill(missed("IC01", 0, "A")), "missed_calls", 538.25)
    # IC01 at C, its terminus: 500 + 1.5 x 300/100 x (f(10)=8 + 10) = 500 + 81
    only(bill(missed("IC01", 2, "C")), "missed_calls", 581.0)
    # S01 at A, H capped at 120: 500 + 1.0 x 100/100 x (153 + 10) = 663
    only(bill(missed("S01", 0, "A")), "missed_calls", 663.0)
    # S03 at A, default H 60: 500 + 1.0 x 100/100 x (93 + 10) = 603
    only(bill(missed("S03", 0, "A")), "missed_calls", 603.0)
    # the same call reported twice is charged once
    only(bill(missed("S03", 0, "A") * 2), "missed_calls", 603.0)
    # a freight or empty-stock train missing a call is not a passenger stop: free
    assert bill(missed("FB01", 0, "A") + missed("E01", 0, "A")).total == 0


def test_missed_connection_made_vs_missed():
    def conn(made):
        detail = {"feeder": "IC01", "connector": "S01", "transfers": 50, "made": made}
        return [ev(41, "CONNECTION", node="B", detail=detail)]

    assert bill(conn(True)).total == 0
    # S01 at B: H = 20 (S02 at 61). 1.0 x 50/100 x (f(20)=23 + 10) = 16.5
    only(bill(conn(False)), "missed_calls", 16.5)
    lines = itemise(TT, conn(False), types=TYPES_OVERRIDE)
    assert len(lines) == 1 and lines[0].train == "S01"
    # transfers omitted from detail falls back to Connection.transfers (50) in the timetable
    fallback = [ev(41, "CONNECTION", node="B", detail={"feeder": "IC01", "connector": "S01", "made": False})]
    only(bill(fallback), "missed_calls", 16.5)

    # Rotation connector visiting B twice: B1 eastbound (dep 44) and B2 westbound (dep 92)
    re_rot = trip("RE10", "RE", [
        Call(platform="A1", arrive=25, depart=27, min_dwell=2, board=100),
        Call(platform="B1", arrive=42, depart=44, min_dwell=2, alight=30, board=30),
        Call(platform="C3", arrive=65, depart=75, min_dwell=10, alight=100, board=100),
        Call(platform="B2", arrive=90, depart=92, min_dwell=2, alight=40, board=40),
    ], entry_node="PW", exit_node="PW")
    re_next = trip("RE11", "RE", [Call(platform="B2", arrive=110, depart=112, board=40)],
                   entry_node="PN", exit_node="PW")
    feeder = trip("S50", "S", [Call(platform="B1", arrive=85, alight=80)], entry_node="PN")
    rot_tt = Timetable(
        net=NET, trips=(re_rot, re_next, feeder), shift_min=180,
        connections=(Connection(feeder="S50", connector="RE10", node="B", transfers=25),),
    )
    # Westbound call 3 at B2 has H = 112 - 92 = 20 (whereas eastbound call 1 has default H = 60).
    # 1.0 x 25/100 x (f(20)=23 + 10) = 8.25
    missed_west = [ev(92, "CONNECTION", node="B", detail={"feeder": "S50", "connector": "RE10", "made": False})]
    only(cost(rot_tt, missed_west), "missed_calls", 8.25)


def test_c3_freight_window_step():
    def fb(tick):
        return [ev(tick, "ARRIVE", train="FB01", element="P1", node="PORT", call=1, booked=70)]

    only(bill(fb(130)), "freight_lateness", 30.0)  # 60 late, window 60: 0.5 x 60
    only(bill(fb(131)), "freight_lateness", 80.5)  # 61 late: 0.5 x 61 + 50
    assert bill(fb(60)).total == 0  # early: nothing

    def fi(dep):  # FI01 exits at PN with no booked exit time: late as it left its last call (booked 31)
        return [ev(dep, "DEPART", train="FI01", element="A3", node="A", call=0, booked=31),
                ev(dep + 8, "EXIT", train="FI01", element="N-PN", node="PN")]

    only(bill(fi(61)), "freight_lateness", 15.0)  # 30 late, window 30: 0.5 x 30
    only(bill(fi(62)), "freight_lateness", 65.5)  # 31 late: 0.5 x 31 + 50
    # a booked exit time wins: 10 late -> 5.0
    only(bill([ev(50, "EXIT", train="FI01", element="N-PN", node="PN", booked=40)]), "freight_lateness", 5.0)


def test_port_cutoff_is_charged_only_where_one_is_given():
    late = [ev(130, "ARRIVE", train="FB01", element="P1", node="PORT", call=1, booked=70)]
    only(bill(late, port_cutoffs={"FB01": 100}), "freight_lateness", 430.0)  # 30 + 400
    only(bill(late, port_cutoffs={"FB01": 130}), "freight_lateness", 30.0)  # on the cutoff minute: made it


def test_c2_unfinished_trains():
    # passenger: 200; its unserved stops come as their own CALL_MISSED events
    only(bill([ev(300, "UNFINISHED", train="IC01", detail={"freight": False})]), "unfinished", 200.0)
    # FB01 at shift end 300, booked at P1 at 70: 230 late. 200 + 1500 + 0.5 x 230 + 50 = 1865
    only(bill([ev(300, "UNFINISHED", train="FB01", detail={"freight": True})]), "unfinished", 1865.0)
    # FI01 has an exit: booked end is leaving A at 31, 269 late. 200 + 1500 + 134.5 + 50 = 1884.5
    only(bill([ev(300, "UNFINISHED", train="FI01", detail={"freight": True})]), "unfinished", 1884.5)
    # empty stock: 200 only
    only(bill([ev(300, "UNFINISHED", train="E01", detail={"freight": False})]), "unfinished", 200.0)


def test_c4_crew_timeout_for_every_class():
    log = [ev(t, "CREW_TIMEOUT", train=tr, element="B1", detail={"blocked_min": 45})
           for t, tr in [(100, "IC01"), (101, "FB01"), (102, "E01")]]
    only(bill(log), "crew_timeout", 120.0)  # 3 x 40


def test_c5_gridlock_and_operations():
    only(bill([ev(90, "GRIDLOCK", detail={"trains": ["FB01", "S01"]})]), "gridlock", 1000.0)
    stops = [ev(5, "SIGNAL_STOP", train="IC01", element="A-JA.u", detail={"restart_min": 1}),
             ev(6, "SIGNAL_STOP", train="FB01", element="A-JA.u", detail={"restart_min": 5}),
             ev(7, "SIGNAL_STOP", train="E01", element="A-JA.u", detail={"restart_min": 1})]
    only(bill(stops), "operations", 5.0)  # passenger 1 + freight 4 + empty stock 0
    # IC01 on time at B2 instead of B1: 1.5 x (200 + 40)/100 x 2 = 7.2
    moved = [
        ev(31, "ARRIVE", train="IC01", element="B2", node="B", call=1, booked=30, detail={"booked_platform": "B1"})
    ]
    only(bill(moved), "operations", 7.2)


def test_empty_stock_pays_no_passenger_or_freight_charges():
    log = [ev(200, "ARRIVE", train="E01", element="A2", node="A", call=0, booked=100, detail={"booked_platform": "A1"}),
           ev(201, "CALL_MISSED", train="E01", call=1, node="B", detail={"why": "cancel"}),
           ev(300, "UNFINISHED", train="E01", detail={"freight": False})]
    only(bill(log), "unfinished", 200.0)


def test_same_log_same_cost_and_itemised_lines_add_up():
    log = EventLog()
    log.emit(23, "ARRIVE", train="S01", element="A2", node="A", call=0, booked=20)
    log.emit(50, "ARRIVE", train="IC01", element="B2", node="B", call=1, booked=30, detail={"booked_platform": "B1"})
    log.emit(90, "GRIDLOCK", detail={"trains": ["FB01"]})
    log.emit(300, "SHIFT_END")
    log.emit(300, "UNFINISHED", train="FB01", detail={"freight": True})
    first, second = bill(log), bill(list(log.events))
    assert first == second
    # 0.4 + 69 + 7.2 + 1000 + 1865
    assert first.total == pytest.approx(2941.6)
    lines = itemise(TT, log, types=TYPES_OVERRIDE)
    assert [(c.index, c.component) for c in lines] == [
        (0, "pax_lateness"), (1, "pax_lateness"), (1, "operations"), (2, "gridlock"), (4, "unfinished"),
        (4, "unfinished"),
    ]
    assert sum(c.amount for c in lines) == pytest.approx(first.total)
    assert "f(20)=23" in lines[1].explanation


def test_do_nothing_misses_everything_and_finishes_nothing():
    tt = Timetable(net=NET, trips=(IC01, S01, FB01), shift_min=180)
    log = [ev(180, "SHIFT_END")]
    for tr in (IC01, S01):
        for i, c in enumerate(tr.calls):
            log.append(ev(180, "CALL_MISSED", train=tr.train.id, call=i, node=NET.element(c.platform).node,
                          booked=c.arrive, detail={"why": "shift_end"}))
    log += [ev(180, "UNFINISHED", train=t, detail={"freight": t == "FB01"}) for t in ("FB01", "IC01", "S01")]
    got = cost(tt, log)
    # Headways here: IC01 A 9, B 9, C 10 (S01 right behind); S01 has nothing behind it: 60 everywhere.
    # IC01: A 500 + 1.5 x 1.5 x 17 = 538.25; B 500 + 1.5 x 2.4 x 17 = 561.2; C 500 + 1.5 x 3 x 18 = 581
    # S01:  A 500 + 1.0 x 103 = 603;       B 500 + 0.5 x 103 = 551.5;     C 500 + 0.8 x 103 = 582.4
    assert got.missed_calls == pytest.approx(3417.35)
    # 3 x 200 + FB01 undelivered 1500 + 110 late at 180 (booked 70): 55 + 50
    assert got.unfinished == pytest.approx(2205.0)
    assert got.total == pytest.approx(5622.35)
    assert got.pax_lateness == got.freight_lateness == got.operations == 0


def test_a_log_naming_an_unknown_train_is_refused():
    with pytest.raises(ValueError, match="not in the timetable"):
        bill([ev(1, "SIGNAL_STOP", train="ZZ9", element="A-JA.u")])


def test_regional_express_and_engineering_classes_end_to_end():
    re01 = trip("RE01", "RE", [Call(platform="A1", arrive=20, alight=150)], entry_node="PW")
    eng01 = Trip(
        train=TrainSpec(id="ENG01", length_m=100, electric=False, passenger=False, max_speed_kmh=60, restart_min=1),
        calls=(Call(platform="B3", arrive=40),),
        entry_node="PW",
    )
    tt = Timetable(net=NET, trips=(re01, eng01), shift_min=180)
    log = [
        # RE01 15 late: v=1.0 x 150/100 x f(15)=18 -> 27.0
        ev(35, "ARRIVE", train="RE01", element="A1", node="A", call=0, booked=20),
        # ENG01 signal stop and late arrival pay 0; crew timeout pays 40; unfinished pays 200
        ev(25, "SIGNAL_STOP", train="ENG01", element="A-JA.u", detail={"restart_min": 1}),
        ev(60, "CREW_TIMEOUT", train="ENG01", element="A-JA.u", detail={"blocked_min": 45}),
        ev(180, "SHIFT_END"),
        ev(180, "UNFINISHED", train="ENG01", detail={"freight": False}),
    ]
    got = cost(tt, log)
    assert got.pax_lateness == pytest.approx(27.0)
    assert got.operations == 0.0
    assert got.crew_timeout == pytest.approx(40.0)
    assert got.unfinished == pytest.approx(200.0)
    assert got.total == pytest.approx(267.0)


def test_worked_example_from_net_cost_doc():
    tt = Timetable(
        net=NET, trips=(IC01, S01, FB01), shift_min=180,
        connections=(Connection(feeder="IC01", connector="S01", node="B", transfers=50),),
    )
    log = [
        ev(28, "SIGNAL_STOP", train="IC01", element="JA-B.u", detail={"restart_min": 1}),
        ev(41, "CONNECTION", node="B", detail={"feeder": "IC01", "connector": "S01", "transfers": 50, "made": False}),
        ev(50, "ARRIVE", train="IC01", element="B2", node="B", call=1, booked=30, detail={"booked_platform": "B1"}),
        ev(60, "CALL_MISSED", train="IC01", call=2, node="C", booked=50, detail={"why": "short_turn"}),
        ev(135, "ARRIVE", train="FB01", element="P1", node="PORT", call=1, booked=70),
        ev(180, "SHIFT_END"),
        ev(180, "UNFINISHED", train="IC01", detail={"freight": False}),
    ]
    got = cost(tt, log)
    assert got == CostBreakdown(
        pax_lateness=69.0,
        missed_calls=632.5,
        unfinished=200.0,
        freight_lateness=82.5,
        crew_timeout=0.0,
        gridlock=0.0,
        operations=8.2,
        total=992.2,
    )


def test_charges_are_rounded_to_six_decimals():
    # S01 at B 3 late: 1.0 x 30/100 x 1 = 0.3 exactly, however floats round
    got = bill([ev(43, "ARRIVE", train="S01", element="B2", node="B", call=1, booked=40)] * 3)
    assert got.pax_lateness == 0.9 and got.total == 0.9
