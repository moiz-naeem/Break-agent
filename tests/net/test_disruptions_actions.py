import pytest

from railway.net.engine import SIGNAL_FAILURE_FACTOR, Engine, track_minutes
from railway.net.model import Call, Connection, Disruption, Timetable, Trip
from railway.net.rules import NetRule
from railway.net.templates import t1_services, trunk_and_branches
from railway.net.timetable import generate, generate_disruptions, with_disruptions
from railway.net.trains import TYPES

NET = trunk_and_branches()


def make_trip(train_id: str, calls: tuple[Call, ...], code: str = "S", **kw) -> Trip:
    return Trip(train=TYPES[code].spec(train_id), calls=calls, **kw)


def run_auto(eng: Engine, limit: int = 300):
    seen = []
    for _ in range(limit):
        d = eng.advance()
        seen.append(d)
        if d.kind == "SHIFT_END":
            return seen
        if d.kind == "CONFLICT":
            for c in d.conflicts:
                assert eng.order(c.key, sorted(c.trains, key=lambda t: eng.trains[t].booked())) is None
    raise AssertionError(f"did not finish: {[(d.tick, d.kind) for d in seen]}")


# ------------------------------------------------------------------ all 7 disruption kinds & no future leak


def test_late_entry_has_no_future_leak_before_reveal_and_delays_entry():
    trip = make_trip(
        "IC1",
        (Call(platform="A1", arrive=35, depart=37), Call(platform="C1", arrive=70)),
        code="IC",
        entry_node="PW",
        entry_tick=25,
    )
    d = Disruption(id="LE1", kind="late_entry", start=25, end=33, reveal=10, train="IC1")
    eng = Engine(NET, [trip], shift_min=120, disruptions=[d])

    # Before reveal (t=0): no bulletin, entry_tick still equals booked entry_tick (25)
    assert eng.bulletins() == []
    assert eng.trains["IC1"].entry_tick == 25

    dec = eng.advance()
    assert (dec.tick, dec.kind, dec.trains, dec.disruptions) == (10, "DISRUPTION", ("IC1",), ("LE1",))
    assert eng.bulletins() == [d]
    assert eng.trains["IC1"].entry_tick == 33

    run_auto(eng)
    enter = eng.log.of("ENTER")[0]
    assert (enter.tick, enter.booked, enter.node) == (33, 25, "PW")
    assert [(e.tick, e.kind, e.node) for e in eng.log.of("DISRUPTION_START", "DISRUPTION_END")] == [
        (25, "DISRUPTION_START", "PW"),
        (33, "DISRUPTION_END", "PW"),
    ]


def test_possession_and_platform_closed_have_no_future_leak_before_reveal():
    """Before reveal, ARS sets routes exactly as if no future possession existed; once revealed, ARS holds
    trains clear of a possession that would start before they clear the track."""
    trip = make_trip("IC1", (Call(platform="A1", depart=10), Call(platform="C1", arrive=45)), code="IC")
    # Possession on JA-B.u from t=16..40, revealed only at t=15: at t=10 IC1 leaves A1 unhampered.
    unrevealed_at_10 = Disruption(id="P_LATE", kind="possession", start=16, end=40, reveal=15, elements=("JA-B.u",))
    eng_late_rev = Engine(NET, [trip], shift_min=100, disruptions=[unrevealed_at_10])
    eng_clean = Engine(NET, [trip], shift_min=100)

    # Compare logs up to t=14 (strictly before reveal=15): must be identical (no future leak!)
    d_rev = eng_late_rev.advance()
    assert (d_rev.tick, d_rev.kind) == (15, "DISRUPTION")
    assert [e for e in eng_late_rev.log.events if e.tick < 15] == [
        e for e in (run_auto(eng_clean) and eng_clean.log.events) if e.tick < 15
    ]

    # With reveal=0 and possession starting at t=14 on A-JA.u, IC1 at A1 (clears A-JA.u at t=15 > 14) is held.
    revealed_early = Disruption(id="P_EARLY", kind="possession", start=14, end=30, reveal=0, elements=("A-JA.u",))
    eng_held = Engine(NET, [trip], shift_min=100, disruptions=[revealed_early])
    assert eng_held.advance().kind == "DISRUPTION"
    run_auto(eng_held)
    depart_tick = [e.tick for e in eng_held.log.of("DEPART") if e.train == "IC1"][0]
    assert depart_tick == 30  # held safely in A1 until possession ends at t=30


def test_platform_closed_blocks_platform_and_replatforming_recovers():
    trip = make_trip("IC1", (Call(platform="A1", depart=5), Call(platform="C1", arrive=40)), code="IC")
    closed = Disruption(id="PC1", kind="platform_closed", start=20, end=80, reveal=0, elements=("C1",))
    eng = Engine(NET, [trip], shift_min=120, disruptions=[closed])
    d = eng.advance()
    assert (d.tick, d.kind, d.disruptions) == (0, "DISRUPTION", ("PC1",))
    assert eng.platform("IC1", "C2") is None
    run_auto(eng)
    arr = eng.log.of("ARRIVE")[0]
    assert (arr.element, arr.detail) == ("C2", {"booked_platform": "C1"})
    assert len(eng.log.of("DISRUPTION_START")) == 1 and len(eng.log.of("DISRUPTION_END")) == 1


def test_breakdown_stops_live_train_and_ignores_offstage_train():
    t_live = make_trip("S1", (Call(platform="A1", depart=0), Call(platform="C1", arrive=40)), code="S")
    t_off = make_trip(
        "S2",
        (Call(platform="A2", arrive=70, depart=72), Call(platform="N1", arrive=95)),
        code="S",
        entry_node="PW",
        entry_tick=60,
    )
    d_live = Disruption(id="B_LIVE", kind="breakdown", start=5, end=20, reveal=5, train="S1")
    d_off = Disruption(id="B_OFF", kind="breakdown", start=10, end=25, reveal=10, train="S2")

    base = Engine(NET, [t_live], shift_min=120)
    run_auto(base)
    base_arr = base.log.of("ARRIVE")[0].tick

    eng = Engine(NET, [t_live, t_off], shift_min=120, disruptions=[d_live, d_off])
    decisions = run_auto(eng)
    dis_decisions = [d for d in decisions if d.kind == "DISRUPTION"]
    # Only B_LIVE fires; B_OFF is at t=10 while S2 is still offstage (entry_tick=60), so nobody hears of B_OFF.
    assert [(d.tick, d.disruptions) for d in dis_decisions] == [(5, ("B_LIVE",))]
    assert [b.id for b in eng.bulletins()] == ["B_LIVE"]
    s1_arr = [e.tick for e in eng.log.of("ARRIVE") if e.train == "S1"][0]
    assert s1_arr == base_arr + (20 - 5) + TYPES["S"].restart_min


def test_speed_restriction_and_signal_failure_slow_tracks_only_while_active():
    trip = make_trip("IC1", (Call(platform="A1", depart=0), Call(platform="C1", arrive=40)), code="IC")
    base = Engine(NET, [trip], shift_min=120)
    run_auto(base)
    base_arr = base.log.of("ARRIVE")[0].tick

    # Signal failure on JA-B.u doubles block traversal time by SIGNAL_FAILURE_FACTOR (2x)
    sf = Disruption(id="SF1", kind="signal_failure", start=2, end=60, reveal=2, elements=("JA-B.u",))
    eng_sf = Engine(NET, [trip], shift_min=120, disruptions=[sf])
    seen_sf = run_auto(eng_sf)
    assert any(d.kind == "DISRUPTION" and d.tick == 2 and d.disruptions == ("SF1",) for d in seen_sf)
    normal_jab = track_minutes(NET.element("JA-B.u"), trip.train)
    sf_arr = eng_sf.log.of("ARRIVE")[0].tick
    assert sf_arr == base_arr + normal_jab * (SIGNAL_FAILURE_FACTOR - 1)

    # Speed restriction 30 km/h on JA-B.u (15 km -> 30 min vs normal 6 min -> +24 min)
    sr = Disruption(id="SR1", kind="speed_restriction", start=2, end=80, reveal=0, elements=("JA-B.u",), speed_kmh=30)
    eng_sr = Engine(NET, [trip], shift_min=120, disruptions=[sr])
    run_auto(eng_sr)
    sr_arr = eng_sr.log.of("ARRIVE")[0].tick
    assert sr_arr == base_arr + (30 - normal_jab)


def test_points_failure_blocks_move_until_cleared_or_diverted():
    # Enter from PW to C1 stopping nowhere except C1: normal path uses A.w1 -> A1 -> A.e1.
    # Points failure on A.w1 blocks entry unless diverted via A2 using route("IC1", ["A2"]).
    trip = make_trip(
        "IC1",
        (Call(platform="C1", arrive=45),),
        code="IC",
        entry_node="PW",
        entry_tick=5,
    )
    pf = Disruption(id="PF1", kind="points_failure", start=4, end=50, reveal=4, elements=("A.w1",))

    # Without diversion: IC1 waits until PF1 clears at t=50
    eng_wait = Engine(NET, [trip], shift_min=120, disruptions=[pf])
    d = eng_wait.advance()
    assert (d.tick, d.kind, d.disruptions) == (4, "DISRUPTION", ("PF1",))
    run_auto(eng_wait)
    assert eng_wait.log.of("ENTER")[0].tick == 50

    # With route("IC1", ["A2"]): IC1 enters right on time at t=5 via A.w3 -> A2 -> A.e3!
    eng_div = Engine(NET, [trip], shift_min=120, disruptions=[pf])
    assert eng_div.advance().kind == "DISRUPTION"
    assert eng_div.route(train="IC1", via=["A2"]) is None
    run_auto(eng_div)
    assert eng_div.log.of("ENTER")[0].tick == 5
    assert ("IC1", "done", "C1") in [r[:3] for r in eng_div.state()]


# ------------------------------------------------------------------ disruption generator


def test_disruption_generator_is_deterministic_valid_and_scales_with_density():
    core, fill = t1_services()
    tt = Timetable(
        net=NET,
        trips=(
            make_trip(
                "IC01",
                (Call(platform="A1", arrive=8, depart=10), Call(platform="C1", arrive=40)),
                code="IC",
                entry_node="PW",
                entry_tick=0,
            ),
            make_trip(
                "SE01",
                (Call(platform="A2", arrive=22, depart=24), Call(platform="N1", arrive=48)),
                code="S",
                entry_node="PW",
                entry_tick=14,
            ),
        ),
        shift_min=180,
        seed=42,
        tier="T1",
    )
    assert generate_disruptions(tt, seed=42, density=0.0) == ()
    with pytest.raises(ValueError, match="density"):
        generate_disruptions(tt, seed=42, density=-0.5)

    d1 = generate_disruptions(tt, seed=42, density=1.0)
    d2 = generate_disruptions(tt, seed=42, density=1.0)
    d_other = generate_disruptions(tt, seed=99, density=1.0)
    d_heavy = generate_disruptions(tt, seed=42, density=2.5)

    assert d1 == d2 and len(d1) >= 3
    assert d1 != d_other
    assert len(d_heavy) > len(d1)

    # Every generated disruption validates on Timetable and runs cleanly in Engine.from_timetable
    seen_kinds = set()
    for s in range(1, 20):
        populated = with_disruptions(tt, seed=s, density=2.0)
        seen_kinds |= {d.kind for d in populated.disruptions}
        eng = Engine.from_timetable(populated)
        run_auto(eng)
        assert eng.ended
    assert seen_kinds == {
        "late_entry",
        "breakdown",
        "possession",
        "speed_restriction",
        "signal_failure",
        "points_failure",
        "platform_closed",
    }

    # Hook on generate() preserves default undisturbed output (density=0.0) and populates when > 0
    small_group = [core[1]]
    tt_clean = generate(NET, small_group, [], 120, 7, "T1")
    tt_dis = generate(NET, small_group, [], 120, 7, "T1", disruption_density=1.0)
    assert tt_clean.disruptions == ()
    assert tt_clean.trips == tt_dis.trips
    assert len(tt_dis.disruptions) >= 2


# ------------------------------------------------------------------ the 5 service actions + refusals


def test_skip_action_legal_and_refusals():
    trip = make_trip(
        "IC1",
        (
            Call(platform="A1", depart=0, min_dwell=1),
            Call(platform="B1", arrive=15, depart=17, min_dwell=2),
            Call(platform="C1", arrive=30, depart=40, min_dwell=10),
            Call(platform="B2", arrive=52, depart=54, min_dwell=2),
        ),
        code="IC",
        exit_node="PW",
    )
    t_term = make_trip("S1", (Call(platform="A2", depart=5), Call(platform="N1", arrive=30)), code="S")
    eng = Engine(NET, [trip, t_term], shift_min=120)

    assert eng.skip("nope", "B").rule == NetRule.UNKNOWN_TRAIN
    assert eng.skip("S1", "C").rule == NetRule.NO_CALL
    assert eng.skip("S1", "N").rule == NetRule.SKIP_DESTINATION  # ends at N with no exit_node
    # Skipping C on IC1 removes its required turnaround, leaving no route from A1 to down platform B2
    assert eng.skip("IC1", "C").rule == NetRule.NO_ROUTE

    # Legal skip of intermediate stop B (outbound call 1), using Session keyword style
    assert eng.skip(train="IC1", station="B") is None
    missed = eng.log.of("CALL_MISSED")
    assert len(missed) == 1 and (missed[0].call, missed[0].detail) == (1, {"why": "skip"})

    run_auto(eng)
    # IC1 passed B1 without stopping (PASS with call=1), then called at C1 and B2, and exited at PW
    assert ("IC1", "done", "PW") in [r[:3] for r in eng.state()]
    assert any(e.train == "IC1" and e.element == "B1" and e.call == 1 for e in eng.log.of("PASS"))
    assert [e.call for e in eng.log.of("ARRIVE") if e.train == "IC1"] == [2, 3]


def test_single_line_and_route_wrong_road_diversion_and_refusals():
    """Possession closes JA-B.d; single_line('JA-B.u', True) + route('RE1', ['JA-B.u']) diverts RE1 wrong-way
    around the blockage (dropping unreachable B2 with why='route'), while un-routed IC2 stays on normal road."""
    re1 = make_trip(
        "RE1",
        (
            Call(platform="C3", depart=0, min_dwell=1),
            Call(platform="B2", arrive=12, depart=14, min_dwell=2),
            Call(platform="A2", arrive=30, depart=35, min_dwell=2),
        ),
        code="RE",
        exit_node="PW",
    )
    ic2 = make_trip(
        "IC2",
        (Call(platform="C1", depart=45, min_dwell=1), Call(platform="A2", arrive=75)),
        code="IC",
    )
    poss = Disruption(id="P_DOWN", kind="possession", start=0, end=40, reveal=0, elements=("JA-B.d",))
    sr = Disruption(id="SR_N", kind="speed_restriction", start=60, end=80, reveal=30, elements=("JA-N",), speed_kmh=50)
    eng = Engine(NET, [re1, ic2], shift_min=140, disruptions=[poss, sr])

    # Refusals on single_line and route before opening single-line working
    assert eng.single_line("nope", True).rule == NetRule.UNKNOWN_ELEMENT
    assert eng.single_line("A-JA.u", True).rule == NetRule.NOT_REVERSIBLE
    assert eng.single_line("JA-B.u", False).rule == NetRule.SINGLE_LINE  # already off
    assert eng.route("RE1", []).rule == NetRule.BAD_ROUTE
    assert eng.route("RE1", ["JA.x1"]).rule == NetRule.UNKNOWN_ELEMENT  # move, not track/platform
    assert eng.route("RE1", ["C3"]).rule == NetRule.NO_ROUTE  # already standing in C3
    assert eng.route("RE1", ["JA-B.u"]).rule == NetRule.NO_ROUTE  # single_line not open yet

    # Open single-line on JA-B.u and route RE1 via JA-B.u
    assert eng.single_line(segment="JA-B.u", on=True) is None
    assert eng.single_line("JA-B.u", True).rule == NetRule.SINGLE_LINE  # already on
    assert eng.route(train="RE1", via=["JA-B.u"]) is None

    # Cannot close single-line while RE1 is routed wrong-way over JA-B.u
    assert eng.single_line("JA-B.u", False).rule == NetRule.SINGLE_LINE

    # B2 was automatically dropped with why="route", while A2 and exit PW were kept
    missed = eng.log.of("CALL_MISSED")
    assert len(missed) == 1 and (missed[0].call, missed[0].element, missed[0].detail) == (1, "B2", {"why": "route"})

    # Advance to t=25 (SR_N reveal): RE1 has cleared JA-B.u and stands in A2, so single_line can now be closed
    for _ in range(20):
        d = eng.advance()
        if eng.trains["RE1"].position == "A2" or d.kind == "SHIFT_END":
            break
    assert eng.trains["RE1"].position == "A2"
    assert eng.single_line("JA-B.u", False) is None

    run_auto(eng)
    assert ("RE1", "done", "PW") in [r[:3] for r in eng.state()]
    assert ("IC2", "done", "A2") in [r[:3] for r in eng.state()]
    # IC2 was never routed wrong-way: it ran via B2 on normal road after possession cleared at t=40
    assert any(e.train == "IC2" and e.element == "B2" for e in eng.log.of("PASS"))


def test_short_turn_before_and_at_station_with_merge_and_refusals():
    # RE1: PW -> A1 -> B1 -> C3 (turn 10m) -> B2 -> exit PW. Short-turning at B merges B2 into B1!
    re1 = make_trip(
        "RE1",
        (
            Call(platform="A1", arrive=8, depart=10, min_dwell=2),
            Call(platform="B1", arrive=24, depart=30, min_dwell=2),
            Call(platform="C3", arrive=42, depart=52, min_dwell=10),
            Call(platform="B2", arrive=64, depart=66, min_dwell=2),
        ),
        code="RE",
        entry_node="PW",
        entry_tick=0,
        exit_node="PW",
    )
    one_way = make_trip("S1", (Call(platform="A2", depart=5), Call(platform="N1", arrive=30)), code="S")

    # Case 1: short_turn called ahead of B (while RE1 is offstage)
    eng1 = Engine(NET, [re1, one_way], shift_min=140)
    assert eng1.short_turn("S1", "A").rule == NetRule.NO_RETURN
    assert eng1.short_turn("RE1", "N").rule == NetRule.NO_CALL
    assert eng1.short_turn("RE1", "C").rule == NetRule.NO_CALL  # C is the turnaround itself, not before it
    # A1 has no move onto PW-A.d and cannot reach B2 without turning at C
    assert eng1.short_turn("RE1", "A").rule == NetRule.NO_ROUTE

    assert eng1.short_turn(train="RE1", station="B") is None
    # A second short_turn on RE1 is refused (merged return call's 8 min dwell is not a second turnaround!)
    assert eng1.short_turn("RE1", "B").rule == NetRule.NO_RETURN
    missed1 = eng1.log.of("CALL_MISSED")
    assert [(e.call, e.detail["why"]) for e in missed1] == [(2, "short_turn")]

    run_auto(eng1)
    assert ("RE1", "done", "PW") in [r[:3] for r in eng1.state()]
    re1_arrives = [(e.element, e.call, e.detail) for e in eng1.log.of("ARRIVE") if e.train == "RE1"]
    assert re1_arrives == [("A1", 0, {}), ("B1", 1, {}), ("B1", 3, {"booked_platform": "B2"})]

    # Case 2: short_turn called while RE1 already stands berthed at B1 (here=True, merge=True)
    poss_c = Disruption(id="PC", kind="possession", start=35, end=80, reveal=26, elements=("JB-C.u",))
    eng2 = Engine(NET, [re1], shift_min=140, disruptions=[poss_c])
    d2 = eng2.advance()
    assert (d2.tick, d2.kind) == (26, "DISRUPTION")
    assert eng2.trains["RE1"].phase == "berthed" and eng2.trains["RE1"].position == "B1"
    assert eng2.short_turn("RE1", "B") is None
    assert eng2.short_turn("RE1", "B").rule == NetRule.NO_RETURN
    run_auto(eng2)
    assert ("RE1", "done", "PW") in [r[:3] for r in eng2.state()]

    # Case 3: short_turn with merge=False (return working does not call at B: B1 -> C1(turn) -> A2 -> PW)
    ic_no_merge = make_trip(
        "IC_NM",
        (
            Call(platform="B1", depart=24, min_dwell=2),
            Call(platform="C1", arrive=36, depart=46, min_dwell=10),
            Call(platform="A2", arrive=70, depart=72, min_dwell=2),
        ),
        code="IC",
        exit_node="PW",
    )
    eng3 = Engine(NET, [ic_no_merge], shift_min=140)
    assert eng3.short_turn("IC_NM", "B") is None
    run_auto(eng3)
    assert ("IC_NM", "done", "PW") in [r[:3] for r in eng3.state()]


def test_cancel_offstage_running_and_stranded_plus_refusals():
    offstage = make_trip(
        "IC_OFF",
        (Call(platform="A1", arrive=15, depart=17), Call(platform="C1", arrive=45)),
        code="IC",
        entry_node="PW",
        entry_tick=8,
    )
    running = make_trip(
        "S_RUN",
        (
            Call(platform="A2", depart=0, min_dwell=1),
            Call(platform="B1", arrive=15, depart=25, min_dwell=5),
            Call(platform="C2", arrive=40),
        ),
        code="S",
    )
    stranded = make_trip(
        "IC_STRAND",
        (Call(platform="A1", depart=5), Call(platform="N1", arrive=30)),  # N1 too short for 260m IC
        code="IC",
    )

    eng = Engine(NET, [offstage, running, stranded], shift_min=100)
    assert eng.cancel("nope").rule == NetRule.UNKNOWN_TRAIN

    # Cancel offstage train: never enters network
    assert eng.cancel(train="IC_OFF") is None
    assert eng.cancel("IC_OFF").rule == NetRule.TRAIN_DONE

    # Advance to t=5 where S_RUN is running and IC_STRAND reports UNROUTABLE
    d = eng.advance()
    if d.kind == "CONFLICT":
        for c in d.conflicts:
            eng.order(c.key, ["S_RUN", *(t for t in c.trains if t != "S_RUN")])
        d = eng.advance()
    assert d.kind == "UNROUTABLE" and d.trains == ("IC_STRAND",)

    # Cancel stranded berthed train: parks cleanly in A1 without reporting BLOCKED
    assert eng.cancel("IC_STRAND") is None
    assert eng.trains["IC_STRAND"].phase == "done"

    # Cancel running train S_RUN: runs empty without waiting for booked depart=25 at B1
    assert eng.cancel("S_RUN") is None
    run_auto(eng)

    assert [e for e in eng.log.of("ENTER") if e.train == "IC_OFF"] == []
    assert ("S_RUN", "done", "C2") in [r[:3] for r in eng.state()]
    s_arr = [e for e in eng.log.of("ARRIVE") if e.train == "S_RUN" and e.element == "B1"][0]
    s_dep = [e for e in eng.log.of("DEPART") if e.train == "S_RUN" and e.element == "B1"][0]
    assert s_arr.call is None and s_dep.call is None and s_dep.tick == s_arr.tick  # 0 dwell when running empty
    assert {e.train for e in eng.log.of("UNFINISHED")} == {"IC_OFF", "IC_STRAND", "S_RUN"}


# ------------------------------------------------------------------ determinism


def test_complex_disrupted_shift_with_actions_produces_identical_event_logs():
    def play_shift() -> Engine:
        trips = [
            make_trip(
                "IC1",
                (
                    Call(platform="A1", arrive=8, depart=10, min_dwell=2, board=120),
                    Call(platform="B1", arrive=22, depart=24, min_dwell=2, alight=30, board=30),
                    Call(platform="C1", arrive=38, depart=48, min_dwell=10, alight=120),
                ),
                code="IC",
                entry_node="PW",
                entry_tick=0,
                exit_node="PW",
                crew_min=205,
            ),
            make_trip(
                "RE1",
                (
                    Call(platform="C3", depart=5, min_dwell=1, board=80),
                    Call(platform="B2", arrive=17, depart=19, min_dwell=2, alight=20, board=20),
                    Call(platform="A2", arrive=35, alight=80),
                ),
                code="RE",
            ),
            make_trip(
                "S1",
                (Call(platform="A2", depart=12, min_dwell=1, board=60), Call(platform="N1", arrive=36, alight=60)),
                code="S",
            ),
        ]
        disruptions = [
            Disruption(id="D1", kind="possession", start=0, end=30, reveal=0, elements=("JA-B.d",)),
            Disruption(id="D2", kind="speed_restriction", start=10, end=50, reveal=2, elements=("JA-N",), speed_kmh=50),
        ]
        conns = [Connection(feeder="IC1", connector="S1", node="A", transfers=20, min_transfer_min=2)]
        eng = Engine(NET, trips, shift_min=150, connections=conns, disruptions=disruptions)

        assert eng.advance().kind == "DISRUPTION"
        assert eng.single_line("JA-B.u", True) is None
        assert eng.route("RE1", ["JA-B.u"]) is None
        assert eng.skip("IC1", "B") is None

        while True:
            d = eng.advance()
            if d.kind == "SHIFT_END":
                return eng
            if d.kind == "CONFLICT":
                for c in d.conflicts:
                    # Exercise JSON-style list key normalization on order()
                    key = list(c.key) if isinstance(c.key, tuple) else c.key
                    assert eng.order(key, sorted(c.trains)) is None
            if d.kind == "CREW" and "IC1" in d.trains and not eng.trains["IC1"].relieve_next:
                if eng.trains["IC1"].crew_since is not None:
                    assert eng.relieve_crew("IC1") is None

    one, two = play_shift(), play_shift()
    assert one.log.events == two.log.events
    assert one.state() == two.state()


def test_running_train_replanning_and_event_log_invariants():
    """Verify skip, route, short_turn and cancel(NO_CALL) on already-running trains, Engine disruption validation,
    and strict EventLog ordering invariants."""
    re1 = make_trip(
        "RE1",
        (
            Call(platform="A1", arrive=8, depart=10, min_dwell=2),
            Call(platform="B1", arrive=24, depart=28, min_dwell=2),
            Call(platform="C3", arrive=40, depart=50, min_dwell=10),
            Call(platform="B2", arrive=62, depart=64, min_dwell=2),
        ),
        code="RE",
        entry_node="PW",
        entry_tick=0,
        exit_node="PW",
    )
    # Trigger decision point at t=2 while RE1 is running on PW-A.u
    d_early = Disruption(
        id="D_E", kind="speed_restriction", start=80, end=90, reveal=2, elements=("JA-N",), speed_kmh=50
    )

    eng = Engine(NET, [re1], shift_min=140, disruptions=[d_early])
    d = eng.advance()
    assert (d.tick, d.kind) == (2, "DISRUPTION") and eng.trains["RE1"].phase == "running"
    # RE1 already holds authority into A1: skip("RE1", "A") extends through A1 without stopping, then short_turn at B
    assert eng.skip("RE1", "A") is None
    assert eng.short_turn("RE1", "B") is None

    # At t=58 (IDLE), RE1 has already completed its merged return call at B1 and only awaits exit -> NO_CALL on cancel
    d2 = eng.advance()
    assert d2 == (58, "IDLE", (), (), ())
    assert eng.cancel("RE1").rule == NetRule.NO_CALL
    run_auto(eng)

    assert any(e.train == "RE1" and e.element == "A1" and e.call == 0 for e in eng.log.of("PASS"))
    # Strict EventLog post-SHIFT_END ordering invariant
    ticks = [e.tick for e in eng.log.events]
    assert ticks == sorted(ticks)
    shift_idx = [i for i, e in enumerate(eng.log.events) if e.kind == "SHIFT_END"]
    assert len(shift_idx) == 1
    tail = eng.log.events[shift_idx[0] + 1 :]
    assert all(e.kind in {"CALL_MISSED", "UNFINISHED"} for e in tail)

    # Direct Engine disruption & connection validation errors
    with pytest.raises(ValueError, match="share an id"):
        Engine(NET, [re1], 100, disruptions=[d_early, d_early])
    with pytest.raises(ValueError, match="unknown train"):
        Engine(NET, [re1], 100, connections=[Connection(feeder="RE1", connector="ZZ", node="A", transfers=5)])
    with pytest.raises(ValueError, match="not a move"):
        Engine(
            NET,
            [re1],
            100,
            disruptions=[Disruption(id="BAD", kind="points_failure", start=5, end=10, reveal=5, elements=("A1",))],
        )
