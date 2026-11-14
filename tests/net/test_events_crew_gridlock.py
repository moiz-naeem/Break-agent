from railway.net.engine import CREW_LIMIT_MIN, CREW_RELIEF_MIN, CREW_TIMEOUT_MIN, CREW_WARNING_MIN, Engine
from railway.net.events import KINDS
from railway.net.model import Call, Connection, Disruption, Trip
from railway.net.rules import NetRule
from railway.net.templates import trunk_and_branches
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


# ------------------------------------------------------------------ all event kinds


def test_every_event_kind_is_emitted_with_valid_schema_fields():
    """Exercise all 17 EventKind values across shifts and verify schema invariants."""
    seen_kinds: set[str] = set()

    # Shift 1: portal ENTER, platform ENTER, ACTION, ARRIVE (with booked_platform), DEPART, PASS,
    # CALL_MISSED, CONNECTION (made & missed), SIGNAL_STOP, EXIT, DISRUPTION_START/END, SHIFT_END, UNFINISHED.
    trips1 = [
        make_trip(
            "IC1",
            (
                Call(platform="A1", arrive=8, depart=10, min_dwell=2, board=100),
                Call(platform="B1", arrive=22, depart=24, min_dwell=2, alight=20, board=20),
                Call(platform="C1", arrive=40, depart=50, min_dwell=10, alight=100),
            ),
            code="IC",
            entry_node="PW",
            entry_tick=0,
            exit_node="PW",
        ),
        make_trip(
            "S1",
            (Call(platform="A2", depart=10, min_dwell=1, board=50), Call(platform="N1", arrive=30, alight=50)),
            code="S",
        ),
        make_trip(
            "F1",
            (Call(platform="B3", depart=200, min_dwell=2), Call(platform="P1", arrive=240)),
            code="FB",
        ),
    ]
    conns1 = [Connection(feeder="IC1", connector="S1", node="A", transfers=15, min_transfer_min=2)]
    disruptions1 = [
        Disruption(id="D1", kind="speed_restriction", start=5, end=20, reveal=0, elements=("A-JA.u",), speed_kmh=60)
    ]
    eng1 = Engine(NET, trips1, shift_min=150, connections=conns1, disruptions=disruptions1)

    d = eng1.advance()
    assert d.kind == "DISRUPTION"
    assert eng1.skip("IC1", "B") is None  # ACTION + CALL_MISSED(why="skip") + later PASS at B1
    d = eng1.advance()
    while d.kind != "SHIFT_END":
        if d.kind == "CONFLICT":
            for c in d.conflicts:
                # Let IC1 go ahead of S1 so S1 waits at signal or vice-versa; also hold S1 for connection
                eng1.order(c.key, ["IC1", *(t for t in c.trains if t != "IC1")])
        if eng1.trains["IC1"].phase == "berthed" and eng1.trains["IC1"].position == "A1":
            eng1.platform("IC1", "C2")  # re-platform C1 -> C2 so ARRIVE logs booked_platform="C1"
        d = eng1.advance()

    # Shift 2: following train stopped at block signal -> SIGNAL_STOP, plus CREW_WARNING, CREW_RELIEF, CREW_TIMEOUT.
    trips2 = [
        make_trip(
            "FB1",
            (Call(platform="A3", depart=0, min_dwell=1), Call(platform="B3", arrive=40)),
            code="FB",
            crew_min=CREW_LIMIT_MIN - CREW_WARNING_MIN - 2,
        ),
        make_trip(
            "S2",
            (
                Call(platform="A1", depart=1, min_dwell=1),
                Call(platform="C1", arrive=35, depart=45, min_dwell=10),
                Call(platform="A2", arrive=80),
            ),
            code="S",
            crew_min=CREW_LIMIT_MIN - CREW_WARNING_MIN - 5,
        ),
    ]
    eng2 = Engine(NET, trips2, shift_min=160)
    while True:
        d = eng2.advance()
        if d.kind == "SHIFT_END":
            break
        if d.kind == "CONFLICT":
            for c in d.conflicts:
                eng2.order(c.key, ["FB1", *(t for t in c.trains if t != "FB1")])
        if d.kind == "CREW" and "S2" in d.trains and not eng2.trains["S2"].relieve_next:
            if eng2.trains["S2"].crew_since is not None:
                assert eng2.relieve_crew("S2") is None

    # Shift 3: head-on single-track standoff -> GRIDLOCK.
    trips3 = [
        make_trip("E1", (Call(platform="A1", depart=0), Call(platform="N1", arrive=30)), code="S"),
        make_trip("W1", (Call(platform="N1", depart=20), Call(platform="A2", arrive=50)), code="S"),
        make_trip("W2", (Call(platform="N2", depart=20), Call(platform="A2", arrive=55)), code="S"),
    ]
    eng3 = Engine(NET, trips3, shift_min=80)
    run_auto(eng3)

    for eng in (eng1, eng2, eng3):
        seen_kinds |= {e.kind for e in eng.log.events}

    assert seen_kinds == set(KINDS)

    # Verify detailed fields on key events
    replat_arrive = [e for e in eng1.log.of("ARRIVE") if e.train == "IC1" and e.element == "C2"]
    assert len(replat_arrive) == 1
    assert replat_arrive[0].call == 2
    assert replat_arrive[0].booked == 40
    assert replat_arrive[0].detail == {"booked_platform": "C1"}

    passes = [e for e in eng1.log.of("PASS") if e.train == "IC1" and e.node == "B"]
    assert [(e.element, e.call, e.booked) for e in passes] == [("B1", 1, 22), ("B2", None, None)]

    conns = eng1.log.of("CONNECTION")
    assert len(conns) == 1
    assert conns[0].train == "S1" and conns[0].call == 0 and conns[0].node == "A"
    assert conns[0].detail == {"feeder": "IC1", "connector": "S1", "transfers": 15, "made": True}

    exits = [e for e in eng1.log.of("EXIT") if e.train == "IC1"]
    assert len(exits) == 1 and exits[0].node == "PW" and exits[0].element == "PW-A.d"

    unfin = [e for e in eng1.log.of("UNFINISHED") if e.train == "F1"]
    assert len(unfin) == 1 and unfin[0].booked == 240 and unfin[0].detail == {"freight": True}


def test_refused_actions_do_not_emit_action_events():
    eng = Engine(
        NET,
        [make_trip("S1", (Call(platform="A1", depart=5), Call(platform="C1", arrive=40)))],
        shift_min=100,
    )
    assert eng.hold("nope", 20).rule == NetRule.UNKNOWN_TRAIN
    assert eng.hold("S1", 0).rule == NetRule.HOLD_PAST
    assert eng.platform("S1", "A1").rule == NetRule.NO_ROUTE
    assert eng.skip("S1", "C").rule == NetRule.SKIP_DESTINATION
    assert eng.route("S1", []).rule == NetRule.BAD_ROUTE
    assert eng.short_turn("S1", "A").rule == NetRule.NO_RETURN
    assert eng.single_line("A-JA.u", True).rule == NetRule.NOT_REVERSIBLE
    assert eng.order(("A-JA.u", 0), []).rule == NetRule.BAD_ROUTE
    assert eng.log.of("ACTION") == []

    assert eng.hold("S1", 15) is None
    actions = eng.log.of("ACTION")
    assert len(actions) == 1
    assert actions[0].train == "S1"
    assert actions[0].detail == {"action": "hold", "args": {"train": "S1", "until": 15}}


def test_connection_pairs_multi_visit_hub_and_ignores_cancelled_feeder():
    """A connector calling at B outbound then B inbound only resolves a late connection on the inbound call,
    and a cancelled feeder running empty through the hub does not count as delivering passengers."""
    trips = [
        make_trip(
            "RE1",
            (
                Call(platform="A1", depart=0, min_dwell=1, board=80),
                Call(platform="B1", arrive=14, depart=16, min_dwell=2, alight=10, board=10),
                Call(platform="C3", arrive=26, depart=36, min_dwell=10, alight=40, board=40),
                Call(platform="B2", arrive=46, depart=55, min_dwell=2, alight=10, board=20),
            ),
            code="RE",
            exit_node="PW",
        ),
        make_trip(
            "IC2",
            (
                Call(platform="C1", depart=36, min_dwell=1, board=100),
                Call(platform="B2", arrive=46, depart=48, min_dwell=2, alight=30, board=10),
                Call(platform="A2", arrive=62, alight=80),
            ),
            code="IC",
        ),
    ]
    conn = Connection(feeder="IC2", connector="RE1", node="B", transfers=12, min_transfer_min=5)
    eng = Engine(NET, trips, shift_min=120, connections=[conn])
    run_auto(eng)
    events = eng.log.of("CONNECTION")
    assert len(events) == 1
    assert events[0].call == 3  # RE1's inbound call at B2, NOT outbound call 1 at B1
    assert events[0].detail["made"] is True

    # Now cancel feeder IC2 before it leaves C1: it still runs empty through B2, but delivers no passengers.
    eng_cancelled = Engine(NET, trips, shift_min=120, connections=[conn])
    assert eng_cancelled.cancel("IC2") is None
    run_auto(eng_cancelled)
    events_cancelled = eng_cancelled.log.of("CONNECTION")
    assert len(events_cancelled) == 1
    assert events_cancelled[0].call == 3
    assert events_cancelled[0].detail["made"] is False


def test_end_shift_closes_episode_without_looping_on_open_conflicts():
    trips = [
        make_trip("IC1", (Call(platform="A1", depart=5), Call(platform="C1", arrive=40)), code="IC"),
        make_trip("L4", (Call(platform="A2", depart=5), Call(platform="N1", arrive=30)), code="S"),
        make_trip("F3", (Call(platform="B3", depart=10), Call(platform="P1", arrive=50)), code="FB"),
    ]
    eng = Engine(NET, trips, shift_min=100)
    first = eng.advance()
    assert first.kind == "CONFLICT"
    # Agent stops early without resolving the IC1/L4 conflict: F3 still finishes, IC1 & L4 stay unfinished.
    end = eng.end_shift()
    assert (end.tick, end.kind) == (100, "SHIFT_END")
    assert eng.end_shift() == end  # idempotent
    assert len(eng.log.of("SHIFT_END")) == 1
    assert {e.train for e in eng.log.of("UNFINISHED")} == {"IC1", "L4"}
    missed = [(e.train, e.call, e.detail["why"]) for e in eng.log.of("CALL_MISSED")]
    assert missed == [("IC1", 1, "shift_end"), ("L4", 1, "shift_end")]
    assert ("F3", "done", "P1") in [r[:3] for r in eng.state()]


# ------------------------------------------------------------------ crew duty


def test_crew_warning_and_immediate_relief_at_crew_platform():
    # Crew starts with 205 min worked -> warning after 5 min (at t=5, 30 min remaining).
    t = make_trip(
        "IC1",
        (Call(platform="A1", depart=20, min_dwell=2), Call(platform="C1", arrive=60)),
        code="IC",
        crew_min=CREW_LIMIT_MIN - CREW_WARNING_MIN - 5,
    )
    eng = Engine(NET, [t], shift_min=120)
    assert eng.crew_remaining("IC1") == 35
    d = eng.advance()
    assert (d.tick, d.kind, d.trains) == (5, "CREW", ("IC1",))
    assert eng.crew_remaining("IC1") == CREW_WARNING_MIN
    warn = eng.log.of("CREW_WARNING")
    assert len(warn) == 1 and warn[0].detail == {"remaining_min": CREW_WARNING_MIN}

    # Immediate relief at A1 (crew_relief=True): holds until t=13, then fresh 240 min crew takes over.
    assert eng.relieve_crew(train="IC1") is None
    assert eng.crew_remaining("IC1") is None
    assert eng.relieve_crew("IC1").rule == NetRule.CREW_RELIEF  # already being relieved
    relief = eng.log.of("CREW_RELIEF")
    assert len(relief) == 1 and relief[0].element == "A1" and relief[0].detail == {"until": 5 + CREW_RELIEF_MIN}

    run_auto(eng)
    assert ("IC1", "done", "C1") in [r[:3] for r in eng.state()]
    assert eng.log.of("CREW_TIMEOUT") == []


def test_crew_deferred_relief_and_refusals():
    # S1 starts at N1 (NOT a crew_relief platform) heading to A2 (crew_relief=True), then PN? A2->C1 via turnaround.
    t_relief = make_trip(
        "S1",
        (
            Call(platform="N1", depart=2, min_dwell=1),
            Call(platform="A2", arrive=22, depart=32, min_dwell=10),
            Call(platform="N1", arrive=58),
        ),
        code="S",
        crew_min=200,
    )
    t_no_point = make_trip(
        "S2",
        (Call(platform="B1", depart=5, min_dwell=1), Call(platform="N2", arrive=35)),
        code="S",
        crew_min=200,
    )
    t_late = make_trip("S3", (Call(platform="C1", depart=130), Call(platform="B2", arrive=145)), code="S")
    eng = Engine(NET, [t_relief, t_no_point, t_late], shift_min=150)
    # S2 calls at B1 and N2, neither is in CREW_RELIEF
    assert eng.relieve_crew("S2").rule == NetRule.NOT_CREW_POINT
    assert eng.relieve_crew("nope").rule == NetRule.UNKNOWN_TRAIN

    # S1 stands at N1 (not a crew point) but calls at A2 (crew point) -> queues relieve_next
    assert eng.relieve_crew("S1") is None
    assert eng.trains["S1"].relieve_next is True
    assert eng.relieve_crew("S1").rule == NetRule.CREW_RELIEF

    while eng.trains["S1"].phase != "done":
        d = eng.advance()
        if d.kind == "CONFLICT":
            for c in d.conflicts:
                eng.order(c.key, sorted(c.trains))
    assert eng.tick < eng.shift_min
    assert eng.relieve_crew("S1").rule == NetRule.TRAIN_DONE

    run_auto(eng)
    reliefs = eng.log.of("CREW_RELIEF")
    assert len(reliefs) == 1 and reliefs[0].train == "S1" and reliefs[0].element == "A2"
    assert ("S1", "done", "N1") in [r[:3] for r in eng.state()]
    assert eng.relieve_crew("S1").rule == NetRule.SHIFT_ENDED


def test_crew_timeout_blocks_for_45_minutes_and_recovers():
    # Crew starts at 235 min on a train departing N1 at t=0 -> times out mid-journey at t=5 for 45 min.
    t = make_trip(
        "S1",
        (Call(platform="N1", depart=0, min_dwell=1), Call(platform="A2", arrive=22)),
        code="S",
        crew_min=235,
    )
    eng = Engine(NET, [t], shift_min=120)
    # Warning fires immediately at t=0 from __init__
    d1 = eng.advance()
    assert (d1.tick, d1.kind, d1.trains) == (0, "CREW", ("S1",))
    d2 = eng.advance()
    assert (d2.tick, d2.kind, d2.trains) == (5, "CREW", ("S1",))
    timeouts = eng.log.of("CREW_TIMEOUT")
    assert len(timeouts) == 1
    assert timeouts[0].tick == 5 and timeouts[0].detail == {"blocked_min": CREW_TIMEOUT_MIN}
    assert eng.crew_remaining("S1") is None

    # Undisturbed N1 -> A2 takes 17 min; +45 min crew timeout + 1 min restart = 63
    base_eng = Engine(
        NET, [make_trip("S1", (Call(platform="N1", depart=0, min_dwell=1), Call(platform="A2", arrive=22)))], 120
    )
    run_auto(base_eng)
    undisturbed_arr = base_eng.log.of("ARRIVE")[0].tick

    run_auto(eng)
    arrivals = [e for e in eng.log.of("ARRIVE") if e.train == "S1"]
    assert len(arrivals) == 1 and arrivals[0].tick == undisturbed_arr + CREW_TIMEOUT_MIN + 1

    # Also check portal entry when crew_min is already at CREW_LIMIT_MIN: must block on entry track for 45 min.
    t_portal = make_trip(
        "IC9",
        (Call(platform="A1", arrive=8, depart=10), Call(platform="C1", arrive=40)),
        code="IC",
        entry_node="PW",
        entry_tick=2,
        crew_min=CREW_LIMIT_MIN,
    )
    eng_p = Engine(NET, [t_portal], shift_min=150)
    seen = run_auto(eng_p)
    assert any(d.kind == "CREW" and d.tick == 2 for d in seen)
    a1_arr = [e for e in eng_p.log.of("ARRIVE") if e.element == "A1"][0]
    assert a1_arr.tick >= 2 + CREW_TIMEOUT_MIN + 5


# ------------------------------------------------------------------ gridlock


def test_gridlock_true_positive_on_cyclic_standoff_while_other_trains_finish():
    """Eastbound E1 on JA-N meets W1 in N1 and W2 in N2 wanting JA-N: true permanent gridlock on the branch,
    while trunk freight F1 still completes its shift unaffected."""
    trips = [
        make_trip("E1", (Call(platform="A1", depart=0), Call(platform="N1", arrive=30)), code="S"),
        make_trip("W1", (Call(platform="N1", depart=20), Call(platform="A2", arrive=50)), code="S"),
        make_trip("W2", (Call(platform="N2", depart=20), Call(platform="A2", arrive=55)), code="S"),
        make_trip("F1", (Call(platform="B3", depart=5), Call(platform="P1", arrive=45)), code="FB"),
    ]
    eng = Engine(NET, trips, shift_min=100)
    decisions = run_auto(eng)
    grid_decisions = [d for d in decisions if d.kind == "GRIDLOCK"]
    assert len(grid_decisions) == 1
    assert set(grid_decisions[0].trains) == {"E1", "W1", "W2"}
    grid_events = eng.log.of("GRIDLOCK")
    assert len(grid_events) == 1 and grid_events[0].detail == {"trains": ["E1", "W1", "W2"]}
    assert ("F1", "done", "P1") in [r[:3] for r in eng.state()]


def test_gridlock_true_positive_when_running_into_permanently_occupied_terminal_platform():
    """P1 terminates in C1 at t=0 (parked for shift); IC2 runs onto JB-C.u booked into C1 with no escape."""
    trips = [
        make_trip("P1", (Call(platform="C1", arrive=0),), code="IC"),
        make_trip("IC2", (Call(platform="B1", depart=0), Call(platform="C1", arrive=20)), code="IC"),
    ]
    eng = Engine(NET, trips, shift_min=80)
    decisions = run_auto(eng)
    assert [d.kind for d in decisions].count("GRIDLOCK") == 1
    assert eng.log.of("GRIDLOCK")[0].detail == {"trains": ["IC2"]}


def test_gridlock_no_false_positive_when_waiting_behind_moving_held_or_closed_obstructions():
    """Trains queued behind a moving train, a held train, a breakdown, a possession, or a divertable intermediate
    stop must never trigger a false GRIDLOCK."""
    # 1. Platoon of 3 trains on trunk behind slow freight + branch train waiting at flat junction
    platoon = [
        make_trip("F1", (Call(platform="A3", depart=0), Call(platform="B3", arrive=40)), code="FB"),
        make_trip("IC1", (Call(platform="A1", depart=1), Call(platform="C1", arrive=45)), code="IC"),
        make_trip("S1", (Call(platform="A2", depart=2), Call(platform="N1", arrive=45)), code="S"),
    ]
    disruptions = [
        Disruption(id="B1", kind="breakdown", start=6, end=16, reveal=6, train="F1"),
        Disruption(id="P1", kind="possession", start=10, end=25, reveal=0, elements=("JA-N",)),
    ]
    eng = Engine(NET, platoon, shift_min=140, disruptions=disruptions)
    run_auto(eng)
    assert eng.log.of("GRIDLOCK") == []
    assert all(r.phase == "done" for r in eng.state())

    # 2. Running train approaching intermediate platform A1 occupied by parked train P0 can divert via A2
    divertable = [
        make_trip("P0", (Call(platform="A1", arrive=0),), code="IC"),
        make_trip(
            "IC2",
            (Call(platform="A1", arrive=8, depart=10), Call(platform="C1", arrive=45)),
            code="IC",
            entry_node="PW",
            entry_tick=0,
        ),
    ]
    eng2 = Engine(NET, divertable, shift_min=100)
    d = eng2.advance()
    assert d.kind == "BLOCKED" and eng2.log.of("GRIDLOCK") == []
    assert eng2.route("IC2", ["A2"]) is None
    run_auto(eng2)
    assert eng2.log.of("GRIDLOCK") == []
    assert ("IC2", "done", "C1") in [r[:3] for r in eng2.state()]
