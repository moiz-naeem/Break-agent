import copy

import pytest
from pydantic import ValidationError

from railway.sim.constants import RULES_VERSION
from railway.sim.cost import freight_cost, pax_cost
from railway.sim.engine import Simulator
from railway.sim.model import Authorize, CancelAuthority, Hold, Scenario
from railway.sim.timing import project, travel_ticks
from railway.util import canonical_json
from tests.helpers import load_fixture


def sim(name="mini3"):
    return Simulator.load(Scenario.model_validate(load_fixture(name)))


def test_T02_01_fixture_roundtrip_and_strict_schema():
    raw = load_fixture("mini3")
    model = Scenario.model_validate(raw)
    assert canonical_json(model.model_dump(mode="json")) == canonical_json(raw)
    for fixture in ("mini2_gridlock", "mini3_noloops"):
        Scenario.model_validate(load_fixture(fixture))
    for patch in ({"foo": 1},):
        bad = copy.deepcopy(raw)
        bad.update(patch)
        with pytest.raises(ValidationError):
            Scenario.model_validate(bad)
    for path in ("network.stations.0", "trains.0", "trains.0.stops.0"):
        bad = copy.deepcopy(raw)
        node = bad
        for part in path.split("."):
            node = node[int(part)] if isinstance(node, list) else node[part]
        node["foo"] = 1
        with pytest.raises(ValidationError):
            Scenario.model_validate(bad)


def test_T02_02_rules_and_factors():
    d = load_fixture("mini3")
    d["rules_version"] = "old"
    with pytest.raises(ValidationError, match="rules_version"):
        Scenario.model_validate(d)


def test_T02_07_next_service_reference_and_direction():
    d = load_fixture("mini3")
    d["trains"][0]["next_service"] = "L5"
    Scenario.model_validate(d)
    d["trains"][0]["next_service"] = "L99"
    with pytest.raises(ValidationError, match="next_service"):
        Scenario.model_validate(d)
    d = load_fixture("mini3")
    next_train = copy.deepcopy(d["trains"][0])
    next_train["id"] = "E2"
    d["trains"][0]["next_service"] = "E2"
    d["trains"].append(next_train)
    d["trains"].sort(key=lambda item: item["id"])
    with pytest.raises(ValidationError, match="opposite direction"):
        Scenario.model_validate(d)


@pytest.mark.parametrize(
    ("length", "line", "train", "restriction", "expected"),
    [
        (6000, 120, 140, None, 3),
        (6000, 120, 100, None, 4),
        (6000, 120, 70, None, 6),
        (6000, 120, 140, 40, 9),
        (1, 140, 140, None, 1),
        (4500, 80, 100, None, 4),
        (8000, 140, 140, None, 4),
        (9500, 140, 70, None, 9),
    ],
)
def test_T02_03_travel_ticks(length, line, train, restriction, expected):
    assert travel_ticks(length, line, train, restriction) == expected


def test_T02_04_authorize_refuse_hold_cancel_and_clone():
    s = sim()
    r = s.apply(Authorize(train="E1", station="S2", track="L1"))
    assert r.ok and r.message == "GRANTED E1 -> S2/L1 (arrive t=6)"
    assert s.state().trains[0].authority.path == ["YW-S1", "S1/main", "S1-S2", "S2/L1"]
    before = canonical_json(s.state().model_dump(mode="json"))
    refused = s.apply(Authorize(train="F21", station="S2", track="L2"))
    assert not refused.ok and refused.rule.value == "R-NOT-IN-NETWORK"
    assert canonical_json(s.state().model_dump(mode="json")) == before
    c = s.clone()
    c.apply(Hold(train="E1", until_tick=5))
    assert canonical_json(c.state().model_dump(mode="json")) != canonical_json(s.state().model_dump(mode="json"))


def test_T02_26_episode_A_exact():
    s = sim()
    assert s.apply(Authorize(train="E1", station="S2", track="L1")).ok
    a = s.advance()
    assert (a.tick, [(x.kind, x.train) for x in a.reasons]) == (2, [("READY", "F21")])
    assert s.apply(Authorize(train="L5", station="S2", track="main")).message.endswith("(arrive t=11)")
    a = s.advance()
    assert a.tick == 6 and [(x.kind, x.train) for x in a.reasons] == [("AUTHORITY_END", "E1")]
    assert s.state().trains[0].arrivals[-1].actual == 6
    assert not s.apply(Authorize(train="E1", station="YE", track="main")).ok
    assert s.advance().tick == 7
    assert s.advance().tick == 11
    assert s.apply(Authorize(train="E1", station="YE", track="main")).ok
    s.end_shift()
    assert s.cost().pax_lateness == 24.0
    assert s.cost().unfinished == 745.0  # F21 freight: 200 + 1000 x 60/240 + 295
    assert s.cost().total == 769.0


def test_T02_32_do_nothing_cost():
    s = sim()
    s.end_shift()
    assert s.tick == 60 and s.cost().total == 1485.0


def test_T02_29_time_cap_and_shift_end():
    d = load_fixture("mini3")
    d["shift_ticks"] = 200
    s = Simulator.load(Scenario.model_validate(d))
    assert s.advance().tick == 2
    assert s.advance().tick == 62
    assert s.advance().tick == 122
    assert s.advance().tick == 182
    end = s.advance()
    assert end.tick == 200 and end.done
    assert [r.kind for r in end.reasons] == ["SHIFT_END"]
    assert s.apply(Authorize(train="E1", station="S2", track="L1")).rule.value == "R-SHIFT-ENDED"
    assert s.advance().done


@pytest.mark.parametrize(
    ("kind", "pax", "late", "terminal", "expected"),
    [
        ("express", 200, 7, False, 30.0),
        ("express", 200, 7, True, 60.0),
        ("local", 100, 2, False, 0.0),
        ("local", 150, 10, True, 24.0),
        ("express", 100, -5, True, 0.0),
    ],
)
def test_T02_35_passenger_cost_formula(kind, pax, late, terminal, expected):
    assert pax_cost(kind, pax, late, terminal) == expected


@pytest.mark.parametrize(("late", "expected"), [(10, 5.0), (-3, 0.0)])
def test_T02_36_freight_cost_formula(late, expected):
    assert freight_cost(late) == expected


def test_T02_34_departure_respects_dwell_and_timetable():
    s = sim()
    s.apply(Authorize(train="L5", station="S2", track="main"))
    assert s.advance(max_tick=4).tick == 2
    assert s.advance(max_tick=4).tick == 4
    assert s.state().trains[2].station == "S3"
    assert s.advance(max_tick=6).tick == 6
    assert s.state().trains[2].phase == "in_section"
    d = load_fixture("mini3")
    d["trains"][2]["stops"][0]["dep"] = 8
    delayed = Simulator.load(Scenario.model_validate(d))
    delayed.apply(Authorize(train="L5", station="S2", track="main"))
    assert delayed.advance(max_tick=7).tick == 2
    assert delayed.advance(max_tick=7).tick == 7
    assert delayed.state().trains[2].station == "S3"
    assert delayed.advance(max_tick=8).tick == 8
    assert delayed.state().trains[2].phase == "in_section"


def test_T02_07_public_schema_excludes_hidden_scenario_data():
    state = sim().state().model_dump(mode="json")
    static = sim().static_info().model_dump(mode="json")
    assert not {"events", "seed", "scenario_id"}.intersection(state)
    assert not {"events", "seed", "scenario_id"}.intersection(static)
    assert RULES_VERSION == "0.2-provisional"


def test_T02_17_section_reserved_refusal_and_resource_release():
    s = sim()
    assert s.apply(Authorize(train="E1", station="S2", track="L1")).ok
    r = s.apply(Authorize(train="L5", station="S1", track="main"))
    assert r.rule.value == "R-SECTION-RESERVED"
    assert r.message == "REFUSED R-SECTION-RESERVED: S1-S2 reserved by E1"
    assert s.apply(CancelAuthority(train="E1")).ok
    assert s.apply(Authorize(train="L5", station="S1", track="main")).ok


def test_T02_20_destination_track_loop_length_platform_and_unknowns():
    s = sim()
    assert s.apply(Authorize(train="E1", station="S3", track="L1")).rule.value == "R-LOOP-LEN"
    assert s.apply(Authorize(train="E1", station="S2", track="L2")).rule.value == "R-PLATFORM"
    assert s.apply(Authorize(train="E1", station="S9", track="L1")).rule.value == "R-UNKNOWN-STATION"
    assert s.apply(Authorize(train="E1", station="S2", track="L7")).rule.value == "R-UNKNOWN-TRACK"
    assert s.apply(Authorize(train="X9", station="S2", track="L1")).rule.value == "R-UNKNOWN-TRAIN"
    assert s.apply(Authorize(train="E1", station="YW", track="main")).rule.value == "R-WRONG-DIRECTION"
    s.advance()
    assert s.apply(Authorize(train="F21", station="S1", track="L1")).rule.value == "R-LOOP-LEN"
    assert s.apply(Authorize(train="F21", station="S2", track="L2")).ok


def test_T02_21_destination_track_is_reserved_until_departure():
    s = sim()
    assert s.apply(Authorize(train="E1", station="S2", track="L1")).ok
    s.advance()
    s.advance(max_tick=6)
    # E1 still occupies its loop after its authority ends.
    assert s.apply(Authorize(train="L5", station="S2", track="L1")).rule.value == "R-TRACK-OCCUPIED"


def test_T02_18_main_track_is_reserved_by_authority():
    s = sim()
    assert s.apply(Authorize(train="L5", station="S2", track="main")).ok
    refusal = s.apply(Authorize(train="E1", station="YE", track="main"))
    assert refusal.rule.value == "R-MAIN-BLOCKED"
    assert refusal.message == "REFUSED R-MAIN-BLOCKED: S2/main reserved by L5"


def test_T02_20_hold_and_cancel_at_station():
    s = sim()
    assert s.apply(Hold(train="E1", until_tick=0)).rule.value == "R-HOLD-PAST"
    assert s.apply(Hold(train="E1", until_tick=5)).ok
    assert s.apply(Authorize(train="E1", station="S2", track="L1")).rule.value == "R-HELD"
    # A hold does not cancel a previously granted authority.
    s.apply(Hold(train="E1", until_tick=6))
    assert s.state().trains[0].authority is None
    s = sim()
    s.apply(Authorize(train="E1", station="S2", track="L1"))
    assert s.apply(Hold(train="E1", until_tick=5)).ok
    assert s.state().trains[0].authority is not None


def test_T02_21_cancel_stationary_frees_reserved_resources():
    s = sim()
    assert s.apply(CancelAuthority(train="E1")).rule.value == "R-NO-AUTHORITY"
    s.apply(Authorize(train="E1", station="S2", track="L1"))
    assert s.apply(CancelAuthority(train="E1")).ok
    assert s.state().trains[0].authority is None
    assert s.apply(Authorize(train="L5", station="S1", track="main")).ok


def test_T02_30_ready_trigger_is_edge_triggered():
    s = sim()
    assert s.advance().tick == 2
    assert [(r.kind, r.train) for r in s.advance().reasons] != [("READY", "F21")]


def test_T02_37_total_is_sum_of_cost_components():
    s = sim()
    s.apply(Authorize(train="E1", station="S2", track="L1"))
    s.advance()
    s.end_shift()
    c = s.cost()
    assert c.total == sum(
        c.model_dump().get(k, 0) for k in ("pax_lateness", "freight_lateness", "crew_timeout", "reversal", "unfinished")
    )


def test_T02_33_actions_after_done_are_refused():
    s = sim()
    s.end_shift()
    assert s.apply(Authorize(train="E1", station="S2", track="L1")).rule.value == "R-SHIFT-ENDED"
    before = canonical_json(s.state().model_dump(mode="json"))
    assert s.advance().done
    assert canonical_json(s.state().model_dump(mode="json")) == before


def test_T02_31_advance_max_tick():
    s = sim()
    s.apply(Authorize(train="E1", station="S2", track="L1"))
    assert s.advance(max_tick=5).tick == 2
    result = s.advance(max_tick=5)
    assert result.tick == 5 and result.reasons == []
    assert s.state().trains[0].section == "S1-S2"
    assert s.state().trains[0].section_exit == 6


def test_T02_22_cancel_while_moving_truncates_to_next_station():
    s = sim()
    s.apply(Authorize(train="E1", station="YE", track="main"))
    assert s.advance(max_tick=1).tick == 1
    assert s.apply(CancelAuthority(train="E1")).ok
    assert s.state().trains[0].authority.dest_station == "S1"
    assert s.state().trains[0].authority.path == ["S1/main"]
    s.advance()
    assert s.advance().tick == 3
    assert s.state().trains[0].station == "S1"
    assert s.state().trains[0].track == "main"


def test_T02_11_extension_while_moving_keeps_timed_station_stop():
    s = sim()
    assert s.apply(Authorize(train="E1", station="S2", track="L1")).ok
    assert s.advance().tick == 2
    assert s.advance(max_tick=4).tick == 4
    assert s.state().trains[0].section == "S1-S2"
    assert s.apply(Authorize(train="E1", station="YE", track="main")).ok
    assert s.advance(max_tick=6).tick == 6
    assert s.state().trains[0].station == "S2"
    assert s.state().trains[0].track == "main"
    assert s.state().trains[0].authority.dest_station == "YE"
    assert s.advance().tick == 13


@pytest.mark.parametrize(
    "patch",
    [
        {"network.sections": []},
        {"network.sections.1.id": "S2-S3", "network.sections.1.west": "S2", "network.sections.1.east": "S3"},
        {"network.sections.1.id": "S1-S3", "network.sections.1.east": "S3"},
        {"network.sections.0.west": "S1"},
        {"network.sections.3.east": "S2"},
        {"network.sections.1.west": "S2", "network.sections.1.east": "S1"},
        {"network.stations.0.loops.0.length_m": 200},
        {
            "network.stations.0.loops": [
                {"id": "L1", "length_m": 300, "kind": "platform"},
                {"id": "L1", "length_m": 300, "kind": "goods"},
            ]
        },
        {
            "network.stations.0.loops": [
                {"id": "L1", "length_m": 300, "kind": "platform"},
                {"id": "L2", "length_m": 300, "kind": "goods"},
                {"id": "L3", "length_m": 300, "kind": "goods"},
                {"id": "L1", "length_m": 700, "kind": "goods"},
            ]
        },
        {"network.stations.0.loops.0.kind": "siding"},
        {"trains.0.id": "X1"},
        {"trains.0.type": "local"},
        {"trains.2.stops.1.station": "S1"},
        {"trains.0.stops.0.dep": 5, "trains.0.stops.0.arr": 6},
        {"trains.0.stops.0.dep": None},
        {"trains.0.entry_tick": -1},
        {"trains.0.stops.0.station": "YE"},
        {
            "trains.1.stops": [
                {"station": "S2", "arr": 10, "dep": 11, "min_dwell": 1},
                {"station": "YE", "arr": 20, "dep": None, "min_dwell": 0},
            ]
        },
        {"trains.1.id": "E1"},
    ],
)
def test_T02_13_schema_rejects_invalid_network_and_train(patch):
    d = load_fixture("mini3")
    for path, value in patch.items():
        *parents, key = path.split(".")
        node = d
        for part in parents:
            node = node[int(part)] if isinstance(node, list) else node[part]
        if isinstance(node, list):
            node[int(key)] = value
        else:
            node[key] = value
    with pytest.raises(ValidationError):
        Scenario.model_validate(d)


def test_T02_14_free_run_project_matches_fixture_timetable():
    s = sim()
    assert project(s, "E1", s._route("YW", "YE", "main"), 0) == [
        ("YW-S1", 0, 3),
        ("S1-S2", 3, 6),
        ("S2-S3", 7, 10),
        ("S3-YE", 10, 13),
    ]
    assert project(s, "L5", s._route("YE", "YW", "main"), 0) == [
        ("S3-YE", 0, 4),
        ("S2-S3", 5, 9),
        ("S1-S2", 10, 14),
        ("YW-S1", 15, 19),
    ]
    assert project(s, "F21", s._route("YW", "YE", "main"), 2)[-1] == ("S3-YE", 20, 26)
