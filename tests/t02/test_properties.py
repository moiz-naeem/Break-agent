import copy
import os
import subprocess
import sys
import time

import pytest

from railway.sim.engine import Simulator
from railway.sim.model import Authorize, Hold, Scenario
from railway.util import canonical_json
from tests.helpers import load_fixture


def make_sim(shift=60):
    data = load_fixture("mini3")
    data["shift_ticks"] = shift
    return Simulator.load(Scenario.model_validate(data))


def test_T02_41_one_train_per_resource():
    sim = make_sim()
    assert sim.apply(Authorize(train="E1", station="S2", track="L1")).ok
    assert sim.apply(Authorize(train="L5", station="S3", track="main")).ok
    assert all(holder in sim._trains for holder in sim._holders.values())
    assert len(sim._holders) == len(set(sim._holders))
    occupied = [
        f"{t.station}/{t.track}"
        for t in sim.state().trains
        if t.phase == "at_station" and t.station in {"S1", "S2", "S3"}
    ]
    assert len(occupied) == len(set(occupied))


def test_T02_42_no_move_without_authority():
    sim = make_sim()
    sim.advance()
    sim.advance(max_tick=20)
    assert all(t.phase != "in_section" for t in sim.state().trains)


def test_T02_43_departure_never_precedes_timed_stop():
    sim = make_sim()
    sim.apply(Authorize(train="L5", station="S2", track="main"))
    sim.advance()  # F21 entry at t=2
    sim.advance(max_tick=4)
    assert sim.state().trains[2].station == "S3"
    sim.advance(max_tick=5)
    assert sim.state().trains[2].section_enter >= 5


def test_T02_44_loop_fit_and_platform_rules_hold():
    sim = make_sim()
    assert sim.apply(Authorize(train="E1", station="S2", track="L2")).rule.value == "R-PLATFORM"
    sim.advance()
    assert sim.apply(Authorize(train="F21", station="S2", track="L2")).ok
    for train in sim.state().trains:
        if train.station in {"S1", "S2", "S3"} and train.track not in {None, "main"}:
            station = sim._scenario.network.stations[int(train.station[1:]) - 1]
            loop = next(x for x in station.loops if x.id == train.track)
            assert loop.length_m >= {"express": 200, "local": 120, "freight": 600}[train.type]
            assert loop.kind != "goods" or train.type in {"freight", "engineering"}


def test_T02_23_refused_actions_are_noop():
    sim = make_sim()
    actions = [
        Authorize(train="X9", station="S2", track="L1"),
        Authorize(train="E1", station="S9", track="main"),
        Authorize(train="E1", station="S3", track="L1"),
        Hold(train="E1", until_tick=0),
    ]
    for action in actions:
        before = canonical_json(sim.state().model_dump(mode="json"))
        applied = sim.apply(action)
        assert not applied.ok
        assert canonical_json(sim.state().model_dump(mode="json")) == before


def test_T02_46_refusal_is_noop():
    sim = make_sim()
    before = canonical_json(sim.state().model_dump(mode="json"))
    result = sim.apply(Authorize(train="F21", station="S2", track="L2"))
    assert not result.ok
    assert canonical_json(sim.state().model_dump(mode="json")) == before


def test_T02_24_check_equals_apply_on_clone():
    actions = [
        Authorize(train="E1", station="S2", track="L1"),
        Authorize(train="X9", station="S2", track="L1"),
        Authorize(train="F21", station="S2", track="L2"),
        Hold(train="E1", until_tick=0),
    ]
    for action in actions:
        checked_on = make_sim()
        applied_on = make_sim()
        before = canonical_json(checked_on.state().model_dump(mode="json"))
        checked = checked_on.check(action)
        applied = applied_on.clone().apply(action)
        assert (checked.ok, checked.rule, checked.message) == (applied.ok, applied.rule, applied.message)
        assert canonical_json(checked_on.state().model_dump(mode="json")) == before


def test_T02_48_clone_and_replay_are_independent_and_deterministic():
    original = make_sim()
    branch = original.clone()
    action = Authorize(train="E1", station="S2", track="L1")
    assert branch.apply(action).ok
    assert original.state().trains[0].authority is None
    # Exact replay of the same action sequence gives the same observable trajectory.
    left, right = make_sim(), make_sim()
    for run in (left, right):
        run.apply(action)
        run.advance()
    assert canonical_json(left.state().model_dump(mode="json")) == canonical_json(right.state().model_dump(mode="json"))
    assert canonical_json(left.cost().model_dump(mode="json")) == canonical_json(right.cost().model_dump(mode="json"))


def test_T02_47_replay_identical():
    actions = [Authorize(train="E1", station="S2", track="L1")]
    runs = [make_sim(), make_sim()]
    for run in runs:
        for action in actions:
            run.apply(action)
        run.advance()
    assert canonical_json(runs[0].state().model_dump(mode="json")) == canonical_json(
        runs[1].state().model_dump(mode="json")
    )


def test_T02_45_cost_is_nonnegative_and_total_is_component_sum():
    sim = make_sim()
    previous = sim.cost().model_dump(mode="json")
    sim.apply(Authorize(train="E1", station="S2", track="L1"))
    sim.advance()
    sim.end_shift()
    current = sim.cost().model_dump(mode="json")
    keys = ("pax_lateness", "freight_lateness", "crew_timeout", "unfinished")
    components = [current[k] for k in keys]
    assert all(v >= 0 for v in components)
    assert current["total"] == sum(components)
    assert all(current[k] >= previous[k] for k in keys)


def test_T02_49_replay_is_stable_across_hash_seeds():
    code = (
        "from railway.sim.engine import Simulator; from railway.sim.model import Scenario,Authorize; "
        "from tests.helpers import load_fixture; from railway.util import canonical_json; "
        "s=Simulator.load(Scenario.model_validate(load_fixture('mini3'))); "
        "s.apply(Authorize(train='E1',station='S2',track='L1')); s.advance(); "
        "print(canonical_json(s.state().model_dump(mode='json')))"
    )
    outs = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        outs.append(proc.stdout)
    assert outs[0] == outs[1]


@pytest.mark.slow
def test_T02_50_clone_and_advance_speed():
    data = load_fixture("mini3")
    data["shift_ticks"] = 360
    originals = copy.deepcopy(data["trains"])
    trains = []
    next_ids = {"express": 1, "freight": 21, "local": 5}
    for i in range(14):
        train = copy.deepcopy(originals[i % len(originals)])
        prefix = {"express": "E", "local": "L", "freight": "F"}[train["type"]]
        next_ids[train["type"]] += 1
        train["id"] = f"{prefix}{next_ids[train['type']]}"
        train["entry_tick"] = i * 5
        trains.append(train)
    data["trains"] = sorted(trains, key=lambda t: t["id"])
    sim = Simulator.load(Scenario.model_validate(data))
    start = time.perf_counter()
    for _ in range(1000):
        clone = sim.clone()
        clone.end_shift()
    assert time.perf_counter() - start < 5
