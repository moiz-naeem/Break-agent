import os
import re
import subprocess
import sys

import pytest

from railway.gen.generator import generate
from railway.gen.presets import PRESETS
from railway.sim.engine import Simulator
from railway.sim.model import Scenario
from railway.util import canonical_json


def test_T04_10_presets_exist():
    assert {"T1", "T2", "T2-S14", "T3"} <= set(PRESETS)
    # T1/T2 recalibrated in T05/T06 (DESIGN said 8/8/120 and 12/14/360); see docs/gates/T06.md.
    assert (PRESETS["T1"].stations, PRESETS["T1"].trains, PRESETS["T1"].shift_ticks) == (8, 8, 240)
    assert (PRESETS["T2"].stations, PRESETS["T2"].trains, PRESETS["T2"].shift_ticks) == (12, 10, 720)
    assert (PRESETS["T2-S14"].stations, PRESETS["T2-S14"].trains) == (14, 16)
    assert (PRESETS["T3"].stations, PRESETS["T3"].trains, PRESETS["T3"].shift_ticks) == (16, 22, 720)


def test_T04_11_generated_scenario_is_deterministic_valid_and_loadable():
    first = generate(PRESETS["T1"], seed=5)
    second = generate(PRESETS["T1"], seed=5)
    assert canonical_json(first.model_dump(mode="json")) == canonical_json(second.model_dump(mode="json"))
    assert isinstance(Scenario.model_validate(first.model_dump(mode="json")), Scenario)
    sim = Simulator.load(first)
    assert sim.end_shift().done
    assert first.scenario_id.startswith("T1-")


@pytest.mark.parametrize("preset", ["T1", "T2", "T2-S14", "T3"])
def test_T04_15_every_preset_validates_and_finishes(preset):
    scenario = generate(PRESETS[preset], seed=7)
    assert isinstance(Scenario.model_validate(scenario.model_dump(mode="json")), Scenario)
    sim = Simulator.load(scenario)
    result = sim.end_shift()
    assert result.done
    assert sim.cost().total > 0
    assert re.fullmatch(rf"{re.escape(preset)}-[0-9a-f]{{12}}", scenario.scenario_id)


def test_T04_12_cross_process_generation_is_hash_seed_independent(tmp_path):
    outputs = []
    for hash_seed in ("1", "2"):
        target = tmp_path / f"seed-{hash_seed}.json"
        subprocess.run(
            [sys.executable, "-m", "railway.gen.cli", "sample", "--config", "T2", "--seed", "7", "--out", str(target)],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(target.read_bytes())
    assert outputs[0] == outputs[1]


def test_T04_13_scenario_ids_change_with_seed():
    ids = {generate(PRESETS["T2"], seed).scenario_id for seed in range(1, 21)}
    assert len(ids) == 20


def test_T04_14_scenario_id_opaque():
    for seed in (123456, 987654):  # single-digit seeds match a hex digit by chance
        scenario_id = generate(PRESETS["T2"], seed).scenario_id
        assert re.fullmatch(r"T2-[0-9a-f]{12}", scenario_id)
        assert str(seed) not in scenario_id


def test_T04_18b_direct_run_is_on_time():
    """Regression: stop times are absolute. A train sent straight to its yard at entry is not late."""
    from railway.sim.model import Authorize

    scenario = generate(PRESETS["T1"], 5)
    train = max(scenario.trains, key=lambda t: t.entry_tick)  # late entry: relative times would show as early
    sim = Simulator.load(scenario)
    while sim.tick < train.entry_tick:
        sim.advance(max_tick=train.entry_tick)
    yard = train.stops[-1].station
    assert sim.apply(Authorize(train=train.id, station=yard, track="main")).ok
    while sim.tick < train.stops[-1].arr:  # advance() also stops at other trains' decision points
        sim.advance(max_tick=train.stops[-1].arr)
    state = next(t for t in sim.state().trains if t.id == train.id)
    assert state.arrivals and all(a.actual <= a.sched for a in state.arrivals), state.arrivals
