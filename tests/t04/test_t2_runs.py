import time

import pytest

from railway.gen.generator import generate
from railway.gen.presets import PRESETS
from railway.gen.seeds import candidate_seeds
from railway.sim.engine import Simulator


@pytest.mark.slow
def test_T04_31_t2_do_nothing_runs_under_one_second_each():
    for seed in candidate_seeds("dev", 20):
        scenario = generate(PRESETS["T2"], seed)
        start = time.perf_counter()
        sim = Simulator.load(scenario)
        result = sim.end_shift()
        assert result.done
        assert sim.cost().total > 0
        assert time.perf_counter() - start < 1
