from railway.sim.cost import COMPONENTS
"""T02 invariants (T02-41..T02-48) re-run with all 11 factors and one event of each kind."""

import os
import subprocess
import sys

from hypothesis import example, given, settings

from railway.sim.constants import TRAIN_TYPES
from railway.sim.model import Authorize, RelieveCrew
from railway.util import canonical_json
from tests.helpers import REPO
from tests.t03.common import all_factor_scenario, dump, load, run_step, steps

COST_KEYS = COMPONENTS


def check_invariants(sim, prev_cost):
    state = sim.state()
    # T02-41: one train per resource.
    tracks = [(t.station, t.track) for t in state.trains if t.phase == "at_station" and t.station not in {"YW", "YE"}]
    assert len(tracks) == len(set(tracks))
    sections = [t.section for t in state.trains if t.phase in {"in_section", "reversing"}]
    assert len(sections) == len(set(sections))
    stops = {t["id"]: {s["station"]: s for s in t["stops"]} for t in sim.static_info().trains}
    stations = {s["id"]: s for s in sim.static_info().network["stations"]}
    for t in state.trains:
        # T02-42: nothing moves without an authority (reversal carries its own).
        if t.phase == "in_section":
            assert t.authority is not None, t.id
        # T02-43: a train never leaves a timed stop before its scheduled departure.
        if t.phase == "in_section" and t.arrivals:
            stop = stops[t.id].get(t.arrivals[-1].station)
            if stop and stop["dep"] is not None:
                assert t.section_enter >= stop["dep"], t.id
        # T02-44: loop length and goods-loop rules hold.
        if t.phase == "at_station" and t.station in stations and t.track != "main":
            loop = next(x for x in stations[t.station]["loops"] if x["id"] == t.track)
            assert loop["length_m"] >= TRAIN_TYPES[t.type]["length_m"]
            assert loop["kind"] != "goods" or t.type in {"freight", "engineering"}
    # T02-45: cost is non-negative, the total is the component sum, components never decrease.
    cost = dump(sim.cost())
    assert all(cost[k] >= 0 for k in COST_KEYS)
    assert cost["total"] == sum(cost[k] for k in COST_KEYS)
    assert all(cost[k] >= prev_cost[k] for k in COST_KEYS)
    return cost


@settings(max_examples=80)
@given(steps)
# Pin a run that moves every train, through the restriction, breakdown and rotation.
@example(
    [
        ([Authorize(train="E1", station="S2", track="L1"), Authorize(train="L5", station="YW", track="main")], None),
        ([Authorize(train="F21", station="S2", track="L2")], None),
        ([RelieveCrew(train="E1"), Authorize(train="E1", station="YE", track="main")], None),
        ([Authorize(train="F21", station="YE", track="main"), Authorize(train="L6", station="YE", track="main")], None),
        ([Authorize(train="F22", station="YE", track="main")], None),
    ]
    * 2
)
def test_T03_37_t02_invariants_with_all_factors(plan):
    sim = load(all_factor_scenario())
    cost = check_invariants(sim, dump(sim.cost()))
    for acts, jump in plan:
        if sim.done:
            break
        for action in acts:
            before = canonical_json(dump(sim.state()))
            checked = sim.check(action)  # T02-24: check == apply on a clone, and check is a no-op
            assert canonical_json(dump(sim.state())) == before
            applied = sim.clone().apply(action)
            assert dump(checked) == dump(applied)
            result = sim.apply(action)
            assert dump(result) == dump(applied)
            if not result.ok:  # T02-46: a refusal changes nothing
                assert canonical_json(dump(sim.state())) == before
        sim.advance() if jump is None else sim.advance(max_tick=sim.tick + jump)
        cost = check_invariants(sim, cost)
    # T02-47/48: replaying the same plan from scratch gives the identical trajectory.
    replay = load(all_factor_scenario())
    for acts, jump in plan:
        if replay.done:
            break
        run_step(replay, acts, jump)
    assert canonical_json(dump(replay.state())) == canonical_json(dump(sim.state()))


def test_T03_38_cross_process_determinism_all_factors():
    code = (
        "from railway.sim.model import Authorize, RelieveCrew; from railway.util import canonical_json; "
        "from tests.t03.common import all_factor_scenario, dump, load; "
        "s = load(all_factor_scenario()); "
        "s.apply(Authorize(train='E1', station='S2', track='L1')); "
        "s.apply(Authorize(train='L5', station='YW', track='main')); "
        "[s.advance() for _ in range(4)]; s.apply(RelieveCrew(train='E1')); s.end_shift(); "
        "print(canonical_json({'state': dump(s.state()), 'cost': dump(s.cost())}))"
    )
    outs = [
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": seed},
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for seed in ("1", "2")
    ]
    assert outs[0] == outs[1] and outs[0].strip()
