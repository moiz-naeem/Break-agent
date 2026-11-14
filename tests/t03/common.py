"""Shared T03 scenario builders (ticket section 5:`vf(**patches)` = mini3 variant + factors)."""

import copy

from hypothesis import strategies as st

from railway.gen.presets import ALL_FACTORS  # noqa: F401  (re-exported for T03 tests)
from railway.sim.engine import Simulator
from railway.sim.model import Authorize, CancelAuthority, Hold, RelieveCrew, Reverse, Scenario
from tests.helpers import load_fixture

L6 = {
    "id": "L6",
    "type": "local",
    "direction": "east",
    "pax": 100,
    "entry_tick": 20,
    "stops": [
        {"station": "S1", "arr": 24, "dep": 25, "min_dwell": 1},
        {"station": "S2", "arr": 29, "dep": 30, "min_dwell": 1},
        {"station": "S3", "arr": 34, "dep": 35, "min_dwell": 1},
        {"station": "YE", "arr": 39, "dep": None, "min_dwell": 0},
    ],
    "next_service": None,
    "crew_used_at_entry": 0,
}
F22 = {
    "id": "F22",
    "type": "freight",
    "direction": "east",
    "pax": 0,
    "entry_tick": 30,
    "stops": [{"station": "YE", "arr": 54, "dep": None, "min_dwell": 0}],
    "next_service": None,
    "crew_used_at_entry": 0,
}


def vf(*factors, events=(), extra=(), shift=None, **train_patches) -> dict:
    """mini3 + `factors`, extra trains, events; `L5={"next_service": "L6"}` patches a train by id."""
    data = load_fixture("mini3")
    data["trains"] = sorted(data["trains"] + [copy.deepcopy(t) for t in extra], key=lambda t: t["id"])
    assert set(train_patches) <= {t["id"] for t in data["trains"]}, f"unknown train in {sorted(train_patches)}"
    for train in data["trains"]:
        train.update(train_patches.get(train["id"], {}))
    data["factors"] = sorted(set(data["factors"]) | set(factors))
    data["events"] = sorted((dict(e) for e in events), key=lambda e: (e["reveal_tick"], e["id"]))
    if shift is not None:
        data["shift_ticks"] = shift
    return data


def load(data: dict) -> Simulator:
    return Simulator.load(Scenario.model_validate(data))


def tr(sim: Simulator, tid: str):
    return next(t for t in sim.state().trains if t.id == tid)


def kinds(result) -> list[tuple[str, str | None]]:
    return [(r.kind, r.train) for r in result.reasons]


def dump(model) -> dict:
    return model.model_dump(mode="json")


def all_factor_scenario() -> dict:
    """mini3 + L6 (rotation from L5) + F22, all 11 factors, one event of each kind."""
    return vf(
        *ALL_FACTORS,
        extra=(L6, F22),
        shift=90,
        L5={"next_service": "L6"},
        events=[
            {"id": "EV1", "kind": "possession", "reveal_tick": 5, "section": "S2-S3", "start": 50, "end": 60},
            {
                "id": "EV2",
                "kind": "speed_restriction",
                "reveal_tick": 3,
                "sections": ["S3-YE"],
                "speed_kmh": 40,
                "start": 10,
                "end": 40,
            },
            {"id": "EV3", "kind": "late_entry", "reveal_tick": 10, "train": "F22", "new_entry_tick": 45},
            {"id": "EV4", "kind": "breakdown", "reveal_tick": 4, "train": "E1", "tick": 4, "duration": 6},
        ],
    )


TRAINS = ["E1", "F21", "L5", "L6", "F22"]
STATIONS = ["YW", "S1", "S2", "S3", "YE"]
TRACKS = ["main", "L1", "L2"]

actions = st.one_of(
    st.builds(
        Authorize, train=st.sampled_from(TRAINS), station=st.sampled_from(STATIONS), track=st.sampled_from(TRACKS)
    ),
    st.builds(Hold, train=st.sampled_from(TRAINS), until_tick=st.integers(0, 90)),
    st.builds(CancelAuthority, train=st.sampled_from(TRAINS)),
    st.builds(RelieveCrew, train=st.sampled_from(TRAINS)),
    st.builds(Reverse, train=st.sampled_from(TRAINS)),
)
# One step: some actions at the current decision point, then advance (None = to the next decision point).
steps = st.lists(st.tuples(st.lists(actions, max_size=3), st.one_of(st.none(), st.integers(1, 8))), max_size=12)


def run_step(sim: Simulator, acts, jump):
    results = [sim.apply(a) for a in acts]
    adv = sim.advance() if jump is None else sim.advance(max_tick=sim.tick + jump)
    return results, adv
