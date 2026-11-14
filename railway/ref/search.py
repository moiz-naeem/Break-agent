"""Anchor search for tiers the exact solver can't cover (events on): a portfolio of reference variants.

Each variant is the scripted reference with other look-ahead settings, optionally *omniscient*: it plans on the
true simulator, so it knows every future breakdown, late entry and possession. That is knowledge no agent has,
which is fine for an anchor: C_best is "the best plan we know exists", and every plan found here replays
exactly through a fresh session. Variants are independent, so they run in parallel (Modal: infra/modal).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from railway.tools.client import InProcClient
from railway.tools.session import Session

from .best import plan_from_log, replay_plan
from .reference import Reference


@dataclass(frozen=True)
class Variant:
    horizon: int = 90
    max_candidates: int = 5
    lookahead_stations: int = 4
    omniscient: bool = False

    @property
    def name(self) -> str:
        return f"h{self.horizon}-c{self.max_candidates}-l{self.lookahead_stations}{'-omni' if self.omniscient else ''}"


PORTFOLIO = [
    Variant(h, c, look, omni)
    for h, c, look, omni in itertools.product((60, 90, 150, 240), (5, 8), (4, 6), (False, True))
]


def play_variant(scenario, variant: Variant) -> tuple[float, list[dict]]:
    """Run one variant through a real session; (cost, replayable plan)."""
    session = Session(scenario)
    ref = Reference(variant.horizon, variant.max_candidates, variant.lookahead_stations)
    if variant.omniscient:
        ref.belief = lambda static, state: session.sim.clone()
    ref.run(InProcClient(session))
    cost = session.close().cost.total
    plan = plan_from_log(session.log)
    replayed = replay_plan(scenario, plan)
    if abs(replayed - cost) > 1e-6:
        raise RuntimeError(f"{variant.name}: plan replays to {replayed}, not {cost}")
    return cost, plan


def variant_from_name(name: str) -> Variant:
    """Inverse of `Variant.name` ("h150-c8-l6-omni")."""
    parts = name.split("-")
    return Variant(int(parts[0][1:]), int(parts[1][1:]), int(parts[2][1:]), parts[-1] == "omni")


def improve_entry(entry: dict, results: dict[str, float]) -> dict:
    """Lower an accepted entry's C_best to the cheapest variant result, if any beats it."""
    name = min(results, key=lambda k: (results[k], k))
    if results[name] < entry["c_best"] - 1e-6:
        return {**entry, "c_best": results[name], "best_source": f"search:{name}", "best_proven": False}
    return entry
