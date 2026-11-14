"""Exhaustive search of the real simulator, for tiny scenarios: the true optimum the CP-SAT model must match.

At every decision point each train standing without authority may: do nothing, wake up next tick (a hold used
as an alarm clock), or take any grantable authority to any place ahead (every track). Trains are decided one
after another, then time advances. States are memoised on everything that affects the future (cost so far is
left out: the future cost of a state doesn't depend on it).
"""

from __future__ import annotations

import json
import sys

from railway.sim.engine import Simulator
from railway.sim.model import Authorize, Hold

sys.setrecursionlimit(100_000)


def _key(sim) -> str:
    state = sim.state().model_dump(mode="json")
    state.pop("cost")
    state.pop("last_decision_tick")
    return json.dumps([state, sorted(sim._holders.items())], sort_keys=True)


def _options(sim, tid):
    t = sim._trains[tid]
    out = [None]
    if t["phase"] != "at_station" or t["auth"] or (t["hold_until"] is not None and sim.tick < t["hold_until"]):
        return out
    out.append(Hold(train=tid, until_tick=sim.tick + 1))
    for node in sorted(sim._graph.nodes):
        tracks = (
            ["main", *sorted(loop.id for loop in sim._stations[node].loops)]
            if node in sim._stations
            else (sorted(sim._graph.tracks[node]) if not sim._graph.unlimited(node) else ["main"])
        )
        for track in tracks:
            action = Authorize(train=tid, station=node, track=track)
            if sim.check(action).ok:
                out.append(action)
    return out


def optimum(scenario, node_limit: int = 2_000_000) -> float:
    memo: dict[str, float] = {}
    nodes = [0]

    def future(sim) -> float:
        if sim.done:
            return 0.0
        key = _key(sim)
        if key in memo:
            return memo[key]
        nodes[0] += 1
        if nodes[0] > node_limit:
            raise RuntimeError("node limit")
        best = float("inf")
        tids = sorted(sim._trains)

        def decide(sim, i):
            nonlocal best
            if i == len(tids):
                after = sim.clone()
                before = after.cost().total
                after.advance()
                if after.done:
                    after._finalize_cost()
                best = min(best, after.cost().total - before + future(after))
                return
            for option in _options(sim, tids[i]):
                branch = sim
                if option is not None:
                    branch = sim.clone()
                    if not branch.apply(option).ok:
                        continue
                decide(branch, i + 1)

        decide(sim, 0)
        memo[key] = best
        return best

    sim = Simulator.load(scenario)
    return round(sim.cost().total + future(sim), 6)
