"""Solver soundness check: its lower bound must not exceed what any real play reaches, and its own plan must
dispatch to exactly the model's objective. Prints one row per scenario; exits 1 on a violation.

    python tests/optimal/check_bound.py G1 0 6
"""

import json
import sys
import time
from pathlib import Path

from railway.gen.generator import generate
from railway.gen.presets import PRESETS
from railway.policies.baselines import Greedy, RandomLegal
from railway.policies.runner import run_policy
from railway.ref.optimal import dispatch, solve
from railway.ref.reference import Reference
from railway.sim.model import Scenario

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def scenarios(args):
    if args[0] == "fixtures":
        for name in ("mini2_gridlock", "mini3_noloops", "mini3"):
            yield name, Scenario.model_validate(json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8")))
        return
    preset, lo, hi = args[0], int(args[1]), int(args[2])
    for seed in range(lo, hi):
        yield f"{preset}-{seed}", generate(PRESETS[preset], seed)


def main(args) -> int:
    bad = 0
    for name, sc in scenarios(args):
        t0 = time.time()
        bound = solve(sc, 120, bound_only=True)
        plan = solve(sc, 120)
        cost, calls, undelivered = dispatch(sc, plan["plan"])
        plays = {"greedy": run_policy(Greedy(), sc).final.cost.total, "reference": run_policy(Reference(), sc).final.cost.total}
        for k in range(5):
            plays[f"random{k}"] = run_policy(RandomLegal(k), sc).final.cost.total
        lowest = min(plays.values())
        sound = bound["cost"] <= lowest + 1e-6 and bound["cost"] <= cost + 1e-6
        exact = plan["cost"] is not None and abs(plan["cost"] - cost) < 1e-6
        bad += not (sound and exact)
        print(
            f"{name}: bound {bound['cost']:.2f} ({bound['status']}) plan {plan['cost']} dispatched {cost:.2f} "
            f"calls {calls} undelivered {len(undelivered)} | best play {lowest:.2f} ({min(plays, key=plays.get)}) "
            f"| sound {sound} exact {exact} acyclic {plan['acyclic']} | {time.time() - t0:.0f}s",
            flush=True,
        )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
