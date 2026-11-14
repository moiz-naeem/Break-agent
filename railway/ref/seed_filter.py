"""Scenario acceptance checks and deterministic accepted-seed list caching."""

from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import itertools
import json
import os
from collections import Counter
from functools import partial
from pathlib import Path
from typing import TypedDict

from railway.sim.constants import RULES_VERSION
from railway.util import content_hash


class FilterReport(TypedDict):
    accepted: bool
    c_greedy: float
    c_ref: float
    c_best: float
    best_source: str
    best_proven: bool
    ref_gridlocks: int
    greedy_unfinished: int
    exploit_costs: dict[str, float]
    reasons: list[str]


FILTER_VERSION = 3  # bump when the acceptance rules or the entry fields change: cached seed lists are recomputed

POLICY_NAMES = (
    "greedy", "reference", "DoNothing", "StopEarly", "SpamAdvance", "HoldFreight", "MainLineHold", "Malformed",
)  # fmt: skip


def run_named(name: str, scenario):
    """One in-process episode of a filter policy: exactly the code path the oracle uses."""
    from railway.policies.baselines import Greedy
    from railway.policies.exploits import EXPLOITS
    from railway.policies.runner import run_policy
    from railway.ref.reference import Reference

    policy = {"greedy": Greedy, "reference": Reference, **EXPLOITS}[name]()
    return run_policy(policy, scenario, transport="inproc")


def best_for(scenario, reference_run):
    """(c_best, source, proven): the reference, or the exact solver where it applies and does better."""
    from railway.ref.best import best_plan

    try:
        best = best_plan(scenario, reference_run=reference_run)
    except RuntimeError:  # a plan that does not replay: fall back to the reference, which always does
        return reference_run.final.cost.total, "reference", False
    return best.cost, best.source, best.proven


def evaluate_scenario(scenario, runner=run_named, best=None) -> FilterReport:
    """Run greedy, the reference and every exploit, then apply the v0.3 conditions (DESIGN §6.3 with the
    headroom rule on C_best, docs/design-changes-v0.3.md §2.3).

    ``runner(name, scenario) -> EpisodeResult`` and ``best(scenario, reference_run)`` are injectable so unit
    tests can stub the runs.
    """
    results = {name: runner(name, scenario) for name in POLICY_NAMES}
    c_greedy = results["greedy"].final.cost.total
    c_ref = results["reference"].final.cost.total
    c_best, best_source, best_proven = (best or best_for)(scenario, results["reference"])
    gridlocks = results["reference"].final.counts["gridlocks"]
    unfinished = results["greedy"].unfinished
    exploit_costs = {name: r.final.cost.total for name, r in results.items() if name not in {"greedy", "reference"}}
    reasons = []
    if c_best > 0.70 * c_greedy:
        reasons.append("best_not_30pct_better")
    if gridlocks:
        reasons.append("ref_gridlock")
    if min(exploit_costs.values()) < c_greedy:
        reasons.append("exploit_beats_greedy")
    if unfinished:
        reasons.append("greedy_unfinished")
    return {
        "accepted": not reasons,
        "c_greedy": c_greedy,
        "c_ref": c_ref,
        "c_best": c_best,
        "best_source": best_source,
        "best_proven": best_proven,
        "ref_gridlocks": gridlocks,
        "greedy_unfinished": unfinished,
        "exploit_costs": exploit_costs,
        "reasons": reasons,
    }


def cost_version() -> str:
    """Hash of the cost rules (constants and formulas): changing any weight invalidates cached anchors."""
    digest = hashlib.sha256()
    sim = Path(__file__).resolve().parents[1] / "sim"
    for name in ("constants.py", "cost.py"):
        text = (sim / name).read_bytes().replace(b"\r\n", b"\n")
        digest.update(name.encode() + b"\0" + text)
    return digest.hexdigest()


def generator_version() -> str:
    """Hash of the generator sources (`railway/gen/*.py`, not the CLI): a changed generator invalidates lists."""
    digest = hashlib.sha256()
    for path in sorted((Path(__file__).resolve().parents[1] / "gen").glob("*.py")):
        if path.name == "cli.py":  # commands, not scenarios
            continue
        text = path.read_bytes().replace(b"\r\n", b"\n")  # the same hash on CRLF and LF checkouts
        digest.update(path.name.encode() + b"\0" + text)
    return digest.hexdigest()


def _evaluate_seed(preset: str, seed: int) -> tuple[dict, FilterReport]:
    from railway.gen.generator import generate
    from railway.gen.presets import PRESETS

    scenario = generate(PRESETS[preset], seed)
    entry = {
        "seed": seed,
        "scenario_id": scenario.scenario_id,
        "scenario_hash": content_hash(scenario.model_dump(mode="json")),
    }
    return entry, evaluate_scenario(scenario)


def accepted_seeds(
    preset: str, set_name: str, count: int, *, output_dir="seeds", workers: int | None = None, mapper=None
):
    """Scan `candidate_seeds(set_name)` in order until `count` are accepted; write and return the list.

    A file with the same rules and generator versions is reused. Candidates are evaluated in parallel
    batches but taken strictly in seed order, so the result doesn't depend on `workers`. `mapper` replaces the
    local process pool (same contract as `map(_evaluate_seed, presets, seeds)`), e.g. a Modal fan-out.
    """
    from railway.gen.seeds import candidate_seeds

    path = Path(output_dir) / f"{preset}-{set_name}.json"
    header = {
        "rules_version": RULES_VERSION,
        "generator_version": generator_version(),
        "filter_version": FILTER_VERSION,
        "cost_version": cost_version(),
    }
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if {k: cached["header"].get(k) for k in header} == header and len(cached["accepted"]) >= count:
            return cached["accepted"][:count]

    workers = workers or os.cpu_count() or 1
    accepted, rejected, seeds = [], Counter(), iter(candidate_seeds(set_name))
    local = workers > 1 and mapper is None
    with concurrent.futures.ProcessPoolExecutor(workers) if local else contextlib.nullcontext() as pool:
        evaluate = mapper or partial(pool.map if pool else map, _evaluate_seed)
        while len(accepted) < count:
            batch = list(itertools.islice(seeds, workers))
            if not batch:
                raise RuntimeError(f"{preset}-{set_name}: candidate seeds ran out at {len(accepted)}/{count} accepted")
            results = evaluate([preset] * len(batch), batch)
            for entry, report in list(results):  # drain the batch: a half-read remote map errors on exit
                if len(accepted) == count:
                    break
                if report["accepted"]:
                    accepted.append(
                        entry
                        | {k: report[k] for k in ("c_greedy", "c_ref", "c_best", "best_source", "best_proven")}
                    )
                else:
                    rejected.update(report["reasons"])
    payload = {"header": header | {"rejected_by_reason": dict(sorted(rejected.items()))}, "accepted": accepted}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return accepted
