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
    ref_gridlocks: int
    greedy_unfinished: int
    exploit_costs: dict[str, float]
    reasons: list[str]


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


def evaluate_scenario(scenario, runner=run_named) -> FilterReport:
    """Run greedy, the reference and every exploit, then apply DESIGN §6.3's four fixed conditions.

    ``runner(name, scenario) -> EpisodeResult`` is injectable so unit tests can stub the runs.
    """
    results = {name: runner(name, scenario) for name in POLICY_NAMES}
    c_greedy = results["greedy"].final.cost.total
    c_ref = results["reference"].final.cost.total
    gridlocks = results["reference"].final.counts["gridlocks"]
    unfinished = results["greedy"].unfinished
    exploit_costs = {name: r.final.cost.total for name, r in results.items() if name not in {"greedy", "reference"}}
    reasons = []
    if c_ref > 0.70 * c_greedy:
        reasons.append("ref_not_30pct_better")
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
        "ref_gridlocks": gridlocks,
        "greedy_unfinished": unfinished,
        "exploit_costs": exploit_costs,
        "reasons": reasons,
    }


def generator_version() -> str:
    """Hash of the generator sources (`railway/gen/*.py`, not the CLI): a changed generator invalidates lists."""
    digest = hashlib.sha256()
    for path in sorted((Path(__file__).resolve().parents[1] / "gen").glob("*.py")):
        if path.name == "cli.py":  # commands, not scenarios
            continue
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
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
    header = {"rules_version": RULES_VERSION, "generator_version": generator_version()}
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
                    accepted.append(entry | {"c_greedy": report["c_greedy"], "c_ref": report["c_ref"]})
                else:
                    rejected.update(report["reasons"])
    payload = {"header": header | {"rejected_by_reason": dict(sorted(rejected.items()))}, "accepted": accepted}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return accepted
