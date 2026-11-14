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
    c_best: float | None  # None when another condition already rejected the scenario (C_best is not computed)
    c_best_kind: str | None
    c_best_source: str | None
    solver_status: str | None
    plan: list[dict] | None  # the tool calls that reach c_best: the Harbor oracle replays them
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


HEADROOM = 0.70  # v0.3 §2.6: C_best must be at least 30% below greedy


def _best_plan(scenario, reference_log):
    from railway.ref.best import best_plan

    return best_plan(scenario, reference_log)


def evaluate_scenario(scenario, runner=run_named, best=_best_plan) -> FilterReport:
    """Run greedy, the reference and every exploit, then apply the v0.3 seed filter (design-changes §2.6):

    1. C_best is at least 30% below greedy (replaces "the reference is at least 30% below greedy");
    2. the reference never gridlocks;
    3. every exploit baseline costs at least as much as greedy;
    4. greedy finishes every train.

    C_best (`railway.ref.best`) is the costly part, so it is computed only when 2-4 already pass.
    ``runner(name, scenario) -> EpisodeResult`` and ``best(scenario, reference_log) -> dict`` are injectable so
    unit tests can stub them.
    """
    results = {name: runner(name, scenario) for name in POLICY_NAMES}
    c_greedy = results["greedy"].final.cost.total
    c_ref = results["reference"].final.cost.total
    gridlocks = results["reference"].final.counts["gridlocks"]
    unfinished = results["greedy"].unfinished
    exploit_costs = {name: r.final.cost.total for name, r in results.items() if name not in {"greedy", "reference"}}
    reasons = []
    if gridlocks:
        reasons.append("ref_gridlock")
    if min(exploit_costs.values()) < c_greedy:
        reasons.append("exploit_beats_greedy")
    if unfinished:
        reasons.append("greedy_unfinished")
    found = {} if reasons else best(scenario, results["reference"].log)
    if found and found["c_best"] > HEADROOM * c_greedy:
        reasons.append("best_not_30pct_better")
    return {
        "accepted": not reasons,
        "c_greedy": c_greedy,
        "c_ref": c_ref,
        "c_best": found.get("c_best"),
        "c_best_kind": found.get("c_best_kind"),
        "c_best_source": found.get("c_best_source"),
        "solver_status": found.get("solver", {}).get("status"),
        "plan": found.get("plan"),
        "ref_gridlocks": gridlocks,
        "greedy_unfinished": unfinished,
        "exploit_costs": exploit_costs,
        "reasons": reasons,
    }


def generator_version() -> str:
    """Hash of the generator sources (`railway/gen/*.py`, not the CLI): a changed generator invalidates lists.

    Line endings are normalised first, so a CRLF checkout (git autocrlf on Windows) hashes like an LF one.
    """
    digest = hashlib.sha256()
    for path in sorted((Path(__file__).resolve().parents[1] / "gen").glob("*.py")):
        if path.name == "cli.py":  # commands, not scenarios
            continue
        digest.update(path.name.encode() + b"\0" + path.read_bytes().replace(b"\r\n", b"\n"))
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


ENTRY_REPORT_KEYS = ("c_greedy", "c_ref", "c_best", "c_best_kind", "c_best_source", "solver_status")


def plan_path(preset: str, set_name: str, scenario_id: str, output_dir="seeds") -> Path:
    """Where `accepted_seeds` stores the c_best plan (tool calls) of an accepted scenario."""
    return Path(output_dir) / "plans" / f"{preset}-{set_name}" / f"{scenario_id}.json"


def load_plan(preset: str, set_name: str, scenario_id: str, output_dir="seeds") -> list[dict]:
    return json.loads(plan_path(preset, set_name, scenario_id, output_dir).read_text(encoding="utf-8"))


def accepted_seeds(
    preset: str, set_name: str, count: int, *, output_dir="seeds", workers: int | None = None, mapper=None
):
    """Scan `candidate_seeds(set_name)` in order until `count` are accepted; write and return the list.

    A file with the same rules and generator versions (and every accepted entry's plan file) is reused.
    Candidates are evaluated in parallel batches but taken strictly in seed order, so the result doesn't depend
    on `workers`. `mapper` replaces the local process pool (same contract as `map(_evaluate_seed, presets,
    seeds)`), e.g. a Modal fan-out. Each accepted entry's plan goes to `plan_path(...)`; the header lists the
    accepted seeds whose exact solve timed out (`solver_timeouts`, labelled best-known).
    """
    from railway.gen.seeds import candidate_seeds

    path = Path(output_dir) / f"{preset}-{set_name}.json"
    header = {"rules_version": RULES_VERSION, "generator_version": generator_version()}
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        entries = cached["accepted"][:count]
        if (
            {k: cached["header"].get(k) for k in header} == header
            and len(cached["accepted"]) >= count
            and all(plan_path(preset, set_name, e["scenario_id"], output_dir).exists() for e in entries)
        ):
            return entries

    workers = workers or os.cpu_count() or 1
    accepted, rejected, seeds, plans = [], Counter(), iter(candidate_seeds(set_name)), {}
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
                    accepted.append(entry | {k: report[k] for k in ENTRY_REPORT_KEYS})
                    plans[entry["scenario_id"]] = report["plan"]
                else:
                    rejected.update(report["reasons"])
    timeouts = sorted(e["seed"] for e in accepted if e["solver_status"] == "timeout")
    extra = {"rejected_by_reason": dict(sorted(rejected.items())), "solver_timeouts": timeouts}
    payload = {"header": header | extra, "accepted": accepted}
    for scenario_id, plan in plans.items():
        target = plan_path(preset, set_name, scenario_id, output_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(plan, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return accepted
