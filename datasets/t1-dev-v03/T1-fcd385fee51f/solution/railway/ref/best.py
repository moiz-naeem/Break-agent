"""The top anchor `C_best` and the plan that reaches it (docs/design-changes-v0.3.md §2).

`C_best` is the cheapest cost any of our planners reach on a scenario, with the plan that reaches it: a list of
tool calls that replays through a fresh session to exactly that cost. Planners, in order:

- the scripted reference (sees only what an agent sees);
- a full-knowledge beam search on the true simulator (`offline_search`);
- the exact CP-SAT solver, where it applies (no events) and `ortools` is installed. When its lower bound
  equals the cost of its own plan, `C_best` is proven optimal (`source = "proven"`).

The Harbor oracle replays the stored plan, so it scores exactly 1.0 by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from railway.tools.session import Session

READ_ONLY = {"manual", "network", "timetable", "status", "bulletins", "kpis", "stringline", "notes"}


@dataclass
class Best:
    cost: float
    source: str  # "reference" | "offline" | "solver"
    proven: bool
    plan: list[dict] = field(repr=False)  # [{"tool": ..., "args": {...}}], mutating calls and advances only
    candidates: dict[str, float] = field(default_factory=dict)


def replay_plan(scenario, plan: list[dict]) -> float:
    """Play a plan through a fresh session; the cost it reaches."""
    session = Session(scenario)
    for call in plan:
        if session.closed:
            break
        session.call(call["tool"], call["args"])
    return session.close().cost.total


def plan_from_log(log: list[dict]) -> list[dict]:
    """The calls of an episode log that change anything (read-only calls are dropped)."""
    return [{"tool": e["tool"], "args": e["args"]} for e in log if e["tool"] not in READ_ONLY]


def best_plan(
    scenario, *, beam: int | None = None, per_train: int = 4, solver: bool = True, solver_seconds: float = 120.0,
    reference_run=None,
) -> Best:
    """`beam=None` skips the beam search (on graph tiers it rarely beats the reference and is slow)."""
    from railway.policies.runner import run_policy

    from .offline_search import offline_plan
    from .reference import Reference

    ref = reference_run or run_policy(Reference(), scenario)
    options = {"reference": (ref.final.cost.total, plan_from_log(ref.log), False)}
    if beam:
        c_off, off_plan = offline_plan(scenario, beam=beam, per_train=per_train)
        options["offline"] = (c_off, off_plan, False)
    if solver:
        exact = _solver_plan(scenario, solver_seconds)
        if exact is not None:
            options["solver"] = exact
    source = min(options, key=lambda k: (options[k][0], k != "solver"))
    cost, plan, proven = options[source]
    replayed = replay_plan(scenario, plan)
    if abs(replayed - cost) > 1e-6:
        raise RuntimeError(f"{scenario.scenario_id}: {source} plan replays to {replayed}, not {cost}")
    return Best(cost, source, proven, plan, {k: v[0] for k, v in options.items()})


def _solver_plan(scenario, seconds: float):
    try:
        from .optimal import dispatch, solve
    except ImportError:  # ortools not installed
        return None
    try:
        plan = solve(scenario, seconds)
    except ValueError:  # events or factors the model does not cover
        return None
    if not plan["plan"]:
        return None
    cost, _, undelivered, log = dispatch(scenario, plan["plan"], with_log=True)
    if undelivered:
        return None
    bound = solve(scenario, seconds, bound_only=True)
    proven = bound["status"] == "OPTIMAL" and abs(bound["cost"] - cost) < 1e-6 and plan["acyclic"]
    return cost, plan_from_log(log), proven
