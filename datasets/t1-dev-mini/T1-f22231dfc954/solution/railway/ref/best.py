"""C_best, the top anchor of the reward (docs/design-changes-v0.3.md §2.4), and the plan that reaches it.

Candidates: the scripted reference, the offline beam search and, where it covers the scenario (factors F1-F4,
no events), the exact CP-SAT solver (`railway/ref/optimal.py`). Every candidate is re-played as tool calls
through a fresh `Session`, and only the re-played cost counts: that is exactly what the Harbor oracle gets
when it replays the stored plan through MCP, so the oracle scores 1.0.

`c_best_kind` is "proven" when the solver's lower bound, its plan cost and the plan's played cost are equal
(both solves OPTIMAL) and no other candidate is cheaper; otherwise "best-known".
"""

from __future__ import annotations

from railway.sim.rules import RuleId
from railway.tools.session import Session

PROVEN, BEST_KNOWN = "proven", "best-known"
SOLVER_TIME_LIMIT = 300.0  # optimal.py's default, per solve (bound and plan)
# Equal costs prefer the solver's plan, then the search's, then the reference's.
SOURCES = ("solver", "offline_search", "reference")


def plan_calls(log: list[dict]) -> list[dict]:
    """The tool calls of an episode log, without calls made after the shift had ended."""
    return [{"tool": e["tool"], "args": e["args"]} for e in log if e["rule"] != RuleId.SHIFT_ENDED.value]


def play(scenario, calls: list[dict]) -> tuple[float, list[dict]]:
    """Play tool calls through a fresh session, as the oracle does over MCP: (final cost, calls actually made)."""
    session = Session(scenario)
    for call in calls:
        if session.closed:
            break
        session.call(call["tool"], call["args"])
    final = session.close()
    return final.cost.total, plan_calls(session.log)


def _solver(scenario, time_limit: float) -> tuple[dict, list[dict] | None]:
    """(solver report, its played tool calls or None). Never raises for a scenario the solver can't take."""
    try:
        from .optimal import dispatch, solve
    except ImportError:  # ortools is a dev dependency; the Harbor images never compute C_best
        return {"status": "unavailable"}, None
    try:
        bound = solve(scenario, time_limit, bound_only=True)
    except ValueError:  # events or factors beyond F1-F4
        return {"status": "not_covered"}, None
    plan = solve(scenario, time_limit)
    log: list[dict] = []
    played, _, undelivered = dispatch(scenario, plan["plan"], log=log)
    statuses = (bound["status"], plan["status"])
    report = {
        "status": "optimal" if statuses == ("OPTIMAL", "OPTIMAL") else "timeout",
        "bound": round(bound["cost"], 6),
        "plan_cost": round(plan["cost"], 6),
        "played": played,
        "undelivered": len(undelivered),
    }
    report["proven"] = (
        report["status"] == "optimal" and not undelivered and report["bound"] == report["plan_cost"] == played
    )
    return report, plan_calls(log)


def best_plan(scenario, reference_log: list[dict], *, time_limit: float = SOLVER_TIME_LIMIT) -> dict:
    """C_best for one scenario. `reference_log` is the reference's episode log (the filter already ran it)."""
    from .offline_search import offline_plan

    solver, solver_calls = _solver(scenario, time_limit)
    plans = {"reference": plan_calls(reference_log), "offline_search": offline_plan(scenario)[1]}
    if solver_calls is not None:
        plans["solver"] = solver_calls
    played = {source: play(scenario, calls) for source, calls in plans.items()}
    costs = {source: cost for source, (cost, _) in played.items()}
    source = min(played, key=lambda s: (costs[s], SOURCES.index(s)))
    c_best = costs[source]
    proven = solver.get("proven", False) and source == "solver"
    if solver.get("proven") and c_best < solver["played"]:
        solver["contradiction"] = True  # a plan beat the "proven" optimum: the solver's model is wrong somewhere
    return {
        "c_best": c_best,
        "c_best_kind": PROVEN if proven else BEST_KNOWN,
        "c_best_source": source,
        "candidates": costs,
        "solver": solver,
        "plan": played[source][1],
    }
