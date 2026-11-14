"""Diagnostics (DESIGN.md §9.7, T09 §2.6) that work on results rows and the simulator.

Built here: failure breakdown (item 1), factor-ablation gaps (item 2, from rows), score vs
episode length (item 4, from rows), the regret formula and deterministic decision-state
sampling (item 3 building blocks).

Not built yet: the regret runner (needs T06 `Reference`/`candidates` and the T08 prompts),
`conflicts()`/`what_if` (T05 Session), the ablation presets (T04 presets).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from railway.eval._stats import ScoreFn
from railway.sim.cost import CostBreakdown
from railway.sim.engine import Simulator
from railway.sim.observe import State, StaticInfo

COMPONENTS = tuple(name for name in CostBreakdown.model_fields if name != "total")
FAILURE_COUNTS = ("gridlocks", "crew_timeouts", "invalid_calls")
ABLATION_FACTORS = ("F5", "F6", "F7", "F8", "F10", "F15", "F16")
LENGTH_PRESETS = ("T1", "T2", "T3")
_ABLATION = re.compile(r"^T2-only-(F\d+)$")


def _ok(rows: Iterable[dict]) -> list[dict]:
    return [r for r in rows if r["status"] == "ok"]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _default_score(rows: list[dict]) -> float:
    from railway.eval._stats import score_ros

    return score_ros(rows)


def group_key(row: dict) -> str:
    return f"{row['model']}|{row['harness']}|{row.get('mode') or ''}"


def group_rows(rows: Iterable[dict]) -> dict[str, list[dict]]:
    """Rows by `model|harness|mode`, groups in first-appearance order."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(group_key(r), []).append(r)
    return groups


def failure_breakdown(rows: list[dict]) -> dict | None:
    """Mean cost by component, mean total and mean failure counts per ok episode. None if no ok row."""
    ok = _ok(rows)
    if not ok:
        return None
    out: dict = {"episodes": len(ok)}
    for comp in COMPONENTS:
        out[comp] = _mean([float(r["cost_by_component"].get(comp, 0.0)) for r in ok])
    out["total"] = _mean([float(r["cost_total"]) for r in ok])
    for key in FAILURE_COUNTS:
        out[key] = _mean([float(r[key]) for r in ok])
    return out


def _factor_order(f: str) -> int:
    return int(f[1:])


def ablation_gaps(
    rows: list[dict], *, small: str = "small", big: str = "big", score_fn: ScoreFn | None = None
) -> dict[str, dict]:
    """Per factor (from presets `T2-only-<F>`): small and big score_ros and big − small (None if missing)."""
    score = score_fn or _default_score
    by_factor: dict[str, dict[str, list[dict]]] = {}
    for r in _ok(rows):
        m = _ABLATION.match(r["preset"])
        if m:
            by_factor.setdefault(m.group(1), {}).setdefault(r["model"], []).append(r)
    out = {}
    for factor in sorted(by_factor, key=_factor_order):
        models = by_factor[factor]
        s = score(models[small]) if small in models else None
        b = score(models[big]) if big in models else None
        out[factor] = {"small": s, "big": b, "gap": b - s if s is not None and b is not None else None}
    return out


def score_by_preset(rows: list[dict], *, score_fn: ScoreFn | None = None) -> dict[tuple[str, str], float]:
    """Score vs episode length: score_ros per (model, preset) for T1, T2, T3."""
    score = score_fn or _default_score
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in _ok(rows):
        if r["preset"] in LENGTH_PRESETS:
            groups.setdefault((r["model"], r["preset"]), []).append(r)
    order = sorted(groups, key=lambda k: (k[0], LENGTH_PRESETS.index(k[1])))
    return {k: score(groups[k]) for k in order}


def regret(action_value: float, candidate_values: Iterable[float]) -> float:
    """`v(action) − min(v(c) for c in candidates ∪ {action})` (lower values are better). Always ≥ 0."""
    values = [float(action_value), *map(float, candidate_values)]
    if any(math.isnan(v) for v in values):
        raise ValueError("NaN value in regret")
    return values[0] - min(values)


def mean_regret(records: Iterable[dict]) -> dict[str, float]:
    """Mean regret per model from `{model, regret}` records, models sorted."""
    by_model: dict[str, list[float]] = {}
    for rec in records:
        by_model.setdefault(rec["model"], []).append(float(rec["regret"]))
    return {m: _mean(v) for m, v in sorted(by_model.items())}


@dataclass(frozen=True)
class SampledState:
    index: int  # decision-point index in the trajectory; 0 = load()
    tick: int
    static: StaticInfo
    state: State


def sample_decision_states(
    sim: Simulator, step: Callable[[Simulator], None], *, every_k: int, limit: int
) -> list[SampledState]:
    """Every k-th decision point of the trajectory `step` drives, up to `limit` states.

    `step(sim)` takes the trajectory policy's actions at the current decision point (for the
    regret diagnostic: the reference's calls). The shift end is never sampled: there is no
    decision left to ask about. Deterministic given a deterministic `step`.
    """
    if every_k < 1 or limit < 0:
        raise ValueError("every_k must be ≥ 1 and limit ≥ 0")
    static = sim.static_info()
    out: list[SampledState] = []
    index = 0
    while not sim.done and len(out) < limit:
        if index % every_k == 0:
            out.append(SampledState(index, sim.tick, static, sim.state().model_copy(deep=True)))
        step(sim)
        sim.advance()
        index += 1
    return out
