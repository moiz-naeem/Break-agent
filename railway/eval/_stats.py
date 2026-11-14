"""The one place T09 reads T08's `railway/eval/stats.py`.

T08 §2.8 fixes the statistics but not the Python names. This adapter assumes
`stats.summarize(rows) -> dict | object` holding the T08 §2.8 quantities and normalizes it
to the flat dict that `report.py` and `pilot.py` use. If T08 lands different names,
change this file only.

Normalized keys: score_ros, ci (lo, hi), reward_mean, unclipped_mean, cost_ratio,
gridlocks, crew_timeouts, invalid_calls, format_errors, tokens, episodes, errors, flagged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Summary = dict[str, Any]
SummarizeFn = Callable[[list[dict]], Summary]
ScoreFn = Callable[[list[dict]], float]

_MISSING = object()


def _get(raw: Any, *names: str) -> Any:
    for source in (raw, _get_one(raw, "means")):
        if source is None or source is _MISSING:
            continue
        for name in names:
            value = _get_one(source, name)
            if value is not _MISSING:
                return value
    return None


def _get_one(source: Any, name: str) -> Any:
    if isinstance(source, dict):
        return source.get(name, _MISSING)
    return getattr(source, name, _MISSING)


def summarize(rows: list[dict]) -> Summary:
    from railway.eval import stats  # T08; ImportError until it lands

    raw = stats.summarize(rows)
    ci = _get(raw, "score_ros_ci", "ci")
    if isinstance(ci, dict):
        ci = ci.get("score_ros")
    return {
        "score_ros": _get(raw, "score_ros"),
        "ci": tuple(ci) if ci is not None else None,
        "reward_mean": _get(raw, "reward_mean"),
        "unclipped_mean": _get(raw, "unclipped_mean"),
        "cost_ratio": _get(raw, "cost_ratio"),
        "gridlocks": _get(raw, "gridlocks", "mean_gridlocks"),
        "crew_timeouts": _get(raw, "crew_timeouts", "mean_crew_timeouts"),
        "invalid_calls": _get(raw, "invalid_calls", "mean_invalid_calls"),
        "format_errors": _get(raw, "format_errors", "mean_format_errors"),
        "tokens": _get(raw, "tokens", "mean_tokens"),
        "episodes": _get(raw, "episodes", "n_episodes"),
        "errors": _get(raw, "errors", "n_errors"),
        "flagged": list(_get(raw, "flagged") or []),
    }


def score_ros(rows: list[dict]) -> float:
    return summarize(rows)["score_ros"]
