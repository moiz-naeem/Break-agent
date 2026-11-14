"""SYNTHETIC results rows for T09 tests. No number here is a real model result.

`ros_test_only` and `summary_test_only` are tiny test-local stand-ins for T08's
`railway/eval/stats.py` (PRD §8 D16, T08 §2.8), used only so T09 logic can be tested
before T08 lands. Remove them at integration once every caller can use `stats.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

COMPONENTS = ("pax_lateness", "freight_lateness", "crew_timeout", "reversal", "unfinished")


def make_row(
    *,
    run: str = "synthetic",
    model: str = "small",
    harness: str = "harbor",
    mode: str | None = None,
    seed: int = 1,
    rep: int = 0,
    c_greedy: float = 1000.0,
    c_ref: float = 600.0,
    cost_total: float | None = None,
    score: float | None = None,
    preset: str = "T1",
    status: str = "ok",
    **extra,
) -> dict:
    """One PRD §7.10 row. Give either `cost_total` or the unclipped `score` it should produce."""
    if cost_total is None:
        cost_total = c_greedy - (score if score is not None else 0.0) * (c_greedy - c_ref)
    unclipped = (c_greedy - cost_total) / (c_greedy - c_ref)
    # Put the whole cost in one component so the breakdown sums to the total.
    by_comp = dict.fromkeys(COMPONENTS, 0.0)
    by_comp["pax_lateness"] = round(cost_total * 0.75, 6)
    by_comp["unfinished"] = round(cost_total - by_comp["pax_lateness"], 6)
    row = {
        "run": run,
        "harness": harness,
        "mode": mode,
        "model": model,
        "rep": rep,
        "scenario_id": f"{preset}-synth{seed:07d}",
        "scenario_hash": f"{seed:064x}",
        "seed": seed,
        "preset": preset,
        "rules_version": "0.1-provisional",
        "cost_total": cost_total,
        "cost_by_component": by_comp,
        "c_greedy": c_greedy,
        "c_ref": c_ref,
        "score_clipped": min(1.0, max(0.0, unclipped)),
        "score_unclipped": unclipped,
        "cost_ratio": cost_total / c_ref,
        "gridlocks": 0,
        "crew_timeouts": 0,
        "reversals": 0,
        "unfinished": 0,
        "refusals": 0,
        "invalid_calls": 0,
        "tool_calls": 100,
        "decision_points": 30,
        "tokens_in": None,
        "tokens_out": None,
        "wall_s": 10.0,
        "status": status,
        "error": None if status == "ok" else "synthetic infra error",
    }
    if harness == "loop":
        row["format_errors"] = 0
    row.update(extra)
    return row


def model_rows(model: str, score: float, *, seeds=range(1, 11), ratio: float = 0.6, **kw) -> list[dict]:
    """Rows for one model on `seeds`, each scoring `score`, with Σc_ref / Σc_greedy == ratio."""
    return [make_row(model=model, seed=s, c_greedy=1000.0, c_ref=1000.0 * ratio, score=score, **kw) for s in seeds]


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    return path


def _per_seed(rows):
    seeds: dict = {}
    for r in rows:
        if r["status"] == "ok":
            seeds.setdefault(r["scenario_id"], []).append(r)
    return seeds


def ros_test_only(rows: list[dict]) -> float:
    """D16 ratio-of-sums, test-local stand-in for stats.py."""
    num = den = 0.0
    for reps in _per_seed(rows).values():
        ca = sum(r["cost_total"] for r in reps) / len(reps)
        num += reps[0]["c_greedy"] - ca
        den += reps[0]["c_greedy"] - reps[0]["c_ref"]
    return num / den


def summary_test_only(rows: list[dict]) -> dict:
    """Test-local stand-in for the stats.py summary, in the normalized shape report.py expects."""
    seeds = _per_seed(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    score = ros_test_only(rows)

    def per_seed_mean(fn):
        return sum(sum(fn(r) for r in reps) / len(reps) for reps in seeds.values()) / len(seeds)

    def mean(key):
        vals = [r.get(key) for r in ok if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    tokens = [r["tokens_in"] + r["tokens_out"] for r in ok if r.get("tokens_in") is not None]
    return {
        "score_ros": score,
        "ci": (score - 0.05, score + 0.05),
        "reward_mean": per_seed_mean(lambda r: r["score_clipped"]),
        "unclipped_mean": per_seed_mean(lambda r: r["score_unclipped"]),
        "cost_ratio": sum(sum(r["cost_total"] for r in v) / len(v) for v in seeds.values())
        / sum(v[0]["c_ref"] for v in seeds.values()),
        "gridlocks": mean("gridlocks"),
        "crew_timeouts": mean("crew_timeouts"),
        "invalid_calls": mean("invalid_calls"),
        "format_errors": mean("format_errors"),
        "tokens": sum(tokens) / len(tokens) if tokens else None,
        "episodes": len(rows),
        "errors": len(rows) - len(ok),
        "flagged": sorted({r["scenario_id"] for r in ok if r["score_unclipped"] > 1.0}),
    }
