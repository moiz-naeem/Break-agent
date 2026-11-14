"""Aggregate statistics over results rows (DESIGN.md §6.2, PRD §8 D16, T08 §2.8, v0.3 reward).

Every statistic is computed per seed (`scenario_id`) first, averaging over the ok reps, so a seed
weighs the same whatever its rep count. Confidence intervals resample **seeds** with replacement.

In v0.3:
- The episode reward is `-(C1 + C2 + C3 + C4 + C5)` (`-cost_total`); `mean_neg_cost` is the headline
  aggregate on a fixed task set.
- Normalised greedy/best scores (`score_ros`, `score_normalized` / `reward_mean`, `unclipped_mean`)
  are secondary report-only columns for reading results across tasks of different size.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

STATS = ("score_ros", "reward_mean", "unclipped_mean", "cost_ratio")
MEAN_FIELDS = (
    "gridlocks",
    "crew_timeouts",
    "misroutes",
    "wrong_terminals",
    "missed_calls",
    "missed_connections",
    "unfinished_trains",
    "refusals",
    "invalid_calls",
    "format_errors",
    "tool_calls",
    "decision_points",
    "tokens_in",
    "tokens_out",
    "tokens_cached",
    "tokens_thinking",
    "usd",
    "wall_s",
    "llm_s",
)
KPI_FIELDS = ("on_time_pct", "passenger_delay_minutes", "freight_delivered_pct", "red_flags", "clean_shift")
N_BOOT = 10_000
INCOMPLETE_ERROR_RATE = 0.10


def dedupe(rows: Iterable[dict]) -> list[dict]:
    """One row per (scenario_id, rep): an ok row supersedes error rows; otherwise the last row wins."""
    best: dict[tuple, dict] = {}
    for r in rows:
        key = (r["scenario_id"], r["rep"])
        if key not in best or r.get("status") == "ok" or best[key].get("status") != "ok":
            best[key] = r
    return [best[k] for k in sorted(best, key=lambda k: (str(k[0]), k[1]))]


class _Seeds:
    """Per-seed arrays: cg, cr (c_best), mean agent cost, report-only normalised scores (max(0, u) and u)."""

    def __init__(self, rows: Iterable[dict]):
        by_seed: dict[str, list[dict]] = {}
        for r in rows:
            if r.get("status") == "ok":
                by_seed.setdefault(r["scenario_id"], []).append(r)
        self.ids = sorted(by_seed)
        cg, cr, ca, clipped, unclipped = [], [], [], [], []
        for sid in self.ids:
            reps = by_seed[sid]
            g, c = float(reps[0]["c_greedy"]), float(reps[0]["c_best"])
            costs = np.array([float(r["cost_total"]) for r in reps])
            u = (g - costs) / (g - c)
            cg.append(g)
            cr.append(c)
            ca.append(costs.mean())
            clipped.append(np.maximum(u, 0.0).mean())
            unclipped.append(u.mean())
        self.cg, self.cr, self.ca = np.array(cg), np.array(cr), np.array(ca)
        self.clipped, self.unclipped = np.array(clipped), np.array(unclipped)
        self.by_seed = by_seed

    def __len__(self):
        return len(self.ids)


def _stat_on(seeds: _Seeds, name: str, idx: np.ndarray) -> np.ndarray:
    """Statistic `name` for each row of the index matrix `idx` (shape (B, n) of seed indices)."""
    if name == "neg_cost_mean":
        return (-seeds.ca[idx]).mean(axis=1)
    if name == "score_ros":
        return (seeds.cg[idx] - seeds.ca[idx]).sum(axis=1) / (seeds.cg[idx] - seeds.cr[idx]).sum(axis=1)
    if name in ("reward_mean", "normalized_mean"):
        return seeds.clipped[idx].mean(axis=1)
    if name == "unclipped_mean":
        return seeds.unclipped[idx].mean(axis=1)
    if name == "cost_ratio":
        num = seeds.ca[idx].sum(axis=1)
        den = seeds.cr[idx].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, num / den, np.nan)
    raise ValueError(f"unknown statistic {name!r}; expected one of {(*STATS, 'neg_cost_mean')}")


def _point(rows, name: str) -> float:
    seeds = rows if isinstance(rows, _Seeds) else _Seeds(dedupe(rows))
    if not len(seeds):
        return float("nan")
    return float(_stat_on(seeds, name, np.arange(len(seeds))[None, :])[0])


def neg_cost_mean(rows) -> float:
    """mean_s mean_reps (−cost_total): the v0.3 reward headline on a task set."""
    return _point(rows, "neg_cost_mean")


def score_ros(rows) -> float:
    """Report-only normalised ratio-of-sums: Σ_s (cg_s − ca_s) / Σ_s (cg_s − c_best_s)."""
    return _point(rows, "score_ros")


def reward_mean(rows) -> float:
    """Report-only mean normalised score: mean_s mean_reps max(0, (cg − ca)/(cg − c_best)) (kept under legacy name)."""
    return _point(rows, "reward_mean")


def normalized_mean(rows) -> float:
    """Report-only mean normalised score: mean_s mean_reps max(0, (cg − ca)/(cg − c_best))."""
    return _point(rows, "normalized_mean")


def unclipped_mean(rows) -> float:
    return _point(rows, "unclipped_mean")


def cost_ratio(rows) -> float:
    """Σ_s ca_s / Σ_s c_best_s."""
    return _point(rows, "cost_ratio")


def flagged(rows) -> list[str]:
    """Seeds where any ok rep scores above 1.0 normalised (`above_best`: beat the stored best plan)."""
    out = []
    for sid, reps in _Seeds(dedupe(rows)).by_seed.items():
        if any((r["c_greedy"] - r["cost_total"]) / (r["c_greedy"] - r["c_best"]) > 1.0 + 1e-6 for r in reps):
            out.append(sid)
    return sorted(out)


def _percentile_ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_ci(rows, name: str, *, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% CI from `n_boot` resamples of seeds, with `numpy.random.default_rng(seed)`."""
    seeds = rows if isinstance(rows, _Seeds) else _Seeds(dedupe(rows))
    n = len(seeds)
    if n == 0:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    return _percentile_ci(_stat_on(seeds, name, idx))


def paired_diff(a_rows, b_rows, name: str, *, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """stat(b) − stat(a) on the seeds both runs have, with a CI from the same seed resamples for both."""
    a, b = _Seeds(dedupe(a_rows)), _Seeds(dedupe(b_rows))
    common = sorted(set(a.ids) & set(b.ids))
    if not common:
        return {"diff": float("nan"), "ci": (float("nan"), float("nan")), "common_seeds": 0}
    a_idx = np.array([a.ids.index(s) for s in common])
    b_idx = np.array([b.ids.index(s) for s in common])
    point = _stat_on(b, name, b_idx[None, :])[0] - _stat_on(a, name, a_idx[None, :])[0]
    pick = np.random.default_rng(seed).integers(0, len(common), size=(n_boot, len(common)))
    diffs = _stat_on(b, name, b_idx[pick]) - _stat_on(a, name, a_idx[pick])
    return {"diff": float(point), "ci": _percentile_ci(diffs), "common_seeds": len(common)}


def _mean(rows: list[dict], field: str):
    vals = [r[field] for r in rows if r.get(field) is not None]
    return float(np.mean(vals)) if vals else None


def _kpi_means(rows: list[dict]) -> dict[str, float | None]:
    krows = [r["kpis"] for r in rows if isinstance(r.get("kpis"), dict)]
    out: dict[str, float | None] = {}
    for k in KPI_FIELDS:
        vals = [float(kr[k]) for kr in krows if kr.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def summarize(rows: Iterable[dict], *, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """The stats table. Error rows are excluded and counted; > 10% errors marks the run incomplete."""
    rows = dedupe(rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    errors = len(rows) - len(ok)
    planned = len(rows)
    seeds = _Seeds(ok)
    usd_vals = [float(r["usd"]) for r in ok if r.get("usd") is not None]
    wall_vals = [float(r["wall_s"]) for r in ok if r.get("wall_s") is not None]
    out = {
        "planned": planned,
        "episodes": len(ok),
        "errors": errors,
        "incomplete": planned > 0 and errors / planned > INCOMPLETE_ERROR_RATE,
        "seeds": len(seeds),
        "flagged": flagged(ok),
        "neg_cost_mean": _point(seeds, "neg_cost_mean"),
        "neg_cost_ci": list(bootstrap_ci(seeds, "neg_cost_mean", n_boot=n_boot, seed=seed)),
        "score_normalized_mean": _point(seeds, "normalized_mean"),
        "ci": {},
        "means": {f: _mean(ok, f) for f in MEAN_FIELDS},
        "kpis": _kpi_means(ok),
        "totals": {
            "usd": round(sum(usd_vals), 6) if usd_vals else None,
            "wall_s": round(sum(wall_vals), 3) if wall_vals else None,
            "tokens_in": int(sum(r.get("tokens_in") or 0 for r in ok)),
            "tokens_out": int(sum(r.get("tokens_out") or 0 for r in ok)),
        },
    }
    for name in STATS:
        out[name] = _point(seeds, name)
        out["ci"][name] = list(bootstrap_ci(seeds, name, n_boot=n_boot, seed=seed))
    return out


def format_summary(s: dict) -> str:
    lines = [
        f"episodes {s['episodes']}/{s['planned']}  errors {s['errors']}  seeds {s['seeds']}"
        + ("  INCOMPLETE" if s["incomplete"] else "")
    ]
    lines.append(f"{'statistic':<24}{'value':>12}{'95% CI':>28}")
    nlo, nhi = s["neg_cost_ci"]
    lines.append(f"{'reward (-cost) mean':<24}{s['neg_cost_mean']:>12.2f}{f'[{nlo:.2f}, {nhi:.2f}]':>28}")
    for name in STATS:
        lo, hi = s["ci"][name]
        label = "norm_score_mean" if name == "reward_mean" else name
        lines.append(f"{label:<24}{s[name]:>12.4f}{f'[{lo:.4f}, {hi:.4f}]':>28}")
    for field, value in s["means"].items():
        if value is not None:
            lines.append(f"{'mean ' + field:<24}{value:>12.2f}")
    kpis = s.get("kpis") or {}
    for field, value in kpis.items():
        if value is not None:
            lines.append(f"{'kpi ' + field:<24}{value:>12.2f}")
    lines.append(f"{'flagged (>1.0)':<24}{', '.join(s['flagged']) or 'none'}")
    return "\n".join(lines)
