"""Pilot gate (DESIGN.md §9.8, PRD §3 G4, T09 §2.1).

Works on PRD §7.10 results rows. The headline score is T08's `score_ros` (PRD §8 D16);
pass `score_fn` only to test the gate logic before `stats.py` exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from railway.eval._stats import ScoreFn
from railway.util import canonical_json

GAP_MIN = 0.3  # (a) big − small ≥ 0.3
BIG_MAX = 0.8  # (b) big ≤ 0.8
REF_RATIO_MAX = 0.7  # (c) Σc_ref ≤ 0.7 × Σc_greedy
USABILITY_MAX = 0.10  # "if a fails": invalid + format errors > 10% of tool calls → tool-usability artifact
MAX_ITERATIONS = 3
_EPS = 1e-9  # the thresholds are inclusive; absorb float noise at the boundary


@dataclass(frozen=True)
class GateResult:
    small: float
    big: float
    gap: float
    sum_c_ref: float
    sum_c_greedy: float
    ref_ratio: float
    a: bool
    b: bool
    c: bool
    seeds: tuple[str, ...]
    errors: dict[str, int]

    @property
    def failed(self) -> tuple[str, ...]:
        return tuple(name for name, ok in (("a", self.a), ("b", self.b), ("c", self.c)) if not ok)

    @property
    def passed(self) -> bool:
        return not self.failed


def _ok(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["status"] == "ok"]


def _norms(rows: list[dict]) -> dict[str, tuple[float, float]]:
    """scenario_id → (c_greedy, c_ref), checking every row of a seed agrees."""
    norms: dict[str, tuple[float, float]] = {}
    for r in rows:
        pair = (float(r["c_greedy"]), float(r["c_ref"]))
        if norms.setdefault(r["scenario_id"], pair) != pair:
            raise ValueError(f"inconsistent c_greedy/c_ref for {r['scenario_id']}: {norms[r['scenario_id']]} vs {pair}")
    return norms


def _default_score(rows: list[dict]) -> float:
    from railway.eval._stats import score_ros

    return score_ros(rows)


def evaluate_gate(small_rows: list[dict], big_rows: list[dict], *, score_fn: ScoreFn | None = None) -> GateResult:
    """The three pilot conditions on the small and big models' rows (same seeds)."""
    score = score_fn or _default_score
    small_ok, big_ok = _ok(small_rows), _ok(big_rows)
    if not small_ok or not big_ok:
        raise ValueError("the pilot needs ok rows for both models")
    if {r["scenario_id"] for r in small_ok} != {r["scenario_id"] for r in big_ok}:
        raise ValueError("small and big must be scored on the same seeds")
    norms = _norms(small_ok + big_ok)
    sum_ref = sum(cr for _, cr in norms.values())
    sum_greedy = sum(cg for cg, _ in norms.values())
    small, big = score(small_ok), score(big_ok)
    gap = big - small
    return GateResult(
        small=small,
        big=big,
        gap=gap,
        sum_c_ref=sum_ref,
        sum_c_greedy=sum_greedy,
        ref_ratio=sum_ref / sum_greedy,
        a=gap >= GAP_MIN - _EPS,
        b=big <= BIG_MAX + _EPS,
        c=sum_ref <= REF_RATIO_MAX * sum_greedy + _EPS,
        seeds=tuple(sorted(norms)),
        errors={"small": len(small_rows) - len(small_ok), "big": len(big_rows) - len(big_ok)},
    )


def usability_rate(rows: list[dict]) -> float:
    """(invalid_calls + format_errors) / tool_calls over ok rows. Harbor rows have no format_errors."""
    ok = _ok(rows)
    calls = sum(r["tool_calls"] for r in ok)
    bad = sum(r["invalid_calls"] + (r.get("format_errors") or 0) for r in ok)
    return bad / calls if calls else 0.0


def next_actions(gate: GateResult, small_rows: list[dict]) -> list[str]:
    """T09 §2.1 "If it fails": what to do before the next iteration, in order, without repeats.

    raise_difficulty      enable more factors (T1+ = T1 + F5, F7, F8, then F6, F10) or raise event_density
    fix_tool_usability    fix the manual or tool descriptions, not the difficulty
    investigate_reference the reference is not ≥ 30% cheaper than greedy
    """
    actions: list[str] = []
    if not gate.a:
        actions.append("fix_tool_usability" if usability_rate(small_rows) > USABILITY_MAX else "raise_difficulty")
    if not gate.b:
        actions.append("raise_difficulty")
    if not gate.c:
        actions.append("investigate_reference")
    return list(dict.fromkeys(actions))


@dataclass(frozen=True)
class Iteration:
    change: str
    gate: GateResult


@dataclass
class PilotLog:
    """Every pilot iteration and what changed. At most MAX_ITERATIONS; then escalate to the user."""

    iterations: list[Iteration] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.iterations:
            return "not_run"
        if self.iterations[-1].gate.passed:
            return "pass"
        return "escalate" if len(self.iterations) >= MAX_ITERATIONS else "iterate"

    @property
    def verdict(self) -> str:
        return "PASS" if self.status == "pass" else "FAIL"

    def record(self, gate: GateResult, change: str) -> None:
        if self.status in ("pass", "escalate"):
            raise RuntimeError(
                f"pilot is over ({self.status}); "
                + ("escalate to the user with the numbers" if self.status == "escalate" else "it passed")
            )
        if "|" in change or "\n" in change:
            raise ValueError("change must be one line without '|'")
        self.iterations.append(Iteration(change, gate))


def _per_seed_unclipped(rows: list[dict]) -> dict[str, float]:
    by_seed: dict[str, list[float]] = {}
    for r in _ok(rows):
        by_seed.setdefault(r["scenario_id"], []).append(r["score_unclipped"])
    return {s: sum(v) / len(v) for s, v in by_seed.items()}


def _mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def render_pilot_md(
    *,
    models: dict[str, str],
    date: str,
    small_rows: list[dict],
    big_rows: list[dict],
    log: PilotLog,
    score_fn: ScoreFn | None = None,
) -> str:
    """`docs/pilot.md` text. `models` maps role → model name (names only, never keys).

    `small_rows`/`big_rows` are the final iteration's rows; they must reproduce its gate.
    """
    if not log.iterations:
        raise ValueError("no pilot iteration recorded")
    final = log.iterations[-1].gate
    gate = evaluate_gate(small_rows, big_rows, score_fn=score_fn)
    if (gate.small, gate.big, gate.ref_ratio) != (final.small, final.big, final.ref_ratio):
        raise ValueError("the rows given do not reproduce the final iteration's gate")
    norms = _norms(_ok(small_rows) + _ok(big_rows))
    small_seed, big_seed = _per_seed_unclipped(small_rows), _per_seed_unclipped(big_rows)
    lines = [
        "# Pilot gate (DESIGN.md §9.8)",
        "",
        f"- Date: {date}",
        "- Models: " + "; ".join(f"{role} = {name}" for role, name in models.items()),
        f"- Seeds: {len(gate.seeds)}",
        f"- Error rows excluded: small {gate.errors['small']}, big {gate.errors['big']}",
        "",
        "## Per-seed results (final iteration; small and big = unclipped score, mean over reps)",
        "",
        "| Seed | c_greedy | c_ref | small | big |",
        "|---|---|---|---|---|",
    ]
    for sid in gate.seeds:
        cg, cr = norms[sid]
        lines.append(f"| {sid} | {cg:.1f} | {cr:.1f} | {small_seed[sid]:.3f} | {big_seed[sid]:.3f} |")
    lines += [
        "",
        "## Conditions (final iteration; scores are score_ros, PRD §8 D16)",
        "",
        f"- (a) big − small = {gate.gap:.3f} (≥ {GAP_MIN}): {_mark(gate.a)}",
        f"- (b) big = {gate.big:.3f} (≤ {BIG_MAX}): {_mark(gate.b)}",
        f"- (c) Σc_ref / Σc_greedy = {gate.ref_ratio:.3f} (≤ {REF_RATIO_MAX}): {_mark(gate.c)}",
        "",
        "## Iterations",
        "",
        "| # | Change | small | big | gap | ref ratio | failed |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, it in enumerate(log.iterations, 1):
        g = it.gate
        failed = ",".join(g.failed) or "none"
        nums = " | ".join(f"{v:.3f}" for v in (g.small, g.big, g.gap, g.ref_ratio))
        lines.append(f"| {i} | {it.change} | {nums} | {failed} |")
    if log.status == "escalate":
        lines += ["", f"Stopped after {MAX_ITERATIONS} failed iterations: escalated to the user."]
    lines += ["", f"Verdict: {log.verdict}", ""]
    return "\n".join(lines)


_NUM = r"(-?\d+(?:\.\d+)?)"


def parse_pilot_md(text: str) -> dict:
    """The three numbers, the iterations' changes and the verdict from a `docs/pilot.md`."""

    def find(pattern: str) -> str:
        m = re.search(pattern, text)
        if not m:
            raise ValueError(f"pilot record lacks {pattern!r}")
        return m.group(1)

    iterations = re.findall(r"^\| (\d+) \| (.+?) \| .+ \|$", text, flags=re.M)
    return {
        "gap": float(find(r"\(a\) big − small = " + _NUM)),
        "big": float(find(r"\(b\) big = " + _NUM)),
        "ref_ratio": float(find(r"\(c\) Σc_ref / Σc_greedy = " + _NUM)),
        "iterations": [change for _, change in iterations],
        "verdict": find(r"(?m)^Verdict: (PASS|FAIL)\s*$"),
    }


def constants_dump() -> str:
    """Canonical JSON of every public constant in `railway.sim.constants` (the freeze file, T09 §2.2).

    Write it to `docs/frozen-constants.json` only after RULES_VERSION is set to "1.0".
    """
    from railway.sim import constants

    data = {name: getattr(constants, name) for name in vars(constants) if name.isupper() and not name.startswith("_")}
    return canonical_json(data) + "\n"
