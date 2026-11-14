"""Strength check (T06 §2.7): a full-knowledge beam search, to see how far the reference is from strong play."""

from __future__ import annotations

import json
from pathlib import Path

from railway.sim.engine import Simulator

from .candidates import WAIT, candidates
from .reference import Reference
from .rollout import terminal_estimate

ROOT = Path(__file__).resolve().parents[2]


def _score(sim) -> float:
    return round(sim.cost().total + terminal_estimate(sim), 6)


def _actionable(sim) -> list[str]:
    """Trains to decide at this decision point, in the reference's priority order (T06 §2.2 steps 3-4)."""
    state = sim.state()
    trains = []
    for train in state.trains:
        held = train.hold_until is not None and state.tick < train.hold_until
        if train.phase in {"finished", "not_entered"} or held:
            continue
        stops = {stop.station for stop in sim._train_defs[train.id].stops}
        extension = (
            "F5" in sim._scenario.factors
            and train.type == "freight"
            and train.authority is not None
            and train.authority.dest_station not in stops
        )
        if (train.phase == "at_station" and train.authority is None) or extension:
            trains.append(train)
    return [t.id for t in sorted(trains, key=Reference()._priority)]


ADVANCE = {"tool": "advance_to_next_event", "args": {}}


def _call(action) -> dict:
    """A simulator action as the tool call that applies it (action kinds are tool names)."""
    return {"tool": action.kind, "args": action.model_dump(mode="json", exclude={"kind"})}


def offline_plan(scenario, beam: int = 16, per_train: int = 4) -> tuple[float, list[dict]]:
    """Beam search over decision points on the *true* simulator (unrevealed events included).

    A partial plan branches per actionable train over its top `per_train` candidates plus WAIT and is
    scored by cost + terminal estimate. Every kept plan runs to the shift end. Returns the best kept plan's
    cost and its tool calls (`authorize`/`hold`/... at each decision point, then `advance_to_next_event`).
    """
    # (sim, trains to decide at this decision point, next index, tool calls so far)
    frontier = [(Simulator.load(scenario), None, 0, ())]
    while not all(item[0].done for item in frontier):
        expanded = []
        for sim, trains, i, calls in frontier:
            if sim.done:
                expanded.append((sim, trains, i, calls))
                continue
            trains = _actionable(sim) if trains is None else trains
            if i == len(trains):
                after = sim.clone()
                after.advance()
                expanded.append((after, None, 0, (*calls, ADVANCE)))
                continue
            for choice in candidates(sim, trains[i], max_candidates=per_train, deadlock_safe=False):
                branch = sim.clone()
                step = calls
                if choice is not WAIT:
                    branch.apply(choice)
                    step = (*calls, _call(choice))
                expanded.append((branch, trains, i + 1, step))
        frontier = sorted(expanded, key=lambda item: _score(item[0]))[:beam]
    sim, _, _, calls = min(frontier, key=lambda item: item[0].cost().total)
    return sim.cost().total, list(calls)


def offline_search(scenario, beam: int = 16, per_train: int = 4) -> tuple[float, list[dict]]:
    """`offline_plan`'s cost, seeded with the reference's own result so it is never worse than the reference."""
    from railway.policies.runner import run_policy

    c_ref = run_policy(Reference(), scenario).final.cost.total
    best, _ = offline_plan(scenario, beam, per_train)
    return min(best, c_ref), [{"beam": beam, "per_train": per_train, "c_ref": c_ref, "best_beam": best}]


def strength_rows(scenarios) -> list[dict]:
    from railway.policies.baselines import Greedy
    from railway.policies.runner import run_policy

    rows = []
    for scenario in scenarios:
        c_ref = run_policy(Reference(), scenario).final.cost.total
        c_offline, _ = offline_search(scenario)
        rows.append(
            {
                "seed": scenario.seed,
                "c_greedy": run_policy(Greedy(), scenario).final.cost.total,
                "c_ref": c_ref,
                "c_offline": c_offline,
                "gap_pct": (c_ref - c_offline) / c_ref * 100 if c_ref else 0.0,
            }
        )
    return rows


def report(rows: list[dict]) -> str:
    lines = ["| seed | c_greedy | c_ref | c_offline | gap_pct |", "|---|---|---|---|---|"]
    lines += [
        f"| {r['seed']} | {r['c_greedy']:.1f} | {r['c_ref']:.1f} | {r['c_offline']:.1f} | {r['gap_pct']:.1f} |"
        for r in rows
    ]
    gaps = [r["gap_pct"] for r in rows]
    worst = max(rows, key=lambda r: r["gap_pct"])
    flag = (
        f"The largest gap ({worst['gap_pct']:.1f}% on seed {worst['seed']}) is above 25%, a red flag: "
        "the reference may be too weak to anchor scores, so this goes to the team in the gate record."
        if worst["gap_pct"] > 25
        else "No gap is above the 25% red-flag line, so the reference is strong enough to anchor scores."
    )
    return "\n".join(
        [
            "# Strength check (T06 §2.7)",
            "",
            "First 5 accepted T1 dev seeds (`seeds/T1-dev.json`). `c_offline` is a beam search (beam 16, 4",
            "candidates per train) on the true simulator with full knowledge, seeded with the reference's own",
            "result. `gap_pct = (c_ref - c_offline) / c_ref * 100`. Regenerate with",
            "`uv run python -m railway.ref.offline_search`.",
            "",
            *lines,
            "",
            "## Interpretation",
            "",
            f"The full-knowledge search improves on the reference by {sum(gaps) / len(gaps):.1f}% on average, "
            f"ranging from {min(gaps):.1f}% to {max(gaps):.1f}%. "
            "Part of any gap is information the reference never has (T1 has no events, so here it is mostly search "
            f"depth). {flag}",
            "",
        ]
    )


def main() -> int:
    from railway.gen.generator import generate
    from railway.gen.presets import PRESETS

    accepted = json.loads((ROOT / "seeds" / "T1-dev.json").read_text(encoding="utf-8"))["accepted"]
    rows = strength_rows([generate(PRESETS["T1"], e["seed"]) for e in accepted[:5]])
    (ROOT / "docs" / "strength-check.md").write_text(report(rows), encoding="utf-8")
    print(report(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
