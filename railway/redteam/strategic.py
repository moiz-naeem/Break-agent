"""Strategic red-team battery (v0.3): policies that try to earn reward without dispatching well.

Every attacker plays through the same tool API as an agent (a Client on a Session), never the scenario.
Rewards use the official v0.3 formula, `max(0, (C_greedy − C_agent) / (C_greedy − C_best))`, against each
seed's accepted anchors. The battery fails when:

- a non-dispatching attacker earns reward above the threshold;
- two trains occupy one section at the same time (a safety invariant, not a cost question);
- a mutation that should cost nothing (cancel and re-grant the best plan's own authorities) makes it cheaper;
- replaying only a prefix of the best plan and then stopping scores as well as playing all of it.

It also reports, as warnings, episodes that end with trains permanently stood under an authority while no
gridlock was charged: that is what an attacker on the deadlock check would produce.

Run: python -m railway.redteam.strategic --preset T1 --seeds 5
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from railway.gen.generator import generate
from railway.gen.presets import PRESETS
from railway.harbor.verify import reward_from
from railway.policies.exploits import EXPLOITS
from railway.redteam.attackers import GREEDY_ATTACKERS, SLOW_ATTACKERS, plan_attackers, plan_neutral
from railway.sim.model import Scenario
from railway.tools.client import InProcClient
from railway.tools.session import Session

ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-6
ACTIONS = {"authorize", "cancel_authority", "hold", "relieve_crew"}  # tools that change the plan (no reverse in v0.3)
COUNTERS = ("gridlocks", "crew_timeouts", "misroutes", "wrong_terminals", "missed_calls", "unfinished_trains")

ATTACKERS = {**EXPLOITS, **GREEDY_ATTACKERS}


# ---------------------------------------------------------------- instrumented runs


class Monitor(InProcClient):
    """A client that also watches the simulator for invariant breaches while an attacker plays.

    This is red-team instrumentation, not something a policy may see: the attackers themselves only ever read
    tool output. Two things are watched after every call:
    - two trains in one section at the same tick (would mean a reservation was released too early);
    - trains left standing under an authority they can never use, with no gridlock charged.
    """

    def __init__(self, session: Session):
        super().__init__(session)
        self.co_occupancy: dict[str, list[str]] = {}

    def call(self, tool: str, args: dict) -> str:
        out = super().call(tool, args)
        sim = self.session.sim
        where: dict[str, list[str]] = {}
        for tid, t in sim._trains.items():
            if t["phase"] == "in_section":
                where.setdefault(t["section"], []).append(tid)
        for section, ids in where.items():
            if len(ids) > 1:
                self.co_occupancy[f"t={sim.tick} {section}"] = sorted(ids)
        return out

    def stalled_under_authority(self) -> list[str]:
        state = self.session.sim.state()
        return sorted(
            t.id for t in state.trains if t.outcome is None and t.phase == "at_station" and t.authority is not None
        )


def play(scenario: Scenario, policy) -> tuple[dict, Monitor]:
    session = Session(scenario)
    monitor = Monitor(session)
    policy.run(monitor)
    final = session.close()
    return {
        "cost": final.cost.total,
        "gridlock_charge": final.cost.gridlock,
        "counters": {k: final.counts[k] for k in COUNTERS},
    }, monitor


# ---------------------------------------------------------------- best plan for a seed


def best_plan(preset: str, entry: dict) -> list[dict] | None:
    """The plan that scores 1.0: stored in the seed list (T1) or in an exported task's `tests/plan.json`."""
    if entry.get("plan"):
        return list(entry["plan"])
    for manifest_path in sorted(ROOT.glob("datasets/*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("preset") != preset:
            continue
        for task in manifest.get("tasks", []):
            if task["scenario_id"] != entry["scenario_id"]:
                continue
            plan_path = manifest_path.parent / task["task"] / "tests" / "plan.json"
            if plan_path.exists():
                return json.loads(plan_path.read_text(encoding="utf-8"))
    return None


# ---------------------------------------------------------------- stop-early curve


def prefix_costs(scenario: Scenario, calls: list[dict]) -> list[tuple[int, int, float, float]]:
    """(k, tick, share of the plan's orders given, cost) when the first k decision points of a plan are
    played and the agent then stops. Replaying a prefix is exact: the simulator is deterministic."""
    cuts = [i + 1 for i, call in enumerate(calls) if call["tool"] == "advance_to_next_event"]
    total_orders = sum(call["tool"] in ACTIONS for call in calls) or 1
    out = []
    for k, cut in enumerate([0, *cuts, len(calls)]):  # the last cut plays the whole plan
        session = Session(scenario)
        for call in calls[:cut]:
            if session.closed:
                break
            session.call(call["tool"], call["args"])
        tick, finished = session.sim.tick, session.sim.done
        share = sum(call["tool"] in ACTIONS for call in calls[:cut]) / total_orders
        out.append((k, tick, share, session.close().cost.total))
        if finished:
            break
    return out


# ---------------------------------------------------------------- battery


@dataclass
class SeedReport:
    seed: int
    scenario_id: str
    c_greedy: float
    c_best: float
    best_source: str
    plan_calls: int | None = None
    attackers: dict[str, dict] = field(default_factory=dict)
    neutral: dict[str, dict] = field(default_factory=dict)
    prefix: list[dict] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _score(entry: dict, cost: float) -> dict:
    s = reward_from(entry["c_greedy"], entry["c_best"], cost)
    return {"cost": cost, "reward": round(s.reward, 6), "unclipped": round(s.score_unclipped, 6)}


def _run_one(
    name: str, scenario, entry, policy, rep: SeedReport, threshold: float, *, neutral_to=None, floor=None
) -> dict:
    """Run one attacker and judge it by its family.

    - plain attacker: it must not earn reward (it does not dispatch well), so reward > threshold fails;
    - `floor` (a mutation of the best plan): the plan already scores 1.0, so the only question is whether
      removing or rewriting those calls makes it *cheaper* than the plan;
    - `neutral_to`: the mutation must change nothing at all.
    """
    result, monitor = play(scenario, policy)
    row = {**_score(entry, result["cost"]), "counters": result["counters"]}
    row["co_occupancy"] = monitor.co_occupancy
    stalled = monitor.stalled_under_authority()
    row["stalled_under_authority"] = stalled
    if monitor.co_occupancy:
        where = ", ".join(f"{k}: {'+'.join(v)}" for k, v in sorted(monitor.co_occupancy.items())[:3])
        rep.failures.append(f"{name}: two trains in one section ({where})")
    refused = list(getattr(policy, "regrant_refusals", ()))
    if refused:
        row["regrant_refusals"] = refused
    if neutral_to is not None:
        if row["cost"] < neutral_to - EPS:
            rep.failures.append(f"{name} costs {row['cost']} < {neutral_to}: a free action lowers cost")
        elif refused:
            # Not a cost finding: the interlocking refused to re-grant an authority it had just granted, so
            # the run left the plan and its cost says nothing about whether cancelling is free.
            row["inconclusive"] = True
            rep.warnings.append(
                f"{name} is inconclusive on this seed: {len(refused)} authority/authorities could not be "
                f"re-granted after being cancelled, e.g. {refused[0]}"
            )
        elif abs(row["cost"] - neutral_to) > EPS:
            rep.warnings.append(f"{name} costs {row['cost']} vs {neutral_to} (not cost-neutral)")
    elif floor is not None:
        if row["cost"] < floor - EPS:
            rep.failures.append(
                f"{name} costs {row['cost']} < the best plan's {floor}: the calls it drops are mispriced"
            )
    elif row["reward"] > threshold:
        rep.failures.append(f"{name} earned reward {row['reward']:.3f} without dispatching")
    if stalled and result["gridlock_charge"] == 0:
        rep.warnings.append(
            f"{name} ended with {len(stalled)} train(s) stood under an authority ({', '.join(stalled[:4])}) "
            "and no gridlock charged"
        )
    return row


def run_seed(preset: str, entry: dict, threshold: float, slow: bool = False) -> SeedReport:
    scenario = generate(PRESETS[preset], entry["seed"])
    rep = SeedReport(
        entry["seed"], scenario.scenario_id, entry["c_greedy"], entry["c_best"], entry.get("best_source", "?")
    )
    attackers = dict(ATTACKERS)
    if slow:
        attackers.update(SLOW_ATTACKERS)
    for name, make in attackers.items():
        rep.attackers[name] = _run_one(name, scenario, entry, make(), rep, threshold)

    plan = best_plan(preset, entry)
    if plan is None:
        rep.warnings.append(
            "no stored best plan for this seed: plan-based attackers and the stop-early curve were skipped"
        )
        return rep
    rep.plan_calls = len(plan)
    honest, _ = play(scenario, Replay(plan))
    if abs(honest["cost"] - entry["c_best"]) > 1e-3:
        rep.warnings.append(f"the stored plan costs {honest['cost']}, accepted c_best {entry['c_best']}")
    for name, policy in plan_attackers(plan).items():
        rep.attackers[name] = _run_one(name, scenario, entry, policy, rep, threshold, floor=honest["cost"])
    for name, policy in plan_neutral(plan).items():
        rep.neutral[name] = _run_one(name, scenario, entry, policy, rep, threshold, neutral_to=honest["cost"])

    full = _score(entry, honest["cost"])["reward"]
    for k, tick, share, cost in prefix_costs(scenario, plan):
        rep.prefix.append({"k": k, "tick": tick, "orders": round(share, 4), **_score(entry, cost)})
    # Stopping once every order is given is the same as finishing; only stops with orders left count.
    early = [p for p in rep.prefix if p["orders"] < 1 - EPS and p["reward"] >= full - EPS]
    if early and full > 0:
        p = early[0]
        rep.failures.append(
            f"stopping after {p['k']} decisions (t={p['tick']}, {p['orders']:.0%} of the orders) "
            f"scores {p['reward']:.3f} >= full {full:.3f}"
        )
    for frac in (0.25, 0.5):
        p = _at_fraction(rep.prefix, frac)
        if p and p["reward"] > 0.5:
            rep.warnings.append(f"stopping with {p['orders']:.0%} of the orders given keeps reward {p['reward']:.3f}")
    return rep


class Replay:
    """Plays a stored plan unchanged: the honest 1.0 run every plan mutation is compared against."""

    def __init__(self, plan: list[dict]):
        self.plan = plan

    def run(self, client) -> None:
        for call in self.plan:
            out = client.call(call["tool"], call["args"])
            if "SHIFT ENDED" in out or "R-SHIFT-ENDED" in out:
                return
        client.call("end_shift", {})


def _at_fraction(prefix: list[dict], share: float) -> dict | None:
    """The last stop at which at most `share` of the plan's orders had been given."""
    before = [p for p in prefix if p["orders"] <= share + EPS]
    return before[-1] if before else None


def accepted(preset: str, set_name: str) -> list[dict]:
    path = ROOT / "seeds" / f"{preset}-{set_name}.json"
    return json.loads(path.read_text(encoding="utf-8"))["accepted"]


def render_markdown(preset: str, reports: list[SeedReport], threshold: float) -> str:
    names = sorted({n for r in reports for n in r.attackers} | {n for r in reports for n in r.neutral})
    lines = [
        f"# Strategic red-team battery ({preset}, {len(reports)} seeds)",
        "",
        f"Reward = max(0, (C_greedy − C_agent) / (C_greedy − C_best)); 1.0 is the stored best plan, 0 is greedy "
        f"or worse. Fail rule: any attacker reward > {threshold}; two trains in one section; a cost-neutral "
        "mutation that lowers cost; a plan prefix that scores as well as the whole plan.",
        "",
        "## Reward per attacker (official reward; unclipped in brackets)",
        "",
        "| Seed | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for r in reports:
        cells = {**r.attackers, **r.neutral}
        row = " | ".join(
            f"{cells[n]['reward']:.2f} ({cells[n]['unclipped']:.2f})" if n in cells else "-" for n in names
        )
        lines.append(f"| {r.seed} | {row} |")
    lines += ["", "## What each attacker produced on the first seed", "",
              "| Attacker | Cost | Reward | gridlocks | crew timeouts | misroutes | wrong terminals |"
              " missed calls | unfinished |",
              "|---|---|---|---|---|---|---|---|---|"]
    first = reports[0]
    for name, row in {**first.attackers, **first.neutral}.items():
        c = row["counters"]
        lines.append(
            f"| {name} | {row['cost']:.1f} | {row['reward']:.2f} | {c['gridlocks']} | {c['crew_timeouts']} | "
            f"{c['misroutes']} | {c['wrong_terminals']} | {c['missed_calls']} | {c['unfinished_trains']} |"
        )
    lines += ["", "## Stop-early curve: reward when the best plan stops after giving a share of its orders", "",
              "| Seed | 10% | 25% | 50% | 75% | 90% | full |", "|---|---|---|---|---|---|---|"]
    for r in reports:
        if not r.prefix:
            lines.append(f"| {r.seed} | " + " | ".join(["-"] * 6) + " |")
            continue
        row = [_at_fraction(r.prefix, f) for f in (0.1, 0.25, 0.5, 0.75, 0.9)]
        cells = [f"{p['reward']:.2f} (t={p['tick']})" if p else "-" for p in row]
        lines.append(f"| {r.seed} | " + " | ".join(cells) + f" | {r.prefix[-1]['reward']:.2f} |")
    lines += ["", "## Findings", ""]
    for r in reports:
        for f in r.failures:
            lines.append(f"- FAIL seed {r.seed}: {f}")
        for w in r.warnings:
            lines.append(f"- WARN seed {r.seed}: {w}")
    if not any(r.failures or r.warnings for r in reports):
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--preset", default="T1", choices=sorted(PRESETS))
    p.add_argument("--set", default="dev", dest="set_name")
    p.add_argument("--seeds", type=int, default=5, help="first N accepted seeds")
    p.add_argument("--seed", type=int, action="append", help="run only these seeds (repeatable)")
    p.add_argument("--threshold", type=float, default=0.02, help="max reward a non-dispatching attacker may get")
    p.add_argument("--slow", action="store_true", help="also run the reference-based attackers (minutes per T2 seed)")
    p.add_argument("--out", default=str(ROOT / "results" / "redteam"))
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    a = p.parse_args(argv)

    entries = accepted(a.preset, a.set_name)
    entries = [e for e in entries if e["seed"] in set(a.seed)] if a.seed else entries[: a.seeds]
    if not entries:
        print("no matching seeds", file=sys.stderr)
        return 2
    jobs = max(1, min(a.jobs, len(entries)))
    args = ([a.preset] * len(entries), entries, [a.threshold] * len(entries), [a.slow] * len(entries))
    if jobs == 1:
        reports = [run_seed(*row) for row in zip(*args, strict=True)]
    else:
        with concurrent.futures.ProcessPoolExecutor(jobs) as pool:
            reports = list(pool.map(run_seed, *args))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"strategic-{a.preset}-{a.set_name}"
    (out / f"{stem}.json").write_text(json.dumps([asdict(r) for r in reports], indent=2), encoding="utf-8")
    md = render_markdown(a.preset, reports, a.threshold)
    (out / f"{stem}.md").write_text(md, encoding="utf-8")
    sys.stdout.write(md)
    failed = sum(len(r.failures) for r in reports)
    print(f"\n{'FAIL' if failed else 'PASS'}: {failed} failure(s); report in {out / (stem + '.md')}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
