"""Strategic red-team battery: policies that try to earn reward without dispatching well.

Every attacker plays through the same tool API as an agent (a Client on a Session), never the scenario.
Rewards use the official formula against each seed's accepted c_greedy and c_ref. The battery fails when:
- a non-dispatching attacker earns reward above the threshold;
- an action that should be cost-neutral (cancel and re-grant) makes the reference cheaper;
- stopping the reference early scores at least as well as letting it finish.

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
from railway.policies.baselines import Greedy, Policy, waiting_trains
from railway.policies.exploits import EXPLOITS
from railway.ref.reference import Reference
from railway.sim.model import Scenario
from railway.tools.client import InProcClient
from railway.tools.observe import Observer
from railway.tools.session import Session

ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-6
ACTIONS = {"authorize", "cancel_authority", "hold", "relieve_crew", "reverse"}  # tools that change the plan


# ---------------------------------------------------------------- extra attackers


class HoldAll(Policy):
    """Hold every train until the shift end, then only advance: tests that `hold` is never rewarded."""

    def play(self, client) -> None:
        observer = Observer(client)
        static, state = observer.observe()
        for t in static.trains:
            client.call("hold", {"train": t["id"], "until_tick": static.shift_ticks})
        while True:
            client.call("advance_to_next_event", {})


class PassengersOnly(Greedy):
    """Greedy that never moves freight: the 'serve the weighted trains, abandon the rest' trade."""

    def choose(self, static, state, belief):
        freight = {t.id for t in state.trains if t.type == "freight"}
        for action in super().choose(static, state, belief):
            if action.train not in freight:
                yield action


class ReverseAbuse(Greedy):
    """Greedy plus `reverse` on every train at every decision point: recovery must never pay."""

    def decide(self, static, state):
        calls = super().decide(static, state)
        reverses = [type(calls[0])("reverse", {"train": t.id}) for t in waiting_trains(state)]
        return [*reverses, *calls]


class CancelChurn(Policy):
    """The reference, but every grant is cancelled and re-requested. Cost must equal the reference's."""

    def play(self, client) -> None:
        reference, observer = Reference(), Observer(client)
        while True:
            for call in reference.decide(*observer.observe()):
                out = client.call(call.tool, call.args)
                if call.tool == "authorize" and out.startswith("GRANTED"):
                    client.call("cancel_authority", {"train": call.args["train"]})
                    client.call(call.tool, call.args)


ATTACKERS = {
    **EXPLOITS,
    "HoldAll": HoldAll,
    "PassengersOnly": PassengersOnly,
    "ReverseAbuse": ReverseAbuse,
}
NEUTRAL = {"CancelChurn": CancelChurn}  # must cost exactly what the reference costs


# ---------------------------------------------------------------- reference prefixes


class _Recorder(InProcClient):
    def __init__(self, session):
        super().__init__(session)
        self.calls: list[tuple[str, dict]] = []

    def call(self, tool, args):
        self.calls.append((tool, args))
        return super().call(tool, args)


def reference_calls(scenario: Scenario) -> tuple[list[tuple[str, dict]], float]:
    session = Session(scenario)
    rec = _Recorder(session)
    Reference().run(rec)
    return rec.calls, session.close().cost.total


def prefix_costs(scenario: Scenario, calls: list[tuple[str, dict]]) -> list[tuple[int, int, float, float]]:
    """(k, tick, share of the reference's orders given, cost) when the reference's first k decision points
    are played and the agent then stops.

    Replaying the recorded calls is exact: the reference sees only tool outputs, and the simulator is
    deterministic, so its first k decision points are the same with or without what comes after.
    """
    cuts = [i for i, (tool, _) in enumerate(calls) if tool == "advance_to_next_event"]
    total_orders = sum(tool in ACTIONS for tool, _ in calls) or 1
    out = []
    for k, cut in enumerate([0, *[c + 1 for c in cuts]]):
        session = Session(scenario)
        for tool, args in calls[:cut]:
            session.call(tool, args)
        tick, finished = session.sim.tick, session.sim.done
        share = sum(tool in ACTIONS for tool, _ in calls[:cut]) / total_orders
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
    c_ref: float
    c_ref_now: float
    attackers: dict[str, dict] = field(default_factory=dict)
    neutral: dict[str, dict] = field(default_factory=dict)
    prefix: list[dict] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _score(entry, cost):
    s = reward_from(entry["c_greedy"], entry["c_ref"], cost)
    return {"cost": cost, "reward": round(s.reward, 6), "unclipped": round(s.score_unclipped, 6)}


def run_seed(preset: str, entry: dict, threshold: float) -> SeedReport:
    scenario = generate(PRESETS[preset], entry["seed"])
    calls, c_ref_now = reference_calls(scenario)
    rep = SeedReport(entry["seed"], scenario.scenario_id, entry["c_greedy"], entry["c_ref"], c_ref_now)
    if abs(c_ref_now - entry["c_ref"]) > EPS:
        rep.warnings.append(f"reference cost {c_ref_now} differs from the accepted c_ref {entry['c_ref']}")

    for name, make in ATTACKERS.items():
        session = Session(scenario)
        make().run(InProcClient(session))
        rep.attackers[name] = r = _score(entry, session.close().cost.total)
        if r["reward"] > threshold:
            rep.failures.append(f"{name} earned reward {r['reward']:.3f} without dispatching")

    for name, make in NEUTRAL.items():
        session = Session(scenario)
        make().run(InProcClient(session))
        rep.neutral[name] = r = _score(entry, session.close().cost.total)
        if r["cost"] < c_ref_now - EPS:
            rep.failures.append(f"{name} costs {r['cost']} < reference {c_ref_now}: a free action lowers cost")
        elif abs(r["cost"] - c_ref_now) > EPS:
            rep.warnings.append(f"{name} costs {r['cost']} vs reference {c_ref_now} (not cost-neutral)")

    prefixes = prefix_costs(scenario, calls)
    full = _score(entry, c_ref_now)["reward"]
    for k, tick, share, cost in prefixes:
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


def _at_fraction(prefix: list[dict], share: float) -> dict | None:
    """The last stop at which at most `share` of the reference's orders had been given."""
    before = [p for p in prefix if p["orders"] <= share + EPS]
    return before[-1] if before else None


def accepted(preset: str, set_name: str) -> list[dict]:
    path = ROOT / "seeds" / f"{preset}-{set_name}.json"
    return json.loads(path.read_text(encoding="utf-8"))["accepted"]


def render_markdown(preset: str, reports: list[SeedReport], threshold: float) -> str:
    names = [*ATTACKERS, *NEUTRAL]
    lines = [
        f"# Strategic red-team battery ({preset}, {len(reports)} seeds)",
        "",
        f"Fail rule: any attacker reward > {threshold}; a cost-neutral action that lowers cost; "
        "a reference prefix that scores at least the full reference.",
        "",
        "## Reward per attacker (official clipped reward; unclipped in brackets)",
        "",
        "| Seed | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for r in reports:
        cells = {**r.attackers, **r.neutral}
        row = " | ".join(f"{cells[n]['reward']:.2f} ({cells[n]['unclipped']:.2f})" for n in names)
        lines.append(f"| {r.seed} | {row} |")
    lines += ["", "## Stop-early curve: reward when the reference stops after giving a share of its orders", "",
              "| Seed | 10% | 25% | 50% | 75% | 90% | full |", "|---|---|---|---|---|---|---|"]
    for r in reports:
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
    p.add_argument("--threshold", type=float, default=0.02, help="max reward a non-dispatching attacker may get")
    p.add_argument("--out", default=str(ROOT / "results" / "redteam"))
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    a = p.parse_args(argv)

    entries = accepted(a.preset, a.set_name)[: a.seeds]
    with concurrent.futures.ProcessPoolExecutor(max(1, min(a.jobs, len(entries)))) as pool:
        reports = list(pool.map(run_seed, [a.preset] * len(entries), entries, [a.threshold] * len(entries)))

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
