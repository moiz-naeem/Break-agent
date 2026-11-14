"""Post-run analysis for `results/net_bench/<run>/` (local; Modal only for `--verify` downloads).

    python infra/modal/net_analysis.py --run net-test-20261115-002930 --run net-test-t23-20261115
    python infra/modal/net_analysis.py --run <run> --do-nothing --verify 3

Adds what `net_bench.py report` does not show, written to `<first run>/analysis.md` (+ `.json`):
  * paired model-vs-model difference per tier on the same seeds (seed-bootstrap CI, wins/ties/losses);
  * run-to-run noise: mean within-seed std of reward over reps, per model and tier;
  * failure profile: unfinished trains, gridlocks, crew timeouts, invalid/refusal/format errors,
    episodes that hit the tool budget, infra errors;
  * do-nothing cost per seed as a raw floor for scale (context only, never a normalizer);
  * `--verify N`: replays N episodes per model and tier from `actions.json` on the transcripts volume and
    checks the replayed cost equals the reported one.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "net_bench"
TRANSCRIPTS_VOLUME = "railway-transcripts"
TRANSCRIPTS_MOUNT = "/transcripts"


def load_rows(runs: list[str]) -> list[dict]:
    """Latest row per (model, scenario, rep) across the given runs."""
    latest: dict[tuple, dict] = {}
    for run in runs:
        path = RESULTS / run / "rows.jsonl"
        if not path.is_file():
            print(f"warning: {path} missing", file=sys.stderr)
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                latest[(r["model"], r["scenario_id"], int(r["rep"]))] = r
    return list(latest.values())


def bootstrap_ci(values: list[float], n: int = 4000, seed: int = 0) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choices(values, k=k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def seed_means(rows: list[dict]) -> dict[tuple[str, str], dict[int, list[float]]]:
    """(model, tier) -> seed -> rewards over reps (ok rows only)."""
    out: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("status") == "ok":
            out[(r["model"], r["tier"])][int(r["seed"])].append(float(r["reward"]))
    return out


def paired(by: dict, tier: str, a: str, b: str) -> dict | None:
    sa, sb = by.get((a, tier), {}), by.get((b, tier), {})
    seeds = sorted(set(sa) & set(sb))
    if not seeds:
        return None
    diffs = [statistics.fmean(sa[s]) - statistics.fmean(sb[s]) for s in seeds]
    lo, hi = bootstrap_ci(diffs)
    return {
        "seeds": len(seeds),
        "mean_diff": statistics.fmean(diffs),
        "ci95": [lo, hi],
        "wins": sum(d > 0 for d in diffs),
        "ties": sum(d == 0 for d in diffs),
        "losses": sum(d < 0 for d in diffs),
    }


def noise(by: dict) -> dict[str, float | None]:
    out = {}
    for (model, tier), seeds in sorted(by.items()):
        stds = [statistics.pstdev(v) for v in seeds.values() if len(v) > 1]
        out[f"{model}|{tier}"] = statistics.fmean(stds) if stds else None
    return out


def failures(rows: list[dict], budget: int) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[f"{r['model']}|{r['tier']}"].append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        ok = [r for r in rs if r.get("status") == "ok"]
        n = len(ok) or 1

        def share(pred, ok=ok, n=n) -> float:
            return sum(1 for r in ok if pred(r)) / n

        out[key] = {
            "episodes": len(rs),
            "infra_errors": len(rs) - len(ok),
            "error_kinds": sorted({(r.get("error") or "")[:80] for r in rs if r.get("status") != "ok"}),
            "unfinished_eps": share(
                lambda r: (r.get("unfinished") or (r.get("counts") or {}).get("unfinished") or 0) > 0
            ),
            "gridlock_eps": share(lambda r: (r.get("cost_by_component") or {}).get("gridlock", 0) > 0),
            "crew_timeout_eps": share(lambda r: (r.get("cost_by_component") or {}).get("crew_timeout", 0) > 0),
            "budget_hit_eps": share(lambda r: (r.get("tool_calls") or 0) >= budget),
            "invalid_per_ep": statistics.fmean([r.get("invalid_calls") or 0 for r in ok]) if ok else None,
            "refusals_per_ep": statistics.fmean([r.get("refusals") or 0 for r in ok]) if ok else None,
            "format_errors_per_ep": statistics.fmean([r.get("format_errors") or 0 for r in ok]) if ok else None,
        }
    return out


def do_nothing_floor(rows: list[dict]) -> dict[str, dict]:
    sys.path.insert(0, str(REPO))
    from railway.net.exploits import build_timetable, do_nothing

    seeds = sorted({(r["tier"], int(r["seed"])) for r in rows})
    per_tier: dict[str, list[float]] = defaultdict(list)
    for tier, seed in seeds:
        per_tier[tier].append(-float(do_nothing(build_timetable(tier, seed)).cost.total))
    return {t: {"seeds": len(v), "mean_reward": statistics.fmean(v)} for t, v in sorted(per_tier.items())}


def verify(rows: list[dict], per_group: int) -> list[dict]:
    """Replay `actions.json` for up to `per_group` ok episodes per (model, tier); compare costs."""
    sys.path.insert(0, str(REPO))
    from railway.net.exploits import build_timetable
    from railway.net.replay import ReplayMismatch, replay

    picked: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["model"], r["tier"], int(r["seed"]), int(r["rep"]))):
        key = f"{r['model']}|{r['tier']}"
        if r.get("status") == "ok" and r.get("transcript_dir") and len(picked[key]) < per_group:
            picked[key].append(r)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for r in itertools.chain.from_iterable(picked.values()):
            remote = r["transcript_dir"].removeprefix(TRANSCRIPTS_MOUNT).strip("/") + "/actions.json"
            local = Path(tmp) / f"{abs(hash(remote))}.json"
            got = subprocess.run(
                ["modal", "volume", "get", TRANSCRIPTS_VOLUME, remote, str(local)], capture_output=True, text=True
            )
            item = {
                "model": r["model"],
                "tier": r["tier"],
                "seed": r["seed"],
                "rep": r["rep"],
                "reported": r["cost_total"],
            }
            if got.returncode != 0 or not local.is_file():
                results.append(item | {"ok": False, "why": f"download failed: {got.stderr.strip()[-120:]}"})
                continue
            entries = json.loads(local.read_text(encoding="utf-8"))
            try:
                final = replay(build_timetable(r["tier"], int(r["seed"])), {"entries": entries})
                same = abs(final.cost.total - float(r["cost_total"])) < 1e-6
                results.append(item | {"ok": same, "replayed": final.cost.total})
            except ReplayMismatch as exc:
                results.append(item | {"ok": False, "why": str(exc)[:160]})
    return results


def fmt(x, nd: int = 1) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def render(runs: list[str], rep: dict) -> str:
    lines = [f"# Network benchmark analysis: {', '.join(f'`{r}`' for r in runs)}", ""]
    lines += [
        "## Paired comparison (same seeds)",
        "",
        "Difference in mean reward (= -cost) per seed, A minus B; positive means A is better.",
        "",
        "| tier | A | B | seeds | mean diff [95% CI] | A wins / ties / B wins |",
        "|---|---|---|---|---|---|",
    ]
    for p in rep["paired"]:
        lo, hi = p["ci95"]
        lines.append(
            f"| {p['tier']} | {p['a']} | {p['b']} | {p['seeds']} | {fmt(p['mean_diff'])} [{fmt(lo)}, {fmt(hi)}] "
            f"| {p['wins']} / {p['ties']} / {p['losses']} |"
        )
    lines += [
        "",
        "## Run-to-run noise",
        "",
        "Mean within-seed standard deviation of reward over reps.",
        "",
        "| model / tier | std |",
        "|---|---|",
    ]
    lines += [f"| {k} | {fmt(v)} |" for k, v in rep["noise"].items()]
    lines += [
        "",
        "## Failure profile",
        "",
        "| model / tier | eps | infra err | unfinished | gridlock | crew timeout | budget hit "
        "| invalid/ep | refusals/ep | format err/ep |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, f in rep["failures"].items():
        pct = lambda x: f"{100 * x:.0f}%"  # noqa: E731
        lines.append(
            f"| {k} | {f['episodes']} | {f['infra_errors']} | {pct(f['unfinished_eps'])} | {pct(f['gridlock_eps'])} "
            f"| {pct(f['crew_timeout_eps'])} | {pct(f['budget_hit_eps'])} | {fmt(f['invalid_per_ep'])} "
            f"| {fmt(f['refusals_per_ep'])} | {fmt(f['format_errors_per_ep'])} |"
        )
    errs = {k: f["error_kinds"] for k, f in rep["failures"].items() if f["error_kinds"]}
    if errs:
        lines += ["", "Infra error kinds:"] + [f"- {k}: {'; '.join(v)}" for k, v in errs.items()]
    if rep.get("do_nothing"):
        lines += ["", "## Do-nothing floor (context only)", "", "| tier | seeds | mean reward |", "|---|---|---|"]
        lines += [f"| {t} | {d['seeds']} | {fmt(d['mean_reward'])} |" for t, d in rep["do_nothing"].items()]
    if rep.get("verify") is not None:
        v = rep["verify"]
        bad = [x for x in v if not x["ok"]]
        lines += [
            "",
            "## Replay verification",
            "",
            f"{len(v) - len(bad)}/{len(v)} replayed episodes match the reported cost.",
        ]
        lines += [
            f"- MISMATCH {x['model']} {x['tier']} seed {x['seed']} r{x['rep']}: {x.get('why') or x.get('replayed')}"
            for x in bad
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="results/net_bench/<run>; repeatable")
    ap.add_argument("--do-nothing", action="store_true", help="compute the do-nothing floor per seed locally")
    ap.add_argument("--verify", type=int, default=0, help="replay N episodes per model and tier")
    ap.add_argument("--budget", type=int, default=600, help="tool budget (railway.net.manual.TOOL_BUDGET)")
    args = ap.parse_args(argv)

    rows = load_rows(args.run)
    by = seed_means(rows)
    models = sorted({m for m, _ in by})
    tiers = sorted({t for _, t in by})
    rep = {
        "paired": [
            p | {"tier": t, "a": a, "b": b}
            for t in tiers
            for a, b in itertools.combinations(models, 2)
            if (p := paired(by, t, a, b))
        ],
        "noise": noise(by),
        "failures": failures(rows, args.budget),
    }
    if args.do_nothing:
        rep["do_nothing"] = do_nothing_floor(rows)
    if args.verify:
        rep["verify"] = verify(rows, args.verify)
    out = RESULTS / args.run[0]
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(rep, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    md = render(args.run, rep)
    (out / "analysis.md").write_text(md, encoding="utf-8", newline="\n")
    print(md)
    return rep


if __name__ == "__main__":
    main()
