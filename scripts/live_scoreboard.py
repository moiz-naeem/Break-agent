"""Live scoreboard for the v0.3 network benchmark (railway/net): Gemini Carbon vs base Qwen vs fine-tuned Qwen.

Reads benchmark rows read-only, from the synced `results/net_bench/<run>/rows.jsonl` files and optionally
straight from the cluster (`--ssh kosh`: `rows.jsonl` + `shards/shard-*.jsonl`, via `cat` only), merges them
(latest row per model, scenario and rep), and writes

    results/scoreboard/scoreboard.json   machine-readable numbers
    results/scoreboard/scoreboard.md     the tables
    docs/benchmark-results.md            the tables with a plain-English reading

Usage:

    python scripts/live_scoreboard.py                       # local synced rows only
    python scripts/live_scoreboard.py --ssh kosh            # plus the cluster's in-flight shard files
    python scripts/live_scoreboard.py --counts              # print row counts per run and exit (polling)

Statistics:
  * reward = -(C1 + C2 + C3 + C4 + C5); every number is averaged per seed first (over reps), then over seeds,
    so each held-out seed has equal weight however many reps finished;
  * 95% CI = seed bootstrap (4000 resamples, fixed RNG seed) of the per-seed means;
  * within-seed std = mean over seeds (with >= 2 reps) of the population std of reward across reps;
  * paired tables use only seeds completed by every model in the comparison;
  * the do-nothing floor is the deterministic cost of ending the shift at once on the same seeds (scale only).

The lineup is fixed: only the three models in LINEUP are ever reported. The offline teacher used to make
the fine-tuning data is not a benchmark competitor and never appears in a table.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import railway  # noqa: E402

railway.__path__ = [str(REPO / "railway")]  # this checkout's package, even if another one is installed

DEFAULT_SOURCE = Path(r"D:\Hoai Anh\Aalto\Hobbies\Afterquery-net\results\net_bench")
CLUSTER_RUNS = "~/projects/afterquery-cluster/runs"
OUT_DIR = REPO / "results" / "scoreboard"
DOC = REPO / "docs" / "benchmark-results.md"
DO_NOTHING_CACHE = OUT_DIR / "do_nothing_cache.json"
TIERS = ("T1", "T2", "T3")
MIN_VERDICT_SEEDS = 5  # fewer paired seeds than this: report the numbers, call no winner

# key -> run, accepted model ids, display label, expected episodes when the run is complete.
LINEUP = {
    "carbon": {
        "run": "net-test-carbon-20261115",
        "models": {"gemini-carbon-high"},
        "label": "Gemini 4 Carbon",
        "id": "gemini-carbon-high",
        "expected": 270,
    },
    "qwen": {
        "run": "net-test-qwen-20261115",
        "models": {"qwen38-think"},
        "label": "Qwen 3.8 27B (base)",
        "id": "qwen38-think",
        "expected": 270,
    },
    "sft": {
        "run": "net-test-qwen-sft-20261115",
        "models": {"qwen38-think-sft", "qwen38-sft"},
        "label": "Qwen 3.8 27B (fine-tuned)",
        "id": "qwen38-think-sft",
        "expected": 30,
    },
}
COMPONENTS = ("C1", "C2", "C3", "C4", "C5")
COMPONENT_NAMES = {
    "C1": "passenger lateness",
    "C2": "unfinished trains / missed calls",
    "C3": "freight lateness",
    "C4": "crew timeouts",
    "C5": "gridlock and operations (unplanned stops, platform changes)",
}


# ---------------------------------------------------------------- loading


def _parse_lines(text: str, origin: str) -> list[dict]:
    rows = []
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"warning: {origin}:{i} is not JSON (partial write?), skipped", file=sys.stderr)
    return rows


def read_local(source: Path, run: str) -> list[dict]:
    path = source / run / "rows.jsonl"
    return _parse_lines(path.read_text(encoding="utf-8"), str(path)) if path.is_file() else []


def read_cluster(host: str, run: str, timeout: int = 60) -> list[dict]:
    """`cat` the run's rows.jsonl and shard files on the cluster (read-only)."""
    cmd = f"cd {CLUSTER_RUNS}/{run} 2>/dev/null && cat rows.jsonl shards/shard-*.jsonl 2>/dev/null; true"
    try:
        got = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        print(f"warning: ssh {host} timed out for {run}", file=sys.stderr)
        return []
    if got.returncode != 0:
        print(f"warning: ssh {host} failed for {run}: {got.stderr.strip()[-200:]}", file=sys.stderr)
        return []
    return _parse_lines(got.stdout, f"{host}:{run}")


def merge(rows: list[dict]) -> list[dict]:
    """One row per (model, scenario, rep); an ok row beats an error row, later beats earlier."""
    best: dict[tuple, dict] = {}
    for r in rows:
        key = (r.get("model"), r.get("scenario_id"), int(r.get("rep", 0)))
        old = best.get(key)
        if old is None or r.get("status") == "ok" or old.get("status") != "ok":
            best[key] = r
    return list(best.values())


def load(source: Path, ssh: str | None) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """key -> merged rows of that lineup model; plus per-run source counts."""
    out, counts = {}, {}
    for key, spec in LINEUP.items():
        local = read_local(source, spec["run"])
        remote = read_cluster(ssh, spec["run"]) if ssh else []
        merged = [r for r in merge(local + remote) if r.get("model") in spec["models"]]
        out[key] = merged
        counts[key] = {"local_rows": len(local), "cluster_rows": len(remote), "merged": len(merged)}
    return out, counts


def accepted_hashes() -> dict[str, str]:
    out = {}
    for tier in TIERS:
        path = REPO / "seeds" / f"net-{tier}-test.json"
        if path.is_file():
            for e in json.loads(path.read_text(encoding="utf-8"))["accepted"]:
                out[e["scenario_id"]] = e["scenario_hash"]
    return out


# ---------------------------------------------------------------- statistics


def bootstrap_ci(values: list[float], n: int = 4000, seed: int = 0) -> list[float | None]:
    if not values:
        return [None, None]
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choices(values, k=k)) / k for _ in range(n))
    return [means[int(0.025 * n)], means[int(0.975 * n) - 1]]


def mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def _metric(r: dict, name: str) -> float:
    if name == "reward":
        return float(r["reward"])
    if name in COMPONENTS:
        return float((r.get("components") or {}).get(name, 0.0))
    kp = r.get("kpis") or {}
    if name == "clean_shift":
        return 100.0 if kp.get("clean_shift") else 0.0
    if name in ("gridlocks", "unfinished"):
        return float(r.get(name) or (r.get("counts") or {}).get(name) or 0)
    return float(kp.get(name) or 0.0)


METRICS = (
    "reward",
    *COMPONENTS,
    "on_time_pct",
    "passenger_delay_minutes",
    "freight_delivered_pct",
    "clean_shift",
    "gridlocks",
    "unfinished",
)


def per_seed(rows: list[dict]) -> dict[tuple[str, int], dict[str, list[float]]]:
    """(tier, seed) -> metric -> values over ok reps."""
    out: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("status") == "ok":
            cell = out[(r["tier"], int(r["seed"]))]
            for m in METRICS:
                cell[m].append(_metric(r, m))
    return out


def tier_seeds(ps: dict, tier: str) -> list[tuple[str, int]]:
    return sorted(k for k in ps if tier == "ALL" or k[0] == tier)


def summarize(rows: list[dict], ps: dict, tier: str, floor: dict[str, float]) -> dict:
    keys = tier_seeds(ps, tier)
    tier_rows = [r for r in rows if tier == "ALL" or r.get("tier") == tier]
    seed_mean = {m: [statistics.fmean(ps[k][m]) for k in keys] for m in METRICS}
    stds = [statistics.pstdev(ps[k]["reward"]) for k in keys if len(ps[k]["reward"]) > 1]
    floors = [floor[f"{t}-{s}"] for t, s in keys if f"{t}-{s}" in floor]
    return {
        "seeds": len(keys),
        "episodes_ok": sum(len(ps[k]["reward"]) for k in keys),
        "episodes_error": sum(1 for r in tier_rows if r.get("status") != "ok"),
        "mean_reward": mean(seed_mean["reward"]),
        "ci95": bootstrap_ci(seed_mean["reward"]),
        "within_seed_std": mean(stds),
        "seeds_with_reps": len(stds),
        **{m: mean(seed_mean[m]) for m in METRICS if m != "reward"},
        "do_nothing_reward": mean(floors),
        "cost_usd": sum(float(r.get("cost_usd") or 0) for r in tier_rows),
    }


def paired(psa: dict, psb: dict, tier: str) -> dict | None:
    keys = sorted(set(tier_seeds(psa, tier)) & set(tier_seeds(psb, tier)))
    if not keys:
        return None
    a = [statistics.fmean(psa[k]["reward"]) for k in keys]
    b = [statistics.fmean(psb[k]["reward"]) for k in keys]
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    return {
        "seeds": len(keys),
        "mean_a": statistics.fmean(a),
        "mean_b": statistics.fmean(b),
        "mean_diff": statistics.fmean(diffs),
        "ci95": bootstrap_ci(diffs),
        "wins": sum(d > 0 for d in diffs),
        "ties": sum(d == 0 for d in diffs),
        "losses": sum(d < 0 for d in diffs),
    }


def post_training(ps: dict[str, dict]) -> dict:
    """T1 seeds finished by all three models: what fine-tuning changed and how much of the gap it closed."""
    keys = sorted(set.intersection(*(set(tier_seeds(ps[k], "T1")) for k in ("carbon", "qwen", "sft"))))
    out: dict = {"seeds": len(keys), "seed_list": [s for _, s in keys]}
    if not keys:
        return out
    sub = {k: {s: ps[k][s] for s in keys} for k in ("carbon", "qwen", "sft")}
    for k in sub:
        vals = [statistics.fmean(sub[k][s]["reward"]) for s in keys]
        out[f"{k}_mean"] = statistics.fmean(vals)
        out[f"{k}_ci95"] = bootstrap_ci(vals)
        for m in ("on_time_pct", "unfinished", "clean_shift"):
            out[f"{k}_{m}"] = statistics.fmean(statistics.fmean(sub[k][s][m]) for s in keys)
    out["sft_minus_base"] = paired(sub["sft"], sub["qwen"], "T1")
    out["carbon_minus_sft"] = paired(sub["carbon"], sub["sft"], "T1")
    out["carbon_minus_base"] = paired(sub["carbon"], sub["qwen"], "T1")
    gap = out["carbon_minus_base"]["mean_diff"]
    gain = out["sft_minus_base"]["mean_diff"]
    out["gap_closed_pct"] = 100.0 * gain / gap if gap > 0 else None
    return out


# ---------------------------------------------------------------- do-nothing floor


def do_nothing_floor(seeds: set[tuple[str, int]]) -> dict[str, float]:
    cache: dict[str, float] = {}
    if DO_NOTHING_CACHE.is_file():
        cache = json.loads(DO_NOTHING_CACHE.read_text(encoding="utf-8"))
    missing = sorted(s for s in seeds if f"{s[0]}-{s[1]}" not in cache)
    if missing:
        from railway.net.exploits import build_timetable, do_nothing

        for tier, seed in missing:
            cache[f"{tier}-{seed}"] = -float(do_nothing(build_timetable(tier, seed)).cost.total)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        DO_NOTHING_CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return cache


# ---------------------------------------------------------------- report


def build(source: Path, ssh: str | None, floor_on: bool = True) -> dict:
    data, counts = load(source, ssh)
    hashes = accepted_hashes()
    ps = {k: per_seed(rows) for k, rows in data.items()}
    all_seeds = {k for p in ps.values() for k in p}
    floor = do_nothing_floor(all_seeds) if floor_on else {}
    rep: dict = {
        "generated_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "sources": {"local": str(source), "cluster": ssh},
        "lineup": {k: {x: v for x, v in s.items() if x != "models"} for k, s in LINEUP.items()},
        "progress": {},
        "overall": {},
        "head_to_head": {},
        "integrity": {},
    }
    for k, spec in LINEUP.items():
        rows = data[k]
        ok = sum(1 for r in rows if r.get("status") == "ok")
        rep["progress"][k] = counts[k] | {
            "ok": ok,
            "errors": len(rows) - ok,
            "expected": spec["expected"],
            "pct": 100.0 * ok / spec["expected"],
            "by_tier": {t: sum(1 for r in rows if r.get("tier") == t and r.get("status") == "ok") for t in TIERS},
        }
        bad_hash = [
            r["scenario_id"]
            for r in rows
            if r.get("scenario_id") in hashes and r.get("scenario_hash") != hashes[r["scenario_id"]]
        ]
        rep["integrity"][k] = {
            "non_test_split": sum(1 for r in rows if r.get("split") != "test"),
            "scenario_hash_mismatch": sorted(set(bad_hash)),
            "unknown_scenarios": sorted({r["scenario_id"] for r in rows if r.get("scenario_id") not in hashes}),
        }
        rep["overall"][k] = {t: summarize(rows, ps[k], t, floor) for t in (*TIERS, "ALL") if tier_seeds(ps[k], t)}
    for t in (*TIERS, "ALL"):
        p = paired(ps["carbon"], ps["qwen"], t)
        if p:
            rep["head_to_head"][t] = p
    rep["post_training"] = post_training(ps)
    p = paired(ps["sft"], ps["qwen"], "T1")
    rep["sft_vs_base_all_t1"] = p
    return rep


def f(x, nd: int = 1) -> str:
    return "–" if x is None else f"{x:,.{nd}f}"


def ci(c) -> str:
    return "–" if not c or c[0] is None else f"[{f(c[0])}, {f(c[1])}]"


def tables(rep: dict) -> list[str]:
    lab = {k: s["label"] for k, s in LINEUP.items()}
    L = [f"_Snapshot: {rep['generated_utc']}._", "", "### Progress", ""]
    L += ["| model | id | episodes ok / expected | T1 | T2 | T3 | errors |", "|---|---|---|---|---|---|---|"]
    for k, p in rep["progress"].items():
        bt = p["by_tier"]
        L.append(
            f"| {lab[k]} | `{LINEUP[k]['id']}` | {p['ok']} / {p['expected']} ({p['pct']:.0f}%) "
            f"| {bt['T1']} | {bt['T2']} | {bt['T3']} | {p['errors']} |"
        )
    L += ["", "### Overall score per tier", ""]
    L += [
        "Reward = −(C1+C2+C3+C4+C5), higher is better; averaged per seed first. "
        "CI = 95% seed bootstrap. Rep std = mean within-seed std across repeated runs.",
        "",
        "| tier | model | seeds | eps | mean reward [95% CI] | rep std | do-nothing floor | "
        "C1 | C2 | C3 | C4 | C5 | on-time % | pax delay min | freight % | clean % | gridlocks | unfinished |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in (*TIERS, "ALL"):
        for k in LINEUP:
            s = rep["overall"][k].get(t)
            if not s:
                continue
            L.append(
                f"| {t} | {lab[k]} | {s['seeds']} | {s['episodes_ok']} | **{f(s['mean_reward'])}** {ci(s['ci95'])} "
                f"| {f(s['within_seed_std'])} | {f(s['do_nothing_reward'])} | "
                + " | ".join(f(s[c]) for c in COMPONENTS)
                + f" | {f(s['on_time_pct'])} | {f(s['passenger_delay_minutes'], 0)} | {f(s['freight_delivered_pct'])} "
                f"| {f(s['clean_shift'])} | {f(s['gridlocks'], 2)} | {f(s['unfinished'], 2)} |"
            )
    L += ["", "### Head-to-head: Gemini 4 Carbon vs Qwen 3.8 27B (base), paired seeds", ""]
    L += [
        "Difference = Carbon − Qwen in mean reward on seeds both finished; positive means Carbon is better.",
        "",
        "| tier | paired seeds | Carbon | Qwen base | difference [95% CI] | Carbon wins / ties / Qwen wins |",
        "|---|---|---|---|---|---|",
    ]
    if not rep["head_to_head"]:
        L.append("| – | 0 | – | – | no seed finished by both yet | – |")
    for t, p in rep["head_to_head"].items():
        L.append(
            f"| {t} | {p['seeds']} | {f(p['mean_a'])} | {f(p['mean_b'])} | **{f(p['mean_diff'])}** {ci(p['ci95'])} "
            f"| {p['wins']} / {p['ties']} / {p['losses']} |"
        )
    pt = rep["post_training"]
    L += ["", "### Post-training: what fine-tuning changed (T1 held-out test seeds)", ""]
    L += [f"Seeds finished by all three models: **{pt['seeds']}**."]
    if pt["seeds"]:
        L += [
            "",
            "| model | mean reward [95% CI] | on-time % | unfinished trains | clean % |",
            "|---|---|---|---|---|",
        ]
        for k in ("carbon", "sft", "qwen"):
            L.append(
                f"| {lab[k]} | **{f(pt[k + '_mean'])}** {ci(pt[k + '_ci95'])} | {f(pt[k + '_on_time_pct'])} "
                f"| {f(pt[k + '_unfinished'], 2)} | {f(pt[k + '_clean_shift'])} |"
            )
        L += ["", "| comparison | difference [95% CI] | A wins / ties / B wins |", "|---|---|---|"]
        for name, key in (
            ("fine-tuned − base Qwen (what Qwen learned)", "sft_minus_base"),
            ("Carbon − fine-tuned Qwen (gap left)", "carbon_minus_sft"),
            ("Carbon − base Qwen (gap before training)", "carbon_minus_base"),
        ):
            p = pt[key]
            wtl = f"{p['wins']} / {p['ties']} / {p['losses']}"
            L.append(f"| {name} | **{f(p['mean_diff'])}** {ci(p['ci95'])} | {wtl} |")
        L += ["", f"Share of the Carbon-to-Qwen gap closed by fine-tuning: **{f(pt['gap_closed_pct'])}%**."]
    sv = rep.get("sft_vs_base_all_t1")
    if sv and sv["seeds"] != pt["seeds"]:
        L += [
            "",
            f"Fine-tuned vs base Qwen on all {sv['seeds']} T1 seeds both finished (Carbon not required): "
            f"{f(sv['mean_diff'])} {ci(sv['ci95'])}, wins/ties/losses {sv['wins']}/{sv['ties']}/{sv['losses']}.",
        ]
    integ = rep["integrity"]
    issues = [
        f"{lab[k]}: {', '.join(f'{n}={v}' for n, v in i.items() if v)}" for k, i in integ.items() if any(i.values())
    ]
    L += ["", "### Integrity checks", ""]
    L += (
        ["- " + x for x in issues]
        if issues
        else ["All rows are held-out `test` split seeds and every scenario hash matches `seeds/net-T*-test.json`."]
    )
    return L


def render_md(rep: dict) -> str:
    return "\n".join(["# Network benchmark scoreboard (v0.3, live)", "", *tables(rep)]) + "\n"


def reading(rep: dict) -> list[str]:
    """Plain-English summary computed from the numbers (no hand-written claims)."""
    lab = {k: s["label"] for k, s in LINEUP.items()}
    out = []
    for k, p in rep["progress"].items():
        out.append(f"{lab[k]} has finished {p['ok']} of {p['expected']} planned episodes ({p['pct']:.0f}%).")
    for t, p in rep["head_to_head"].items():
        if t == "ALL":
            continue
        lo, hi = p["ci95"]
        if p["seeds"] < MIN_VERDICT_SEEDS:
            verdict = f"too few paired seeds for a verdict (need {MIN_VERDICT_SEEDS})"
        elif lo is not None and lo > 0:
            verdict = "Carbon is clearly better"
        elif hi is not None and hi < 0:
            verdict = "base Qwen is clearly better"
        else:
            verdict = "the difference is not yet distinguishable from noise"
        out.append(
            f"On {t}, over {p['seeds']} seeds both models finished, Carbon scores {f(p['mean_a'])} and base Qwen "
            f"{f(p['mean_b'])} ({f(p['mean_diff'])} points apart, 95% CI {ci(p['ci95'])}): {verdict}."
        )
    pt = rep["post_training"]
    if pt["seeds"]:
        g = pt["sft_minus_base"]
        out.append(
            f"On the {pt['seeds']} T1 test seeds all three models finished, fine-tuning moved Qwen from "
            f"{f(pt['qwen_mean'])} to {f(pt['sft_mean'])} ({'+' if g['mean_diff'] >= 0 else ''}{f(g['mean_diff'])}, "
            f"95% CI {ci(g['ci95'])}; better on {g['wins']} seeds, worse on {g['losses']}), against "
            f"{f(pt['carbon_mean'])} for Carbon"
            + (f", closing {f(pt['gap_closed_pct'])}% of the gap." if pt["gap_closed_pct"] is not None else ".")
        )
    else:
        out.append("No T1 seed has been finished by all three models yet, so the post-training table is still empty.")
    return out


def render_doc(rep: dict) -> str:
    L = [
        "# Benchmark results: railway network (v0.3)",
        "",
        "Live results on the held-out **test** seeds (10000–19999) of the v0.3 network benchmark, regenerated by",
        "`scripts/live_scoreboard.py` while episodes arrive. Three models are compared: Gemini 4 Carbon",
        "(`gemini-carbon-high`), Qwen 3.8 27B before post-training (`qwen38-think`) and the same Qwen after",
        "supervised fine-tuning (`qwen38-think-sft`). The fine-tuning data came from an offline teacher on the",
        "disjoint train seeds (20000+), so no test seed was ever seen in training.",
        "",
        "## How to read the numbers",
        "",
        "- **Reward** is the negative total cost, −(C1+C2+C3+C4+C5); closer to zero is better. The cost parts are "
        + "; ".join(f"{c} {n}" for c, n in COMPONENT_NAMES.items())
        + ".",
        "- Every seed counts equally: reps of a seed are averaged first, then seeds are averaged.",
        "- The **95% CI** resamples seeds (bootstrap). **Rep std** is how much a model's reward moves between "
        "repeated runs of the same seed: the benchmark's run-to-run noise.",
        "- **Paired** comparisons use only seeds both models finished, so a hard seed cannot favour either side.",
        "- The **do-nothing floor** is the deterministic score of ending the shift at once on the same seeds; "
        "it shows the scale, it is not a competitor.",
        "",
        "## Reading of the current snapshot",
        "",
        *[f"- {x}" for x in reading(rep)],
        "",
        "## Tables",
        "",
        *tables(rep),
    ]
    return "\n".join(L) + "\n"


def write(rep: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "scoreboard.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8", newline="\n")
    (OUT_DIR / "scoreboard.md").write_text(render_md(rep), encoding="utf-8", newline="\n")
    DOC.write_text(render_doc(rep), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="synced results/net_bench directory")
    ap.add_argument("--ssh", default=None, help="also read rows and shards from this cluster host (read-only)")
    ap.add_argument("--counts", action="store_true", help="print merged row counts per run and exit")
    ap.add_argument("--no-floor", action="store_true", help="skip the do-nothing floor")
    args = ap.parse_args(argv)
    if args.counts:
        data, counts = load(args.source, args.ssh)
        print(json.dumps(counts, sort_keys=True))
        return 0
    rep = build(args.source, args.ssh, floor_on=not args.no_floor)
    write(rep)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_md(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
