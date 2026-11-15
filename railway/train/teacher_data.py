"""Harbor teacher jobs -> a filtered, loop-format SFT dataset (samples.jsonl + episodes.jsonl + report.json).

    python -m railway.train.teacher_data --jobs jobs/harbor/teacher-opus55-droid-T1-train-* \
        --out data/teacher/v1 [--baseline results/net_bench/<run>/rows.jsonl --baseline-model qwen38-nothink]
        [--per-seed 1] [--max-advance-share 0.3]

Per trial: `result.json` (Harbor's exception info), `verifier/details.json` (tier, seed, cost) and
`verifier/action_log.json` (the session's episode log). An episode is kept only if
- Harbor recorded no exception (timeouts and infra errors are dropped) and the verifier wrote a reward;
- it is a train seed (never dev / test / validation) and its scenario hash matches the seed list;
- the replay matches every logged (tick, status, rule) and the final cost;
- the shift was closed normally: run to its end or ended with `end_shift`, never by the tool budget;
- it is among the `per_seed` cheapest episodes of its seed (and beats the baseline there, if given).
Then targets (OK / GRANTED / INFO, grounded in the windowed prompt) are deduplicated across episodes and
`advance_to_next_event` is capped at `max_advance_share`. The report counts every drop and its reason.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from railway.train.env import is_net_train_seed
from railway.train.teacher import (
    ReplayMismatch,
    TeacherEpisode,
    dedupe,
    rebalance,
    render_episode,
    select_episodes,
    write_samples_jsonl,
)

CLEAN_ENDS = ("end_shift", "shift_end")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def disruption_events(trial_dir: Path) -> int:
    path = trial_dir / "verifier" / "events.jsonl"
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if '"DISRUPTION_' in line)


def load_harbor_trials(job_dirs: list[Path], *, model: str = "claude-opus-5-5", harness: str = "droid-acp") -> list[TeacherEpisode]:
    episodes: list[TeacherEpisode] = []
    for job in job_dirs:
        for res_path in sorted(Path(job).glob("*/result.json")):
            tdir = res_path.parent
            res = _read_json(res_path) or {}
            details = _read_json(tdir / "verifier" / "details.json")
            log = _read_json(tdir / "verifier" / "action_log.json")
            exc = res.get("exception_info")
            if exc:
                status = "timeout" if "timeout" in json.dumps(exc).lower() else "error"
            elif details is None or log is None or not (tdir / "verifier" / "reward.txt").is_file():
                status = "error"
            else:
                status = "ok"
            tier = (details or {}).get("tier") or str((log or {}).get("scenario_id", "?-0")).split("-")[0]
            seed = int((details or {}).get("seed") or str((log or {}).get("scenario_id", "?-0")).rsplit("-", 1)[-1])
            episodes.append(
                TeacherEpisode(
                    tier=tier,
                    seed=seed,
                    rep=0,
                    model=model,
                    harness=harness,
                    status=status,
                    cost=(details or {}).get("cost_total"),
                    scenario_hash=(log or {}).get("scenario_hash"),
                    entries=(log or {}).get("entries") or [],
                    source=str(tdir),
                    extra={
                        "final_cost": (log or {}).get("final_cost"),
                        "disruption_events": disruption_events(tdir),
                        "exception": exc,
                    },
                )
            )
    # number the reps of each seed in a stable order
    seen: Counter = Counter()
    for ep in sorted(episodes, key=lambda e: (e.tier, e.seed, e.source)):
        ep.rep = seen[(ep.tier, ep.seed)]
        seen[(ep.tier, ep.seed)] += 1
    return episodes


def load_baseline(rows_path: Path, model: str) -> dict[tuple[str, int], float]:
    """Mean cost per (tier, seed) of `model` from a results rows.jsonl (e.g. base Qwen on train seeds)."""
    sums: dict[tuple[str, int], list[float]] = {}
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("model") != model or row.get("status") != "ok" or row.get("cost_total") is None:
            continue
        sums.setdefault((row["preset"], int(row["seed"])), []).append(float(row["cost_total"]))
    return {k: sum(v) / len(v) for k, v in sums.items()}


def build_dataset(
    episodes: list[TeacherEpisode],
    out: Path,
    *,
    per_seed: int = 1,
    baseline: dict[tuple[str, int], float] | None = None,
    max_advance_share: float = 0.3,
    rng_seed: int = 0,
) -> dict:
    from railway.net.exploits import build_timetable
    from railway.util import content_hash

    report: dict = {"episodes_in": len(episodes), "dropped": Counter(), "closed_by": Counter()}
    eligible = []
    for ep in episodes:
        if not ep.ok:
            report["dropped"][f"status:{ep.status}"] += 1
        elif not is_net_train_seed(ep.seed):
            report["dropped"]["not_a_train_seed"] += 1
        else:
            eligible.append(ep)
    rendered = []
    for ep in eligible:
        tt = build_timetable(ep.tier, ep.seed)
        if ep.scenario_hash and content_hash(tt.model_dump(mode="json")) != ep.scenario_hash:
            report["dropped"]["scenario_hash_mismatch"] += 1
            continue
        meta = {"tier": ep.tier, "seed": ep.seed, "rep": ep.rep, "source": ep.source}
        try:
            r = render_episode(tt, ep.entries, expected_final_cost=ep.extra.get("final_cost"), meta=meta)
        except ReplayMismatch as exc:
            report["dropped"]["replay_mismatch"] += 1
            report.setdefault("replay_errors", []).append(f"{ep.source}: {exc}")
            continue
        report["closed_by"][r.closed_by] += 1
        if r.closed_by not in CLEAN_ENDS:
            report["dropped"][f"closed_by:{r.closed_by}"] += 1
            continue
        ep.cost = r.final_cost
        rendered.append((ep, r))
    chosen = {id(e) for e in select_episodes([e for e, _ in rendered], per_seed=per_seed, baseline_cost=baseline)}
    report["dropped"]["not_selected_per_seed_or_baseline"] = sum(1 for e, _ in rendered if id(e) not in chosen)

    targets, n_samples, n_ungrounded, n_bad_status, pruned = [], 0, 0, 0, 0
    episodes_out = []
    for ep, r in rendered:
        if id(ep) not in chosen:
            continue
        n_samples += len(r.samples)
        pruned += r.pruned_reads
        n_ungrounded += sum(1 for s in r.samples if not s.meta["grounded"])
        n_bad_status += sum(1 for s in r.samples if s.meta["status"] not in ("OK", "GRANTED", "INFO"))
        targets.extend(r.targets)
        episodes_out.append(
            {"tier": ep.tier, "seed": ep.seed, "rep": ep.rep, "cost": r.final_cost, "closed_by": r.closed_by,
             "actions": r.n_actions, "targets": len(r.targets), "pruned_reads": r.pruned_reads,
             "disruption_events": ep.extra.get("disruption_events"), "source": ep.source}
        )  # fmt: skip
    unique = dedupe(targets)
    final = rebalance(unique, max_share=max_advance_share, seed=rng_seed)
    out.mkdir(parents=True, exist_ok=True)
    write_samples_jsonl(out / "samples.jsonl", final)
    (out / "episodes.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in episodes_out), encoding="utf-8")
    report.update(
        {
            "episodes_used": len(episodes_out),
            "episodes_without_disruption_events": sum(1 for e in episodes_out if not e["disruption_events"]),
            "actions_replayed": n_samples,
            "pruned_reads": pruned,
            "dropped_ungrounded": n_ungrounded,
            "dropped_refused_or_invalid": n_bad_status,
            "targets": len(targets),
            "after_dedupe": len(unique),
            "after_rebalance": len(final),
            "by_tool": Counter(s.tool_calls[0]["name"] for s in final),
            "by_tier": Counter(s.meta.get("tier") for s in final),
            "max_advance_share": max_advance_share,
        }
    )
    report = json.loads(json.dumps(report, default=dict))
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="railway.train.teacher_data")
    p.add_argument("--jobs", nargs="+", required=True, help="Harbor job directories (globs allowed)")
    p.add_argument("--out", required=True)
    p.add_argument("--per-seed", type=int, default=1)
    p.add_argument("--baseline", help="rows.jsonl with baseline episodes on the same train seeds")
    p.add_argument("--baseline-model", default="qwen38-nothink")
    p.add_argument("--max-advance-share", type=float, default=0.3)
    args = p.parse_args(argv)
    jobs = sorted({Path(x) for pattern in args.jobs for x in (glob.glob(pattern) or [pattern])})
    episodes = load_harbor_trials(jobs)
    baseline = load_baseline(Path(args.baseline), args.baseline_model) if args.baseline else None
    report = build_dataset(
        episodes, Path(args.out), per_seed=args.per_seed, baseline=baseline, max_advance_share=args.max_advance_share
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
