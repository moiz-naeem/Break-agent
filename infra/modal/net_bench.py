"""v0.3 network benchmark on Modal: `(model, tier, seed, rep)` episodes in parallel CPU containers.

Each container loads its cached scenario from the `railway-scenarios` volume (written by `net_seeds.py`),
plays one `railway.loop.runner.play_episode` against `railway.net.session.Session`, saves the episode bundle
(`transcript.jsonl` with reasoning, `actions.json`, `events.jsonl`, `result.json`) to the `railway-transcripts`
volume, and returns the results row plus the train-position trajectory, which land locally under
`results/net_bench/<run>/` (`rows.jsonl`, `trajectories/<model>/<scenario_id>-r<rep>.json`).

API keys reach containers only as encrypted Modal secrets: `vllm-key` (named secret) plus
`Secret.from_local_environ` over the provider key names set locally (values are never printed or logged).

The headline is mean -cost, `reward = -(C1 + C2 + C3 + C4 + C5)` (docs/decisions.md #2, #11), with a 95% seed
bootstrap CI. There is no greedy or reference row or column anywhere (#4).

    # dry run (dev seeds, 2 per tier, 1 rep per model), then the full held-out test benchmark straight away (#9)
    modal run infra/modal/net_bench.py::dry_run_then_full --tiers T1,T2,T3 --reps 3

    modal run infra/modal/net_bench.py::run --split dev --tiers T1 --seeds 2 --models qwen38-think
    python infra/modal/net_bench.py report --run <run-name>

Ships `git archive HEAD` (`railway`, `seeds`), so commit (including `seeds/net-*.json`) before `modal run`.
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import modal

SCENARIOS_VOLUME = "railway-scenarios"
SCENARIOS_MOUNT = "/scenarios"
TRANSCRIPTS_VOLUME = "railway-transcripts"
TRANSCRIPTS_MOUNT = "/transcripts"
RESULTS_DIR = "results/net_bench"
MODE = "windowed"

# First runs: Carbon + Qwen only (no Fable credits). Add "fable-high", "nemotron-super" via --models.
DEFAULT_MODELS = ("gemini-carbon-high", "qwen38-think")
# Provider key names forwarded from the local environment (the vLLM key comes from the `vllm-key` secret).
LOCAL_KEY_ENVS = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY", "VERDA_INFERENCE_KEY")

# Used only when Lane F's `railway.loop.providers` is not on this branch yet; mirrors its BENCHMARK_MODELS.
FALLBACK_MODELS: dict[str, dict] = {
    "fable-high": {"provider": "anthropic", "base_url": "https://api.anthropic.com/v1/messages",
                   "model": "claude-fable-5-2", "api_key_env": "ANTHROPIC_API_KEY", "max_tokens": 32768},
    "qwen38-think": {"provider": "vllm", "base_url": "https://hoaianh517--railway-qwen-think.us-east.modal.direct/v1",
                     "model": "qwen38", "api_key_env": "VLLM_API_KEY", "max_tokens": 32768},
    "gemini-carbon-high": {"provider": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta",
                           "model": "gemini-4-carbon", "api_key_env": "GEMINI_API_KEY",
                           "thinking_budget": 4096, "max_tokens": 32768},
    "nemotron-super": {"provider": "verda",
                       "base_url": "https://containers.datacrunch.io/nemotron-3-5-super-vl-b300/v1",
                       "model": "nemotron-3.5-super-vl", "api_key_env": "VERDA_INFERENCE_KEY", "max_tokens": 32768},
}  # fmt: skip

# Results-row fields that belong to the removed greedy/reference normalisation; never written.
DROPPED_FIELDS = ("c_greedy", "c_ref", "c_best", "score_clipped", "score_unclipped", "cost_ratio", "score_normalized")
COMPONENTS = {
    "C1": ("pax_lateness",),
    "C2": ("missed_calls", "unfinished"),
    "C3": ("freight_lateness",),
    "C4": ("crew_timeout",),
    "C5": ("gridlock", "operations"),
}
KPI_KEYS = ("on_time_pct", "passenger_delay_minutes", "freight_delivered_pct")

# Locally this file is <repo>/infra/modal/net_bench.py; in the container it is /root/net_bench.py,
# which has no parents[2], so fall back to the file's own folder (REPO is only used locally).
_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "httpx>=0.28.1", "mcp>=2.3.0", "numpy>=2.4.6", "pydantic>=2.14.3", "pyyaml>=6.0.3",
)  # fmt: skip
if modal.is_local() and __name__ != "__main__":
    source = Path(tempfile.mkdtemp(prefix="railway-net-bench-"))
    archive = subprocess.run(
        ["git", "-C", str(REPO), "archive", "HEAD", "railway", "seeds"], capture_output=True, check=True
    )
    tarfile.open(fileobj=io.BytesIO(archive.stdout)).extractall(source, filter="data")
    image = image.add_local_dir(source, "/src")

app = modal.App("railway-net-bench", image=image)
scenarios = modal.Volume.from_name(SCENARIOS_VOLUME, create_if_missing=True)
transcripts = modal.Volume.from_name(TRANSCRIPTS_VOLUME, create_if_missing=True)


def _load_dotenv() -> None:
    """Fill missing provider key names from a local `.env` into this process only (never printed)."""
    for candidate in (REPO / ".env", REPO.parent / "Afterquery" / ".env"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line.strip())
            if m and m.group(1) in LOCAL_KEY_ENVS:
                os.environ.setdefault(m.group(1), m.group(2).strip().strip('"'))
        return


def _secrets() -> list:
    # Always exactly two secrets: the container rebuilds this list and must see the same number of
    # dependencies as the local definition (otherwise "Function has N dependencies but got N+1 ids").
    # In the container the second entry is a placeholder that Modal hydrates from the local one.
    out = [modal.Secret.from_name("vllm-key")]
    if modal.is_local():
        _load_dotenv()
        out.append(modal.Secret.from_dict({k: os.environ[k] for k in LOCAL_KEY_ENVS if os.environ.get(k)}))
    else:
        out.append(modal.Secret.from_dict({}))
    return out


def _setup() -> None:
    os.chdir("/src")
    if "/src" not in sys.path:
        sys.path.insert(0, "/src")


# -- container side ---------------------------------------------------------------------------------


def model_preset(model_id: str) -> dict:
    try:
        from railway.loop.providers import BENCHMARK_MODELS

        return dict(BENCHMARK_MODELS[model_id])
    except ImportError:
        return dict(FALLBACK_MODELS[model_id])


def make_model_client(model_id: str):
    """Lane F's `make_benchmark_client` when merged; otherwise the OpenAI-compatible client (vLLM / Verda only)."""
    try:
        from railway.loop.providers import make_benchmark_client
    except ImportError:
        make_benchmark_client = None
    if make_benchmark_client is not None:
        return make_benchmark_client(model_id)
    from railway.loop.model_client import ModelClient

    cfg = FALLBACK_MODELS[model_id]
    if cfg["provider"] not in ("vllm", "verda"):
        raise RuntimeError(f"{model_id} needs railway.loop.providers (Lane F), which is not on this branch")
    return ModelClient(cfg["base_url"], cfg["model"], cfg["api_key_env"], max_tokens=cfg["max_tokens"])


def warm_if_needed(base_url: str, timeout_s: float = 900) -> None:
    """Block until a `.modal.direct` vLLM endpoint answers `/health` (cold starts take minutes)."""
    if ".modal.direct" not in (base_url or ""):
        return
    import httpx

    health = re.sub(r"/v1/?$", "", base_url.rstrip("/")) + "/health"
    t0 = time.monotonic()
    while True:
        try:
            if httpx.get(health, timeout=30).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        if time.monotonic() - t0 > timeout_s:
            raise RuntimeError(f"{health} not healthy after {timeout_s:.0f}s")
        time.sleep(10)


def load_timetable(tier: str, seed: int, expected_hash: str | None = None):
    """Cached scenario from the volume; regenerate when missing or its hash differs from the seed list."""
    from railway.net.exploits import build_timetable
    from railway.net.model import Timetable
    from railway.util import content_hash

    path = Path(SCENARIOS_MOUNT) / "net" / tier / f"{seed}.json"
    if path.is_file():
        tt = Timetable.model_validate_json(path.read_text(encoding="utf-8"))
        if expected_hash is None or content_hash(tt.model_dump(mode="json")) == expected_hash:
            return tt
    tt = build_timetable(tier, seed)
    if expected_hash is not None and content_hash(tt.model_dump(mode="json")) != expected_hash:
        raise RuntimeError(f"{tier}-{seed}: generator output no longer matches the seed list; regenerate seeds")
    return tt


def trajectory(session) -> dict:
    """Per-train platform visits and final position/phase, plus the engine event log when one exists."""
    eng = session.engine
    trains = {}
    for tid in sorted(eng.trains):
        mv = eng.trains[tid]
        trains[tid] = {
            "passenger": mv.train.passenger,
            "phase": mv.phase,
            "position": mv.position,
            "cancelled": bool(getattr(mv, "cancelled", False)),
            "visits": [{"platform": v.platform, "arrived": v.arrived, "departed": v.departed} for v in mv.visits],
        }
    log = getattr(eng, "log", None)
    events = [e.model_dump(mode="json") for e in log.events] if log is not None and hasattr(log, "events") else None
    tt = session.timetable
    return {
        "tier": tt.tier,
        "seed": tt.seed,
        "shift_min": tt.shift_min,
        "tick": eng.tick,
        "trains": trains,
        "events": events,
    }


def _bundle_dir(job: dict) -> Path:
    return Path(TRANSCRIPTS_MOUNT, job["run"], job["model"], job["split"], job["scenario_id"], str(job["rep"]))


def _write_missing_bundle_files(bundle: Path, session, row: dict) -> None:
    """Lane F's runner writes the full bundle; before it merges, fill in actions/events/result here."""
    bundle.mkdir(parents=True, exist_ok=True)
    if not (bundle / "actions.json").exists():
        (bundle / "actions.json").write_text(json.dumps(session.log, indent=2, default=str) + "\n", encoding="utf-8")
    log = getattr(session.engine, "log", None)
    if not (bundle / "events.jsonl").exists() and log is not None and hasattr(log, "events"):
        lines = [json.dumps(e.model_dump(mode="json"), sort_keys=True) for e in log.events]
        (bundle / "events.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (bundle / "result.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def components(cost: dict | None) -> dict[str, float] | None:
    if not cost:
        return None
    return {c: round(sum(float(cost.get(f, 0.0)) for f in fields), 4) for c, fields in COMPONENTS.items()}


def play_job(job: dict) -> dict:
    """Run one episode; any infra failure becomes an `error` row instead of an exception."""
    import inspect

    base = {k: job[k] for k in ("run", "model", "tier", "split", "seed", "rep", "scenario_id")}
    t0 = time.monotonic()
    session = None
    try:
        from railway.loop.runner import play_episode
        from railway.net.session import Session

        tt = load_timetable(job["tier"], job["seed"], job.get("scenario_hash"))
        warm_if_needed(model_preset(job["model"]).get("base_url", ""))
        client = make_model_client(job["model"])
        session = Session(tt)
        params = inspect.signature(play_episode).parameters
        bundle = _bundle_dir(job)
        kwargs = {"session": session, "run": job["run"], "model": job["model"], "rep": job["rep"]}
        if "transcript_root" in params:
            kwargs |= {"transcript_root": TRANSCRIPTS_MOUNT, "seed_set": job["split"]}
        else:
            bundle.mkdir(parents=True, exist_ok=True)
            kwargs["transcript_path"] = bundle / "transcript.jsonl"
        try:
            from railway.net.manual import INSTRUCTION

            kwargs["instruction"] = INSTRUCTION
        except ImportError:
            pass
        try:
            rec = play_episode(tt, client, MODE, **kwargs)
        finally:
            if hasattr(client, "close"):
                client.close()
        final = session.close()
        row = {k: v for k, v in rec.to_row().items() if k not in DROPPED_FIELDS}
        cost = final.cost.model_dump()
        row |= {
            **base,
            "cost_total": float(cost["total"]),
            "reward": -float(cost["total"]),
            "cost_by_component": cost,
            "components": components(cost),
            "kpis": dict(final.kpis),
            "counts": dict(final.counts),
        }
        bundle = Path(row.get("transcript_dir") or bundle)
        row["transcript_dir"] = str(bundle)
        traj = trajectory(session)
        _write_missing_bundle_files(bundle, session, row)
        if traj is not None and not (bundle / "trajectory.json").exists():
            (bundle / "trajectory.json").write_text(json.dumps(traj, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - infra errors are reported, never scored
        row = {**base, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        traj = trajectory(session) if session is not None else None
    row.setdefault("wall_s", round(time.monotonic() - t0, 3))
    return {"row": row, "trajectory": traj}


@app.function(
    cpu=1,
    memory=2048,
    timeout=24 * 3600,  # Modal maximum: uncapped Qwen thinking can run a T3 episode for many hours
    max_containers=64,
    secrets=_secrets(),
    volumes={SCENARIOS_MOUNT: scenarios, TRANSCRIPTS_MOUNT: transcripts},
)
def run_episode(job: dict) -> dict:
    _setup()
    os.environ.setdefault("RAILWAY_TRANSCRIPTS_DIR", TRANSCRIPTS_MOUNT)
    out = play_job(job)
    transcripts.commit()
    return out


@app.function(
    cpu=1,
    memory=1024,
    timeout=24 * 3600,
    volumes={TRANSCRIPTS_MOUNT: transcripts},
)
def orchestrate_cloud(run_name: str, jobs: list[dict]) -> dict:
    """Server-side orchestrator so `modal run --detach ...::launch_cloud` survives if the laptop disconnects."""
    _setup()
    for url in sorted({model_preset(j["model"]).get("base_url", "") for j in jobs}):
        warm_if_needed(url)
    print(f"[{run_name}] cloud orchestrator dispatching {len(jobs)} episodes", flush=True)
    ok = errors = 0
    for res in run_episode.map(jobs, order_outputs=False):
        row = res["row"]
        if row.get("status") == "ok":
            ok += 1
            tail = f"reward={row.get('reward')}"
        else:
            errors += 1
            tail = f"ERROR {row.get('error')}"
            err_dir = _bundle_dir(
                {
                    "run": run_name,
                    "model": row["model"],
                    "split": row["split"],
                    "scenario_id": row["scenario_id"],
                    "rep": row["rep"],
                }
            )
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "error.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            transcripts.commit()
        msg = f"[{run_name}] ({ok + errors}/{len(jobs)}) {row['model']} {row['scenario_id']} r{row['rep']}: {tail}"
        print(msg, flush=True)
    return {"run": run_name, "total": len(jobs), "ok": ok, "errors": errors}


@app.local_entrypoint()
def launch_cloud(
    models: str = ",".join(DEFAULT_MODELS),
    tiers: str = "T1",
    split: str = "test",
    seeds: int = 0,
    reps: int = 1,
    run_name: str = "",
):
    """Spawn `orchestrate_cloud` inside Modal (`--detach`) so `.map()` runs 100% in the cloud."""
    name = run_name or _default_run(f"net-{split}")
    collect(name)  # skip any episodes already saved on the volume
    jobs = build_jobs(name, _split_list(models), _split_list(tiers), split, seeds, reps)
    if not jobs:
        print(f"[{name}] all episodes already complete")
        write_report(name)
        return
    call = orchestrate_cloud.spawn(name, jobs)
    print(f"[{name}] spawned cloud orchestrator {call.object_id} for {len(jobs)} episodes", flush=True)


# -- local side ----------------------------------------------------------------------------------------


def _seed_entries(tier: str, split: str, limit: int) -> list[dict]:
    path = REPO / "seeds" / f"net-{tier}-{split}.json"
    if not path.is_file():
        raise SystemExit(f"missing {path.relative_to(REPO)}: run infra/modal/net_seeds.py first")
    entries = json.loads(path.read_text(encoding="utf-8"))["accepted"]
    return entries[:limit] if limit > 0 else entries


def _done(rows_path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if rows_path.is_file():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("status") == "ok":
                    done.add((r["model"], r["scenario_id"], int(r["rep"])))
    return done


def _split_list(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def build_jobs(run: str, models: list[str], tiers: list[str], split: str, seeds: int, reps: int) -> list[dict]:
    done = _done(REPO / RESULTS_DIR / run / "rows.jsonl")
    jobs = []
    for tier in tiers:
        for entry in _seed_entries(tier, split, seeds):
            for model in models:
                for rep in range(reps):
                    if (model, entry["scenario_id"], rep) in done:
                        continue
                    jobs.append(
                        {
                            "run": run,
                            "model": model,
                            "tier": tier,
                            "split": split,
                            "seed": entry["seed"],
                            "scenario_id": entry["scenario_id"],
                            "scenario_hash": entry.get("scenario_hash"),
                            "rep": rep,
                        }
                    )
    return jobs


def dispatch(run: str, jobs: list[dict]) -> int:
    """Fan the jobs out; append rows/trajectories locally as they finish. Returns the number of error rows."""
    out_root = REPO / RESULTS_DIR / run
    out_root.mkdir(parents=True, exist_ok=True)
    for url in sorted({FALLBACK_MODELS.get(j["model"], {}).get("base_url", "") for j in jobs}):
        warm_if_needed(url)
    print(f"[{run}] dispatching {len(jobs)} episodes", flush=True)
    errors = 0
    for res in run_episode.map(jobs, order_outputs=False):
        row = res["row"]
        errors += row.get("status") != "ok"
        with (out_root / "rows.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        if res.get("trajectory"):
            tp = out_root / "trajectories" / row["model"] / f"{row['scenario_id']}-r{row['rep']}.json"
            tp.parent.mkdir(parents=True, exist_ok=True)
            tp.write_text(json.dumps(res["trajectory"], sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        tail = f"reward={row.get('reward')}" if row.get("status") == "ok" else f"ERROR {row.get('error')}"
        print(f"[{run}] {row['model']} {row['scenario_id']} r{row['rep']}: {tail}", flush=True)
    return errors


def _default_run(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


@app.local_entrypoint()
def run(
    models: str = ",".join(DEFAULT_MODELS),
    tiers: str = "T1",
    split: str = "test",
    seeds: int = 0,
    reps: int = 1,
    run_name: str = "",
):
    """Run `models x tiers x seeds x reps` on one split (`seeds=0`: every accepted seed), then report."""
    name = run_name or _default_run(f"net-{split}")
    jobs = build_jobs(name, _split_list(models), _split_list(tiers), split, seeds, reps)
    errors = dispatch(name, jobs)
    write_report(name)
    if errors:
        print(f"[{name}] {errors} infra error rows; rerun with --run-name {name} to retry only those")


@app.local_entrypoint()
def dry_run_then_full(
    models: str = ",".join(DEFAULT_MODELS),
    tiers: str = "T1",
    dry_seeds: int = 2,
    reps: int = 3,
    test_seeds: int = 0,
    run_name: str = "",
):
    """Dry run on `dev` seeds (1 rep per model); with 0 infra errors, launch the full `test` benchmark at once."""
    stamp = run_name or time.strftime("%Y%m%d-%H%M%S")
    model_list, tier_list = _split_list(models), _split_list(tiers)
    dry = f"net-dry-{stamp}"
    errors = dispatch(dry, build_jobs(dry, model_list, tier_list, "dev", dry_seeds, 1))
    write_report(dry)
    if errors:
        raise SystemExit(f"[{dry}] dry run had {errors} infra error rows; fix them before the full run")
    full = f"net-test-{stamp}"
    print(f"[{dry}] clean; launching the full held-out benchmark as {full}", flush=True)
    errors = dispatch(full, build_jobs(full, model_list, tier_list, "test", test_seeds, reps))
    write_report(full)
    if errors:
        print(f"[{full}] {errors} infra error rows; rerun `run --split test --run-name {full}` to retry them")


@app.local_entrypoint()
def report(run_name: str):
    write_report(run_name)


# -- report ------------------------------------------------------------------------------------------


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def bootstrap_ci(values: list[float], *, n: int = 2000, seed: int = 0) -> tuple[float | None, float | None]:
    """95% percentile bootstrap CI of the mean, resampling per-seed values."""
    if len(values) < 2:
        return (values[0], values[0]) if values else (None, None)
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choices(values, k=k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def summarize_group(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    per_seed: dict[tuple[str, int], list[float]] = {}
    for r in ok:
        per_seed.setdefault((r["tier"], int(r["seed"])), []).append(float(r["reward"]))
    seed_means = [sum(v) / len(v) for _, v in sorted(per_seed.items())]
    lo, hi = bootstrap_ci(seed_means)
    tool_calls = sum(int(r.get("tool_calls") or 0) for r in ok)

    def avg(get) -> float | None:
        vals = [v for r in ok if (v := get(r)) is not None]
        return _mean([float(v) for v in vals])

    return {
        "episodes": len(rows),
        "ok": len(ok),
        "infra_errors": len(rows) - len(ok),
        "seeds": len(per_seed),
        "mean_reward": _mean(seed_means),
        "ci95": [lo, hi],
        "components": {c: avg(lambda r, c=c: (r.get("components") or {}).get(c)) for c in COMPONENTS},
        "kpis": {
            **{k: avg(lambda r, k=k: (r.get("kpis") or {}).get(k)) for k in KPI_KEYS},
            "clean_shift_rate": avg(lambda r: float(bool((r.get("kpis") or {}).get("clean_shift")))),
        },
        "invalid_rate": sum(int(r.get("invalid_calls") or 0) for r in ok) / tool_calls if tool_calls else None,
        "refusal_rate": sum(int(r.get("refusals") or 0) for r in ok) / tool_calls if tool_calls else None,
        "tool_calls_mean": avg(lambda r: r.get("tool_calls")),
        "tokens_mean": {k: avg(lambda r, k=k: r.get(f"tokens_{k}")) for k in ("in", "out", "reasoning", "cached")},
        "usd_total": sum(float(r.get("cost_usd") or 0.0) for r in ok),
        "usd_mean": avg(lambda r: r.get("cost_usd")),
        "wall_s_mean": avg(lambda r: r.get("wall_s")),
    }


def summarize(rows: list[dict]) -> dict:
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["tier"]), []).append(r)
        groups.setdefault((r["model"], "ALL"), []).append(r)
    return {f"{m}|{t}": summarize_group(g) for (m, t), g in sorted(groups.items())}


def _f(x, nd: int = 1) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def render_markdown(run_name: str, summary: dict) -> str:
    head = (
        "| model | tier | seeds | ok/eps | mean -cost [95% CI] | C1 | C2 | C3 | C4 | C5 | on-time % | pax delay min "
        "| freight % | clean % | invalid % | refusal % | tok in | tok out | tok reason | tok cached | USD | wall s |"
    )
    lines = [
        f"# Network benchmark `{run_name}`",
        "",
        "reward = -(C1 + C2 + C3 + C4 + C5); CI = 95% seed bootstrap.",
        "",
        head,
        "|" + "---|" * 22,
    ]
    for key, s in summary.items():
        model, tier = key.split("|")
        k, c, t = s["kpis"], s["components"], s["tokens_mean"]
        pct = lambda x: None if x is None else 100 * x  # noqa: E731
        lines.append(
            f"| {model} | {tier} | {s['seeds']} | {s['ok']}/{s['episodes']} | {_f(s['mean_reward'])} "
            f"[{_f(s['ci95'][0])}, {_f(s['ci95'][1])}] | "
            + " | ".join(_f(c[x]) for x in COMPONENTS)
            + f" | {_f(k['on_time_pct'])} | {_f(k['passenger_delay_minutes'])} | {_f(k['freight_delivered_pct'])} "
            f"| {_f(pct(k['clean_shift_rate']))} | {_f(pct(s['invalid_rate']), 2)} | {_f(pct(s['refusal_rate']), 2)} "
            f"| {_f(t['in'], 0)} | {_f(t['out'], 0)} | {_f(t['reasoning'], 0)} | {_f(t['cached'], 0)} "
            f"| {_f(s['usd_total'], 2)} | {_f(s['wall_s_mean'], 0)} |"
        )
    return "\n".join(lines) + "\n"


def write_report(run_name: str) -> dict:
    out_root = REPO / RESULTS_DIR / run_name
    rows_path = out_root / "rows.jsonl"
    rows = (
        [json.loads(x) for x in rows_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if rows_path.is_file()
        else []
    )
    # A retried episode keeps only its latest row.
    latest: dict[tuple, dict] = {}
    for r in rows:
        latest[(r["model"], r["scenario_id"], int(r["rep"]))] = r
    summary = summarize(list(latest.values()))
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    md = render_markdown(run_name, summary)
    (out_root / "summary.md").write_text(md, encoding="utf-8", newline="\n")
    print(md)
    return summary


def collect(run_name: str) -> int:
    """Rebuild `rows.jsonl` from the `result.json` bundles on the transcripts volume.

    For runs whose local collector died (e.g. a detached `modal run` whose terminal went away): every finished
    episode wrote `<run>/<model>/<split>/<scenario_id>/<rep>/result.json` on the volume. Infra-error rows write
    no bundle, so they are missing here; rerunning `run --run-name <run>` retries exactly the missing episodes.
    """
    vol = modal.Volume.from_name(TRANSCRIPTS_VOLUME)
    out_root = REPO / RESULTS_DIR / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    rows: dict[tuple, dict] = {}
    try:
        entries = list(vol.iterdir(run_name, recursive=True))
    except modal.exception.NotFoundError:  # no episode of this run has finished yet
        entries = []
    for entry in entries:
        if not entry.path.endswith("/result.json"):
            continue
        row = json.loads(b"".join(vol.read_file(entry.path)).decode("utf-8"))
        rows[(row["model"], row["scenario_id"], int(row["rep"]))] = row
    lines = [json.dumps(r, sort_keys=True, ensure_ascii=False) for _, r in sorted(rows.items())]
    body = "".join(line + "\n" for line in lines)
    (out_root / "rows.jsonl").write_text(body, encoding="utf-8", newline="\n")
    print(f"[{run_name}] collected {len(rows)} finished episodes from {TRANSCRIPTS_VOLUME}")
    return len(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize a results/net_bench/<run> directory.")
    parser.add_argument("command", choices=["report", "collect"])
    parser.add_argument("--run", required=True)
    cli = parser.parse_args()
    if cli.command == "collect":
        collect(cli.run)
    write_report(cli.run)
