"""Validation and held-out evaluation for Qwen 3.8 27B base & LoRA checkpoints on the v0.3 network.

Two modes share one row schema (100% compatible with `infra/modal/net_bench.py` `rows.jsonl`):
1. **Modal evaluation (`eval_vllm_run`, wrapped by `infra/modal/train.py::eval_models`):**
   Boots one isolated vLLM server on a dedicated GPU (`qwen38-nothink` by default, or `qwen38` with thinking),
   evaluates base Qwen and/or any list of LoRA adapters (`/adapters/<run>/sft`, `/adapters/<run>/dagger1`, ...)
   on `val`, `dev`, `test`, or `train` seeds across `T1,T2,T3` via `railway.loop.runner.play_episode`,
   and saves resumable `rows.jsonl`, full episode transcripts, `summary.json`, `paired.json`, and `report.md`.
   Never touches the shared `railway-qwen` or `railway-net-bench` deployments.
2. **Offline statistical report & paired comparison (`python -m railway.train.evaluate`):**
   Merges one or more `rows.jsonl` files (including `net_bench` runs), aggregates per seed first so every seed
   carries equal weight regardless of rep count, and computes paired `(cand - base)` seed-bootstrap 95% CIs on
   `reward = -cost`, raw `cost_total`, `C1..C5` components, operational KPIs/failures, and inference latency/cost.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from railway.train.env import is_net_train_seed, is_val_seed, net_seed_list

MODE = "windowed"
N_BOOT = 10_000
INCOMPLETE_ERROR_RATE = 0.10

DROPPED_FIELDS = (
    "c_greedy",
    "c_ref",
    "c_best",
    "score_clipped",
    "score_unclipped",
    "cost_ratio",
    "score_normalized",
)
COMPONENTS: dict[str, tuple[str, ...]] = {
    "C1": ("pax_lateness",),
    "C2": ("missed_calls", "unfinished"),
    "C3": ("freight_lateness",),
    "C4": ("crew_timeout",),
    "C5": ("gridlock", "operations"),
}
KPI_KEYS = ("on_time_pct", "passenger_delay_minutes", "freight_delivered_pct")
SPLITS = ("val", "dev", "test", "train")


@dataclass(frozen=True)
class EvalCandidate:
    """One model or LoRA checkpoint to evaluate."""

    label: str
    adapter_path: str | None = None


def parse_candidates(specs: Iterable[str], *, thinking: bool = False) -> list[EvalCandidate]:
    """Parse candidate specs: `'base'` / `'qwen38-nothink'`, `'sft=/adapters/r/sft'`, or `'/adapters/r/sft'`."""
    base_default = "qwen38-think" if thinking else "qwen38-nothink"
    out: list[EvalCandidate] = []
    seen: set[str] = set()
    for raw in specs:
        item = raw.strip()
        if not item:
            continue
        if "=" in item:
            label, path = (x.strip() for x in item.split("=", 1))
            cand = EvalCandidate(label=label, adapter_path=path or None)
        elif item in ("base", "qwen38", "qwen38-nothink", "qwen38-think"):
            label = base_default if item == "base" else item
            cand = EvalCandidate(label=label, adapter_path=None)
        else:
            p = Path(item.rstrip("/"))
            label = f"{p.parent.name}-{p.name}" if p.parent.name and p.parent.name != "adapters" else p.name
            cand = EvalCandidate(label=label, adapter_path=item)
        if cand.label in seen:
            raise ValueError(f"duplicate candidate label {cand.label!r}; give explicit 'label=/adapters/...' names")
        seen.add(cand.label)
        out.append(cand)
    if not out:
        raise ValueError("at least one candidate is required")
    return out


# -- Seed selection & scenario loading ---------------------------------------------------------------


def select_eval_seeds(tier: str, split: str, *, limit: int = 0, repo_root: Path | None = None) -> list[dict[str, Any]]:
    """Return seed entries `[{tier, split, seed, scenario_id, scenario_hash}, ...]` with strict split separation."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    file_split = "train" if split in ("val", "train") else split
    if repo_root is not None:
        path = Path(repo_root) / "seeds" / f"net-{tier}-{file_split}.json"
        entries = json.loads(path.read_text(encoding="utf-8"))["accepted"]
    else:
        entries = net_seed_list(tier, file_split)

    filtered: list[dict[str, Any]] = []
    for e in entries:
        s = int(e["seed"])
        if split == "val" and not is_val_seed(s):
            continue
        if split == "train" and not is_net_train_seed(s):
            continue
        filtered.append(
            {
                "tier": tier,
                "split": split,
                "seed": s,
                "scenario_id": e.get("scenario_id") or f"{tier}-{s}",
                "scenario_hash": e.get("scenario_hash"),
            }
        )
    return filtered[:limit] if limit > 0 else filtered


def load_net_timetable(
    tier: str,
    seed: int,
    expected_hash: str | None = None,
    *,
    scenarios_dir: str | Path | None = "/scenarios",
):
    """Load cached timetable from `scenarios_dir` when present, else rebuild via `build_timetable`; check hash."""
    from railway.net.exploits import build_timetable
    from railway.net.model import Timetable
    from railway.util import content_hash

    if scenarios_dir is not None:
        cached = Path(scenarios_dir) / "net" / tier / f"{seed}.json"
        if cached.is_file():
            tt = Timetable.model_validate_json(cached.read_text(encoding="utf-8"))
            if expected_hash is None or content_hash(tt.model_dump(mode="json")) == expected_hash:
                return tt
    tt = build_timetable(tier, seed)
    if expected_hash is not None and content_hash(tt.model_dump(mode="json")) != expected_hash:
        raise RuntimeError(f"{tier}-{seed}: scenario hash mismatch against seed list")
    return tt


# -- Episode execution (loop harness, net_bench-compatible rows) -------------------------------------


def compute_components(cost: dict | None) -> dict[str, float] | None:
    if not cost:
        return None
    return {c: round(sum(float(cost.get(f, 0.0)) for f in fields), 4) for c, fields in COMPONENTS.items()}


def play_eval_job(
    job: dict[str, Any],
    client_factory: Callable[[], Any],
    *,
    mode: str = MODE,
    max_turns: int | None = None,
    scenarios_dir: str | Path | None = "/scenarios",
    transcript_root: str | Path | None = None,
) -> dict[str, Any]:
    """Play one `(model, tier, seed, rep)` episode through `railway.loop.runner.play_episode`; never raises."""
    base = {k: job[k] for k in ("run", "model", "tier", "split", "seed", "rep", "scenario_id")}
    if job.get("adapter_path"):
        base["adapter_path"] = job["adapter_path"]
    t0 = time.monotonic()
    session = None
    try:
        from railway.loop.runner import play_episode
        from railway.net.manual import INSTRUCTION
        from railway.net.session import Session

        tt = load_net_timetable(job["tier"], int(job["seed"]), job.get("scenario_hash"), scenarios_dir=scenarios_dir)
        session = Session(tt)
        client = client_factory()
        kwargs: dict[str, Any] = {
            "session": session,
            "instruction": INSTRUCTION,
            "run": job["run"],
            "model": job["model"],
            "rep": int(job["rep"]),
            "seed_set": job["split"],
        }
        if transcript_root is not None:
            kwargs["transcript_root"] = transcript_root
        if max_turns is not None:
            kwargs["max_turns"] = max_turns
        try:
            rec = play_episode(tt, client, mode, **kwargs)
        finally:
            if hasattr(client, "close"):
                client.close()
        final = session.close()
        row = {k: v for k, v in rec.to_row().items() if k not in DROPPED_FIELDS}
        cost = final.cost.model_dump()
        row |= {
            **base,
            "preset": job["tier"],
            "cost_total": float(cost["total"]),
            "reward": -float(cost["total"]),
            "cost_by_component": cost,
            "components": compute_components(cost),
            "kpis": dict(final.kpis),
            "counts": dict(final.counts),
        }
    except Exception as exc:  # noqa: BLE001 - infra errors become `status="error"` rows, never silent drops
        if session is not None and not session.closed:
            try:
                session.close()
            except Exception:  # noqa: BLE001
                pass
        row = {
            **base,
            "preset": job["tier"],
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    row.setdefault("wall_s", round(time.monotonic() - t0, 3))
    return row


def read_rows_jsonl(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in paths:
        p = Path(raw)
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(normalize_row(json.loads(line)))
    return rows


def _iso_seconds(started: str | None, finished: str | None) -> float | None:
    if not started or not finished:
        return None
    try:
        t0 = datetime.fromisoformat(started.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(finished.replace("Z", "+00:00"))
        return round(max(0.0, (t1 - t0).total_seconds()), 3)
    except ValueError:
        return None


def harbor_jobs_to_rows(
    job_patterns: Iterable[str | Path],
    *,
    model_label: str = "opus55-droid",
    split: str = "test",
) -> list[dict[str, Any]]:
    """Convert Harbor trial folders (`<job>/<trial>/{result.json,verifier/details.json}`) into standard eval rows."""
    job_dirs = sorted(
        {Path(x) for pat in job_patterns for x in (glob.glob(str(pat)) or [str(pat)]) if Path(x).is_dir()}
    )
    raw_trials: list[tuple[str, int, str, dict[str, Any]]] = []
    for jdir in job_dirs:
        for res_path in sorted(jdir.glob("*/result.json")):
            tdir = res_path.parent
            res = json.loads(res_path.read_text(encoding="utf-8"))
            details_path = tdir / "verifier" / "details.json"
            reward_path = tdir / "verifier" / "reward.txt"
            details = json.loads(details_path.read_text(encoding="utf-8")) if details_path.is_file() else {}
            exc = res.get("exception_info")
            ok = (exc is None) and bool(details) and reward_path.is_file()
            task_name = str(details.get("scenario_id") or res.get("task_name") or tdir.name.split("__", 1)[0])
            tier = str(details.get("tier") or task_name.split("-", 1)[0])
            seed = int(details.get("seed") if details.get("seed") is not None else task_name.rsplit("-", 1)[-1])
            wall_s = _iso_seconds(res.get("started_at"), res.get("finished_at"))
            counts = dict(details.get("counts") or {})
            row: dict[str, Any] = {
                "run": jdir.name,
                "harness": "harbor-droid",
                "model": model_label,
                "provider": "factory-droid",
                "tier": tier,
                "preset": tier,
                "split": split,
                "seed": seed,
                "scenario_id": f"{tier}-{seed}",
                "scenario_hash": details.get("scenario_hash"),
                "rules_version": details.get("rules_version") or "0.3-net",
                "status": "ok" if ok else "error",
                "error": json.dumps(exc)[:300] if exc else (None if ok else "missing verifier output"),
                "wall_s": wall_s,
                "tool_calls": details.get("tool_calls"),
                "gridlocks": counts.get("gridlocks"),
                "crew_timeouts": counts.get("crew_timeouts"),
                "unfinished": counts.get("unfinished"),
                "refusals": counts.get("refusals"),
                "invalid_calls": counts.get("invalid_calls"),
                "decision_points": counts.get("decision_points"),
                "format_errors": 0,
            }
            if ok:
                cost = dict(details.get("cost_by_component") or {})
                cost_total = float(details["cost_total"])
                cost.setdefault("total", cost_total)
                row |= {
                    "cost_total": cost_total,
                    "reward": float(reward_path.read_text(encoding="utf-8").strip()),
                    "cost_by_component": cost,
                    "components": compute_components(cost),
                    "kpis": dict(details.get("kpis") or {}),
                    "counts": counts,
                }
            raw_trials.append((tier, seed, str(tdir), row))

    rep_counter: Counter[tuple[str, int]] = Counter()
    rows: list[dict[str, Any]] = []
    for tier, seed, _src, row in sorted(raw_trials, key=lambda x: (x[0], x[1], x[2])):
        row["rep"] = rep_counter[(tier, seed)]
        rep_counter[(tier, seed)] += 1
        rows.append(normalize_row(row))
    return rows


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Ensure `tier`, `seed`, `scenario_id`, `rep`, `reward`, and `components` exist on any loop/net_bench row."""
    r = dict(row)
    tier = r.get("tier") or r.get("preset")
    if tier is None and r.get("scenario_id"):
        tier = str(r["scenario_id"]).split("-", 1)[0]
    r["tier"] = tier
    r.setdefault("preset", tier)
    if r.get("seed") is not None:
        r["seed"] = int(r["seed"])
    r["rep"] = int(r.get("rep") or 0)
    if not r.get("scenario_id") and tier is not None and r.get("seed") is not None:
        r["scenario_id"] = f"{tier}-{r['seed']}"
    if r.get("status") == "ok" and r.get("cost_total") is not None:
        r["cost_total"] = float(r["cost_total"])
        if r.get("reward") is None:
            r["reward"] = -r["cost_total"]
        else:
            r["reward"] = float(r["reward"])
        if not r.get("components") and isinstance(r.get("cost_by_component"), dict):
            r["components"] = compute_components(r["cost_by_component"])
    return r


def dedupe_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per `(model, scenario_id, rep)`: an `ok` row supersedes `error` rows; latest wins on ties."""
    best: dict[tuple[str, str, int], dict[str, Any]] = {}
    for raw in rows:
        r = normalize_row(raw)
        key = (str(r.get("model") or ""), str(r.get("scenario_id") or ""), int(r.get("rep") or 0))
        prev = best.get(key)
        if prev is None or r.get("status") == "ok" or prev.get("status") != "ok":
            best[key] = r
    return [best[k] for k in sorted(best)]


def done_keys(rows_path: Path) -> set[tuple[str, str, int]]:
    if not rows_path.is_file():
        return set()
    return {
        (str(r["model"]), str(r["scenario_id"]), int(r["rep"]))
        for r in dedupe_rows(read_rows_jsonl([rows_path]))
        if r.get("status") == "ok"
    }


# -- Modal dedicated-GPU vLLM runner -----------------------------------------------------------------


def eval_vllm_run(
    *,
    run: str,
    candidates: list[str],
    tiers: list[str],
    split: str = "val",
    seeds_limit: int = 0,
    reps: int = 1,
    thinking: bool = False,
    temperature: float | None = None,
    max_tokens: int = 32768,
    max_model_len: int = 65536,
    concurrency: int = 16,
    max_turns: int | None = None,
    base_model: str | None = None,
    extra_rows_blobs: dict[str, bytes] | None = None,
    out_dir: str | Path = "/data/eval",
    scenarios_dir: str | Path | None = "/scenarios",
    transcript_root: str | Path | None = "/transcripts",
    commit: Callable[[], None] = lambda: None,
    log: Callable[..., None] = print,
) -> dict[str, Any]:
    """Boot one local vLLM server on GPU 0 and evaluate all `candidates` on identical `(tier, seed, rep)` jobs."""
    from railway.loop.model_client import ModelClient
    from railway.train.serving import LocalVLLM

    parsed_cands = parse_candidates(candidates, thinking=thinking)
    for c in parsed_cands:
        if c.adapter_path and not Path(c.adapter_path).is_dir():
            raise FileNotFoundError(f"adapter directory not found: {c.adapter_path}")

    seed_entries: list[dict[str, Any]] = []
    for tier in tiers:
        seed_entries.extend(select_eval_seeds(tier, split, limit=seeds_limit))
    if not seed_entries:
        raise ValueError(f"no seeds selected for tiers={tiers}, split={split!r}")

    run_dir = Path(out_dir) / run
    run_dir.mkdir(parents=True, exist_ok=True)
    rows_path = run_dir / "rows.jsonl"
    already_done = done_keys(rows_path)

    served_base = "qwen38" if thinking else "qwen38-nothink"
    extra_args: tuple[str, ...] = ()
    if not thinking:
        extra_args = ("--default-chat-template-kwargs", '{"enable_thinking": false}')

    gpus = os.environ.get("CUDA_VISIBLE_DEVICES") or "0"
    server = LocalVLLM(
        served_name=served_base,
        gpus=gpus,
        max_model_len=max_model_len,
        extra_args=extra_args,
    )
    write_lock = threading.Lock()
    t_boot = time.monotonic()
    server.start()
    try:
        ready_s = round(server.wait(), 1)
        log(f"[eval {run}] vLLM ({served_base}) healthy in {ready_s}s; seeds={len(seed_entries)} x reps={reps}")
        for idx, cand in enumerate(parsed_cands):
            if cand.adapter_path is None:
                served_model = served_base
            else:
                served_model = f"{run}-lora{idx}-{cand.label}"
                t_swap = time.monotonic()
                server.swap_lora(served_model, cand.adapter_path)
                log(f"[eval {run}] loaded LoRA {cand.label} ({cand.adapter_path}) in {time.monotonic() - t_swap:.1f}s")

            jobs = [
                {
                    "run": run,
                    "model": cand.label,
                    "adapter_path": cand.adapter_path,
                    "tier": e["tier"],
                    "split": e["split"],
                    "seed": e["seed"],
                    "scenario_id": e["scenario_id"],
                    "scenario_hash": e["scenario_hash"],
                    "rep": rep,
                }
                for e in seed_entries
                for rep in range(reps)
                if (cand.label, e["scenario_id"], rep) not in already_done
            ]
            if not jobs:
                log(f"[eval {run}] {cand.label}: all {len(seed_entries) * reps} episodes already done")
                continue

            log(f"[eval {run}] {cand.label}: running {len(jobs)} episodes (concurrency={concurrency})")
            extra_body: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": thinking}}
            if thinking:
                extra_body["thinking_token_budget"] = 4096

            def make_client(sm: str = served_model, eb: dict = extra_body) -> ModelClient:
                return ModelClient(
                    server.base_url,
                    sm,
                    api_key_env=None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_s=900,
                    provider="vllm",
                    extra_body=eb,
                )

            def run_one(j: dict[str, Any]) -> dict[str, Any]:
                row = play_eval_job(
                    j,
                    make_client,
                    max_turns=max_turns,
                    scenarios_dir=scenarios_dir,
                    transcript_root=transcript_root,
                )
                with write_lock:
                    with rows_path.open("a", encoding="utf-8", newline="\n") as fh:
                        fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                    commit()
                status_str = f"cost={row.get('cost_total')}" if row.get("status") == "ok" else f"ERR {row.get('error')}"
                log(f"[eval {run}] {row['model']} {row['scenario_id']} r{row['rep']}: {status_str}")
                return row

            with ThreadPoolExecutor(max_workers=min(concurrency, len(jobs))) as pool:
                list(pool.map(run_one, jobs))
            commit()
    finally:
        server.stop()

    extra_paths: list[Path] = []
    for idx, (name, blob) in enumerate((extra_rows_blobs or {}).items()):
        p = run_dir / f"extra_{idx}_{Path(name).name}"
        p.write_bytes(blob)
        extra_paths.append(p)

    resolved_base = base_model or parsed_cands[0].label
    bundle = build_eval_report(
        run_name=run,
        row_paths=[rows_path, *extra_paths],
        base_model=resolved_base,
        out_dir=run_dir,
    )
    bundle["vllm_boot_s"] = round(time.monotonic() - t_boot, 1)
    bundle["rows_jsonl"] = rows_path.read_text(encoding="utf-8") if rows_path.is_file() else ""
    commit()
    return bundle


def shard_entries(entries: list[dict[str, Any]], n_shards: int) -> list[list[dict[str, Any]]]:
    """Split `entries` evenly into up to `n_shards` non-empty round-robin shards."""
    k = max(1, min(n_shards, len(entries)))
    return [entries[i::k] for i in range(k) if entries[i::k]]


def eval_shard_vllm(
    *,
    run: str,
    shard_id: str,
    model_label: str,
    adapter_path: str | None,
    seed_entries: list[dict[str, Any]],
    reps: int = 1,
    thinking: bool = True,
    temperature: float | None = None,
    max_tokens: int = 24576,
    max_model_len: int = 65536,
    concurrency: int = 8,
    out_dir: str | Path = "/data/eval",
    scenarios_dir: str | Path | None = "/scenarios",
    transcript_root: str | Path | None = "/transcripts",
    commit: Callable[[], None] = lambda: None,
    reload_volumes: Callable[[], None] = lambda: None,
    log: Callable[..., None] = print,
) -> list[dict[str, Any]]:
    """Run one GPU shard of `(model_label, seed_entries x reps)`, committing each finished row immediately."""
    from railway.loop.model_client import ModelClient
    from railway.train.serving import LocalVLLM

    reload_volumes()
    if adapter_path and not Path(adapter_path).is_dir():
        raise FileNotFoundError(f"adapter directory not found on volume: {adapter_path}")

    shards_dir = Path(out_dir) / run / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shards_dir / f"{shard_id}.jsonl"
    already_done = done_keys(shard_path)

    jobs = [
        {
            "run": run,
            "model": model_label,
            "adapter_path": adapter_path,
            "tier": e["tier"],
            "split": e["split"],
            "seed": e["seed"],
            "scenario_id": e["scenario_id"],
            "scenario_hash": e["scenario_hash"],
            "rep": rep,
        }
        for e in seed_entries
        for rep in range(reps)
        if (model_label, e["scenario_id"], rep) not in already_done
    ]
    if not jobs:
        log(f"[shard {shard_id}] all {len(seed_entries) * reps} episodes already done")
        return dedupe_rows(read_rows_jsonl([shard_path]))

    served_base = "qwen38" if thinking else "qwen38-nothink"
    extra_args: tuple[str, ...] = () if thinking else ("--default-chat-template-kwargs", '{"enable_thinking": false}')
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES") or "0"
    server = LocalVLLM(served_name=served_base, gpus=gpus, max_model_len=max_model_len, extra_args=extra_args)
    write_lock = threading.Lock()
    server.start()
    try:
        ready_s = round(server.wait(), 1)
        served_model = served_base
        if adapter_path:
            served_model = f"{shard_id}-lora"
            server.swap_lora(served_model, adapter_path)
        log(f"[shard {shard_id}] vLLM ready in {ready_s}s; running {len(jobs)} jobs ({model_label})")
        extra_body: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": thinking}}
        if thinking:
            extra_body["thinking_token_budget"] = 4096

        def make_client(sm: str = served_model, eb: dict = extra_body) -> ModelClient:
            return ModelClient(
                server.base_url,
                sm,
                api_key_env=None,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=900,
                provider="vllm",
                extra_body=eb,
            )

        def run_one(j: dict[str, Any]) -> dict[str, Any]:
            row = play_eval_job(
                j,
                make_client,
                scenarios_dir=scenarios_dir,
                transcript_root=transcript_root,
            )
            with write_lock:
                with shard_path.open("a", encoding="utf-8", newline="\n") as fh:
                    fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                commit()
            tag = f"cost={row.get('cost_total')}" if row.get("status") == "ok" else f"ERR {row.get('error')}"
            log(f"[shard {shard_id}] {row['model']} {row['scenario_id']} r{row['rep']}: {tag}")
            return row

        with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(jobs)))) as pool:
            list(pool.map(run_one, jobs))
    finally:
        server.stop()

    return dedupe_rows(read_rows_jsonl([shard_path]))


# -- Aggregation & paired seed-bootstrap comparison --------------------------------------------------


def _mean_floats(xs: Iterable[float]) -> float | None:
    vals = list(xs)
    return float(np.mean(vals)) if vals else None


def _seed_bootstrap_ci(values: np.ndarray, *, n_boot: int = N_BOOT, seed: int = 0) -> list[float | None]:
    n = len(values)
    if n == 0:
        return [None, None]
    if n == 1:
        v = float(values[0])
        return [v, v]
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def _per_seed_table(ok_rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    """Average metrics over `ok` reps within each `(tier, seed)` so every seed weighs equally."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in ok_rows:
        if r.get("tier") is None or r.get("seed") is None:
            continue
        grouped.setdefault((str(r["tier"]), int(r["seed"])), []).append(r)

    table: dict[tuple[str, int], dict[str, Any]] = {}
    for key, reps in sorted(grouped.items()):
        def avg(fn, items: list[dict[str, Any]] = reps) -> float | None:
            vals = [float(v) for x in items if (v := fn(x)) is not None]
            return float(np.mean(vals)) if vals else None

        calls = sum(int(x.get("tool_calls") or 0) for x in reps)
        inv = sum(int(x.get("invalid_calls") or 0) for x in reps)
        ref = sum(int(x.get("refusals") or 0) for x in reps)
        table[key] = {
            "reps": len(reps),
            "cost": float(np.mean([float(x["cost_total"]) for x in reps])),
            "reward": float(np.mean([float(x["reward"]) for x in reps])),
            "components": {c: avg(lambda x, c=c: (x.get("components") or {}).get(c)) for c in COMPONENTS},
            "on_time_pct": avg(lambda x: (x.get("kpis") or {}).get("on_time_pct")),
            "passenger_delay_minutes": avg(lambda x: (x.get("kpis") or {}).get("passenger_delay_minutes")),
            "freight_delivered_pct": avg(lambda x: (x.get("kpis") or {}).get("freight_delivered_pct")),
            "clean_shift": avg(
                lambda x: None
                if (x.get("kpis") or {}).get("clean_shift") is None
                else float(bool((x.get("kpis") or {}).get("clean_shift")))
            ),
            "gridlocks": avg(lambda x: x.get("gridlocks", (x.get("counts") or {}).get("gridlocks"))),
            "crew_timeouts": avg(lambda x: x.get("crew_timeouts", (x.get("counts") or {}).get("crew_timeouts"))),
            "unfinished": avg(lambda x: x.get("unfinished", (x.get("counts") or {}).get("unfinished"))),
            "format_errors": avg(lambda x: x.get("format_errors")),
            "tool_calls": avg(lambda x: x.get("tool_calls")),
            "invalid_rate": (inv / calls) if calls > 0 else 0.0,
            "refusal_rate": (ref / calls) if calls > 0 else 0.0,
            "tokens_in": avg(lambda x: x.get("tokens_in")),
            "tokens_out": avg(lambda x: x.get("tokens_out")),
            "tokens_reasoning": avg(lambda x: x.get("tokens_reasoning")),
            "tokens_cached": avg(lambda x: x.get("tokens_cached")),
            "cost_usd": avg(lambda x: x.get("cost_usd")),
            "wall_s": avg(lambda x: x.get("wall_s")),
        }
    return table


def summarize_group(rows: list[dict[str, Any]], *, n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok" and r.get("cost_total") is not None]
    planned = len(rows)
    errors = planned - len(ok)
    error_rate = (errors / planned) if planned > 0 else 0.0
    per_seed = _per_seed_table(ok)
    costs = np.array([v["cost"] for v in per_seed.values()], dtype=float)
    rewards = np.array([v["reward"] for v in per_seed.values()], dtype=float)

    def seed_avg(field: str) -> float | None:
        return _mean_floats(v[field] for v in per_seed.values() if v.get(field) is not None)

    return {
        "episodes": planned,
        "ok": len(ok),
        "infra_errors": errors,
        "error_rate": round(error_rate, 4),
        "incomplete": planned > 0 and error_rate > INCOMPLETE_ERROR_RATE,
        "seeds": len(per_seed),
        "mean_cost": float(costs.mean()) if len(costs) else None,
        "cost_ci95": _seed_bootstrap_ci(costs, n_boot=n_boot, seed=seed),
        "mean_reward": float(rewards.mean()) if len(rewards) else None,
        "ci95": _seed_bootstrap_ci(rewards, n_boot=n_boot, seed=seed),
        "components": {
            c: _mean_floats(v["components"][c] for v in per_seed.values() if v["components"].get(c) is not None)
            for c in COMPONENTS
        },
        "kpis": {
            "on_time_pct": seed_avg("on_time_pct"),
            "passenger_delay_minutes": seed_avg("passenger_delay_minutes"),
            "freight_delivered_pct": seed_avg("freight_delivered_pct"),
            "clean_shift_rate": seed_avg("clean_shift"),
        },
        "failures": {
            "gridlocks": seed_avg("gridlocks"),
            "crew_timeouts": seed_avg("crew_timeouts"),
            "unfinished": seed_avg("unfinished"),
            "format_errors": seed_avg("format_errors"),
            "invalid_rate": seed_avg("invalid_rate"),
            "refusal_rate": seed_avg("refusal_rate"),
        },
        "tool_calls_mean": seed_avg("tool_calls"),
        "tokens_mean": {
            "in": seed_avg("tokens_in"),
            "out": seed_avg("tokens_out"),
            "reasoning": seed_avg("tokens_reasoning"),
            "cached": seed_avg("tokens_cached"),
        },
        "usd_total": round(sum(float(r.get("cost_usd") or 0.0) for r in ok), 6),
        "usd_mean": seed_avg("cost_usd"),
        "wall_s_mean": seed_avg("wall_s"),
    }


def summarize_rows(rows: Iterable[dict[str, Any]], *, n_boot: int = N_BOOT, seed: int = 0) -> dict[str, dict[str, Any]]:
    deduped = dedupe_rows(rows)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in deduped:
        m, t = str(r.get("model") or "unknown"), str(r.get("tier") or "UNKNOWN")
        groups.setdefault((m, t), []).append(r)
        groups.setdefault((m, "ALL"), []).append(r)
    return {f"{m}|{t}": summarize_group(g, n_boot=n_boot, seed=seed) for (m, t), g in sorted(groups.items())}


def _paired_slice(
    base_rows: list[dict[str, Any]],
    cand_rows: list[dict[str, Any]],
    *,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> dict[str, Any]:
    base_sum = summarize_group(base_rows, n_boot=n_boot, seed=seed)
    cand_sum = summarize_group(cand_rows, n_boot=n_boot, seed=seed)
    base_ok = [r for r in base_rows if r.get("status") == "ok" and r.get("cost_total") is not None]
    cand_ok = [r for r in cand_rows if r.get("status") == "ok" and r.get("cost_total") is not None]
    b_table = _per_seed_table(base_ok)
    c_table = _per_seed_table(cand_ok)

    b_keys, c_keys = set(b_table), set(c_table)
    common = sorted(b_keys & c_keys)
    n = len(common)
    if n == 0:
        return {
            "common_seeds": 0,
            "base_only_seeds": len(b_keys - c_keys),
            "cand_only_seeds": len(c_keys - b_keys),
            "base_incomplete": base_sum["incomplete"],
            "cand_incomplete": cand_sum["incomplete"],
            "mean_base_cost": None,
            "mean_cand_cost": None,
            "cost_delta": None,
            "cost_delta_ci95": [None, None],
            "mean_base_reward": None,
            "mean_cand_reward": None,
            "reward_delta": None,
            "reward_delta_ci95": [None, None],
            "median_reward_delta": None,
            "cost_reduction_pct": None,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "component_deltas": {c: None for c in COMPONENTS},
            "kpi_deltas": {},
            "failure_deltas": {},
            "inference_deltas": {},
            "significant_improvement": False,
            "verdict": "INCOMPLETE (0 common seeds)",
        }

    b_cost = np.array([b_table[k]["cost"] for k in common], dtype=float)
    c_cost = np.array([c_table[k]["cost"] for k in common], dtype=float)
    cost_diffs = c_cost - b_cost  # negative = cand is cheaper
    reward_diffs = -cost_diffs  # positive = cand has higher -cost reward

    cost_ci = _seed_bootstrap_ci(cost_diffs, n_boot=n_boot, seed=seed)
    reward_ci = _seed_bootstrap_ci(reward_diffs, n_boot=n_boot, seed=seed)

    wins = int(np.sum(c_cost < b_cost - 1e-6))
    losses = int(np.sum(c_cost > b_cost + 1e-6))
    ties = n - wins - losses

    def paired_field_delta(getter) -> float | None:
        ds = [
            float(cv) - float(bv)
            for k in common
            if (bv := getter(b_table[k])) is not None and (cv := getter(c_table[k])) is not None
        ]
        return float(np.mean(ds)) if ds else None

    incomplete = bool(base_sum["incomplete"] or cand_sum["incomplete"] or n < 2)
    sig_up = (not incomplete) and reward_ci[0] is not None and reward_ci[0] > 0.0
    sig_down = (not incomplete) and reward_ci[1] is not None and reward_ci[1] < 0.0
    if incomplete:
        verdict = "INCOMPLETE (>10% errors or <2 common seeds)"
    elif sig_up:
        verdict = "IMPROVED (95% paired CI > 0)"
    elif sig_down:
        verdict = "REGRESSED (95% paired CI < 0)"
    else:
        verdict = "INCONCLUSIVE (95% paired CI spans 0)"

    mean_b_cost = float(b_cost.mean())
    mean_c_cost = float(c_cost.mean())
    pct_red = (100.0 * (mean_b_cost - mean_c_cost) / mean_b_cost) if mean_b_cost > 1e-9 else None

    return {
        "common_seeds": n,
        "base_only_seeds": len(b_keys - c_keys),
        "cand_only_seeds": len(c_keys - b_keys),
        "base_incomplete": base_sum["incomplete"],
        "cand_incomplete": cand_sum["incomplete"],
        "mean_base_cost": mean_b_cost,
        "mean_cand_cost": mean_c_cost,
        "cost_delta": float(cost_diffs.mean()),
        "cost_delta_ci95": cost_ci,
        "mean_base_reward": float((-b_cost).mean()),
        "mean_cand_reward": float((-c_cost).mean()),
        "reward_delta": float(reward_diffs.mean()),
        "reward_delta_ci95": reward_ci,
        "median_reward_delta": float(np.median(reward_diffs)),
        "cost_reduction_pct": pct_red,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "component_deltas": {
            c: paired_field_delta(lambda row, c=c: row["components"].get(c)) for c in COMPONENTS
        },
        "kpi_deltas": {
            "on_time_pct": paired_field_delta(lambda row: row["on_time_pct"]),
            "passenger_delay_minutes": paired_field_delta(lambda row: row["passenger_delay_minutes"]),
            "freight_delivered_pct": paired_field_delta(lambda row: row["freight_delivered_pct"]),
            "clean_shift_rate": paired_field_delta(lambda row: row["clean_shift"]),
        },
        "failure_deltas": {
            "gridlocks": paired_field_delta(lambda row: row["gridlocks"]),
            "crew_timeouts": paired_field_delta(lambda row: row["crew_timeouts"]),
            "unfinished": paired_field_delta(lambda row: row["unfinished"]),
            "format_errors": paired_field_delta(lambda row: row["format_errors"]),
            "invalid_rate": paired_field_delta(lambda row: row["invalid_rate"]),
            "refusal_rate": paired_field_delta(lambda row: row["refusal_rate"]),
        },
        "inference_deltas": {
            "tokens_in": paired_field_delta(lambda row: row["tokens_in"]),
            "tokens_out": paired_field_delta(lambda row: row["tokens_out"]),
            "tokens_reasoning": paired_field_delta(lambda row: row["tokens_reasoning"]),
            "cost_usd": paired_field_delta(lambda row: row["cost_usd"]),
            "wall_s": paired_field_delta(lambda row: row["wall_s"]),
        },
        "significant_improvement": sig_up,
        "verdict": verdict,
    }


def paired_compare(
    rows: Iterable[dict[str, Any]],
    *,
    base_model: str,
    cand_model: str,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Paired `(cand_model - base_model)` comparison across `'ALL'` and each present tier."""
    deduped = dedupe_rows(rows)
    b_all = [r for r in deduped if r.get("model") == base_model]
    c_all = [r for r in deduped if r.get("model") == cand_model]
    tiers = sorted({str(r["tier"]) for r in (b_all + c_all) if r.get("tier")})
    out: dict[str, dict[str, Any]] = {}
    for tier in tiers:
        out[tier] = _paired_slice(
            [r for r in b_all if r.get("tier") == tier],
            [r for r in c_all if r.get("tier") == tier],
            n_boot=n_boot,
            seed=seed,
        )
    out["ALL"] = _paired_slice(b_all, c_all, n_boot=n_boot, seed=seed)
    return out


# -- Markdown & JSON report builder ------------------------------------------------------------------


def _f(x: float | None, nd: int = 1, signed: bool = False) -> str:
    if x is None:
        return "-"
    fmt = f"+.{nd}f" if signed else f".{nd}f"
    return format(x, fmt)


def render_eval_markdown(
    run_name: str,
    summary: dict[str, dict[str, Any]],
    paired: dict[str, dict[str, dict[str, Any]]],
    *,
    base_model: str | None = None,
) -> str:
    lines = [
        f"# Evaluation Report `{run_name}`",
        "",
        "Reward = `-(C1 + C2 + C3 + C4 + C5)`; seed-level means & 95% bootstrap CIs (10,000 resamples).",
        "",
        "## 1. Per-Model Summary",
        "",
        "| model | tier | seeds | ok/eps | mean cost | mean -cost [95% CI] | C1 | C2 | C3 | C4 | C5 "
        "| on-time % | clean % | gridlocks | invalid % | refusal % | tok in | tok out | USD | wall s |",
        "|" + "---|" * 20,
    ]
    for key, s in summary.items():
        model, tier = key.split("|", 1)
        c, k, fl, t = s["components"], s["kpis"], s["failures"], s["tokens_mean"]
        pct = lambda v: None if v is None else 100.0 * v  # noqa: E731
        lines.append(
            f"| {model} | {tier} | {s['seeds']} | {s['ok']}/{s['episodes']} | {_f(s['mean_cost'])} "
            f"| {_f(s['mean_reward'])} [{_f(s['ci95'][0])}, {_f(s['ci95'][1])}] | "
            + " | ".join(_f(c[x]) for x in COMPONENTS)
            + f" | {_f(k['on_time_pct'])} | {_f(pct(k['clean_shift_rate']))} | {_f(fl['gridlocks'], 2)} "
            + f"| {_f(pct(fl['invalid_rate']), 2)} | {_f(pct(fl['refusal_rate']), 2)} "
            + f"| {_f(t['in'], 0)} | {_f(t['out'], 0)} | {_f(s['usd_total'], 2)} | {_f(s['wall_s_mean'], 0)} |"
        )

    if paired and base_model:
        lines.extend(
            [
                "",
                f"## 2. Paired Comparison vs. `{base_model}` (`candidate - base`)",
                "",
                "| candidate | tier | paired seeds | base cost | cand cost | Δ reward (-Δ cost) [95% CI] | cost Δ % "
                "| W / T / L | Δ clean % | Δ invalid % | Δ wall s | verdict |",
                "|" + "---|" * 12,
            ]
        )
        for cand_model, by_tier in sorted(paired.items()):
            for tier, p in by_tier.items():
                pct = lambda v: None if v is None else 100.0 * v  # noqa: E731
                lines.append(
                    f"| {cand_model} | {tier} | {p['common_seeds']} | {_f(p['mean_base_cost'])} "
                    f"| {_f(p['mean_cand_cost'])} | {_f(p['reward_delta'], signed=True)} "
                    f"[{_f(p['reward_delta_ci95'][0], signed=True)}, {_f(p['reward_delta_ci95'][1], signed=True)}] "
                    f"| {_f(None if p['cost_reduction_pct'] is None else -p['cost_reduction_pct'], signed=True)}% "
                    f"| {p['wins']}/{p['ties']}/{p['losses']} "
                    f"| {_f(pct(p['kpi_deltas'].get('clean_shift_rate')), 1, signed=True)} "
                    f"| {_f(pct(p['failure_deltas'].get('invalid_rate')), 2, signed=True)} "
                    f"| {_f(p['inference_deltas'].get('wall_s'), 0, signed=True)} | {p['verdict']} |"
                )
    return "\n".join(lines) + "\n"


def build_eval_report(
    *,
    run_name: str,
    row_paths: Iterable[str | Path] = (),
    extra_rows: Iterable[dict[str, Any]] = (),
    harbor_jobs: Iterable[str | Path] = (),
    harbor_model_label: str = "opus55-droid",
    base_model: str | None = None,
    out_dir: str | Path | None = None,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> dict[str, Any]:
    combined = [
        *read_rows_jsonl(row_paths),
        *(normalize_row(r) for r in extra_rows),
        *harbor_jobs_to_rows(harbor_jobs, model_label=harbor_model_label),
    ]
    rows = dedupe_rows(combined)
    models = sorted({str(r["model"]) for r in rows if r.get("model")})
    resolved_base = base_model if base_model in models else (models[0] if models else None)
    summary = summarize_rows(rows, n_boot=n_boot, seed=seed)
    paired: dict[str, dict[str, dict[str, Any]]] = {}
    if resolved_base is not None:
        for m in models:
            if m != resolved_base:
                paired[m] = paired_compare(rows, base_model=resolved_base, cand_model=m, n_boot=n_boot, seed=seed)
    md = render_eval_markdown(run_name, summary, paired, base_model=resolved_base)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out / "paired.json").write_text(json.dumps(paired, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out / "report.md").write_text(md, encoding="utf-8")
    return {
        "run": run_name,
        "base_model": resolved_base,
        "models": models,
        "summary": summary,
        "paired": paired,
        "markdown": md,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="railway.train.evaluate")
    sub = p.add_subparsers(dest="cmd", required=True)

    rep = sub.add_parser("report", help="Summarize one or more rows.jsonl files and compute paired CIs.")
    rep.add_argument("--rows", nargs="*", default=[], help="Path(s) to rows.jsonl files")
    rep.add_argument("--harbor-jobs", nargs="*", default=[], help="Optional Harbor job dir(s)/globs (e.g. Opus 5.5)")
    rep.add_argument("--harbor-label", default="opus55-droid", help="Model label for --harbor-jobs rows")
    rep.add_argument("--base", default="", help="Baseline model name for paired comparisons")
    rep.add_argument("--run-name", default="eval")
    rep.add_argument("--out", default="", help="Optional output folder for summary.json, paired.json, report.md")
    rep.add_argument("--n-boot", type=int, default=N_BOOT)

    args = p.parse_args(argv)
    bundle = build_eval_report(
        run_name=args.run_name,
        row_paths=args.rows,
        harbor_jobs=args.harbor_jobs,
        harbor_model_label=args.harbor_label,
        base_model=args.base or None,
        out_dir=args.out or None,
        n_boot=args.n_boot,
    )
    print(bundle["markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
