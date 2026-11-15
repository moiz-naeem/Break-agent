"""End-to-end training drivers that run inside one GPU container (infra/modal/train.py wraps them).

Layout: vLLM on the first GPU(s) samples episodes; the trainer holds the bf16 base + LoRA on the last GPU.
After each update the adapter is saved (to the `railway-adapters` volume) and hot-loaded into the local
vLLM under a new name, so the next rollouts use it. The caller must set CUDA_VISIBLE_DEVICES to the
trainer GPU before torch is imported; vLLM gets its GPUs through its own environment.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

from railway.train.advantages import assign_advantages
from railway.train.env import make_env, train_range
from railway.train.grpo import GRPOConfig, grpo_step, training_turns
from railway.train.rollout import CaptureClient, rollout_groups, summarize
from railway.train.samples import Trajectory, read_jsonl, sft_filter, write_jsonl
from railway.train.serving import LocalVLLM
from railway.train.sft import SFTConfig, render_prompt_ids, sft_train


def log(*parts) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *parts, flush=True)


def gpu_snapshot() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return out.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        return [f"nvidia-smi failed: {exc}"]


def train_seeds(count: int, seed: int, env=None) -> list[int]:
    """Training seeds: the network env draws from its accepted train seed list (validation excluded)."""
    if env is not None and hasattr(env, "train_seeds"):
        return env.train_seeds(count, seed)
    lo, hi = train_range()
    return random.Random(seed).sample(range(lo, hi), count)


class Timer:
    def __init__(self):
        self.phases: dict[str, float] = {}

    def run(self, name: str, fn: Callable):
        t0 = time.monotonic()
        log(f"phase {name}: start")
        out = fn()
        self.phases[name] = round(time.monotonic() - t0, 1)
        log(f"phase {name}: {self.phases[name]} s")
        return out


def _client_factory(server: LocalVLLM, name: str, *, thinking: bool, max_tokens: int, temperature: float):
    return lambda: CaptureClient(
        server.base_url, name, temperature=temperature, max_tokens=max_tokens, thinking=thinking
    )


def best_per_group(trajectories: list[Trajectory]) -> list[Trajectory]:
    best: dict[str, Trajectory] = {}
    for t in trajectories:
        if t.ok and (t.group not in best or t.reward > best[t.group].reward):
            best[t.group] = t
    return list(best.values())


def _sft_turns(trajectories: list[Trajectory]) -> list:
    return [s for t in trajectories for s in t.turns if len(s.tool_calls) == 1 and s.has_tokens]


class Workbench:
    """vLLM + trainer in one container. `start()` launches vLLM and loads the trainer in parallel."""

    def __init__(self, *, adapter: str | None = None, vllm_gpus: str | None = None, max_model_len: int = 32768):
        gpus = vllm_gpus or os.environ.get("RAILWAY_VLLM_GPUS", "0")
        self.server = LocalVLLM(gpus=gpus, max_model_len=max_model_len)
        self.start_adapter = adapter
        self.model = self.tokenizer = self.think_end = None
        self.timer = Timer()

    def start(self) -> Workbench:
        from railway.train.policy import load_policy, load_tokenizer, think_end_id

        self.server.start()
        loaded: dict = {}

        def load():
            t0 = time.monotonic()
            loaded["tok"] = load_tokenizer()
            loaded["model"] = load_policy(adapter=self.start_adapter)
            loaded["s"] = round(time.monotonic() - t0, 1)

        th = threading.Thread(target=load)
        th.start()
        self.timer.phases["vllm_start"] = round(self.server.wait(), 1)
        log("vLLM healthy after", self.timer.phases["vllm_start"], "s")
        th.join()
        if "model" not in loaded:
            raise RuntimeError("trainer failed to load; see the traceback above")
        self.timer.phases["trainer_load"] = loaded["s"]
        self.tokenizer, self.model = loaded["tok"], loaded["model"]
        self.think_end = think_end_id(self.tokenizer)
        log("trainer loaded in", loaded["s"], "s; </think> id", self.think_end)
        return self

    def save(self, path: Path, name: str, commit: Callable[[], None]) -> None:
        self.model.save_pretrained(str(path))
        commit()
        t0 = time.monotonic()
        self.server.swap_lora(name, str(path))
        log(f"adapter {name} saved to {path} and loaded into vLLM in {time.monotonic() - t0:.1f} s")

    def stop(self) -> None:
        self.server.stop()


def smoke(
    *,
    out_dir: str,
    adapters_dir: str,
    run: str,
    commit: Callable[[], None] = lambda: None,
    env_name: str = "standin",
    seeds: tuple[int, ...] = (10000, 10001),
    group: int = 4,
    max_turns: int = 10,
    thinking: bool = True,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    sft_batch: int = 16,
    grpo_turns_per_episode: int | None = 5,
    final_group: int = 4,
) -> dict:
    out, adir = Path(out_dir) / run, Path(adapters_dir) / run
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "run": run,
        "config": dict(
            seeds=list(seeds),
            group=group,
            max_turns=max_turns,
            thinking=thinking,
            max_tokens=max_tokens,
            temperature=temperature,
            env=env_name,
        ),
    }
    env = make_env(env_name)
    wb = Workbench().start()
    timer = wb.timer
    think_end = wb.think_end if thinking else None
    try:

        def roll(name: str, tag: str, g: int) -> list[Trajectory]:
            factory = _client_factory(
                wb.server, name, thinking=thinking, max_tokens=max_tokens, temperature=temperature
            )
            trajs = rollout_groups(env, list(seeds), g, factory, tag=tag, max_turns=max_turns)
            write_jsonl(out / f"{tag}.jsonl", trajs)
            report[f"rollout_{tag}"] = summarize(trajs)
            log(tag, json.dumps(report[f"rollout_{tag}"]))
            return trajs

        # Fail fast: an untrained adapter must save, hot-load and generate before any long rollout.
        timer.run("adapter_probe", lambda: wb.save(adir / "init", f"{run}-init", commit))
        probe_client = CaptureClient(wb.server.base_url, f"{run}-init", max_tokens=64, thinking=False)
        report["adapter_probe"] = probe_client.chat([{"role": "user", "content": "Reply with the word OK."}], [])
        probe_client.close()
        log("adapter probe reply", report["adapter_probe"])

        base = timer.run("rollout_base", lambda: roll("qwen38", "base", group))

        # The chat template, rendered by us, must give vLLM's exact prompt tokens (SFT on teacher data relies on it).
        probe = next((s for t in base for s in t.turns if s.prompt_token_ids), None)
        if probe is not None:
            mine = render_prompt_ids(wb.tokenizer, probe, thinking=thinking)
            theirs = probe.prompt_token_ids
            diff = next((i for i, (a, b) in enumerate(zip(mine, theirs, strict=False)) if a != b), None)
            report["render_check"] = {
                "equal": mine == theirs,
                "ours": len(mine),
                "vllm": len(theirs),
                "first_diff": diff,
            }
            log("render check", report["render_check"])

        chosen = sft_filter(base) or best_per_group(base)
        report["sft_source"] = {"beat_greedy": len(sft_filter(base)), "episodes": len(chosen)}
        sft_hist = timer.run(
            "sft_step",
            lambda: sft_train(
                wb.model,
                wb.tokenizer,
                _sft_turns(chosen),
                SFTConfig(steps=1, batch=sft_batch),
                think_end=think_end,
                log=log,
            ),
        )
        report["sft"] = sft_hist
        timer.run("sft_save_load", lambda: wb.save(adir / "sft", f"{run}-sft", commit))

        it1 = timer.run("rollout_it1", lambda: roll(f"{run}-sft", "it1", group))
        cfg = GRPOConfig(max_turns_per_episode=grpo_turns_per_episode)
        kept = assign_advantages(it1, scale=cfg.scale)
        turns = training_turns(kept, cfg)
        report["grpo_batch"] = {
            "episodes": len(kept),
            "turns": len(turns),
            "advantages": {t.episode_id: round(t.turns[0].advantage, 3) for t in kept if t.turns},
        }
        report["grpo"] = timer.run("grpo_step", lambda: grpo_step(wb.model, turns, cfg, think_end=think_end, log=log))
        timer.run("grpo_save_load", lambda: wb.save(adir / "grpo1", f"{run}-grpo1", commit))

        if final_group:
            timer.run("rollout_after", lambda: roll(f"{run}-grpo1", "after", final_group))
        report["gpus_end"] = gpu_snapshot()
    finally:
        report["phases_s"] = timer.phases
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        commit()
        wb.stop()
    return report


def grpo_run(
    *,
    out_dir: str,
    adapters_dir: str,
    run: str,
    commit: Callable[[], None] = lambda: None,
    env_name: str = "standin",
    start_adapter: str | None = None,
    iterations: int = 10,
    seeds_per_iter: int = 4,
    group: int = 8,
    max_turns: int | None = None,
    thinking: bool = True,
    max_tokens: int = 8192,
    temperature: float = 1.0,
    cfg: GRPOConfig | None = None,
    seed: int = 0,
) -> dict:
    """The real loop: `iterations` x (rollouts on fresh training seeds -> one GRPO update -> hot reload)."""
    cfg = cfg or GRPOConfig()
    out, adir = Path(out_dir) / run, Path(adapters_dir) / run
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(env_name)
    wb = Workbench(adapter=start_adapter).start()
    history = []
    name = "qwen38"
    if start_adapter:
        name = f"{run}-start"
        wb.server.swap_lora(name, start_adapter)
    import torch

    from railway.train.policy import trainable_parameters

    optimizer = torch.optim.AdamW(trainable_parameters(wb.model), lr=cfg.lr)
    seeds_all = train_seeds(iterations * seeds_per_iter, seed, env)
    try:
        for it in range(iterations):
            seeds = seeds_all[it * seeds_per_iter : (it + 1) * seeds_per_iter]
            t0 = time.monotonic()
            factory = _client_factory(
                wb.server, name, thinking=thinking, max_tokens=max_tokens, temperature=temperature
            )
            trajs = rollout_groups(env, seeds, group, factory, tag=f"it{it}", max_turns=max_turns)
            write_jsonl(out / f"it{it}.jsonl", trajs)
            t_roll = time.monotonic() - t0
            kept = assign_advantages(trajs, scale=cfg.scale)
            step = grpo_step(
                wb.model,
                training_turns(kept, cfg),
                cfg,
                think_end=wb.think_end if thinking else None,
                optimizer=optimizer,
                log=log,
            )
            name = f"{run}-it{it}"
            wb.save(adir / f"it{it}", name, commit)
            row = {
                "iteration": it,
                "seeds": seeds,
                "rollout": summarize(trajs),
                "step": step,
                "rollout_s": round(t_roll, 1),
                "iteration_s": round(time.monotonic() - t0, 1),
            }
            history.append(row)
            log(json.dumps(row)[:2000])
            (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            commit()
    finally:
        wb.stop()
    return {"iterations": history}


def sft_run(
    *,
    data: list[str],
    uploaded_blobs: dict[str, bytes] | None = None,
    adapters_dir: str,
    out_dir: str = "/data",
    run: str,
    commit: Callable[[], None] = lambda: None,
    steps: int = 200,
    batch: int = 16,
    lr: float = 1e-4,
    think: str = "train",
    thinking: bool = False,
) -> dict:
    """SFT warm start. `data` files are either teacher `samples.jsonl` (one TurnSample per line, from
    `railway.train.teacher_data`) or our trajectory JSONL (one episode per line, filtered by `sft_filter`).
    Teacher samples carry no reasoning: train them with `thinking=False` (the `qwen38-nothink` serving)."""
    from railway.train.policy import load_policy, load_tokenizer, think_end_id

    run_dir = Path(out_dir) / run
    run_dir.mkdir(parents=True, exist_ok=True)
    blobs = uploaded_blobs or {}
    resolved = []
    for idx, p in enumerate(data):
        if p in blobs:
            staged = run_dir / f"input_{idx}_{Path(p).name}"
            staged.write_bytes(blobs[p])
            resolved.append(str(staged))
        else:
            resolved.append(p)

    tok = load_tokenizer()
    model = load_policy()
    samples = load_any_samples(resolved)
    log(f"{len(samples)} SFT samples from {len(resolved)} files (thinking={'on' if thinking else 'off'})")
    hist = sft_train(
        model,
        tok,
        samples,
        SFTConfig(steps=steps, batch=batch, lr=lr, think=think),
        think_end=think_end_id(tok) if thinking else None,
        log=log,
    )
    path = Path(adapters_dir) / run / "sft"
    model.save_pretrained(str(path))
    result = {"run": run, "adapter": str(path), "samples": len(samples), "thinking": thinking, "history": hist}
    (run_dir / "sft_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    commit()
    return result


def load_any_samples(paths: list[str]) -> list:
    """TurnSamples from teacher samples.jsonl files and/or trajectory JSONL files (format detected per file)."""
    import json

    from railway.train.samples import load_sft_samples
    from railway.train.teacher import read_samples_jsonl

    out: list = []
    for p in paths:
        path = Path(p)
        with path.open(encoding="utf-8") as fh:
            first = next((json.loads(line) for line in fh if line.strip()), None)
        if first is None:
            continue
        if "messages" in first and "tool_calls" in first:  # one TurnSample per line
            out.extend(s for s in read_samples_jsonl(path) if len(s.tool_calls) == 1)
        else:
            out.extend(load_sft_samples([path]))
    return out


def load_trajectories(paths: list[str]) -> list[Trajectory]:
    return [t for p in paths for t in read_jsonl(Path(p))]


def sft_contextualized_run(
    *,
    run: str,
    teacher_samples_blob: bytes,
    out_dir: str = "/data",
    adapters_dir: str = "/adapters",
    commit: Callable[[], None] = lambda: None,
    reload_volumes: Callable[[], None] = lambda: None,
    max_sft_samples: int = 1500,
    think_max_tokens: int = 1024,
    sft_steps: int = 80,
    sft_batch: int = 16,
    sft_lr: float = 4e-5,
) -> dict:
    """Populate genuine Qwen `<think>` prefixes on teacher samples and train masked tool-call LoRA SFT (`2x H200`)."""
    from railway.train.sft import populate_thinking_prefixes
    from railway.train.teacher import read_samples_jsonl, write_samples_jsonl

    reload_volumes()
    out, adir = Path(out_dir) / run, Path(adapters_dir) / run
    out.mkdir(parents=True, exist_ok=True)
    adir.mkdir(parents=True, exist_ok=True)
    sft_dir = adir / "sft"
    if (sft_dir / "adapter_config.json").is_file():
        log(f"[sft {run}] adapter already exists at {sft_dir}; skipping re-train")
        return {"run": run, "adapter_path": str(sft_dir), "resumed": True}

    enriched_path = out / "sft_enriched.jsonl"
    if enriched_path.is_file():
        from railway.train.policy import load_policy, load_tokenizer, think_end_id

        timer = Timer()
        enriched = [s for s in read_samples_jsonl(enriched_path) if len(s.tool_calls) == 1]
        prefix_stats = {"resumed_enriched": len(enriched)}
        log(f"[sft {run}] fast-path: resumed {len(enriched)} enriched samples from {enriched_path} (no vLLM boot needed)")  # noqa: E501
        tok = load_tokenizer()
        model = timer.run("trainer_load", lambda: load_policy())
        sft_hist = timer.run(
            "sft_train",
            lambda: sft_train(
                model,
                tok,
                enriched,
                SFTConfig(lr=sft_lr, steps=sft_steps, batch=sft_batch, think="mask", max_len=32768),
                think_end=think_end_id(tok),
                log=log,
            ),
        )
        model.save_pretrained(str(sft_dir))
        commit()
        result = {
            "run": run,
            "adapter_path": str(sft_dir),
            "prefix_stats": prefix_stats,
            "sft_history": sft_hist,
            "phases_s": timer.phases,
        }
        (out / "sft_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        commit()
        return result

    wb = Workbench(max_model_len=65536).start()
    timer = wb.timer
    try:
        if enriched_path.is_file():
            enriched = [s for s in read_samples_jsonl(enriched_path) if len(s.tool_calls) == 1]
            prefix_stats = {"resumed_enriched": len(enriched)}
            log(f"[sft {run}] resumed {len(enriched)} enriched samples from {enriched_path}")
        else:
            teacher_path = out / "teacher_samples.jsonl"
            teacher_path.write_bytes(teacher_samples_blob)
            raw = [s for s in read_samples_jsonl(teacher_path) if len(s.tool_calls) == 1]
            enriched, prefix_stats = timer.run(
                "populate_think_prefixes",
                lambda: populate_thinking_prefixes(
                    raw,
                    wb.server.base_url,
                    "qwen38",
                    max_samples=max_sft_samples,
                    max_tokens=think_max_tokens,
                    concurrency=24,
                    log=log,
                ),
            )
            write_samples_jsonl(enriched_path, enriched)
            commit()

        sft_hist = timer.run(
            "sft_train",
            lambda: sft_train(
                wb.model,
                wb.tokenizer,
                enriched,
                SFTConfig(lr=sft_lr, steps=sft_steps, batch=sft_batch, think="mask", max_len=32768),
                think_end=wb.think_end,
                log=log,
            ),
        )
        wb.save(sft_dir, f"{run}-sft", commit)
        result = {
            "run": run,
            "adapter_path": str(sft_dir),
            "prefix_stats": prefix_stats,
            "sft_history": sft_hist,
            "phases_s": timer.phases,
        }
        (out / "sft_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        commit()
        return result
    finally:
        wb.stop()


def select_val_winner(val_summary: dict[str, dict], candidates: list[tuple[str, str, str]]) -> dict:
    """Pick `(stage, adapter_path, val_model_label)` strictly by highest `val` `mean_reward` among valid runs."""
    scored = []
    for stage, adapter_path, label in candidates:
        s = val_summary.get(f"{label}|T1") or val_summary.get(f"{label}|ALL")
        if not s or s.get("mean_reward") is None:
            continue
        scored.append(
            {
                "stage": stage,
                "adapter_path": str(adapter_path),
                "val_label": label,
                "incomplete": bool(s.get("incomplete")),
                "mean_reward": float(s["mean_reward"]),
                "mean_cost": float(s["mean_cost"]),
                "ci95": s.get("ci95"),
            }
        )
    if not scored:
        raise RuntimeError("no valid tuned candidates scored on validation split")
    valid = [c for c in scored if not c["incomplete"]] or scored
    winner = max(valid, key=lambda c: c["mean_reward"])
    return {"winner": winner, "candidates": scored}


def _eval_split_local_vllm(
    server: LocalVLLM,
    served_model: str,
    model_label: str,
    *,
    run: str,
    tier: str = "T1",
    split: str = "val",
    reps: int = 1,
    concurrency: int = 16,
    max_tokens: int = 32768,
    temperature: float | None = None,
    adapter_path: str | None = None,
    scenarios_dir: str | None = "/scenarios",
    transcript_root: str | None = "/transcripts",
) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor

    from railway.loop.model_client import ModelClient
    from railway.train import evaluate

    entries = evaluate.select_eval_seeds(tier, split)
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
        for e in entries
        for rep in range(reps)
    ]
    extra_body = {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 4096}

    def make_client() -> ModelClient:
        return ModelClient(
            server.base_url,
            served_model,
            api_key_env=None,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=900,
            provider="vllm",
            extra_body=extra_body,
        )

    def one(j: dict) -> dict:
        row = evaluate.play_eval_job(
            j,
            make_client,
            scenarios_dir=scenarios_dir,
            transcript_root=transcript_root,
        )
        tag = f"cost={row.get('cost_total')}" if row.get("status") == "ok" else f"ERR {row.get('error')}"
        log(f"[{split}] {model_label} {row['scenario_id']} r{row['rep']}: {tag}")
        return row

    with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(jobs)))) as pool:
        return list(pool.map(one, jobs))


def train_t1_pipeline(
    *,
    run: str,
    teacher_samples_blob: bytes,
    base_val_rows_blob: bytes | None = None,
    run_base_val_if_missing: bool = False,
    out_dir: str = "/data",
    adapters_dir: str = "/adapters",
    scenarios_dir: str = "/scenarios",
    transcript_root: str = "/transcripts",
    commit: Callable[[], None] = lambda: None,
    max_sft_samples: int = 1500,
    think_max_tokens: int = 1200,
    sft_steps: int = 80,
    sft_batch: int = 16,
    sft_lr: float = 4e-5,
    grpo_iterations: int = 6,
    grpo_seeds_per_iter: int = 3,
    grpo_group: int = 6,
    grpo_lr: float = 1e-5,
    grpo_beta: float = 0.02,
    grpo_max_turns: int = 24,
    rollout_max_tokens: int = 16384,
    teacher_seed_offset: int = 8,
    run_test_on_winner: bool = True,
) -> dict:
    """End-to-end T1 (`thinking=True`): contextualized tool-call SFT -> val eval -> GRPO -> val selection -> test."""
    import torch

    from railway.train import evaluate
    from railway.train.env import NetEnv, is_net_train_seed, net_seed_list
    from railway.train.policy import trainable_parameters
    from railway.train.sft import populate_thinking_prefixes
    from railway.train.teacher import read_samples_jsonl, write_samples_jsonl

    out, adir = Path(out_dir) / run, Path(adapters_dir) / run
    out.mkdir(parents=True, exist_ok=True)
    adir.mkdir(parents=True, exist_ok=True)

    teacher_path = out / "teacher_samples.jsonl"
    teacher_path.write_bytes(teacher_samples_blob)
    raw_samples = [s for s in read_samples_jsonl(teacher_path) if len(s.tool_calls) == 1]
    if not raw_samples:
        raise ValueError("no single-call teacher samples provided")

    wb = Workbench(max_model_len=65536).start()
    timer = wb.timer
    report: dict = {"run": run}
    try:
        # 0. Base validation rows (reuse pre-computed `eval-t1-val-pre` when supplied, else optional).
        val_base_rows: list[dict] = []
        if base_val_rows_blob:
            val_base_path = out / "val_base_rows.jsonl"
            val_base_path.write_bytes(base_val_rows_blob)
            val_base_rows = evaluate.read_rows_jsonl([val_base_path])
            log(f"loaded {len(val_base_rows)} pre-computed base T1 val rows")
        elif run_base_val_if_missing:
            val_base_rows = timer.run(
                "val_base",
                lambda: _eval_split_local_vllm(
                    wb.server,
                    "qwen38",
                    "qwen38-think",
                    run=run,
                    tier="T1",
                    split="val",
                    concurrency=19,
                    scenarios_dir=scenarios_dir,
                    transcript_root=transcript_root,
                ),
            )
            (out / "val_base_rows.jsonl").write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in val_base_rows),
                encoding="utf-8",
            )

        # 1. Populate genuine Qwen `<think>` prefixes on teacher prompts, then train masked tool-call LoRA SFT.
        enriched, prefix_stats = timer.run(
            "populate_think_prefixes",
            lambda: populate_thinking_prefixes(
                raw_samples,
                wb.server.base_url,
                "qwen38",
                max_samples=max_sft_samples,
                max_tokens=think_max_tokens,
                concurrency=24,
                log=log,
            ),
        )
        report["prefix_stats"] = prefix_stats
        write_samples_jsonl(out / "sft_enriched.jsonl", enriched)

        sft_hist = timer.run(
            "sft_train",
            lambda: sft_train(
                wb.model,
                wb.tokenizer,
                enriched,
                SFTConfig(lr=sft_lr, steps=sft_steps, batch=sft_batch, think="mask", max_len=32768),
                think_end=wb.think_end,
                log=log,
            ),
        )
        report["sft_history"] = sft_hist
        sft_dir = adir / "sft"
        timer.run("sft_save_load", lambda: wb.save(sft_dir, f"{run}-sft", commit))

        # 2. Evaluate SFT checkpoint on all 19 T1 validation seeds.
        val_sft_rows = timer.run(
            "val_sft",
            lambda: _eval_split_local_vllm(
                wb.server,
                f"{run}-sft",
                "qwen38-think-sft",
                run=run,
                tier="T1",
                split="val",
                concurrency=19,
                adapter_path=str(sft_dir),
                scenarios_dir=scenarios_dir,
                transcript_root=transcript_root,
            ),
        )
        (out / "val_sft_rows.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in val_sft_rows),
            encoding="utf-8",
        )
        commit()

        candidates: list[tuple[str, str, str]] = [("sft", str(sft_dir), "qwen38-think-sft")]
        val_all_rows = [*val_base_rows, *val_sft_rows]

        # 3. Optional on-policy multi-turn GRPO (`thinking=True`) on fresh T1 train seeds (disjoint from teacher slice).
        if grpo_iterations > 0:
            env = NetEnv("T1")
            all_train_pool = [
                int(r["seed"]) for r in net_seed_list("T1", "train") if is_net_train_seed(int(r["seed"]))
            ]
            needed = grpo_iterations * grpo_seeds_per_iter
            fresh_pool = all_train_pool[teacher_seed_offset:]
            pool_to_use = fresh_pool if len(fresh_pool) >= needed else all_train_pool
            grpo_seeds = random.Random(42).sample(pool_to_use, needed)

            grpo_cfg = GRPOConfig(
                lr=grpo_lr,
                beta=grpo_beta,
                think="train",
                max_turns_per_episode=grpo_max_turns,
                max_len=32768,
            )
            optimizer = torch.optim.AdamW(trainable_parameters(wb.model), lr=grpo_cfg.lr)
            cur_name = f"{run}-sft"
            grpo_hist = []
            for it in range(grpo_iterations):
                iter_seeds = grpo_seeds[it * grpo_seeds_per_iter : (it + 1) * grpo_seeds_per_iter]
                t_it = time.monotonic()
                factory = _client_factory(
                    wb.server,
                    cur_name,
                    thinking=True,
                    max_tokens=rollout_max_tokens,
                    temperature=0.9,
                )
                trajs = rollout_groups(env, iter_seeds, grpo_group, factory, tag=f"it{it}")
                write_jsonl(out / f"grpo_it{it}.jsonl", trajs)
                t_roll = round(time.monotonic() - t_it, 1)
                kept = assign_advantages(trajs, scale=grpo_cfg.scale)
                step_info = grpo_step(
                    wb.model,
                    training_turns(kept, grpo_cfg),
                    grpo_cfg,
                    think_end=wb.think_end,
                    optimizer=optimizer,
                    log=log,
                )
                cur_name = f"{run}-grpo-it{it}"
                it_dir = adir / f"grpo_it{it}"
                wb.save(it_dir, cur_name, commit)
                row_it = {
                    "iteration": it,
                    "seeds": iter_seeds,
                    "rollout": summarize(trajs),
                    "step": step_info,
                    "rollout_s": t_roll,
                    "iteration_s": round(time.monotonic() - t_it, 1),
                }
                grpo_hist.append(row_it)
                log("grpo_iter:", json.dumps(row_it)[:1500])
                if grpo_iterations >= 4 and (it + 1) == (grpo_iterations // 2):
                    mid_label = f"qwen38-think-grpo-it{it}"
                    mid_rows = timer.run(
                        f"val_grpo_it{it}",
                        lambda cname=cur_name, mlabel=mid_label, mdir=it_dir: _eval_split_local_vllm(
                            wb.server,
                            cname,
                            mlabel,
                            run=run,
                            tier="T1",
                            split="val",
                            concurrency=19,
                            adapter_path=str(mdir),
                            scenarios_dir=scenarios_dir,
                            transcript_root=transcript_root,
                        ),
                    )
                    (out / f"val_grpo_it{it}_rows.jsonl").write_text(
                        "".join(json.dumps(r, sort_keys=True) + "\n" for r in mid_rows),
                        encoding="utf-8",
                    )
                    candidates.append((f"grpo_it{it}", str(it_dir), mid_label))
                    val_all_rows.extend(mid_rows)
                    commit()
            grpo_dir = adir / "grpo"
            wb.save(grpo_dir, f"{run}-grpo", commit)
            report["grpo_history"] = grpo_hist
            (out / "grpo_history.json").write_text(json.dumps(grpo_hist, indent=2) + "\n", encoding="utf-8")

            val_grpo_rows = timer.run(
                "val_grpo",
                lambda: _eval_split_local_vllm(
                    wb.server,
                    f"{run}-grpo",
                    "qwen38-think-grpo",
                    run=run,
                    tier="T1",
                    split="val",
                    concurrency=19,
                    adapter_path=str(grpo_dir),
                    scenarios_dir=scenarios_dir,
                    transcript_root=transcript_root,
                ),
            )
            (out / "val_grpo_rows.jsonl").write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in val_grpo_rows),
                encoding="utf-8",
            )
            candidates.append(("grpo", str(grpo_dir), "qwen38-think-grpo"))
            val_all_rows.extend(val_grpo_rows)
            commit()

        # 4. Select winner strictly on T1 validation split (never peeking at test).
        val_all_path = out / "val_all_rows.jsonl"
        val_all_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in val_all_rows), encoding="utf-8")
        val_bundle = evaluate.build_eval_report(
            run_name=f"{run}-val",
            row_paths=[val_all_path],
            base_model="qwen38-think",
            out_dir=out / "val_report",
        )
        selection = select_val_winner(val_bundle["summary"], candidates)
        report["val_selection"] = selection
        report["val_markdown"] = val_bundle["markdown"]
        (out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
        log("VAL SELECTION:", json.dumps(selection))

        # 5. Evaluate ONLY the validation-selected winner once on the 30 untouched T1 test seeds.
        test_rows_text = ""
        if run_test_on_winner:
            win_path = selection["winner"]["adapter_path"]
            wb.server.swap_lora(f"{run}-winner", win_path)
            test_rows = timer.run(
                "test_winner",
                lambda: _eval_split_local_vllm(
                    wb.server,
                    f"{run}-winner",
                    "qwen38-think-tuned",
                    run=run,
                    tier="T1",
                    split="test",
                    concurrency=16,
                    adapter_path=win_path,
                    scenarios_dir=scenarios_dir,
                    transcript_root=transcript_root,
                ),
            )
            test_rows_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in test_rows)
            (out / "test_tuned_rows.jsonl").write_text(test_rows_text, encoding="utf-8")
        report["val_all_rows_jsonl"] = val_all_path.read_text(encoding="utf-8")
        report["test_tuned_rows_jsonl"] = test_rows_text
    finally:
        report["phases_s"] = timer.phases
        (out / "pipeline_report.json").write_text(
            json.dumps({k: v for k, v in report.items() if not k.endswith("_jsonl")}, indent=2) + "\n",
            encoding="utf-8",
        )
        commit()
        wb.stop()
    return report
