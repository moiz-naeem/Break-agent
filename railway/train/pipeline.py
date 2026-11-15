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
