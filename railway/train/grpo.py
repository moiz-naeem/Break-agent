"""One multi-turn GRPO update (needs torch for the step; advantages live in `advantages.py`).

Per iteration: sample G episodes per seed with the current adapter (rollout.py), give every assistant
turn its episode's group-normalised advantage, then one optimizer step on

    loss = - sum_turns sum_tokens min(r_t A, clip(r_t, 1-eps, 1+eps) A) / (loss tokens in the batch)
           + beta * KL_k3(policy || base)

with r_t = exp(logp_t - logp_old_t). With `epochs=1` the update is strictly on-policy (logp_old is
the trainer's own log-prob, detached), so r_t = 1 and the step is REINFORCE with a group baseline;
`epochs>1` reuses the batch and the clip becomes active. vLLM's sampling log-probs are kept only to
measure the sampler/trainer mismatch.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from railway.train.samples import IGNORE, Trajectory, TurnSample, token_example


@dataclass
class GRPOConfig:
    lr: float = 2e-5
    clip_eps: float = 0.2
    beta: float = 0.0  # KL to the base model; the base is the adapter-disabled policy, no second copy
    epochs: int = 1
    think: str = "train"
    max_turns_per_episode: int | None = None  # subsample turns per episode (unbiased up to scale)
    max_len: int = 32768
    scale: str = "std"
    grad_clip: float = 1.0
    seed: int = 0


def training_turns(trajectories: list[Trajectory], cfg: GRPOConfig) -> list[TurnSample]:
    """Turns with exact tokens and a non-zero advantage, optionally subsampled per episode."""
    rng = random.Random(cfg.seed)
    out = []
    for t in trajectories:
        turns = [s for s in t.turns if s.has_tokens and s.advantage]
        turns = [s for s in turns if len(s.prompt_token_ids) + len(s.completion_token_ids) <= cfg.max_len]
        if cfg.max_turns_per_episode is not None and len(turns) > cfg.max_turns_per_episode:
            turns = sorted(rng.sample(turns, cfg.max_turns_per_episode), key=lambda s: s.turn)
        out.extend(turns)
    return out


def grpo_step(
    model, samples: list[TurnSample], cfg: GRPOConfig, *, think_end: int | None, optimizer=None, log=print
) -> dict:
    import torch

    from railway.train.policy import gpu_memory_gb, token_logprobs, trainable_parameters

    if not samples:
        return {"skipped": "no turns with a non-zero advantage"}
    params = trainable_parameters(model)
    optimizer = optimizer or torch.optim.AdamW(params, lr=cfg.lr)
    examples = []
    for s in samples:
        ids, labels = token_example(s, think_end, cfg.think)
        n = len(s.completion_token_ids)
        examples.append((s, ids, n, [lab != IGNORE for lab in labels[-n:]]))
    n_tokens = sum(sum(m) for *_, m in examples)
    t0 = time.monotonic()
    model.train()
    old: list = [None] * len(examples)
    if cfg.epochs > 1:
        with torch.no_grad():
            old = [token_logprobs(model, ids, n) for _, ids, n, _ in examples]
    stats = []
    for epoch in range(cfg.epochs):
        total, kl_sum, mismatch, clipped = 0.0, 0.0, [], 0
        for k, (s, ids, n, keep) in enumerate(examples):
            mask = torch.tensor(keep, device=params[0].device, dtype=torch.float32)
            lp = token_logprobs(model, ids, n)
            lp_old = old[k] if old[k] is not None else lp.detach()
            if epoch == 0 and s.logprobs and len(s.logprobs) == n:
                vllm = torch.tensor(s.logprobs, device=lp.device)
                mismatch.append((lp.detach() - vllm).abs().mean().item())
            ratio = torch.exp(lp - lp_old)
            adv = float(s.advantage)
            surr = torch.minimum(ratio * adv, ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv)
            clipped += int(((ratio - 1).abs() > cfg.clip_eps).logical_and(mask.bool()).sum().item())
            per_tok = -surr
            if cfg.beta > 0:
                with torch.no_grad(), model.disable_adapter():
                    ref = token_logprobs(model, ids, n)
                d = ref - lp
                kl = torch.exp(d) - d - 1
                per_tok = per_tok + cfg.beta * kl
                kl_sum += (kl.detach() * mask).sum().item()
            loss = (per_tok * mask).sum() / n_tokens
            loss.backward()
            total += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        row = {
            "epoch": epoch,
            "loss": round(total, 6),
            "grad_norm": round(grad_norm, 4),
            "kl": round(kl_sum / n_tokens, 6) if cfg.beta > 0 else None,
            "clipped_tokens": clipped,
            "turns": len(examples),
            "loss_tokens": n_tokens,
            "seq_tokens": sum(len(ids) for _, ids, _, _ in examples),
            "sampler_mismatch": round(sum(mismatch) / len(mismatch), 4) if mismatch else None,
            "wall_s": round(time.monotonic() - t0, 1),
            "gpu_gb": gpu_memory_gb(),
        }
        log(row)
        stats.append(row)
    return {"epochs": stats}
