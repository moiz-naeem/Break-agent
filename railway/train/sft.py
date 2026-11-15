"""LoRA SFT on per-turn samples: loss on the assistant completion only (needs torch).

A sample with exact token ids (our own rollouts) trains on those tokens. A sample without them (teacher
transcripts) is rendered with the model's chat template: prompt = messages + generation prompt, target =
one assistant message with `reasoning_content` (the teacher's thinking, if captured) and the tool call.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from railway.train.samples import IGNORE, TurnSample, token_example


def _template_messages(messages: list[dict]) -> list[dict]:
    """Chat templates iterate tool-call arguments as a mapping; the loop keeps them as JSON strings."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c["function"])
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"] or "{}")
                    except ValueError:
                        fn["arguments"] = {}
                calls.append({**c, "function": fn})
            m["tool_calls"] = calls
        out.append(m)
    return out


def render_prompt_ids(tokenizer, sample: TurnSample, *, thinking: bool = True) -> list[int]:
    text = tokenizer.apply_chat_template(
        _template_messages(sample.messages),
        tools=sample.tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def render_sample(tokenizer, sample: TurnSample, *, thinking: bool = True) -> TurnSample:
    """A copy of `sample` with prompt/completion token ids from the chat template."""
    if len(sample.tool_calls) != 1:
        raise ValueError("an SFT target needs exactly one tool call")
    call = sample.tool_calls[0]
    target = {
        "role": "assistant",
        "content": sample.content or "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": call["name"], "arguments": json.loads(call["arguments_raw"] or "{}")},
            }
        ],
    }
    if thinking:
        target["reasoning_content"] = sample.reasoning or ""
    messages = _template_messages(sample.messages)
    kwargs = {"tools": sample.tools, "tokenize": False, "enable_thinking": thinking}
    head = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(messages + [target], **kwargs)
    if not full.startswith(head):
        raise ValueError("chat template: the prompt is not a prefix of prompt + target")
    prompt_ids = tokenizer(head, add_special_tokens=False)["input_ids"]
    completion = tokenizer(full[len(head) :], add_special_tokens=False)["input_ids"]
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if end in completion:  # drop the newline the template adds after the end-of-turn token
        completion = completion[: completion.index(end) + 1]
    out = TurnSample(**{**sample.__dict__})
    out.prompt_token_ids, out.completion_token_ids, out.logprobs = prompt_ids, completion, None
    return out


def to_example(tokenizer, sample: TurnSample, think_end: int | None, think: str) -> tuple[list[int], list[int]]:
    effective_think = think
    if not sample.has_tokens:
        has_reasoning = bool(sample.reasoning and sample.reasoning.strip())
        if think_end is not None and not has_reasoning:
            # Never train Qwen to emit an immediate empty `</think>` when teacher data lacks reasoning text.
            effective_think = "mask"
        sample = render_sample(tokenizer, sample, thinking=think_end is not None)
    return token_example(sample, think_end, effective_think)


def stratified_subsample(samples: list[TurnSample], max_samples: int, *, seed: int = 0) -> list[TurnSample]:
    """Keep up to `max_samples` while preserving rare tools first and subsampling frequent tools evenly."""
    if max_samples <= 0 or len(samples) <= max_samples:
        return list(samples)
    by_tool: dict[str, list[int]] = {}
    for idx, s in enumerate(samples):
        name = s.tool_calls[0]["name"] if s.tool_calls else "unknown"
        by_tool.setdefault(name, []).append(idx)
    rng = random.Random(seed)
    for idxs in by_tool.values():
        rng.shuffle(idxs)
    picked: set[int] = set()
    # Round-robin across tools so rare action tools (`route`, `skip`, `relieve_crew`, ...) are never squeezed out.
    active = sorted(by_tool)
    cursor = {k: 0 for k in active}
    while len(picked) < max_samples and active:
        next_active = []
        for k in active:
            pos = cursor[k]
            if pos < len(by_tool[k]):
                picked.add(by_tool[k][pos])
                cursor[k] = pos + 1
                if len(picked) >= max_samples:
                    break
                if cursor[k] < len(by_tool[k]):
                    next_active.append(k)
        active = next_active
    return [samples[i] for i in sorted(picked)]


def _canonical_call(call: dict) -> tuple[str, str]:
    try:
        parsed = json.loads(call.get("arguments_raw") or "{}")
        args = json.dumps(parsed, sort_keys=True, ensure_ascii=False) if isinstance(parsed, dict) else ""
    except ValueError:
        args = str(call.get("arguments_raw") or "")
    return str(call.get("name") or ""), args


def populate_thinking_prefixes(
    samples: list[TurnSample],
    base_url: str,
    model: str,
    *,
    max_samples: int = 1500,
    max_tokens: int = 1200,
    temperature: float = 0.6,
    concurrency: int = 24,
    seed: int = 0,
    client_factory=None,
    log=print,
) -> tuple[list[TurnSample], dict]:
    """Query `model` (`thinking=True`) on each teacher prompt to attach a genuine Qwen `<think>` prefix.

    During subsequent `think='mask'` SFT, `<think>...</think>` carries `-100` (zero loss, so Qwen's reasoning
    distribution is never corrupted), while `<tool_call>` is trained conditioned on a realistic thinking context
    instead of an out-of-distribution empty `<think></think>`.
    """
    from railway.loop.model_client import ModelClient

    chosen = stratified_subsample(samples, max_samples, seed=seed)

    def make_default_client():
        return ModelClient(
            base_url,
            model,
            api_key_env=None,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=600,
            provider="vllm",
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )

    factory = client_factory or make_default_client

    def enrich_one(s: TurnSample) -> tuple[TurnSample, bool, bool, bool]:
        if s.has_tokens or (s.reasoning and s.reasoning.strip()):
            return s, True, False, False
        client = factory()
        try:
            reply = client.chat(s.messages, s.tools)
        except Exception:  # noqa: BLE001
            return s, False, False, True
        finally:
            if hasattr(client, "close"):
                client.close()
        thought = reply.get("reasoning")
        if not thought and isinstance(reply.get("content"), str):
            raw = reply["content"].replace("<think>", "").replace("</think>", "").strip()
            thought = raw or None
        calls = reply.get("tool_calls") or []
        matched = bool(calls and s.tool_calls and _canonical_call(calls[0]) == _canonical_call(s.tool_calls[0]))
        out = copy.deepcopy(s)
        if thought and thought.strip():
            # Cap extreme run-on thoughts so prompt + thought + tool_call always fits comfortably in max_len.
            out.reasoning = thought.strip()[:6000]
            out.meta = {**out.meta, "think_populated": True, "think_matched_teacher": matched}
            return out, True, matched, False
        return out, False, matched, False

    t0 = time.monotonic()
    workers = max(1, min(concurrency, len(chosen)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(enrich_one, chosen))

    enriched = [r[0] for r in results]
    populated = sum(1 for r in results if r[1])
    matched = sum(1 for r in results if r[2])
    errors = sum(1 for r in results if r[3])
    lengths = [len(s.reasoning) for s in enriched if s.reasoning]
    stats = {
        "input_samples": len(samples),
        "selected_samples": len(enriched),
        "populated": populated,
        "matched_teacher": matched,
        "matched_rate": round(matched / len(enriched), 4) if enriched else 0.0,
        "errors": errors,
        "mean_reasoning_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        "wall_s": round(time.monotonic() - t0, 1),
    }
    log("populate_thinking_prefixes:", stats)
    return enriched, stats


@dataclass
class SFTConfig:
    lr: float = 1e-4
    steps: int = 1
    batch: int = 16  # samples per optimizer step (gradient accumulation, one sequence at a time)
    max_len: int = 32768
    think: str = "train"
    seed: int = 0
    grad_clip: float = 1.0
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.10


def _step_lr(step: int, total_steps: int, base_lr: float, warmup_ratio: float, min_lr_ratio: float) -> float:
    if total_steps <= 1:
        return base_lr
    warmup = max(1, int(round(total_steps * warmup_ratio))) if warmup_ratio > 0 else 0
    if step < warmup:
        return base_lr * float(step + 1) / float(warmup)
    progress = float(step - warmup) / float(max(1, total_steps - warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def sft_train(
    model, tokenizer, samples: list[TurnSample], cfg: SFTConfig, *, think_end: int | None, log=print
) -> list[dict]:
    """`cfg.steps` optimizer steps, each on `cfg.batch` samples; loss = mean NLL over all loss tokens of the batch."""
    import torch

    from railway.train.policy import gpu_memory_gb, token_logprobs, trainable_parameters

    examples = [to_example(tokenizer, s, think_end, cfg.think) for s in samples]
    examples = [e for e in examples if len(e[0]) <= cfg.max_len and any(lab != IGNORE for lab in e[1])]
    if not examples:
        raise ValueError("no usable SFT examples")
    rng = random.Random(cfg.seed)
    deck: list[tuple[list[int], list[int]]] = []
    params = trainable_parameters(model)
    optimizer = torch.optim.AdamW(params, lr=cfg.lr)
    model.train()
    history = []
    batch_size = min(cfg.batch, len(examples))
    for step in range(cfg.steps):
        t0 = time.monotonic()
        lr_now = _step_lr(step, cfg.steps, cfg.lr, cfg.warmup_ratio, cfg.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr_now
        batch: list[tuple[list[int], list[int]]] = []
        while len(batch) < batch_size:
            if not deck:
                deck = list(examples)
                rng.shuffle(deck)
            take = min(batch_size - len(batch), len(deck))
            batch.extend(deck[:take])
            deck = deck[take:]
        n_tokens = sum(sum(lab != IGNORE for lab in labels) for _, labels in batch)
        total, seq_tokens = 0.0, 0
        for ids, labels in batch:
            first = next(i for i, lab in enumerate(labels) if lab != IGNORE)
            n_comp = len(ids) - first
            mask = torch.tensor([lab != IGNORE for lab in labels[first:]], device=params[0].device)
            lp = token_logprobs(model, ids, n_comp)
            loss = -(lp * mask).sum() / n_tokens
            loss.backward()
            total += loss.item()
            seq_tokens += len(ids)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        row = {
            "step": step,
            "lr": round(lr_now, 8),
            "loss": round(total, 4),
            "grad_norm": round(grad_norm, 4),
            "samples": len(batch),
            "loss_tokens": n_tokens,
            "seq_tokens": seq_tokens,
            "wall_s": round(time.monotonic() - t0, 1),
            "gpu_gb": gpu_memory_gb(),
        }
        log(row)
        history.append(row)
    return history
