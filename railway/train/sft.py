"""LoRA SFT on per-turn samples: loss on the assistant completion only (needs torch).

A sample with exact token ids (our own rollouts) trains on those tokens. A sample without them (teacher
transcripts) is rendered with the model's chat template: prompt = messages + generation prompt, target =
one assistant message with `reasoning_content` (the teacher's thinking, if captured) and the tool call.
"""

from __future__ import annotations

import json
import random
import time
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
    if not sample.has_tokens:
        sample = render_sample(tokenizer, sample, thinking=think_end is not None)
    return token_example(sample, think_end, think)


@dataclass
class SFTConfig:
    lr: float = 1e-4
    steps: int = 1
    batch: int = 16  # samples per optimizer step (gradient accumulation, one sequence at a time)
    max_len: int = 32768
    think: str = "train"
    seed: int = 0
    grad_clip: float = 1.0


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
    params = trainable_parameters(model)
    optimizer = torch.optim.AdamW(params, lr=cfg.lr)
    model.train()
    history = []
    for step in range(cfg.steps):
        t0 = time.monotonic()
        batch = rng.sample(examples, min(cfg.batch, len(examples)))
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
