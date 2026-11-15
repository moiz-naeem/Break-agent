"""The trainable policy: Qwen base in bf16 + a LoRA adapter, and exact per-token log-probs (needs torch)."""

from __future__ import annotations

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_RANK = 16  # serve_qwen.py starts vLLM with --max-lora-rank 16


def load_tokenizer(model: str = MODEL, revision: str = REVISION):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, revision=revision)


def think_end_id(tokenizer) -> int | None:
    tid = tokenizer.convert_tokens_to_ids("</think>")
    return tid if isinstance(tid, int) and tid != tokenizer.unk_token_id else None


def load_policy(
    model: str = MODEL,
    revision: str = REVISION,
    *,
    adapter: str | None = None,
    rank: int = LORA_RANK,
    alpha: int = 32,
    checkpointing: bool = True,
    device: str = "cuda:0",
):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText

    # The model is multimodal (Qwen3_5ForConditionalGeneration); the causal-LM auto class does not apply.
    base = AutoModelForImageTextToText.from_pretrained(
        model, revision=revision, dtype=torch.bfloat16, device_map={"": device}
    )
    base.config.use_cache = False
    if adapter:
        policy = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    else:
        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,  # dropout would make the trainer's log-probs differ from the sampler's
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGETS,
            exclude_modules=r".*(visual|vision).*",
        )
        policy = get_peft_model(base, config)
    if checkpointing:
        policy.enable_input_require_grads()
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return policy


def token_logprobs(model, input_ids: list[int], n_completion: int, *, chunk: int = 1024):
    """log p(token_t | tokens_<t) for the last `n_completion` tokens of `input_ids`, as a float32 tensor.

    Only the completion positions get logits (`logits_to_keep`), so a long prompt costs no vocab-sized
    activations; the log-softmax runs in chunks to cap the float32 copy.
    """
    import torch

    device = next(p for p in model.parameters()).device
    ids = torch.tensor([input_ids], device=device)
    keep = n_completion + 1
    try:
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=keep).logits
    except TypeError:
        logits = model(input_ids=ids, use_cache=False).logits
    logits = logits[0, -keep:-1]
    targets = ids[0, -n_completion:]
    out = []
    for i in range(0, n_completion, chunk):
        part = logits[i : i + chunk].float()
        out.append(part.gather(-1, targets[i : i + chunk, None]).squeeze(-1) - torch.logsumexp(part, dim=-1))
    return torch.cat(out)


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def gpu_memory_gb() -> dict:
    import torch

    return {
        "allocated": round(torch.cuda.memory_allocated() / 1e9, 1),
        "peak": round(torch.cuda.max_memory_allocated() / 1e9, 1),
    }
