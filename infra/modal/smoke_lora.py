"""LoRA smoke test for Qwen3.8-27B on Modal: 30 steps on dummy tool-call examples.

Proves three things before real training data exists: the model loads and trains in bf16 on one
GPU, the chat template renders a tool-call target, and an adapter is saved where the server
(serve_qwen.py) can load it.

Run:  modal run infra/modal/smoke_lora.py            (prompt of about 10k tokens, the real size)
      modal run infra/modal/smoke_lora.py --repeat 5 (short prompt)
"""

import os

import modal

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
GPU = os.environ.get("RAILWAY_GPU", "H200:1")

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "torch", "transformers==5.8.1", "peft", "accelerate", "pillow", "torchvision"
).env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
adapters = modal.Volume.from_name("railway-adapters", create_if_missing=True)
app = modal.App("railway-qwen-lora-smoke")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "authorize",
            "description": "Give a train a movement authority to a station track.",
            "parameters": {
                "type": "object",
                "properties": {
                    "train": {"type": "string"},
                    "station": {"type": "string"},
                    "track": {"type": "string"},
                },
                "required": ["train", "station", "track"],
            },
        },
    }
]


@app.function(
    image=image,
    gpu=GPU,
    timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache, "/adapters": adapters},
)
def smoke(repeat: int = 430, steps: int = 30, checkpointing: bool = True):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    # The model is multimodal (Qwen3_5ForConditionalGeneration), so the causal-LM auto class does not apply.
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="auto"
    )
    print("loaded; GPU GB after load:", round(torch.cuda.memory_allocated() / 1e9, 1))

    # Language-model linear layers only; the vision tower stays frozen and untouched.
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        exclude_modules=r".*(visual|vision).*",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    if checkpointing:
        # Without this, a 5.8k-token prompt ran out of memory on a 140 GB H200: every layer's
        # activations are kept for the backward pass. Checkpointing recomputes them instead.
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    def example(i: int):
        status = f"E{i} express at S1/main auth:- delay:+{i} crew:96 next:S2@{i + 6}\n" * repeat
        prompt = [{"role": "user", "content": f"t={i} | gridlock: none\n{status}Decide the next action."}]
        target = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "authorize",
                        "arguments": {"train": f"E{i}", "station": "S2", "track": "L1"},
                    },
                }
            ],
        }
        head = tok.apply_chat_template(
            prompt, tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        full = tok.apply_chat_template(prompt + [target], tools=TOOLS, tokenize=False, enable_thinking=False)
        return head, full

    head, full = example(0)
    print("---- target the model is trained to produce ----")
    print(full[len(head) :] if full.startswith(head) else full[-600:])
    print("---- prompt tokens:", len(tok(head)["input_ids"]), "----")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    losses = []
    for step in range(steps):
        head, full = example(step)
        batch = tok(full, return_tensors="pt").to(model.device)
        labels = batch["input_ids"].clone()
        labels[:, : len(tok(head)["input_ids"])] = -100  # loss on the assistant tool call only
        loss = model(**batch, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        print(step, round(loss.item(), 4), "peak GPU GB", round(torch.cuda.max_memory_allocated() / 1e9, 1))

    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    print(f"mean loss first 5 steps {first:.4f}, last 5 steps {last:.4f}:", "PASS" if last < first else "FAIL")
    model.save_pretrained("/adapters/smoke")
    adapters.commit()
    print("adapter saved to the railway-adapters volume at /adapters/smoke")


@app.local_entrypoint()
def main(repeat: int = 430, steps: int = 30, checkpointing: bool = True):
    smoke.remote(repeat=repeat, steps=steps, checkpointing=checkpointing)
