# Training: SFT warm start + multi-turn GRPO (lane E)

Goal: a LoRA on Qwen 3.8 27B (thinking) that dispatches more cheaply. Reward = −(episode cost), nothing
else (decisions.md #2). Training seeds only (`seeds/ranges.json` `train`); `railway.train.env` refuses
others. Code: `railway/train/`, Modal entrypoints: `infra/modal/train.py`.

## Trainer choice: a small custom loop

- **Why not a library.** TRL's `GRPOTrainer`, verl, SkyRL, ART and prime-rl each bring their own rollout
  engine and agent loop. Our episodes must run through `railway/loop` (the benchmark loop) so the prompts
  are identical to the evaluation, and the model is a multimodal hybrid (`Qwen3_5ForConditionalGeneration`)
  whose support in those stacks we would have to prove first. Windowed prompts make every turn an
  independent, short sequence, so the trainer only needs "log-probs of sampled tokens given a prompt",
  which is about 100 lines. The custom loop is the faster way to a correct result here.
- **Sampler.** vLLM 0.21 (same flags as `serve_qwen.py`) with `--enable-lora`. Requests set
  `return_token_ids` and `logprobs`, so every turn records vLLM's exact prompt and completion token ids
  (no re-tokenisation drift) and the sampling log-probs.
- **Co-located, not the deployed endpoint.** vLLM runs as a subprocess on GPU 0 of the training container;
  the trainer holds the bf16 base + LoRA on the last GPU. A saved adapter is on local disk, so
  `POST /v1/load_lora_adapter` sees it at once (a separately deployed server would need a volume reload),
  and training never restarts the shared benchmark endpoint. Each update is loaded under a new name
  (`<run>-it<k>`) and the old one unloaded, so nothing stale is cached.
- **Update.** For each seed, G episodes with the current adapter; advantage A = (r − mean) / (std + ε)
  within the group (`std` or Dr. GRPO `none`), put on every assistant turn of the episode. Loss per
  token: −min(ρA, clip(ρ, 1±0.2)A) + β·KL₃(π‖base), summed and divided by the loss tokens in the batch.
  With one epoch per batch the step is on-policy (ρ = 1, REINFORCE with a group baseline); more epochs
  make the clip active. The base for KL is the same model with the adapter disabled (no second copy);
  β = 0 by default. vLLM's log-probs are used only to report the sampler/trainer mismatch.
- **SFT warm start.** Same LoRA, NLL on the completion tokens of good episodes (cheaper than greedy on
  their training seed, `sft_filter`). Samples with token ids train on them directly; samples without
  (teacher transcripts) are rendered with the chat template, with the teacher's reasoning as
  `reasoning_content`.

## Data formats

- **Trajectory JSONL** (`railway/train/samples.py`, one episode per line): episode id, env, seed, group,
  model, mode, status, cost, reward (= −cost), greedy cost, the loop's `EpisodeRecord` row, and
  `turns`. A **TurnSample** is: the exact `messages` and `tools` sent, `content`, `reasoning`,
  `tool_calls`, `finish_reason`, `prompt_token_ids`, `completion_token_ids`, `logprobs`, `advantage`.
- This is a stop-gap for SFT: lane F owns the episode transcript format (`railway/loop/transcripts.py`).
  SFT reads only through `load_sft_samples`, the one function to swap when F lands. GRPO keeps this
  format, because it needs the sampled token ids.

## Thinking tokens

The thinking template puts `<think>` in the generation prompt, so a completion is
`reasoning … </think> tool call <|im_end|>`. Earlier turns in the prompt carry only their tool calls
(the loop drops reasoning), which keeps prompts short. GRPO trains on the whole completion by default
(`think="train"`): the thinking is the policy's own sampled tokens and carries the decision.
`think="mask"` puts loss only after `</think>`. A turn cut off by `max_tokens` inside the thinking has no
tool call; it is a format error in the loop and is still trained with its episode's advantage. With
thinking off (`--no-thinking`) there is no `</think>` and every completion token carries loss.

## GPU and memory plan

- One container, 2× H200 (141 GB): GPU 0 vLLM (`--gpu-memory-utilization 0.9`, `--max-model-len 32768`),
  GPU 1 trainer: 27B bf16 weights ≈ 54 GB + LoRA r=16 on all attention/MLP projections + AdamW state
  (small) + activations with gradient checkpointing; logits only on completion positions
  (`logits_to_keep`), log-softmax in 1k-token chunks. One sequence at a time, gradient accumulation.
- Scale-up: more rollout GPUs (`RAILWAY_TRAIN_GPU=H200:4` gives vLLM 3 GPUs with tensor parallel; data
  parallel replicas are the better next step), one trainer GPU is enough for LoRA.

## Throughput (measured in the smoke; see below)

MEASURED_PLACEHOLDER

## Commands

    modal run infra/modal/download_qwen.py                     # once: weights into huggingface-cache
    modal run infra/modal/train.py::smoke                      # thinking smoke, 2 seeds x G=4, 10 turns
    modal run infra/modal/train.py::smoke --no-thinking
    modal run infra/modal/train.py::sft --data /data/<run>/base.jsonl --steps 200
    modal run infra/modal/train.py::grpo --iterations 10 --seeds-per-iter 4 --group 8 --start-adapter /adapters/<run>/sft

Outputs: `/data/<run>/` on the `railway-train` volume (trajectories, `report.json`, `history.json`),
`/adapters/<run>/` on `railway-adapters`.

## Switching to the network environment

`railway/train/env.py` has `NetEnv`, a marked slot. It needs: a network scenario per seed, the network
session (`call`, `closed`, `close() -> cost.total`, plus `peek` for the loop's CURRENT block), the network
tool schemas and the network greedy cost. `railway.loop.runner.play_episode` must accept the network
session (lane F). Then `--env net` (the `env_name` argument) is the only change in the drivers.
