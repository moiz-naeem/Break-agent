"""Training on Modal: vLLM sampler + LoRA trainer in one container (railway/train/pipeline.py).

    modal run infra/modal/train.py::smoke                       # thinking, 2 seeds x G=4, 10 turns
    modal run infra/modal/train.py::smoke --no-thinking         # same with thinking off
    modal run infra/modal/train.py::grpo --iterations 10 --seeds-per-iter 4 --group 8
        [--start-adapter /adapters/<run>/sft]
    modal run infra/modal/train.py::sft --data /data/<run>/base.jsonl --steps 200

GPU 0 runs vLLM (the same flags as serve_qwen.py), the last GPU trains. Adapters go to the
`railway-adapters` volume (/adapters/<run>/...), trajectories and reports to `railway-train` (/data/<run>/).
Weights come from the `huggingface-cache` volume (fill it once with download_qwen.py).
Set RAILWAY_TRAIN_GPU (default H200:2) at launch to change the GPUs, e.g. B200:2.
"""

import os
import time
from pathlib import Path

import modal

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else Path("/root")  # inside the container: /root/train.py
GPU = os.environ.get("RAILWAY_TRAIN_GPU", "H200:2")
HOURS = 60 * 60

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.21.0",
        "transformers==5.8.1",
        "peft",
        "accelerate",
        "huggingface_hub",
        "httpx",
        "pydantic",
        "numpy",
        "pyyaml",
        "mcp",
        "matplotlib",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONPATH": "/root",
        }
    )
    .add_local_dir(REPO / "seeds", "/root/seeds")
    .add_local_dir(REPO / "railway", "/root/railway", ignore=["**/__pycache__/**", "**/*.pyc"])
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
adapters = modal.Volume.from_name("railway-adapters", create_if_missing=True)
data = modal.Volume.from_name("railway-train", create_if_missing=True)
scenarios = modal.Volume.from_name("railway-scenarios", create_if_missing=True)
transcripts = modal.Volume.from_name("railway-transcripts", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache, "/adapters": adapters, "/data": data}
EVAL_VOLUMES = {**VOLUMES, "/scenarios": scenarios, "/transcripts": transcripts}

app = modal.App("railway-train")


def _split_gpus() -> str:
    """vLLM gets every GPU but the last; this process (the trainer) keeps the last. Call before torch loads."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = visible.split(",") if visible else [str(i) for i in range(int(GPU.split(":")[-1]) if ":" in GPU else 1)]
    if len(ids) < 2:
        raise RuntimeError("training needs at least 2 GPUs: one for vLLM, one for the trainer")
    os.environ["CUDA_VISIBLE_DEVICES"] = ids[-1]
    os.environ["RAILWAY_VLLM_GPUS"] = ",".join(ids[:-1])
    return os.environ["RAILWAY_VLLM_GPUS"]


def _commit() -> None:
    adapters.commit()
    data.commit()


def _run_name(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


FN = dict(image=image, gpu=GPU, volumes=VOLUMES, timeout=12 * HOURS)


@app.function(**FN)
def smoke_remote(**kwargs) -> dict:
    vllm_gpus = _split_gpus()
    from railway.train import pipeline

    pipeline.log("GPUs:", pipeline.gpu_snapshot(), "| vLLM on", vllm_gpus)
    return pipeline.smoke(out_dir="/data", adapters_dir="/adapters", commit=_commit, **kwargs)


@app.function(**FN)
def grpo_remote(**kwargs) -> dict:
    _split_gpus()
    from railway.train import pipeline

    return pipeline.grpo_run(out_dir="/data", adapters_dir="/adapters", commit=_commit, **kwargs)


@app.function(image=image, gpu=GPU.split(":")[0], volumes=VOLUMES, timeout=12 * HOURS)
def sft_remote(**kwargs) -> dict:
    from railway.train import pipeline

    return pipeline.sft_run(adapters_dir="/adapters", out_dir="/data", commit=_commit, **kwargs)


@app.function(image=image, gpu=GPU.split(":")[0], volumes=EVAL_VOLUMES, timeout=12 * HOURS)
def eval_remote(**kwargs) -> dict:
    from railway.train import evaluate, pipeline

    def commit_eval() -> None:
        data.commit()
        transcripts.commit()

    return evaluate.eval_vllm_run(
        out_dir="/data/eval",
        scenarios_dir="/scenarios",
        transcript_root="/transcripts",
        commit=commit_eval,
        log=pipeline.log,
        **kwargs,
    )


def _print(report: dict) -> None:
    import json

    print(json.dumps(report, indent=2)[:20000])


@app.local_entrypoint()
def smoke(
    thinking: bool = True,
    group: int = 4,
    max_turns: int = 10,
    max_tokens: int = 4096,
    final_group: int = 4,
    seeds: str = "10000,10001",
    run: str = "",
):
    run = run or _run_name("smoke-think" if thinking else "smoke-nothink")
    _print(
        smoke_remote.remote(
            run=run,
            thinking=thinking,
            group=group,
            max_turns=max_turns,
            max_tokens=max_tokens,
            final_group=final_group,
            seeds=tuple(int(s) for s in seeds.split(",")),
        )
    )


@app.local_entrypoint()
def grpo(
    iterations: int = 10,
    seeds_per_iter: int = 4,
    group: int = 8,
    max_turns: int = 0,
    thinking: bool = True,
    max_tokens: int = 8192,
    start_adapter: str = "",
    run: str = "",
):
    _print(
        grpo_remote.remote(
            run=run or _run_name("grpo"),
            iterations=iterations,
            seeds_per_iter=seeds_per_iter,
            group=group,
            max_turns=max_turns or None,
            thinking=thinking,
            max_tokens=max_tokens,
            start_adapter=start_adapter or None,
        )
    )


@app.local_entrypoint()
def sft(
    data: str,
    steps: int = 200,
    batch: int = 16,
    lr: float = 1e-4,
    think: str = "train",
    thinking: bool = False,
    run: str = "",
):
    """`--data` paths (local files are uploaded automatically; `/data/...` read from volume). Thinking is off by
    default: teacher samples carry no reasoning and the tuned model is served as `qwen38-nothink`."""
    items = [x.strip() for x in data.split(",") if x.strip()]
    uploaded = {p: Path(p).read_bytes() for p in items if Path(p).is_file()}
    _print(
        sft_remote.remote(
            data=items,
            uploaded_blobs=uploaded,
            run=run or _run_name("sft"),
            steps=steps,
            batch=batch,
            lr=lr,
            think=think,
            thinking=thinking,
        )
    )


@app.function(image=image, volumes=VOLUMES, timeout=HOURS, cpu=4, memory=16384)
def check_render_remote(rows: list[dict], thinking: bool = False) -> dict:
    """Render teacher samples with Qwen 3.8 27B's chat template and parse each completion back with vLLM's
    `qwen3_coder` tool parser (the one `serve_qwen.py` runs); every call must come back unchanged."""
    import json

    from railway.train.policy import load_tokenizer
    from railway.train.samples import TurnSample
    from railway.train.sft import render_sample

    try:
        from vllm.tool_parsers import ToolParserManager
    except ImportError:
        from vllm.entrypoints.openai.tool_parsers import ToolParserManager
    try:
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    except ImportError:
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest

    tok = load_tokenizer()
    parser_cls = ToolParserManager.get_tool_parser("qwen3_coder")
    results, failures = [], []
    for row in rows:
        s = TurnSample(**row)
        r = render_sample(tok, s, thinking=thinking)
        prompt_text = tok.decode(r.prompt_token_ids, skip_special_tokens=False)
        text = tok.decode(r.completion_token_ids, skip_special_tokens=False).replace("<|im_end|>", "")
        req = ChatCompletionRequest(model="qwen38", messages=[{"role": "user", "content": "x"}], tools=s.tools)
        # vLLM 0.21: the parser gets the request's tools in its constructor (that is how the server builds it);
        # without them it cannot convert parameter types and returns every value as a string.
        info = parser_cls(tok, req.tools).extract_tool_calls(text, request=req)
        want = s.tool_calls[0]
        got = [(c.function.name, json.loads(c.function.arguments or "{}")) for c in info.tool_calls]
        ok = got == [(want["name"], json.loads(want["arguments_raw"] or "{}"))]
        item = {
            "tool": want["name"],
            "ok": ok,
            "completion": text[-400:],
            "prompt_tokens": len(r.prompt_token_ids),
            "completion_tokens": len(r.completion_token_ids),
            "think_block_in_prompt_tail": "<think>" in prompt_text[-200:],
            "think_in_completion": "<think>" in text or "</think>" in text,
        }
        results.append(item)
        if not ok:
            failures.append({**item, "got": got, "want": want})
    return {"n": len(rows), "ok": sum(r["ok"] for r in results), "failures": failures[:5], "examples": results[:3]}


@app.local_entrypoint()
def check_render(samples: str, n: int = 20, thinking: bool = False):
    """Round-trip check on the first `n` samples of a local teacher samples.jsonl (spread over the tools)."""
    import json

    rows = [json.loads(line) for line in Path(samples).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_tool: dict[str, list[dict]] = {}
    for row in rows:
        by_tool.setdefault(row["tool_calls"][0]["name"], []).append(row)
    picked: list[dict] = []
    while len(picked) < min(n, len(rows)):
        for group in by_tool.values():
            if group and len(picked) < n:
                picked.append(group.pop(0))
    _print(check_render_remote.remote(picked, thinking=thinking))


@app.local_entrypoint()
def eval_models(
    candidates: str = "base",
    tiers: str = "T1,T2,T3",
    split: str = "val",
    seeds: int = 0,
    reps: int = 1,
    thinking: bool = False,
    temperature: float = -1.0,
    max_tokens: int = 32768,
    max_model_len: int = 65536,
    concurrency: int = 16,
    max_turns: int = 0,
    base_model: str = "",
    extra_rows: str = "",
    run: str = "",
    out_dir: str = "results/eval",
):
    """Evaluate base Qwen and/or LoRA checkpoints (`label=/adapters/<run>/sft`) on `val`, `dev`, `test`, or `train`."""
    import json

    resolved_run = run or _run_name(f"eval-{split}")
    cand_list = [x.strip() for x in candidates.split(",") if x.strip()]
    tier_list = [x.strip() for x in tiers.split(",") if x.strip()]
    extra_blobs = {
        p: Path(p).read_bytes()
        for p in (x.strip() for x in extra_rows.split(",") if x.strip())
        if Path(p).is_file()
    }
    bundle = eval_remote.remote(
        run=resolved_run,
        candidates=cand_list,
        tiers=tier_list,
        split=split,
        seeds_limit=seeds,
        reps=reps,
        thinking=thinking,
        temperature=None if temperature < 0 else temperature,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        concurrency=concurrency,
        max_turns=max_turns or None,
        base_model=base_model or None,
        extra_rows_blobs=extra_blobs,
    )
    if out_dir:
        local = REPO / out_dir / resolved_run
        local.mkdir(parents=True, exist_ok=True)
        if bundle.get("rows_jsonl"):
            (local / "rows.jsonl").write_text(bundle["rows_jsonl"], encoding="utf-8", newline="\n")
        summary_json = json.dumps(bundle["summary"], indent=2, sort_keys=True) + "\n"
        paired_json = json.dumps(bundle["paired"], indent=2, sort_keys=True) + "\n"
        (local / "summary.json").write_text(summary_json, encoding="utf-8")
        (local / "paired.json").write_text(paired_json, encoding="utf-8")
        (local / "report.md").write_text(bundle["markdown"], encoding="utf-8")
        print(f"Saved local evaluation artifacts -> {local}")
    print(bundle["markdown"])
