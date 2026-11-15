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


@app.function(**{**FN, "volumes": EVAL_VOLUMES})
def sft_contextualized_remote(**kwargs) -> dict:
    vllm_gpus = _split_gpus()
    from railway.train import pipeline

    def commit_sft() -> None:
        adapters.commit()
        data.commit()

    def reload_sft() -> None:
        adapters.reload()
        data.reload()

    pipeline.log("GPUs:", pipeline.gpu_snapshot(), "| vLLM on", vllm_gpus)
    return pipeline.sft_contextualized_run(
        out_dir="/data",
        adapters_dir="/adapters",
        commit=commit_sft,
        reload_volumes=reload_sft,
        **kwargs,
    )


@app.function(image=image, gpu=GPU.split(":")[0], volumes=EVAL_VOLUMES, timeout=12 * HOURS)
def eval_shard_remote(**kwargs) -> list[dict]:
    from railway.train import evaluate, pipeline

    def commit_shard() -> None:
        data.commit()
        transcripts.commit()

    def reload_shard() -> None:
        adapters.reload()
        data.reload()
        scenarios.reload()

    return evaluate.eval_shard_vllm(
        out_dir="/data/eval",
        scenarios_dir="/scenarios",
        transcript_root="/transcripts",
        commit=commit_shard,
        reload_volumes=reload_shard,
        log=pipeline.log,
        **kwargs,
    )


@app.function(image=image, volumes=EVAL_VOLUMES, cpu=2, memory=4096, timeout=12 * HOURS, nonpreemptible=True)
def run_t1_comparison_remote(
    *,
    run: str,
    teacher_samples_blob: bytes,
    opus_test_rows_blob: bytes | None = None,
    val_shards: int = 3,
    test_shards: int = 5,
    sft_steps: int = 80,
    sft_batch: int = 16,
    sft_lr: float = 4e-5,
    think_max_tokens: int = 1024,
) -> dict:
    """Server-side detached coordinator: runs Qwen Pre (val+test) + SFT in parallel, then Qwen SFT (val+test)."""
    import json

    from railway.train import evaluate, pipeline

    data.reload()
    eval_root = Path("/data/eval") / run
    eval_root.mkdir(parents=True, exist_ok=True)

    val_entries = evaluate.select_eval_seeds("T1", "val")
    test_entries = evaluate.select_eval_seeds("T1", "test")
    val_parts = evaluate.shard_entries(val_entries, val_shards)
    test_parts = evaluate.shard_entries(test_entries, test_shards)

    pipeline.log(
        f"[orchestrator {run}] starting: sft (2x GPU) + pre_val ({len(val_parts)} shards) "
        f"+ pre_test ({len(test_parts)} shards)"
    )

    sft_call = sft_contextualized_remote.spawn(
        run=run,
        teacher_samples_blob=teacher_samples_blob,
        sft_steps=sft_steps,
        sft_batch=sft_batch,
        sft_lr=sft_lr,
        think_max_tokens=think_max_tokens,
    )
    pre_val_calls = [
        eval_shard_remote.spawn(
            run=run,
            shard_id=f"pre-val-s{i}",
            model_label="qwen38-think",
            adapter_path=None,
            seed_entries=part,
            thinking=True,
            concurrency=len(part),
        )
        for i, part in enumerate(val_parts)
    ]
    pre_test_calls = [
        eval_shard_remote.spawn(
            run=run,
            shard_id=f"pre-test-s{i}",
            model_label="qwen38-think",
            adapter_path=None,
            seed_entries=part,
            thinking=True,
            concurrency=len(part),
        )
        for i, part in enumerate(test_parts)
    ]

    # As soon as SFT finishes (~20 min), launch post-SFT val + test shards immediately!
    sft_res = sft_call.get()
    adapter_path = sft_res["adapter_path"]
    pipeline.log(f"[orchestrator {run}] SFT ready at {adapter_path}; spawning post-SFT val & test shards")

    post_test_parts = evaluate.shard_entries(test_entries, max(test_shards, 4))
    post_val_calls = [
        eval_shard_remote.spawn(
            run=run,
            shard_id=f"sft-val-s{i}",
            model_label="qwen38-think-sft",
            adapter_path=adapter_path,
            seed_entries=part,
            thinking=True,
            concurrency=len(part),
        )
        for i, part in enumerate(val_parts)
    ]
    post_test_calls = [
        eval_shard_remote.spawn(
            run=run,
            shard_id=f"sft-test-s{i}",
            model_label="qwen38-think-sft",
            adapter_path=adapter_path,
            seed_entries=part,
            thinking=True,
            concurrency=len(part),
        )
        for i, part in enumerate(post_test_parts)
    ]

    val_rows = [r for c in (*pre_val_calls, *post_val_calls) for r in c.get()]
    val_dir = eval_root / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    val_rows_path = val_dir / "rows.jsonl"
    val_rows_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in val_rows), encoding="utf-8")
    val_bundle = evaluate.build_eval_report(
        run_name=f"{run}-val",
        row_paths=[val_rows_path],
        base_model="qwen38-think",
        out_dir=val_dir,
    )
    data.commit()
    pipeline.log(f"[orchestrator {run}] VAL DONE:\n{val_bundle['markdown']}")

    test_rows = [r for c in (*pre_test_calls, *post_test_calls) for r in c.get()]
    test_dir = eval_root / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_rows_path = test_dir / "rows.jsonl"
    test_rows_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in test_rows), encoding="utf-8")
    row_paths = [test_rows_path]
    if opus_test_rows_blob:
        opus_path = test_dir / "opus_rows.jsonl"
        opus_path.write_bytes(opus_test_rows_blob)
        row_paths.append(opus_path)
    test_bundle = evaluate.build_eval_report(
        run_name=f"{run}-test",
        row_paths=row_paths,
        base_model="qwen38-think",
        out_dir=test_dir,
    )
    status_doc = {
        "run": run,
        "status": "completed",
        "adapter_path": adapter_path,
        "sft": sft_res,
        "val_markdown": val_bundle["markdown"],
        "test_markdown": test_bundle["markdown"],
    }
    (eval_root / "status.json").write_text(json.dumps(status_doc, indent=2) + "\n", encoding="utf-8")
    data.commit()
    pipeline.log(f"[orchestrator {run}] TEST DONE:\n{test_bundle['markdown']}")
    return status_doc


@app.function(**{**FN, "volumes": EVAL_VOLUMES})
def train_t1_remote(**kwargs) -> dict:
    vllm_gpus = _split_gpus()
    from railway.train import pipeline

    def commit_all() -> None:
        adapters.commit()
        data.commit()
        transcripts.commit()

    pipeline.log("GPUs:", pipeline.gpu_snapshot(), "| vLLM on", vllm_gpus)
    return pipeline.train_t1_pipeline(
        out_dir="/data",
        adapters_dir="/adapters",
        scenarios_dir="/scenarios",
        transcript_root="/transcripts",
        commit=commit_all,
        **kwargs,
    )


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


@app.local_entrypoint()
def train_t1(
    samples: str,
    base_val_rows: str = "",
    run_base_val_if_missing: bool = False,
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
    run: str = "",
    out_dir: str = "results/eval",
):
    """Run full T1 `thinking=True` pipeline (contextualized SFT -> val -> GRPO -> val selection -> T1 test eval)."""
    import json

    resolved_run = run or _run_name("t1-think-train")
    samples_blob = Path(samples).read_bytes()
    val_blob = Path(base_val_rows).read_bytes() if base_val_rows and Path(base_val_rows).is_file() else None
    rep = train_t1_remote.remote(
        run=resolved_run,
        teacher_samples_blob=samples_blob,
        base_val_rows_blob=val_blob,
        run_base_val_if_missing=run_base_val_if_missing,
        max_sft_samples=max_sft_samples,
        think_max_tokens=think_max_tokens,
        sft_steps=sft_steps,
        sft_batch=sft_batch,
        sft_lr=sft_lr,
        grpo_iterations=grpo_iterations,
        grpo_seeds_per_iter=grpo_seeds_per_iter,
        grpo_group=grpo_group,
        grpo_lr=grpo_lr,
        grpo_beta=grpo_beta,
        grpo_max_turns=grpo_max_turns,
        rollout_max_tokens=rollout_max_tokens,
        teacher_seed_offset=teacher_seed_offset,
        run_test_on_winner=run_test_on_winner,
    )
    if out_dir:
        local = REPO / out_dir / resolved_run
        local.mkdir(parents=True, exist_ok=True)
        if rep.get("val_all_rows_jsonl"):
            (local / "val_all_rows.jsonl").write_text(rep["val_all_rows_jsonl"], encoding="utf-8", newline="\n")
        if rep.get("test_tuned_rows_jsonl"):
            (local / "test_tuned_rows.jsonl").write_text(rep["test_tuned_rows_jsonl"], encoding="utf-8", newline="\n")
        if rep.get("val_markdown"):
            (local / "val_report.md").write_text(rep["val_markdown"], encoding="utf-8")
        meta = {k: v for k, v in rep.items() if not k.endswith("_jsonl")}
        (local / "pipeline_report.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Saved T1 training & evaluation artifacts -> {local}")
    if rep.get("val_markdown"):
        print(rep["val_markdown"])
    print("Winner:", json.dumps(rep.get("val_selection", {}).get("winner"), indent=2))


@app.local_entrypoint()
def launch_t1_comparison(
    samples: str = "data/teacher/t1-clean/samples.jsonl",
    opus_rows: str = "results/eval/opus55-t1-test/rows.jsonl",
    val_shards: int = 3,
    test_shards: int = 5,
    sft_steps: int = 80,
    sft_batch: int = 16,
    sft_lr: float = 4e-5,
    think_max_tokens: int = 1024,
    run: str = "t1-cmp-v1",
):
    """Launch detached server-side T1 comparison (`modal run --detach infra/modal/train.py::launch_t1_comparison`)."""
    s_path = Path(samples) if Path(samples).is_file() else (REPO / samples)
    o_path = Path(opus_rows) if Path(opus_rows).is_file() else (REPO / opus_rows)
    call = run_t1_comparison_remote.spawn(
        run=run,
        teacher_samples_blob=s_path.read_bytes(),
        opus_test_rows_blob=o_path.read_bytes() if o_path.is_file() else None,
        val_shards=val_shards,
        test_shards=test_shards,
        sft_steps=sft_steps,
        sft_batch=sft_batch,
        sft_lr=sft_lr,
        think_max_tokens=think_max_tokens,
    )
    print(f"Spawned cloud T1 comparison coordinator ({run}): {call.object_id}", flush=True)
