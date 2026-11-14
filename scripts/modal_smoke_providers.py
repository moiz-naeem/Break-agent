"""Run 1 short live episode per benchmark provider on Modal, writing to the `railway-transcripts` volume.

Usage:
    PYTHONIOENCODING=utf-8 modal run scripts/modal_smoke_providers.py
    PYTHONIOENCODING=utf-8 modal run scripts/modal_smoke_providers.py --only "fable-high qwen38-think"

API keys are read by environment variable NAME only (from `os.environ` or the repo's git-ignored `.env`)
and passed exclusively via encrypted `modal.Secret` objects (plus the `vllm-key` Modal secret).
Keys are never passed as function arguments, printed, logged, or written to disk.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
IGNORE = [
    ".git",
    ".venv",
    "jobs",
    "artifacts",
    "data",
    "**/__pycache__",
    ".hypothesis",
    ".pytest_cache",
    ".ruff_cache",
    ".env*",
]

VOLUME_NAME = "railway-transcripts"
MOUNT_PATH = "/transcripts"
PROVIDER_KEY_NAMES = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY", "VERDA_INFERENCE_KEY")


def _load_local_env_by_name(names: tuple[str, ...]) -> dict[str, str]:
    """Read requested key names from `os.environ` or git-ignored `.env` (worktree or common git dir)."""
    if not modal.is_local():
        return {}
    found: dict[str, str] = {}
    for name in names:
        if os.environ.get(name):
            found[name] = os.environ[name]

    candidates = [ROOT / ".env"]
    try:
        common = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        if common:
            candidates.append((ROOT / common).resolve().parent / ".env")
    except Exception:  # noqa: BLE001
        pass

    for env_path in candidates:
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line.strip())
            if m and m.group(1) in names and m.group(1) not in found:
                val = m.group(2).strip().strip('"').strip("'")
                if val:
                    found[m.group(1)] = val
    return found


image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(ROOT), groups=["dev"])
    .add_local_dir(str(ROOT), "/repo", ignore=IGNORE)
)
transcripts_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
_SECRETS = [
    modal.Secret.from_name("vllm-key"),
    modal.Secret.from_dict(_load_local_env_by_name(PROVIDER_KEY_NAMES)),
]
app = modal.App("railway-provider-smoke", image=image)


@app.function(
    cpu=1.0,
    timeout=1800,
    volumes={MOUNT_PATH: transcripts_vol},
    secrets=_SECRETS,
)
def run_one_provider(preset_id: str, max_turns: int = 4, run_name: str = "smoke-laneF") -> dict:
    import sys

    import httpx

    sys.path.insert(0, "/repo")
    from railway.gen.generator import generate
    from railway.gen.presets import PRESETS
    from railway.loop.providers import BENCHMARK_MODELS, make_benchmark_client
    from railway.loop.runner import play_episode
    from railway.loop.transcripts import load_training_samples, load_transcript

    cfg = BENCHMARK_MODELS[preset_id]
    url = cfg["base_url"]

    # Wake Modal scale-to-zero vLLM server (`qwen38`) before starting the episode timer.
    warm_s = 0.0
    if ".modal.direct" in url:
        health = re.sub(r"/v1/?$", "", url.rstrip("/")) + "/health"
        t0 = time.monotonic()
        while True:
            try:
                if httpx.get(health, timeout=30).status_code == 200:
                    warm_s = round(time.monotonic() - t0, 1)
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() - t0 > 1200:
                return {"preset_id": preset_id, "status": "error", "error": f"health check timed out: {health}"}
            time.sleep(10)

    listing = json.loads((Path("/repo") / "seeds" / "T1-dev.json").read_text(encoding="utf-8"))
    acc = next(a for a in listing["accepted"] if a["seed"] == 5)
    scenario = generate(PRESETS["T1"], 5)

    client = make_benchmark_client(preset_id)
    try:
        rec = play_episode(
            scenario,
            client,
            "windowed",
            max_turns=max_turns,
            transcript_root=MOUNT_PATH,
            seed_set="dev",
            run=run_name,
            model=preset_id,
            rep=0,
            c_greedy=float(acc["c_greedy"]),
            c_ref=float(acc["c_ref"]),
        )
    finally:
        client.close()

    transcripts_vol.commit()

    bundle_dir = Path(rec.transcript_dir)
    files_written = sorted(p.name for p in bundle_dir.iterdir() if p.is_file())
    loaded = load_transcript(bundle_dir)
    samples = load_training_samples(bundle_dir, skip_format_errors=False)
    clean_samples = load_training_samples(bundle_dir, skip_format_errors=True)

    reasoning_chars = [len(t.get("reasoning") or "") for t in loaded.turns]
    turns_with_reasoning = sum(1 for c in reasoning_chars if c > 0)
    first_reasoning = next((t["reasoning"] for t in loaded.turns if t.get("reasoning")), None)
    preview = (
        first_reasoning[:160].replace("\n", " ") + ("..." if len(first_reasoning) > 160 else "")
        if first_reasoning
        else None
    )

    per_turn_usage = [
        {
            "turn": t["turn"],
            "executed_tool": (t.get("executed") or {}).get("tool"),
            "format_error": t["format_error"],
            "finish_reason": t.get("finish_reason"),
            "reasoning_chars": len(t.get("reasoning") or ""),
            **t["usage"],
        }
        for t in loaded.turns
    ]
    n_turns = len(loaded.turns)
    total_usd = float((loaded.footer or {}).get("usage_total", {}).get("cost_usd") or 0.0)

    return {
        "preset_id": preset_id,
        "provider": loaded.header["provider"],
        "model_name": loaded.header["model_name"],
        "status": rec.status,
        "error": rec.error,
        "warm_s": warm_s,
        "wall_s": rec.wall_s,
        "volume_dir": bundle_dir.as_posix(),
        "transcript_bytes": (bundle_dir / "transcript.jsonl").stat().st_size,
        "files_written": files_written,
        "turns": n_turns,
        "tool_calls": rec.tool_calls,
        "format_errors": rec.format_errors,
        "reasoning_captured": turns_with_reasoning > 0,
        "turns_with_reasoning": turns_with_reasoning,
        "reasoning_chars_per_turn": reasoning_chars,
        "reasoning_preview": preview,
        "sft_samples_clean": len(clean_samples),
        "sft_samples_all": len(samples),
        "sample_prompt_messages_turn1": len(samples[0].messages) if samples else 0,
        "sample_prompt_chars_turn1": len(json.dumps(samples[0].messages)) if samples else 0,
        "sample_tools_count": len(samples[0].tools) if samples else 0,
        "usage_total": (loaded.footer or {}).get("usage_total"),
        "cost_usd_total": round(total_usd, 6),
        "mean_cost_usd_per_turn": round(total_usd / n_turns, 6) if n_turns else 0.0,
        "per_turn_usage": per_turn_usage,
    }


@app.local_entrypoint()
def main(only: str = "", max_turns: int = 4, run_name: str = "smoke-laneF"):
    import sys

    sys.path.insert(0, str(ROOT))
    from railway.loop.providers import BENCHMARK_MODELS

    wanted = only.split() if only.strip() else list(BENCHMARK_MODELS)
    inputs = [(preset_id, max_turns, run_name) for preset_id in wanted]
    results = list(run_one_provider.starmap(inputs))
    print(json.dumps(results, indent=2, sort_keys=True))
    failed = [r for r in results if r.get("status") != "ok" or not r.get("reasoning_captured")]
    if failed:
        raise SystemExit(f"{len(failed)} smoke run(s) failed or lacked reasoning")
