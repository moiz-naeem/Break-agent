"""Live provider smoke tests (`pytest -m model tests/loop/test_live_smoke.py`).

Plays a short multi-turn T1 episode (`max_turns=4`) against each configured benchmark provider on
`railway.tools.session.Session`, verifying multi-turn tool use with thinking on, saved transcript
bundles, and `(prompt, reasoning, completion)` training-sample reconstruction.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import httpx
import pytest

from railway.loop.providers import BENCHMARK_MODELS, make_benchmark_client
from railway.loop.runner import play_episode
from railway.loop.transcripts import load_training_samples, load_transcript

pytestmark = pytest.mark.model

REPO_ROOT = Path(__file__).resolve().parents[2]


def _warm_modal_if_needed(url: str, timeout_s: float = 1200) -> None:
    if ".modal.direct" not in url:
        return
    health = re.sub(r"/v1/?$", "", url.rstrip("/")) + "/health"
    t0 = time.monotonic()
    while True:
        try:
            if httpx.get(health, timeout=30).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError(f"Modal vLLM endpoint {health} not healthy after {timeout_s:.0f}s")
        time.sleep(10)


def _t1_seed5():
    from railway.gen.generator import generate
    from railway.gen.presets import PRESETS

    listing = json.loads((REPO_ROOT / "seeds" / "T1-dev.json").read_text(encoding="utf-8"))
    acc = next(a for a in listing["accepted"] if a["seed"] == 5)
    return generate(PRESETS["T1"], 5), float(acc["c_greedy"]), float(acc["c_ref"])


@pytest.mark.parametrize("preset_id", ["fable-high", "qwen38-think", "gemini-carbon-high", "nemotron-super"])
def test_live_provider_smoke_captures_reasoning(preset_id: str, tmp_path: Path):
    cfg = BENCHMARK_MODELS[preset_id]
    key_env = cfg["api_key_env"]
    if not os.environ.get(key_env):
        pytest.skip(f"{key_env} not set in environment")

    _warm_modal_if_needed(cfg["base_url"])
    scenario, c_greedy, c_ref = _t1_seed5()

    client = make_benchmark_client(preset_id)
    try:
        rec = play_episode(
            scenario,
            client,
            "windowed",
            max_turns=4,
            transcript_root=tmp_path / "transcripts",
            seed_set="dev",
            run="pytest-live-smoke",
            model=preset_id,
            rep=0,
            c_greedy=c_greedy,
            c_ref=c_ref,
        )
    finally:
        client.close()

    assert rec.status == "ok", rec.error
    assert rec.transcript_dir is not None
    bundle_dir = Path(rec.transcript_dir)
    assert (bundle_dir / "transcript.jsonl").is_file()
    assert (bundle_dir / "actions.json").is_file()
    assert (bundle_dir / "result.json").is_file()

    loaded = load_transcript(bundle_dir)
    assert len(loaded.turns) >= 2
    assert any(isinstance(t.get("reasoning"), str) and len(t["reasoning"].strip()) > 0 for t in loaded.turns)

    samples = load_training_samples(bundle_dir)
    assert len(samples) >= 1
    prompt, reasoning, completion = samples[0]
    assert prompt["messages"] and prompt["tools"]
    assert completion["role"] == "assistant"
