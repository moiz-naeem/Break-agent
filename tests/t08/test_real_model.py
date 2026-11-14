"""T08-M1: one real-model episode (marker `model`; optional for the T08 gate)."""

from __future__ import annotations

import os

import pytest

from railway.loop.model_client import ModelClient
from railway.loop.runner import play_episode

pytestmark = pytest.mark.model


def test_T08_M1_real_model_smoke():
    base_url = os.environ.get("MODEL_SMOKE_BASE_URL")
    name = os.environ.get("MODEL_SMOKE_NAME")
    if not (base_url and name):
        pytest.skip("set MODEL_SMOKE_BASE_URL, MODEL_SMOKE_NAME (and MODEL_SMOKE_KEY_ENV naming the key's env var)")
    from railway.eval.run import episodes_from_scenarios, reward_from_t07

    ep = episodes_from_scenarios("T1", "dev", 1)[0]
    client = ModelClient(base_url, name, os.environ.get("MODEL_SMOKE_KEY_ENV"))
    rec = play_episode(ep.scenario, client, "windowed", c_greedy=ep.c_greedy, c_best=ep.c_best, score_fn=reward_from_t07)
    assert rec.status == "ok", rec.error
    assert rec.tool_calls >= 5
    assert 0.0 <= rec.score_clipped <= 1.0
