"""Teacher replay -> loop-format samples: exact replay, read pruning, labels, normalisation, filters."""

from __future__ import annotations

import json

import pytest

from railway.harbor.net_export import generate_timetable
from railway.loop.prompts import WINDOW_TURNS
from railway.net.session import Session
from railway.train.samples import TurnSample
from railway.train.teacher import (
    ADVANCE,
    ReplayMismatch,
    dedupe,
    normalize_args,
    prune_reads,
    read_samples_jsonl,
    rebalance,
    render_episode,
    select_episodes,
    state_at,
    TeacherEpisode,
    write_samples_jsonl,
)


@pytest.fixture(scope="module")
def tt():
    return generate_timetable("T1", 1, 180)


def _play(tt, calls):
    s = Session(tt)
    for tool, args in calls:
        if s.closed:
            break
        s.call(tool, args)
    return s.episode_log()


SCRIPT = [
    ("manual", {}),
    ("network", {}),
    ("status", {}),
    ("status", {}),  # repeat with no state change in between: pruned
    ("order", {"resource": "NO-SUCH-RESOURCE", "trains": ["X"]}),  # refused / invalid: not a target
    (ADVANCE, {}),
    ("status", {}),  # state changed since the last status: kept
    *[(ADVANCE, {})] * 12,
]


def test_replay_matches_and_prunes(tt):
    log = _play(tt, SCRIPT)
    r = render_episode(tt, log["entries"], expected_final_cost=log["final_cost"])
    assert r.pruned_reads == 2  # manual + the repeated status
    assert r.final_cost == log["final_cost"]
    tools = [s.meta["tool"] for s in r.samples]
    assert "manual" not in tools and tools[:3] == ["network", "status", "order"]
    bad = next(s for s in r.samples if s.meta["tool"] == "order")
    assert bad.meta["status"] in ("REFUSED", "INVALID") and not bad.meta["target"]
    assert all(s.meta["target"] for s in r.samples if s.meta["tool"] == ADVANCE)


def test_prompts_are_windowed_loop_prompts(tt):
    log = _play(tt, SCRIPT)
    r = render_episode(tt, log["entries"])
    last = r.samples[-1]
    roles = [m["role"] for m in last.messages]
    assert roles[:2] == ["system", "user"] and roles[-1] == "user"
    assert roles.count("assistant") == min(WINDOW_TURNS, len(r.samples) - 1)
    assert last.messages[-1]["content"].startswith("CURRENT STATE")
    assert last.messages[1]["content"].startswith("MANUAL\n")
    assert {t["function"]["name"] for t in last.tools} >= {"order", ADVANCE, "skip"}
    json.loads(last.tool_calls[0]["arguments_raw"])


def test_replay_mismatch_is_detected(tt):
    log = _play(tt, SCRIPT)
    entries = [dict(e) for e in log["entries"]]
    adv = next(e for e in entries if e["tool"] == ADVANCE)
    adv["tick"] = adv["tick"] + 999
    with pytest.raises(ReplayMismatch):
        render_episode(tt, entries, prune=False)
    with pytest.raises(ReplayMismatch):
        render_episode(tt, log["entries"], expected_final_cost=log["final_cost"] + 1)


def test_state_at_is_the_branch_point(tt):
    log = _play(tt, SCRIPT)
    k = 6
    s = state_at(tt, log["entries"], k)
    assert len(s.log) == k and s.log[-1]["tick"] == log["entries"][k - 1]["tick"]


def test_prune_reads_resets_after_state_change():
    e = lambda t: {"tool": t, "args": {}}  # noqa: E731
    kept, n = prune_reads([e("status"), e("status"), e(ADVANCE), e("status"), e("manual")])
    assert [x["tool"] for x in kept] == ["status", ADVANCE, "status"] and n == 2


def test_normalize_platform_to_station(tt):
    s = Session(tt)
    plat = tt.net.platforms[0]
    out = normalize_args(s, "skip", {"train": "X", "station": plat.id})
    assert out["station"] == plat.node
    assert normalize_args(s, "skip", {"train": "X", "station": plat.node})["station"] == plat.node


def _sample(tool, i):
    return TurnSample(turn=i, messages=[{"role": "user", "content": str(i)}], tools=[],
                      tool_calls=[{"id": "c", "name": tool, "arguments_raw": "{}"}], meta={"tool": tool})  # fmt: skip


def test_rebalance_and_dedupe(tmp_path):
    samples = [_sample(ADVANCE, i) for i in range(20)] + [_sample("order", 100 + i) for i in range(10)]
    out = rebalance(samples, max_share=0.25)
    adv = sum(s.tool_calls[0]["name"] == ADVANCE for s in out)
    assert adv / len(out) <= 0.25 + 1e-9 and len(out) - adv == 10
    assert len(dedupe(samples + samples)) == len(samples)
    p = tmp_path / "s.jsonl"
    write_samples_jsonl(p, out)
    back = read_samples_jsonl(p)
    assert [b.meta for b in back] == [s.meta for s in out]


def test_select_episodes_per_seed_and_baseline():
    ep = lambda seed, rep, cost, status="ok": TeacherEpisode("T1", seed, rep, "opus", "droid", status, cost, None, [{}])  # noqa: E731
    eps = [ep(1, 0, 50), ep(1, 1, 40), ep(2, 0, 90), ep(2, 1, 10, "error")]
    assert [(e.seed, e.cost) for e in select_episodes(eps)] == [(1, 40), (2, 90)]
    kept = select_episodes(eps, baseline_cost={("T1", 1): 45, ("T1", 2): 80})
    assert [(e.seed, e.cost) for e in kept] == [(1, 40)]
