import json

import pytest

from railway.loop.fake_model import FakeModelServer, Scripted, _completion
from railway.train.env import NetEnv, StandInEnv, is_train_seed, make_env
from railway.train.rollout import CaptureClient, rollout_groups, run_episode, summarize
from railway.train.samples import read_jsonl, write_jsonl

SEED = 10000


def test_train_seed_guard():
    assert is_train_seed(SEED) and not is_train_seed(5) and not is_train_seed(1_000_000)
    with pytest.raises(ValueError):
        StandInEnv().reset(5)
    assert StandInEnv(allow_any_seed=True).reset(5).seed == 5


def test_standin_step_done_reward():
    env = make_env("standin")
    ep = env.reset(SEED)
    names = {t["function"]["name"] for t in ep.tools}
    assert {"authorize", "advance_to_next_event", "end_shift"} <= names
    assert not ep.done
    out = ep.step("advance_to_next_event", {})
    assert isinstance(out, str) and out
    assert "REFUSED" in ep.step("authorize", {"train": "NOPE", "station": "S1", "track": "L1"})
    ep.step("end_shift", {})
    assert ep.done
    reward = ep.close()
    assert reward == -ep.session.final.cost.total == ep.reward
    assert reward < 0


def test_standin_is_deterministic_and_greedy_is_cheaper_than_nothing():
    env = StandInEnv()
    a, b = env.reset(SEED), env.reset(SEED)
    assert a.step("advance_to_next_event", {}) == b.step("advance_to_next_event", {})
    nothing = -env.reset(SEED).close()
    assert env.greedy_cost(SEED) < nothing


def test_net_env_splits_and_reset():
    from railway.train.env import is_net_train_seed, is_val_seed

    env = NetEnv()
    for bad in (1, 10_001, 20_000):  # dev, test (held out), a validation seed
        with pytest.raises(ValueError):
            env.reset(bad)
    assert is_val_seed(20_000) and not is_net_train_seed(20_000) and is_net_train_seed(20_001)
    ep = env.reset(20_001)
    assert not ep.done and {t["function"]["name"] for t in ep.tools} >= {"order", "advance_to_next_event"}
    assert ep.step("status", {"compact": True})
    assert isinstance(ep.close(), float) and env.greedy_cost(20_001) is None


class WithTokens(Scripted):
    """Scripted replies that also carry vLLM's return_token_ids / logprobs fields."""

    def respond(self, body, n):
        assert body["return_token_ids"] is True and body["logprobs"] is True
        status, data = super().respond(body, n)
        data["prompt_token_ids"] = [1, 2, n]
        data["choices"][0]["token_ids"] = [7, 8, 9]
        data["choices"][0]["logprobs"] = {"content": [{"token": "t", "logprob": -0.25}] * 3}
        data["choices"][0]["message"]["reasoning_content"] = f"thinking {n}"
        return status, data


ADV = {"name": "advance_to_next_event", "arguments": {}}


def test_run_episode_captures_every_turn(tmp_path):
    env = StandInEnv()
    with FakeModelServer(WithTokens([ADV, ADV, None, {"name": "end_shift", "arguments": {}}])) as srv:
        traj = run_episode(env, SEED, lambda: CaptureClient(srv.base_url, "fake"), group="g", rep=0, max_turns=10)
        sent = [r["messages"] for r in srv.requests]
    assert traj.status == "ok" and traj.turns_played == 4 == len(sent)
    assert [s.messages for s in traj.turns] == sent  # exactly what the model was sent
    assert [len(s.tool_calls) for s in traj.turns] == [1, 1, 0, 1]
    assert traj.turns[0].prompt_token_ids == [1, 2, 1] and traj.turns[3].completion_token_ids == [7, 8, 9]
    assert traj.turns[1].logprobs == [-0.25] * 3 and traj.turns[2].reasoning == "thinking 3"
    # the reward is minus the cost of the same actions replayed directly
    ep = env.reset(SEED)
    ep.step("advance_to_next_event", {})
    ep.step("advance_to_next_event", {})
    ep.step("end_shift", {})
    assert traj.reward == ep.close() == -traj.cost
    write_jsonl(tmp_path / "x.jsonl", [traj])
    assert read_jsonl(tmp_path / "x.jsonl")[0].to_json() == traj.to_json()


def test_turn_cap_ends_the_shift():
    with FakeModelServer(WithTokens([ADV] * 50)) as srv:
        traj = run_episode(
            StandInEnv(), SEED, lambda: CaptureClient(srv.base_url, "fake"), group="g", rep=0, max_turns=3
        )
    assert traj.turns_played == 3 and traj.ok


def test_server_without_token_ids_still_records():
    with FakeModelServer(Scripted([ADV])) as srv:
        traj = run_episode(
            StandInEnv(), SEED, lambda: CaptureClient(srv.base_url, "fake"), group="g", rep=0, max_turns=5
        )
    assert traj.ok and traj.turns[0].prompt_token_ids is None and not traj.turns[0].has_tokens


def test_rollout_groups_shapes():
    with FakeModelServer(_Fixed()) as srv:
        trajs = rollout_groups(
            StandInEnv(), [SEED, SEED + 1], 3, lambda: CaptureClient(srv.base_url, "fake"), tag="t", max_turns=2
        )
    assert sorted(t.episode_id for t in trajs) == sorted(f"s{s}-t-r{r}" for s in (SEED, SEED + 1) for r in range(3))
    assert all(t.greedy_cost is not None for t in trajs)
    summary = summarize(trajs)
    assert summary["episodes"] == 6 and summary["ok"] == 6
    json.dumps(summary)


class _Fixed:
    """Thread-safe: always `advance_to_next_event` (the server serializes calls anyway)."""

    def respond(self, body, n):
        return 200, _completion(body, [ADV], None, n)
