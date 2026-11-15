import pytest

from railway.train.grpo import GRPOConfig, training_turns
from railway.train.samples import (
    IGNORE,
    Trajectory,
    TurnSample,
    completion_mask,
    load_sft_samples,
    read_jsonl,
    sft_filter,
    think_span,
    token_example,
    write_jsonl,
)

THINK_END = 99


def sample(turn=1, prompt=(1, 2, 3), completion=(10, 11, THINK_END, 20, 21), calls=1, adv=None):
    return TurnSample(
        turn=turn,
        messages=[{"role": "user", "content": "x"}],
        tools=[],
        tool_calls=[{"name": "advance_to_next_event", "arguments_raw": "{}"}] * calls,
        prompt_token_ids=list(prompt),
        completion_token_ids=list(completion),
        logprobs=[-0.5] * len(completion),
        advantage=adv,
    )


def traj(eid, cost, greedy, status="ok", turns=None):
    return Trajectory(
        episode_id=eid,
        env="standin-v02",
        seed=10000,
        group="g",
        model="m",
        mode="windowed",
        status=status,
        cost=cost,
        reward=None if cost is None else -cost,
        greedy_cost=greedy,
        turns=turns or [sample()],
    )


def test_jsonl_roundtrip(tmp_path):
    ts = [traj("a", 10.0, 20.0), traj("b", None, 20.0, status="error")]
    path = tmp_path / "t.jsonl"
    assert write_jsonl(path, ts) == 2
    back = read_jsonl(path)
    assert [t.to_json() for t in back] == [t.to_json() for t in ts]
    assert isinstance(back[0].turns[0], TurnSample)


def test_think_span():
    assert think_span([10, 11, THINK_END, 20], THINK_END) == 3
    assert think_span([10, 11], THINK_END) == 2  # truncated inside the thinking
    assert think_span([10, 11], None) == 0  # thinking off


def test_completion_mask_modes():
    comp = [10, 11, THINK_END, 20, 21]
    assert completion_mask(comp, THINK_END, "train") == [True] * 5
    assert completion_mask(comp, THINK_END, "mask") == [False, False, False, True, True]
    with pytest.raises(ValueError):
        completion_mask(comp, THINK_END, "drop")


def test_token_example_masks_prompt_and_thinking():
    ids, labels = token_example(sample(), THINK_END, "mask")
    assert ids == [1, 2, 3, 10, 11, THINK_END, 20, 21]
    assert labels == [IGNORE] * 6 + [20, 21]
    ids, labels = token_example(sample(), THINK_END, "train")
    assert labels == [IGNORE] * 3 + [10, 11, THINK_END, 20, 21]


def test_token_example_needs_tokens():
    s = sample()
    s.prompt_token_ids = None
    with pytest.raises(ValueError):
        token_example(s, THINK_END)


def test_sft_filter_beats_greedy(tmp_path):
    ts = [traj("good", 10.0, 20.0), traj("tie", 20.0, 20.0), traj("bad", 30.0, 20.0), traj("err", None, 20.0, "error")]
    assert [t.episode_id for t in sft_filter(ts)] == ["good"]
    assert [t.episode_id for t in sft_filter(ts, beat_greedy=False)] == ["good", "tie", "bad"]
    two_calls = traj("good2", 5.0, 20.0, turns=[sample(1), sample(2, calls=2), sample(3, calls=0)])
    path = tmp_path / "d.jsonl"
    write_jsonl(path, ts + [two_calls])
    got = list(load_sft_samples([path]))
    assert [s.turn for s in got] == [1, 1]  # good's turn and good2's single-call turn only


def test_training_turns_filters_and_subsamples():
    turns = [sample(i, adv=0.5) for i in range(1, 9)] + [sample(9, adv=0.0)]
    no_tok = sample(10, adv=0.5)
    no_tok.completion_token_ids = None
    t = traj("a", 1.0, 2.0, turns=turns + [no_tok])
    assert [s.turn for s in training_turns([t], GRPOConfig())] == list(range(1, 9))
    picked = training_turns([t], GRPOConfig(max_turns_per_episode=3, seed=1))
    assert len(picked) == 3 and [s.turn for s in picked] == sorted(s.turn for s in picked)
    assert training_turns([t], GRPOConfig(max_len=7)) == []  # 3 prompt + 5 completion tokens > 7


def test_load_sft_samples_reads_teacher_samples_jsonl(tmp_path):
    from railway.train.teacher import write_samples_jsonl

    s1 = sample(1, calls=1)
    s2 = sample(2, calls=2)  # multi-call should be skipped
    s3 = sample(3, calls=1)
    path = tmp_path / "samples.jsonl"
    write_samples_jsonl(path, [s1, s2, s3])
    got = list(load_sft_samples([path]))
    assert [s.turn for s in got] == [1, 3]


def test_populate_thinking_prefixes_and_stratified_subsample():
    from railway.train.sft import _step_lr, populate_thinking_prefixes, stratified_subsample

    raw = [
        TurnSample(
            turn=i,
            messages=[{"role": "user", "content": f"state {i}"}],
            tools=[],
            tool_calls=[{"name": "route" if i == 0 else "order", "arguments_raw": "{\"a\": 1}"}],
        )
        for i in range(10)
    ]
    sub = stratified_subsample(raw, 4, seed=0)
    assert len(sub) == 4
    # Rare tool `route` (1 instance) must be kept by round-robin stratification.
    assert any(s.tool_calls[0]["name"] == "route" for s in sub)

    class _FakeThinkClient:
        def chat(self, messages, tools):
            return {
                "reasoning": "check conflict at J1 and prioritize express",
                "content": None,
                "tool_calls": [{"id": "c1", "name": "order", "arguments_raw": "{\"a\":1}"}],
                "usage": {},
            }

        def close(self):
            pass

    enriched, stats = populate_thinking_prefixes(
        raw,
        "http://localhost:8001/v1",
        "qwen38",
        max_samples=5,
        concurrency=2,
        client_factory=_FakeThinkClient,
    )
    assert len(enriched) == 5
    assert stats["populated"] == 5 and stats["errors"] == 0
    assert all(s.reasoning == "check conflict at J1 and prioritize express" for s in enriched)
    assert _step_lr(0, 100, 1e-4, 0.05, 0.1) < 1e-4
    assert _step_lr(99, 100, 1e-4, 0.05, 0.1) == pytest.approx(1e-5, rel=1e-2)
