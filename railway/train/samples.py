"""Per-turn training samples and episode trajectories (the internal format).

This is a stop-gap until lane F's transcript reader (`railway/loop/transcripts.py`) lands. SFT reads
samples only through `load_sft_samples`, the single swap point. GRPO keeps this format regardless,
because it needs the exact sampled token ids, which transcripts don't promise.

Trajectory JSONL: one episode per line.
    {"episode_id", "env", "seed", "group", "model", "mode", "status", "error",
     "cost", "reward", "greedy_cost", "turns_played", "row": {...EpisodeRecord row...},
     "turns": [TurnSample, ...]}
TurnSample:
    {"turn", "messages", "tools", "content", "reasoning", "tool_calls", "finish_reason",
     "prompt_token_ids", "completion_token_ids", "logprobs", "advantage", "meta"}
`messages` is exactly what the model was sent; `prompt_token_ids` / `completion_token_ids` are what vLLM
tokenized and sampled (null from servers that don't return them). `advantage` is filled in by GRPO.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

IGNORE = -100
THINK_MODES = ("train", "mask")


@dataclass
class TurnSample:
    turn: int
    messages: list[dict]
    tools: list[dict]
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    advantage: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def has_tokens(self) -> bool:
        return bool(self.prompt_token_ids) and bool(self.completion_token_ids)


@dataclass
class Trajectory:
    episode_id: str
    env: str
    seed: int
    group: str
    model: str | None
    mode: str
    status: str
    cost: float | None
    reward: float | None
    error: str | None = None
    greedy_cost: float | None = None
    turns_played: int = 0
    row: dict = field(default_factory=dict)
    turns: list[TurnSample] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> Trajectory:
        data = dict(data)
        turns = [TurnSample(**t) for t in data.pop("turns", [])]
        return cls(**data, turns=turns)

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.reward is not None


def write_jsonl(path: Path, trajectories: Iterable[Trajectory]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for traj in trajectories:
            fh.write(json.dumps(traj.to_json(), ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[Trajectory]:
    with Path(path).open(encoding="utf-8") as fh:
        return [Trajectory.from_json(json.loads(line)) for line in fh if line.strip()]


def sft_filter(trajectories: Iterable[Trajectory], *, beat_greedy: bool = True) -> list[Trajectory]:
    """Episodes good enough to imitate: finished OK and (by default) cheaper than greedy on their seed."""
    keep = []
    for t in trajectories:
        if not t.ok:
            continue
        if beat_greedy and (t.greedy_cost is None or t.cost is None or t.cost >= t.greedy_cost):
            continue
        keep.append(t)
    return keep


def _first_json_object(path: Path) -> dict | None:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                return json.loads(line)
    return None


def load_sft_samples(paths: Iterable[Path], *, beat_greedy: bool = True) -> Iterator[TurnSample]:
    """Load single-tool-call `TurnSample`s from teacher `samples.jsonl`, rollout trajectories, or loop transcripts."""
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            first = {"type": "header"}
        else:
            first = _first_json_object(path)
            if first is None:
                continue
        if first.get("type") == "header":
            from railway.loop.transcripts import load_training_samples

            for ts in load_training_samples(path, beat_greedy_only=beat_greedy):
                calls = [
                    {
                        "id": tc.get("id", "call_1"),
                        "name": tc["function"]["name"],
                        "arguments_raw": tc["function"]["arguments"],
                    }
                    for tc in (ts.completion.get("tool_calls") or [])
                ]
                if len(calls) == 1:
                    yield TurnSample(
                        turn=int(ts.meta.get("turn") or 1),
                        messages=ts.prompt["messages"],
                        tools=ts.prompt["tools"],
                        content=ts.completion.get("content"),
                        reasoning=ts.reasoning,
                        tool_calls=calls,
                        meta=dict(ts.meta),
                    )
        elif "turns" in first:
            for traj in sft_filter(read_jsonl(path), beat_greedy=beat_greedy):
                for turn in traj.turns:
                    if len(turn.tool_calls) == 1:
                        yield turn
        else:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    turn = TurnSample(**json.loads(line))
                    if len(turn.tool_calls) == 1:
                        yield turn


def think_span(completion_ids: list[int], think_end_id: int | None) -> int:
    """Number of leading completion tokens that are thinking, including the closing `</think>`.

    The thinking chat template puts `<think>` in the prompt, so a completion is
    `reasoning ... </think> answer`. No `</think>` (thinking off, or truncated) -> 0 if thinking is off
    (`think_end_id is None`), else the whole completion is thinking.
    """
    if think_end_id is None:
        return 0
    try:
        return completion_ids.index(think_end_id) + 1
    except ValueError:
        return len(completion_ids)


def completion_mask(completion_ids: list[int], think_end_id: int | None, think: str = "train") -> list[bool]:
    """Which completion tokens carry loss: all of them (`train`) or only those after `</think>` (`mask`)."""
    if think not in THINK_MODES:
        raise ValueError(f"think must be one of {THINK_MODES}, not {think!r}")
    if think == "train":
        return [True] * len(completion_ids)
    n = think_span(completion_ids, think_end_id)
    return [False] * n + [True] * (len(completion_ids) - n)


def token_example(sample: TurnSample, think_end_id: int | None, think: str = "train") -> tuple[list[int], list[int]]:
    """(input_ids, labels) from the exact sampled tokens; labels are IGNORE on the prompt and masked tokens."""
    if not sample.has_tokens:
        raise ValueError("sample has no token ids; render it with the chat template instead")
    prompt, completion = sample.prompt_token_ids, sample.completion_token_ids
    mask = completion_mask(completion, think_end_id, think)
    labels = [IGNORE] * len(prompt) + [tok if keep else IGNORE for tok, keep in zip(completion, mask, strict=True)]
    return prompt + completion, labels
