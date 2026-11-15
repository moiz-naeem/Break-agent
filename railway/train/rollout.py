"""Rollouts for GRPO and SFT: G episodes per seed, in parallel, through an OpenAI-compatible vLLM server.

Episodes run through the benchmark loop (`railway.loop.runner.play_episode`), so prompts are identical to
the evaluation. The model client here is a thin capture wrapper: it asks vLLM for the exact prompt and
completion token ids (`return_token_ids`) and per-token log-probs, and records one TurnSample per call.
The loop calls `chat` exactly once per turn, so captured samples line up with the loop's transcript.
"""

from __future__ import annotations

import copy
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import httpx

from railway.loop.model_client import BACKOFF_S, ModelError, ModelReply, clean_openai_messages, parse_reply
from railway.loop.runner import play_episode
from railway.train.env import Env
from railway.train.samples import Trajectory, TurnSample


class CaptureClient:
    """`chat(messages, tools) -> ModelReply` like `railway.loop.model_client.ModelClient`, plus token capture."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key_env: str | None = None,
        temperature: float | None = 1.0,
        max_tokens: int = 8192,
        thinking: bool | None = None,
        thinking_budget: int | None = 4096,
        timeout_s: float = 900,
        max_retries: int = 3,
        capture_tokens: bool = True,
        http: httpx.Client | None = None,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max(max_tokens, thinking_budget + 2048) if (thinking is not False and thinking_budget) else max_tokens  # noqa: E501
        self.thinking = thinking
        self.thinking_budget = thinking_budget
        self.max_retries = max_retries
        self.capture_tokens = capture_tokens
        headers = {"Content-Type": "application/json"}
        if api_key_env:
            key = os.environ.get(api_key_env)
            if not key:
                raise ModelError(f"environment variable {api_key_env} is not set")
            headers["Authorization"] = f"Bearer {key}"
        self._http = http or httpx.Client(headers=headers, timeout=timeout_s)
        self.samples: list[TurnSample] = []

    def __repr__(self) -> str:
        return f"CaptureClient(url={self.url!r}, model={self.model!r})"

    def close(self) -> None:
        self._http.close()

    def request_body(self, messages: list[dict], tools: list[dict]) -> dict:
        body = {"model": self.model, "messages": clean_openai_messages(messages), "max_tokens": self.max_tokens}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.capture_tokens:
            body["logprobs"] = True
            body["return_token_ids"] = True
        if self.thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": self.thinking}
        if self.thinking_budget is not None and self.thinking is not False:
            body["thinking_token_budget"] = self.thinking_budget
        return body

    def _post(self, body: dict) -> dict:
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._http.post(self.url, json=body)
                status, detail = resp.status_code, resp.text[:300]
            except httpx.HTTPError as exc:
                status, detail = 599, type(exc).__name__
            if status == 200:
                return resp.json()
            if status != 429 and status < 500:
                raise ModelError(f"HTTP {status}: {detail}")
            if attempt < self.max_retries:
                time.sleep(BACKOFF_S[attempt])
        raise ModelError(f"HTTP {status} after {self.max_retries} retries: {detail}")

    def chat(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        data = self._post(self.request_body(messages, tools))
        reply = parse_reply(data)
        self.samples.append(capture(data, reply, messages, tools))
        return reply


def capture(data: dict, reply: ModelReply, messages: list[dict], tools: list[dict]) -> TurnSample:
    choice = data["choices"][0]
    message = choice.get("message") or {}
    logprobs = None
    content_lp = (choice.get("logprobs") or {}).get("content")
    if content_lp:
        logprobs = [float(x["logprob"]) for x in content_lp]
    return TurnSample(
        turn=0,
        messages=copy.deepcopy(messages),
        tools=tools,
        content=reply["content"],
        reasoning=message.get("reasoning_content") or message.get("reasoning"),
        tool_calls=[{"name": c["name"], "arguments_raw": c["arguments_raw"]} for c in reply["tool_calls"]],
        finish_reason=choice.get("finish_reason"),
        prompt_token_ids=data.get("prompt_token_ids") or choice.get("prompt_token_ids"),
        completion_token_ids=choice.get("token_ids"),
        logprobs=logprobs,
    )


ClientFactory = Callable[[], CaptureClient]


def run_episode(
    env: Env,
    seed: int,
    client_factory: ClientFactory,
    *,
    group: str,
    rep: int,
    mode: str = "windowed",
    max_turns: int | None = None,
    greedy_cost: float | None = None,
) -> Trajectory:
    episode = env.reset(seed)
    client = client_factory()
    kwargs: dict = {"transcript_root": os.environ.get("RAILWAY_TRANSCRIPTS_DIR") or "/tmp/rollout_transcripts"}
    if max_turns is not None:
        kwargs["max_turns"] = max_turns
    try:
        rec = play_episode(
            episode.scenario, client, mode, session=episode.session, model=client.model, rep=rep, **kwargs
        )
    finally:
        client.close()
    turns = client.samples
    for n, sample in enumerate(turns, start=1):
        sample.turn = n
    ok = rec.status == "ok" and rec.cost_total is not None
    return Trajectory(
        episode_id=f"{group}-r{rep}",
        env=env.name,
        seed=seed,
        group=group,
        model=client.model,
        mode=mode,
        status=rec.status,
        error=rec.error,
        cost=rec.cost_total,
        reward=-float(rec.cost_total) if ok else None,
        greedy_cost=greedy_cost,
        turns_played=len(turns),
        row=rec.to_row(),
        turns=turns,
    )


def rollout_groups(
    env: Env,
    seeds: list[int],
    group_size: int,
    client_factory: ClientFactory,
    *,
    tag: str = "it0",
    mode: str = "windowed",
    max_turns: int | None = None,
    with_greedy: bool = True,
    workers: int | None = None,
) -> list[Trajectory]:
    """`group_size` episodes of each seed, all in parallel; groups are named `s<seed>-<tag>`."""
    greedy = {s: env.greedy_cost(s) for s in seeds} if with_greedy else {}  # None on the network env
    jobs = [(s, r) for s in seeds for r in range(group_size)]
    with ThreadPoolExecutor(max_workers=workers or len(jobs)) as pool:
        futures = [
            pool.submit(
                run_episode,
                env,
                seed,
                client_factory,
                group=f"s{seed}-{tag}",
                rep=rep,
                mode=mode,
                max_turns=max_turns,
                greedy_cost=greedy.get(seed),
            )
            for seed, rep in jobs
        ]
        return [f.result() for f in futures]


def summarize(trajectories: list[Trajectory]) -> dict:
    ok = [t for t in trajectories if t.ok]
    turns = [s for t in trajectories for s in t.turns]
    comp = [len(s.completion_token_ids) for s in turns if s.completion_token_ids]
    prompt = [len(s.prompt_token_ids) for s in turns if s.prompt_token_ids]
    return {
        "episodes": len(trajectories),
        "ok": len(ok),
        "mean_cost": round(sum(t.cost for t in ok) / len(ok), 2) if ok else None,
        "mean_greedy": (
            round(sum(t.greedy_cost for t in ok) / len(ok), 2)
            if ok and all(t.greedy_cost is not None for t in ok)
            else None
        ),
        "turns": len(turns),
        "tool_call_turns": sum(len(s.tool_calls) == 1 for s in turns),
        "truncated_turns": sum(s.finish_reason == "length" for s in turns),
        "mean_prompt_tokens": round(sum(prompt) / len(prompt)) if prompt else None,
        "mean_completion_tokens": round(sum(comp) / len(comp)) if comp else None,
        "max_completion_tokens": max(comp) if comp else None,
        "errors": sorted({t.error for t in trajectories if t.error}),
    }
