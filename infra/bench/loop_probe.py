"""v0.3 T08-loop probe & dry-run driver: scripted baselines + model configs with full telemetry & trajectories.

    python infra/bench/loop_probe.py ping   [--only ID ...]
    python infra/bench/loop_probe.py run    [--only ID ...] [--datasets datasets/t1-dev-v03 datasets/t2-dev-v03]
    python infra/bench/loop_probe.py report [--results-dir results/v03_dryrun] [--json-out results/v03_dryrun/summary.json]

Every run executes through `play_episode` (including scripted `greedy` and `reference`), saving:
- `<results-dir>/<run>.jsonl` (v0.3 anchor, KPIs, counters, tokens, cached/thinking tokens, USD, wall/LLM time, prompt_stats)
- `<results-dir>/<run>/transcripts/<scenario_id>-r<rep>.jsonl` (per-turn tool calls, latency, usage, prompt char breakdown)
- `<results-dir>/<run>/trajectories/<scenario_id>-r<rep>.json` (continuous [t0, t1] train position segments, movements, snapshots)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from railway.eval import stats as eval_stats  # noqa: E402
from railway.eval.manifest import Manifest  # noqa: E402
from railway.eval.run import (  # noqa: E402
    EpisodeSpec,
    LoopDeps,
    _done_pairs,
    _Writer,
    episodes_from_dataset,
    episodes_from_scenarios,
    read_rows,
    reward_from_t07,
    run_manifest,
)
from railway.loop.fake_model import END_SHIFT, _wire  # noqa: E402
from railway.loop.model_client import BACKOFF_S, ModelClient, ModelError, ModelReply, RawToolCall, _ssl_context  # noqa: E402
from railway.loop.prompts import load_instruction  # noqa: E402
from railway.loop.runner import make_session, play_episode  # noqa: E402
from railway.sim.engine import Simulator  # noqa: E402
from railway.sim.observe import State  # noqa: E402

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
VERDA_URL = "https://containers.datacrunch.io/nemotron-3-5-super-vl-b300/v1"
QWEN_THINK_URL = "https://hoaianh517--railway-qwen-think.us-east.modal.direct/v1"
SCRIPTED_URL = "scripted://inproc"

ANTHROPIC_CAP_USD = 10.0

# USD per Mtok: (uncached_input, output_and_thinking, cache_read, cache_write)
MODEL_PRICE: dict[str, tuple[float, float, float, float]] = {
    "gemini-3.8-flash": (0.75, 3.75, 0.075, 0.0),
    "gemini-4-hydrogen": (1.50, 7.50, 0.075, 0.0),
    "gemini-4-carbon": (2.00, 10.00, 0.10, 0.0),
    "claude-fable-5-2": (8.00, 40.00, 0.20, 10.00),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    "qwen38": (0.0, 0.0, 0.0, 0.0),
    "nemotron-3.5-super-vl": (0.0, 0.0, 0.0, 0.0),
    "greedy": (0.0, 0.0, 0.0, 0.0),
    "reference": (0.0, 0.0, 0.0, 0.0),
}

# (id, base_url, model, key_env, extra_body, max_tokens)
CONFIGS = [
    # Scripted baselines through the exact T08 loop
    ("greedy", SCRIPTED_URL, "greedy", None, {}, 1024),
    ("reference", SCRIPTED_URL, "reference", None, {}, 1024),
    # Cheap smoke / dry-run model
    ("gemini-flash-low", GEMINI_URL, "gemini-3.8-flash", "GEMINI_API_KEY", {"reasoning_effort": "low"}, 16384),
    ("gemini-flash-medium", GEMINI_URL, "gemini-3.8-flash", "GEMINI_API_KEY", {"reasoning_effort": "medium"}, 32768),
    ("gemini-flash-high", GEMINI_URL, "gemini-3.8-flash", "GEMINI_API_KEY", {"reasoning_effort": "high"}, 32768),
    # Candidate pilot models (hold until model list & spend cap are finalized)
    ("qwen38-think", QWEN_THINK_URL, "qwen38", "VLLM_API_KEY", {}, 32768),
    ("gemini-hydrogen-low", GEMINI_URL, "gemini-4-hydrogen", "GEMINI_API_KEY", {"reasoning_effort": "low"}, 32768),
    ("gemini-hydrogen-medium", GEMINI_URL, "gemini-4-hydrogen", "GEMINI_API_KEY", {"reasoning_effort": "medium"}, 32768),
    ("gemini-hydrogen-high", GEMINI_URL, "gemini-4-hydrogen", "GEMINI_API_KEY", {"reasoning_effort": "high"}, 32768),
    ("gemini-hydrogen-xhigh", GEMINI_URL, "gemini-4-hydrogen", "GEMINI_API_KEY", {"reasoning_effort": "xhigh"}, 32768),
    ("gemini-carbon-low", GEMINI_URL, "gemini-4-carbon", "GEMINI_API_KEY", {"reasoning_effort": "low"}, 32768),
    ("gemini-carbon-medium", GEMINI_URL, "gemini-4-carbon", "GEMINI_API_KEY", {"reasoning_effort": "medium"}, 32768),
    ("gemini-carbon-high", GEMINI_URL, "gemini-4-carbon", "GEMINI_API_KEY", {"reasoning_effort": "high"}, 32768),
    ("fable-high", ANTHROPIC_MESSAGES_URL, "claude-fable-5-2", "ANTHROPIC_API_KEY", {"reasoning_effort": "high"}, 32768),
    ("haiku-high", ANTHROPIC_MESSAGES_URL, "claude-haiku-5-5", "ANTHROPIC_API_KEY", {"reasoning_effort": "high"}, 32768),
    ("nemotron-super", VERDA_URL, "nemotron-3.5-super-vl", "VERDA_INFERENCE_KEY", {}, 32768),
]
CONFIG_BY_ID = {c[0]: c for c in CONFIGS}
DEFAULT_DATASETS = ["datasets/t1-dev-v03", "datasets/t2-dev-v03"]


class Spend:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.usd = json.loads(path.read_text(encoding="utf-8"))["usd"] if path.exists() else 0.0

    def add(self, usd: float) -> float:
        with self.lock:
            self.usd += usd
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"usd": round(self.usd, 6)}), encoding="utf-8")
            return self.usd


class PolicyLoopClient:
    """In-process T08 loop client for deterministic scripted policies (`greedy`, `reference`)."""

    def __init__(self, policy_name: str, session, scenario):
        from railway.policies.baselines import Greedy
        from railway.ref.reference import Reference

        self.decide_fn = Greedy().decide if policy_name == "greedy" else Reference().decide
        self.session = session
        self.static = Simulator.load(scenario).static_info()
        self.queue: deque = deque()
        self.turn = 0

    def close(self) -> None:
        pass

    def chat(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        self.turn += 1
        if getattr(self.session, "closed", False):
            call = END_SHIFT
        else:
            if not self.queue:
                state = State.model_validate_json(self.session.peek("status", {}))
                self.queue.extend(self.decide_fn(self.static, state) or [END_SHIFT])
            call = self.queue.popleft()
        name, raw = _wire(call)
        # Approximate 4 chars/token on scripted runs so prompt-size comparison is available alongside real tokenizers.
        prompt_chars = len(json.dumps(messages, ensure_ascii=False)) + len(json.dumps(tools, ensure_ascii=False))
        comp_chars = len(name) + len(raw) + 8
        return {
            "tool_calls": [{"id": f"script_{self.turn}", "name": name, "arguments_raw": raw}],
            "content": None,
            "usage": {
                "prompt_tokens": prompt_chars // 4 + 1,
                "completion_tokens": comp_chars // 4 + 1,
                "cached_tokens": 0,
                "thinking_tokens": 0,
                "usd": 0.0,
            },
        }


class ProbeClient(ModelClient):
    """OpenAI-compatible client that records cached/thinking tokens and exact USD per turn."""

    def __init__(self, *a, extra: dict, price: tuple[float, float, float, float], **kw):
        super().__init__(*a, **kw)
        self.extra = extra
        self.price = price

    def request_body(self, messages, tools):
        return {**super().request_body(messages, tools), **self.extra}

    def chat(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        body = self.request_body(messages, tools)
        attempt = 0
        while True:
            try:
                resp = self._http.post(self.url, json=body)
                status = resp.status_code
                detail = "(body withheld)" if status in (401, 403) else resp.text[:200]
            except httpx.HTTPError as exc:
                status, detail = 599, type(exc).__name__
            if status == 200:
                try:
                    data = resp.json()
                except ValueError:
                    raise ModelError("response is not JSON") from None
                from railway.loop.model_client import parse_reply

                reply = parse_reply(data)
                usage = data.get("usage") or {}
                prompt = reply["usage"]["prompt_tokens"]
                visible_out = reply["usage"]["completion_tokens"]
                total = int(usage.get("total_tokens") or 0)
                p_details = usage.get("prompt_tokens_details") or {}
                c_details = usage.get("completion_tokens_details") or {}
                cached = int(p_details.get("cached_tokens") or usage.get("cached_tokens") or 0)
                cached = min(cached, prompt)
                # Gemini's OpenAI shim puts thinking tokens in total_tokens - prompt_tokens - completion_tokens;
                # OpenAI/vLLM shims put reasoning_tokens inside completion_tokens_details.
                extra_think = max(0, total - prompt - visible_out)
                reported_think = int(c_details.get("reasoning_tokens") or usage.get("thinking_tokens") or 0)
                if extra_think > 0:
                    out_total = visible_out + extra_think
                    thinking = extra_think
                else:
                    out_total = visible_out
                    thinking = reported_think

                p_in, p_out, p_cr, _ = self.price
                uncached = max(0, prompt - cached)
                usd = (uncached * p_in + cached * p_cr + out_total * p_out) / 1e6
                reply["usage"] = {
                    "prompt_tokens": prompt,
                    "completion_tokens": out_total,
                    "cached_tokens": cached,
                    "thinking_tokens": thinking,
                    "usd": round(usd, 8),
                }
                return reply
            if not (status == 429 or status >= 500):
                raise ModelError(f"HTTP {status}: {detail}")
            if attempt >= self.max_retries:
                raise ModelError(f"HTTP {status} after {attempt} retries: {detail}")
            time.sleep(BACKOFF_S[attempt])
            attempt += 1


def _parse_tool_input(raw: str) -> dict:
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {"__raw__": raw}
    return parsed if isinstance(parsed, dict) else {"__raw__": raw}


class AnthropicMessagesClient:
    """Native Anthropic `/v1/messages` client with ephemeral prefix caching on MANUAL + adaptive thinking."""

    def __init__(
        self,
        url: str,
        model: str,
        api_key_env: str,
        *,
        effort: str | None,
        max_tokens: int,
        timeout_s: float = 300,
        max_retries: int = 5,
        price: tuple[float, float, float, float],
        spend: Spend,
    ):
        key = os.environ.get(api_key_env)
        if not key:
            raise ModelError(f"environment variable {api_key_env} is not set")
        self.url = url
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.price = price
        self.spend = spend
        self._http = httpx.Client(
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
            timeout=timeout_s,
            verify=_ssl_context(),
        )

    def close(self) -> None:
        self._http.close()

    def _convert(self, messages: list[dict], tools: list[dict]) -> dict:
        system_text: str | None = None
        anth_msgs: list[dict] = []
        cached_manual = False

        def append_user_block(block: dict) -> None:
            if anth_msgs and anth_msgs[-1]["role"] == "user":
                anth_msgs[-1]["content"].append(block)
            else:
                anth_msgs.append({"role": "user", "content": [block]})

        for msg in messages:
            role = msg["role"]
            if role == "system":
                system_text = msg["content"]
            elif role == "user":
                text = msg["content"] or ""
                block: dict = {"type": "text", "text": text}
                if not cached_manual and text.startswith("MANUAL\n"):
                    block["cache_control"] = {"type": "ephemeral"}
                    cached_manual = True
                append_user_block(block)
            elif role == "assistant":
                blocks: list[dict] = []
                if msg.get("content"):
                    blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": str(tc.get("id") or "call_0"),
                            "name": str(fn.get("name") or ""),
                            "input": _parse_tool_input(fn.get("arguments") or ""),
                        }
                    )
                anth_msgs.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                append_user_block(
                    {
                        "type": "tool_result",
                        "tool_use_id": str(msg["tool_call_id"]),
                        "content": msg.get("content") or "",
                    }
                )

        body: dict = {"model": self.model, "messages": anth_msgs, "max_tokens": self.max_tokens}
        if system_text:
            body["system"] = system_text
        if tools:
            body["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"]["parameters"],
                }
                for t in tools
            ]
        if self.effort:
            body["thinking"] = {"type": "adaptive"}
            body["output_config"] = {"effort": self.effort}
        return body

    def chat(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        if self.spend.usd >= ANTHROPIC_CAP_USD:
            raise ModelError(f"anthropic cap ${ANTHROPIC_CAP_USD} reached")
        body = self._convert(messages, tools)
        attempt = 0
        while True:
            try:
                resp = self._http.post(self.url, json=body)
                status = resp.status_code
                detail = "(body withheld)" if status in (401, 403) else resp.text[:200]
            except httpx.HTTPError as exc:
                status, detail = 599, type(exc).__name__
            if status == 200:
                data = resp.json()
                calls: list[RawToolCall] = []
                texts: list[str] = []
                for i, blk in enumerate(data.get("content") or []):
                    btype = blk.get("type")
                    if btype == "tool_use":
                        inp = blk.get("input")
                        raw = json.dumps(inp) if isinstance(inp, dict) else str(inp or "")
                        calls.append(
                            {"id": str(blk.get("id") or f"call_{i}"), "name": str(blk.get("name") or ""), "arguments_raw": raw}
                        )
                    elif btype == "text" and blk.get("text"):
                        texts.append(blk["text"])
                u = data.get("usage") or {}
                in_tok = int(u.get("input_tokens") or 0)
                cw_tok = int(u.get("cache_creation_input_tokens") or 0)
                cr_tok = int(u.get("cache_read_input_tokens") or 0)
                out_tok = int(u.get("output_tokens") or 0)
                think_tok = int((u.get("output_tokens_details") or {}).get("thinking_tokens") or 0)
                p_in, p_out, p_cr, p_cw = self.price
                usd = (in_tok * p_in + out_tok * p_out + cr_tok * p_cr + cw_tok * p_cw) / 1e6
                self.spend.add(usd)
                return {
                    "tool_calls": calls,
                    "content": "\n".join(texts) if texts else None,
                    "usage": {
                        "prompt_tokens": in_tok + cw_tok + cr_tok,
                        "completion_tokens": out_tok,
                        "cached_tokens": cr_tok,
                        "thinking_tokens": think_tok,
                        "usd": round(usd, 8),
                    },
                }
            if not (status == 429 or status >= 500):
                raise ModelError(f"HTTP {status}: {detail}")
            if attempt >= self.max_retries:
                raise ModelError(f"HTTP {status} after {attempt} retries: {detail}")
            time.sleep(BACKOFF_S[attempt])
            attempt += 1


def load_env() -> None:
    for candidate in (ROOT / ".env", ROOT.parent / "Afterquery" / ".env"):
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line.strip())
                if m:
                    os.environ.setdefault(m.group(1), m.group(2).strip().strip('"'))
            break


def warm_modal_if_needed(url: str, timeout_s: float = 900) -> None:
    if ".modal.direct" not in url:
        return
    health = re.sub(r"/v1/?$", "", url.rstrip("/")) + "/health"
    t0 = time.monotonic()
    announced = False
    while True:
        try:
            if httpx.get(health, timeout=30).status_code == 200:
                if announced:
                    print(f"WARM {health} ready after {time.monotonic() - t0:.0f}s", flush=True)
                return
        except httpx.HTTPError:
            pass
        if not announced:
            print(f"WARM waiting for Modal cold start at {health} ...", flush=True)
            announced = True
        if time.monotonic() - t0 > timeout_s:
            raise ModelError(f"Modal endpoint {health} did not become healthy within {timeout_s:.0f}s")
        time.sleep(10)


def make_client(cfg, spend: Spend, *, session=None, ep: EpisodeSpec | None = None, max_retries: int = 5):
    cid, url, model, key, extra, max_tokens = cfg
    if url == SCRIPTED_URL:
        return PolicyLoopClient(model, session, ep.scenario)
    price = MODEL_PRICE.get(model, (0.0, 0.0, 0.0, 0.0))
    if url == ANTHROPIC_MESSAGES_URL:
        return AnthropicMessagesClient(
            url,
            model,
            key,
            effort=extra.get("reasoning_effort"),
            max_tokens=max_tokens,
            timeout_s=300,
            max_retries=max_retries,
            price=price,
            spend=spend,
        )
    return ProbeClient(
        url, model, key, max_tokens=max_tokens, timeout_s=300, max_retries=max_retries, extra=extra, price=price
    )


def enrich_c_ref(episodes: list[EpisodeSpec]) -> list[EpisodeSpec]:
    """Attach `c_ref` from `seeds/<preset>-dev.json` when loading from an exported dataset."""
    c_ref_by_hash: dict[str, float] = {}
    for p in (ROOT / "seeds").glob("*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        for a in data.get("accepted") or []:
            if "scenario_hash" in a and a.get("c_ref") is not None:
                c_ref_by_hash[a["scenario_hash"]] = float(a["c_ref"])
    from railway.util import content_hash

    for ep in episodes:
        if ep.c_ref is None:
            ep.c_ref = c_ref_by_hash.get(content_hash(ep.scenario.model_dump(mode="json")))
    return episodes


def _run_scripted_episode(args_tuple) -> dict:
    """Worker entry point so CPU-bound `Reference` rollouts run across multiple CPU processes."""
    ep, rep, run_name, model_id, policy_name, results_dir_str = args_tuple
    results_dir = Path(results_dir_str)
    sid = ep.scenario.scenario_id
    session = make_session(ep.scenario)
    client = PolicyLoopClient(policy_name, session, ep.scenario)
    rec = play_episode(
        ep.scenario,
        client,
        "windowed",
        session=session,
        instruction=load_instruction(),
        transcript_path=results_dir / run_name / "transcripts" / f"{sid}-r{rep}.jsonl",
        trajectory_path=results_dir / run_name / "trajectories" / f"{sid}-r{rep}.json",
        run=run_name,
        model=model_id,
        rep=rep,
        c_greedy=ep.c_greedy,
        c_best=ep.c_best,
        c_ref=ep.c_ref,
        best_source=ep.best_source,
        best_proven=ep.best_proven,
        score_fn=reward_from_t07,
    )
    return rec.to_row()


def run_scripted(cfg, episodes: list[EpisodeSpec], run_name: str, reps: int, results_dir: Path, workers: int) -> dict:
    cid, _, policy_name, *_ = cfg
    out = results_dir / f"{run_name}.jsonl"
    done = _done_pairs(out)
    todo = [
        (ep, rep, run_name, cid, policy_name, str(results_dir))
        for ep in episodes
        for rep in range(reps)
        if (ep.scenario.scenario_id, rep) not in done
    ]
    if not todo:
        return {"executed": 0, "skipped": len(episodes) * reps, "results": str(out)}
    writer = _Writer(out)
    try:
        max_w = max(1, min(workers, len(todo)))
        with cf.ProcessPoolExecutor(max_workers=max_w) as pool:
            for row in pool.map(_run_scripted_episode, todo):
                writer.put(row)
    finally:
        writer.close()
    return {"executed": len(todo), "skipped": len(episodes) * reps - len(todo), "results": str(out)}


def ping(args, spend: Spend) -> None:
    tool = {
        "type": "function",
        "function": {"name": "advance", "description": "Advance time.", "parameters": {"type": "object", "properties": {}}},
    }
    msgs = [{"role": "user", "content": "Call the advance tool once."}]
    for cfg in CONFIGS:
        if cfg[1] == SCRIPTED_URL:
            continue
        if args.only and cfg[0] not in args.only:
            continue
        try:
            warm_modal_if_needed(cfg[1])
            c = make_client(cfg, spend, max_retries=2)
            r = c.chat(msgs, [tool])
            print(f"OK   {cfg[0]:<24} calls={[t['name'] for t in r['tool_calls']]} usage={r['usage']}", flush=True)
            c.close()
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {cfg[0]:<24} {type(exc).__name__}: {str(exc)[:300]}", flush=True)


def _dataset_tag(dataset_path: Path) -> str:
    manifest_file = dataset_path / "manifest.json"
    if manifest_file.exists():
        m = json.loads(manifest_file.read_text(encoding="utf-8"))
        return f"{m['preset'].lower()}-{m['set']}"
    return dataset_path.name


def run(args, spend: Spend) -> None:
    results_dir = Path(args.results_dir)
    selected = [c for c in CONFIGS if (not args.only or c[0] in args.only)]
    if not selected:
        raise SystemExit(f"no configs matched --only {args.only}")

    for ds in args.datasets:
        ds_path = (ROOT / ds) if not Path(ds).is_absolute() else Path(ds)
        tag = _dataset_tag(ds_path)
        episodes = enrich_c_ref(episodes_from_dataset(ds_path))
        for cfg in selected:
            cid, url, model, key, extra, max_tokens = cfg
            run_name = f"{tag}-{cid}"
            print(f"RUN  {run_name} ({len(episodes)} tasks x {args.reps} reps)", flush=True)
            t0 = time.monotonic()
            if url == SCRIPTED_URL:
                rep_info = run_scripted(cfg, episodes, run_name, args.reps, results_dir, workers=max(args.concurrency, 5))
            else:
                warm_modal_if_needed(url)
                # Ensure enriched c_ref flows into _run_loop episodes too.
                import railway.eval.run as eval_run_mod

                orig_loader = eval_run_mod.episodes_from_dataset
                eval_run_mod.episodes_from_dataset = lambda _d, eps=episodes: eps
                try:
                    manifest = Manifest(
                        name=run_name,
                        harness="loop",
                        mode="windowed",
                        model={"id": cid, "name": model, "base_url": url, "api_key_env": key},
                        max_tokens=max_tokens,
                        dataset=str(ds_path),
                        reps=args.reps,
                        concurrency=args.concurrency,
                    )
                    deps = LoopDeps(client_factory=lambda session, ep, cfg=cfg: make_client(cfg, spend, session=session, ep=ep))
                    rep_info = run_manifest(manifest, results_dir=results_dir, deps=deps)
                finally:
                    eval_run_mod.episodes_from_dataset = orig_loader
            dt = round(time.monotonic() - t0, 1)
            print(f"DONE {run_name} in {dt}s -> {rep_info}", flush=True)


def _prompt_summary(rows: list[dict]) -> dict:
    ps_list = [r["prompt_stats"] for r in rows if r.get("status") == "ok" and isinstance(r.get("prompt_stats"), dict)]
    if not ps_list:
        return {}

    def avg(vals):
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    chars_keys = (
        "system",
        "manual",
        "tools",
        "status_compact_mean",
        "status_compact_max",
        "window_mean",
        "window_max",
        "bulletins_max",
        "notes_max",
    )
    return {
        "mean_turns_per_episode": avg([p.get("turns", 0) for p in ps_list]),
        "mean_decision_points": avg([p.get("decision_points", 0) or 0 for p in ps_list]),
        "tokens_in_per_turn_mean": avg([p.get("tokens_in_per_turn_mean", 0) for p in ps_list]),
        "tokens_in_per_turn_max": max((p.get("tokens_in_per_turn_max", 0) for p in ps_list), default=0),
        "tokens_in_at_decision_mean": avg([p.get("tokens_in_at_decision_mean", 0) for p in ps_list]),
        "tokens_in_at_decision_max": max((p.get("tokens_in_at_decision_max", 0) for p in ps_list), default=0),
        "tokens_in_per_decision_total_mean": avg([p.get("tokens_in_per_decision_total", 0) for p in ps_list]),
        "chars": {
            k: (
                max(p["chars"].get(k, 0) for p in ps_list if "chars" in p)
                if k.endswith("_max")
                else avg([p["chars"].get(k, 0) for p in ps_list if "chars" in p])
            )
            for k in chars_keys
        },
    }


def report(args) -> None:
    results_dir = Path(args.results_dir)
    files = sorted(results_dir.glob("*.jsonl"))
    full_report: dict[str, dict] = {}
    for f in files:
        rows = eval_stats.dedupe(read_rows(f))
        if not rows:
            continue
        s = eval_stats.summarize(rows)
        ps = _prompt_summary(rows)
        full_report[f.stem] = {
            "summary": s,
            "prompt_stats": ps,
            "episodes": [
                {
                    "scenario_id": r["scenario_id"],
                    "seed": r["seed"],
                    "preset": r["preset"],
                    "status": r["status"],
                    "reward": r.get("reward") if r.get("reward") is not None else (-r["cost_total"] if r.get("cost_total") is not None else None),
                    "cost_total": r.get("cost_total"),
                    "c_best": r.get("c_best"),
                    "c_ref": r.get("c_ref"),
                    "c_greedy": r.get("c_greedy"),
                    "best_source": r.get("best_source"),
                    "best_proven": r.get("best_proven"),
                    "score_normalized": r.get("score_normalized") if r.get("score_normalized") is not None else r.get("score_clipped"),
                    "score_unclipped": r.get("score_unclipped"),
                    "above_best": r.get("above_best"),
                    "kpis": r.get("kpis"),
                    "counts": {
                        k: r.get(k)
                        for k in (
                            "gridlocks",
                            "crew_timeouts",
                            "misroutes",
                            "wrong_terminals",
                            "missed_calls",
                            "missed_connections",
                            "unfinished_trains",
                            "refusals",
                            "invalid_calls",
                            "format_errors",
                        )
                    },
                    "tool_calls": r.get("tool_calls"),
                    "decision_points": r.get("decision_points"),
                    "tokens_in": r.get("tokens_in"),
                    "tokens_cached": r.get("tokens_cached"),
                    "tokens_out": r.get("tokens_out"),
                    "tokens_thinking": r.get("tokens_thinking"),
                    "usd": r.get("usd"),
                    "wall_s": r.get("wall_s"),
                    "llm_s": r.get("llm_s"),
                    "prompt_stats": r.get("prompt_stats"),
                    "error": r.get("error"),
                }
                for r in rows
            ],
        }
        kpis = s.get("kpis") or {}
        totals = s.get("totals") or {}
        nlo, nhi = s["neg_cost_ci"]
        print(
            f"=== {f.stem} ({s['episodes']}/{s['planned']} ok) ===\n"
            f"  HEADLINE reward (-cost) mean={s['neg_cost_mean']:.2f}  95% CI=[{nlo:.2f}, {nhi:.2f}]\n"
            f"  Normalised (report-only): norm_mean={s['score_normalized_mean']:.4f}  score_ros={s['score_ros']:.4f}  "
            f"unclipped={s['unclipped_mean']:.4f}  cost_ratio={s['cost_ratio']:.2f}  above_best={s['flagged']}\n"
            f"  KPIs: on_time={kpis.get('on_time_pct')}%  pax_delay_min={kpis.get('passenger_delay_minutes')}  "
            f"freight_deliv={kpis.get('freight_delivered_pct')}%  red_flags={kpis.get('red_flags')}  "
            f"clean_shift={kpis.get('clean_shift')}\n"
            f"  Tokens & Spend: in={totals.get('tokens_in')}  out={totals.get('tokens_out')}  "
            f"total_usd=${totals.get('usd') or 0.0:.4f}  total_wall={totals.get('wall_s')}s\n"
            f"  Prompt: per_turn={ps.get('tokens_in_per_turn_mean')} tok (at_decision={ps.get('tokens_in_at_decision_mean')} tok, "
            f"total_per_decision={ps.get('tokens_in_per_decision_total_mean')} tok) | "
            f"status_compact={ps.get('chars', {}).get('status_compact_mean')} chars (max {ps.get('chars', {}).get('status_compact_max')})"
        )
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(full_report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote JSON report -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["ping", "run", "report"])
    p.add_argument("--only", nargs="*")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--results-dir", default=str(ROOT / "results" / "v03_dryrun"))
    p.add_argument("--json-out", default=None)
    args = p.parse_args()
    load_env()
    spend = Spend(Path(args.results_dir) / "anthropic_spend.json")
    if args.cmd == "ping":
        ping(args, spend)
    elif args.cmd == "run":
        run(args, spend)
    else:
        report(args)


if __name__ == "__main__":
    main()
