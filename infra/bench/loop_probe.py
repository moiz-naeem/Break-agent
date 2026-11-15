"""Quick T08-loop probe across every planned model: one scenario, one rep per config.

    python infra/bench/loop_probe.py ping [--only ID ...]
    python infra/bench/loop_probe.py run  [--only ID ...] [--preset T1 --set dev --seeds 5]

Anthropic models run through native `/v1/messages` with ephemeral prompt caching on `tools + system +
MANUAL` (~6.7k tokens/turn cached at 97.5% discount) and adaptive thinking at the requested effort.
 Spend is tallied from exact cached/uncached usage and stops once `ANTHROPIC_CAP_USD` is reached.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import railway.eval.run as eval_run  # noqa: E402
from railway.eval.manifest import Manifest  # noqa: E402
from railway.eval.run import EpisodeSpec, LoopDeps, run_manifest  # noqa: E402
from railway.loop.model_client import BACKOFF_S, ModelClient, ModelError, ModelReply, RawToolCall, _ssl_context  # noqa: E402

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
VERDA_URL = "https://containers.datacrunch.io/nemotron-3-5-super-vl-b300/v1"
QWEN_THINK_URL = "https://hoaianh517--railway-qwen-think.us-east.modal.direct/v1"

ANTHROPIC_CAP_USD = 10.0
# USD per Mtok: (input, output, cache_read, cache_write)
ANTHROPIC_PRICE: dict[str, tuple[float, float, float, float]] = {
    "claude-fable-5-2": (8.0, 40.0, 0.2, 10.0),
    "claude-haiku-5-5": (0.1, 0.5, 0.01, 0.125),
}

# Final lineup: Qwen Think, Hydrogen, Carbon, Fable -> Haiku, NVIDIA (Nemotron on Verda).
# (id, base_url, model, key env, extra body, max_tokens)
CONFIGS = [
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


class Spend:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.usd = json.loads(path.read_text())["usd"] if path.exists() else 0.0

    def add(self, usd: float) -> float:
        with self.lock:
            self.usd += usd
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"usd": round(self.usd, 4)}))
            return self.usd


class ProbeClient(ModelClient):
    """OpenAI-compatible client that passes extra fields and includes hidden thinking tokens in tokens_out."""

    def __init__(self, *a, extra: dict, **kw):
        super().__init__(*a, **kw)
        self.extra = extra

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
                total = int(usage.get("total_tokens") or 0)
                prompt = reply["usage"]["prompt_tokens"]
                if total > prompt + reply["usage"]["completion_tokens"]:
                    # Gemini's OpenAI shim counts thinking tokens in total_tokens - prompt_tokens.
                    reply["usage"]["completion_tokens"] = total - prompt
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
                    blocks.append({
                        "type": "tool_use",
                        "id": str(tc.get("id") or "call_0"),
                        "name": str(fn.get("name") or ""),
                        "input": _parse_tool_input(fn.get("arguments") or ""),
                    })
                anth_msgs.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                append_user_block({
                    "type": "tool_result",
                    "tool_use_id": str(msg["tool_call_id"]),
                    "content": msg.get("content") or "",
                })

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
                        calls.append({"id": str(blk.get("id") or f"call_{i}"), "name": str(blk.get("name") or ""), "arguments_raw": raw})
                    elif btype == "text" and blk.get("text"):
                        texts.append(blk["text"])
                u = data.get("usage") or {}
                in_tok = int(u.get("input_tokens") or 0)
                cw_tok = int(u.get("cache_creation_input_tokens") or 0)
                cr_tok = int(u.get("cache_read_input_tokens") or 0)
                out_tok = int(u.get("output_tokens") or 0)
                p_in, p_out, p_cr, p_cw = self.price
                cost = (in_tok * p_in + out_tok * p_out + cr_tok * p_cr + cw_tok * p_cw) / 1e6
                self.spend.add(cost)
                return {
                    "tool_calls": calls,
                    "content": "\n".join(texts) if texts else None,
                    "usage": {"prompt_tokens": in_tok + cw_tok + cr_tok, "completion_tokens": out_tok},
                }
            if not (status == 429 or status >= 500):
                raise ModelError(f"HTTP {status}: {detail}")
            if attempt >= self.max_retries:
                raise ModelError(f"HTTP {status} after {attempt} retries: {detail}")
            time.sleep(BACKOFF_S[attempt])
            attempt += 1


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line.strip())
        if m:
            os.environ.setdefault(m.group(1), m.group(2).strip().strip('"'))


def warm_modal_if_needed(url: str, timeout_s: float = 900) -> None:
    """Modal scale-to-zero vLLM servers return 503 while cold-starting H200 + 55 GB weights."""
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


def make_client(cfg, spend: Spend, max_retries: int = 5):
    cid, url, model, key, extra, max_tokens = cfg
    if url == ANTHROPIC_MESSAGES_URL:
        return AnthropicMessagesClient(
            url,
            model,
            key,
            effort=extra.get("reasoning_effort"),
            max_tokens=max_tokens,
            timeout_s=300,
            max_retries=max_retries,
            price=ANTHROPIC_PRICE[model],
            spend=spend,
        )
    return ProbeClient(url, model, key, max_tokens=max_tokens, timeout_s=300, max_retries=max_retries, extra=extra)


def ping(args, spend: Spend) -> None:
    tool = {"type": "function", "function": {"name": "advance", "description": "Advance time.",
            "parameters": {"type": "object", "properties": {}}}}  # fmt: skip
    msgs = [{"role": "user", "content": "Call the advance tool once."}]
    for cfg in CONFIGS:
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


def committed_episodes(preset: str, set_name: str, seeds: list[int]) -> list[EpisodeSpec]:
    """Read the committed seed list instead of re-running the seed filter (which spawns a process pool)."""
    from railway.gen.generator import generate
    from railway.gen.presets import PRESETS
    from railway.util import content_hash

    listing = json.loads((ROOT / "seeds" / f"{preset}-{set_name}.json").read_text())
    accepted = {a["seed"]: a for a in listing["accepted"]}
    out = []
    for seed in seeds:
        acc = accepted[seed]
        scenario = generate(PRESETS[preset], seed)
        if content_hash(scenario.model_dump(mode="json")) != acc["scenario_hash"]:
            raise ValueError(f"{preset} seed {seed}: scenario hash differs from the committed list")
        out.append(EpisodeSpec(scenario, float(acc["c_greedy"]), float(acc["c_ref"])))
    return out


def run(args, spend: Spend) -> None:
    episodes = committed_episodes(args.preset, args.set, args.seeds)
    eval_run.episodes_from_scenarios = lambda *_a, **_k: episodes
    for cfg in CONFIGS:
        if args.only and cfg[0] not in args.only:
            continue
        cid, url, model, key, extra, max_tokens = cfg
        name = f"probe-{args.preset.lower()}-s{'-'.join(map(str, args.seeds))}-{cid}"
        manifest = Manifest(
            name=name, harness="loop", mode="windowed",
            model={"id": cid, "name": model, "base_url": url, "api_key_env": key},
            max_tokens=max_tokens, scenarios={"preset": args.preset, "set": args.set, "count": args.count},
            reps=1, concurrency=1,
        )  # fmt: skip
        warm_modal_if_needed(url)
        deps = LoopDeps(client_factory=lambda session, ep, cfg=cfg: make_client(cfg, spend))
        print(f"RUN  {cid} (anthropic spend so far ${spend.usd:.2f})", flush=True)
        report = run_manifest(manifest, results_dir=ROOT / "results" / "probe", deps=deps)
        print(f"DONE {cid} {report} (anthropic spend ${spend.usd:.2f})", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["ping", "run"])
    p.add_argument("--only", nargs="*")
    p.add_argument("--preset", default="T1")
    p.add_argument("--set", default="dev")
    p.add_argument("--seeds", type=int, nargs="+", default=[5])
    p.add_argument("--count", type=int, default=1)  # kept for the manifest; episodes come from --seeds
    args = p.parse_args()
    load_env()
    spend = Spend(ROOT / "results" / "probe" / "anthropic_spend.json")
    ping(args, spend) if args.cmd == "ping" else run(args, spend)


if __name__ == "__main__":
    main()
