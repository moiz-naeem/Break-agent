"""The episode loop of our own harness (T08 §2.4, Lane F): one tool call per turn, full transcripts.

Each turn: build the messages, `chat`, execute exactly one tool call through `Session.call`.
- Zero tool calls: a format error, no session call; the next CURRENT starts with the error line;
  3 consecutive zero-call replies stop the episode.
- More than one tool call: only the first runs; a format error.
- Arguments that aren't a JSON object are passed as `{"__raw__": ...}`, which the session refuses
  with R-BAD-ARGS (and which counts against the budget).
The episode ends when the session closes, after `max_turns`, or on the 3-strikes stop; then
`session.close()` runs (stopping counts as end_shift).

Every episode unconditionally writes a complete transcript bundle (`transcript.jsonl` with header,
per-turn deltas + reasoning + usage/cost, and footer; `actions.json`; `events.jsonl` when the
session/engine exposes an event log; and `result.json`), whether or not `transcript_path` is passed.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from railway.loop.model_client import clean_openai_messages
from railway.loop.prompts import (
    MODES,
    WINDOW_TURNS,
    build_messages,
    load_instruction,
    render_current,
    system_message,
    turn_messages,
)
from railway.loop.providers import client_provider, client_settings
from railway.loop.transcripts import (
    FORMAT_VERSION,
    episode_bundle_dir,
    extract_action_log,
    extract_event_log,
    reconstruct_turn_messages,
    utc_iso_now,
    write_episode_bundle,
)
from railway.sim.constants import TOOL_BUDGET
from railway.util import content_hash

MAX_STRIKES = 3
COUNT_FIELDS = ("gridlocks", "crew_timeouts", "reversals", "refusals", "invalid_calls", "decision_points")
USAGE_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "cache_write_tokens",
)


@dataclass
class EpisodeRecord:
    """PRD §7.10 results row, plus `format_errors`, reasoning/cache token counts, USD cost, and `transcript_dir`."""

    run: str | None = None
    harness: str = "loop"
    mode: str | None = None
    model: str | None = None
    provider: str | None = None
    rep: int = 0
    scenario_id: str | None = None
    scenario_hash: str | None = None
    seed: int | None = None
    preset: str | None = None
    rules_version: str | None = None
    cost_total: float | None = None
    cost_by_component: dict | None = None
    c_greedy: float | None = None
    c_ref: float | None = None
    score_clipped: float | None = None
    score_unclipped: float | None = None
    cost_ratio: float | None = None
    gridlocks: int | None = None
    crew_timeouts: int | None = None
    reversals: int | None = None
    unfinished: int | None = None
    refusals: int | None = None
    invalid_calls: int | None = None
    tool_calls: int | None = 0
    decision_points: int | None = None
    tokens_in: int | None = 0
    tokens_out: int | None = 0
    tokens_reasoning: int | None = 0
    tokens_cached: int | None = 0
    cost_usd: float | None = 0.0
    wall_s: float | None = None
    status: str = "ok"
    error: str | None = None
    format_errors: int | None = 0
    transcript_dir: str | None = None
    transcript: list[dict] = field(default_factory=list, repr=False)

    def to_row(self) -> dict:
        row = asdict(self)
        row.pop("transcript")
        return row


ROW_FIELDS = tuple(k for k in EpisodeRecord.__dataclass_fields__ if k != "transcript")


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_dict(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return dict(vars(obj))


def _is_net_timetable(scenario) -> bool:
    return _get(scenario, "tier") is not None and hasattr(scenario, "net") and hasattr(scenario, "trips")


def scenario_meta(scenario) -> dict:
    dumped = _as_dict(scenario)
    sc_hash = content_hash(dumped)
    preset = _get(scenario, "preset") or _get(scenario, "tier")
    seed = _get(scenario, "seed")
    default_sid = (
        f"{preset}-{seed}" if (preset is not None and seed is not None) else f"{preset or 'scn'}-{sc_hash[:12]}"
    )
    sid = _get(scenario, "scenario_id") or default_sid
    rules_version = _get(scenario, "rules_version")
    if rules_version is None and _get(scenario, "tier") is not None:
        rules_version = "0.3-net"
    return {
        "scenario_id": sid,
        "scenario_hash": sc_hash,
        "seed": seed,
        "preset": preset,
        "rules_version": rules_version,
    }


def normalize_reward(result) -> tuple[float, float]:
    """(reward, score_unclipped) from whatever T07's `reward_from` returns (mapping, object or tuple)."""
    if isinstance(result, tuple | list):
        return float(result[0]), float(result[1])
    return float(_get(result, "reward")), float(_get(result, "score_unclipped"))


def make_session(scenario, *, disabled=(), diagnostics=False, session_cls=None):
    if session_cls is None:
        if _is_net_timetable(scenario):
            try:
                from railway.net.session import Session as session_cls
            except ImportError:
                from railway.tools.session import Session as session_cls
        else:
            from railway.tools.session import Session as session_cls
    kwargs = {}
    if disabled:
        kwargs["disabled"] = tuple(disabled)
    if diagnostics:
        kwargs["diagnostics"] = True
    return session_cls(scenario, **kwargs)


def parse_args(raw: str) -> dict:
    if raw is None or not raw.strip():
        return {}  # many servers send "" for a call with no arguments
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {"__raw__": raw}
    return parsed if isinstance(parsed, dict) else {"__raw__": raw}


def observe_current(session, error: bool = False) -> str:
    """The CURRENT block, read with `Session.peek` (free, unlogged), so context doesn't grow with the episode."""
    status = session.peek("status", {"compact": True})
    bulletins = session.peek("bulletins", {})
    notes = json.loads(session.peek("status", {})).get("notes", "")
    budget = json.loads(session.peek("kpis", {}))["budget"]
    return render_current(status, bulletins, notes, used=budget["used"], total=budget["total"], error=error)


def _tool_specs(session, disabled):
    if hasattr(session, "tool_schemas") and callable(session.tool_schemas):
        return session.tool_schemas(disabled)
    from railway.tools.schemas import tool_schemas

    return tool_schemas(disabled, registry=getattr(session, "registry", None))


def _current_tick(session) -> int | None:
    for owner in (session, getattr(session, "sim", None), getattr(session, "engine", None)):
        if owner is not None and hasattr(owner, "tick"):
            val = getattr(owner, "tick", None)
            if isinstance(val, int):
                return val
    return None


def _safe_kpis(session) -> dict:
    try:
        return json.loads(session.peek("kpis", {}))
    except Exception:  # noqa: BLE001
        return {}


def _unfinished(session) -> int | None:
    try:
        state = json.loads(session.peek("status", {}))
        return sum(1 for t in state["trains"] if t.get("finished_tick") is None)
    except Exception:  # noqa: BLE001 - a closed session may refuse; then the count is unknown
        return None


def _normalize_usage(raw_usage: dict | None) -> dict:
    u = raw_usage or {}
    prompt = int(u.get("prompt_tokens") or 0)
    comp = int(u.get("completion_tokens") or 0)
    cached = int(u.get("cached_tokens") or 0)
    cw = int(u.get("cache_write_tokens") or 0)
    reason = int(u.get("reasoning_tokens") or 0)
    inp = int(u["input_tokens"]) if "input_tokens" in u else max(0, prompt - cached - cw)
    out = int(u["output_tokens"]) if "output_tokens" in u else max(0, comp - reason)
    cost = float(u.get("cost_usd") or 0.0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": comp,
        "input_tokens": inp,
        "output_tokens": out,
        "reasoning_tokens": reason,
        "cached_tokens": cached,
        "cache_write_tokens": cw,
        "cost_usd": cost,
    }


def _record_outcome(rec: EpisodeRecord, session, final, score_fn) -> dict:
    cost = _as_dict(_get(final, "cost"))
    rec.cost_by_component = cost
    rec.cost_total = cost.get("total")
    counts = _as_dict(_get(final, "counts"))
    kpis = _safe_kpis(session)
    kpi_counts = kpis.get("counts", {}) if isinstance(kpis, dict) else {}
    for key in COUNT_FIELDS:
        setattr(rec, key, counts.get(key, kpi_counts.get(key)))
    rec.unfinished = counts["unfinished"] if "unfinished" in counts else _unfinished(session)
    if rec.c_ref and rec.cost_total is not None:
        rec.cost_ratio = rec.cost_total / rec.c_ref
    if score_fn is not None and rec.c_greedy is not None and rec.c_ref is not None and rec.cost_total is not None:
        rec.score_clipped, rec.score_unclipped = normalize_reward(score_fn(rec.c_greedy, rec.c_ref, rec.cost_total))
    return kpis


def _loop(
    rec: EpisodeRecord, session, model_client, mode: str, system: str, manual: str, tools: list[dict], max_turns: int
) -> dict:
    history: list[tuple[int, dict, dict]] = []
    turns_by_num: dict[int, dict] = {}
    header_stub = {"system_prompt": system, "manual": manual}
    usage_totals: dict[str, int | float] = {k: 0 for k in USAGE_KEYS}
    usage_totals["cost_usd"] = 0.0

    strikes = 0
    for n in range(1, max_turns + 1):
        if session.closed:
            break
        tick_before = _current_tick(session)
        shown = history[-WINDOW_TURNS:] if mode == "windowed" else history
        history_turn_nums = [t_num for t_num, _, _ in shown]
        shown_pairs = [(a, t) for _, a, t in shown]

        current = observe_current(session, error=strikes > 0)
        messages = build_messages(system, manual, shown_pairs, current, mode)
        clean_msgs = clean_openai_messages(messages)

        started_at = utc_iso_now()
        t0 = time.monotonic()
        reply = model_client.chat(messages, tools)
        latency_s = round(time.monotonic() - t0, 4)

        usage = _normalize_usage(reply.get("usage"))
        rec.tokens_in = (rec.tokens_in or 0) + usage["prompt_tokens"]
        rec.tokens_out = (rec.tokens_out or 0) + usage["completion_tokens"]
        rec.tokens_reasoning = (rec.tokens_reasoning or 0) + usage["reasoning_tokens"]
        rec.tokens_cached = (rec.tokens_cached or 0) + usage["cached_tokens"]
        rec.cost_usd = round((rec.cost_usd or 0.0) + usage["cost_usd"], 8)
        for k in USAGE_KEYS:
            usage_totals[k] = int(usage_totals[k]) + int(usage[k])
        usage_totals["cost_usd"] = round(float(usage_totals["cost_usd"]) + float(usage["cost_usd"]), 8)

        calls = reply.get("tool_calls") or []
        format_error = len(calls) != 1
        format_error_kind = "no_tool_call" if not calls else ("multiple_tool_calls" if len(calls) > 1 else None)

        prompt_ref = {
            "from_header": True,
            "history_turns": history_turn_nums,
            "current": current,
        }
        entry: dict = {
            "type": "turn",
            "turn": n,
            "tick": tick_before,
            "started_at": started_at,
            "ended_at": started_at,
            "latency_s": latency_s,
            "prompt_ref": prompt_ref,
            "reasoning": reply.get("reasoning"),
            "content": reply.get("content"),
            "tool_calls": calls,
            "provider_blocks": reply.get("provider_blocks"),
            "executed": None,
            "output": None,
            "tool_outputs": [],
            "delta_messages": [],
            "usage": usage,
            "cost_usd": usage["cost_usd"],
            "format_error": format_error,
            "format_error_kind": format_error_kind,
            "finish_reason": reply.get("finish_reason"),
            "stop": None,
        }
        # Guarantee lossless reconstruction: fall back to inline `messages` only if `prompt_ref` ever differed.
        if reconstruct_turn_messages(header_stub, turns_by_num, entry) != clean_msgs:
            entry["messages"] = clean_msgs

        rec.transcript.append(entry)
        turns_by_num[n] = entry

        if not calls:
            rec.format_errors = (rec.format_errors or 0) + 1
            strikes += 1
            entry["ended_at"] = utc_iso_now()
            if strikes >= MAX_STRIKES:
                entry["stop"] = "three_strikes"
                break
            continue

        strikes = 0
        if len(calls) > 1:
            rec.format_errors = (rec.format_errors or 0) + 1

        call = calls[0]
        args = parse_args(call["arguments_raw"])
        output = session.call(call["name"], args)
        rec.tool_calls = (rec.tool_calls or 0) + 1

        call_idx = len(history) + 1
        call_id = f"call_{call_idx}"
        assistant_msg, tool_msg = turn_messages(
            call_idx,
            call["name"],
            call["arguments_raw"],
            output,
            provider_blocks=reply.get("provider_blocks"),
        )
        history.append((n, assistant_msg, tool_msg))

        entry["executed"] = {
            "tool": call["name"],
            "args": args,
            "arguments_raw": call["arguments_raw"],
            "call_id": call_id,
        }
        entry["output"] = output
        entry["tool_outputs"] = [{"call_id": call_id, "tool": call["name"], "output": output}]
        entry["delta_messages"] = clean_openai_messages([assistant_msg, tool_msg])
        entry["ended_at"] = utc_iso_now()

    return usage_totals


def play_episode(
    scenario,
    model_client,
    mode: str,
    *,
    disabled=(),
    max_turns: int = TOOL_BUDGET + 20,
    session=None,
    session_cls=None,
    diagnostics: bool = False,
    instruction: str | None = None,
    transcript_path: Path | None = None,
    transcript_root: Path | str | None = None,
    seed_set: str | None = None,
    run: str | None = None,
    model: str | None = None,
    rep: int = 0,
    c_greedy: float | None = None,
    c_ref: float | None = None,
    score_fn=None,
) -> EpisodeRecord:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    start_mono = time.monotonic()
    started_at = utc_iso_now()
    provider_name = client_provider(model_client)
    rec = EpisodeRecord(
        run=run,
        mode=mode,
        model=model,
        provider=provider_name,
        rep=rep,
        c_greedy=c_greedy,
        c_ref=c_ref,
        **scenario_meta(scenario),
    )
    if session is None:
        session = make_session(scenario, disabled=disabled, diagnostics=diagnostics, session_cls=session_cls)

    system = ""
    manual = ""
    tools: list[dict] = []
    usage_totals: dict[str, int | float] = {k: 0 for k in USAGE_KEYS}
    usage_totals["cost_usd"] = 0.0

    try:
        if instruction is None:
            if _is_net_timetable(scenario):
                try:
                    from railway.net.manual import INSTRUCTION as net_instruction

                    instruction = net_instruction
                except ImportError:
                    instruction = load_instruction()
            else:
                instruction = load_instruction()
        system = system_message(instruction)
        manual = session.peek("manual", {})
        tools = _tool_specs(session, disabled)
        usage_totals = _loop(rec, session, model_client, mode, system, manual, tools, max_turns)
    except Exception as exc:  # noqa: BLE001 - an infra failure is an `error` row, never a score
        rec.status, rec.error = "error", f"{type(exc).__name__}: {exc}"
        rec.tokens_in = None
        rec.tokens_out = None
        rec.tokens_reasoning = None
        rec.tokens_cached = None
        rec.cost_usd = None

    final = session.close()
    kpis: dict = {}
    if rec.status == "ok":
        kpis = _record_outcome(rec, session, final, score_fn)
    else:
        kpis = _safe_kpis(session)
    rec.wall_s = round(time.monotonic() - start_mono, 3)
    ended_at = utc_iso_now()

    model_name = str(getattr(model_client, "model", None) or model or "unknown")
    model_label = model or model_name
    resolved_seed_set = seed_set or _get(scenario, "seed_set") or _get(scenario, "set") or "unspecified"

    bundle_dir = episode_bundle_dir(
        transcript_root,
        run=run,
        model=model_label,
        seed_set=resolved_seed_set,
        scenario_id=rec.scenario_id,
        rep=rep,
    )
    rec.transcript_dir = str(bundle_dir)

    settings = {
        **client_settings(model_client),
        "mode": mode,
        "max_turns": max_turns,
        "disabled_tools": list(disabled),
        "diagnostics": diagnostics,
    }
    header = {
        "type": "header",
        "format_version": FORMAT_VERSION,
        "run": run,
        "model": model_label,
        "model_name": model_name,
        "provider": provider_name,
        "settings": settings,
        "seed_set": resolved_seed_set,
        "preset": rec.preset,
        "seed": rec.seed,
        "scenario_id": rec.scenario_id,
        "scenario_hash": rec.scenario_hash,
        "rep": rep,
        "rules_version": rec.rules_version,
        "manual_hash": content_hash(manual),
        "manual": manual,
        "system_prompt": system,
        "tools": tools,
        "started_at": started_at,
    }
    counts_summary = {k: getattr(rec, k) for k in (*COUNT_FIELDS, "unfinished")}
    reward = -float(rec.cost_total) if rec.cost_total is not None else None
    footer = {
        "type": "footer",
        "status": rec.status,
        "error": rec.error,
        "ended_at": ended_at,
        "wall_s": rec.wall_s,
        "turns": len(rec.transcript),
        "tool_calls": rec.tool_calls,
        "format_errors": rec.format_errors,
        "cost_total": rec.cost_total,
        "reward": reward,
        "cost_by_component": rec.cost_by_component,
        "c_greedy": rec.c_greedy,
        "c_ref": rec.c_ref,
        "score_clipped": rec.score_clipped,
        "score_unclipped": rec.score_unclipped,
        "kpis": kpis,
        "counts": counts_summary,
        "usage_total": usage_totals,
    }

    action_log = extract_action_log(session, rec, final)
    event_log = extract_event_log(session)
    extra_path = Path(transcript_path) if transcript_path is not None else None
    write_episode_bundle(
        bundle_dir,
        header=header,
        turns=rec.transcript,
        footer=footer,
        action_log=action_log,
        event_log=event_log,
        result_row=rec.to_row(),
        extra_transcript_path=extra_path,
    )

    if os.environ.get("RAILWAY_UPLOAD_TRANSCRIPTS") == "1":
        from railway.loop.transcripts import upload_episode_to_volume

        upload_episode_to_volume(bundle_dir, root=transcript_root)

    return rec
