"""The episode loop of our own harness (T08 §2.4): one tool call per turn, strict format.

Each turn: build the messages, `chat`, execute exactly one tool call through `Session.call`.
- Zero tool calls: a format error, no session call; the next CURRENT starts with the error line;
  3 consecutive zero-call replies stop the episode.
- More than one tool call: only the first runs; a format error.
- Arguments that aren't a JSON object are passed as `{"__raw__": ...}`, which the session refuses
  with R-BAD-ARGS (and which counts against the budget).
The episode ends when the session closes, after `max_turns`, or on the 3-strikes stop; then
`session.close()` runs (stopping counts as end_shift).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from railway.loop.prompts import MODES, build_messages, load_instruction, render_current, system_message, turn_messages
from railway.sim.constants import TOOL_BUDGET
from railway.util import content_hash

MAX_STRIKES = 3
COUNT_FIELDS = (
    "gridlocks",
    "crew_timeouts",
    "misroutes",
    "wrong_terminals",
    "missed_calls",
    "missed_connections",
    "unfinished_trains",
    "refusals",
    "invalid_calls",
    "decision_points",
)


@dataclass
class EpisodeRecord:
    """PRD §7.10 results row (v0.3), plus `format_errors` and run telemetry.

    `reward` is `-(C1 + C2 + C3 + C4 + C5)` (`-cost_total`).
    `score_normalized` (`max(0, u)`) and `score_unclipped` (`u`) are report-only comparisons against
    `c_greedy` and `c_best` (`score_clipped` mirrors `score_normalized` for T08 compatibility).
    """

    run: str | None = None
    harness: str = "loop"
    mode: str | None = None
    model: str | None = None
    rep: int = 0
    scenario_id: str | None = None
    scenario_hash: str | None = None
    seed: int | None = None
    preset: str | None = None
    rules_version: str | None = None
    cost_total: float | None = None
    cost_by_component: dict | None = None
    reward: float | None = None
    c_greedy: float | None = None
    c_best: float | None = None
    c_ref: float | None = None
    best_source: str | None = None
    best_proven: bool | None = None
    score_normalized: float | None = None
    score_clipped: float | None = None
    score_unclipped: float | None = None
    above_best: bool | None = None
    cost_ratio: float | None = None
    gridlocks: int | None = None
    crew_timeouts: int | None = None
    misroutes: int | None = None
    wrong_terminals: int | None = None
    missed_calls: int | None = None
    missed_connections: int | None = None
    unfinished_trains: int | None = None
    kpis: dict | None = None
    unfinished: int | None = None
    refusals: int | None = None
    invalid_calls: int | None = None
    tool_calls: int | None = 0
    decision_points: int | None = None
    tokens_in: int | None = 0
    tokens_out: int | None = 0
    tokens_cached: int | None = 0
    tokens_thinking: int | None = 0
    usd: float | None = None
    wall_s: float | None = None
    llm_s: float | None = 0.0
    prompt_stats: dict | None = None
    status: str = "ok"
    error: str | None = None
    format_errors: int | None = 0
    transcript: list[dict] = field(default_factory=list, repr=False)

    def to_row(self) -> dict:
        row = asdict(self)
        row.pop("transcript")
        return row


ROW_FIELDS = tuple(k for k in EpisodeRecord.__dataclass_fields__ if k != "transcript")


def scenario_meta(scenario) -> dict:
    return {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": content_hash(scenario.model_dump(mode="json")),
        "seed": scenario.seed,
        "preset": scenario.preset,
        "rules_version": scenario.rules_version,
    }


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


def normalize_reward(result) -> tuple[float, float]:
    """(score_normalized, score_unclipped) from whatever T07's `reward_from` returns (mapping, object or tuple)."""
    if isinstance(result, tuple | list):
        return float(result[0]), float(result[1])
    normalized = _get(result, "score_normalized")  # v0.3: the reward itself is −cost; rows keep the normalised score
    return float(normalized if normalized is not None else _get(result, "reward")), float(_get(result, "score_unclipped"))


def make_session(scenario, *, disabled=(), diagnostics=False, session_cls=None):
    if session_cls is None:
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


def _observe_parts(session, error: bool = False) -> tuple[str, dict[str, int]]:
    status = session.peek("status", {"compact": True})
    bulletins = session.peek("bulletins", {})
    notes = json.loads(session.peek("status", {})).get("notes", "")
    budget = json.loads(session.peek("kpis", {}))["budget"]
    cur = render_current(status, bulletins, notes, budget["used"], budget["total"], error)
    sizes = {
        "status_compact": len(status),
        "bulletins": len(bulletins),
        "notes": len(notes),
        "current": len(cur),
    }
    return cur, sizes


def observe_current(session, error: bool = False) -> str:
    """The CURRENT block, read with `Session.peek` (free, unlogged), so context doesn't grow with the episode."""
    return _observe_parts(session, error=error)[0]


def _tool_specs(session, disabled):
    from railway.tools.schemas import tool_schemas

    return tool_schemas(disabled, registry=getattr(session, "registry", None))


def _unfinished(session) -> int | None:
    try:
        state = json.loads(session.peek("status", {}))
        return sum(1 for t in state["trains"] if t.get("finished_tick") is None)
    except Exception:  # noqa: BLE001 - a closed session may refuse; then the count is unknown
        return None


def _snapshot_decision(session) -> dict | None:
    sim = getattr(session, "sim", None)
    if sim is None:
        return None
    state = sim.state()
    trains = []
    for t in state.trains:
        if t.phase in ("not_entered", "finished"):
            continue
        trains.append(
            {
                "id": t.id,
                "type": t.type,
                "phase": t.phase,
                "station": t.station,
                "track": t.track,
                "section": t.section,
                "section_from": t.section_from,
                "section_enter": t.section_enter,
                "section_exit": t.section_exit,
                "delay": t.delay,
                "crew_remaining": t.crew_remaining,
                "off_route": t.off_route,
                "auth": f"{t.authority.dest_station}/{t.authority.dest_track}" if t.authority else None,
            }
        )
    return {
        "tick": state.tick,
        "cost_total": round(state.cost.total, 4),
        "gridlock": list(state.gridlock),
        "trains": trains,
    }


def build_position_segments(scenario, session) -> list[dict]:
    """Exact continuous [t0, t1] dwell and section-traversal intervals per train across the shift."""
    history = getattr(session, "history", None)
    sim = getattr(session, "sim", None)
    if history is None or sim is None:
        return []
    sections = {s.id: (s.west, s.east) for s in scenario.network.sections}
    terminals = set(scenario.network.terminal_ids)
    final_state = sim.state()
    final_tick = final_state.tick
    by_train_state = {t.id: t for t in final_state.trains}

    out = []
    for tdef in scenario.trains:
        tid = tdef.id
        tstate = by_train_state.get(tid)
        events = [c for c in history if c.train == tid]
        segments: list[dict] = []
        cur: dict | None = None
        cur_node: str | None = None

        def close_cur(tick: int) -> None:
            nonlocal cur
            if cur is not None:
                cur["t1"] = tick
                segments.append(cur)
                cur = None

        for c in events:
            if c.kind == "ENTRY":
                close_cur(c.tick)
                node = c.text.rsplit(" ", 1)[1]
                cur_node = node
                cur = {"kind": "at_node", "node": node, "track": None, "t0": c.tick, "t1": None}
            elif c.kind == "ENTER":
                close_cur(c.tick)
                sec_id = c.text.rsplit(" ", 1)[1]
                w, e = sections.get(sec_id, (cur_node, None))
                from_node = cur_node if cur_node in (w, e) else w
                to_node = e if from_node == w else w
                cur = {
                    "kind": "in_section",
                    "section": sec_id,
                    "from_node": from_node,
                    "to_node": to_node,
                    "t0": c.tick,
                    "t1": None,
                }
                cur_node = to_node
            elif c.kind == "ARRIVE":
                close_cur(c.tick)
                place = c.text.rsplit(" ", 1)[1]
                node, track = place.split("/", 1) if "/" in place else (place, None)
                cur_node = node
                if tstate is not None and tstate.finished_tick == c.tick and node in terminals:
                    segments.append({"kind": "finished", "node": node, "track": track, "t0": c.tick, "t1": c.tick})
                else:
                    cur = {"kind": "at_node", "node": node, "track": track, "t0": c.tick, "t1": None}

        if cur is not None:
            end_t = tstate.finished_tick if (tstate and tstate.finished_tick is not None) else final_tick
            cur["t1"] = end_t
            if cur["kind"] == "in_section" and tstate and tstate.section_exit is not None:
                cur["projected_exit_tick"] = tstate.section_exit
            segments.append(cur)

        out.append(
            {
                "id": tid,
                "type": tdef.type,
                "origin": tdef.origin,
                "destination": tdef.destination,
                "entry_tick": tdef.entry_tick,
                "route": list(tdef.route or []),
                "outcome": tstate.outcome if tstate else None,
                "finished_tick": tstate.finished_tick if tstate else None,
                "delay": tstate.delay if tstate else 0,
                "off_route": tstate.off_route if tstate else False,
                "arrivals": [a.model_dump(mode="json") for a in tstate.arrivals] if tstate else [],
                "segments": segments,
            }
        )
    return out


def _write_trajectory(path: Path, rec: EpisodeRecord, scenario, session, snapshots: list[dict]) -> None:
    history = getattr(session, "history", None)
    if history is None:
        return
    net = scenario.network.model_dump(mode="json")
    turns = [
        {
            "turn": e["turn"],
            "tick": e.get("tick"),
            "tool": (e.get("executed") or {}).get("tool"),
            "args": (e.get("executed") or {}).get("args"),
            "format_error": e.get("format_error", False),
            "latency_s": e.get("latency_s"),
            "usage": e.get("usage"),
            "chars": e.get("chars"),
        }
        for e in rec.transcript
    ]
    payload = {
        "scenario_id": rec.scenario_id,
        "scenario_hash": rec.scenario_hash,
        "preset": rec.preset,
        "seed": rec.seed,
        "rules_version": rec.rules_version,
        "shift_ticks": scenario.shift_ticks,
        "run": rec.run,
        "model": rec.model,
        "rep": rec.rep,
        "summary": {
            "reward": rec.reward,
            "cost_total": rec.cost_total,
            "cost_by_component": rec.cost_by_component,
            "c_greedy": rec.c_greedy,
            "c_best": rec.c_best,
            "c_ref": rec.c_ref,
            "best_source": rec.best_source,
            "best_proven": rec.best_proven,
            "score_normalized": rec.score_normalized,
            "score_unclipped": rec.score_unclipped,
            "above_best": rec.above_best,
            "kpis": rec.kpis,
            "tokens_in": rec.tokens_in,
            "tokens_out": rec.tokens_out,
            "tokens_cached": rec.tokens_cached,
            "tokens_thinking": rec.tokens_thinking,
            "usd": rec.usd,
            "wall_s": rec.wall_s,
            "llm_s": rec.llm_s,
            "tool_calls": rec.tool_calls,
            "decision_points": rec.decision_points,
            "prompt_stats": rec.prompt_stats,
        },
        "network": {k: net[k] for k in ("terminals", "stations", "sections", "yards") if k in net},
        "trains": build_position_segments(scenario, session),
        "movements": [c.model_dump(mode="json") for c in history],
        "decision_snapshots": snapshots,
        "actions": list(session.log),
        "turns": turns,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _record_outcome(rec: EpisodeRecord, session, final, score_fn) -> None:
    cost = _as_dict(_get(final, "cost"))
    rec.cost_by_component = cost
    rec.cost_total = cost.get("total")
    if rec.cost_total is not None:
        rec.reward = round(-float(rec.cost_total), 6)
    counts = _as_dict(_get(final, "counts"))
    try:
        kpi_counts = json.loads(session.peek("kpis", {})).get("counts", {})
    except Exception:  # noqa: BLE001
        kpi_counts = {}
    for key in COUNT_FIELDS:
        setattr(rec, key, counts.get(key, kpi_counts.get(key)))
    rec.unfinished = counts["unfinished_trains"] if "unfinished_trains" in counts else _unfinished(session)
    if rec.unfinished_trains is None:
        rec.unfinished_trains = rec.unfinished
    kpis = _get(final, "kpis")
    rec.kpis = dict(kpis) if kpis else None
    if rec.c_best:
        rec.cost_ratio = rec.cost_total / rec.c_best
    if score_fn is not None and rec.c_greedy is not None and rec.c_best is not None:
        res = score_fn(rec.c_greedy, rec.c_best, rec.cost_total)
        norm, unclipped = normalize_reward(res)
        rec.score_normalized = norm
        rec.score_clipped = norm
        rec.score_unclipped = unclipped
        above = None if isinstance(res, tuple | list) else _get(res, "above_best")
        rec.above_best = bool(above) if above is not None else (unclipped > 1.0 + 1e-6)
    if rec.prompt_stats is not None:
        decisions = max(1, (rec.decision_points or 0) + 1)
        rec.prompt_stats["decision_points"] = rec.decision_points
        rec.prompt_stats["tokens_in_per_decision_total"] = round((rec.tokens_in or 0) / decisions, 1)


def _mean_int(vals: list[int | float]) -> float:
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def _loop(rec, session, model_client, mode, system, tools, max_turns) -> list[dict]:
    manual = session.peek("manual", {})
    manual_msg = f"MANUAL\n{manual}"
    system_chars = len(system)
    manual_chars = len(manual_msg)
    tools_chars = len(json.dumps(tools))

    turns: list[tuple[dict, dict]] = []
    snapshots: list[dict] = []
    if (snap0 := _snapshot_decision(session)) is not None:
        snapshots.append(snap0)

    strikes = 0
    usd_sum = 0.0
    has_usd = False
    at_decision_start = True
    decision_prompt_tokens: list[int] = []
    turn_prompt_tokens: list[int] = []
    status_sizes: list[int] = []
    window_sizes: list[int] = []
    bulletins_sizes: list[int] = []
    notes_sizes: list[int] = []

    for n in range(1, max_turns + 1):
        if session.closed:
            break
        tick_now = getattr(getattr(session, "sim", None), "tick", None)
        current, cur_sizes = _observe_parts(session, error=strikes > 0)
        msgs = build_messages(system, manual, turns, current, mode)
        win_chars = len(json.dumps(msgs[2:-1])) if len(msgs) > 3 else 0

        t0 = time.monotonic()
        reply = model_client.chat(msgs, tools)
        dt = round(time.monotonic() - t0, 3)
        rec.llm_s = round((rec.llm_s or 0.0) + dt, 3)

        usage = reply["usage"]
        p_tok = int(usage.get("prompt_tokens") or 0)
        c_tok = int(usage.get("completion_tokens") or 0)
        cache_tok = int(usage.get("cached_tokens") or 0)
        think_tok = int(usage.get("thinking_tokens") or 0)
        rec.tokens_in += p_tok
        rec.tokens_out += c_tok
        rec.tokens_cached += cache_tok
        rec.tokens_thinking += think_tok
        if usage.get("usd") is not None:
            usd_sum += float(usage["usd"])
            has_usd = True

        turn_prompt_tokens.append(p_tok)
        if at_decision_start:
            decision_prompt_tokens.append(p_tok)
        status_sizes.append(cur_sizes["status_compact"])
        window_sizes.append(win_chars)
        bulletins_sizes.append(cur_sizes["bulletins"])
        notes_sizes.append(cur_sizes["notes"])

        calls = reply["tool_calls"]
        entry = {
            "turn": n,
            "tick": tick_now,
            "content": reply["content"],
            "tool_calls": calls,
            "executed": None,
            "output": None,
            "format_error": len(calls) != 1,
            "latency_s": dt,
            "usage": dict(usage),
            "chars": {
                "system": system_chars,
                "manual": manual_chars,
                "tools": tools_chars,
                "window": win_chars,
                **cur_sizes,
            },
        }
        rec.transcript.append(entry)
        if not calls:
            rec.format_errors += 1
            strikes += 1
            if strikes >= MAX_STRIKES:
                entry["stop"] = "three_strikes"
                break
            continue
        strikes = 0
        if len(calls) > 1:
            rec.format_errors += 1
        call = calls[0]
        args = parse_args(call["arguments_raw"])
        output = session.call(call["name"], args)
        rec.tool_calls += 1
        turns.append(turn_messages(len(turns) + 1, call["name"], call["arguments_raw"], output))
        entry["executed"] = {"tool": call["name"], "args": args}
        entry["output"] = output
        if call["name"] == "advance_to_next_event":
            at_decision_start = True
            if (snap := _snapshot_decision(session)) is not None:
                snapshots.append(snap)
        else:
            at_decision_start = False

    if has_usd:
        rec.usd = round(usd_sum, 6)
    rec.prompt_stats = {
        "turns": len(turn_prompt_tokens),
        "tokens_in_per_turn_mean": _mean_int(turn_prompt_tokens),
        "tokens_in_per_turn_max": max(turn_prompt_tokens) if turn_prompt_tokens else 0,
        "tokens_in_at_decision_mean": _mean_int(decision_prompt_tokens),
        "tokens_in_at_decision_max": max(decision_prompt_tokens) if decision_prompt_tokens else 0,
        "chars": {
            "system": system_chars,
            "manual": manual_chars,
            "tools": tools_chars,
            "status_compact_mean": _mean_int(status_sizes),
            "status_compact_max": max(status_sizes) if status_sizes else 0,
            "window_mean": _mean_int(window_sizes),
            "window_max": max(window_sizes) if window_sizes else 0,
            "bulletins_max": max(bulletins_sizes) if bulletins_sizes else 0,
            "notes_max": max(notes_sizes) if notes_sizes else 0,
        },
    }
    return snapshots


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
    trajectory_path: Path | None = None,
    run: str | None = None,
    model: str | None = None,
    rep: int = 0,
    c_greedy: float | None = None,
    c_best: float | None = None,
    c_ref: float | None = None,
    best_source: str | None = None,
    best_proven: bool | None = None,
    score_fn=None,
) -> EpisodeRecord:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    start = time.monotonic()
    rec = EpisodeRecord(
        run=run,
        mode=mode,
        model=model,
        rep=rep,
        c_greedy=c_greedy,
        c_best=c_best,
        c_ref=c_ref,
        best_source=best_source,
        best_proven=best_proven,
        **scenario_meta(scenario),
    )
    if session is None:
        session = make_session(scenario, disabled=disabled, diagnostics=diagnostics, session_cls=session_cls)
    snapshots: list[dict] = []
    try:
        system = system_message(load_instruction() if instruction is None else instruction)
        snapshots = _loop(rec, session, model_client, mode, system, _tool_specs(session, disabled), max_turns)
    except Exception as exc:  # noqa: BLE001 - an infra failure is an `error` row, never a score
        rec.status, rec.error = "error", f"{type(exc).__name__}: {exc}"
    final = session.close()
    if rec.status == "ok":
        _record_outcome(rec, session, final, score_fn)
    rec.wall_s = round(time.monotonic() - start, 3)
    if transcript_path is not None:
        transcript_path = Path(transcript_path)
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with transcript_path.open("w", encoding="utf-8", newline="\n") as fh:
            for entry in rec.transcript:
                fh.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
    if trajectory_path is not None and rec.status == "ok":
        _write_trajectory(Path(trajectory_path), rec, scenario, session, snapshots)
    return rec
