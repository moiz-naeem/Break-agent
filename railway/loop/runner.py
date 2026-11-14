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
COUNT_FIELDS = ("gridlocks", "crew_timeouts", "misroutes", "missed_calls", "refusals", "invalid_calls", "decision_points")


@dataclass
class EpisodeRecord:
    """PRD §7.10 results row, plus `format_errors`."""

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
    c_greedy: float | None = None
    c_best: float | None = None
    score_clipped: float | None = None
    score_unclipped: float | None = None
    cost_ratio: float | None = None
    gridlocks: int | None = None
    crew_timeouts: int | None = None
    misroutes: int | None = None
    missed_calls: int | None = None
    kpis: dict | None = None
    unfinished: int | None = None
    refusals: int | None = None
    invalid_calls: int | None = None
    tool_calls: int | None = 0
    decision_points: int | None = None
    tokens_in: int | None = 0
    tokens_out: int | None = 0
    wall_s: float | None = None
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
    """(reward, score_unclipped) from whatever T07's `reward_from` returns (mapping, object or tuple)."""
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


def observe_current(session, error: bool = False) -> str:
    """The CURRENT block, read with `Session.peek` (free, unlogged), so context doesn't grow with the episode."""
    status = session.peek("status", {"compact": True})
    bulletins = session.peek("bulletins", {})
    notes = json.loads(session.peek("status", {})).get("notes", "")
    budget = json.loads(session.peek("kpis", {}))["budget"]
    return render_current(status, bulletins, notes, budget["used"], budget["total"], error)


def _tool_specs(session, disabled):
    from railway.tools.schemas import tool_schemas

    return tool_schemas(disabled, registry=getattr(session, "registry", None))


def _unfinished(session) -> int | None:
    try:
        state = json.loads(session.peek("status", {}))
        return sum(1 for t in state["trains"] if t.get("finished_tick") is None)
    except Exception:  # noqa: BLE001 - a closed session may refuse; then the count is unknown
        return None


def _record_outcome(rec: EpisodeRecord, session, final, score_fn) -> None:
    cost = _as_dict(_get(final, "cost"))
    rec.cost_by_component = cost
    rec.cost_total = cost.get("total")
    counts = _as_dict(_get(final, "counts"))
    try:
        kpi_counts = json.loads(session.peek("kpis", {})).get("counts", {})
    except Exception:  # noqa: BLE001
        kpi_counts = {}
    for key in COUNT_FIELDS:
        setattr(rec, key, counts.get(key, kpi_counts.get(key)))
    rec.unfinished = counts["unfinished_trains"] if "unfinished_trains" in counts else _unfinished(session)
    kpis = _get(final, "kpis")
    rec.kpis = dict(kpis) if kpis else None
    if rec.c_best:
        rec.cost_ratio = rec.cost_total / rec.c_best
    if score_fn is not None and rec.c_greedy is not None and rec.c_best is not None:
        rec.score_clipped, rec.score_unclipped = normalize_reward(score_fn(rec.c_greedy, rec.c_best, rec.cost_total))


def _loop(rec, session, model_client, mode, system, tools, max_turns):
    manual = session.peek("manual", {})
    turns: list[tuple[dict, dict]] = []
    strikes = 0
    for n in range(1, max_turns + 1):
        if session.closed:
            break
        current = observe_current(session, error=strikes > 0)
        reply = model_client.chat(build_messages(system, manual, turns, current, mode), tools)
        rec.tokens_in += reply["usage"]["prompt_tokens"]
        rec.tokens_out += reply["usage"]["completion_tokens"]
        calls = reply["tool_calls"]
        entry = {
            "turn": n,
            "content": reply["content"],
            "tool_calls": calls,
            "executed": None,
            "output": None,
            "format_error": len(calls) != 1,
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
    run: str | None = None,
    model: str | None = None,
    rep: int = 0,
    c_greedy: float | None = None,
    c_best: float | None = None,
    score_fn=None,
) -> EpisodeRecord:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    start = time.monotonic()
    rec = EpisodeRecord(
        run=run, mode=mode, model=model, rep=rep, c_greedy=c_greedy, c_best=c_best, **scenario_meta(scenario)
    )
    if session is None:
        session = make_session(scenario, disabled=disabled, diagnostics=diagnostics, session_cls=session_cls)
    try:
        system = system_message(load_instruction() if instruction is None else instruction)
        _loop(rec, session, model_client, mode, system, _tool_specs(session, disabled), max_turns)
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
    return rec
