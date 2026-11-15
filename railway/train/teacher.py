"""Teacher episodes from any harness -> per-decision SFT samples in the exact `railway/loop` prompt format.

The teacher (Opus 5.5 through Droid, run by Harbor) plays in its own harness: Droid's system prompt, its
full history, its own tool descriptions. None of that is what Qwen will see. What carries over is the
action sequence, `Session.episode_log()["entries"]` (Harbor's `verifier/action_log.json`, or the MCP
server's `/episode_log`). `render_episode` replays it on a fresh network `Session` and, before every
action, rebuilds the prompt `railway.loop.runner` would send at that point: system + MANUAL + the last
`WINDOW_TURNS` calls with their outputs + the CURRENT block, with the loop's own tool definitions. The
teacher's call is the target. The replay checks every logged (tick, status, rule) and the final cost.

Trace hygiene (why each step exists):
- `prune_reads`: Droid's coding-agent prompt makes Opus re-read a lot. A read repeated with the same
  arguments since the last state-changing call returns the same text, and `manual` is already in the
  prompt, so both are removed before rendering. Reads change nothing but the budget counter, so the
  replay and the final cost are unaffected. Other reads stay in as turns, so what Opus looked up is in
  Qwen's window.
- `normalize_args`: arguments in the loop's canonical form (`skip`/`short_turn` take a station node,
  not a platform: `B1` -> `B`), JSON with sorted keys.
- labels: only calls with status OK / GRANTED / INFO are targets. REFUSED / INVALID calls and anything
  after the session closed stay in the history (the state is what it is) but are never trained on, so a
  retry contributes only its corrected call.
- `grounded`: every identifier in a target's arguments must appear in the prompt Qwen sees. A decision
  that relies on something Opus read more than `WINDOW_TURNS` calls ago fails this and is dropped.
- `end_reason`: an episode counts only if the shift ran to its end or the teacher called `end_shift`;
  budget stops and timeouts (Harbor `exception_info`) are dropped by the dataset builder.
- `dedupe` (same prompt + same call, e.g. the opening reads across reps of one seed) and `rebalance`
  (cap the share of `advance_to_next_event`, which otherwise dominates the targets).

Selection (`select_episodes`) replaces the removed beat-greedy filter (decisions.md #4): per seed, the
cheapest teacher episodes, optionally only those cheaper than a baseline cost on the same seed.

`state_at(timetable, entries, k)` gives the session after the first `k` actions: the DAgger branch point.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from railway.loop.model_client import clean_openai_messages
from railway.loop.prompts import build_messages, system_message, turn_messages
from railway.loop.runner import _tool_specs, observe_current
from railway.train.samples import TurnSample

TRAINABLE_STATUSES = frozenset({"OK", "GRANTED", "INFO"})
# Tools that only read; they change nothing but the session's budget counter.
READ_TOOLS = frozenset({"manual", "network", "status", "conflicts", "timetable", "bulletins", "kpis", "stringline"})
ADVANCE = "advance_to_next_event"
# Targets whose arguments carry no identifiers to ground.
UNGROUNDED_OK = frozenset({ADVANCE, "end_shift", *READ_TOOLS})


class ReplayMismatch(Exception):
    pass


def net_instruction() -> str:
    from railway.net.manual import INSTRUCTION

    return INSTRUCTION


def canonical_arguments(args: dict) -> str:
    """The argument string the student is trained to emit: JSON, keys sorted."""
    return json.dumps(args or {}, sort_keys=True, ensure_ascii=False)


def normalize_args(session, tool: str, args: dict) -> dict:
    """The loop's canonical form of `args` (same effect in the engine)."""
    args = dict(args or {})
    if tool in ("skip", "short_turn") and isinstance(args.get("station"), str):
        from railway.net.session import _resolve_station_arg

        args["station"] = _resolve_station_arg(session, args["station"])
    return args


def prune_reads(entries: list[dict]) -> tuple[list[dict], int]:
    """Drop `manual` reads and reads repeated (same tool + args) since the last state-changing call."""
    kept: list[dict] = []
    seen: set[str] = set()
    for e in entries:
        tool = e["tool"]
        if tool in READ_TOOLS:
            if tool == "manual":
                continue
            key = tool + canonical_arguments(e.get("args") or {})
            if key in seen:
                continue
            seen.add(key)
        else:
            seen.clear()
        kept.append(e)
    return kept, len(entries) - len(kept)


def _identifiers(args: dict) -> list[str]:
    out: list[str] = []
    for v in args.values():
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            out.extend(x for x in v if isinstance(x, str))
    return [x for x in out if x]


def grounded(messages: list[dict], tool: str, args: dict) -> bool:
    """Every identifier argument of a decision appears somewhere in the prompt the student sees."""
    if tool in UNGROUNDED_OK or tool == "notes":
        return True
    visible = "\n".join(str(m.get("content") or "") for m in messages)
    return all(x in visible for x in _identifiers(args))


@dataclass
class RenderedEpisode:
    samples: list[TurnSample]
    final_cost: float
    final_tick: int
    final_cost_by_component: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    n_actions: int = 0
    pruned_reads: int = 0
    closed_by: str = "unknown"  # end_shift | shift_end | budget | open

    @property
    def targets(self) -> list[TurnSample]:
        return [s for s in self.samples if s.meta["target"]]


def _new_session(timetable, budget: int | None):
    from railway.net.session import Session

    return Session(timetable) if budget is None else Session(timetable, budget=budget)


def _check(entry: dict, got: dict, k: int) -> None:
    want = (entry.get("tick"), entry.get("status"), entry.get("rule"))
    have = (got["tick"], got["status"], got["rule"])
    if want != have:
        raise ReplayMismatch(f"action {k} {entry.get('tool')}: logged {want}, replayed {have}")


def state_at(timetable, entries: list[dict], k: int, *, budget: int | None = None):
    """A fresh session with the first `k` logged actions replayed (checked); the DAgger branch point."""
    session = _new_session(timetable, budget)
    for i, entry in enumerate(entries[:k]):
        session.call(entry["tool"], entry["args"])
        _check(entry, session.log[-1], i)
    return session


def render_episode(
    timetable,
    entries: list[dict],
    *,
    mode: str = "windowed",
    budget: int | None = None,
    disabled: tuple[str, ...] = (),
    prune: bool = True,
    expected_final_cost: float | None = None,
    meta: dict | None = None,
) -> RenderedEpisode:
    """Replay `entries` and emit one TurnSample per action, prompt rebuilt exactly as `railway/loop` does.

    A sample's `meta["target"]` says whether it is a training target (status OK/GRANTED/INFO, made while
    the session was open, and grounded in the visible prompt)."""
    pruned = 0
    if prune:
        entries, pruned = prune_reads(entries)
    session = _new_session(timetable, budget)
    system = system_message(net_instruction())
    manual = session.peek("manual", {})
    tools = _tool_specs(session, disabled)
    history: list[tuple[dict, dict]] = []
    samples: list[TurnSample] = []
    closed_by = "open"
    for k, entry in enumerate(entries):
        if session.closed:
            break  # calls after the end are refusals; the loop never sends them
        current = observe_current(session)
        messages = clean_openai_messages(build_messages(system, manual, history, current, mode))
        args = normalize_args(session, entry["tool"], entry["args"])
        args_raw = canonical_arguments(args)
        output = session.call(entry["tool"], args)
        _check(entry, session.log[-1], k)
        status = entry.get("status")
        is_grounded = grounded(messages, entry["tool"], args)
        call_index = len(history) + 1
        samples.append(
            TurnSample(
                turn=call_index,
                messages=messages,
                tools=tools,
                tool_calls=[{"id": f"call_{call_index}", "name": entry["tool"], "arguments_raw": args_raw}],
                meta={
                    "seq": entry.get("seq"),
                    "tick": entry.get("tick"),
                    "tool": entry["tool"],
                    "status": status,
                    "rule": entry.get("rule"),
                    "grounded": is_grounded,
                    "target": status in TRAINABLE_STATUSES and is_grounded,
                    **(meta or {}),
                },
            )
        )
        history.append(turn_messages(call_index, entry["tool"], args_raw, output))
        if session.closed:
            if entry["tool"] == "end_shift":
                closed_by = "end_shift"
            elif session.used >= session.budget:
                closed_by = "budget"
            else:
                closed_by = "shift_end"
    if not session.closed:
        closed_by = "open"  # the teacher stopped without end_shift; closing runs the remainder
    final = session.close()
    if expected_final_cost is not None and abs(final.cost.total - float(expected_final_cost)) > 1e-6:
        raise ReplayMismatch(f"final cost: logged {expected_final_cost}, replayed {final.cost.total}")
    return RenderedEpisode(
        samples=samples,
        final_cost=float(final.cost.total),
        final_tick=int(final.tick),
        final_cost_by_component=final.cost.model_dump(mode="json", exclude={"total"}),
        counts=dict(final.counts),
        n_actions=len(samples),
        pruned_reads=pruned,
        closed_by=closed_by,
    )


# -- dataset-level filters --------------------------------------------------------------------------


def sample_key(s: TurnSample) -> str:
    blob = json.dumps([s.messages, s.tool_calls[0]["name"], s.tool_calls[0]["arguments_raw"]], sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def dedupe(samples: Iterable[TurnSample]) -> list[TurnSample]:
    """Drop exact repeats (same prompt, same call), e.g. the opening reads of several reps of one seed."""
    seen: set[str] = set()
    out = []
    for s in samples:
        k = sample_key(s)
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def rebalance(samples: list[TurnSample], *, tool: str = ADVANCE, max_share: float = 0.3, seed: int = 0) -> list[TurnSample]:
    """Subsample `tool` targets so they are at most `max_share` of the result (order kept)."""
    hit = [i for i, s in enumerate(samples) if s.tool_calls[0]["name"] == tool]
    rest = len(samples) - len(hit)
    if not hit or max_share >= 1:
        return samples
    cap = int(max_share * rest / (1 - max_share)) if max_share > 0 else 0
    if len(hit) <= cap:
        return samples
    keep = set(random.Random(seed).sample(hit, cap))
    return [s for i, s in enumerate(samples) if i not in hit or i in keep]


# -- episode records and selection ------------------------------------------------------------------


@dataclass
class TeacherEpisode:
    """One teacher episode as stored: identity, outcome and the action log entries."""

    tier: str
    seed: int
    rep: int
    model: str
    harness: str
    status: str  # ok | error | timeout
    cost: float | None
    scenario_hash: str | None
    entries: list[dict]
    source: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.cost is not None and bool(self.entries)

    def to_json(self) -> dict:
        return dict(self.__dict__)


def select_episodes(
    episodes: Iterable[TeacherEpisode],
    *,
    per_seed: int = 1,
    baseline_cost: dict[tuple[str, int], float] | None = None,
    margin: float = 0.0,
) -> list[TeacherEpisode]:
    """Per (tier, seed): the `per_seed` cheapest OK episodes, and with `baseline_cost` only those cheaper
    than `baseline - margin` on that seed (a seed with no baseline keeps its best episodes)."""
    by_seed: dict[tuple[str, int], list[TeacherEpisode]] = defaultdict(list)
    for ep in episodes:
        if ep.ok:
            by_seed[(ep.tier, ep.seed)].append(ep)
    keep: list[TeacherEpisode] = []
    for key, eps in sorted(by_seed.items()):
        eps.sort(key=lambda e: e.cost)
        for ep in eps[:per_seed]:
            if baseline_cost is not None and key in baseline_cost and ep.cost >= baseline_cost[key] - margin:
                continue
            keep.append(ep)
    return keep


def write_samples_jsonl(path: Path, samples: Iterable[TurnSample]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for s in samples:
            fh.write(json.dumps(dict(s.__dict__), ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_samples_jsonl(path: Path) -> list[TurnSample]:
    with Path(path).open(encoding="utf-8") as fh:
        return [TurnSample(**json.loads(line)) for line in fh if line.strip()]
