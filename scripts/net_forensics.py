"""Failure forensics & thinking-cap diagnostics for v0.3 railway/net benchmark episodes.

Reads episode rows (`rows.jsonl`) and episode bundles (`transcript.jsonl`, `actions.json`,
`events.jsonl`, `result.json`) for Gemini Carbon and Qwen 3.8 and computes:

* thinking-budget & truncation stats (reasoning tokens per turn, share of turns at the 4,096 cap,
  format errors, invalid calls, refusals by rule);
* context-window & polling behaviour (share of read-only calls, read streaks without advancing
  time, identical re-reads at the same minute after the first read scrolled out of the 5-turn
  window, use of `notes()`);
* cost-loss attribution against the basic schedule-order policy
  (`booked_order(build_timetable(tier, seed))`) across C1..C5;
* infrastructure failures (provider errors) kept apart from model behaviour.

Usage (PowerShell, from the repo root):

    python scripts/net_forensics.py summary --rows <rows.jsonl> ... --bundles <dir> ... \
        --out results/forensics/forensics-summary.json
    python scripts/net_forensics.py trace <bundle_dir> [--actions-only] [--chars 400]
    python scripts/net_forensics.py diverge <bundle_dir_a> <bundle_dir_b>

Only `gemini-carbon-high`, `qwen38-think` and `qwen38-sft` episodes are analysed (see
`MODELS_IN_SCOPE`); everything else found under the given paths is ignored.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
try:  # an editable install elsewhere may shadow this checkout's package
    import railway

    railway.__path__ = [str(REPO / "railway")]
except ImportError:  # pragma: no cover - railway is always importable inside the repo
    railway = None

MODELS_IN_SCOPE = ("gemini-carbon-high", "qwen38-think", "qwen38-sft")
MODEL_LABELS = {
    "gemini-carbon-high": "Gemini 4 Carbon",
    "qwen38-think": "Qwen 3.8 27B (base)",
    "qwen38-sft": "Qwen 3.8 27B (SFT)",
}
READ_TOOLS = frozenset({"manual", "network", "status", "conflicts", "timetable", "bulletins", "kpis", "stringline"})
ACTION_TOOLS = frozenset(
    {"order", "hold", "platform", "route", "skip", "short_turn", "cancel", "single_line", "relieve_crew"}
)
TIME_TOOLS = frozenset({"advance_to_next_event", "end_shift"})
WINDOW = 5  # windowed mode keeps the last 5 turns of tool results in context
DEFAULT_CAP = 4096
CAP_NEAR = 3950
CAP_HARD = 4090
LENGTH_FINISH = frozenset({"length", "max_tokens", "MAX_TOKENS"})
COMPONENTS = ("C1", "C2", "C3", "C4", "C5")
COMPONENT_NAMES = {
    "C1": "passenger lateness",
    "C2": "missed stops + unfinished trains",
    "C3": "freight lateness",
    "C4": "crew time-outs",
    "C5": "gridlock + operations",
}


# --------------------------------------------------------------------------------------- loading


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:  # a live file may end mid-line
                    continue
    return out


@dataclass
class Episode:
    model: str
    scenario_id: str
    rep: int
    row: dict[str, Any] = field(default_factory=dict)
    bundle: Path | None = None
    header: dict[str, Any] | None = None
    turns: list[dict[str, Any]] = field(default_factory=list)
    footer: dict[str, Any] | None = None
    actions: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.model, self.scenario_id, self.rep)

    @property
    def tier(self) -> str:
        return self.scenario_id.split("-")[0]

    @property
    def seed(self) -> int:
        return int(self.scenario_id.split("-")[1])

    @property
    def error(self) -> str | None:
        return self.row.get("error") or (self.footer or {}).get("error")

    @property
    def infra_failed(self) -> bool:
        """Provider/infrastructure failure (HTTP errors, context overflow), not a dispatching decision."""
        err = self.error or ""
        return bool(err) and ("ModelError" in err or "HTTP" in err or "Timeout" in err)

    @property
    def complete(self) -> bool:
        return bool(self.row) and not self.infra_failed and self.row.get("status", "ok") == "ok"

    @property
    def thinking_cap(self) -> int | None:
        s = (self.header or {}).get("settings") or {}
        cap = s.get("thinking_budget") or (s.get("extra_body") or {}).get("thinking_token_budget")
        return int(cap) if cap else None


def load_bundle(path: Path) -> Episode | None:
    tpath = path / "transcript.jsonl"
    lines = _read_jsonl(tpath) if tpath.exists() else []
    header = next((x for x in lines if x.get("type") == "header"), None)
    result = {}
    if (path / "result.json").exists():
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    model = (header or {}).get("model") or result.get("model")
    sid = (header or {}).get("scenario_id") or result.get("scenario_id")
    rep = (header or {}).get("rep", result.get("rep"))
    if not model or not sid or rep is None:
        return None
    ep = Episode(model=model, scenario_id=sid, rep=int(rep), row=result, bundle=path, header=header)
    ep.turns = [x for x in lines if x.get("type") == "turn"]
    ep.footer = next((x for x in lines if x.get("type") == "footer"), None)
    if (path / "actions.json").exists():
        ep.actions = json.loads((path / "actions.json").read_text(encoding="utf-8")).get("entries", [])
    if (path / "events.jsonl").exists():
        ep.events = _read_jsonl(path / "events.jsonl")
    return ep


def discover(rows_files: list[Path], bundle_roots: list[Path]) -> dict[tuple[str, str, int], Episode]:
    eps: dict[tuple[str, str, int], Episode] = {}
    for root in bundle_roots:
        for tfile in sorted(root.rglob("transcript.jsonl")):
            ep = load_bundle(tfile.parent)
            if ep is None or ep.model not in MODELS_IN_SCOPE:
                continue
            old = eps.get(ep.key)
            # keep the most informative copy (more turns; a finished one beats a partial one)
            if old is None or (len(ep.turns), ep.footer is not None) > (len(old.turns), old.footer is not None):
                if old is not None and not ep.row:
                    ep.row = old.row
                eps[ep.key] = ep
    for rf in rows_files:
        if not rf.exists():
            continue
        for row in _read_jsonl(rf):
            if row.get("model") not in MODELS_IN_SCOPE:
                continue
            key = (row["model"], row["scenario_id"], int(row["rep"]))
            ep = eps.setdefault(key, Episode(model=key[0], scenario_id=key[1], rep=key[2]))
            ep.row = row  # rows.jsonl is the canonical scored record
    return eps


# --------------------------------------------------------------------------------------- per-turn


def turn_tool(t: dict[str, Any]) -> str | None:
    ex = t.get("executed")
    if ex:
        return ex.get("tool")
    calls = t.get("tool_calls") or []
    return calls[0].get("name") if len(calls) == 1 else None


def turn_args(t: dict[str, Any]) -> dict[str, Any]:
    ex = t.get("executed") or {}
    return ex.get("args") or {}


def thinking_tokens(t: dict[str, Any], reports_reasoning: bool) -> int:
    u = t.get("usage") or {}
    if reports_reasoning:
        return int(u.get("reasoning_tokens") or 0)
    # vLLM without a reasoning split: completion tokens = thinking + the tool call itself
    return int(u.get("completion_tokens") or u.get("output_tokens") or 0)


def output_status(output: str | None) -> tuple[str, str | None]:
    if not output:
        return ("", None)
    if output.startswith("REFUSED "):
        code = output.split(":", 1)[0].removeprefix("REFUSED ").strip()
        return ("REFUSED", code)
    return ("OK", None)


def _pct(n: float, d: float) -> float | None:
    return round(100.0 * n / d, 2) if d else None


def _dist(xs: list[float]) -> dict[str, float | int | None]:
    if not xs:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    s = sorted(xs)
    p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
    return {
        "n": len(s),
        "mean": round(statistics.fmean(s), 1),
        "median": round(statistics.median(s), 1),
        "p95": round(float(p95), 1),
        "max": round(float(s[-1]), 1),
    }


# --------------------------------------------------------------------------------------- metrics


def episode_metrics(ep: Episode) -> dict[str, Any]:
    """Behavioural metrics derived from one bundle's turns and action log."""
    reports_reasoning = any((t.get("usage") or {}).get("reasoning_tokens") for t in ep.turns)
    cap = ep.thinking_cap
    think = [thinking_tokens(t, reports_reasoning) for t in ep.turns]
    tools = [turn_tool(t) for t in ep.turns]

    near = hard = length_cut = overshoot = 0
    for t, n in zip(ep.turns, think, strict=True):
        fr = t.get("finish_reason")
        cut = fr in LENGTH_FINISH
        length_cut += cut
        # uncapped runs: the count shows how often a 4,096 cap *would* have cut the turn
        near += (n >= CAP_NEAR) or cut
        hard += (n >= CAP_HARD) or cut
        if cap and n > cap + 64 and not cut:  # budget ignored: thought past it and still answered
            overshoot += 1
    three_strikes = any(t.get("stop") == "three_strikes" for t in ep.turns)

    fmt = Counter(t.get("format_error_kind") for t in ep.turns if t.get("format_error"))

    # refusals / invalid calls by rule: action log is authoritative, transcript is the fallback
    refusals: Counter[str] = Counter()
    invalid: Counter[str] = Counter()
    if ep.actions:
        for e in ep.actions:
            if e.get("status") == "REFUSED":
                refusals[e.get("rule") or "?"] += 1
            elif e.get("status") == "INVALID":
                invalid[e.get("rule") or "?"] += 1
    else:
        for t in ep.turns:
            st, code = output_status(t.get("output"))
            if st == "REFUSED":
                (invalid if code in {"R-BAD-ARGS", "R-UNKNOWN-TRAIN", "R-UNKNOWN-ELEMENT"} else refusals)[code] += 1

    # polling: read streaks between non-read calls; identical re-reads at the same minute
    streaks: list[int] = []
    cur = 0
    for tool in tools:
        if tool in READ_TOOLS:
            cur += 1
        else:
            if cur:
                streaks.append(cur)
            cur = 0
    if cur:
        streaks.append(cur)

    seen: dict[tuple, int] = {}
    rereads_in_window = rereads_after_scroll = 0
    for i, t in enumerate(ep.turns):
        tool = tools[i]
        if tool not in READ_TOOLS:
            continue
        sig = (t.get("tick"), tool, json.dumps(turn_args(t), sort_keys=True))
        if sig in seen:
            if i - seen[sig] > WINDOW:
                rereads_after_scroll += 1
            else:
                rereads_in_window += 1
        seen[sig] = i
    # static references (never change during a shift) fetched again with identical arguments
    static_seen: Counter[tuple[str, str]] = Counter(
        (tool, json.dumps(turn_args(t), sort_keys=True))
        for t, tool in zip(ep.turns, tools, strict=True)
        if tool in {"network", "timetable", "manual"}
    )
    static_reads = {f"{k[0]}{k[1] if k[1] != '{}' else ''}": v for k, v in static_seen.items()}

    notes_calls = [t for t in ep.turns if turn_tool(t) == "notes"]
    notes_text = [str(turn_args(t).get("text", "")) for t in notes_calls]

    n = len(ep.turns)
    reads = sum(tool in READ_TOOLS for tool in tools)
    actions = sum(tool in ACTION_TOOLS for tool in tools)
    advances = sum(tool == "advance_to_next_event" for tool in tools)
    return {
        "turns": n,
        "thinking_source": "reasoning_tokens" if reports_reasoning else "completion_tokens",
        "thinking_cap": cap,
        "thinking_per_turn": think,
        "cap_near_turns": near,
        "cap_hard_turns": hard,
        "length_cut_turns": length_cut,
        "overshoot_turns": overshoot,
        "three_strikes_stop": three_strikes,
        "format_errors": dict(fmt),
        "refusals_by_rule": dict(refusals),
        "invalid_by_rule": dict(invalid),
        "tool_counts": dict(Counter(t for t in tools if t)),
        "reads": reads,
        "actions": actions,
        "advances": advances,
        "notes_calls": len(notes_calls),
        "notes_max_chars": max((len(x) for x in notes_text), default=0),
        "read_streaks": streaks,
        "rereads_in_window": rereads_in_window,
        "rereads_after_scroll": rereads_after_scroll,
        "static_reads": dict(static_reads),
        "max_tick": max((t.get("tick") or 0 for t in ep.turns), default=0),
        "gridlock_events": [
            {"tick": e.get("tick"), "trains": (e.get("detail") or {}).get("trains")}
            for e in ep.events
            if e.get("kind") == "GRIDLOCK"
        ],
        "unfinished_trains": sorted({e.get("train") for e in ep.events if e.get("kind") == "UNFINISHED"}),
    }


# --------------------------------------------------------------------------------------- baseline


def components_of(cost: dict[str, float]) -> dict[str, float]:
    return {
        "C1": cost.get("pax_lateness", 0.0),
        "C2": cost.get("missed_calls", 0.0) + cost.get("unfinished", 0.0),
        "C3": cost.get("freight_lateness", 0.0),
        "C4": cost.get("crew_timeout", 0.0),
        "C5": cost.get("gridlock", 0.0) + cost.get("operations", 0.0),
    }


_BASELINE_CACHE: dict[str, dict[str, Any]] = {}


def schedule_order_baseline(scenario_id: str) -> dict[str, Any]:
    """Cost of settling every conflict in booked timetable order (no look-ahead, no re-planning)."""
    if scenario_id in _BASELINE_CACHE:
        return _BASELINE_CACHE[scenario_id]
    from railway.net.exploits import booked_order, build_timetable

    tier, seed = scenario_id.split("-")
    res = booked_order(build_timetable(tier, int(seed)))
    cost = res.cost.model_dump() if hasattr(res.cost, "model_dump") else dict(res.cost)
    out = {
        "cost_total": round(float(cost["total"]), 3),
        "components": {k: round(v, 3) for k, v in components_of(cost).items()},
        "cost_by_component": cost,
        "counts": res.counts,
        "kpis": res.kpis,
    }
    _BASELINE_CACHE[scenario_id] = out
    return out


def failure_modes(row: dict[str, Any], base: dict[str, Any] | None) -> list[str]:
    modes = []
    cbc = row.get("cost_by_component") or {}
    kpis = row.get("kpis") or {}
    if (row.get("gridlocks") or 0) > 0:
        modes.append("gridlock")
    if kpis.get("freight_delivered_pct", 100.0) < 100.0:
        modes.append("stranded_freight")
    if (row.get("unfinished") or 0) > 0:
        modes.append("unfinished_trains")
    if cbc.get("missed_calls", 0) > 0:
        modes.append("missed_stops")
    if (row.get("crew_timeouts") or 0) > 0:
        modes.append("crew_timeout")
    if base is not None:
        if row["cost_total"] <= base["cost_total"]:
            modes.append("beat_schedule_order")
        elif cbc.get("pax_lateness", 0) > base["cost_by_component"].get("pax_lateness", 0) + 100:
            modes.append("extra_passenger_delay")
    return modes or ["clean"]


# --------------------------------------------------------------------------------------- summary


def summarize(eps: dict[tuple[str, str, int], Episode], with_baseline: bool) -> dict[str, Any]:
    by_group: dict[tuple[str, str], list[Episode]] = defaultdict(list)
    for ep in eps.values():
        by_group[(ep.model, ep.tier)].append(ep)

    episodes_out = []
    groups_out = []
    for (model, tier), group in sorted(by_group.items()):
        g_think: list[int] = []
        g = Counter()
        g_ref: Counter[str] = Counter()
        g_inv: Counter[str] = Counter()
        g_fmt: Counter[str] = Counter()
        g_modes: Counter[str] = Counter()
        g_streaks: list[int] = []
        g_delta = defaultdict(list)
        notes_eps = 0
        caps = set()
        for ep in sorted(group, key=lambda e: (e.seed, e.rep)):
            rec: dict[str, Any] = {
                "model": model,
                "scenario_id": ep.scenario_id,
                "rep": ep.rep,
                "complete": ep.complete,
                "infra_failed": ep.infra_failed,
                "error": (ep.error or "")[:160] or None,
                "has_transcript": bool(ep.turns),
                "reward": ep.row.get("reward"),
                "cost_total": ep.row.get("cost_total"),
                "components": ep.row.get("components"),
            }
            g["episodes"] += 1
            g["complete"] += ep.complete
            g["infra_failed"] += ep.infra_failed
            if ep.turns:
                m = episode_metrics(ep)
                caps.add(m["thinking_cap"])
                g["with_transcript"] += 1
                g["turns"] += m["turns"]
                g["cap_near"] += m["cap_near_turns"]
                g["cap_hard"] += m["cap_hard_turns"]
                g["length_cut"] += m["length_cut_turns"]
                g["overshoot"] += m["overshoot_turns"]
                g["three_strikes"] += m["three_strikes_stop"]
                g["reads"] += m["reads"]
                g["actions"] += m["actions"]
                g["advances"] += m["advances"]
                g["notes_calls"] += m["notes_calls"]
                g["rereads_in_window"] += m["rereads_in_window"]
                g["rereads_after_scroll"] += m["rereads_after_scroll"]
                g["static_rereads"] += sum(max(0, v - 1) for v in m["static_reads"].values())
                notes_eps += m["notes_calls"] > 0
                g_think += m["thinking_per_turn"]
                g_streaks += m["read_streaks"]
                g_ref.update(m["refusals_by_rule"])
                g_inv.update(m["invalid_by_rule"])
                g_fmt.update({k or "?": v for k, v in m["format_errors"].items()})
                rec["metrics"] = {k: v for k, v in m.items() if k not in ("thinking_per_turn", "read_streaks")}
                rec["metrics"]["thinking"] = _dist(m["thinking_per_turn"])
                rec["metrics"]["longest_read_streak"] = max(m["read_streaks"], default=0)
            if ep.complete:
                g["row_format_errors"] += ep.row.get("format_errors") or 0
                g["row_refusals"] += ep.row.get("refusals") or 0
                g["row_invalid"] += ep.row.get("invalid_calls") or 0
                g["row_tool_calls"] += ep.row.get("tool_calls") or 0
                base = schedule_order_baseline(ep.scenario_id) if with_baseline else None
                if base is not None:
                    rec["schedule_order"] = {"cost_total": base["cost_total"], "components": base["components"]}
                    comps = ep.row.get("components") or {}
                    delta = {c: round(comps.get(c, 0.0) - base["components"][c], 3) for c in COMPONENTS}
                    delta["total"] = round(ep.row["cost_total"] - base["cost_total"], 3)
                    rec["loss_vs_schedule_order"] = delta
                    for k, v in delta.items():
                        g_delta[k].append(v)
                    g["beat_schedule_order"] += ep.row["cost_total"] <= base["cost_total"]
                    g["strictly_beat"] += ep.row["cost_total"] < base["cost_total"] - 0.01
                    g["lost_by_100_plus"] += ep.row["cost_total"] > base["cost_total"] + 100
                modes = failure_modes(ep.row, base)
                rec["failure_modes"] = modes
                g_modes.update(modes)
            episodes_out.append(rec)

        turns = g["turns"]
        calls = g["reads"] + g["actions"] + g["advances"] + g["notes_calls"]
        groups_out.append(
            {
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "tier": tier,
                "episodes": g["episodes"],
                "complete_episodes": g["complete"],
                "infra_failed_episodes": g["infra_failed"],
                "episodes_with_transcript": g["with_transcript"],
                "thinking_caps_seen": sorted(c for c in caps if c) or [None],
                "thinking": {
                    **_dist(g_think),
                    "turns": turns,
                    "pct_turns_ge_3950": _pct(g["cap_near"], turns),
                    "pct_turns_ge_4090": _pct(g["cap_hard"], turns),
                    "length_cut_turns": g["length_cut"],
                    "pct_turns_over_budget_but_answered": _pct(g["overshoot"], turns),
                    "episodes_ended_by_three_strikes": g["three_strikes"],
                },
                "format_errors_by_kind": dict(g_fmt),
                "refusals_by_rule": dict(g_ref.most_common()),
                "invalid_by_rule": dict(g_inv.most_common()),
                "rows_totals_complete_eps": {
                    "tool_calls": g["row_tool_calls"],
                    "format_errors": g["row_format_errors"],
                    "refusals": g["row_refusals"],
                    "invalid_calls": g["row_invalid"],
                },
                "polling": {
                    "pct_calls_read_only": _pct(g["reads"], calls),
                    "pct_calls_actions": _pct(g["actions"], calls),
                    "pct_calls_advance": _pct(g["advances"], calls),
                    "reads_per_advance": round(g["reads"] / g["advances"], 2) if g["advances"] else None,
                    "read_streaks": _dist(g_streaks),
                    "read_streaks_ge_window": sum(s >= WINDOW for s in g_streaks),
                    "identical_rereads_same_minute_within_window": g["rereads_in_window"],
                    "identical_rereads_same_minute_after_scroll_out": g["rereads_after_scroll"],
                    "static_reference_rereads": g["static_rereads"],
                    "notes_calls": g["notes_calls"],
                    "pct_episodes_using_notes": _pct(notes_eps, g["with_transcript"]),
                },
                "failure_modes": dict(g_modes.most_common()),
                "beat_schedule_order": g["beat_schedule_order"] if with_baseline else None,
                "strictly_beat_schedule_order": g["strictly_beat"] if with_baseline else None,
                "lost_to_schedule_order_by_100_plus": g["lost_by_100_plus"] if with_baseline else None,
                "mean_loss_vs_schedule_order": (
                    {k: round(statistics.fmean(v), 1) for k, v in g_delta.items()} if g_delta else None
                ),
            }
        )
    return {
        "generated_by": "scripts/net_forensics.py",
        "models_in_scope": list(MODELS_IN_SCOPE),
        "thresholds": {"cap": DEFAULT_CAP, "near": CAP_NEAR, "hard": CAP_HARD, "window_turns": WINDOW},
        "component_names": COMPONENT_NAMES,
        "groups": groups_out,
        "schedule_order_baselines": _BASELINE_CACHE,
        "episodes": episodes_out,
    }


# --------------------------------------------------------------------------------------- trace


def _short(x: Any, n: int) -> str:
    s = x if isinstance(x, str) else json.dumps(x, separators=(",", ":"))
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def trace(bundle: Path, actions_only: bool, chars: int, start: int, end: int | None) -> None:
    ep = load_bundle(bundle)
    if ep is None:
        sys.exit(f"no episode bundle at {bundle}")
    reports = any((t.get("usage") or {}).get("reasoning_tokens") for t in ep.turns)
    print(f"# {ep.model} {ep.scenario_id} rep {ep.rep}  reward={ep.row.get('reward')}  turns={len(ep.turns)}")
    for t in ep.turns:
        i = t.get("turn")
        if i < start or (end is not None and i > end):
            continue
        tool = turn_tool(t)
        if actions_only and tool in READ_TOOLS:
            continue
        st, code = output_status(t.get("output"))
        flag = f" [{code}]" if code else ""
        fe = f" FORMAT_ERROR({t.get('format_error_kind')})" if t.get("format_error") else ""
        n = thinking_tokens(t, reports)
        print(
            f"\n## turn {i} t={t.get('tick')} {tool}({_short(turn_args(t), 200)}){flag}{fe} think={n}"
            f" finish={t.get('finish_reason')}"
        )
        if chars and t.get("reasoning"):
            print("  THINK: " + _short(t["reasoning"], chars))
        if chars and tool not in READ_TOOLS and t.get("output"):
            print("  OUT: " + _short(t["output"], min(chars, 300)))
    for g in episode_metrics(ep)["gridlock_events"]:
        print(f"\nGRIDLOCK at t={g['tick']}: {g['trains']}")


def diverge(a: Path, b: Path) -> None:
    ea, eb = load_bundle(a), load_bundle(b)
    if ea is None or eb is None:
        sys.exit("both paths must be episode bundles")

    def acts(ep: Episode) -> list[tuple[int, Any, str, str]]:
        return [
            (t["turn"], t.get("tick"), turn_tool(t), json.dumps(turn_args(t), sort_keys=True))
            for t in ep.turns
            if turn_tool(t) in ACTION_TOOLS
        ]

    xa, xb = acts(ea), acts(eb)
    for k, (pa, pb) in enumerate(zip(xa, xb, strict=False)):
        if (pa[1], pa[2], pa[3]) != (pb[1], pb[2], pb[3]):
            print(f"first divergent action #{k + 1}:")
            print(f"  A turn {pa[0]} t={pa[1]} {pa[2]} {pa[3]}")
            print(f"  B turn {pb[0]} t={pb[1]} {pb[2]} {pb[3]}")
            return
    print(f"no divergence in the first {min(len(xa), len(xb))} actions")


# --------------------------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary", help="aggregate forensics over rows + bundles")
    s.add_argument("--rows", nargs="*", type=Path, default=[])
    s.add_argument("--bundles", nargs="*", type=Path, default=[])
    s.add_argument("--out", type=Path, default=REPO / "results" / "forensics" / "forensics-summary.json")
    s.add_argument("--no-baseline", action="store_true", help="skip booked_order baselines")
    t = sub.add_parser("trace", help="print a turn-by-turn trace of one bundle")
    t.add_argument("bundle", type=Path)
    t.add_argument("--actions-only", action="store_true")
    t.add_argument("--chars", type=int, default=400)
    t.add_argument("--start", type=int, default=0)
    t.add_argument("--end", type=int, default=None)
    d = sub.add_parser("diverge", help="first differing action between two bundles of one scenario")
    d.add_argument("a", type=Path)
    d.add_argument("b", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "trace":
        trace(args.bundle, args.actions_only, args.chars, args.start, args.end)
    elif args.cmd == "diverge":
        diverge(args.a, args.b)
    else:
        eps = discover(args.rows, args.bundles)
        out = summarize(eps, with_baseline=not args.no_baseline)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        for grp in out["groups"]:
            th, po = grp["thinking"], grp["polling"]
            print(
                f"{grp['model']:<20} {grp['tier']} eps={grp['episodes']} complete={grp['complete_episodes']} "
                f"infra={grp['infra_failed_episodes']} tx={grp['episodes_with_transcript']} "
                f"think_med={th['median']} p95={th['p95']} max={th['max']} >=3950={th['pct_turns_ge_3950']}% "
                f"reads={po['pct_calls_read_only']}% notes_eps={po['pct_episodes_using_notes']}% "
                f"beat={grp['beat_schedule_order']} loss={grp['mean_loss_vs_schedule_order']}"
            )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
