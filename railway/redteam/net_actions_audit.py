"""Loophole audit of the network track's service-regulation tools (docs/redteam-net.md, section "Action tools").

`platform()` was the F1 hole (a call served at the wrong station). This audit asks the same question of the
other action tools, `route`, `short_turn`, `skip`, `cancel` and `single_line`: can any of them make a booked
call disappear without the bill, or move work somewhere it does not count?

Per seed it measures:

* **Call conservation** for every granted `route` / `short_turn` / `skip` / `cancel`: the booked calls the
  action removes from the train's plan are exactly the calls it emits `CALL_MISSED` for, at the moment it is
  granted, and every passenger `CALL_MISSED` is billed (`missed_calls` >= 500) when the shift is scored.
* **`route` exhaustively**: every passenger and freight train x every track and platform as a one-element
  `via`, at shift start and at mid-shift. Granted diversions must keep a freight train's destination (final
  call or exit), and their planned path must respect electrification, road length and track direction;
  via elements that break one of those must be refused. A sample of diversions that drop passenger calls is
  played to shift end to check the missed calls and broken connections are billed.
* **`short_turn`** at every booked station of every train: trains with no turnaround left must be refused
  with `R-NO-RETURN`; granted turns must bill every skipped stop.
* **No call is ever made at the wrong station** (`ARRIVE` with `call=j` at a node other than call `j`'s).
* **Strategic avoidance**: `cancel_all_freight`, `cancel_delayed_pax`, `short_turn_all_possible` and
  `skip_all_intermediate`, each layered on the same regulation loops as the honest dispatch references, are
  compared with the best honest dispatch (`dispatch` = cheapest of `booked_order`, `pax_first` and
  `freight_hold_sweep`; none of them withdraws a call or a train).

    python -m railway.redteam.net_actions_audit --tiers T1,T2,T3 --out results/redteam-net/actions-audit-dev.json
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from railway.net import exploits as X
from railway.net.cost import itemise
from railway.net.model import Platform, Timetable, Track
from railway.net.session import FinalResult, Session

REPO = Path(__file__).resolve().parents[2]
BIG_BUDGET = 10**7
DELAY_CANCEL_MIN = 15
ROUTE_PLAYOUTS = 4
Hook = Callable[[Session], bool]


def dev_seeds(tier: str) -> list[int]:
    """The accepted dev seeds of a tier, from `seeds/net-<tier>-dev.json`."""
    data = json.loads((REPO / "seeds" / f"net-{tier}-dev.json").read_text(encoding="utf-8"))
    return [int(row["seed"]) for row in data["accepted"]]


# ---------------------------------------------------------------- regulation loop


def regulate(s: Session, hook: Hook | None = None, *, stop_at: int | None = None, pax_first: bool = False) -> Session:
    """Booked-order regulation (as `exploits.booked_order`), with `hook` applied at shift start and each stop.

    `pax_first` orders every conflict passenger trains first, freight behind (freight still runs). With
    `stop_at`, returns the still-open session once `tick >= stop_at` (a mid-shift checkpoint).
    """
    freight = frozenset(t.train.id for t in s.timetable.trips if not t.train.passenger) if pax_first else frozenset()
    last_sig: tuple | None = None
    stalls = 0
    if hook:
        hook(s)
    X._relieve_warned_crews(s)
    for _ in range(X._probe_budget(s.timetable) * 4):
        if s.closed:
            break
        s.call("advance_to_next_event", {})
        if s.closed or (stop_at is not None and s.engine.tick >= stop_at):
            break
        sig = X._state_signature(s)
        acted = bool(hook and hook(s))
        acted = X._relieve_warned_crews(s) or acted
        acted = X._rescue_stranded(s) or acted
        acted = X._order_open_conflicts(s, deprioritize=freight) or acted
        if sig == last_sig and not acted:
            stalls += 1
            if stalls >= X.STALL_LIMIT:
                s.call("end_shift", {})
                break
        else:
            stalls, last_sig = 0, sig
    return s


def play(tt: Timetable, hook: Hook | None = None, *, pax_first: bool = False) -> tuple[FinalResult, Session]:
    s = regulate(Session(tt, budget=BIG_BUDGET), hook, pax_first=pax_first)
    return s.close(), s


# ---------------------------------------------------------------- strategic avoidance policies


def _live_ids(s: Session, passenger: bool | None = None) -> list[str]:
    out = []
    for tid, mv in sorted(s.engine.trains.items()):
        if mv.phase == "done" or mv.cancelled:
            continue
        if passenger is not None and mv.train.passenger != passenger:
            continue
        out.append(tid)
    return out


def _once(fn: Hook) -> Hook:
    done = [False]

    def hook(s: Session) -> bool:
        if done[0]:
            return False
        done[0] = True
        return fn(s)

    return hook


def _cancel_all_freight(s: Session) -> bool:
    return any(s.call("cancel", {"train": t}).startswith("OK") for t in _live_ids(s, passenger=False))


def _cancel_delayed_pax(s: Session) -> bool:
    acted = False
    for tid in _live_ids(s, passenger=True):
        if s.engine.trains[tid].lateness(s.engine.tick) >= DELAY_CANCEL_MIN:
            acted = s.call("cancel", {"train": tid}).startswith("OK") or acted
    return acted


def _stations(s: Session, tid: str) -> list[str]:
    mv = s.engine.trains[tid]
    return [s.timetable.net.element(c.platform).node for c in mv.plan]


def _short_turn_all_possible(s: Session) -> bool:
    acted = False
    for tid in _live_ids(s):
        for station in _stations(s, tid):
            if s.call("short_turn", {"train": tid, "station": station}).startswith("OK"):
                acted = True
                break
    return acted


def _skip_all_intermediate(s: Session) -> bool:
    acted = False
    for tid in _live_ids(s, passenger=True):
        for station in _stations(s, tid)[1:-1]:
            acted = s.call("skip", {"train": tid, "station": station}).startswith("OK") or acted
    return acted


# Honest dispatch references: settle every conflict, rescue stranded trains, relieve crews; nothing withdrawn.
AVOID_TOOLS = ("cancel", "skip", "short_turn")
DISPATCHERS = {"booked_order": False, "pax_first": True}  # name -> pax_first ordering
DISPATCH_REFS = (*DISPATCHERS, "freight_hold_sweep")
# Avoidance shortcuts, each layered on both dispatch references (the cheaper of the two counts).
POLICIES: dict[str, Callable[[], Hook]] = {
    "cancel_all_freight": lambda: _once(_cancel_all_freight),
    "cancel_delayed_pax": lambda: _cancel_delayed_pax,
    "short_turn_all_possible": lambda: _once(_short_turn_all_possible),
    "skip_all_intermediate": lambda: _once(_skip_all_intermediate),
}


def freight_hold_sweep(tt: Timetable, audit: Audit) -> tuple[float, dict | None]:
    """Honest dispatch that keeps freight out of the passenger peak: hold every freight train until `T` at
    shift start (it still runs and is delivered, late), for `T` on an 18-step grid, under both orderings."""
    freight = [t.train.id for t in tt.trips if not t.train.passenger]
    best: tuple[float, dict | None] = (float("inf"), None)
    if not freight:
        return best
    step = max(10, tt.shift_min // 18)
    for until in range(step, tt.shift_min, step):
        for name, pax_first in DISPATCHERS.items():

            def hold(s: Session, until: int = until) -> bool:
                return any(s.call("hold", {"train": f, "until": until}).startswith("OK") for f in freight)

            res, s = play(tt, _once(hold), pax_first=pax_first)
            if res.cost.total < best[0]:
                check_episode(tt, s, audit, f"freight_hold_sweep:{until}:{name}")
                best = (res.cost.total, {"until": until, "ordering": name})
    return best


# ---------------------------------------------------------------- invariants


class Audit:
    """Counters and the first few violations of each kind."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.violations: list[dict] = []

    def bad(self, kind: str, **info) -> None:
        self.counts[f"violation:{kind}"] += 1
        if len(self.violations) < 50:
            self.violations.append({"kind": kind, **info})


def _plan_calls(s: Session, tid: str) -> set[int]:
    return {c for c in s.engine.trains[tid].plan_calls if c is not None}


def act_checked(s: Session, audit: Audit, tool: str, args: dict, tid: str) -> tuple[str, list[int]]:
    """Run one action and check call conservation: removed booked calls == CALL_MISSED emitted by it."""
    mv = s.engine.trains[tid]
    before_plan, before_missed = _plan_calls(s, tid), set(mv.missed)
    n_ev = len(s.engine.log.events)
    out = s.call(tool, args)
    if not out.startswith(("OK", "GRANTED")):
        return out, []
    emitted = sorted(e.call for e in s.engine.log.events[n_ev:] if e.kind == "CALL_MISSED" and e.train == tid)
    removed = sorted(before_plan - _plan_calls(s, tid))
    new_missed = sorted(set(mv.missed) - before_missed)
    audit.counts[f"{tool}:granted"] += 1
    audit.counts[f"{tool}:calls_dropped"] += len(removed)
    if emitted != removed or new_missed != removed:
        audit.bad("call_conservation", tool=tool, args=args, removed=removed, emitted=emitted, missed=new_missed)
    return out, removed


def check_episode(tt: Timetable, s: Session, audit: Audit, label: str) -> None:
    """Whole-shift invariants on a closed session: no call made at the wrong station; every missed passenger
    call billed; every connection whose feeder call was dropped but whose connector left is billed."""
    net = tt.net
    trips = {t.train.id: t for t in tt.trips}
    events = s.engine.log.events
    for e in events:
        if e.kind == "ARRIVE" and e.call is not None:
            booked_node = net.element(trips[e.train].calls[e.call].platform).node
            if e.node != booked_node:
                audit.bad("call_at_wrong_station", episode=label, train=e.train, call=e.call, node=e.node)
    charges = itemise(tt, s.engine.log)
    billed = {(c.train, c.event.call) for c in charges if c.component == "missed_calls" and c.amount >= 500}
    dropped_by_action = set()
    for e in events:
        if e.kind != "CALL_MISSED" or not trips[e.train].train.passenger:
            continue
        audit.counts["pax_calls_missed"] += 1
        if e.detail.get("why") != "shift_end":
            dropped_by_action.add((e.train, e.call))
        if (e.train, e.call) not in billed:
            audit.bad("missed_call_unbilled", episode=label, train=e.train, call=e.call, why=e.detail.get("why"))
    broken = {
        (c.event.detail.get("feeder"), c.event.detail.get("connector"), c.event.node)
        for c in charges
        if c.event.kind == "CONNECTION" and c.amount > 0
    }
    for i, link in enumerate(s.engine.connections):
        fc, cc = s.engine._link_calls[i]
        if fc is None or (link.feeder, fc) not in dropped_by_action:
            continue
        connector_left = any(
            e.kind == "CONNECTION"
            and e.detail.get("feeder") == link.feeder
            and e.detail.get("connector") == link.connector
            and e.node == link.node
            and (cc is None or e.call == cc)
            for e in events
        )
        if not connector_left:  # connector never made that call: its own CALL_MISSED bills the boarders
            audit.counts["connections_dropped_with_connector_missed"] += 1
            continue
        audit.counts["connections_broken_by_action"] += 1
        if (link.feeder, link.connector, link.node) not in broken:
            audit.bad("broken_connection_unbilled", episode=label, feeder=link.feeder, connector=link.connector)


def physical_violation(s: Session, tid: str) -> str | None:
    """Check the train's planned path against electrification, road length and track direction."""
    mv = s.engine.trains[tid]
    path = s.engine._journey(mv) or []
    net = s.engine.net
    train = mv.train
    for i, eid in enumerate(path):
        el = net.element(eid)
        if isinstance(el, Track):
            if train.electric and not el.electrified:
                return f"non-electrified {eid}"
            prev = net.element(path[i - 1]) if i else None
            came = getattr(prev, "node", None) if prev is not None else None
            if came is not None and el.direction != "both" and el.travel_from(came) != el.direction:
                return f"wrong direction on {eid}"
        elif isinstance(el, Platform):
            if el.length_m < train.length_m:
                return f"{eid} shorter than {tid}"
            if train.electric and not el.electrified:
                return f"non-electrified {eid}"
    return None


def _constraint_breaker(s: Session, tid: str, eid: str) -> bool:
    """Would running via `eid` break a hard physical constraint for this train, whatever the path?"""
    el, train = s.engine.net.element(eid), s.engine.trains[tid].train
    if train.electric and not getattr(el, "electrified", True):
        return True
    return isinstance(el, Platform) and el.length_m < train.length_m


# ---------------------------------------------------------------- route / short_turn probes


def probe_route(tt: Timetable, base: Session, audit: Audit, label: str, playouts: list) -> None:
    elements = [t.id for t in tt.net.tracks] + [p.id for p in tt.net.platforms]
    scratch = copy.deepcopy(base)
    for tid in _live_ids(base):
        trip = base.engine.trains[tid].trip
        for eid in elements:
            breaker = _constraint_breaker(scratch, tid, eid)
            out, removed = act_checked(scratch, audit, "route", {"train": tid, "via": [eid]}, tid)
            audit.counts["route:attempts"] += 1
            if not out.startswith("GRANTED"):
                audit.counts[f"route:refused:{out.split(':')[0].removeprefix('REFUSED ')}"] += 1
                continue
            if breaker:
                audit.bad("route_granted_through_constraint", at=label, train=tid, via=eid)
            if why := physical_violation(scratch, tid):
                audit.bad("route_physical", at=label, train=tid, via=eid, why=why)
            if not trip.train.passenger:
                audit.counts["route:freight_granted"] += 1
                mv = scratch.engine.trains[tid]
                last = len(trip.calls) - 1
                if not trip.exit_node and last not in mv.plan_calls and last not in mv.made:
                    audit.bad("freight_destination_dropped", at=label, train=tid, via=eid)
            elif removed:
                audit.counts["route:pax_diversions_dropping_calls"] += 1
                if len(playouts) < ROUTE_PLAYOUTS:
                    playouts.append((label, tid, eid, scratch))
            scratch = copy.deepcopy(base)


def _wrong_way(tt: Timetable, s: Session, tid: str, track_id: str) -> bool:
    """Does the train's planned path run over `track_id` against its normal (timetable) direction?"""
    normal = tt.net.element(track_id)
    path = s.engine._journey(s.engine.trains[tid]) or []
    for i, eid in enumerate(path):
        if eid == track_id and i:
            came = getattr(s.engine.net.element(path[i - 1]), "node", None)
            if came is not None and normal.travel_from(came) != normal.direction:
                return True
    return False


def probe_single_line(tt: Timetable, base: Session, audit: Audit, label: str) -> None:
    """Open each reversible track to both directions, route every train over it, and check that a wrong-way
    diversion still conserves calls and respects physics, and that the track cannot be closed under it."""
    reversible = [t.id for t in tt.net.tracks if t.reversible]
    for track_id in reversible:
        opened = copy.deepcopy(base)
        if not opened.call("single_line", {"segment": track_id, "on": True}).startswith("OK"):
            continue
        audit.counts["single_line:opened"] += 1
        scratch = copy.deepcopy(opened)
        for tid in _live_ids(opened):
            out, _ = act_checked(scratch, audit, "route", {"train": tid, "via": [track_id]}, tid)
            if not out.startswith("GRANTED"):
                continue
            audit.counts["single_line:routes_granted"] += 1
            if why := physical_violation(scratch, tid):
                audit.bad("single_line_physical", at=label, train=tid, via=track_id, why=why)
            if _wrong_way(tt, scratch, tid, track_id):
                audit.counts["single_line:wrong_way_routes"] += 1
                off = scratch.call("single_line", {"segment": track_id, "on": False})
                if not off.startswith("REFUSED R-SINGLE-LINE"):
                    audit.bad("single_line_closed_under_train", at=label, train=tid, segment=track_id, out=off[:80])
            scratch = copy.deepcopy(opened)


def probe_short_turn(tt: Timetable, base: Session, audit: Audit, label: str) -> None:
    from railway.net.engine import TURNAROUND_MIN

    scratch = copy.deepcopy(base)
    for tid in _live_ids(base):
        mv = scratch.engine.trains[tid]
        has_return = any(
            c is not None
            and mv.trip.calls[c].min_dwell >= TURNAROUND_MIN
            and (p < len(mv.plan) - 1 or mv.trip.exit_node)
            for p, c in enumerate(mv.plan_calls)
        )
        for station in dict.fromkeys(_stations(scratch, tid)):
            out, _ = act_checked(scratch, audit, "short_turn", {"train": tid, "station": station}, tid)
            audit.counts["short_turn:attempts"] += 1
            if not has_return:
                audit.counts["short_turn:no_return_attempts"] += 1
                if not out.startswith("REFUSED R-NO-RETURN"):
                    audit.bad("short_turn_without_return", at=label, train=tid, station=station, out=out[:80])
            if out.startswith("OK"):
                scratch = copy.deepcopy(base)
                break


# ---------------------------------------------------------------- per seed


def audit_seed(tier: str, seed: int) -> dict:
    t0 = time.time()
    tt = X.build_timetable(tier, seed)
    audit = Audit()
    costs: dict[str, float] = {}
    by_base: dict[str, dict[str, float]] = {}
    actions: dict[str, int] = {}
    with X._cached_routing():
        for name, pax_first in DISPATCHERS.items():
            res, s = play(tt, pax_first=pax_first)
            costs[name] = res.cost.total
            check_episode(tt, s, audit, name)
        hold_cost, hold_at = freight_hold_sweep(tt, audit)
        costs["freight_hold_sweep"] = hold_cost if hold_at else costs["booked_order"]
        for name, make in POLICIES.items():
            by_base[name] = {}
            for base_name, pax_first in DISPATCHERS.items():
                res, s = play(tt, make(), pax_first=pax_first)
                by_base[name][base_name] = res.cost.total
                n_acts = sum(1 for e in s.log if e["tool"] in AVOID_TOOLS and e["status"] == "OK")
                actions[name] = max(actions.get(name, 0), n_acts)
                check_episode(tt, s, audit, f"{name}+{base_name}")
            costs[name] = min(by_base[name].values())
        start = Session(tt, budget=BIG_BUDGET)
        mid = regulate(Session(tt, budget=BIG_BUDGET), stop_at=tt.shift_min // 2)
        playouts: list = []
        for label, base in (("start", start), ("mid", mid)):
            probe_route(tt, base, audit, label, playouts)
            probe_short_turn(tt, base, audit, label)
            probe_single_line(tt, base, audit, label)
            for tid in _live_ids(base):
                scratch = copy.deepcopy(base)
                stations = _stations(scratch, tid)
                if len(stations) > 2:
                    act_checked(scratch, audit, "skip", {"train": tid, "station": stations[1]}, tid)
                act_checked(copy.deepcopy(base), audit, "cancel", {"train": tid}, tid)
        for label, tid, eid, s in playouts:
            regulate(s)
            s.close()
            check_episode(tt, s, audit, f"route:{label}:{tid}via{eid}")
            audit.counts["route:playouts"] += 1
    dispatch = min(costs[k] for k in DISPATCH_REFS)
    return {
        "tier": tier,
        "seed": seed,
        "dispatch": dispatch,
        "freight_hold_best": hold_at,
        "costs": costs,
        "avoidance_by_base": by_base,
        "avoidance_actions": actions,
        # True: strictly costlier than honest dispatch; "no-op": the policy found nothing to withdraw on this seed.
        "avoidance_loses": {k: (costs[k] > dispatch) if actions[k] else "no-op" for k in POLICIES},
        "counts": dict(sorted(audit.counts.items())),
        "violations": audit.violations,
        "secs": round(time.time() - t0, 1),
    }


def summarise(rows: list[dict]) -> dict:
    out: dict = {"seeds": len(rows), "by_tier": {}, "all": {}}

    def stats(group: list[dict]) -> dict:
        disp = [r["dispatch"] for r in group]
        res: dict = {"dispatch": {"mean_cost": round(statistics.mean(disp), 2)}}
        for n in group[0]["costs"]:
            vals = [r["costs"][n] for r in group]
            pairs = list(zip(vals, disp, strict=True))
            res[n] = {
                "mean_cost": round(statistics.mean(vals), 2),
                "mean_ratio_vs_dispatch": round(statistics.mean(v / d for v, d in pairs), 3),
                "min_ratio_vs_dispatch": round(min(v / d for v, d in pairs), 3),
            }
            if n in POLICIES:
                flags = [r["avoidance_loses"][n] for r in group]
                acted = [f for f in flags if f != "no-op"]
                res[n]["strictly_worse_on"] = f"{sum(f is True for f in acted)}/{len(acted)} seeds where it acted"
                res[n]["no_op_seeds"] = len(flags) - len(acted)
        return res

    for tier in sorted({r["tier"] for r in rows}):
        out["by_tier"][tier] = stats([r for r in rows if r["tier"] == tier])
    out["all"] = stats(rows)
    totals: Counter[str] = Counter()
    for r in rows:
        totals.update(r["counts"])
    out["counts"] = dict(sorted(totals.items()))
    out["violations"] = sum(v for k, v in totals.items() if k.startswith("violation:"))
    out["every_avoidance_policy_strictly_loses"] = all(
        f is not False for r in rows for f in r["avoidance_loses"].values()
    )
    out["avoidance_wins"] = [
        {"seed": f"{r['tier']}-{r['seed']}", "policy": k, "cost": r["costs"][k], "dispatch": r["dispatch"]}
        for r in rows
        for k, f in r["avoidance_loses"].items()
        if f is False
    ]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiers", default="T1,T2,T3")
    ap.add_argument("--seeds", default=None, help="comma list; default: the tier's accepted dev seeds")
    ap.add_argument("--out", default=None)
    ap.add_argument("--merge", nargs="+", default=None, help="combine the rows of earlier --out files instead")
    a = ap.parse_args(argv)
    rows = []
    for path in a.merge or []:
        rows += json.loads(Path(path).read_text(encoding="utf-8"))["rows"]
    rows.sort(key=lambda r: (r["tier"], r["seed"]))
    for tier in [] if a.merge else a.tiers.split(","):
        seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else dev_seeds(tier)
        for seed in seeds:
            r = audit_seed(tier, seed)
            rows.append(r)
            c = r["costs"]
            bad = sum(v for k, v in r["counts"].items() if k.startswith("violation:"))
            print(
                f"{tier}-{seed}: dispatch {r['dispatch']:9.2f} | "
                + " | ".join(f"{k} {v:9.2f}" for k, v in c.items())
                + f" | route granted {r['counts'].get('route:granted', 0)}/{r['counts'].get('route:attempts', 0)}"
                f" | violations {bad} | {r['secs']}s",
                flush=True,
            )
    result = {"summary": summarise(rows), "rows": rows}
    print(json.dumps(result["summary"], indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
