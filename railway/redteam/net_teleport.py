"""Red-team probe F1: is a booked call served at another station still billed as served? (docs/redteam-net.md)

`platform(train, station, n)` re-platforms a train's next booked call. The engine checks only that the
target is a platform the train can reach. It does not check that the platform is at the station the call is
booked at. The train then berths at another station, `ARRIVE` carries the original call index, and the cost
bills an on-time (or early) arrival plus a two-minute platform-change charge instead of the 500+ missed call.

The probe is a paired lookahead search: one search, two option sets.

* `control`: re-platform the next call only to another platform at its booked station (what the tool is for);
* `exploit`: also allow platforms at any other station (what the tool accepts today).

At every decision point the search tries each option on a copy of the session, plays the rest of the shift with
booked-order regulation (`railway.net.exploits`), and keeps the best option only if it lowers the final cost.
Because both arms run the same search and differ only in the option set, the cost difference is the value of
the hole and nothing else.

    python -m railway.redteam.net_teleport --tier T1 --seeds 1,3,4,5,7
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import pickle
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor

from railway.net import exploits as X
from railway.net.model import Platform, Timetable
from railway.net.session import Session

MAX_STEPS = 3000


def _clone(s: Session) -> Session:
    memo = {
        id(s.timetable): s.timetable,
        id(s.timetable.net): s.timetable.net,
        id(s.engine._normal_net): s.engine._normal_net,
    }
    return copy.deepcopy(s, memo)


def _regulate(s: Session) -> None:
    X._relieve_warned_crews(s)
    X._rescue_stranded(s)
    X._order_open_conflicts(s)


def rollout(s: Session, *, clone: bool = True) -> float:
    """Final cost if the rest of the shift is played with booked-order regulation."""
    if clone:
        s = _clone(s)
    for _ in range(MAX_STEPS):
        if s.closed:
            break
        _regulate(s)
        s.call("advance_to_next_event", {})
    return s.close().cost.total


def _chain_len(s: Session, mv, platform: str) -> int | None:
    eng = s.engine
    stops = [(v, False) for v in mv.via] + [(platform, False)]
    legs = eng._chain(mv, mv.position, mv.facing(eng.net), eng._side_in(mv), stops, via_count=len(mv.via))
    return None if legs is None else sum(len(leg) for leg in legs)


def options(s: Session, cross: bool, k: int, horizon: int | None) -> Iterator[tuple[str, Platform]]:
    """Up to `k` re-platform targets per passenger train, nearest first; `cross` admits other stations."""
    eng, net = s.engine, s.timetable.net
    for tid, mv in sorted(eng.trains.items()):
        if not mv.train.passenger or mv.phase not in ("berthed", "offstage") or not mv.plan:
            continue
        if mv.plan_calls[0] is None:
            continue
        if horizon is not None and (mv.plan[0].arrive or 10**9) - eng.tick > horizon:
            continue
        booked_node = net.element(mv.plan[0].platform).node
        found = []
        for p in net.platforms:
            if p.id in (mv.plan[0].platform, mv.position) or p.kind == "siding" or p.length_m < mv.train.length_m:
                continue
            if mv.train.electric and not p.electrified:
                continue
            if p.node != booked_node and not cross:
                continue
            n = _chain_len(s, mv, p.id)
            if n is not None:
                found.append((n, p.id, p))
        for _, _, p in sorted(found)[:k]:
            yield tid, p


def _eval_chunk(
    payload: tuple[bytes, float, list[tuple[int, str, str, str]]],
) -> tuple[float, int, str, str, str] | None:
    state_bytes, base, chunk = payload
    s: Session = pickle.loads(state_bytes)
    best: tuple[float, int, str, str, str] | None = None
    for idx, tid, node, pid in chunk:
        trial = _clone(s)
        if not trial.call("platform", {"train": tid, "station": node, "n": pid}).startswith("OK"):
            continue
        v = rollout(trial, clone=False)
        if v < base - 1e-6 and (best is None or (v, idx) < (best[0], best[1])):
            best = (v, idx, tid, node, pid)
    return best


def run_arm(
    tt: Timetable,
    cross: bool,
    *,
    k: int = 3,
    horizon: int | None = None,
    workers: int = 1,
    include_entries: bool = True,
) -> dict:
    t0 = time.time()
    s = Session(tt, budget=100_000)
    moves: list[dict] = []
    with contextlib.ExitStack() as stack:
        pool = stack.enter_context(ProcessPoolExecutor(max_workers=workers)) if workers > 1 else None
        for _ in range(MAX_STEPS):
            if s.closed:
                break
            _regulate(s)
            if s.closed:
                break
            cands = list(options(s, cross, k, horizon))
            if not cands:
                s.call("advance_to_next_event", {})
                continue
            base = rollout(s)
            best: tuple[float, int, str, Platform] | None = None
            if pool is not None and len(cands) > 1:
                indexed = [(i, tid, p.node, p.id) for i, (tid, p) in enumerate(cands)]
                n_chunks = min(workers, len(indexed))
                state_bytes = pickle.dumps(s, protocol=5)
                chunks = [(state_bytes, base, indexed[i::n_chunks]) for i in range(n_chunks)]
                for res in pool.map(_eval_chunk, chunks):
                    if res is not None and (best is None or (res[0], res[1]) < (best[0], best[1])):
                        v, idx, tid, _, pid = res
                        best = (v, idx, tid, tt.net.element(pid))  # type: ignore[arg-type]
            else:
                for idx, (tid, p) in enumerate(cands):
                    trial = _clone(s)
                    if not trial.call("platform", {"train": tid, "station": p.node, "n": p.id}).startswith("OK"):
                        continue
                    v = rollout(trial, clone=False)
                    if v < base - 1e-6 and (best is None or (v, idx) < (best[0], best[1])):
                        best = (v, idx, tid, p)
            if best is None:
                s.call("advance_to_next_event", {})
                continue
            _, _, tid, p = best
            booked_node = tt.net.element(s.engine.trains[tid].plan[0].platform).node
            s.call("platform", {"train": tid, "station": p.node, "n": p.id})
            moves.append(
                {
                    "tick": s.engine.tick,
                    "train": tid,
                    "booked_node": booked_node,
                    "to": p.id,
                    "cross_station": p.node != booked_node,
                    "gain": round(base - best[0], 2),
                }
            )
    final = s.close()
    trip_by_id = {t.train.id: t for t in tt.trips}
    served_elsewhere = [
        {
            "tick": e.tick,
            "train": e.train,
            "call": e.call,
            "at": e.element,
            "booked_node": tt.net.element(trip_by_id[e.train].calls[e.call].platform).node,
        }
        for e in s.engine.log.events
        if e.kind == "ARRIVE"
        and e.call is not None
        and e.detail.get("booked_platform")
        and tt.net.element(e.detail["booked_platform"]).node != e.node
    ]
    out = {
        "tier": tt.tier,
        "seed": tt.seed,
        "arm": "exploit" if cross else "control",
        "k": k,
        "horizon": horizon,
        "cost": final.cost.model_dump(),
        "counts": final.counts,
        "moves": moves,
        "served_elsewhere": served_elsewhere,
        "secs": round(time.time() - t0),
    }
    if include_entries:
        out["entries"] = s.log
    return out


def pair(
    tier: str,
    seed: int,
    *,
    k: int = 3,
    horizon: int | None = None,
    workers: int = 1,
    include_entries: bool = True,
) -> dict:
    tt = X.build_timetable(tier, seed)
    booked = X.booked_order(tt).cost.total
    control = run_arm(tt, False, k=k, horizon=horizon, workers=workers, include_entries=include_entries)
    exploit = run_arm(tt, True, k=k, horizon=horizon, workers=workers, include_entries=include_entries)
    return {
        "tier": tier,
        "seed": seed,
        "booked": booked,
        "control": control,
        "exploit": exploit,
        "gain": round(control["cost"]["total"] - exploit["cost"]["total"], 6),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tier", default="T1")
    ap.add_argument("--seeds", default="1,3,4,5,7")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = []
    for seed in (int(x) for x in a.seeds.split(",")):
        r = pair(a.tier, seed, k=a.k, horizon=a.horizon, workers=a.workers)
        rows.append(r)
        print(
            f"{a.tier}-{seed}: booked {r['booked']:10.2f} | control {r['control']['cost']['total']:10.2f} | "
            f"exploit {r['exploit']['cost']['total']:10.2f} | gain {r['gain']:9.2f} | "
            f"calls served elsewhere {len(r['exploit']['served_elsewhere'])}",
            flush=True,
        )
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
