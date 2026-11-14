"""Proven-optimal dispatching for scenarios without events (factors F1-F4, F11 off), by CP-SAT, under the v0.3 cost.

Needs `ortools` (not a project dependency: `pip install ortools`); the tests skip without it.

Model:
  - every train runs its timetabled route hop by hop; from the grant of a hop it holds the section until it
    arrives and the chosen track at the next station until it leaves that station;
  - one train per section and per station track at a time; a terminal's tracks of one kind (platform or goods)
    are a pool with as many places as tracks: the origin's from entry until departure, the destination's from
    the grant until the turnaround ends;
  - a station track left at tick t can be granted again at t only if the leaving train had its onward
    authority before t (it then leaves by itself); otherwise from t+1;
  - scheduled stops: dwell >= min_dwell and no departure before `dep`;
  - times may run past the shift end: a stop reached after it is a missed call, a destination reached after it
    an unfinished train, exactly as the simulator charges them. So partial runs are modelled, not just
    "run or not".

`solve(bound_only=True)` relaxes three things into a lower bound on any legal play: an early grant may fall on
any tick of the dwell, an entry may be later than the timetable, and a terminal track frees on the departure
tick. `solve()` gives a plan that `dispatch` plays through the real tool session. When the bound equals the
dispatched cost, the plan is proven optimal among plans that keep every train on its timetabled route, which on
a network without loops is every plan. `cancel_authority`, `hold` (other than as an alarm clock) and leaving
the timetabled route are not modelled.

Run:  python -m railway.ref.optimal --config T1 --set dev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ortools.sat.python import cp_model

from railway.policies.routes import call_args, sim_route, via
from railway.sim.constants import (
    FREIGHT_LATE_STEP,
    LATE_STEPS,
    MISSED_CALL_FLAT,
    MISSED_CALL_HASSLE,
    PAX_GRACE,
    PLATFORM_CHANGE_MINUTES,
    TOOL_BUDGET,
    TRAIN_TYPES,
    TURNAROUND,
    UNFINISHED_COST,
    VALUE,
    W_FREIGHT,
)
from railway.sim.cost import lateness_curve, unfinished_cost
from railway.sim.engine import Simulator
from railway.sim.model import Authorize
from railway.sim.timing import travel_ticks
from railway.tools.session import Session

SCALE = 1000
MODELLED = {"F1", "F2", "F3", "F4"}


def _acyclic(sc) -> bool:
    nodes = len(sc.network.stations) + len(sc.network.terminal_ids)
    return len(sc.network.sections) == nodes - 1


def solve(sc, time_limit=300.0, bound_only=False):
    if not set(sc.factors) <= MODELLED or sc.events:
        raise ValueError("only scenarios with factors F1-F4 and no events are modelled")
    sim = Simulator.load(sc)
    graph = sim._graph
    section = {frozenset((s.west, s.east)): s for s in sc.network.sections}
    horizon = sc.shift_ticks * 3
    shift = sc.shift_ticks
    m = cp_model.CpModel()
    by_resource: dict = {}
    pools: dict = {}  # (terminal, track kind) -> intervals
    objective, trains = [], {}

    def late_terms(arr, sched, weight_milli: int, prefix: str):
        """weight_milli x f(arr - sched), f the passenger lateness curve, as linear terms."""
        over = m.NewIntVar(0, horizon, "")
        m.Add(over >= arr - sched - PAX_GRACE)
        terms = [weight_milli * over]
        for at, add in LATE_STEPS:
            step = m.NewBoolVar(f"{prefix}_ge{at}")
            m.Add(arr - sched <= at - 1).OnlyEnforceIf(step.Not())
            terms.append(weight_milli * add * step)
        return terms

    for tr in sc.trains:
        spec = TRAIN_TYPES[tr.type]
        route = sim_route(sim, tr.id)
        stops = {s.station: (i, s) for i, s in enumerate(tr.stops)}
        dep, arr, choice, grant = {}, {}, {}, {}
        for j, place in enumerate(route):
            if j > 0:
                sec = section[frozenset((route[j - 1], place))]
                run = travel_ticks(sec.length_m, sec.line_speed_kmh, spec["max_speed_kmh"])
                arr[j] = m.NewIntVar(0, horizon, f"{tr.id}_arr_{place}")
                m.Add(arr[j] == dep[j - 1] + run)
                held = m.NewIntVar(1, horizon, "")  # the section is reserved from the grant, not from the departure
                by_resource.setdefault(("sec", sec.id), []).append(
                    m.NewIntervalVar(grant[j], held, arr[j], f"{tr.id}_{sec.id}")
                )
            if j == len(route) - 1:
                break
            dep[j] = m.NewIntVar(0, horizon, f"{tr.id}_dep_{place}")
            grant[j + 1] = m.NewIntVar(0, horizon, f"{tr.id}_grant_{place}")  # when the hop out of `place` is granted
            if j == 0:
                entry = m.NewIntVar(tr.entry_tick, horizon, f"{tr.id}_entry")
                if not bound_only:
                    m.Add(entry == tr.entry_tick)
                m.Add(dep[0] >= entry)
                m.Add(grant[1] == dep[0])
                if place in graph.terminals and not graph.unlimited(place):
                    kind = "platform" if tr.type in {"express", "local"} else "goods"
                    size = m.NewIntVar(0, horizon + 1, "")
                    release = m.NewIntVar(0, horizon + 1, "")
                    m.Add(release == dep[0] + (0 if bound_only else 1))
                    pools.setdefault((place, kind), []).append(m.NewIntervalVar(entry, size, release, ""))
                continue
            index, stop = stops.get(place, (None, None))
            earliest = m.NewIntVar(0, horizon, "")
            m.AddMaxEquality(earliest, [arr[j] + (stop.min_dwell if stop else 0), (stop.dep or 0) if stop else 0])
            m.Add(dep[j] >= earliest)
            tracks = ["main"] + [
                loop.id
                for loop in sim._stations[place].loops
                if loop.length_m >= spec["length_m"] and not (loop.kind == "goods" and tr.type in ("express", "local"))
            ]
            # early: the onward hop is granted before the departure tick, so the train leaves by itself as soon
            # as it may, and its track is free for a grant at that same tick. Otherwise the grant and the
            # departure share a tick, and the track can only be granted to another train one tick later.
            # thru: granted together with the hop into this station (one long authority over its main).
            early, thru = m.NewBoolVar(f"{tr.id}_{place}_early"), m.NewBoolVar(f"{tr.id}_{place}_thru")
            m.AddImplication(thru, early)
            m.Add(grant[j + 1] == dep[j]).OnlyEnforceIf(early.Not())
            m.Add(grant[j + 1] <= dep[j] - 1).OnlyEnforceIf(early)
            m.Add(dep[j] == earliest).OnlyEnforceIf(early)
            m.Add(grant[j + 1] == grant[j]).OnlyEnforceIf(thru)
            m.Add(grant[j + 1] >= arr[j]).OnlyEnforceIf([early, thru.Not()])
            if not bound_only:
                m.Add(grant[j + 1] == arr[j]).OnlyEnforceIf([early, thru.Not()])
            picks = []
            for k in tracks:
                x = m.NewBoolVar(f"{tr.id}_{place}_{k}")
                size = m.NewIntVar(1, horizon + 1, "")
                release = m.NewIntVar(0, horizon + 1, "")
                m.Add(release == dep[j] + 1 - early)
                by_resource.setdefault(("trk", place, k), []).append(
                    m.NewOptionalIntervalVar(grant[j], size, release, x, f"{tr.id}_{place}_{k}_iv")
                )
                picks.append(x)
                choice[(j, k)] = x
            m.AddImplication(thru, choice[(j, "main")])
            m.AddExactlyOne(picks)
            if stop is not None and stop.platform is not None and tr.type in VALUE:
                alight, board = tr.flow(index)
                weight = round(VALUE[tr.type] * (alight + board) / 100 * PLATFORM_CHANGE_MINUTES * SCALE)
                if stop.platform in tracks:
                    objective.append(weight * (1 - choice[(j, stop.platform)]))
                else:
                    objective.append(weight)
        last = len(route) - 1
        end = route[last]
        if end in graph.terminals and not graph.unlimited(end):
            kind = "platform" if tr.type in {"express", "local"} else "goods"
            size = m.NewIntVar(0, horizon + TURNAROUND + 1, "")
            release = m.NewIntVar(0, horizon + TURNAROUND + 1, "")
            m.Add(release == arr[last] + TURNAROUND)
            pools.setdefault((end, kind), []).append(m.NewIntervalVar(grant[last], size, release, ""))
        # Cost: each stop on time-ish (lateness) or after the shift end (missed); unfinished if the end is late.
        finished = m.NewBoolVar(f"{tr.id}_finished")
        m.Add(arr[last] <= shift).OnlyEnforceIf(finished)
        m.Add(arr[last] >= shift + 1).OnlyEnforceIf(finished.Not())
        objective.append(round(unfinished_cost(tr, shift) * SCALE) * (1 - finished))
        for i, stop in enumerate(tr.stops):
            j = route.index(stop.station)
            served = m.NewBoolVar(f"{tr.id}_s{i}")
            m.Add(arr[j] <= shift).OnlyEnforceIf(served)
            m.Add(arr[j] >= shift + 1).OnlyEnforceIf(served.Not())
            if tr.type in VALUE:
                alight, board = tr.flow(i)
                missed = MISSED_CALL_FLAT + VALUE[tr.type] * (alight + board) / 100 * (
                    lateness_curve(stop.headway or 60) + MISSED_CALL_HASSLE
                )
                objective.append(round(missed * SCALE) * (1 - served))
                weight = round(VALUE[tr.type] * alight / 100 * SCALE)
                if weight:
                    # only charged when served: a missed stop is not also late
                    late = m.NewIntVar(0, horizon, "")
                    m.Add(late == arr[j]).OnlyEnforceIf(served)
                    m.Add(late == stop.arr).OnlyEnforceIf(served.Not())
                    objective.extend(late_terms(late, stop.arr, weight, f"{tr.id}_{i}"))
            elif tr.type == "freight" and i == len(tr.stops) - 1:
                late = m.NewIntVar(0, horizon, "")
                m.Add(late >= arr[j] - stop.arr).OnlyEnforceIf(finished)  # unfinished: its lateness is in unfinished_cost
                objective.append(round(W_FREIGHT * SCALE) * late)
                at, step_cost = FREIGHT_LATE_STEP
                step = m.NewBoolVar("")
                m.Add(arr[j] - stop.arr <= at).OnlyEnforceIf([step.Not(), finished])
                objective.append(round(step_cost * SCALE) * step)
        trains[tr.id] = (route, grant, arr, choice)
    for intervals in by_resource.values():
        m.AddNoOverlap(intervals)
    capacity = {}
    for t in sc.network.terminals:
        for track in t.tracks:
            capacity[(t.id, track.kind)] = capacity.get((t.id, track.kind), 0) + 1
    for key, intervals in pools.items():
        m.AddCumulative(intervals, [1] * len(intervals), capacity.get(key, 0))
    m.Minimize(sum(objective))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    status = solver.Solve(m)
    plan = {}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for tid, (route, grant, _arr, choice) in trains.items():
            hops = []
            for j in range(1, len(route)):
                track = "main"
                if j < len(route) - 1:
                    track = next(k for (jj, k), x in choice.items() if jj == j and solver.Value(x))
                elif route[j] in graph.terminals and not graph.unlimited(route[j]):
                    track = None  # any free fitting terminal track, picked at the grant
                hops.append((solver.Value(grant[j]), route[j], track))
            plan[tid] = hops
    return {
        "status": solver.StatusName(status),
        "cost": solver.ObjectiveValue() / SCALE if plan else None,
        "bound": solver.BestObjectiveBound() / SCALE,
        "plan": plan,
        "acyclic": _acyclic(sc),
    }


def dispatch(sc, plan, budget=TOOL_BUDGET, with_log=False):
    """Dispatch the plan through the real tool session. `hold` is used as an alarm clock for a waiting train."""
    session = Session(sc, budget=budget)
    graph = not session.sim._graph.chain
    todo = {tid: list(hops) for tid, hops in plan.items()}
    alarms = set()

    def grant(tid, place, track):
        sim = session.sim
        t = sim._trains[tid]
        start = t["auth"]["dest_station"] if t["auth"] else t["station"]
        route = sim_route(sim, tid)
        tracks = [track] if track is not None else sorted(sim._graph.tracks[place])
        for k in tracks:
            action = Authorize(train=tid, station=place, track=k, via=via(route, start, place, graph))
            if track is None and not sim.check(action).ok:
                continue
            return session.call("authorize", call_args(action))
        return "REFUSED no free terminal track"

    while not session.closed:
        tick = session.sim.tick
        for _ in range(2):  # second pass: a grant may free what another grant needs
            for tid in sorted(todo, key=lambda x: (todo[x][0][0] if todo[x] else 10**9, x)):
                if todo[tid] and todo[tid][0][0] <= tick:
                    _, place, track = todo[tid][0]
                    out = grant(tid, place, track)
                    if session.closed:
                        break
                    if out.startswith("GRANTED"):
                        todo[tid].pop(0)
        state = {t.id: t for t in session.sim.state().trains}  # after the grants: a hold only arms a waiting train
        for tid, hops in todo.items():
            train = state.get(tid)
            if not hops or session.closed or train is None or train.phase != "at_station" or train.authority:
                continue
            wake = max(hops[0][0], tick + 1)
            if (tid, wake) not in alarms:
                alarms.add((tid, wake))
                session.call("hold", {"train": tid, "until_tick": wake})
        if session.closed:
            break
        if not any(todo.values()):
            session.call("end_shift", {})
            break
        session.call("advance_to_next_event", {})
    final = session.close()
    undelivered = {tid: hops[0] for tid, hops in todo.items() if hops and hops[0][0] < sc.shift_ticks}
    if with_log:
        return final.cost.total, len(session.log), undelivered, session.log
    return final.cost.total, len(session.log), undelivered


def main(argv=None) -> int:
    from railway.gen.generator import generate
    from railway.gen.presets import PRESETS

    parser = argparse.ArgumentParser(prog="railway.ref.optimal")
    parser.add_argument("--config", default="T1")
    parser.add_argument("--set", default="dev")
    parser.add_argument("--time-limit", type=float, default=300.0)
    args = parser.parse_args(argv)
    path = Path(__file__).resolve().parents[2] / "seeds" / f"{args.config}-{args.set}.json"
    print("| seed | c_greedy | c_best | lower bound | plan | dispatched | tool calls | proven |")
    print("|---|---|---|---|---|---|---|---|")
    for entry in json.loads(path.read_text(encoding="utf-8"))["accepted"]:
        scenario = generate(PRESETS[args.config], entry["seed"])
        bound, plan = solve(scenario, args.time_limit, bound_only=True), solve(scenario, args.time_limit)
        cost, calls, undelivered = dispatch(scenario, plan["plan"])
        proven = (
            bound["status"] == plan["status"] == "OPTIMAL"
            and abs(bound["cost"] - cost) < 1e-6
            and not undelivered
            and plan["acyclic"]
        )
        print(
            f"| {entry['seed']} | {entry['c_greedy']:.1f} | {entry.get('c_best', float('nan')):.1f} | "
            f"{bound['cost']:.1f} | {plan['cost']:.1f} | {cost:.1f} | {calls} | {'yes' if proven else 'NO'} |",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
