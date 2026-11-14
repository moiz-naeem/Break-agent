"""Proven-optimal dispatching for scenarios without events (factors F1-F4, i.e. T1), by CP-SAT.

Needs `ortools` (not a project dependency: `pip install ortools`); the tests skip without it.

Model:
  - a train runs hop by hop; from the grant of a hop it holds the section until it arrives and the
    chosen track at the next station until it leaves that station;
  - one train per section and per station track at a time;
  - a station track left at tick t can be granted again at t only if the leaving train had its
    onward authority before t (it then leaves by itself); otherwise from t+1;
  - scheduled stops: dwell >= min_dwell and no departure before `dep`;
  - a train may be left out entirely (never authorized) for its unfinished cost. A train that runs
    part of the way is never cheaper than that, so it is not modelled.

`solve(bound_only=True)` is a lower bound on any legal play; `solve()` gives a plan that `dispatch`
plays through the real tool session. When the three costs are equal the plan is proven optimal.
`cancel_authority`, `hold` (other than as an alarm clock) and `reverse` are not modelled.

Run:  python -m railway.ref.optimal --config T1 --set dev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ortools.sat.python import cp_model

from railway.sim.constants import PAX_GRACE, TERMINUS_MULT, TOOL_BUDGET, TRAIN_TYPES, W_EXPRESS, W_FREIGHT, W_LOCAL
from railway.sim.cost import unfinished_cost
from railway.sim.timing import travel_ticks
from railway.tools.session import Session

SCALE = 1000


def solve(sc, time_limit=300.0, bound_only=False):
    """bound_only: let an early grant fall on any tick of the dwell (a lower bound); otherwise only on the
    arrival tick, where the simulator is sure to give the dispatcher a decision point."""
    if not set(sc.factors) <= {"F1", "F2", "F3", "F4"} or sc.events:
        raise ValueError("only scenarios with factors F1-F4 and no events are modelled")
    places = ["YW", *[s.id for s in sc.network.stations], "YE"]
    station = {s.id: s for s in sc.network.stations}
    section = {}
    for s in sc.network.sections:
        section[(s.west, s.east)] = section[(s.east, s.west)] = s
    horizon = sc.shift_ticks
    m = cp_model.CpModel()
    by_resource: dict = {}
    objective, trains = [], {}
    for tr in sc.trains:
        spec = TRAIN_TYPES[tr.type]
        route = places if tr.direction == "east" else places[::-1]
        end = tr.stops[-1].station
        if end not in ("YW", "YE"):
            raise ValueError(f"{tr.id} does not finish in a yard")
        route = route[: route.index(end) + 1]
        stops = {s.station: s for s in tr.stops}
        present = m.NewBoolVar(f"{tr.id}_present")
        dep, arr, choice, grant = {}, {}, {}, {}
        for j, place in enumerate(route):
            if j > 0:
                sec = section[(route[j - 1], place)]
                run = travel_ticks(sec.length_m, sec.line_speed_kmh, spec["max_speed_kmh"])
                arr[j] = m.NewIntVar(0, horizon, f"{tr.id}_arr_{place}")
                m.Add(arr[j] == dep[j - 1] + run)
                held = m.NewIntVar(1, horizon, "")  # the section is reserved from the grant, not from the departure
                by_resource.setdefault(("sec", sec.id), []).append(
                    m.NewOptionalIntervalVar(grant[j], held, arr[j], present, f"{tr.id}_{sec.id}")
                )
            if j == len(route) - 1:
                break
            dep[j] = m.NewIntVar(0, horizon, f"{tr.id}_dep_{place}")
            grant[j + 1] = m.NewIntVar(0, horizon, f"{tr.id}_grant_{place}")  # when the hop out of `place` is granted
            if j == 0:
                m.Add(dep[0] >= tr.entry_tick)
                m.Add(grant[1] == dep[0])
                continue
            stop = stops.get(place)
            earliest = m.NewIntVar(0, horizon, "")
            m.AddMaxEquality(earliest, [arr[j] + (stop.min_dwell if stop else 0), (stop.dep or 0) if stop else 0])
            m.Add(dep[j] >= earliest)
            tracks = ["main"] + [
                loop.id
                for loop in station[place].loops
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
            m.Add(sum(picks) == present)
        for stop in tr.stops:
            j = route.index(stop.station)
            terminus = stop is tr.stops[-1]
            if tr.type == "freight":
                coef, grace = (W_FREIGHT if terminus else 0), 0
            elif tr.type in ("express", "local"):
                weight = W_EXPRESS if tr.type == "express" else W_LOCAL
                coef, grace = weight * tr.pax / 100 * (TERMINUS_MULT if terminus else 1), PAX_GRACE
            else:
                coef, grace = 0, 0
            if coef:
                over = m.NewIntVar(0, horizon, "")
                m.Add(over >= arr[j] - stop.arr - grace)
                objective.append(round(coef * SCALE) * over)
        objective.append(round(unfinished_cost(tr, sc.shift_ticks) * SCALE) * (1 - present))
        trains[tr.id] = (route, present, grant, arr, choice)
    for intervals in by_resource.values():
        m.AddNoOverlap(intervals)
    m.Minimize(sum(objective))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    status = solver.Solve(m)
    plan = {}
    for tid, (route, present, grant, _arr, choice) in trains.items():
        if not solver.Value(present):
            continue
        hops = []
        for j in range(1, len(route)):
            track = "main"
            if j < len(route) - 1:
                track = next(k for (jj, k), x in choice.items() if jj == j and solver.Value(x))
            hops.append((solver.Value(grant[j]), route[j], track))
        plan[tid] = hops
    return {
        "status": solver.StatusName(status),
        "cost": solver.ObjectiveValue() / SCALE,
        "bound": solver.BestObjectiveBound() / SCALE,
        "plan": plan,
    }


def dispatch(sc, plan, budget=TOOL_BUDGET):
    """Dispatch the plan through the real tool session. `hold` is used as an alarm clock for a waiting train."""
    session = Session(sc, budget=budget)
    todo = {tid: list(hops) for tid, hops in plan.items()}
    alarms = set()
    while not session.closed:
        tick = session.sim.tick
        for _ in range(2):  # second pass: a grant may free what another grant needs
            for tid in sorted(todo, key=lambda x: (todo[x][0][0] if todo[x] else 10**9, x)):
                if todo[tid] and todo[tid][0][0] <= tick:
                    _, place, track = todo[tid][0]
                    out = session.call("authorize", {"train": tid, "station": place, "track": track})
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
    undelivered = {tid: hops[0] for tid, hops in todo.items() if hops}
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
    print("| seed | c_greedy | c_ref | lower bound | plan | dispatched | tool calls | proven |")
    print("|---|---|---|---|---|---|---|---|")
    for entry in json.loads(path.read_text(encoding="utf-8"))["accepted"]:
        scenario = generate(PRESETS[args.config], entry["seed"])
        bound, plan = solve(scenario, args.time_limit, bound_only=True), solve(scenario, args.time_limit)
        cost, calls, undelivered = dispatch(scenario, plan["plan"])
        proven = bound["status"] == plan["status"] == "OPTIMAL" and bound["cost"] == cost and not undelivered
        print(
            f"| {entry['seed']} | {entry['c_greedy']:.1f} | {entry['c_ref']:.1f} | {bound['cost']:.1f} | "
            f"{plan['cost']:.1f} | {cost:.1f} | {calls} | {'yes' if proven else 'NO'} |",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
