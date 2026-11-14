"""Offline search with full knowledge of the scenario (T06 §2.7).

It plays the **true** simulator (unrevealed events included) through `clone`, `apply` and `advance`,
so it measures how far any dispatcher is from the best possible plan for that scenario.

Two entry points:

- `exhaustive_search(scenario)`: depth-first branch and bound over every tick of the shift. Exact
  (minimum cost) on small scenarios such as the `tests/fixtures/` ones, within the canonical action
  timing described below.
- `offline_search(scenario, beam=16, per_train=4)`: the T06 beam search for larger scenarios. It is
  seeded with an optional known plan (`seed_log`, e.g. the reference's), so its result is never worse
  than that plan.

`offline_search` returns `(cost, log)` as T06 specifies; `exhaustive_search` returns a `SearchResult`
whose `plan` is `[(tick, [actions])]` and whose `log` is the same in JSON form. `replay_plan` replays
a plan on a fresh simulator (apply the tick's actions, then advance one tick, until the shift ends).

**Canonical timing (why "exhaustive" stays tractable).** Granting an authority before the train can
use it only reserves track earlier, which can block other trains but never helps this one, so a
stationary train is only authorized on ticks where it could depart; a moving train is only extended
on the last tick before it reaches its authority end (the latest moment that still avoids the stop).
Both choices are revisited every tick, so waiting is always explored.

**Soundness filter.** By default a branch is dropped as soon as two trains occupy one resource or a
train occupies a resource it does not hold. Several simulator defects allow exactly that (see
`tests/hardening/test_bugs.py`); `exploit_bugs=True` turns the filter off and also allows
`cancel_authority`, to measure how much a bug-exploiting agent could gain.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass

from railway.sim.constants import TRAIN_TYPES
from railway.sim.cost import freight_cost, pax_cost, unfinished_cost
from railway.sim.engine import Simulator
from railway.sim.model import Action, ActionAdapter, Authorize, CancelAuthority, Reverse, Scenario
from railway.sim.timing import travel_ticks

COMPONENTS = ("pax_lateness", "freight_lateness", "crew_timeout", "reversal", "unfinished")
YARDS = ("YW", "YE")

ActionFilter = Callable[[Simulator, Action], bool]


class SearchLimit(RuntimeError):
    """Raised when `max_nodes` is exceeded (the result would not be proven optimal)."""


# ---------------------------------------------------------------------------------------------
# Simulator helpers
# ---------------------------------------------------------------------------------------------

_SHARED = (
    "_scenario",
    "_train_defs",
    "_stations",
    "_section_by_id",
    "_section_by_pair",
    "_station_ids",
    "_station_idx",
    "_events",
    "_rotation_predecessor",
)


def fast_clone(sim: Simulator) -> Simulator:
    """`sim.clone()` without copying the read-only scenario objects (same semantics, much faster)."""
    memo = {}
    for name in _SHARED:
        value = getattr(sim, name)
        memo[id(value)] = value
    return copy.deepcopy(sim, memo)


def accumulated(sim: Simulator) -> float:
    cost = sim._cost
    return sum(getattr(cost, k) for k in COMPONENTS)


def final_cost(sim: Simulator) -> float:
    return round(sim.cost().total, 6)


def sound(sim: Simulator) -> bool:
    """No resource has two occupants; every occupant holds what it occupies."""
    seen = {}
    for tid in sorted(sim._trains):
        t = sim._trains[tid]
        if t["phase"] == "at_station" and t["station"] in sim._stations:
            res = f"{t['station']}/{t['track']}"
        elif t["phase"] in {"in_section", "reversing"}:
            res = t["section"]
        else:
            continue
        if res in seen or sim._holders.get(res) != tid:
            return False
        seen[res] = tid
    return True


def state_key(sim: Simulator) -> str:
    """Everything that determines the future cost (not the edge-trigger bookkeeping)."""
    trains = []
    for tid in sorted(sim._trains):
        t = dict(sim._trains[tid])
        t.pop("ready_reported_tick", None)
        trains.append(repr(sorted(t.items(), key=lambda kv: kv[0])))
    return repr(
        (
            sim.tick,
            sim.done,
            trains,
            sorted(sim._holders.items()),
            sorted(sim._revealed_events),
            repr(sim._possessions),
            repr(sim._restrictions),
            list(sim._gridlock),
        )
    )


def _front(sim: Simulator, tid: str) -> str | None:
    t = sim._trains[tid]
    if t["auth"]:
        return t["auth"]["dest_station"]
    if t["phase"] == "at_station":
        return t["station"]
    if t["phase"] == "not_entered":
        return t["position"]
    return None


def _tracks(sim: Simulator, station: str) -> list[str]:
    if station in YARDS:
        return ["main"]
    return ["main", *[loop.id for loop in sim._stations[station].loops]]


def _ahead(sim: Simulator, tid: str, front: str) -> list[str]:
    order = sim._station_ids
    i = order.index(front)
    if sim._trains[tid]["direction"] == "east":
        return order[i + 1 :]
    return list(reversed(order[:i]))


def _can_depart_now(sim: Simulator, t: dict) -> bool:
    tick = sim.tick
    timers = (t["entry_tick"], t["dwell_until"], t["relief_until"], t["breakdown_until"], t["blocked_until"])
    if any(x is not None and tick < x for x in timers):
        return False
    if t["hold_until"] is not None and tick < t["hold_until"]:
        return False
    if t["station"] in sim._stations:
        stop = next((s for s in sim._train_defs[t["id"]].stops if s.station == t["station"]), None)
        served = any(a["station"] == t["station"] for a in t["arrivals"])
        if stop and served and stop.dep is not None and tick < stop.dep:
            return False
    return True


def candidate_actions(sim: Simulator, tid: str, *, exploit_bugs: bool = False) -> list[Action]:
    """Actions worth trying for one train at this tick (canonical timing, see module doc)."""
    t = sim._trains[tid]
    out: list[Action] = []
    if t["phase"] in {"finished", "reversing"}:
        return out
    if t["phase"] == "not_entered":
        if sim.tick < t["entry_tick"] or t["waits_for"]:
            return out
    stationary = t["phase"] in {"at_station", "not_entered"}
    if stationary and t["auth"] is None and _can_depart_now(sim, t):
        front = _front(sim, tid)
    elif t["phase"] == "in_section" and t["auth"]:
        path = t["auth"]["path"]
        last_leg = len([r for r in path if r in sim._section_by_id]) == 1
        if not (last_leg and t["section_exit"] == sim.tick + 1):
            front = None
        else:
            front = _front(sim, tid)
    else:
        front = None
    if front is not None:
        for station in reversed(_ahead(sim, tid, front)):  # furthest first: good incumbents early
            for track in _tracks(sim, station):
                out.append(Authorize(train=tid, station=station, track=track))
    if tid in sim._gridlock:
        out.append(Reverse(train=tid))
    if exploit_bugs and t["auth"]:
        out.append(CancelAuthority(train=tid))
    return out


def _joint_options(sim: Simulator, *, exploit_bugs: bool, action_filter: ActionFilter | None, per_train=None):
    """Yield (actions, sim_after) for every combination of at most one granted action per train."""
    tids = sorted(sim._trains)

    def rec(i: int, current: Simulator, chosen: list):
        if i == len(tids):
            yield list(chosen), current
            return
        tid = tids[i]
        options = []
        for action in candidate_actions(current, tid, exploit_bugs=exploit_bugs):
            if action_filter is not None and not action_filter(current, action):
                continue
            if not current.check(action).ok:
                continue
            options.append(action)
            if per_train is not None and len(options) >= per_train:
                break
        for action in options:
            nxt = fast_clone(current)
            nxt.apply(action)
            if not exploit_bugs and not sound(nxt):
                continue
            chosen.append(action)
            yield from rec(i + 1, nxt, chosen)
            chosen.pop()
        yield from rec(i + 1, current, chosen)  # do nothing with this train

    yield from rec(0, sim, [])


def _step(sim: Simulator) -> None:
    """Advance by one tick (repeating while `advance` returns at the same tick)."""
    start = sim.tick
    while not sim.done and sim.tick == start:
        sim.advance(max_tick=start + 1)


# ---------------------------------------------------------------------------------------------
# Lower bound (admissible)
# ---------------------------------------------------------------------------------------------


def lower_bound(sim: Simulator, never_move: frozenset[str] = frozenset()) -> float:
    """Accumulated cost + for every unfinished train min(free-run finish cost, unfinished charge).

    Free running ignores other trains, speed restrictions, restart penalties and future events, so it
    is never later than the true arrival; every cost term is non-decreasing in arrival time.
    """
    if sim.done:
        return final_cost(sim)
    total = accumulated(sim)
    shift = sim._scenario.shift_ticks
    order = sim._station_ids
    for tid in sorted(sim._trains):
        t = sim._trains[tid]
        if t["phase"] == "finished":
            continue
        d = sim._train_defs[tid]
        if tid in never_move and t["auth"] is None and t["phase"] not in {"in_section", "reversing"}:
            total += unfinished_cost(d, shift)
            continue
        east = d.direction == "east"
        speed = TRAIN_TYPES[d.type]["max_speed_kmh"]
        remaining = d.stops[t["stop_idx"] :]
        if t["phase"] in {"in_section", "reversing"}:
            if t["phase"] == "reversing":
                station = t["reversing_to"][0]
            else:
                sec = sim._section_by_id[t["section"]]
                station = sec.east if east else sec.west
            clock = t["section_exit"]
            arrived_now = True
        else:
            station = t["station"] or t["position"]
            timers = [sim.tick, t["entry_tick"]]
            timers += [x for x in (t["dwell_until"], t["relief_until"], t["breakdown_until"], t["blocked_until"]) if x]
            clock = max(timers)
            arrived_now = False
        cost = 0.0
        idx = order.index(station)
        step = 1 if east else -1
        stops = {s.station: s for s in remaining}
        feasible = True
        pos = idx
        while True:
            name = order[pos]
            stop = stops.get(name)
            if stop is not None and (arrived_now or pos != idx):
                late = clock - stop.arr
                if d.type in {"express", "local"}:
                    cost += pax_cost(d.type, d.pax, late, name in YARDS)
                elif d.type == "freight" and name in YARDS:
                    cost += freight_cost(late)
                if name in YARDS:
                    break
                clock = max(clock + stop.min_dwell, stop.dep if stop.dep is not None else clock)
            if name in YARDS and pos != idx:
                break
            nxt = pos + step
            if nxt < 0 or nxt >= len(order):
                break
            a, b = order[pos], order[nxt]
            sec = sim._section_by_pair.get((a, b)) or sim._section_by_pair.get((b, a))
            clock += travel_ticks(sec.length_m, sec.line_speed_kmh, speed)
            pos = nxt
            if clock > shift:
                feasible = False
                break
        finish = cost if feasible and clock <= shift else math.inf
        total += min(finish, unfinished_cost(d, shift))
    return total


# ---------------------------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------------------------


def replay_plan(scenario: Scenario | dict, plan: list[tuple[int, list[Action]]]) -> Simulator:
    sim = Simulator.load(scenario if isinstance(scenario, Scenario) else Scenario.model_validate(scenario))
    by_tick: dict[int, list[Action]] = {}
    for tick, actions in plan:
        by_tick.setdefault(tick, []).extend(actions)
    while not sim.done:
        for action in by_tick.get(sim.tick, []):
            result = sim.apply(action)
            if not result.ok:
                raise AssertionError(f"t={sim.tick} replay refused {action}: {result.message}")
        if sim.tick >= max(by_tick, default=-1):
            sim.end_shift()
            break
        _step(sim)
    return sim


# ---------------------------------------------------------------------------------------------
# Exhaustive branch and bound
# ---------------------------------------------------------------------------------------------


@dataclass
class SearchResult:
    cost: float
    plan: list[tuple[int, list[Action]]]
    nodes: int
    proven: bool

    @property
    def log(self) -> list[dict]:
        return [{"tick": tick, "actions": [a.model_dump(mode="json") for a in actions]} for tick, actions in self.plan]


def exhaustive_search(
    scenario: Scenario | dict,
    *,
    action_filter: ActionFilter | None = None,
    exploit_bugs: bool = False,
    max_nodes: int = 2_000_000,
    incumbent: float = math.inf,
    must_finish: frozenset[str] = frozenset(),
    never_move: frozenset[str] = frozenset(),
) -> SearchResult:
    """Minimum-cost plan with full knowledge (exact within canonical timing; see module doc).

    `must_finish`: only plans in which these trains finish count (to price "finish it anyway").
    `never_move`: these trains get no authority at all (e.g. "hold all freight").
    """
    if never_move:
        user_filter = action_filter

        def action_filter(sim, action, _f=user_filter):
            return action.train not in never_move and (_f is None or _f(sim, action))

    def valid_end(sim: Simulator) -> bool:
        return all(sim._trains[tid]["phase"] == "finished" for tid in must_finish)

    root = Simulator.load(scenario if isinstance(scenario, Scenario) else Scenario.model_validate(scenario))
    best = {"cost": incumbent, "plan": None}
    memo: dict[str, float] = {}
    nodes = 0

    # Doing nothing at all is always a valid plan and a first incumbent.
    idle = fast_clone(root)
    idle.end_shift()
    if final_cost(idle) < best["cost"] and valid_end(idle):
        best.update(cost=final_cost(idle), plan=[])

    def visit(sim: Simulator, plan: list):
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            raise SearchLimit(f"more than {max_nodes} nodes")
        if sim.done:
            cost = final_cost(sim)
            if cost < best["cost"] - 1e-9 and valid_end(sim):
                best.update(cost=cost, plan=list(plan))
            return
        if lower_bound(sim, never_move) >= best["cost"] - 1e-9:
            return
        key = state_key(sim)
        acc = accumulated(sim)
        if memo.get(key, math.inf) <= acc + 1e-9:
            return
        memo[key] = acc
        options = list(_joint_options(sim, exploit_bugs=exploit_bugs, action_filter=action_filter))
        # Ending the shift now is a terminal option (no further authorities).
        stop = fast_clone(sim)
        stop.end_shift()
        if final_cost(stop) < best["cost"] - 1e-9 and (exploit_bugs or sound(stop)) and valid_end(stop):
            best.update(cost=final_cost(stop), plan=list(plan))
        for actions, after in options:
            nxt = fast_clone(after) if after is sim else after
            _step(nxt)
            if not exploit_bugs and not sound(nxt):
                continue
            if actions:
                plan.append((sim.tick, actions))
            visit(nxt, plan)
            if actions:
                plan.pop()

    proven = True
    try:
        visit(root, [])
    except SearchLimit:
        proven = False
    return SearchResult(cost=best["cost"], plan=best["plan"] or [], nodes=nodes, proven=proven)


# ---------------------------------------------------------------------------------------------
# Beam search (T06 §2.7)
# ---------------------------------------------------------------------------------------------


def offline_search(
    scenario: Scenario | dict,
    beam: int = 16,
    per_train: int = 4,
    *,
    seed_plan: list[tuple[int, list[Action]]] | None = None,
    action_filter: ActionFilter | None = None,
    exploit_bugs: bool = False,
) -> tuple[float, list[dict]]:
    """Beam search over ticks with full knowledge; returns (cost, log). Never worse than `seed_plan`."""
    root = Simulator.load(scenario if isinstance(scenario, Scenario) else Scenario.model_validate(scenario))
    frontier: list[tuple[float, Simulator, list]] = [(lower_bound(root), root, [])]
    idle = fast_clone(root)
    idle.end_shift()
    best_cost, best_plan = final_cost(idle), []
    if seed_plan is not None:
        seeded = replay_plan(scenario, seed_plan)
        if final_cost(seeded) < best_cost:
            best_cost, best_plan = final_cost(seeded), list(seed_plan)
    while frontier:
        children: dict[str, tuple[float, Simulator, list]] = {}
        for _, sim, plan in frontier:
            for actions, after in _joint_options(
                sim, exploit_bugs=exploit_bugs, action_filter=action_filter, per_train=per_train
            ):
                nxt = fast_clone(after) if after is sim else after
                _step(nxt)
                if not exploit_bugs and not sound(nxt):
                    continue
                new_plan = plan + ([(sim.tick, actions)] if actions else [])
                if nxt.done:
                    if final_cost(nxt) < best_cost:
                        best_cost, best_plan = final_cost(nxt), new_plan
                    continue
                score = lower_bound(nxt)
                key = state_key(nxt)
                if key not in children or score < children[key][0]:
                    children[key] = (score, nxt, new_plan)
        frontier = sorted(children.values(), key=lambda x: x[0])[:beam]
        frontier = [x for x in frontier if x[0] < best_cost]
    return best_cost, [{"tick": tick, "actions": [a.model_dump(mode="json") for a in acts]} for tick, acts in best_plan]


def plan_from_log(log: list[dict]) -> list[tuple[int, list[Action]]]:
    return [(entry["tick"], [ActionAdapter.validate_python(a) for a in entry["actions"]]) for entry in log]
