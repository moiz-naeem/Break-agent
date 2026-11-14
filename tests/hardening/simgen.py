"""Random scenarios, random action sequences and the invariant checker used by the hardening tests.

Generators take a *chooser* so the same code runs under Hypothesis (shrinkable draws) and under a
plain `random.Random(seed)` (for the frequency statistics). Everything here only reads the simulator;
private attributes are read to check reservation integrity, which the public `State` cannot show.
"""

from __future__ import annotations

import contextlib
import copy
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from unittest import mock

from hypothesis import strategies as st

from railway.sim.constants import TIME_CAP, TRAIN_TYPES, TURNAROUND
from railway.sim.cost import PAX_GRACE, TERMINUS_MULT, UNFINISHED_COST, W_EXPRESS, W_FREIGHT, W_LOCAL
from railway.sim.engine import Simulator
from railway.sim.model import Authorize, CancelAuthority, Hold, RelieveCrew, Reverse, Scenario
from railway.sim.timing import travel_ticks
from railway.util import canonical_json

COMPONENTS = ("pax_lateness", "freight_lateness", "crew_timeout", "reversal", "unfinished")
EXTRA_FACTORS = ("F5", "F6", "F7", "F8", "F10", "F15", "F16")
BASE_FACTORS = ("F1", "F2", "F3", "F4")

# Global frequency counters (read by the statistics test and the final report).
STATS: Counter = Counter()


# ---------------------------------------------------------------------------------------------
# Choosers
# ---------------------------------------------------------------------------------------------


class HypChooser:
    def __init__(self, data):
        self.data = data

    def int(self, lo: int, hi: int) -> int:
        return self.data.draw(st.integers(lo, hi))

    def choice(self, seq):
        seq = list(seq)
        return seq[self.int(0, len(seq) - 1)]

    def bool(self, p: float = 0.5) -> bool:
        return self.int(0, 99) < round(p * 100)

    def weighted(self, pairs):
        total = sum(w for _, w in pairs)
        x = self.int(0, total - 1)
        for value, weight in pairs:
            if x < weight:
                return value
            x -= weight
        raise AssertionError("unreachable")


class RandChooser(HypChooser):
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def int(self, lo: int, hi: int) -> int:
        return self.rng.randint(lo, hi)


# ---------------------------------------------------------------------------------------------
# Scenario generator
# ---------------------------------------------------------------------------------------------


def gen_scenario(
    c,
    *,
    factors: tuple[str, ...] | None = None,
    events: bool = True,
    max_stations: int = 3,
    max_trains: int = 4,
    opposing_bias: float = 0.9,
    gridlock_bias: bool = False,
) -> dict:
    """A small valid scenario dict. Biased toward opposing trains and few or no loops.

    `gridlock_bias` makes head-on meetings likely: 2-3 stations, mostly no loops, early entries.
    """
    n = c.int(2, max(2, max_stations)) if gridlock_bias else c.int(1, max_stations)
    loop_mode = c.weighted([("none", 5), ("few", 3)] if gridlock_bias else [("none", 3), ("few", 4), ("rich", 2)])
    stations = []
    for i in range(1, n + 1):
        if loop_mode == "none":
            nloops = 0
        elif loop_mode == "few":
            nloops = c.int(0, 1)
        else:
            nloops = c.int(1, 3)
        loops = [
            {
                "id": f"L{k}",
                "length_m": c.choice([150, 300, 700]),
                "kind": c.weighted([("platform", 3), ("goods", 1)]),
            }
            for k in range(1, nloops + 1)
        ]
        stations.append({"id": f"S{i}", "major": c.bool(0.5), "crew_station": c.bool(0.5), "loops": loops})
    ids = ["YW", *[s["id"] for s in stations], "YE"]
    sections = [
        {
            "id": f"{a}-{b}",
            "west": a,
            "east": b,
            "length_m": c.choice([2000, 4000, 6000, 9000]),
            "line_speed_kmh": c.choice([80, 120, 140]),
        }
        for a, b in zip(ids, ids[1:], strict=False)
    ]
    if factors is None:
        extra = [f for f in EXTRA_FACTORS if c.bool(0.5)]
    else:
        extra = [f for f in factors if f not in BASE_FACTORS]
    fs = sorted(set(BASE_FACTORS) | set(extra))  # the schema wants lexical order
    shift = c.choice([60, 90, 120])

    ntrains = c.int(2, max_trains)
    opposing = c.bool(opposing_bias)
    counters = {"E": 0, "L": 0, "F": 0, "N": 0}
    trains = []
    for k in range(ntrains):
        ttype = c.weighted([("express", 4), ("local", 5), ("freight", 4), ("engineering", 1)])
        if opposing and k < 2:
            direction = "east" if k == 0 else "west"
        else:
            direction = c.choice(["east", "west"])
        prefix = {"express": "E", "local": "L", "freight": "F", "engineering": "N"}[ttype]
        counters[prefix] += 1
        tid = f"{prefix}{counters[prefix]}"
        entry = c.int(0, 4) if gridlock_bias else c.weighted([(0, 3), (c.int(1, 25), 4)])
        route = ids if direction == "east" else list(reversed(ids))
        speed = TRAIN_TYPES[ttype]["max_speed_kmh"]
        tick = entry
        stops = []
        for a, b in zip(route, route[1:], strict=False):
            sec = next(s for s in sections if {s["west"], s["east"]} == {a, b})
            tick += travel_ticks(sec["length_m"], sec["line_speed_kmh"], speed)
            if b in {"YW", "YE"}:
                stops.append({"station": b, "arr": tick, "dep": None, "min_dwell": 0})
                break
            st_def = next(s for s in stations if s["id"] == b)
            serves = ttype == "local" or (ttype == "express" and st_def["major"])
            if serves:
                dwell = c.int(1, 2)
                dep = tick + dwell + c.int(0, 1)
                stops.append({"station": b, "arr": tick, "dep": dep, "min_dwell": dwell})
                tick = dep
        crew = 0
        if "F10" in fs:
            crew = c.weighted([(0, 2), (c.int(150, 236), 3)])
        trains.append(
            {
                "id": tid,
                "type": ttype,
                "direction": direction,
                "pax": c.choice([50, 100, 200]) if ttype in {"express", "local"} else 0,
                "entry_tick": entry,
                "stops": stops,
                "next_service": None,
                "crew_used_at_entry": crew,
            }
        )
    trains.sort(key=lambda t: t["id"])
    if "F6" in fs:
        easts = [t for t in trains if t["direction"] == "east"]
        wests = [t for t in trains if t["direction"] == "west"]
        if easts and wests and c.bool(0.8):
            a, b = (c.choice(easts), c.choice(wests))
            if c.bool(0.5):
                a, b = b, a
            a["next_service"] = b["id"]

    evs = []
    if events:
        sec_ids = [s["id"] for s in sections]
        if "F7" in fs:
            for _ in range(c.int(0, 2)):
                start = c.int(1, shift - 5)
                end = min(shift, start + c.int(3, 30))
                evs.append(
                    {
                        "kind": "possession",
                        "reveal_tick": max(0, start - c.int(0, 60)),
                        "section": c.choice(sec_ids),
                        "start": start,
                        "end": end,
                    }
                )
        if "F8" in fs:
            for _ in range(c.int(0, 2)):
                first = c.choice(sec_ids)
                secs = [first]
                second = c.choice(sec_ids)
                if second != first and c.bool(0.4):
                    secs.append(second)
                start = c.int(0, shift - 1)
                evs.append(
                    {
                        "kind": "speed_restriction",
                        "reveal_tick": max(0, start - c.int(0, 60)),
                        "sections": secs,
                        "speed_kmh": c.choice([20, 40, 60, 100]),
                        "start": start,
                        "end": start + c.int(5, 60),
                    }
                )
        if "F16" in fs:
            late_ok = [t for t in trains if t["entry_tick"] >= 1]
            if late_ok and c.bool(0.8):
                t = c.choice(late_ok)
                evs.append(
                    {
                        "kind": "late_entry",
                        "reveal_tick": c.choice([max(0, t["entry_tick"] - 20), c.int(0, t["entry_tick"] - 1)]),
                        "train": t["id"],
                        "new_entry_tick": t["entry_tick"] + c.int(1, 30),
                    }
                )
        if "F15" in fs:
            for _ in range(c.int(0, 2)):
                tick = c.int(0, shift - 1)
                evs.append(
                    {
                        "kind": "breakdown",
                        "reveal_tick": tick,
                        "train": c.choice(trains)["id"],
                        "tick": tick,
                        "duration": c.int(1, 20),
                    }
                )
    for i, ev in enumerate(evs, start=1):
        ev["id"] = f"EV{i}"
    evs.sort(key=lambda e: (e["reveal_tick"], e["id"]))
    scenario = {
        "schema_version": 1,
        "rules_version": "0.1-provisional",
        "scenario_id": "FIX-hardening",
        "seed": 0,
        "preset": "fixture",
        "shift_ticks": shift,
        "factors": fs,
        "network": {"yards": {"west": "YW", "east": "YE"}, "stations": stations, "sections": sections},
        "trains": trains,
        "events": evs,
    }
    Scenario.model_validate(scenario)  # a generator bug must fail loudly
    return scenario


# ---------------------------------------------------------------------------------------------
# Action generator
# ---------------------------------------------------------------------------------------------

ALL_KINDS = (
    "authorize",
    "extend",  # pseudo-kind: when absent, only trains without an authority are authorized
    "advance",
    "advance_to",
    "cancel",
    "hold",
    "reverse",
    "relieve",
    "junk",
    "end_shift",
)
KNOWN_BUG_KINDS = ("cancel", "reverse", "hold", "extend")  # see tests/hardening/test_bugs.py
SAFE_KINDS = tuple(k for k in ALL_KINDS if k not in KNOWN_BUG_KINDS)


def _route(sim: Simulator) -> list[str]:
    return sim._station_ids


def train_front(sim: Simulator, tid: str) -> str:
    """The station an authority for `tid` would start from (mirrors PRD §7.5 rule 2)."""
    t = sim._trains[tid]
    if t["auth"]:
        return t["auth"]["dest_station"]
    if t["phase"] in {"in_section", "reversing"}:
        sec = sim._section_by_id[t["section"]]
        return sec.east if t["direction"] == "east" else sec.west
    return t["station"] or t["position"]


def gen_step(c, sim: Simulator, kinds=ALL_KINDS, gridlock_bias: bool = False):
    """One step: ("apply", action) | ("advance", max_tick|None) | ("end_shift",).

    `gridlock_bias` parks trains on the next station's main line, which drives head-on gridlocks.
    """
    s = sim._trains
    ids = sorted(s)
    if sim._gridlock and "reverse" in kinds and c.bool(0.5):
        return ("apply", Reverse(train=c.choice(sim._gridlock)))
    reversing = [t for t in ids if s[t]["phase"] == "reversing"]
    if reversing and "reverse" in kinds and c.bool(0.3):
        # Attack the weakest area: try to authorize a train while it is reversing.
        return ("apply", Authorize(train=c.choice(reversing), station=c.choice(sim._station_ids), track="main"))
    weights = {
        "authorize": 45,
        "advance": 25,
        "advance_to": 10,
        "cancel": 5,
        "hold": 4,
        "reverse": 3,
        "relieve": 3,
        "junk": 3,
        "end_shift": 1,
    }
    kind = c.weighted([(k, weights[k]) for k in kinds if k in weights])
    if kind == "advance":
        return ("advance", None)
    if kind == "advance_to":
        return ("advance", sim.tick + c.int(1, 6))
    if kind == "end_shift":
        return ("end_shift",)
    live = [t for t in ids if s[t]["phase"] != "finished"] or ids
    if kind == "authorize" and "extend" not in kinds:
        live = [t for t in live if s[t]["auth"] is None] or live
        if s[live[0]]["auth"] is not None:
            return ("advance", None)
    tid = c.choice(live)
    if kind == "authorize":
        route = _route(sim)
        front = train_front(sim, tid)
        idx = route.index(front)
        step = 1 if s[tid]["direction"] == "east" else -1
        ahead = [route[i] for i in range(idx + step, len(route) if step == 1 else -1, step)]
        if not ahead or c.bool(0.05):
            station = c.choice(route)
        else:
            dist = c.weighted([(1, 9), (2, 1)] if gridlock_bias else [(1, 6), (2, 3), (len(ahead), 1)])
            station = ahead[min(dist, len(ahead)) - 1]
        if station in sim._stations and gridlock_bias and c.bool(0.8):
            track = "main"
        elif station in sim._stations:
            loops = [x.id for x in sim._stations[station].loops]
            track = c.weighted([("main", 5), ("loop", 4 if loops else 0), ("any", 1)])
            if track == "loop":
                track = c.choice(loops)
            elif track == "any":
                track = c.choice(["main", "L1", "L2", "L3"])
        else:
            track = "main" if c.bool(0.95) else "L1"
        return ("apply", Authorize(train=tid, station=station, track=track))
    if kind == "cancel":
        return ("apply", CancelAuthority(train=tid))
    if kind == "hold":
        return ("apply", Hold(train=tid, until_tick=sim.tick + c.int(-1, 10)))
    if kind == "reverse":
        return ("apply", Reverse(train=tid))
    if kind == "relieve":
        return ("apply", RelieveCrew(train=tid))
    # junk
    junk = c.choice(
        [
            {"kind": "authorize", "train": tid},
            {"kind": "hold", "train": tid, "until_tick": "soon"},
            Authorize(train="X9", station="S1", track="main"),
            Authorize(train=tid, station="S9", track="main"),
            Authorize(train=tid, station="YE", track="L1"),
            {"kind": "teleport", "train": tid},
        ]
    )
    return ("apply", junk)


def run_step(sim: Simulator, step):
    if step[0] == "apply":
        return sim.apply(step[1])
    if step[0] == "advance":
        return sim.advance(max_tick=step[1])
    return sim.end_shift()


# ---------------------------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------------------------

_INTERNAL = (
    "tick",
    "done",
    "_last_dp",
    "_cost_finalized",
    "_trains",
    "_holders",
    "_entered",
    "_revealed_events",
    "_restrictions",
    "_possessions",
    "_gridlock",
)


def internal(sim: Simulator) -> dict:
    out = {k: copy.deepcopy(getattr(sim, k)) for k in _INTERNAL}
    out["_cost"] = sim._cost.model_dump()
    out["_pending_changes"] = [x.model_dump() for x in sim._pending_changes]
    return out


def public(sim: Simulator) -> str:
    return canonical_json(sim.state().model_dump(mode="json"))


def dump(result) -> str:
    return canonical_json(result.model_dump(mode="json"))


# ---------------------------------------------------------------------------------------------
# Departure recorder (wraps the engine's departure routine; read-only)
# ---------------------------------------------------------------------------------------------


@contextlib.contextmanager
def record_departures(log: list):
    original = Simulator._start_move

    def wrapped(self, t):
        before = {
            "phase": t["phase"],
            "station": t["station"],
            "had_auth": t["auth"] is not None,
            "hold_until": t["hold_until"],
            "dwell_until": t["dwell_until"],
            "relief_until": t["relief_until"],
            "breakdown_until": t["breakdown_until"],
            "blocked_until": t["blocked_until"],
            "entry_tick": t["entry_tick"],
            "arrivals": list(t["arrivals"]),
        }
        original(self, t)
        if before["phase"] == "at_station" and t["phase"] == "in_section":
            log.append({"sim": self, "tick": self.tick, "train": t["id"], "section": t["section"], **before})

    with mock.patch.object(Simulator, "_start_move", wrapped):
        yield


# ---------------------------------------------------------------------------------------------
# Independent oracles
# ---------------------------------------------------------------------------------------------


def occupancy(sim: Simulator) -> dict[str, list[str]]:
    occ: dict[str, list[str]] = {}
    for tid in sorted(sim._trains):
        t = sim._trains[tid]
        if t["phase"] == "at_station" and t["station"] in sim._stations:
            occ.setdefault(f"{t['station']}/{t['track']}", []).append(tid)
        elif t["phase"] in {"in_section", "reversing"}:
            occ.setdefault(t["section"], []).append(tid)
    return occ


def oracle_gridlock(sim: Simulator) -> list[str]:
    """Trains that are deadlocked for good, from public state only (occupation + authority paths).

    PRD §7.5 rule 10's intent: a blocked train is in a gridlock when it can never move again without a
    reversal. A train is *stuck* when every feasible next-station move is blocked by another stuck train
    (a move blocked only by a train that can still move will clear). Gridlock = stuck trains in a cycle.
    """
    state = sim.state()
    stations = sim._stations
    order = sim._station_ids
    occ: dict[str, set[str]] = {}
    for t in state.trains:
        if t.phase == "at_station" and t.station in stations:
            occ.setdefault(f"{t.station}/{t.track}", set()).add(t.id)
        elif t.phase in {"in_section", "reversing"}:
            occ.setdefault(t.section, set()).add(t.id)
        if t.authority:
            for res in t.authority.path:
                occ.setdefault(res, set()).add(t.id)
    blocked = {
        t.id: t for t in state.trains if t.phase == "at_station" and t.station in stations and t.authority is None
    }
    # moves[T] = list of blocker sets, one per feasible next-station move (empty set: a free move).
    moves: dict[str, list[set[str]]] = {}
    for tid, t in blocked.items():
        i = order.index(t.station)
        nxt = order[i + (1 if t.direction == "east" else -1)]
        sec = next(s.id for s in sim._scenario.network.sections if {s.west, s.east} == {t.station, nxt})
        sec_block = occ.get(sec, set()) - {tid}
        tracks: list[str | None] = []
        if nxt in stations:
            tracks.append(f"{nxt}/main")
            length = TRAIN_TYPES[t.type]["length_m"]
            for loop in stations[nxt].loops:
                if length <= loop.length_m and (loop.kind == "platform" or t.type not in {"express", "local"}):
                    tracks.append(f"{nxt}/{loop.id}")
        else:
            tracks.append(None)  # a yard never blocks
        moves[tid] = [sec_block | ((occ.get(track, set()) - {tid}) if track else set()) for track in tracks]
    # Stuck forever = greatest set S where every move of every member has a blocker inside S.
    stuck = set(moves)
    changed = True
    while changed:
        changed = False
        for tid in sorted(stuck):
            if any(not (m & stuck) for m in moves[tid]):
                stuck.discard(tid)
                changed = True
    graph = {tid: set().union(*moves[tid]) & stuck for tid in stuck}

    def reach(a):
        seen, todo = set(), [a]
        while todo:
            x = todo.pop()
            for y in graph.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    todo.append(y)
        return seen

    reaches = {a: reach(a) for a in graph}
    members = {a for a in graph if any(b != a and a in reaches[b] and b in reaches[a] for b in graph)}
    return sorted(members)


def oracle_cost(sim: Simulator) -> dict[str, float]:
    """Recompute pax/freight/unfinished from the recorded arrivals and the PRD §7.6 formulas."""
    pax = freight = unfinished = 0.0
    defs = sim._train_defs
    state = sim.state()
    for t in state.trains:
        d = defs[t.id]
        for a in t.arrivals:
            late = a.actual - a.sched
            terminus = a.station in {"YW", "YE"}
            if d.type in {"express", "local"}:
                w = W_EXPRESS if d.type == "express" else W_LOCAL
                pax += w * d.pax / 100 * max(0, late - PAX_GRACE) * (TERMINUS_MULT if terminus else 1)
            elif d.type == "freight" and terminus:
                freight += W_FREIGHT * max(0, late)
        if state.done and t.phase != "finished":
            late = state.shift_ticks - d.stops[-1].arr
            if d.type in {"express", "local"}:
                w = W_EXPRESS if d.type == "express" else W_LOCAL
                unfinished += UNFINISHED_COST + w * d.pax / 100 * max(0, late - PAX_GRACE) * TERMINUS_MULT
            elif d.type == "freight":
                unfinished += UNFINISHED_COST + W_FREIGHT * max(0, late)
            else:
                unfinished += UNFINISHED_COST
    return {"pax_lateness": pax, "freight_lateness": freight, "unfinished": unfinished}


def unrevealed(sim: Simulator) -> list:
    return [e for e in sim._scenario.events if e.id not in sim._revealed_events]


def next_reveal(sim: Simulator) -> float:
    pending = unrevealed(sim)
    return min((e.reveal_tick for e in pending), default=math.inf)


# ---------------------------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------------------------


@dataclass
class Violation(Exception):
    name: str
    detail: str

    def __str__(self):
        return f"{self.name}: {self.detail}"


def fail(name: str, detail: str):
    raise Violation(name, detail)


def check_resources(sim: Simulator):
    """One train per resource, and every occupied or authorized resource is reserved by its train."""
    occ = occupancy(sim)
    for res, tids in occ.items():
        if len(tids) > 1:
            fail("one_train_per_resource", f"t={sim.tick} {res} occupied by {tids}")
    for res, tids in occ.items():
        holder = sim._holders.get(res)
        if holder != tids[0]:
            fail("occupied_not_reserved", f"t={sim.tick} {res} occupied by {tids[0]} but reserved by {holder}")
    for tid, t in sim._trains.items():
        if t["auth"]:
            for res in t["auth"]["path"]:
                if res in {"YW", "YE"}:
                    continue
                if sim._holders.get(res) != tid:
                    fail(
                        "authority_not_reserved",
                        f"t={sim.tick} {tid} path has {res} but holder is {sim._holders.get(res)}",
                    )


def check_loops(sim: Simulator):
    for tid, t in sim._trains.items():
        if t["phase"] != "at_station" or t["station"] not in sim._stations or t["track"] in (None, "main"):
            continue
        loop = next((x for x in sim._stations[t["station"]].loops if x.id == t["track"]), None)
        if loop is None:
            fail("unknown_track_occupied", f"{tid} on {t['station']}/{t['track']}")
        if TRAIN_TYPES[t["type"]]["length_m"] > loop.length_m:
            fail("loop_length", f"{tid} on {t['station']}/{loop.id} ({loop.length_m} m)")
        if loop.kind == "goods" and t["type"] in {"express", "local"}:
            fail("platform", f"{tid} on goods loop {t['station']}/{loop.id}")


def check_cost(sim: Simulator, previous: dict | None) -> dict:
    cost = sim.cost().model_dump()
    for k in COMPONENTS:
        if cost[k] < 0:
            fail("cost_negative", f"{k}={cost[k]}")
        if previous is not None and cost[k] < previous[k] - 1e-9:
            fail("cost_decreased", f"{k} {previous[k]} -> {cost[k]}")
    if abs(cost["total"] - sum(cost[k] for k in COMPONENTS)) > 1e-6:
        fail("cost_total", f"total {cost['total']} != sum")
    oracle = oracle_cost(sim)
    for k, v in oracle.items():
        if abs(cost[k] - v) > 1e-4:
            fail("cost_oracle", f"t={sim.tick} {k}: sim {cost[k]} vs formula {v}")
    return cost


def _position(t: dict):
    return (t["phase"], t["station"], t["track"], t["section"])


def check_movement(before: dict, sim: Simulator, step, result):
    """No movement without authority (or a granted reversal / an entry at the origin yard)."""
    for tid, b in before["_trains"].items():
        a = sim._trains[tid]
        if _position(b) == _position(a):
            continue
        if b["auth"] is not None or b["phase"] == "reversing":
            continue
        if b["phase"] == "not_entered" and a["phase"] == "at_station" and a["station"] in {"YW", "YE"}:
            continue
        if (
            step[0] == "apply"
            and isinstance(step[1], Reverse)
            and step[1].train == tid
            and result.ok
            and a["phase"] == "reversing"
        ):
            continue
        fail("move_without_authority", f"{tid}: {_position(b)} -> {_position(a)} on {step}")


def check_departures(log: list, sim: Simulator, *, check_hold: bool = False):
    defs = sim._train_defs
    for d in log:
        if d["sim"] is not sim:
            continue
        tid = d["train"]
        sec = sim._section_by_id[d["section"]]
        direction = defs[tid].direction
        origin = sec.west if direction == "east" else sec.east
        if origin != d["station"]:
            fail("non_contiguous_move", f"t={d['tick']} {tid} entered {sec.id} from {d['station']}")
        if not d["had_auth"]:
            fail("move_without_authority", f"{tid} departed without authority")
        if d["tick"] < d["entry_tick"]:
            fail("early_departure", f"{tid} left the origin at {d['tick']} < entry {d['entry_tick']}")
        for timer in ("relief_until", "breakdown_until", "blocked_until", "dwell_until"):
            if d[timer] is not None and d["tick"] < d[timer]:
                fail("early_departure", f"{tid} departed t={d['tick']} before {timer}={d[timer]}")
        if check_hold and d["hold_until"] is not None and d["tick"] < d["hold_until"]:
            fail("departed_while_held", f"{tid} departed t={d['tick']} held until {d['hold_until']}")
        stop = next((s for s in defs[tid].stops if s.station == d["station"]), None)
        served = [a for a in d["arrivals"] if a["station"] == d["station"]]
        if stop and stop.dep is not None and served:
            arrival = served[-1]["actual"]
            if d["tick"] < stop.dep or d["tick"] < arrival + stop.min_dwell:
                fail(
                    "early_departure",
                    f"{tid} left {d['station']} at {d['tick']} (arr {arrival}, dwell {stop.min_dwell}, dep {stop.dep})",
                )
    log.clear()


def check_rotation(sim: Simulator):
    """F6: a successor is never in the network before its predecessor finished + TURNAROUND."""
    if "F6" not in sim._scenario.factors:
        return
    for succ, pred in sim._rotation_predecessor.items():
        s, p = sim._trains[succ], sim._trains[pred]
        if s["phase"] == "not_entered":
            continue
        if p["phase"] != "finished":
            fail("rotation_violated", f"t={sim.tick} {succ} is {s['phase']} while {pred} is {p['phase']}")
        if sim.tick < p["finished_tick"] + TURNAROUND:
            fail("rotation_violated", f"t={sim.tick} {succ} entered before {pred} finished+{TURNAROUND}")


def check_no_unrevealed(sim: Simulator):
    state = sim.state()
    text = canonical_json(state.model_dump(mode="json"))
    static = canonical_json(sim.static_info().model_dump(mode="json"))
    for key in ("events", "reveal_tick", "scenario_id", "seed"):
        if f'"{key}"' in static:
            fail("static_leak", key)
    shown = {x["id"] for x in state.possessions} | {x["id"] for x in state.restrictions}
    for e in unrevealed(sim):
        if e.id in shown or f'"{e.id}"' in text:
            fail("unrevealed_event_visible", f"t={sim.tick} {e.id} ({e.kind}, reveal {e.reveal_tick})")
        if e.kind == "late_entry":
            t = next(x for x in state.trains if x.id == e.train)
            if t.entry_tick == e.new_entry_tick and sim._train_defs[e.train].entry_tick != e.new_entry_tick:
                pred = sim._rotation_predecessor.get(e.train)
                if pred is None or "F6" not in sim._scenario.factors:
                    fail("unrevealed_late_entry_visible", f"t={sim.tick} {e.train} entry {t.entry_tick}")
        if e.kind == "breakdown":
            t = next(x for x in state.trains if x.id == e.train)
            if t.breakdown_until == e.tick + e.duration:
                fail("unrevealed_breakdown_visible", f"t={sim.tick} {e.train}")


# ---------------------------------------------------------------------------------------------
# Episode driver
# ---------------------------------------------------------------------------------------------

ALL_CHECKS = frozenset(
    {
        "resources",
        "loops",
        "rotation",
        "cost",
        "movement",
        "departures",
        "hold",
        "refusal_noop",
        "check_equals_apply",
        "no_unrevealed",
        "gridlock_sound",
        "gridlock_complete",
        "decision_points",
        "belief",
    }
)


@dataclass
class Trace:
    scenario: dict
    steps: list = field(default_factory=list)
    results: list = field(default_factory=list)
    states: list = field(default_factory=list)
    gridlocks: int = 0
    reversals: int = 0
    belief_checks: int = 0


def run_episode(
    c, scenario: dict, *, n_steps: int, checks=ALL_CHECKS, kinds=ALL_KINDS, finish=True, gridlock_bias=False
) -> Trace:
    sim = Simulator.load(Scenario.model_validate(scenario))
    trace = Trace(scenario=scenario)
    log: list = []
    prev_cost = None
    belief = None
    belief_horizon = math.inf
    STATS["episodes"] += 1
    had_gridlock = had_reverse = False
    with record_departures(log):
        if "no_unrevealed" in checks:
            check_no_unrevealed(sim)
        for i in range(n_steps + (1 if finish else 0)):
            if i == n_steps:
                step = ("end_shift",)
            else:
                step = gen_step(c, sim, kinds, gridlock_bias)
            before = internal(sim)
            before_public = public(sim)
            grid_before = set(sim._gridlock)
            if step[0] == "apply" and "check_equals_apply" in checks:
                checked = sim.check(step[1])
                if internal(sim) != before or public(sim) != before_public:
                    fail("check_mutated_state", f"{step}")
            result = run_step(sim, step)
            trace.steps.append(step)
            trace.results.append(dump(result))
            trace.states.append(public(sim))
            if step[0] == "apply":
                if "check_equals_apply" in checks and dump(checked) != dump(result):
                    fail("check_differs_from_apply", f"{step}: check {dump(checked)} apply {dump(result)}")
                if not result.ok and "refusal_noop" in checks:
                    if public(sim) != before_public:
                        fail("refusal_changed_public_state", f"{step} -> {result.message}")
                    if internal(sim) != before:
                        changed = sorted(k for k in before if before[k] != internal(sim)[k])
                        fail("refusal_changed_internal_state", f"{step} -> {result.message}; changed {changed}")
                if isinstance(step[1], Reverse) and result.ok:
                    trace.reversals += 1
                    STATS["reversals_granted"] += 1
                    had_reverse = True
                elif isinstance(step[1], Reverse):
                    STATS[f"reverse_refused_{result.rule}"] += 1
            else:
                kinds_seen = [r.kind for r in result.reasons]
                if "GRIDLOCK" in kinds_seen:
                    trace.gridlocks += 1
                    STATS["gridlock_triggers"] += 1
                    had_gridlock = True
                if "decision_points" in checks and step[0] == "advance":
                    limit = step[1]
                    if not result.done and not result.reasons and (limit is None or result.tick < limit):
                        fail("advance_without_reason", f"t={result.tick} {step}")
                    if not result.done and result.tick > before["_last_dp"] + TIME_CAP:
                        fail("time_cap_exceeded", f"{before['_last_dp']} -> {result.tick}")
                    new_members = set(sim._gridlock) - grid_before
                    if new_members and "GRIDLOCK" not in kinds_seen and not result.done:
                        fail("gridlock_not_reported", f"t={result.tick} new members {sorted(new_members)}")
                if not sim.done and ({"gridlock_sound", "gridlock_complete"} & set(checks)):
                    expect = set(oracle_gridlock(sim))
                    got = set(sim.state().gridlock)
                    if "gridlock_sound" in checks and got - expect:
                        fail(
                            "gridlock_false_positive", f"t={sim.tick} sim {sorted(got)} vs deadlocked {sorted(expect)}"
                        )
                    if "gridlock_complete" in checks and expect - got:
                        fail(
                            "gridlock_false_negative", f"t={sim.tick} sim {sorted(got)} vs deadlocked {sorted(expect)}"
                        )
            if "resources" in checks:
                check_resources(sim)
            if "loops" in checks:
                check_loops(sim)
            if "rotation" in checks:
                check_rotation(sim)
            if "cost" in checks:
                prev_cost = check_cost(sim, prev_cost)
            if "movement" in checks:
                check_movement(before, sim, step, result)
            if "departures" in checks:
                check_departures(log, sim, check_hold="hold" in checks)
            if "no_unrevealed" in checks:
                check_no_unrevealed(sim)
            # Belief simulator: compare with the real one until the next reveal.
            if belief is not None:
                belief_result = run_step(belief, step)
                if step[0] != "apply" and sim.tick >= belief_horizon:
                    belief = None
                else:
                    trace.belief_checks += 1
                    if dump(belief_result) != dump(result):
                        fail("belief_result_differs", f"t={sim.tick} {step}: {dump(belief_result)} vs {dump(result)}")
                    if public(belief) != public(sim):
                        fail("belief_state_differs", f"t={sim.tick} after {step}")
            if (
                "belief" in checks
                and belief is None
                and step[0] == "advance"
                and not sim.done
                and next_reveal(sim) > sim.tick
            ):
                belief = Simulator.from_observation(sim.static_info(), sim.state())
                belief_horizon = next_reveal(sim)
                if public(belief) != public(sim):
                    fail("belief_roundtrip", f"t={sim.tick}")
            if sim.done and i < n_steps:
                break
    if had_gridlock:
        STATS["episodes_with_gridlock"] += 1
    if had_reverse:
        STATS["episodes_with_reversal"] += 1
    trace.final = sim
    return trace


def replay(scenario: dict, steps) -> tuple[list, list, Simulator]:
    sim = Simulator.load(Scenario.model_validate(scenario))
    results, states = [], []
    for step in steps:
        results.append(dump(run_step(sim, step)))
        states.append(public(sim))
    return results, states, sim
