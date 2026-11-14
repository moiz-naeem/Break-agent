"""The v0.3 timetable generator (docs/network-v0.3.md §4.1), its utilisation target and its feasibility check.

A service group is one clockface line: a train type, the portals it enters and leaves by, and the platforms it
calls at. The generator repeats each group across the shift, fills in passenger counts, adds freight paths until
the busiest resource sits in the utilisation band, links the connections passengers rely on, and then publishes
the times its own undisturbed run of the shift produced.

Two clocks, on purpose. The provisional times `_trip` works out from `leg_minutes` are the planner's *intent*:
they are what "settled in timetable order" means, because both `plan_run` and the engine's own ARS tie-break
order trains by their booked departure. The times the run produces are what gets published, because only those
are deliverable. `leg_minutes` therefore has to agree with the engine's clock, and
`test_leg_minutes_matches_what_the_engine_really_takes` is what keeps the two from drifting apart.

Rotations are not a separate object: a unit that turns at a terminal and runs back is one trip whose middle call
has a turnaround dwell, so its late arrival carries straight into the return service.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from math import ceil

from .engine import LOOKAHEAD_MIN, TURNAROUND_MIN, Engine, Movement, block_minutes, move_minutes, track_minutes
from .interlocking import Interlocking
from .model import Call, Connection, Move, Network, Platform, ServiceGroup, Timetable, Track, TrainSpec, Trip
from .routing import find_route, portal_track
from .trains import TYPES

LOW, HIGH = 0.75, 0.90  # §1.2: the busiest resource is filled to this share of the shift
MARGIN_MIN = 2  # approach locking before a movement and sectional release after it: the gap two trains need
MIN_TRANSFER_MIN = 5  # §4.1: passengers need this long to change trains
MAX_WAIT_MIN = 30  # a later train than this is a different journey, not a connection
CUSHION_MIN = 15  # the longest a congested run may overrun its provisional times before it is booked too late
REBOOK_PASSES = 4  # publishing new times can reorder the next run; in practice one more pass settles it
MAX_DECISIONS_PER_TRAIN = 40  # a trip crosses about 15 sections, each able to raise a few conflicts
CHECKPOINT_MIN = 15  # snapshot interval so late filler/drop trials resume near the change instead of t=0

LegsKey = tuple[str | None, str | None, tuple[tuple[str, int], ...], int, bool, bool]
OccKey = tuple[str | None, str | None, tuple[tuple[str, int], ...], int, bool, bool, int]
_MISSING = object()
_STD_ENG_KEYS = frozenset({"net", "il", "tick", "shift_min", "orders", "stranded", "reported_blocked", "trains"})
_STD_MV_KEYS = frozenset(Movement.__dataclass_fields__.keys())


@dataclass(frozen=True)
class _TripOcc:
    all_busy: tuple[tuple[str, int], ...]
    running: tuple[tuple[str, int], ...]
    tail: int
    min_from_entry: int
    min_from_call_arrive: tuple[int, ...]  # minimum minutes from arriving at call j to phase == "done"
    min_after_call_depart: tuple[int, ...]  # minimum minutes from departing call j to phase == "done"


@dataclass
class _Checkpoint:
    tick: int
    trips_by_id: dict[str, Trip]
    engine: Engine


@dataclass
class _NetTimetableCache:
    tracks: tuple[Track, ...]
    platforms: tuple[Platform, ...]
    moves: tuple[Move, ...]
    crossings: tuple[tuple[str, str], ...]
    legs: dict[LegsKey, tuple[tuple[str, ...], ...] | None] = field(default_factory=dict)
    occ: dict[OccKey, _TripOcc | None] = field(default_factory=dict)
    can_patch: bool | None = None
    fast_clone: bool | None = None


def _tt_cache(net: Network) -> _NetTimetableCache:
    priv = net.__pydantic_private__
    cache: _NetTimetableCache | None = priv.get("_tcache")
    if (
        cache is not None
        and cache.tracks is net.tracks
        and cache.platforms is net.platforms
        and cache.moves is net.moves
        and cache.crossings is net.crossings
    ):
        return cache
    cache = _NetTimetableCache(tracks=net.tracks, platforms=net.platforms, moves=net.moves, crossings=net.crossings)
    priv["_tcache"] = cache
    return cache


# ---------------------------------------------------------------------- run times


def leg_minutes(net: Network, path: list[str], train: TrainSpec) -> int:
    """How long one leg takes, counted exactly as the engine runs it (railway/net/engine.py `_enter`/`_step`)."""
    first = net.element(path[0])
    total = track_minutes(first, train) if isinstance(first, Track) else 1  # a minute to pull out of a platform
    for i, element_id in enumerate(path[1:], start=1):
        element = net.element(element_id)
        if isinstance(element, Track):
            total += track_minutes(element, train)
        elif isinstance(element, Move):
            total += move_minutes(element, train)
        elif i < len(path) - 1:
            total += 1  # running through a platform it does not call at
    return total


def _walk(
    net: Network, train: TrainSpec, start: str, facing: str, targets: list[str], dwells: list[int]
) -> list[list[str]] | None:
    """Route each leg in turn, carrying the platform side forward so a turnaround is the only way back out.

    The single place that decides where a trip goes. `dwells[i]` is the dwell booked at `targets[i]`, which is
    what decides whether the train may leave it the way it came in; this mirrors `Engine._side_in`.
    """
    side_in, out = None, []
    for target, dwell in zip(targets, dwells, strict=True):
        path = find_route(net, start, facing, target, train, side_in)
        if path is None:
            return None
        out.append(path)
        element = net.element(target)
        side_in = None if dwell >= TURNAROUND_MIN else net.element(path[-2]).side
        start = target
        facing = element.node if isinstance(element, Platform) else element.other(facing)
    return out


def _exit_track(net: Network, named: str, exit_node: str) -> str:
    track = portal_track(net, exit_node, inbound=False)
    if track is None:
        raise ValueError(f"{named}: no track leads out of the network to {exit_node}")
    return track


def _entry_track(net: Network, named: str, entry_node: str) -> str:
    track = portal_track(net, entry_node, inbound=True)
    if track is None:
        raise ValueError(f"{named}: no track leads into the network from {entry_node}")
    return track


def legs(net: Network, trip: Trip) -> list[list[str]] | None:
    """Every leg of a trip in order: entry to the first call, call to call, last call out to the exit portal.

    Cached by route shape rather than booked times: a trip's paths never change when only its times do, and the
    generator measures the same trips over and over as it tries filler paths.
    """
    cache = _tt_cache(net)
    lkey: LegsKey = (
        trip.entry_node,
        trip.exit_node,
        tuple((c.platform, c.min_dwell) for c in trip.calls),
        trip.train.length_m,
        trip.train.electric,
        trip.train.passenger,
    )
    if lkey in cache.legs:
        hit = cache.legs[lkey]
        return [list(p) for p in hit] if hit is not None else None
    if trip.entry_node:
        start = _entry_track(net, trip.train.id, trip.entry_node)
        facing, routed = net.element(start).other(trip.entry_node), list(trip.calls)
    else:
        start = trip.calls[0].platform
        if not isinstance(net.element(start), Platform):
            raise ValueError(f"{trip.train.id}: {start} is not a platform it can start from")
        facing, routed = net.element(start).node, list(trip.calls[1:])
    targets = [call.platform for call in routed]
    dwells = [call.min_dwell for call in routed]
    if trip.exit_node:
        targets.append(_exit_track(net, trip.train.id, trip.exit_node))
        dwells.append(0)
    walked = _walk(net, trip.train, start, facing, targets, dwells)
    cache.legs[lkey] = tuple(tuple(p) for p in walked) if walked is not None else None
    return walked


# ---------------------------------------------------------------------- utilisation


def _trip_occ_or_none(net: Network, trip: Trip) -> _TripOcc | None:
    """Cached per-trip element occupancy (excluding end-of-shift platform parking) and minimum remaining run times."""
    cache = _tt_cache(net)
    okey: OccKey = (
        trip.entry_node,
        trip.exit_node,
        tuple((c.platform, c.min_dwell) for c in trip.calls),
        trip.train.length_m,
        trip.train.electric,
        trip.train.passenger,
        trip.train.max_speed_kmh,
    )
    if okey in cache.occ:
        return cache.occ[okey]
    paths = legs(net, trip)
    if paths is None:
        cache.occ[okey] = None
        return None
    busy: dict[str, int] = {}

    def add(element_id: str, minutes: int) -> None:
        busy[element_id] = busy.get(element_id, 0) + minutes

    called_at = {call.platform for call in trip.calls}
    for path in paths:
        for element_id in path:
            element = net.element(element_id)
            if isinstance(element, Move):
                minutes = move_minutes(element, trip.train) + MARGIN_MIN
                for other in (element_id, *net.conflicts(element_id)):
                    add(other, minutes)
            elif isinstance(element, Track):
                held = (
                    track_minutes(element, trip.train)
                    if element.direction == "both"
                    else max(block_minutes(element, trip.train, i) for i in range(element.blocks))
                )
                add(element_id, held + MARGIN_MIN)
            elif element_id not in called_at:
                add(element_id, 1)  # running through a platform it does not call at
    for call in trip.calls:  # once per call, not once per leg that touches the platform
        add(call.platform, max(1, call.min_dwell))
    all_items = tuple(sorted(busy.items()))
    running_items = tuple((k, v) for k, v in all_items if not isinstance(net.element(k), Platform))
    leg_times = [leg_minutes(net, p, trip.train) for p in paths]
    tail = leg_times[-1] if trip.exit_node else 0
    n_calls = len(trip.calls)
    offset = 1 if trip.entry_node else 0
    after_dep = [0] * n_calls
    from_arr = [0] * n_calls
    for j in range(n_calls - 1, -1, -1):
        terminates = j == n_calls - 1 and trip.exit_node is None
        if terminates:
            after_dep[j] = 0
            from_arr[j] = 0
        else:
            nxt_leg = leg_times[j + offset]
            nxt_rest = from_arr[j + 1] if j + 1 < n_calls else 0
            after_dep[j] = nxt_leg + nxt_rest
            from_arr[j] = trip.calls[j].min_dwell + after_dep[j]
    min_entry = (leg_times[0] + from_arr[0]) if trip.entry_node else after_dep[0]
    occ = _TripOcc(
        all_busy=all_items,
        running=running_items,
        tail=tail,
        min_from_entry=min_entry,
        min_from_call_arrive=tuple(from_arr),
        min_after_call_depart=tuple(after_dep),
    )
    cache.occ[okey] = occ
    return occ


def _trip_occ(net: Network, trip: Trip) -> _TripOcc:
    occ = _trip_occ_or_none(net, trip)
    if occ is None:
        raise ValueError(f"{trip.train.id} cannot be routed, so how much of the network it uses is unknown")
    return occ


def utilisation(net: Network, trips: tuple[Trip, ...], shift_min: int) -> dict[str, float]:
    """The share of the shift each element is busy: §1.2's measure of how full the network is.

    A move is busy for its own clearing time and for every conflicting move's, because only one of them can be
    set at a time. A one-way track is busy for its critical block, the one that sets the following headway; a
    bidirectional track is busy for the whole transit, because the first train to take it locks out the other
    direction. Both carry MARGIN_MIN on top, the real gap between two movements rather than the physical
    occupancy. A platform is busy for the dwell booked there, and for the whole rest of the shift when a train
    terminates and stays put.
    """
    busy: dict[str, int] = {}
    for trip in trips:
        occ = _trip_occ(net, trip)
        for element_id, minutes in occ.all_busy:
            busy[element_id] = busy.get(element_id, 0) + minutes
        last = trip.calls[-1]
        if trip.exit_node is None:  # it terminates here and holds the platform for what is left of the shift
            busy[last.platform] = busy.get(last.platform, 0) + max(0, shift_min - (last.arrive or 0))
    return {element_id: minutes / shift_min for element_id, minutes in sorted(busy.items())}


def peak_utilisation(net: Network, trips: tuple[Trip, ...], shift_min: int) -> float:
    """The busiest junction or line. Platforms are left out: a train parked in a siding pins one at 100% without
    costing the network any capacity, and §1.2's target is about the bottlenecks trains have to run through."""
    busy: dict[str, int] = {}
    for trip in trips:
        for element_id, minutes in _trip_occ(net, trip).running:
            busy[element_id] = busy.get(element_id, 0) + minutes
    return max(busy.values(), default=0) / shift_min


# ---------------------------------------------------------------------- building trips


def _peak(shift_min: int) -> tuple[int, int]:
    """The busy hour: the middle third of the shift carries the fullest trains and the longest dwells (§4.1)."""
    return shift_min // 3, 2 * shift_min // 3


def _loads(group: ServiceGroup, stops: int, in_peak: bool, rng: random.Random) -> list[tuple[int, int]]:
    """Passengers alighting and boarding at each stop. Everyone boards at the origin and alights at the end.

    Counts are rounded to tens, as published loadings are. The peak half of a type's range stands for the busy
    hour (§3 gives a commuter 300 off-peak and 800 in the peak). A fifth to a third of the train turns over at an
    intermediate stop, so the load stays roughly flat along the journey; a service that crosses the area with a
    single call turns over more, because that one call is doing the work of an origin and a destination.
    """
    low, high = TYPES[group.type].load
    if high == 0:
        return [(0, 0)] * stops
    total = rng.randrange(high - (high - low) // 2, high + 1, 10) if in_peak else rng.randrange(low, high + 1, 10)
    if stops == 1:
        turnover = round(total * rng.uniform(0.3, 0.6))
        return [(turnover, turnover)]
    out = [(0, total)]
    for _ in range(stops - 2):
        turnover = round(total * rng.uniform(0.15, 0.35))
        out.append((turnover, turnover))
    return [*out, (total, 0)]


def _dwells(group: ServiceGroup, in_peak: bool) -> list[int]:
    """The dwell booked at each stop. The peak adds a minute, but never enough to turn a stop into a turnaround."""
    out = []
    for i, dwell in enumerate(group.dwell):
        terminates = i == len(group.dwell) - 1 and group.exit_node is None
        bump = 1 if in_peak and not terminates and dwell + 1 < TURNAROUND_MIN else 0
        out.append(dwell + bump)
    return out


def _trip(net: Network, group: ServiceGroup, train_id: str, entry: int, shift_min: int, rng: random.Random) -> Trip:
    """Time every call of one service from the run times the engine will produce, plus the group's slack."""
    if group.type not in TYPES:
        raise ValueError(f"group {group.code}: no train type {group.type}")
    spec = TYPES[group.type].spec(train_id)
    peak_from, peak_to = _peak(shift_min)
    in_peak = peak_from <= entry <= peak_to
    dwells = _dwells(group, in_peak)
    loads = _loads(group, len(group.stops), in_peak, rng)
    start = _entry_track(net, f"group {group.code}", group.entry_node)
    targets, walked = list(group.stops), list(dwells)
    if group.exit_node:
        targets.append(_exit_track(net, f"group {group.code}", group.exit_node))
        walked.append(dwells[-1])  # the last call's dwell decides whether it may reverse out of the platform
    paths = _walk(net, spec, start, net.element(start).other(group.entry_node), targets, walked)
    if paths is None:
        raise ValueError(f"group {group.code}: {train_id} cannot run {' -> '.join(targets)}")
    clock, calls = entry, []
    for i, (platform, dwell) in enumerate(zip(group.stops, dwells, strict=True)):
        run = leg_minutes(net, paths[i], spec)
        clock += run + ceil(run * group.slack_pct / 100)
        terminates = i == len(group.stops) - 1 and group.exit_node is None
        alight, board = loads[i]
        calls.append(
            Call(
                platform=platform,
                arrive=clock,
                depart=None if terminates else clock + dwell,
                min_dwell=dwell,
                alight=alight,
                board=board,
            )
        )
        clock += 0 if terminates else dwell
    return Trip(train=spec, calls=tuple(calls), entry_node=group.entry_node, exit_node=group.exit_node,
                entry_tick=entry)  # fmt: skip


def _finishes_by(net: Network, trip: Trip, limit: int) -> bool:
    """Whether the whole run, including the leg out to the exit portal, is over by `limit`."""
    occ = _trip_occ_or_none(net, trip)
    if occ is None:
        return False
    return (trip.calls[-1].arrive or 0) + occ.tail <= limit


def _services(net: Network, group: ServiceGroup, shift_min: int, rng: random.Random) -> list[Trip]:
    """One clockface line across the shift. The seed moves the whole line's phase, keeping the interval even."""
    out = []
    offset = rng.randrange(group.every_min)
    for i in range(1 + shift_min // group.every_min):
        entry = group.first_min + offset + i * group.every_min
        if entry >= shift_min:
            break
        trip = _trip(net, group, f"{group.code}{i + 1:02d}", entry, shift_min, rng)
        if _finishes_by(net, trip, shift_min - CUSHION_MIN):
            out.append(trip)
    return out


def _shift_times(trip: Trip, by: int) -> Trip:
    calls = tuple(
        call.model_copy(
            update={
                "arrive": None if call.arrive is None else call.arrive + by,
                "depart": None if call.depart is None else call.depart + by,
            }
        )
        for call in trip.calls
    )
    return trip.model_copy(update={"calls": calls, "entry_tick": trip.entry_tick + by})


def _space_entries(net: Network, trips: list[Trip], shift_min: int) -> list[Trip]:
    """Push entries apart so no train waits outside: one entry block plus the signalling margin, as on the line."""
    clear: dict[str, int] = {}
    out = []
    for trip in sorted(trips, key=lambda t: (t.entry_tick, t.train.id)):
        track = net.element(_entry_track(net, trip.train.id, trip.entry_node))
        headway = block_minutes(track, trip.train, 0) + MARGIN_MIN
        entry = max(trip.entry_tick, clear.get(trip.entry_node, 0))
        clear[trip.entry_node] = entry + headway
        if entry != trip.entry_tick:
            trip = _shift_times(trip, entry - trip.entry_tick)
        if _finishes_by(net, trip, shift_min - CUSHION_MIN):
            out.append(trip)
    return out


# ---------------------------------------------------------------------- connections


def _connections(net: Network, trips: tuple[Trip, ...], lines: dict[str, str]) -> tuple[Connection, ...]:
    """Link each passenger arrival to the next passenger departure from the same node by a different line.

    `transfers` is a tenth of the feeder's load: the people whose journey breaks if the connector leaves first.
    """
    by_node: dict[str, list[tuple[Call, Trip]]] = {}
    for trip in trips:
        if not trip.train.passenger:
            continue
        for call in trip.calls:
            node = net.element(call.platform).node
            by_node.setdefault(node, []).append((call, trip))
    for stops in by_node.values():
        stops.sort(key=lambda s: ((s[0].depart or 0), s[1].train.id))
    out = []
    for node in sorted(by_node):
        stops = by_node[node]
        for call, trip in sorted(stops, key=lambda s: (s[0].arrive or 0, s[1].train.id)):
            if call.arrive is None or not call.alight:
                continue
            onward = next(
                (
                    other
                    for candidate, other in stops
                    if other.train.id != trip.train.id
                    and lines[other.train.id] != lines[trip.train.id]
                    and candidate.board
                    and MIN_TRANSFER_MIN <= (candidate.depart or 0) - call.arrive <= MAX_WAIT_MIN
                ),
                None,
            )
            if onward is not None:
                out.append(
                    Connection(
                        feeder=trip.train.id,
                        connector=onward.train.id,
                        node=node,
                        transfers=max(1, round(call.alight / 10)),
                        min_transfer_min=MIN_TRANSFER_MIN,
                    )
                )
    return tuple(out)


# ---------------------------------------------------------------------- the planner's run


def _clone_mv_fast(mv: Movement) -> Movement:
    return Movement(
        trip=mv.trip,
        plan=list(mv.plan),
        position=mv.position,
        behind=mv.behind,
        phase=mv.phase,
        side_in=mv.side_in,
        path=list(mv.path),
        idx=mv.idx,
        authority=mv.authority,
        block=mv.block,
        next_tick=mv.next_tick,
        waiting=mv.waiting,
        held_until=mv.held_until,
        arrived=mv.arrived,
        at=mv.at,
        worst_late=mv.worst_late,
        visits=list(mv.visits),
    )


def _clone_engine(eng: Engine) -> Engine:
    cache = _tt_cache(eng.net)
    if cache.fast_clone is None:
        sample_mv = next(iter(eng.trains.values()), None)
        cache.fast_clone = (
            set(vars(eng)) == _STD_ENG_KEYS
            and set(vars(eng.il)) == {"net", "holder", "directions"}
            and (sample_mv is None or set(vars(sample_mv)) == _STD_MV_KEYS)
        )
    if cache.fast_clone:
        il = Interlocking.__new__(Interlocking)
        il.net = eng.net
        il.holder = dict(eng.il.holder)
        il.directions = {k: dict(v) for k, v in eng.il.directions.items()}
        out = Engine.__new__(Engine)
        out.net = eng.net
        out.il = il
        out.tick = eng.tick
        out.shift_min = eng.shift_min
        out.orders = {k: list(v) for k, v in eng.orders.items()}
        out.stranded = set(eng.stranded)
        out.reported_blocked = eng.reported_blocked
        out.trains = {tid: _clone_mv_fast(mv) for tid, mv in eng.trains.items()}
        return out
    memo: dict[int, object] = {id(eng.net): eng.net}
    for mv in eng.trains.values():
        memo[id(mv.trip)] = mv.trip
        for call in mv.trip.calls:
            memo[id(call)] = call
        for call in mv.plan:
            memo[id(call)] = call
        if mv.at is not None:
            memo[id(mv.at)] = mv.at
    return copy.deepcopy(eng, memo)


def _can_patch_offstage(net: Network, trips: list[Trip], shift_min: int) -> bool:
    """Verify once per network that Engine stores per-train state only in `trains` for offstage portal trips."""
    cache = _tt_cache(net)
    if cache.can_patch is not None:
        return cache.can_patch
    portal_trips = [t for t in trips if t.entry_node is not None][:2]
    if len(portal_trips) < 2:
        return False
    e1 = Engine(net, portal_trips[:1], shift_min)
    e2 = Engine(net, portal_trips, shift_min)
    ok = True
    for k, v2 in vars(e2).items():
        if k == "trains":
            continue
        v1 = getattr(e1, k, _MISSING)
        if k == "il":
            if not (v1.holder == v2.holder == {} and v1.directions == v2.directions == {}):
                ok = False
                break
        elif k == "log":
            if getattr(v1, "events", None) != [] or getattr(v2, "events", None) != []:
                ok = False
                break
        elif v1 != v2:
            ok = False
            break
    cache.can_patch = ok
    return ok


def _stuck_at_signal(net: Network, mv: Movement) -> bool:
    if mv.phase == "berthed":
        return mv.working
    if mv.phase == "running" and mv.waiting and mv.authority <= mv.idx:
        el = net.element(mv.position)
        if isinstance(el, Track):
            if mv.block < el.blocks - 1 or mv.idx == len(mv.path) - 1:
                return False
        return True
    return False


def _next_target_edge(net: Network, mv: Movement) -> tuple[str, str] | None:
    """Return `(from_node, next_track_or_platform_id)` that `mv` needs next, or None."""
    if mv.phase == "running":
        seg = mv.path[mv.idx : mv.idx + 3]
        if len(seg) < 2:
            return None
        el = net.element(mv.position)
        node = el.other(mv.behind) if isinstance(el, Track) else mv.behind
        return node, seg[-1]
    paths = legs(net, mv.trip)
    if not paths:
        return None
    leg_idx = (len(mv.visits) - 1) + (1 if mv.trip.entry_node else 0)
    if leg_idx >= len(paths):
        return None
    seg = paths[leg_idx][:3]
    return (mv.behind, seg[-1]) if len(seg) >= 2 else None


def _cannot_finish(net: Network, engine: Engine, shift_min: int) -> bool:
    """True when at least one train already cannot physically reach `done` by `shift_min`."""
    now = engine.tick
    stopped: dict[str, Movement] = {}
    for tid, mv in engine.trains.items():
        if mv.phase == "done":
            continue
        occ = _trip_occ_or_none(net, mv.trip)
        if occ is None:
            return True
        if mv.phase == "offstage":
            earliest = max(now, mv.ready_at() or now) + occ.min_from_entry
        elif mv.phase == "berthed":
            j = len(mv.visits) - 1
            earliest = max(now, mv.ready_at() or now) + occ.min_after_call_depart[j]
        else:  # running
            rem = len(mv.plan)
            after = occ.min_from_call_arrive[len(mv.trip.calls) - rem] if rem else 0
            earliest = (
                mv.next_tick if mv.next_tick is not None else now + (mv.train.restart_min if mv.waiting else 0)
            ) + after
        if earliest > shift_min:
            return True
        if _stuck_at_signal(net, mv):
            stopped[tid] = mv
    if len(stopped) >= 2:
        holder = engine.il.holder
        directions = engine.il.directions
        wait_for: dict[str, set[str]] = {}
        for tid, mv in stopped.items():
            nxt = _next_target_edge(net, mv)
            if nxt is None:
                continue
            node, target_id = nxt
            target = net.element(target_id)
            deps: set[str] = set()
            if isinstance(target, Platform):
                if (h := holder.get(target_id)) in stopped and h != tid:
                    deps.add(h)
            elif isinstance(target, Track):
                travel = target.travel_from(node)
                entry_blk = 0 if travel == "ab" else target.blocks - 1
                if (h := holder.get((target_id, entry_blk))) in stopped and h != tid:
                    deps.add(h)
                for h, d in directions.get(target_id, {}).items():
                    if d != travel and h in stopped and h != tid:
                        deps.add(h)
            if deps:
                wait_for[tid] = deps
        state: dict[str, int] = {}  # 0/missing=unvisited, 1=visiting, 2=done
        for start in wait_for:
            if state.get(start):
                continue
            stack: list[tuple[str, list[str], int]] = [(start, list(wait_for.get(start, ())), 0)]
            state[start] = 1
            while stack:
                u, nbrs, idx = stack[-1]
                if idx >= len(nbrs):
                    state[u] = 2
                    stack.pop()
                    continue
                stack[-1] = (u, nbrs, idx + 1)
                v = nbrs[idx]
                st = state.get(v, 0)
                if st == 1:
                    return True
                if st == 0 and v in wait_for:
                    state[v] = 1
                    stack.append((v, list(wait_for[v]), 0))
    return False


def _run_with_checkpoints(
    net: Network,
    trips: list[Trip],
    shift_min: int,
    checkpoints: list[_Checkpoint],
    *,
    abort_if_impossible: bool = False,
) -> tuple[Engine | None, list[_Checkpoint]]:
    """Run `trips` to SHIFT_END, resuming from the latest unaffected checkpoint when safe."""
    if not trips or not _can_patch_offstage(net, trips, shift_min):
        return plan_run(net, trips, shift_min), []
    target_by_id = {t.train.id: t for t in trips}
    engine: Engine | None = None
    kept: list[_Checkpoint] = []
    for cp in reversed(checkpoints):
        changed = [
            tid
            for tid in (cp.trips_by_id.keys() | target_by_id.keys())
            if cp.trips_by_id.get(tid) != target_by_id.get(tid)
        ]
        if any(
            (old := cp.trips_by_id.get(tid)) is not None
            and (
                old.entry_node is None
                or old.entry_tick <= cp.tick + LOOKAHEAD_MIN
                or cp.engine.trains[tid].phase != "offstage"
            )
            or (new := target_by_id.get(tid)) is not None
            and (new.entry_node is None or new.entry_tick <= cp.tick + LOOKAHEAD_MIN)
            for tid in changed
        ):
            continue
        engine = _clone_engine(cp.engine)
        if changed:
            to_add = [target_by_id[tid] for tid in changed if tid in target_by_id]
            added = Engine(net, to_add, shift_min).trains if to_add else {}
            merged = {tid: mv for tid, mv in engine.trains.items() if tid not in changed}
            merged.update(added)
            engine.trains = {tid: merged[tid] for tid in sorted(merged)}
        kept = [c for c in checkpoints if c.tick <= cp.tick]
        break
    if engine is None:
        engine = Engine(net, trips, shift_min)
        kept = []
    next_snap = ((kept[-1].tick // CHECKPOINT_MIN) + 1) * CHECKPOINT_MIN if kept else CHECKPOINT_MIN
    for _ in range(MAX_DECISIONS_PER_TRAIN * len(trips) + 100):
        decision = engine.advance()
        if decision.kind == "SHIFT_END":
            return engine, kept
        if abort_if_impossible and _cannot_finish(net, engine, shift_min):
            return None, []
        for conflict in decision.conflicts:
            engine.order(conflict.key, sorted(conflict.trains, key=lambda t: engine.trains[t].booked()))
        if decision.tick >= next_snap and decision.tick < shift_min:
            kept.append(_Checkpoint(decision.tick, target_by_id, _clone_engine(engine)))
            next_snap = ((decision.tick // CHECKPOINT_MIN) + 1) * CHECKPOINT_MIN
    raise RuntimeError(f"the engine did not reach the end of a {shift_min} min shift for {len(trips)} trains")


def plan_run(net: Network, trips: list[Trip], shift_min: int) -> Engine:
    """Run the shift with no disruption and no agent judgement: every conflict is settled in timetable order.

    This is the planner's own run. Its result is what the generator books, so the published timetable is one the
    network can really deliver, and so an undisrupted shift costs nothing (DESIGN.md §6.6).
    """
    engine = Engine(net, trips, shift_min)
    for _ in range(MAX_DECISIONS_PER_TRAIN * len(trips) + 100):
        decision = engine.advance()
        if decision.kind == "SHIFT_END":
            return engine
        for conflict in decision.conflicts:
            engine.order(conflict.key, sorted(conflict.trains, key=lambda t: engine.trains[t].booked()))
    raise RuntimeError(f"the engine did not reach the end of a {shift_min} min shift for {len(trips)} trains")


def off_book(engine: Engine) -> list[str]:
    """Trains the planner's run could not deliver as booked: the feasibility test, in one place."""
    reasons = []
    for train_id, mv in sorted(engine.trains.items()):
        if mv.phase != "done":
            reasons.append(f"{train_id} is still {mv.phase} at {mv.position} when the shift ends")
        elif mv.worst_late > 0:
            reasons.append(f"{train_id} runs {mv.worst_late} min late with nothing in its way")
    return reasons


def infeasible(timetable: Timetable) -> list[str]:
    """Why this timetable will not run as booked with no help from the agent; empty means it is feasible (§4.1)."""
    return off_book(plan_run(timetable.net, list(timetable.trips), timetable.shift_min))


# ---------------------------------------------------------------------- the generator


def _rebook(trips: list[Trip], engine: Engine) -> list[Trip]:
    """Publish the times the planner's run actually produced, keeping each call's dwell and passenger counts."""
    out = []
    for trip in sorted(trips, key=lambda t: t.train.id):
        visits = engine.trains[trip.train.id].visits
        if len(visits) != len(trip.calls):
            raise RuntimeError(f"{trip.train.id} made {len(visits)} of {len(trip.calls)} calls: nothing to book")
        calls = tuple(
            call.model_copy(
                update={
                    "arrive": None if call.arrive is None else visit.arrived,
                    "depart": None if call.depart is None else max(visit.departed or 0, visit.arrived),
                }
            )
            for call, visit in zip(trip.calls, visits, strict=True)
        )
        out.append(trip.model_copy(update={"calls": calls}))
    return out


def _all_services(
    net: Network, groups: list[ServiceGroup], shift_min: int, rng: random.Random, lines: dict[str, str]
) -> list[Trip]:
    """Every service of every group, recording which line each train belongs to so connections can tell them apart."""
    out = []
    for group in groups:
        for trip in _services(net, group, shift_min, rng):
            lines[trip.train.id] = group.code
            out.append(trip)
    return out


def _delivers(net: Network, trips: list[Trip], shift_min: int) -> Engine | None:
    """The planner's run, or None if it cannot get every train home inside the shift."""
    engine = plan_run(net, trips, shift_min)
    return None if any(mv.phase != "done" for mv in engine.trains.values()) else engine


def _drop_until_delivered(
    net: Network, trips: list[Trip], shift_min: int, checkpoints: list[_Checkpoint] | None = None
) -> tuple[list[Trip], Engine, list[_Checkpoint]] | None:
    """Give up services, latest booked path first, until the run gets every remaining train home."""
    snaps = list(checkpoints) if checkpoints else []
    for _ in range(len(trips) + 1):
        engine, snaps = _run_with_checkpoints(net, trips, shift_min, snaps)
        assert engine is not None
        stuck = [mv for mv in engine.trains.values() if mv.phase != "done"]
        if not stuck:
            return trips, engine, snaps
        latest = max(stuck, key=lambda m: (m.trip.entry_tick, m.trip.train.id)).trip.train.id
        trips = [t for t in trips if t.train.id != latest]
    return None


def generate(
    net: Network, core: list[ServiceGroup], fill: list[ServiceGroup], shift_min: int, seed: int, tier: str
) -> Timetable:
    """Build the shift: the core service, then filler paths until the busiest junction is in the band (§1.2).

    Freight is the traffic a real timetable fits into the gaps, so it is what the generator adds to reach the
    utilisation target and the first thing it gives up when the result cannot be run. Booked times come from the
    planner's own run, repeated until re-running the published book changes nothing, because publishing new
    departures can reorder the next run.
    """
    rng = random.Random(f"{seed}:{tier}")
    lines: dict[str, str] = {}
    spaced = _space_entries(net, _all_services(net, core, shift_min, rng, lines), shift_min)
    delivered = _drop_until_delivered(net, spaced, shift_min)
    if delivered is None:
        raise ValueError(f"seed {seed} rejected: no part of the core service can be run to completion")
    trips, engine, snaps = delivered
    running_busy: dict[str, int] = {}
    for trip in trips:
        for el_id, mins in _trip_occ(net, trip).running:
            running_busy[el_id] = running_busy.get(el_id, 0) + mins
    running_peak = max(running_busy.values(), default=0)
    # Filler paths are offered in the order they would run; the seed has already moved each line's phase,
    # so which path gets the last gap varies with it.
    paths = _all_services(net, fill, shift_min, rng, lines)
    for path in sorted(paths, key=lambda t: (t.entry_tick, t.train.id)):
        if running_peak / shift_min >= LOW:
            break
        candidate = _space_entries(net, [*trips, path], shift_min)
        if len(candidate) != len(trips) + 1:
            continue  # re-spacing lost a service: this path costs more than it adds
        path_running = _trip_occ(net, path).running
        cand_peak = max(
            running_peak,
            max((running_busy.get(el_id, 0) + mins for el_id, mins in path_running), default=0),
        )
        if cand_peak / shift_min > HIGH:
            continue
        cand_engine, cand_snaps = _run_with_checkpoints(net, candidate, shift_min, snaps, abort_if_impossible=True)
        if cand_engine is not None and all(mv.phase == "done" for mv in cand_engine.trains.values()):
            trips, engine, snaps = candidate, cand_engine, cand_snaps
            for el_id, mins in path_running:
                running_busy[el_id] = running_busy.get(el_id, 0) + mins
            running_peak = cand_peak
    for _ in range(REBOOK_PASSES):
        # Later times can push a service past the end of the shift, so every pass re-proves delivery as well.
        delivered = _drop_until_delivered(net, _rebook(trips, engine), shift_min, snaps)
        if delivered is None:
            raise ValueError(f"seed {seed} rejected: the service cannot be booked to times it can keep")
        trips, engine, snaps = delivered
        if not off_book(engine):
            break
    else:
        raise ValueError(f"seed {seed} rejected: {off_book(engine)[0]}")
    booked = tuple(trips)
    return Timetable(
        net=net,
        trips=booked,
        connections=_connections(net, booked, lines),
        shift_min=shift_min,
        seed=seed,
        tier=tier,
    )
