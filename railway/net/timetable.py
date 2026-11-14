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

import random
from functools import lru_cache
from math import ceil

from .engine import TURNAROUND_MIN, Engine, block_minutes, move_minutes, track_minutes
from .model import (
    Call,
    Connection,
    Disruption,
    DisruptionKind,
    Move,
    Network,
    Platform,
    ServiceGroup,
    Timetable,
    Track,
    TrainSpec,
    Trip,
)
from .routing import find_route, portal_track
from .trains import TYPES

LOW, HIGH = 0.75, 0.90  # §1.2: the busiest resource is filled to this share of the shift
MARGIN_MIN = 2  # approach locking before a movement and sectional release after it: the gap two trains need
MIN_TRANSFER_MIN = 5  # §4.1: passengers need this long to change trains
MAX_WAIT_MIN = 30  # a later train than this is a different journey, not a connection
CUSHION_MIN = 15  # the longest a congested run may overrun its provisional times before it is booked too late
REBOOK_PASSES = 4  # publishing new times can reorder the next run; in practice one more pass settles it
MAX_DECISIONS_PER_TRAIN = 40  # a trip crosses about 15 sections, each able to raise a few conflicts


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


@lru_cache(maxsize=4096)
def legs(net: Network, trip: Trip) -> list[list[str]] | None:
    """Every leg of a trip in order: entry to the first call, call to call, last call out to the exit portal.

    Cached: a trip's paths never change when only its times do, and the generator measures the same trips over
    and over as it tries filler paths.
    """
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
    return _walk(net, trip.train, start, facing, targets, dwells)


# ---------------------------------------------------------------------- utilisation


def utilisation(net: Network, trips: tuple[Trip, ...], shift_min: int) -> dict[str, float]:
    """The share of the shift each element is busy: §1.2's measure of how full the network is.

    A move is busy for its own clearing time and for every conflicting move's, because only one of them can be
    set at a time. A one-way track is busy for its critical block, the one that sets the following headway; a
    bidirectional track is busy for the whole transit, because the first train to take it locks out the other
    direction. Both carry MARGIN_MIN on top, the real gap between two movements rather than the physical
    occupancy. A platform is busy for the dwell booked there, and for the whole rest of the shift when a train
    terminates and stays put.
    """
    busy: dict[str, float] = {}

    def add(element_id: str, minutes: float) -> None:
        busy[element_id] = busy.get(element_id, 0.0) + minutes

    for trip in trips:
        paths = legs(net, trip)
        if paths is None:
            raise ValueError(f"{trip.train.id} cannot be routed, so how much of the network it uses is unknown")
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
        last = trip.calls[-1]
        if trip.exit_node is None:  # it terminates here and holds the platform for what is left of the shift
            add(last.platform, max(0, shift_min - (last.arrive or 0)))
    return {element_id: minutes / shift_min for element_id, minutes in sorted(busy.items())}


def peak_utilisation(net: Network, trips: tuple[Trip, ...], shift_min: int) -> float:
    """The busiest junction or line. Platforms are left out: a train parked in a siding pins one at 100% without
    costing the network any capacity, and §1.2's target is about the bottlenecks trains have to run through."""
    loads = utilisation(net, trips, shift_min)
    running = [v for k, v in loads.items() if not isinstance(net.element(k), Platform)]
    return max(running, default=0.0)


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
    paths = legs(net, trip)
    if paths is None:
        return False
    tail = leg_minutes(net, paths[-1], trip.train) if trip.exit_node else 0
    return (trip.calls[-1].arrive or 0) + tail <= limit


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
    stops = sorted(
        (
            (net.element(call.platform).node, call, trip)
            for trip in trips
            if trip.train.passenger
            for call in trip.calls
        ),
        key=lambda s: ((s[1].depart or 0), s[2].train.id),
    )
    out = []
    for node, call, trip in sorted(stops, key=lambda s: (s[0], s[1].arrive or 0, s[2].train.id)):
        if call.arrive is None or not call.alight:
            continue
        onward = next(
            (
                other
                for other_node, candidate, other in stops
                if other_node == node
                and other.train.id != trip.train.id
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


def _drop_until_delivered(net: Network, trips: list[Trip], shift_min: int) -> tuple[list[Trip], Engine] | None:
    """Give up services, latest booked path first, until the run gets every remaining train home."""
    for _ in range(len(trips) + 1):
        engine = plan_run(net, trips, shift_min)
        stuck = [mv for mv in engine.trains.values() if mv.phase != "done"]
        if not stuck:
            return trips, engine
        latest = max(stuck, key=lambda m: (m.trip.entry_tick, m.trip.train.id)).trip.train.id
        trips = [t for t in trips if t.train.id != latest]
    return None


def _draw_incident(
    net: Network,
    trips: tuple[Trip, ...],
    shift_min: int,
    rng: random.Random,
    used_breakdowns: set[str],
) -> dict | None:
    """Draw one infrastructure or breakdown disruption inside `shift_min`, or None if nothing fits."""
    if shift_min < 8:
        return None
    kinds: list[DisruptionKind] = []
    live_trips = [
        t
        for t in trips
        if t.train.id not in used_breakdowns
        and (t.calls[-1].arrive or shift_min) > t.entry_tick + 4
        and t.entry_tick + 2 < shift_min - 2
    ]
    if live_trips:
        kinds.append("breakdown")
    if net.tracks:
        kinds += ["possession", "speed_restriction", "signal_failure"]
    if net.moves:
        kinds.append("points_failure")
    if net.platforms:
        kinds.append("platform_closed")
    if not kinds:
        return None

    kind = rng.choice(kinds)
    low = min(15, max(1, shift_min // 6))
    high = max(low, shift_min - max(6, shift_min // 5))
    start = rng.randint(low, high)

    if kind == "breakdown":
        trip = rng.choice(live_trips)
        used_breakdowns.add(trip.train.id)
        hi = min(shift_min - 2, (trip.calls[-1].arrive or shift_min) - 1)
        lo = min(hi, trip.entry_tick + 2)
        start = rng.randint(lo, hi)
        dur = rng.randint(10, 22)
        end = min(shift_min, max(start + 1, start + dur))
        return {"kind": kind, "start": start, "end": end, "reveal": start, "train": trip.train.id}

    if kind == "possession":
        reversible = [t for t in net.tracks if t.reversible]
        track = rng.choice(reversible or list(net.tracks))
        dur = rng.randint(15, 30)
        end = min(shift_min, max(start + 1, start + dur))
        reveal = max(0, start - rng.randint(30, 60))
        return {"kind": kind, "start": start, "end": end, "reveal": reveal, "elements": (track.id,)}

    if kind == "speed_restriction":
        fast = [t for t in net.tracks if t.line_speed_kmh >= 80]
        track = rng.choice(fast or list(net.tracks))
        speed = 40 if track.line_speed_kmh > 60 else max(20, track.line_speed_kmh // 2)
        dur = rng.randint(20, 45)
        end = min(shift_min, max(start + 1, start + dur))
        reveal = max(0, start - rng.randint(30, 60))
        return {
            "kind": kind,
            "start": start,
            "end": end,
            "reveal": reveal,
            "elements": (track.id,),
            "speed_kmh": speed,
        }

    if kind == "signal_failure":
        track = rng.choice(net.tracks)
        dur = rng.randint(15, 35)
        end = min(shift_min, max(start + 1, start + dur))
        return {"kind": kind, "start": start, "end": end, "reveal": start, "elements": (track.id,)}

    if kind == "points_failure":
        move = rng.choice(net.moves)
        dur = rng.randint(12, 28)
        end = min(shift_min, max(start + 1, start + dur))
        return {"kind": kind, "start": start, "end": end, "reveal": start, "elements": (move.id,)}

    by_node: dict[str, list[Platform]] = {}
    for p in net.platforms:
        by_node.setdefault(p.node, []).append(p)
    multi = [p for group in by_node.values() if len(group) >= 2 for p in group]
    plat = rng.choice(multi or list(net.platforms))
    dur = rng.randint(15, 35)
    end = min(shift_min, max(start + 1, start + dur))
    reveal = max(0, start - 30)
    return {"kind": "platform_closed", "start": start, "end": end, "reveal": reveal, "elements": (plat.id,)}


def generate_disruptions(
    timetable: Timetable,
    seed: int | None = None,
    density: float = 1.0,
) -> tuple[Disruption, ...]:
    """Deterministically draw §4.2 disruptions for `timetable` from `(seed, density)`.

    `density <= 0` draws none; `density == 1.0` matches §6's preset density (~3-4 incidents on a 3 h T1 shift plus
    late portal entries), and larger values scale both entry-delay frequency and incident count up monotonically.
    """
    if density < 0:
        raise ValueError(f"disruption density must be >= 0, got {density}")
    if density == 0:
        return ()
    use_seed = timetable.seed if seed is None else seed
    rng = random.Random(f"{use_seed}:{timetable.tier}:disruptions")
    raw: list[dict] = []

    entry_prob = min(0.85, 0.25 * density)
    for trip in sorted(timetable.trips, key=lambda t: t.train.id):
        if trip.entry_node is None or trip.entry_tick + 2 >= timetable.shift_min:
            continue
        roll = rng.random()
        max_delay = min(10, timetable.shift_min - trip.entry_tick - 1)
        delay = rng.randint(2, max(2, max_delay))
        if roll < entry_prob and max_delay >= 2:
            raw.append(
                {
                    "kind": "late_entry",
                    "start": trip.entry_tick,
                    "end": trip.entry_tick + delay,
                    "reveal": max(0, trip.entry_tick - 20),
                    "train": trip.train.id,
                }
            )

    target = max(1, round((timetable.shift_min / 60) * 1.2 * density))
    used_breakdowns: set[str] = set()
    for _ in range(target):
        item = _draw_incident(timetable.net, timetable.trips, timetable.shift_min, rng, used_breakdowns)
        if item is not None:
            raw.append(item)

    raw.sort(key=lambda d: (d["start"], d["kind"], d.get("train") or "", d.get("elements", ())))
    return tuple(Disruption(id=f"D{i + 1:02d}", **d) for i, d in enumerate(raw))


def with_disruptions(
    timetable: Timetable,
    seed: int | None = None,
    density: float = 1.0,
) -> Timetable:
    """Return a validated copy of `timetable` populated with deterministic disruptions."""
    disruptions = generate_disruptions(timetable, seed=seed, density=density)
    return Timetable(
        net=timetable.net,
        trips=timetable.trips,
        connections=timetable.connections,
        shift_min=timetable.shift_min,
        seed=timetable.seed if seed is None else seed,
        tier=timetable.tier,
        disruptions=disruptions,
    )


def generate(
    net: Network,
    core: list[ServiceGroup],
    fill: list[ServiceGroup],
    shift_min: int,
    seed: int,
    tier: str,
    disruption_density: float = 0.0,
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
    trips, engine = delivered
    # Filler paths are offered in the order they would run; the seed has already moved each line's phase,
    # so which path gets the last gap varies with it.
    paths = _all_services(net, fill, shift_min, rng, lines)
    for path in sorted(paths, key=lambda t: (t.entry_tick, t.train.id)):
        if peak_utilisation(net, tuple(trips), shift_min) >= LOW:
            break
        candidate = _space_entries(net, [*trips, path], shift_min)
        if len(candidate) != len(trips) + 1:
            continue  # re-spacing lost a service: this path costs more than it adds
        if peak_utilisation(net, tuple(candidate), shift_min) > HIGH:
            continue
        if run := _delivers(net, candidate, shift_min):
            trips, engine = candidate, run
    for _ in range(REBOOK_PASSES):
        # Later times can push a service past the end of the shift, so every pass re-proves delivery as well.
        delivered = _drop_until_delivered(net, _rebook(trips, engine), shift_min)
        if delivered is None:
            raise ValueError(f"seed {seed} rejected: the service cannot be booked to times it can keep")
        trips, engine = delivered
        if not off_book(engine):
            break
    else:
        raise ValueError(f"seed {seed} rejected: {off_book(engine)[0]}")
    booked = tuple(trips)
    timetable = Timetable(
        net=net,
        trips=booked,
        connections=_connections(net, booked, lines),
        shift_min=shift_min,
        seed=seed,
        tier=tier,
    )
    if disruption_density > 0:
        return with_disruptions(timetable, seed=seed, density=disruption_density)
    if disruption_density < 0:
        raise ValueError(f"disruption density must be >= 0, got {disruption_density}")
    return timetable
