"""The v0.3 engine: trains run under automatic route setting, and the agent decides every conflict (§5).

Time is in minutes and advances only in `advance()`. A train is given authority one section at a time, as real
block signalling works: the moves up to the next track or platform, and on a track only its entry block. It then
runs on block by block, releasing behind it, and stops at a signal when the next section is not free.

ARS extends a train's authority as soon as nothing else wants that section. When two or more trains want the same
section within the next 15 minutes it raises a conflict instead of choosing, and `advance()` returns so the agent
can answer with `order`, `hold` or `platform`, or change the service with `route`, `skip`, `short_turn`,
`cancel`, `single_line` and `relieve_crew`.

Everything that happens is appended to `log` (railway/net/events.py), the one record the cost is computed from.
Disruptions (§4.2) are data in the scenario, applied at their minute and revealed to the agent at theirs; crews
(F10) work at most CREW_LIMIT_MIN minutes. `advance()` also stops when a disruption is revealed or happens, on a
crew warning or timeout, and when a set of trains is gridlocked: none of them can ever move again.

`Movement.behind` is the node the train last passed, so on a track it is always the end the train entered from.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import ceil, inf
from typing import Literal, NamedTuple

from .events import EventLog
from .interlocking import Interlocking, Refusal, Resource, claims, walk
from .model import (
    UNANNOUNCED,
    Call,
    Connection,
    Disruption,
    Move,
    Network,
    Platform,
    Side,
    Timetable,
    Track,
    TrainSpec,
    Trip,
)
from .routing import find_route, portal_track
from .rules import NetRule

IDLE_MIN = 30  # §5.2 item 7: look in at least this often
LOOKAHEAD_MIN = 15  # §5.1: contention is raised this far ahead, not only for trains ready this minute
TURNAROUND_MIN = 8  # §2.3: a booked stop at least this long is a turnaround, so the driver can change ends
CREW_LIMIT_MIN = 240  # F10, as v0.2: the most a crew may work
CREW_WARNING_MIN = 30  # a warning, and a decision point, when this much duty is left
CREW_RELIEF_MIN = 8  # a relief at a crew-relief platform holds the train this long
CREW_TIMEOUT_MIN = 45  # a crew that runs out stops the train where it is until a new crew reaches it
SIGNAL_FAILURE_FACTOR = 2  # §4.2: a track under a signal failure is worked at twice its running time
NOTICES = ("DISRUPTION", "CREW", "GRIDLOCK")  # the order decision points of one minute are handed over in
Phase = Literal["offstage", "berthed", "running", "done"]
Key = Resource | tuple[str, str]  # a claimed resource, or a pair of moves whose paths cross
CLOSES = {
    "possession": NetRule.POSSESSION,
    "points_failure": NetRule.POINTS_FAILED,
    "platform_closed": NetRule.PLATFORM_CLOSED,
}


def track_minutes(track: Track, train: TrainSpec) -> int:
    """Minutes to run the whole track. Never less than one minute per block, which is the signalling floor."""
    speed = min(track.line_speed_kmh, train.max_speed_kmh)
    return max(track.blocks, ceil(track.length_m * 60 / (speed * 1000)))


def block_minutes(track: Track, train: TrainSpec, index: int) -> int:
    """Minutes in one block. Every block takes a minute; the last one carries whatever the run time has left."""
    if index < track.blocks - 1:
        return 1
    return track_minutes(track, train) - (track.blocks - 1)


def move_minutes(move: Move, train: TrainSpec) -> int:
    speed = min(move.speed_kmh, train.max_speed_kmh)
    return 1 if speed >= 80 else 2  # clearing time over the points; slow turnouts hold a junction longer


class Conflict(NamedTuple):
    key: Key
    trains: tuple[str, ...]


class Decision(NamedTuple):
    tick: int
    kind: Literal["CONFLICT", "UNROUTABLE", "BLOCKED", "IDLE", "SHIFT_END", "DISRUPTION", "CREW", "GRIDLOCK"]
    conflicts: tuple[Conflict, ...] = ()
    trains: tuple[str, ...] = ()
    disruptions: tuple[str, ...] = ()  # the disruptions revealed or begun, for DISRUPTION


class Visit(NamedTuple):
    """When a train really stood at a platform, which is what the generator books and the reward charges."""

    platform: str
    arrived: int
    departed: int | None = None


class Row(NamedTuple):
    train: str
    phase: str
    position: str
    delay: int


@dataclass
class Movement:
    """One train's runtime state. `plan` is the calls it still has to make, so the agent's actions edit a list.

    `plan_calls[i]` is the index in `trip.calls` of `plan[i]`, the booked call it serves, or None when the train
    only runs there empty (a cancelled service). Booked calls end up in `made` or `missed`, never both.
    """

    trip: Trip
    plan: list[Call]
    position: str
    behind: str  # the node the train last passed; its own node while it stands in a platform
    phase: Phase = "offstage"
    side_in: Side | None = None  # the side it entered its current platform by
    path: list[str] = field(default_factory=list)  # the whole leg; position is path[idx]
    idx: int = 0
    authority: int = 0  # the furthest index in `path` this train holds
    block: int = 0  # blocks completed on the current track, in travel order
    next_tick: int | None = None
    waiting: bool = False  # stopped at a signal, with no authority to go on
    held_until: int | None = None
    arrived: int | None = None
    at: Call | None = None  # the call it stands at
    worst_late: int = 0  # the most it has been late at any call so far
    visits: list[Visit] = field(default_factory=list)  # every call it has actually made
    plan_calls: list[int | None] = field(default_factory=list)
    at_call: int | None = None  # the booked call it stands at
    entry_tick: int = 0  # when it is due to enter; a late entry moves it
    via: list[str] = field(default_factory=list)  # waypoints its next leg must run through, set by route()
    made: set[int] = field(default_factory=set)
    missed: dict[int, str] = field(default_factory=dict)  # booked call -> why it was missed
    passed: set[int] = field(default_factory=set)  # missed calls it has since run through
    stopped_until: int | None = None  # broken down, out of crew or being relieved: it cannot move before this
    stop_since: int | None = None  # the minute it stopped at a signal, until that stop is logged
    cancelled: bool = False
    crew_base: int = 0  # minutes the crew had worked at crew_since
    crew_since: int | None = None  # None while no crew is at work: not entered, finished, relief or timeout
    crew_back: int | None = None  # when a new crew takes over
    crew_warned: bool = False
    relieve_next: bool = False  # relieve the crew at the next crew-relief platform it calls at
    turn_here: bool = False  # a short turn put its next call in the platform it will stand in

    @property
    def train(self) -> TrainSpec:
        return self.trip.train

    def facing(self, net: Network) -> str:
        element = net.element(self.position)
        return element.other(self.behind) if isinstance(element, Track) else self.behind

    @property
    def working(self) -> bool:
        """Still has somewhere to go: another booked call, or the portal it runs out of the area by."""
        return bool(self.plan) or (self.phase == "berthed" and self.trip.exit_node is not None)

    def ready_at(self) -> int | None:
        """When this train next wants to move, or None if it is running or finished."""
        if self.phase == "offstage":
            due = self.entry_tick
        elif self.phase == "berthed" and self.working:
            due = max(self.at.depart or 0, (self.arrived or 0) + self.at.min_dwell)
        else:
            return None
        return max(due, self.held_until or 0, self.stopped_until or 0)

    def booked(self) -> int:
        """The time the timetable has this train away from where it stands: what "timetable order" sorts on."""
        if self.at is None or self.at.depart is None:
            return self.trip.entry_tick
        return self.at.depart

    def lateness(self, now: int) -> int:
        """How late the train is: measured at its last call, or against the call it is running to."""
        if self.at is not None and self.at.arrive is not None and self.phase in {"berthed", "done"}:
            return (self.arrived or now) - self.at.arrive
        if self.plan and self.plan[0].arrive is not None:
            return max(0, now - self.plan[0].arrive)
        return 0

    def crew_used(self, now: int) -> int:
        return self.crew_base + (now - self.crew_since if self.crew_since is not None else 0)


class Engine:
    def __init__(
        self,
        net: Network,
        trips: list[Trip],
        shift_min: int,
        connections: tuple[Connection, ...] | list[Connection] = (),
        disruptions: tuple[Disruption, ...] | list[Disruption] = (),
    ):
        self.net = net
        self._normal_net = net  # normal one-way track directions; single_line() only opens self.net for route()
        self.il = Interlocking(net)
        self.log = EventLog()
        self.tick = 0
        self.shift_min = shift_min
        self.ended = False
        self.orders: dict[Key, list[str]] = {}
        self.stranded: set[str] = set()
        self.reported_blocked: tuple[str, ...] | None = None
        self.gridlocked: set[str] = set()
        self.single_lines: set[str] = set()
        self._tracks = {t.id: t for t in net.tracks}  # as built: single_line() puts a track back to this
        self._pending: dict[str, tuple[list[str], list[str]]] = {}
        self.trains: dict[str, Movement] = {}
        for trip in sorted(trips, key=lambda t: t.train.id):
            if trip.train.id in self.trains:
                raise ValueError(f"two trips for train {trip.train.id}: a unit's whole day is one trip")
            self.trains[trip.train.id] = self._place(trip)
        self.connections = tuple(connections)
        self._links: dict[str, list[int]] = {}
        self._link_calls: dict[int, tuple[int | None, int | None]] = {}
        for i, link in enumerate(self.connections):
            if link.feeder not in self.trains or link.connector not in self.trains:
                missing = link.feeder if link.feeder not in self.trains else link.connector
                raise ValueError(f"connection at {link.node} names unknown train {missing}")
            self._links.setdefault(link.connector, []).append(i)
            self._link_calls[i] = self._pair_connection(link)
        self._resolved: set[int] = set()
        self._first_arrival: dict[tuple[str, str], int] = {}
        self._call_arrival: dict[tuple[str, int], int] = {}
        self._setup_disruptions(disruptions)
        for mv in self.trains.values():
            self._crew_check(mv)

    @classmethod
    def from_timetable(cls, timetable: Timetable) -> Engine:
        return cls(
            timetable.net, list(timetable.trips), timetable.shift_min, timetable.connections, timetable.disruptions
        )

    def _place(self, trip: Trip) -> Movement:
        plan = list(trip.calls)
        if trip.exit_node is not None and portal_track(self.net, trip.exit_node, inbound=False) is None:
            raise ValueError(f"{trip.train.id}: no track leads out of the network to {trip.exit_node}")
        if trip.entry_node is not None:
            entry = portal_track(self.net, trip.entry_node, inbound=True)
            if entry is None:
                raise ValueError(f"{trip.train.id}: no track leads into the network from {trip.entry_node}")
            return Movement(
                trip,
                plan,
                entry,
                trip.entry_node,
                plan_calls=list(range(len(plan))),
                entry_tick=trip.entry_tick,
                crew_base=trip.crew_min,
            )
        start = self.net.element(plan[0].platform)
        if not isinstance(start, Platform):
            raise ValueError(f"{trip.train.id}: {plan[0].platform} is not a platform it can start from")
        if start.length_m < trip.train.length_m:  # it would stand there with no route it could ever be given
            raise ValueError(f"{trip.train.id} is {trip.train.length_m} m and cannot stand in {start.id}")
        if blocker := self.il.occupy(trip.train.id, start.id):
            raise ValueError(f"{trip.train.id} and {blocker} both start in {start.id}")
        visits = [Visit(start.id, trip.entry_tick)]
        phase: Phase = "berthed" if len(plan) > 1 or trip.exit_node is not None else "done"
        mv = Movement(
            trip,
            plan[1:],
            start.id,
            start.node,
            phase,
            at=plan[0],
            arrived=trip.entry_tick,
            visits=visits,
            plan_calls=list(range(1, len(plan))),
            at_call=0,
            entry_tick=trip.entry_tick,
            made={0},
            crew_base=trip.crew_min,
            crew_since=0 if phase == "berthed" else None,
        )
        self.log.emit(0, "ENTER", train=trip.train.id, element=start.id, call=0, booked=trip.entry_tick)
        return mv

    def _pair_connection(self, link: Connection) -> tuple[int | None, int | None]:
        """Pair a Connection with the exact booked feeder arrival call and connector departure call at `link.node`."""
        f_calls = [i for i, c in enumerate(self.trains[link.feeder].trip.calls) if self._node(c.platform) == link.node]
        c_calls = [
            i for i, c in enumerate(self.trains[link.connector].trip.calls) if self._node(c.platform) == link.node
        ]
        for fc in f_calls:
            fa = self.trains[link.feeder].trip.calls[fc].arrive
            for cc in c_calls:
                cd = self.trains[link.connector].trip.calls[cc].depart
                if fa is not None and cd is not None and cd >= fa + link.min_transfer_min:
                    return fc, cc
        for fc in f_calls:
            fa = self.trains[link.feeder].trip.calls[fc].arrive or 0
            for cc in c_calls:
                cd = self.trains[link.connector].trip.calls[cc].depart
                if cd is not None and cd > fa:
                    return fc, cc
        return (f_calls[0] if f_calls else None), (c_calls[0] if c_calls else None)

    def _setup_disruptions(self, disruptions) -> None:
        self.disruptions = tuple(sorted(disruptions, key=lambda d: (d.start, d.id)))
        self._by_id = {d.id: d for d in self.disruptions}
        if len(self._by_id) != len(self.disruptions):
            raise ValueError("two disruptions share an id")
        self._closures: dict[str, list[Disruption]] = {}
        self._slow: dict[str, list[Disruption]] = {}
        self._failed: dict[str, list[Disruption]] = {}
        wanted = {"points_failure": Move, "platform_closed": Platform}
        schedule = []
        for d in self.disruptions:
            if d.train is not None and d.train not in self.trains:
                raise ValueError(f"disruption {d.id} names unknown train {d.train}")
            if d.kind == "late_entry":
                trip = self.trains[d.train].trip
                if trip.entry_node is None or trip.entry_tick != d.start:
                    raise ValueError(
                        f"disruption {d.id}: a late entry starts at the booked entry of a train from a portal"
                    )
            expected = wanted.get(d.kind, Track)
            index = (
                self._slow
                if d.kind == "speed_restriction"
                else self._failed
                if d.kind == "signal_failure"
                else (self._closures if d.kind in CLOSES else None)
            )
            for element in d.elements:
                if not isinstance(self.net.element(element), expected):
                    raise ValueError(f"disruption {d.id}: {element} is not a {expected.__name__.lower()}")
                index.setdefault(element, []).append(d)
            # Within a minute: what ends goes first, then what is announced, then what begins.
            schedule += [(d.end, 0, d.id, "end"), (d.start, 2, d.id, "start")]
            if d.kind not in UNANNOUNCED:
                schedule.append((d.reveal, 1, d.id, "reveal"))
        self._schedule = sorted(schedule)
        self._next_sched = 0
        self._active: set[str] = set()
        self._revealed: list[str] = []

    # ------------------------------------------------------------------ reading

    def bulletins(self) -> list[Disruption]:
        """Every disruption the agent has been told of so far, in the order it was told."""
        return [self._by_id[i] for i in self._revealed]

    def crew_remaining(self, train_id: str) -> int | None:
        """Minutes of duty its crew has left, or None while no crew is at work on it."""
        mv = self.trains[train_id]
        return None if mv.crew_since is None else CREW_LIMIT_MIN - mv.crew_used(self.tick)

    def state(self) -> list[Row]:
        """One row per train: id, phase, where it is, how late. Sorted, so runs can be compared."""
        return sorted(Row(mv.train.id, mv.phase, mv.position, mv.lateness(self.tick)) for mv in self.trains.values())

    # ------------------------------------------------------------------ actions

    def _act(self, name: str, args: dict, refusal: Refusal | None = None) -> Refusal | None:
        """Log an accepted agent action; refusals change nothing and are not appended to the shift log."""
        if refusal is None and not self.ended:
            self.log.emit(self.tick, "ACTION", train=args.get("train"), detail={"action": name, "args": args})
        return refusal

    def _live(self, train_id: str, finished_ok: bool = False) -> Movement | Refusal:
        if self.tick >= self.shift_min:
            return Refusal(NetRule.SHIFT_ENDED, "the shift is over")
        mv = self.trains.get(train_id)
        if mv is None:
            return Refusal(NetRule.UNKNOWN_TRAIN, f"no train {train_id}")
        if not finished_ok and (mv.phase == "done" or mv.cancelled):
            return Refusal(NetRule.TRAIN_DONE, f"{train_id} is {'cancelled' if mv.cancelled else 'finished'}")
        return mv

    def order(self, key: Key | list, train_ids: Sequence[str]) -> Refusal | None:
        """Say who takes a contested section first. Trains left out keep timetable order behind those listed."""
        norm_key: Key = tuple(key) if isinstance(key, list) else key  # type: ignore[assignment]
        args = {"key": list(norm_key) if isinstance(norm_key, tuple) else norm_key, "trains": list(train_ids)}
        if self.tick >= self.shift_min:
            return self._act("order", args, Refusal(NetRule.SHIFT_ENDED, "the shift is over"))
        if not train_ids:
            return self._act("order", args, Refusal(NetRule.BAD_ROUTE, "order needs at least one train"))
        if unknown := [t for t in train_ids if t not in self.trains]:
            return self._act("order", args, Refusal(NetRule.UNKNOWN_TRAIN, f"no train {unknown[0]}"))
        self.orders[norm_key] = list(train_ids)
        return self._act("order", args)

    def hold(self, train_id: str, until_tick: int) -> Refusal | None:
        args = {"train": train_id, "until": until_tick}
        mv = self._live(train_id, finished_ok=True)
        if isinstance(mv, Refusal):
            return self._act("hold", args, mv)
        if mv.phase in {"running", "done"}:
            return self._act("hold", args, Refusal(NetRule.ALREADY_RUNNING, f"{train_id} is {mv.phase}"))
        if until_tick <= self.tick:
            return self._act("hold", args, Refusal(NetRule.HOLD_PAST, f"t={until_tick} is not in the future"))
        mv.held_until = until_tick
        return self._act("hold", args)

    def platform(self, train_id: str, platform_id: str) -> Refusal | None:
        """Re-platform the call the train is next heading to."""
        args = {"train": train_id, "platform": platform_id}
        mv = self._live(train_id, finished_ok=True)
        if isinstance(mv, Refusal):
            return self._act("platform", args, mv)
        if mv.phase in {"running", "done"} or not mv.plan:
            return self._act("platform", args, Refusal(NetRule.ALREADY_RUNNING, f"{train_id} is {mv.phase}"))
        if platform_id == mv.position:
            return self._act("platform", args, Refusal(NetRule.NO_ROUTE, f"{train_id} already stands in {platform_id}"))
        if not isinstance(self.net.element(platform_id), Platform):
            return self._act("platform", args, Refusal(NetRule.NO_ROUTE, f"{platform_id} is not a platform"))
        booked_node = self._node(mv.plan[0].platform)
        if self._node(platform_id) != booked_node:
            refusal = Refusal(NetRule.NO_ROUTE, f"{platform_id} is not at {booked_node}, where {train_id} calls next")
            return self._act("platform", args, refusal)
        stops = [(v, False) for v in mv.via] + [(platform_id, False)]
        if self._chain(mv, mv.position, mv.facing(self.net), self._side_in(mv), stops, via_count=len(mv.via)) is None:
            refusal = Refusal(NetRule.NO_ROUTE, f"no route from {mv.position} to {platform_id} for {train_id}")
            return self._act("platform", args, refusal)
        mv.plan[0] = mv.plan[0].model_copy(update={"platform": platform_id})
        self.stranded.discard(train_id)
        return self._act("platform", args)

    def skip(self, train: str, station: str) -> Refusal | None:
        """Run through the next booked call at `station` without stopping. The call is missed."""
        args = {"train": train, "station": station}
        mv = self._live(train)
        if isinstance(mv, Refusal):
            return self._act("skip", args, mv)
        p = self._find_call(mv, station, len(mv.plan))
        if p is None:
            return self._act("skip", args, Refusal(NetRule.NO_CALL, f"{train} has no call left at {station}"))
        if p == len(mv.plan) - 1 and mv.trip.exit_node is None:
            refusal = Refusal(NetRule.SKIP_DESTINATION, f"{train} ends at {station}; short_turn or cancel it")
            return self._act("skip", args, refusal)
        call = mv.plan_calls[p]
        plan, calls = mv.plan[:p] + mv.plan[p + 1 :], mv.plan_calls[:p] + mv.plan_calls[p + 1 :]
        if refusal := self._replan(mv, plan, calls, mv.via):
            return self._act("skip", args, refusal)
        self._act("skip", args)
        self._miss(mv, call, "skip")
        return None

    def route(self, train: str, via: Sequence[str]) -> Refusal | None:
        """Send the train's next leg through `via` (tracks or platforms, run through in order, never called at).

        Booked calls the diversion can no longer reach are given up, earliest first, until the rest of the plan
        can be routed; each is missed. The final call, or the exit, always stays.
        """
        via_list = list(via)
        args = {"train": train, "via": via_list}
        mv = self._live(train)
        if isinstance(mv, Refusal):
            return self._act("route", args, mv)
        if not via_list:
            return self._act("route", args, Refusal(NetRule.BAD_ROUTE, "route needs at least one element to run via"))
        for element in via_list:
            if not isinstance(self.net.element(element), Track | Platform):
                return self._act("route", args, Refusal(NetRule.UNKNOWN_ELEMENT, f"no track or platform {element!r}"))
        last = len(mv.plan) if mv.trip.exit_node is not None else len(mv.plan) - 1
        refusal = Refusal(NetRule.NO_ROUTE, f"{train} cannot be routed via {' -> '.join(via_list)}")
        for j in range(max(0, last) + 1):
            if self._replan(mv, mv.plan[j:], mv.plan_calls[j:], via_list) is None:
                break
        else:
            return self._act("route", args, refusal)
        self._act("route", args)
        for call in self._dropped:
            self._miss(mv, call, "route")
        return None

    def short_turn(self, train: str, station: str) -> Refusal | None:
        """Turn the train back at `station` and start its return working there.

        The calls from `station` out to the turnaround, and the turnaround itself, are missed. If the return
        working calls at `station` too, the train makes that call where it already stands.
        """
        args = {"train": train, "station": station}
        mv = self._live(train)
        if isinstance(mv, Refusal):
            return self._act("short_turn", args, mv)
        k = next(
            (
                p
                for p, c in enumerate(mv.plan_calls)
                if c is not None
                and mv.trip.calls[c].min_dwell >= TURNAROUND_MIN
                and (p < len(mv.plan) - 1 or mv.trip.exit_node)
            ),
            None,
        )
        if k is None:
            refusal = Refusal(NetRule.NO_RETURN, f"{train} has no return working to start; skip or cancel instead")
            return self._act("short_turn", args, refusal)
        here = (
            mv.phase == "berthed" and not mv.turn_here and self._node(mv.position) == station and mv.at_call is not None
        )
        o = -1 if here else self._find_call(mv, station, k)
        if o is None:
            refusal = Refusal(NetRule.NO_CALL, f"{train} has no call at {station} before it turns")
            return self._act("short_turn", args, refusal)
        dropped = [c for c in mv.plan_calls[o + 1 : k + 1] if c is not None]
        rest, rest_calls = list(mv.plan[k + 1 :]), list(mv.plan_calls[k + 1 :])
        turn_at = mv.position if here else mv.plan[o].platform
        merge = bool(rest) and self._node(rest[0].platform) == station
        if merge:  # the return call there is made in the platform it turns in
            rest[0] = rest[0].model_copy(
                update={"platform": turn_at, "min_dwell": max(rest[0].min_dwell, TURNAROUND_MIN)}
            )
        before_at, before_turn = mv.at, mv.turn_here
        mv.turn_here = merge
        if here:
            plan, calls = rest, rest_calls
            if not merge:
                mv.at = mv.at.model_copy(update={"min_dwell": max(mv.at.min_dwell, TURNAROUND_MIN)})
        else:
            turn = mv.plan[o]
            if not merge:
                turn = turn.model_copy(update={"min_dwell": max(turn.min_dwell, TURNAROUND_MIN)})
            plan, calls = [*mv.plan[:o], turn, *rest], [*mv.plan_calls[: o + 1], *rest_calls]
        if refusal := self._replan(mv, plan, calls, mv.via):
            mv.at, mv.turn_here = before_at, before_turn
            return self._act("short_turn", args, refusal)
        self._act("short_turn", args)
        for call in dropped:
            self._miss(mv, call, "short_turn")
        return None

    def cancel(self, train: str) -> Refusal | None:
        """Withdraw the service. Every call it has left is missed; a train not yet in the area never enters,
        and one already running keeps its path empty, calling nowhere, out to its exit or its last platform."""
        args = {"train": train}
        mv = self._live(train)
        if isinstance(mv, Refusal):
            return self._act("cancel", args, mv)
        dropped = [c for c in mv.plan_calls if c is not None]
        if not dropped:
            return self._act("cancel", args, Refusal(NetRule.NO_CALL, f"{train} has no calls left to cancel"))
        mv.cancelled, mv.relieve_next, mv.held_until = True, False, None
        if mv.phase == "offstage":
            mv.phase, mv.plan, mv.plan_calls = "done", [], []
        else:

            def empty(call: Call) -> Call:  # no dwell, no booked minute, unless it must turn there
                keep = call.min_dwell if call.min_dwell >= TURNAROUND_MIN else 0
                return call.model_copy(update={"depart": None, "min_dwell": keep})

            mv.plan, mv.plan_calls = [empty(c) for c in mv.plan], [None] * len(mv.plan)
            if mv.phase == "berthed":
                mv.at, mv.at_call = empty(mv.at), None
                if not mv.working or (not self._in_place_due(mv) and self._journey(mv) is None):
                    mv.plan, mv.plan_calls = [], []
                    mv.phase, mv.crew_since = "done", None
                    self.stranded.discard(train)
        self._act("cancel", args)
        for call in dropped:
            self._miss(mv, call, "cancel")
        return None

    def single_line(self, segment: str, on: bool) -> Refusal | None:
        """Open a reversible track to both directions, or return it to its normal one (§2.1, §5.3).

        Trains only run on it the wrong way when the agent routes them there.
        """
        args = {"segment": segment, "on": on}
        if self.tick >= self.shift_min:
            return self._act("single_line", args, Refusal(NetRule.SHIFT_ENDED, "the shift is over"))
        built = self._tracks.get(segment)
        if built is None:
            return self._act("single_line", args, Refusal(NetRule.UNKNOWN_ELEMENT, f"no track {segment!r}"))
        if not built.reversible:
            refusal = Refusal(NetRule.NOT_REVERSIBLE, f"{segment} is not signalled for running both ways")
            return self._act("single_line", args, refusal)
        if on == (segment in self.single_lines):
            refusal = Refusal(
                NetRule.SINGLE_LINE, f"single-line working on {segment} is already {'on' if on else 'off'}"
            )
            return self._act("single_line", args, refusal)
        if not on and (user := self._wrong_way_user(built)):
            refusal = Refusal(NetRule.SINGLE_LINE, f"{user} is routed the wrong way over {segment}")
            return self._act("single_line", args, refusal)
        if on:
            self.single_lines.add(segment)
            self._rebuild({segment: built.model_copy(update={"direction": "both", "reversible": False})})
        else:
            self.single_lines.discard(segment)
            self._rebuild({segment: built})
        return self._act("single_line", args)

    def relieve_crew(self, train: str) -> Refusal | None:
        """Relieve the crew: now, if the train stands at a crew-relief platform, else at the next one it calls at."""
        args = {"train": train}
        mv = self._live(train)
        if isinstance(mv, Refusal):
            return self._act("relieve_crew", args, mv)
        if mv.relieve_next or (mv.crew_back is not None and mv.crew_back > self.tick):
            refusal = Refusal(NetRule.CREW_RELIEF, f"a new crew is already on its way to {train}")
            return self._act("relieve_crew", args, refusal)
        here = self.net.element(mv.position)
        if mv.phase == "berthed" and isinstance(here, Platform) and here.crew_relief:
            self._act("relieve_crew", args)
            self._start_relief(mv, here)
            return None
        if not any(self.net.element(c.platform).crew_relief for c in mv.plan):
            refusal = Refusal(NetRule.NOT_CREW_POINT, f"{train} calls at no crew-relief platform")
            return self._act("relieve_crew", args, refusal)
        mv.relieve_next = True
        return self._act("relieve_crew", args)

    # ------------------------------------------------------------------ plan editing

    def _node(self, element_id: str) -> str | None:
        element = self.net.element(element_id)
        return element.node if isinstance(element, Platform | Move) else None

    def _find_call(self, mv: Movement, station: str, before: int) -> int | None:
        """The first place in the plan, before `before`, that is a booked call at `station`."""
        for p in range(min(before, len(mv.plan))):
            if mv.plan_calls[p] is not None and self._node(mv.plan[p].platform) == station:
                return p
        return None

    def _stops(self, mv: Movement, plan: list[Call]) -> list[tuple[str, bool]]:
        """Where the train still has to get to, and whether it may leave each the way it came in."""
        out = [(c.platform, c.min_dwell >= TURNAROUND_MIN) for c in plan]
        if mv.trip.exit_node is not None:
            out.append((portal_track(self.net, mv.trip.exit_node, inbound=False), False))
        return out

    def _leg_start(self, mv: Movement) -> tuple[list[str], str, str, Side | None]:
        """Where a new leg is routed from: the path already authorised, its last element, facing, side entered."""
        if mv.phase != "running":
            return [], mv.position, mv.facing(self.net), self._side_in(mv)
        a = mv.authority
        end = self.net.element(mv.path[a])
        prev = self.net.element(mv.path[a - 1]) if a else None
        if isinstance(end, Track):
            facing = end.other(mv.behind) if a == mv.idx else end.other(prev.node)
            return mv.path[: a + 1], end.id, facing, None
        return mv.path[: a + 1], end.id, end.node, prev.side if isinstance(prev, Move) else None

    def _chain(
        self, mv: Movement, start: str, facing: str, side: Side | None, stops, via_count: int = 0
    ) -> list[list[str]] | None:
        """Route through each stop in turn, carrying the side a platform was entered by; None if one is out of reach."""
        out = []
        for i, (target, turns) in enumerate(stops):
            is_via = i < via_count
            from_via = via_count > 0 and i == via_count
            net = self.net if via_count > 0 and i <= via_count else self._normal_net
            if target == start:
                if is_via or from_via or not turns:
                    return None
                path = [start]
            elif (path := find_route(net, start, facing, target, mv.train, side)) is None:
                return None
            out.append(path)
            element = self.net.element(target)
            if len(path) > 1:
                prev = self.net.element(path[-2])
                if isinstance(element, Track):
                    facing, side = element.other(prev.node), None
                else:
                    facing, side = element.node, prev.side
            if turns:
                side = None
            start = target
        return out

    def _replan(self, mv: Movement, plan: list[Call], calls: list[int | None], via) -> Refusal | None:
        """Give the train a new plan if it can still be routed through every stop of it; otherwise change nothing.

        A running train keeps what it has already been given and has its leg re-routed from the end of it.
        The calls cut from the front of the plan are left in `self._dropped`.
        """
        prefix, start, facing, side = self._leg_start(mv)
        stops = [(v, False) for v in via] + self._stops(mv, plan)
        if not stops or (isinstance(self.net.element(stops[-1][0]), Track) and mv.trip.exit_node is None):
            return Refusal(NetRule.NO_ROUTE, f"{mv.train.id} would be left with nowhere to go")
        legs = self._chain(mv, start, facing, side, stops, via_count=len(via))
        if legs is None:
            return Refusal(NetRule.NO_ROUTE, f"{mv.train.id} cannot be routed {' -> '.join(t for t, _ in stops)}")
        if mv.phase == "running":
            leg = list(prefix)
            for path in legs[: len(via) + 1]:
                leg += path[1:]
            if len(leg) - 1 == mv.idx and isinstance(self.net.element(leg[-1]), Platform):
                return Refusal(NetRule.NO_ROUTE, f"{mv.train.id} is already running through {leg[-1]}")
            mv.path, via = leg, []
        kept = set(calls)
        self._dropped = [c for c in mv.plan_calls if c is not None and c not in kept]
        mv.plan, mv.plan_calls, mv.via = list(plan), list(calls), list(via)
        if mv.phase == "berthed" and mv.turn_here and (not plan or plan[0].platform != mv.position):
            mv.turn_here = False
        self.stranded.discard(mv.train.id)
        return None

    def _miss(self, mv: Movement, call: int, why: str) -> None:
        booked = mv.trip.calls[call]
        mv.missed[call] = why
        self.log.emit(
            self.tick,
            "CALL_MISSED",
            train=mv.train.id,
            element=booked.platform,
            node=self._node(booked.platform),
            call=call,
            booked=booked.arrive,
            detail={"why": why},
        )

    def _wrong_way_user(self, built: Track) -> str | None:
        """A train that holds, or is routed over, this track against its normal direction."""
        for mv in sorted(self.trains.values(), key=lambda m: m.train.id):
            if any(d != built.direction for t, d in self.il.directions.get(built.id, {}).items() if t == mv.train.id):
                return mv.train.id
            path = mv.path[mv.idx :] if mv.phase == "running" else ((self._journey(mv) or []) if mv.via else [])
            for i, eid in enumerate(path):
                if eid == built.id:
                    came = mv.behind if i == 0 else self.net.element(path[i - 1]).node
                    if built.travel_from(came) != built.direction:
                        return mv.train.id
        return None

    def _rebuild(self, changed: dict[str, Track]) -> None:
        tracks = tuple(changed.get(t.id, t) for t in self.net.tracks)
        net = self.net
        self.net = Network(
            nodes=net.nodes, tracks=tracks, platforms=net.platforms, moves=net.moves, crossings=net.crossings
        )
        self.il.net = self.net

    # ------------------------------------------------------------------ time

    def advance(self) -> Decision:
        """Run until the agent is needed, or the shift ends."""
        return self._advance_loop(unattended=False)

    def end_shift(self) -> Decision:
        """Close out the rest of the shift unattended and emit SHIFT_END, CALL_MISSED and UNFINISHED."""
        if self.ended:
            return Decision(self.tick, "SHIFT_END")
        return self._advance_loop(unattended=True)

    def _advance_loop(self, unattended: bool) -> Decision:
        while self.tick < self.shift_min:
            self._clock()
            if unattended:
                self._pending.clear()
            elif notice := self._notice():
                return notice
            if decision := self._set_routes(unattended=unattended):
                return decision
            self._gridlock()
            if unattended:
                self._pending.clear()
            elif notice := self._notice():
                return notice
            nxt = self._next_event()
            if nxt is None:
                if not unattended and (decision := self._blocked()):
                    return decision
                break
            self._flush_stops()
            if not unattended and nxt - self.tick > IDLE_MIN:
                self.tick = min(self.tick + IDLE_MIN, self.shift_min)
                if self.tick >= self.shift_min:
                    break
                return Decision(self.tick, "IDLE")
            self.tick = min(nxt, self.shift_min)
            self._clock()
            self._run_transitions()
        self._flush_stops()
        self.tick = self.shift_min
        self._finish()
        return Decision(self.tick, "SHIFT_END")

    def _next_event(self) -> int | None:
        times = [mv.next_tick for mv in self.trains.values() if mv.next_tick is not None]
        for mv in self.trains.values():
            times += [t for t in (mv.ready_at(), mv.stopped_until, mv.crew_back) if t is not None and t > self.tick]
        if self._next_sched < len(self._schedule):
            sched = self._schedule[self._next_sched][0]
            if sched <= self.shift_min:
                times.append(sched)
        if not times:
            return None
        for mv in self.trains.values():
            if mv.crew_since is not None:
                due = mv.crew_since + CREW_LIMIT_MIN - mv.crew_base
                warn = due - CREW_WARNING_MIN
                if self.tick < due <= self.shift_min:
                    times.append(due)
                if not mv.crew_warned and self.tick < warn <= self.shift_min:
                    times.append(warn)
        return min(times)

    def _blocked(self) -> Decision | None:
        """Nothing can move and nothing is due: the trains still out there are deadlocked (§5.2 item 6).

        Reported once per set of stuck trains; if the agent cannot free them, the shift simply runs out.
        """
        stuck = tuple(
            sorted(
                mv.train.id
                for mv in self.trains.values()
                if mv.waiting or ((due := mv.ready_at()) is not None and due <= self.tick)
            )
        )
        if not stuck or stuck == self.reported_blocked or set(stuck) <= self.gridlocked:
            return None
        self.reported_blocked = stuck
        return Decision(self.tick, "BLOCKED", trains=stuck)

    def _notify(self, kind: str, train: str | None = None, disruption: str | None = None) -> None:
        trains, ids = self._pending.setdefault(kind, ([], []))
        if train is not None:
            trains.append(train)
        if disruption is not None:
            ids.append(disruption)

    def _notice(self) -> Decision | None:
        for kind in NOTICES:
            if kind in self._pending:
                trains, ids = self._pending.pop(kind)
                return Decision(self.tick, kind, trains=tuple(sorted(set(trains))), disruptions=tuple(ids))
        return None

    def _flush_stops(self) -> None:
        """Log the trains still standing at a signal as time moves on: a section granted in the same minute
        the last one ran out is a clear signal, not a stop."""
        for mv in sorted(self.trains.values(), key=lambda m: m.train.id):
            if mv.waiting and mv.stop_since is not None:
                self.log.emit(
                    self.tick,
                    "SIGNAL_STOP",
                    train=mv.train.id,
                    element=mv.position,
                    detail={"restart_min": mv.train.restart_min},
                )
                mv.stop_since = None

    def _finish(self) -> None:
        """SHIFT_END, then every booked call never made and every train whose work is not done."""
        if self.ended:
            return
        self.log.emit(self.tick, "SHIFT_END")
        for train_id, mv in sorted(self.trains.items()):
            for call in range(len(mv.trip.calls)):
                if call not in mv.made and call not in mv.missed:
                    self._miss(mv, call, "shift_end")
            if mv.phase != "done" or mv.cancelled:
                last = mv.trip.calls[-1]
                booked = getattr(mv.trip, "exit_tick", None)
                if booked is None:
                    booked = last.arrive if last.arrive is not None else last.depart
                self.log.emit(
                    self.tick,
                    "UNFINISHED",
                    train=train_id,
                    booked=booked,
                    detail={"freight": not mv.train.passenger},
                )
        self.ended = True

    # ------------------------------------------------------------------ disruptions and crews

    def _clock(self) -> None:
        """Apply everything due by now: disruptions beginning, ending and being announced, and crew duty."""
        while self._next_sched < len(self._schedule) and self._schedule[self._next_sched][0] <= self.tick:
            _, _, disruption_id, what = self._schedule[self._next_sched]
            self._next_sched += 1
            d = self._by_id[disruption_id]
            if what == "reveal":
                self._reveal(d)
            elif what == "start":
                self._begin(d)
            elif d.id in self._active:
                self._active.discard(d.id)
                self._disruption_event("DISRUPTION_END", d)
        for mv in sorted(self.trains.values(), key=lambda m: m.train.id):
            self._crew_check(mv)

    def _reveal(self, d: Disruption) -> None:
        self._revealed.append(d.id)
        if d.kind == "late_entry":
            mv = self.trains[d.train]
            if mv.phase == "offstage":
                mv.entry_tick = d.end
        self._notify("DISRUPTION", d.train, d.id)

    def _begin(self, d: Disruption) -> None:
        mv = self.trains.get(d.train) if d.train else None
        if d.kind == "breakdown":
            if mv.phase not in {"running", "berthed"}:
                return  # it is not in the area at that minute, so nothing happens and nobody hears of it
            self._stop(mv, d.end)
        self._active.add(d.id)
        self._disruption_event("DISRUPTION_START", d)
        if d.kind in UNANNOUNCED:
            self._revealed.append(d.id)
            self._notify("DISRUPTION", d.train, d.id)

    def _disruption_event(self, kind, d: Disruption) -> None:
        if d.kind == "breakdown":
            element = self.trains[d.train].position
        elif d.kind == "late_entry":
            element = portal_track(self.net, self.trains[d.train].trip.entry_node, inbound=True)
        else:
            element = d.elements[0] if len(d.elements) == 1 else None
        node = self._node(element) if element else None
        if d.kind == "late_entry":
            node = self.trains[d.train].trip.entry_node
        detail = {"id": d.id, "kind": d.kind, "train": d.train, "elements": list(d.elements)}
        self.log.emit(self.tick, kind, element=element, node=node, detail=detail)

    def _stop(self, mv: Movement, until: int) -> None:
        """Stop a train where it stands until `until`; one already moving loses the time and restarts after it."""
        before = mv.stopped_until if mv.stopped_until is not None and mv.stopped_until > self.tick else None
        mv.stopped_until = max(until, before or 0)
        if mv.phase == "running" and mv.next_tick is not None:
            if before is None:
                mv.next_tick = mv.stopped_until + (mv.next_tick - self.tick) + mv.train.restart_min
            else:
                mv.next_tick += mv.stopped_until - before

    def _crew_check(self, mv: Movement) -> None:
        if mv.crew_back is not None and mv.crew_back <= self.tick:
            if mv.phase in {"running", "berthed"}:
                mv.crew_base, mv.crew_since, mv.crew_warned = 0, mv.crew_back, False
            mv.crew_back = None
        if mv.crew_since is None:
            return
        left = CREW_LIMIT_MIN - mv.crew_used(self.tick)
        if left <= CREW_WARNING_MIN and not mv.crew_warned:
            mv.crew_warned = True
            self.log.emit(self.tick, "CREW_WARNING", train=mv.train.id, detail={"remaining_min": left})
            self._notify("CREW", mv.train.id)
        if left <= 0:
            self.log.emit(
                self.tick,
                "CREW_TIMEOUT",
                train=mv.train.id,
                element=mv.position,
                detail={"blocked_min": CREW_TIMEOUT_MIN},
            )
            mv.crew_since, mv.crew_back = None, self.tick + CREW_TIMEOUT_MIN
            self._stop(mv, mv.crew_back)
            self._notify("CREW", mv.train.id)

    def _start_relief(self, mv: Movement, platform: Platform) -> None:
        until = self.tick + CREW_RELIEF_MIN
        self.log.emit(
            self.tick,
            "CREW_RELIEF",
            train=mv.train.id,
            element=platform.id,
            node=platform.node,
            detail={"until": until},
        )
        mv.crew_since, mv.crew_back, mv.relieve_next = None, until, False
        self._stop(mv, until)

    def _closed(self, mv: Movement, segment: list[str]) -> Refusal | None:
        """A section crossing something closed: active now, or (once revealed) starting before the train clears it."""
        t = self.tick
        for i, element_id in enumerate(segment):
            element = self.net.element(element_id)
            if i == 0 and mv.phase != "offstage":
                t += mv.train.restart_min if mv.phase == "running" and mv.waiting else 1
                continue  # where it stands
            dur = (
                self._track_min(element, mv.train)
                if isinstance(element, Track)
                else (move_minutes(element, mv.train) if isinstance(element, Move) else 1)
            )
            for d in self._closures.get(element_id, ()):
                if d.kind in UNANNOUNCED or d.id not in self._revealed:
                    hit = d.id in self._active
                else:
                    hit = d.start < t + dur and self.tick < d.end
                if hit:
                    return Refusal(CLOSES[d.kind], f"{element_id} is closed by {d.id} until t={d.end}")
            t += dur
        return None

    def _track_min(self, track: Track, train: TrainSpec) -> int:
        speed = min(track.line_speed_kmh, train.max_speed_kmh)
        for d in self._slow.get(track.id, ()):
            if d.id in self._active:
                speed = min(speed, d.speed_kmh)
        return max(track.blocks, ceil(track.length_m * 60 / (speed * 1000)))

    def _block_min(self, track: Track, train: TrainSpec, index: int) -> int:
        """block_minutes, slowed by any speed restriction or signal failure in force as the block is entered."""
        per = SIGNAL_FAILURE_FACTOR if any(d.id in self._active for d in self._failed.get(track.id, ())) else 1
        if index < track.blocks - 1:
            return per
        return per * (self._track_min(track, train) - (track.blocks - 1))

    # ------------------------------------------------------------------ gridlock

    def _waits_on(self, mv: Movement) -> set[str]:
        """The trains this one is waiting for, if it wants to move now and only other trains stop it."""
        if mv.phase not in {"running", "berthed"} or (mv.stopped_until or 0) > self.tick:
            return set()
        if mv.phase == "running":
            if not mv.waiting:
                return set()
            if self._blocks_left(mv):
                track = self.net.element(mv.position)
                order = list(range(track.blocks))
                if track.travel_from(mv.behind) == "ba":
                    order.reverse()
                holder = self.il.holder.get((track.id, order[mv.block + 1]))
                return {holder} - {None, mv.train.id}
        elif (due := mv.ready_at()) is None or due > self.tick:
            return set()
        path = self._journey(mv)
        if path is None:
            return set()
        segment = self._segment(mv, path)
        if self._closed(mv, segment) is not None:
            return set()
        return self.il.blockers(mv.train, mv.behind, segment)

    def _can_divert(self, mv: Movement) -> bool:
        """Whether a running train blocked at an intermediate platform can divert around it via another road."""
        if mv.phase != "running" or self._blocks_left(mv) or not mv.path:
            return False
        segment = self._segment(mv, mv.path)
        if len(segment) < 3 or not isinstance(target := self.net.element(segment[-1]), Platform):
            return False
        if len(mv.plan) <= (0 if mv.trip.exit_node else 1):
            return False
        end_stop = self._stops(mv, mv.plan)[-1]
        for move in self.net.moves:
            if move.node != target.node or mv.position not in move.ends:
                continue
            alt = self.net.element(move.other(mv.position))
            if not isinstance(alt, Platform) or alt.id == target.id:
                continue
            alt_seg = [mv.position, move.id, alt.id]
            if self._closed(mv, alt_seg) is not None or self.il.check(mv.train, mv.behind, alt_seg) != walk(
                self.net, mv.behind, alt_seg
            ):
                continue
            if self._chain(mv, mv.position, mv.facing(self.net), None, [(alt.id, False), end_stop], via_count=1):
                return True
        return False

    def _immobile(self, mv: Movement) -> bool:
        """Never going to move again on its own: parked for the shift, or due away with nowhere it can be routed."""
        if mv.phase == "done":
            return True
        due = mv.ready_at()
        return (
            mv.phase == "berthed"
            and due is not None
            and due <= self.tick
            and not self._in_place_due(mv)
            and self._journey(mv) is None
        )

    def _gridlock(self) -> None:
        """Find the trains that can never move again: each waits for a train that is itself stuck, or parked.

        A train waiting for one that is free to move, or for anything but a train (an order, a hold, a closure,
        a breakdown), is not gridlocked: that will clear. GRIDLOCK is logged once for each new group.
        """
        waits = {t: w for t, mv in sorted(self.trains.items()) if (w := self._waits_on(mv))}
        if not waits and not self.gridlocked:
            return
        fixed = {t for t, mv in self.trains.items() if t not in waits and t in self._holders() and self._immobile(mv)}
        stuck = set(waits)
        changed = True
        while changed:
            changed = False
            for t in sorted(stuck):
                if not (waits[t] & (stuck | fixed)) or (not (waits[t] & stuck) and self._can_divert(self.trains[t])):
                    stuck.discard(t)
                    changed = True
        fresh = [group for group in self._groups(stuck, waits) if not group & self.gridlocked]
        self.gridlocked |= stuck
        for group in fresh:
            trains = sorted(group)
            self.log.emit(self.tick, "GRIDLOCK", detail={"trains": trains})
            for t in trains:
                self._notify("GRIDLOCK", t)

    def _holders(self) -> set[str]:
        return set(self.il.holder.values())

    @staticmethod
    def _groups(stuck: set[str], waits: dict[str, set[str]]) -> list[set[str]]:
        """The stuck trains split into groups that wait on one another."""
        groups, seen = [], set()
        for start in sorted(stuck):
            if start in seen:
                continue
            group, todo = set(), [start]
            while todo:
                t = todo.pop()
                if t in group:
                    continue
                group.add(t)
                todo += [w for w in waits[t] if w in stuck]
                todo += [o for o in stuck if t in waits[o]]
            seen |= group
            groups.append(group)
        return groups

    # ------------------------------------------------------------------ route setting

    def _side_in(self, mv: Movement) -> Side | None:
        """The platform side it may not leave by; a turnaround dwell clears it, because the driver changes ends."""
        if mv.at is not None and mv.at.min_dwell >= TURNAROUND_MIN:
            return None
        return mv.side_in

    def _journey(self, mv: Movement) -> list[str] | None:
        if mv.path:
            return mv.path
        if mv.plan:
            target = mv.plan[0].platform
        elif mv.working:  # its calls are made: it runs off the area through its exit portal
            target = portal_track(self.net, mv.trip.exit_node, inbound=False)
        else:
            return None
        if not mv.via:
            return find_route(self._normal_net, mv.position, mv.facing(self.net), target, mv.train, self._side_in(mv))
        stops = [(v, False) for v in mv.via] + [(target, False)]
        legs = self._chain(mv, mv.position, mv.facing(self.net), self._side_in(mv), stops, via_count=len(mv.via))
        if legs is None:
            return None
        path = legs[0]
        for leg in legs[1:]:
            path = path + leg[1:]
        return path if len(path) > 1 else None

    def _segment(self, mv: Movement, path: list[str]) -> list[str]:
        """The next section to ask for: the moves up to, and including, the next track or platform."""
        return path[mv.idx : mv.idx + 3]

    def _wanting(self, window: int) -> dict[str, list[str]]:
        """Every train that wants a new section within `window` minutes, and the section it wants."""
        out: dict[str, list[str]] = {}
        for mv in sorted(self.trains.values(), key=lambda m: m.train.id):
            due = mv.ready_at()
            soon = due is not None and due <= self.tick + window
            if not (soon or (mv.phase == "running" and mv.waiting)):
                continue
            if mv.phase == "running" and (mv.stopped_until or 0) > self.tick:
                continue  # broken down or out of crew: it wants nothing until it can move
            if self._blocks_left(mv):
                continue  # stopped by the train ahead in its own section: it needs that block, not a new route
            path = self._journey(mv)
            if path is None:
                continue
            segment = self._segment(mv, path)
            if len(segment) >= 2:
                out[mv.train.id] = segment
        return out

    def _blocks_left(self, mv: Movement) -> bool:
        """Whether the train still has track to run before the signal that guards its next section."""
        track = self.net.element(mv.position)
        return mv.phase == "running" and isinstance(track, Track) and mv.block < track.blocks - 1

    def _running_out(self, mv: Movement) -> bool:
        """Whether the train is on the last element of its route and that element is a track: the leg to a portal.

        Only an exit leg ends on a track; every other route ends at the platform the train has been called to.
        """
        return mv.idx == len(mv.path) - 1 and isinstance(self.net.element(mv.position), Track)

    def _halt(self, mv: Movement) -> None:
        """Stand a running train at the signal ahead of it."""
        if not mv.waiting:
            mv.stop_since = self.tick
        mv.waiting, mv.next_tick = True, None

    def _advance_block(self, mv: Movement, track: Track, restarting: bool) -> None:
        """Move a train up one block inside the track it already holds, or stand it at the signal."""
        if self.il.step_block(mv.train.id, track.id, track.travel_from(mv.behind), mv.block) is not None:
            return self._halt(mv)  # the block ahead is taken: stand at the signal
        mv.block, mv.waiting, mv.stop_since = mv.block + 1, False, None
        restart = mv.train.restart_min if restarting else 0  # heavy trains take longer to get going again
        mv.next_tick = self.tick + restart + self._block_min(track, mv.train, mv.block)

    def _resume(self) -> None:
        """A train held by the one ahead in its own section moves up as soon as that block clears.

        Setting a route past it instead would let a following train lock the junction its leader still needs,
        which deadlocks the pair: the leader cannot clear and the follower cannot pass it.
        """
        for mv in sorted(self.trains.values(), key=lambda m: m.train.id):
            if mv.waiting and self._blocks_left(mv) and (mv.stopped_until or 0) <= self.tick:
                self._advance_block(mv, self.net.element(mv.position), restarting=True)

    def _in_place_due(self, mv: Movement) -> bool:
        return mv.turn_here and mv.phase == "berthed" and bool(mv.plan) and mv.plan[0].platform == mv.position

    def _in_place(self) -> None:
        """A train whose next call is in the platform it stands in (a short turn's return call) makes it there."""
        for mv in sorted(self.trains.values(), key=lambda m: m.train.id):
            due = mv.ready_at()
            if self._in_place_due(mv) and due is not None and due <= self.tick:
                platform = self.net.element(mv.position)
                self._departed(mv)
                mv.visits[-1] = mv.visits[-1]._replace(departed=self.tick)
                mv.held_until, mv.turn_here = None, False
                self._berthed(mv, platform)

    def _set_routes(self, unattended: bool = False) -> Decision | None:
        """One ARS pass: extend authority where nothing competes, and hand real contention to the agent."""
        self._resume()
        self._in_place()
        if (fresh := self._unroutable()) and not unattended:
            return Decision(self.tick, "UNROUTABLE", trains=fresh)
        soon = self._wanting(LOOKAHEAD_MIN)
        conflicts = self._conflicts(soon)
        if conflicts and not unattended:
            names = tuple(sorted({t for c in conflicts for t in c.trains}))
            return Decision(self.tick, "CONFLICT", conflicts, names)
        contested = {t for c in conflicts for t in c.trains} if unattended else set()
        now = {t: s for t, s in self._wanting(0).items() if t in soon and t not in contested}
        for train_id in self._priority(now):
            self._extend(self.trains[train_id], now[train_id])
        # An order stands until the trains it names have finished. Pruning it by what is wanted this minute would
        # drop it while one of them is held behind another, and the same conflict would simply be raised again.
        self.orders = {
            key: order
            for key, order in self.orders.items()
            if any(self.trains[t].working for t in order if t in self.trains)
        }
        return None

    def _unroutable(self) -> tuple[str, ...]:
        lost = set()
        for mv in self.trains.values():
            due = mv.ready_at()
            if due is not None and due <= self.tick and mv.working and self._journey(mv) is None:
                lost.add(mv.train.id)
        fresh = tuple(sorted(lost - self.stranded))
        self.stranded = (self.stranded | lost) & lost
        return fresh

    def _claims(self, mv: Movement, segment: list[str]) -> list[Resource]:
        steps = walk(self.net, mv.behind, segment)
        if isinstance(steps, Refusal):
            return []
        return [resource for resource, _ in claims(steps, self.il.entering(mv.behind, steps))]

    def _keys(self, segment: list[str]) -> tuple[set[str], set[str]]:
        """The moves a section sets, and the tracks it runs over."""
        moves = {e for e in segment if isinstance(self.net.element(e), Move)}
        tracks = {e for e in segment if isinstance(self.net.element(e), Track)}
        return moves, tracks

    def _keys_in_play(self, wanted: dict[str, list[str]]) -> set[Key]:
        keys: set[Key] = set()
        for train_id, segment in wanted.items():
            keys |= set(self._claims(self.trains[train_id], segment))
            moves, tracks = self._keys(segment)
            keys |= moves | tracks
            for move in moves:
                keys |= {tuple(sorted((move, other))) for other in self.net.conflicts(move)}
        return keys

    def _conflicts(self, wanted: dict[str, list[str]]) -> tuple[Conflict, ...]:
        """Who competes for what: shared resources, moves that cross, and tracks wanted both ways."""
        users: dict[Key, set[str]] = {}

        def note(key: Key, train_id: str) -> None:
            users.setdefault(key, set()).add(train_id)

        move_users: dict[str, set[str]] = {}
        for train_id, segment in wanted.items():
            mv = self.trains[train_id]
            for resource in self._claims(mv, segment):
                note(resource, train_id)
            moves, _ = self._keys(segment)
            for move in moves:
                move_users.setdefault(move, set()).add(train_id)
        for move, holders in move_users.items():
            for other in self.net.conflicts(move):
                for train_id in move_users.get(other, ()):
                    if train_id not in holders:
                        note(tuple(sorted((move, other))), train_id)
                        for mine in holders:
                            note(tuple(sorted((move, other))), mine)
        found = [
            Conflict(key, tuple(sorted(ids)))
            for key, ids in sorted(users.items(), key=lambda kv: str(kv[0]))
            if len(ids) > 1 and not (set(ids) & set(self.orders.get(key, ())))
        ]
        return tuple(found)

    def _priority(self, wanted: dict[str, list[str]]) -> list[str]:
        """Explicit orders first, then the timetable: earliest booked departure, then train id."""
        keys = {t: self._keys_in_play({t: s}) for t, s in wanted.items()}

        def key(train_id: str) -> tuple[float, int, str]:
            ranks = [
                order.index(train_id) for k, order in self.orders.items() if train_id in order and k in keys[train_id]
            ]
            return (min(ranks) if ranks else inf, self.trains[train_id].booked(), train_id)

        return sorted(wanted, key=key)

    def _extend(self, mv: Movement, segment: list[str]) -> bool:
        """Ask the interlocking for the next section. On success the train starts, or starts again."""
        if self._closed(mv, segment) is not None or self.il.lock(mv.train, mv.behind, segment) is not None:
            if mv.phase == "running":
                self._halt(mv)
            return False
        mv.authority = mv.idx + len(segment) - 1
        if mv.phase != "running":
            entering = mv.phase == "offstage"
            if entering:
                self._entered(mv)
            else:
                self._departed(mv)
            if mv.visits:
                mv.visits[-1] = mv.visits[-1]._replace(departed=self.tick)
            mv.path, mv.idx, mv.phase, mv.held_until = self._journey(mv), 0, "running", None
            mv.via = []
            mv.authority = len(segment) - 1
            element = self.net.element(mv.position)
            start = self._block_min(element, mv.train, 0) if isinstance(element, Track) else 1
            mv.next_tick = self.tick + start
            if entering:
                mv.crew_base, mv.crew_since = mv.trip.crew_min, self.tick
                self._crew_check(mv)
        elif mv.waiting:
            mv.waiting, mv.stop_since = False, None
            mv.next_tick = self.tick + mv.train.restart_min  # heavy trains take longer to get going again
        return True

    # ------------------------------------------------------------------ movement

    def _run_transitions(self) -> None:
        for mv in sorted(self.trains.values(), key=lambda m: m.train.id):
            if mv.phase == "running" and mv.next_tick == self.tick:
                self._step(mv)

    def _step(self, mv: Movement) -> None:
        current = self.net.element(mv.path[mv.idx])
        if self._blocks_left(mv):
            return self._advance_block(mv, current, restarting=False)
        if self._running_out(mv):
            return self._leave(mv, current)
        if mv.authority <= mv.idx:
            return self._halt(mv)
        if isinstance(current, Track):
            mv.behind = current.other(mv.behind)
        self.il.release(mv.train.id, current.id)
        mv.idx += 1
        self._enter(mv)

    def _enter(self, mv: Movement) -> None:
        element = self.net.element(mv.path[mv.idx])
        mv.position = element.id
        if isinstance(element, Track):
            mv.block = 0
            mv.next_tick = self.tick + self._block_min(element, mv.train, 0)
        elif isinstance(element, Move):
            mv.next_tick = self.tick + move_minutes(element, mv.train)
        elif mv.idx == len(mv.path) - 1:
            self._berth(mv, element)
        else:
            self._passed(mv, element)
            mv.next_tick = self.tick + 1  # running through a platform it does not call at

    def _leave(self, mv: Movement, track: Track) -> None:
        """It has run off the area through a portal: it gives everything back and its work for the shift is over.

        Its `position` becomes the portal's node id, the one place a Movement names a node rather than an element,
        because where a train went is the only useful thing left to say about it.
        """
        self.il.release_train(mv.train.id)
        mv.position = mv.behind = track.other(mv.behind)
        mv.phase, mv.next_tick, mv.waiting = "done", None, False
        mv.path, mv.idx, mv.authority = [], 0, 0
        mv.crew_since = mv.crew_back = None
        self.log.emit(
            self.tick,
            "EXIT",
            train=mv.train.id,
            element=track.id,
            node=mv.position,
            booked=getattr(mv.trip, "exit_tick", None),
        )

    def _berth(self, mv: Movement, platform: Platform) -> None:
        entered = self.net.element(mv.path[mv.idx - 1])
        mv.side_in = entered.side if isinstance(entered, Move) else None
        mv.path, mv.idx, mv.authority, mv.waiting = [], 0, 0, False
        self._berthed(mv, platform)

    def _berthed(self, mv: Movement, platform: Platform) -> None:
        """It has come to a stand in `platform` for the call at the head of its plan."""
        if mv.plan:
            mv.at, mv.at_call = mv.plan.pop(0), mv.plan_calls.pop(0)
        if mv.at is not None and mv.at.arrive is not None and mv.at_call is not None:
            mv.worst_late = max(mv.worst_late, self.tick - mv.at.arrive)
        mv.visits.append(Visit(platform.id, self.tick))
        mv.phase = "berthed" if mv.plan or mv.trip.exit_node else "done"
        mv.behind, mv.arrived, mv.next_tick = platform.node, self.tick, None
        call = mv.at_call
        booked = mv.trip.calls[call] if call is not None else None
        detail = {"booked_platform": booked.platform} if booked is not None and booked.platform != platform.id else {}
        self.log.emit(
            self.tick,
            "ARRIVE",
            train=mv.train.id,
            element=platform.id,
            node=platform.node,
            call=call,
            booked=booked.arrive if booked else None,
            detail=detail,
        )
        if call is not None:
            mv.made.add(call)
            self._first_arrival.setdefault((mv.train.id, platform.node), self.tick)
            self._call_arrival.setdefault((mv.train.id, call), self.tick)
        if mv.phase == "done":
            mv.crew_since = mv.crew_back = None
        elif mv.relieve_next and platform.crew_relief:
            self._start_relief(mv, platform)

    def _entered(self, mv: Movement) -> None:
        self.log.emit(
            self.tick,
            "ENTER",
            train=mv.train.id,
            element=mv.position,
            node=mv.trip.entry_node,
            booked=mv.trip.entry_tick,
        )

    def _departed(self, mv: Movement) -> None:
        platform = self.net.element(mv.position)
        call = mv.at_call
        booked = mv.trip.calls[call].depart if call is not None else None
        self.log.emit(
            self.tick, "DEPART", train=mv.train.id, element=platform.id, node=platform.node, call=call, booked=booked
        )
        if call is None:
            return
        for i in self._links.get(mv.train.id, ()):
            link = self.connections[i]
            fc, cc = self._link_calls[i]
            if link.node != platform.node or i in self._resolved or (cc is not None and call != cc):
                continue
            self._resolved.add(i)
            arrived = (
                self._call_arrival.get((link.feeder, fc))
                if fc is not None
                else self._first_arrival.get((link.feeder, link.node))
            )
            made = arrived is not None and arrived + link.min_transfer_min <= self.tick
            self.log.emit(
                self.tick,
                "CONNECTION",
                train=mv.train.id,
                node=link.node,
                call=call,
                detail={"feeder": link.feeder, "connector": link.connector, "transfers": link.transfers, "made": made},
            )

    def _passed(self, mv: Movement, platform: Platform) -> None:
        call = next(
            (
                c
                for c in sorted(mv.missed)
                if c not in mv.passed and self._node(mv.trip.calls[c].platform) == platform.node
            ),
            None,
        )
        if call is not None:
            mv.passed.add(call)
        booked = mv.trip.calls[call].arrive if call is not None else None
        self.log.emit(
            self.tick, "PASS", train=mv.train.id, element=platform.id, node=platform.node, call=call, booked=booked
        )
