"""The v0.3 engine: trains run under automatic route setting, and the agent decides every conflict (§5).

Time is in minutes and advances only in `advance()`. A train is given authority one section at a time, as real
block signalling works: the moves up to the next track or platform, and on a track only its entry block. It then
runs on block by block, releasing behind it, and stops at a signal when the next section is not free.

ARS extends a train's authority as soon as nothing else wants that section. When two or more trains want the same
section within the next 15 minutes it raises a conflict instead of choosing, and `advance()` returns so the agent
can answer with `order`, `hold` or `platform`.

`Movement.behind` is the node the train last passed, so on a track it is always the end the train entered from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, inf
from typing import Literal, NamedTuple

from .interlocking import Interlocking, Refusal, Resource, claims, walk
from .model import Call, Move, Network, Platform, Side, Track, TrainSpec, Trip
from .routing import find_route, portal_track
from .rules import NetRule

IDLE_MIN = 30  # §5.2 item 7: look in at least this often
LOOKAHEAD_MIN = 15  # §5.1: contention is raised this far ahead, not only for trains ready this minute
TURNAROUND_MIN = 8  # §2.3: a booked stop at least this long is a turnaround, so the driver can change ends
Phase = Literal["offstage", "berthed", "running", "done"]
Key = Resource | tuple[str, str]  # a claimed resource, or a pair of moves whose paths cross


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
    kind: Literal["CONFLICT", "UNROUTABLE", "BLOCKED", "IDLE", "SHIFT_END"]
    conflicts: tuple[Conflict, ...] = ()
    trains: tuple[str, ...] = ()


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
    """One train's runtime state. `plan` is the calls it still has to make, so the agent's actions edit a list."""

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
            due = self.trip.entry_tick
        elif self.phase == "berthed" and self.working:
            due = max(self.at.depart or 0, (self.arrived or 0) + self.at.min_dwell)
        else:
            return None
        return max(due, self.held_until or 0)

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


class Engine:
    def __init__(self, net: Network, trips: list[Trip], shift_min: int):
        self.net = net
        self.il = Interlocking(net)
        self.tick = 0
        self.shift_min = shift_min
        self.orders: dict[Key, list[str]] = {}
        self.stranded: set[str] = set()
        self.reported_blocked: tuple[str, ...] | None = None
        self.trains: dict[str, Movement] = {}
        for trip in sorted(trips, key=lambda t: t.train.id):
            if trip.train.id in self.trains:
                raise ValueError(f"two trips for train {trip.train.id}: a unit's whole day is one trip")
            self.trains[trip.train.id] = self._place(trip)

    def _place(self, trip: Trip) -> Movement:
        plan = list(trip.calls)
        if trip.exit_node is not None and portal_track(self.net, trip.exit_node, inbound=False) is None:
            raise ValueError(f"{trip.train.id}: no track leads out of the network to {trip.exit_node}")
        if trip.entry_node is not None:
            entry = portal_track(self.net, trip.entry_node, inbound=True)
            if entry is None:
                raise ValueError(f"{trip.train.id}: no track leads into the network from {trip.entry_node}")
            return Movement(trip, plan, entry, trip.entry_node)
        start = self.net.element(plan[0].platform)
        if not isinstance(start, Platform):
            raise ValueError(f"{trip.train.id}: {plan[0].platform} is not a platform it can start from")
        if start.length_m < trip.train.length_m:  # it would stand there with no route it could ever be given
            raise ValueError(f"{trip.train.id} is {trip.train.length_m} m and cannot stand in {start.id}")
        if blocker := self.il.occupy(trip.train.id, start.id):
            raise ValueError(f"{trip.train.id} and {blocker} both start in {start.id}")
        visits = [Visit(start.id, trip.entry_tick)]
        return Movement(
            trip, plan[1:], start.id, start.node, "berthed", at=plan[0], arrived=trip.entry_tick, visits=visits
        )

    # ------------------------------------------------------------------ actions

    def order(self, key: Key, train_ids: list[str]) -> Refusal | None:
        """Say who takes a contested section first. Trains left out keep timetable order behind those listed."""
        if self.tick >= self.shift_min:
            return Refusal(NetRule.SHIFT_ENDED, "the shift is over")
        if unknown := [t for t in train_ids if t not in self.trains]:
            return Refusal(NetRule.UNKNOWN_TRAIN, f"no train {unknown[0]}")
        self.orders[key] = list(train_ids)
        return None

    def hold(self, train_id: str, until_tick: int) -> Refusal | None:
        if self.tick >= self.shift_min:
            return Refusal(NetRule.SHIFT_ENDED, "the shift is over")
        mv = self.trains.get(train_id)
        if mv is None:
            return Refusal(NetRule.UNKNOWN_TRAIN, f"no train {train_id}")
        if mv.phase in {"running", "done"}:
            return Refusal(NetRule.ALREADY_RUNNING, f"{train_id} is {mv.phase}")
        if until_tick <= self.tick:
            return Refusal(NetRule.HOLD_PAST, f"t={until_tick} is not in the future")
        mv.held_until = until_tick
        return None

    def platform(self, train_id: str, platform_id: str) -> Refusal | None:
        """Re-platform the call the train is next heading to."""
        if self.tick >= self.shift_min:
            return Refusal(NetRule.SHIFT_ENDED, "the shift is over")
        mv = self.trains.get(train_id)
        if mv is None:
            return Refusal(NetRule.UNKNOWN_TRAIN, f"no train {train_id}")
        if mv.phase in {"running", "done"} or not mv.plan:
            return Refusal(NetRule.ALREADY_RUNNING, f"{train_id} is {mv.phase}")
        if platform_id == mv.position:
            return Refusal(NetRule.NO_ROUTE, f"{train_id} already stands in {platform_id}")
        if not isinstance(self.net.element(platform_id), Platform):
            return Refusal(NetRule.NO_ROUTE, f"{platform_id} is not a platform")
        if find_route(self.net, mv.position, mv.facing(self.net), platform_id, mv.train, self._side_in(mv)) is None:
            return Refusal(NetRule.NO_ROUTE, f"no route from {mv.position} to {platform_id} for {train_id}")
        mv.plan[0] = mv.plan[0].model_copy(update={"platform": platform_id})
        self.stranded.discard(train_id)
        return None

    # ------------------------------------------------------------------ time

    def advance(self) -> Decision:
        """Run until the agent is needed, or the shift ends."""
        while self.tick < self.shift_min:
            if decision := self._set_routes():
                return decision
            nxt = self._next_event()
            if nxt is None:
                if decision := self._blocked():
                    return decision
                break
            if nxt - self.tick > IDLE_MIN:
                self.tick = min(self.tick + IDLE_MIN, self.shift_min)
                if self.tick >= self.shift_min:
                    break
                return Decision(self.tick, "IDLE")
            self.tick = min(nxt, self.shift_min)
            self._run_transitions()
        self.tick = self.shift_min
        return Decision(self.tick, "SHIFT_END")

    def _next_event(self) -> int | None:
        times = [mv.next_tick for mv in self.trains.values() if mv.next_tick is not None]
        times += [t for mv in self.trains.values() if (t := mv.ready_at()) is not None and t > self.tick]
        return min(times) if times else None

    def _blocked(self) -> Decision | None:
        """Nothing can move and nothing is due: the trains still out there are deadlocked (§5.2 item 6).

        Reported once per set of stuck trains; if the agent cannot free them, the shift simply runs out.
        """
        stuck = tuple(sorted(mv.train.id for mv in self.trains.values() if mv.waiting or mv.ready_at() is not None))
        if not stuck or stuck == self.reported_blocked:
            return None
        self.reported_blocked = stuck
        return Decision(self.tick, "BLOCKED", trains=stuck)

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
        return find_route(self.net, mv.position, mv.facing(self.net), target, mv.train, self._side_in(mv))

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

    def _advance_block(self, mv: Movement, track: Track, restarting: bool) -> None:
        """Move a train up one block inside the track it already holds, or stand it at the signal."""
        if self.il.step_block(mv.train.id, track.id, track.travel_from(mv.behind), mv.block) is not None:
            mv.waiting, mv.next_tick = True, None  # the block ahead is taken: stand at the signal
            return
        mv.block, mv.waiting = mv.block + 1, False
        restart = mv.train.restart_min if restarting else 0  # heavy trains take longer to get going again
        mv.next_tick = self.tick + restart + block_minutes(track, mv.train, mv.block)

    def _resume(self) -> None:
        """A train held by the one ahead in its own section moves up as soon as that block clears.

        Setting a route past it instead would let a following train lock the junction its leader still needs,
        which deadlocks the pair: the leader cannot clear and the follower cannot pass it.
        """
        for mv in sorted(self.trains.values(), key=lambda m: m.train.id):
            if mv.waiting and self._blocks_left(mv):
                self._advance_block(mv, self.net.element(mv.position), restarting=True)

    def _set_routes(self) -> Decision | None:
        """One ARS pass: extend authority where nothing competes, and hand real contention to the agent."""
        self._resume()
        if fresh := self._unroutable():
            return Decision(self.tick, "UNROUTABLE", trains=fresh)
        soon = self._wanting(LOOKAHEAD_MIN)
        if conflicts := self._conflicts(soon):
            names = tuple(sorted({t for c in conflicts for t in c.trains}))
            return Decision(self.tick, "CONFLICT", conflicts, names)
        now = {t: s for t, s in self._wanting(0).items() if t in soon}
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
        if self.il.lock(mv.train, mv.behind, segment) is not None:
            mv.waiting = mv.phase == "running"
            return False
        mv.authority = mv.idx + len(segment) - 1
        if mv.phase != "running":
            if mv.visits:
                mv.visits[-1] = mv.visits[-1]._replace(departed=self.tick)
            mv.path, mv.idx, mv.phase, mv.held_until = self._journey(mv), 0, "running", None
            mv.authority = len(segment) - 1
            element = self.net.element(mv.position)
            start = block_minutes(element, mv.train, 0) if isinstance(element, Track) else 1
            mv.next_tick = self.tick + start
        elif mv.waiting:
            mv.waiting = False
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
            mv.waiting, mv.next_tick = True, None
            return
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
            mv.next_tick = self.tick + block_minutes(element, mv.train, 0)
        elif isinstance(element, Move):
            mv.next_tick = self.tick + move_minutes(element, mv.train)
        elif mv.idx == len(mv.path) - 1:
            self._berth(mv, element)
        else:
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

    def _berth(self, mv: Movement, platform: Platform) -> None:
        entered = self.net.element(mv.path[mv.idx - 1])
        mv.at = mv.plan.pop(0) if mv.plan else mv.at
        if mv.at is not None and mv.at.arrive is not None:
            mv.worst_late = max(mv.worst_late, self.tick - mv.at.arrive)
        mv.visits.append(Visit(platform.id, self.tick))
        mv.phase = "berthed" if mv.plan or mv.trip.exit_node else "done"
        mv.behind, mv.arrived, mv.next_tick = platform.node, self.tick, None
        mv.side_in = entered.side if isinstance(entered, Move) else None
        mv.path, mv.idx, mv.authority, mv.waiting = [], 0, 0, False

    # ------------------------------------------------------------------ reading

    def state(self) -> list[Row]:
        """One row per train: id, phase, where it is, how late. Sorted, so runs can be compared."""
        return sorted(Row(mv.train.id, mv.phase, mv.position, mv.lateness(self.tick)) for mv in self.trains.values())
