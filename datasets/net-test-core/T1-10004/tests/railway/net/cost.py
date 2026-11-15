"""The v0.3 network cost (DESIGN.md §6.6, amended by docs/design-changes-v0.3.md §3 and §6). The reward is -total.

The cost is a pure function of the timetable and the event log (railway/net/events.py): the same log always
costs the same, a replay can be audited charge by charge (`itemise`), and a log cut mid-shift costs what has
happened so far. Every charge is rounded to six decimals; totals are sums of rounded charges, rounded again.
See docs/net-cost.md for the formulas and a worked example.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .cost_constants import (
    DEFAULT_HEADWAY,
    FREIGHT_LATE_STEP,
    GRIDLOCK_COST,
    LATE_STEPS,
    MAX_HEADWAY,
    MISSED_CALL_FLAT,
    MISSED_CALL_HASSLE,
    PAX_GRACE,
    PLATFORM_CHANGE_MINUTES,
    PORT_CUTOFF_COST,
    TIMEOUT_COST,
    UNDELIVERED_FREIGHT,
    UNFINISHED_COST,
    UNPLANNED_STOP_COST,
    VALUE,
    W_FREIGHT,
)
from .events import Event, EventLog
from .model import Call, Connection, Timetable, Trip
from .trains import TYPES

Component = Literal[
    "pax_lateness", "missed_calls", "unfinished", "freight_lateness", "crew_timeout", "gridlock", "operations"
]
COMPONENTS: tuple[str, ...] = Component.__args__  # type: ignore[attr-defined]
RewardClass = Literal["express", "local", "freight", "none"]

FREIGHT_WINDOW = {"FI": 30, "FB": 60}  # docs/network-v0.3.md §3: minutes late before the +50 step
NO_REWARD_TYPES = ("ECS", "ENG")  # empty stock and engineering: only UNFINISHED, crew and gridlock reach them
TURNAROUND_MIN = 8  # docs/network-v0.3.md §2.3: a booked stop at least this long reverses direction


class CostBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pax_lateness: float = 0.0  # C1
    missed_calls: float = 0.0  # C2: unserved passenger stops and missed connections
    unfinished: float = 0.0  # C2: trains not finished; freight adds undelivered cargo and lateness at shift end
    freight_lateness: float = 0.0  # C3: lateness at delivery, port cutoff
    crew_timeout: float = 0.0  # C4
    gridlock: float = 0.0  # C5
    operations: float = 0.0  # C5: unplanned stops, platform changes
    total: float = 0.0


class Charge(BaseModel):
    """One line of the bill: which event caused it, where it goes, how much, and why."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int  # position of the event in the log
    event: Event
    train: str | None
    component: Component
    amount: float
    explanation: str


def lateness_curve(late: int) -> int:
    """f(L) = max(0, L - 2) + 5[L >= 15] + 10[L >= 30] + 20[L >= 60], per passenger."""
    return max(0, late - PAX_GRACE) + sum(step for at, step in LATE_STEPS if late >= at)


def freight_lateness_cost(late: int, window: int) -> float:
    """C3: 0.5 x late, plus 50 once it is later than its window."""
    return _r(W_FREIGHT * max(0, late) + (FREIGHT_LATE_STEP if late > window else 0))


def stranded_cost(value: float, people: int, headway: int) -> float:
    """People left behind, waiting `headway` minutes for the next service: v x people/100 x (f(H) + 10)."""
    return _r(value * people / 100 * (lateness_curve(headway) + MISSED_CALL_HASSLE))


@dataclass(frozen=True)
class TrainClass:
    code: str | None  # the §3 type code, or None if it could not be told
    reward: RewardClass
    window: int | None = None  # freight only

    @property
    def value(self) -> float:
        return VALUE.get(self.reward, 0.0)

    @property
    def passenger(self) -> bool:
        return self.reward in VALUE


def classify(trip: Trip, code: str | None = None) -> TrainClass:
    """The reward class of a trip. `code` names its type; otherwise it is the §3 type whose spec it is exactly.

    Trips carry no type code (ids like SE01 or FN01 name the line, not the type), so without `code` the class is
    read off the TrainSpec, which the generator builds from TYPES. A spec that matches no type has class "none".
    """
    if code is None:
        code = getattr(trip, "type", None) or getattr(trip.train, "type", None)
    if code is None:
        code = next((c for c, t in TYPES.items() if t.spec(trip.train.id) == trip.train), None)
    if code is None or code in NO_REWARD_TYPES:
        return TrainClass(code, "none")
    if code not in TYPES:
        raise ValueError(f"{trip.train.id}: unknown train type {code!r}")
    reward = TYPES[code].reward_class
    return TrainClass(code, reward, FREIGHT_WINDOW.get(code) if reward == "freight" else None)


class _Scenario:
    """What the cost needs from the timetable, worked out once per call."""

    def __init__(self, timetable: Timetable, types: Mapping[str, str] | None, cutoffs: Mapping[str, int] | None):
        self.tt = timetable
        self.trips = {t.train.id: t for t in timetable.trips}
        types = types or {}
        if unknown := sorted(set(types) - set(self.trips)):
            raise ValueError(f"types names unknown trains: {unknown}")
        self.classes = {i: classify(t, types.get(i)) for i, t in self.trips.items()}
        base_cutoffs = dict(getattr(timetable, "port_cutoffs", None) or {})
        for tid, t in self.trips.items():
            if (cut := getattr(t, "cutoff_tick", None)) is not None:
                base_cutoffs[tid] = cut
        base_cutoffs.update(cutoffs or {})
        self.cutoffs = base_cutoffs
        self.connections = {(c.feeder, c.connector, c.node): c for c in timetable.connections}
        self._headways: dict[tuple[str, int], tuple[int, bool]] = {}

    def node_of(self, platform: str) -> str:
        return self.tt.net.element(platform).node  # type: ignore[union-attr]

    def _corridors(self, trip: Trip) -> list[tuple[frozenset[str], frozenset[str]]]:
        """For each call of `trip`, the (earlier, onward) nodes on the same leg, stopping at turnarounds."""
        nodes = [self.node_of(c.platform) for c in trip.calls]
        out: list[tuple[frozenset[str], frozenset[str]]] = []
        for i, node in enumerate(nodes):
            seen, onward, turned = {node}, set(), False
            for j in range(i + 1, len(nodes)):
                n = nodes[j]
                if n in seen:
                    turned = True
                    break
                seen.add(n)
                onward.add(n)
                if trip.calls[j].min_dwell >= TURNAROUND_MIN:
                    turned = True
                    break
            if not turned and trip.exit_node and trip.exit_node not in seen:
                onward.add(trip.exit_node)

            seen, earlier, turned = {node}, set(), False
            for j in range(i - 1, -1, -1):
                n = nodes[j]
                if n in seen:
                    turned = True
                    break
                seen.add(n)
                earlier.add(n)
                if trip.calls[j].min_dwell >= TURNAROUND_MIN:
                    turned = True
                    break
            if not turned and trip.entry_node and trip.entry_node not in seen:
                earlier.add(trip.entry_node)

            out.append((frozenset(earlier), frozenset(onward)))
        return out

    @cached_property
    def _corridor_by_call(self) -> dict[tuple[str, int], tuple[str, frozenset[str], frozenset[str]]]:
        out: dict[tuple[str, int], tuple[str, frozenset[str], frozenset[str]]] = {}
        for tid, trip in self.trips.items():
            for i, (earlier, onward) in enumerate(self._corridors(trip)):
                out[(tid, i)] = (self.node_of(trip.calls[i].platform), earlier, onward)
        return out

    @cached_property
    def _passenger_calls(
        self,
    ) -> dict[str, list[tuple[str, int, frozenset[str], frozenset[str], int | None, int | None]]]:
        """node -> (train, call, earlier nodes, onward nodes, arrive, depart) for every passenger call."""
        out: dict[str, list] = {}
        for tid, trip in self.trips.items():
            if not self.classes[tid].passenger:
                continue
            for i, call in enumerate(trip.calls):
                node, earlier, onward = self._corridor_by_call[(tid, i)]
                out.setdefault(node, []).append((tid, i, earlier, onward, call.arrive, call.depart))
        return out

    def headway(self, train: str, call: int) -> tuple[int, bool]:
        """Minutes to the next passenger service calling at the same node and going the same way, capped at 120.

        Going the same way: sharing an onward stop/portal before the next turnaround (or, at a terminating call,
        sharing an earlier stop/portal after the previous turnaround). Returns (H, derived); H is 60 when no such
        service exists in the timetable.
        """
        key = (train, call)
        if key not in self._headways:
            self._headways[key] = self._headway(train, call)
        return self._headways[key]

    def _headway(self, train: str, call: int) -> tuple[int, bool]:
        booked = self.trips[train].calls[call]
        node, earlier, onward = self._corridor_by_call[(train, call)]
        if onward:
            t0 = booked.depart if booked.depart is not None else booked.arrive
        else:
            t0 = booked.arrive if booked.arrive is not None else booked.depart
        if t0 is None:
            return DEFAULT_HEADWAY, False
        best = None
        for tid, j, other_earlier, other_onward, arr, dep in self._passenger_calls.get(node, ()):
            if (tid, j) == (train, call):
                continue
            same_way = bool(onward & other_onward) if onward else bool(earlier & other_earlier)
            t = (dep if dep is not None else arr) if onward else (arr if arr is not None else dep)
            if same_way and t is not None and t > t0:
                best = t - t0 if best is None else min(best, t - t0)
        return (DEFAULT_HEADWAY, False) if best is None else (min(best, MAX_HEADWAY), True)


def _r(x: float) -> float:
    return round(x, 6)


def _events(events: EventLog | Iterable[Event]) -> list[Event]:
    return list(events.events if isinstance(events, EventLog) else events)


def itemise(
    timetable: Timetable,
    events: EventLog | Iterable[Event],
    *,
    types: Mapping[str, str] | None = None,
    port_cutoffs: Mapping[str, int] | None = None,
) -> list[Charge]:
    """Every charge the log incurs, in log order. Zero charges are left out.

    `types` maps train id -> §3 type code (IC, RE, S, FI, FB, ECS, ENG) where the TrainSpec alone cannot say it.
    `port_cutoffs` maps a freight train id -> the minute its ship leaves (F13); no cutoff, no 400 charge.
    """
    sc = _Scenario(timetable, types, port_cutoffs)
    log = _events(events)
    out: list[Charge] = []
    missed: set[tuple[str, int]] = set()
    delivered: set[str] = set()
    last_departure_late: dict[str, int] = {}

    def charge(
        i: int, e: Event, component: Component, amount: float, why: str, *, train: str | None = None
    ) -> None:
        rounded = _r(amount)
        if rounded:
            out.append(
                Charge(
                    index=i,
                    event=e,
                    train=e.train if e.train is not None else train,
                    component=component,
                    amount=rounded,
                    explanation=why,
                )
            )

    def trip_of(e: Event) -> tuple[Trip, TrainClass]:
        if e.train not in sc.trips:
            raise ValueError(f"event {e.kind} at t={e.tick} names train {e.train!r}, which is not in the timetable")
        return sc.trips[e.train], sc.classes[e.train]

    def call_of(trip: Trip, e: Event) -> Call:
        assert e.call is not None
        if e.call >= len(trip.calls):
            raise ValueError(
                f"event {e.kind} at t={e.tick} names call {e.call} of {e.train}, which has {len(trip.calls)} calls"
            )
        return trip.calls[e.call]

    def deliver(i: int, e: Event, late: int, cls: TrainClass, where: str) -> None:
        assert e.train is not None
        delivered.add(e.train)
        charge(i, e, "freight_lateness", freight_lateness_cost(late, cls.window or 0),
               f"{e.train} delivered {where} {late} min late: 0.5 x {max(0, late)}"
               + (f" + {FREIGHT_LATE_STEP} (window {cls.window})" if late > (cls.window or 0) else ""))
        cutoff = sc.cutoffs.get(e.train)
        if (cutoff is not None and e.tick > cutoff) or e.detail.get("port_cutoff_missed") is True:
            charge(i, e, "freight_lateness", PORT_CUTOFF_COST, f"{e.train} missed its port cutoff at t={cutoff}")

    for i, e in enumerate(log):
        k = e.kind
        if k == "GRIDLOCK":
            trains = ", ".join(e.detail.get("trains", []))
            charge(i, e, "gridlock", GRIDLOCK_COST, f"gridlock at t={e.tick} ({trains})")
            continue
        if k == "CONNECTION":  # names its trains in detail; `train`, if set, is the connector
            _connection(sc, i, e, charge)
            continue
        if e.train is None or k in ("SHIFT_END", "DISRUPTION_START", "DISRUPTION_END", "ACTION"):
            continue
        trip, cls = trip_of(e)
        if k == "ARRIVE" and e.call is not None:
            booked = call_of(trip, e)
            if cls.passenger:
                due = e.booked if e.booked is not None else booked.arrive
                if due is not None and booked.alight:
                    late = e.tick - due
                    charge(i, e, "pax_lateness", cls.value * booked.alight / 100 * lateness_curve(late),
                           f"{e.train} at {e.node} {late} min late: {cls.value} x {booked.alight}/100 x "
                           f"f({late})={lateness_curve(late)}")
                booked_plat = e.detail.get("booked_platform")
                if booked_plat and booked_plat != e.element:
                    people = booked.alight + booked.board
                    charge(i, e, "operations", cls.value * people / 100 * PLATFORM_CHANGE_MINUTES,
                           f"{e.train} at {e.element} instead of {booked_plat}: "
                           f"{cls.value} x {people}/100 x {PLATFORM_CHANGE_MINUTES}")
            elif cls.reward == "freight" and not trip.exit_node and e.call == len(trip.calls) - 1:
                due = e.booked if e.booked is not None else booked.arrive
                if due is not None and e.train not in delivered:
                    deliver(i, e, e.tick - due, cls, f"at {e.node}")
        elif k == "DEPART" and e.call == len(trip.calls) - 1 and cls.reward == "freight":
            due = e.booked if e.booked is not None else trip.calls[-1].depart
            if due is not None:
                last_departure_late[e.train] = e.tick - due
        elif k == "EXIT" and cls.reward == "freight" and e.train not in delivered:
            if e.booked is not None:
                late = e.tick - e.booked
            elif e.train in last_departure_late:  # no booked exit time: lateness leaving its last call
                late = last_departure_late[e.train]
            else:  # neither: the booked departure from its last call (it skipped that call)
                late = e.tick - (trip.calls[-1].depart if trip.calls[-1].depart is not None else e.tick)
            deliver(i, e, late, cls, f"out through {e.node}")
        elif k == "CALL_MISSED" and e.call is not None and cls.passenger and (e.train, e.call) not in missed:
            booked = call_of(trip, e)
            missed.add((e.train, e.call))
            h, derived = sc.headway(e.train, e.call)
            people = booked.alight + booked.board
            why = e.detail.get("why", "?")
            charge(i, e, "missed_calls", MISSED_CALL_FLAT + stranded_cost(cls.value, people, h),
                   f"{e.train} missed call {e.call} at {e.node} ({why}): 500 + {cls.value} x {people}/100 x "
                   f"(f({h})={lateness_curve(h)} + 10), H {'from timetable' if derived else 'default'}")
        elif k == "UNFINISHED":
            charge(i, e, "unfinished", UNFINISHED_COST, f"{e.train} did not finish")
            if cls.reward == "freight" and e.train not in delivered:
                if e.booked is not None:
                    late = e.tick - e.booked
                else:
                    last = trip.calls[-1]
                    ends = (last.depart, last.arrive) if trip.exit_node else (last.arrive, last.depart)
                    late = e.tick - next((t for t in ends if t is not None), 0)
                charge(i, e, "unfinished", UNDELIVERED_FREIGHT + freight_lateness_cost(late, cls.window or 0),
                       f"{e.train} undelivered: 1500 + freight lateness at shift end ({late} min, window "
                       f"{cls.window})")
                cutoff = sc.cutoffs.get(e.train)
                if (cutoff is not None and e.tick >= cutoff) or e.detail.get("port_cutoff_missed") is True:
                    charge(
                        i, e, "freight_lateness", PORT_CUTOFF_COST, f"{e.train} missed its port cutoff at t={cutoff}"
                    )
        elif k == "SIGNAL_STOP" and cls.reward != "none":
            group = "passenger" if cls.passenger else "freight"
            charge(i, e, "operations", UNPLANNED_STOP_COST[group], f"{e.train} unplanned stop at {e.element}")
        elif k == "CREW_TIMEOUT":
            charge(i, e, "crew_timeout", TIMEOUT_COST, f"{e.train} crew timed out at {e.element}")
    return out


def _connector_call(
    sc: _Scenario, e: Event, feeder: str | None, connector: str, link: Connection | None
) -> int | None:
    trip = sc.trips[connector]
    if e.call is not None:
        if e.call >= len(trip.calls):
            raise ValueError(
                f"CONNECTION at t={e.tick} names call {e.call} of {connector}, which has {len(trip.calls)} calls"
            )
        return e.call
    candidates = [j for j, c in enumerate(trip.calls) if sc.node_of(c.platform) == e.node]
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    departing = [j for j in candidates if trip.calls[j].depart is not None] or candidates
    if e.booked is not None:
        for j in departing:
            if trip.calls[j].depart == e.booked:
                return j
    if feeder in sc.trips:
        min_tr = link.min_transfer_min if link else 5
        matched = [
            j
            for j in departing
            for fc in sc.trips[feeder].calls
            if sc.node_of(fc.platform) == e.node
            and fc.arrive is not None
            and trip.calls[j].depart is not None
            and min_tr <= trip.calls[j].depart - fc.arrive <= 30
        ]
        if len(matched) == 1:
            return matched[0]
    due = [j for j in departing if (trip.calls[j].depart or 0) <= e.tick]
    return due[-1] if due else departing[0]


def _connection(sc: _Scenario, i: int, e: Event, charge) -> None:
    """A missed connection: v x n/100 x (f(H) + 10), with v and H those of the connector at the hub."""
    if e.detail.get("made", True):
        return
    feeder, connector = e.detail.get("feeder"), e.detail.get("connector")
    if feeder is not None and feeder not in sc.trips:
        raise ValueError(f"CONNECTION at t={e.tick} names feeder {feeder!r}, which is not in the timetable")
    if connector not in sc.trips:
        raise ValueError(f"CONNECTION at t={e.tick} names connector {connector!r}, which is not in the timetable")
    link = sc.connections.get((feeder, connector, e.node))
    n = e.detail.get("transfers")
    if n is None:
        n = link.transfers if link else 0
    cls = sc.classes[connector]
    if not cls.passenger or not n:
        return
    call = _connector_call(sc, e, feeder, connector, link)
    h, derived = sc.headway(connector, call) if call is not None else (DEFAULT_HEADWAY, False)
    charge(i, e, "missed_calls", stranded_cost(cls.value, n, h),
           f"{feeder}->{connector} connection missed at {e.node}: {cls.value} x {n}/100 x "
           f"(f({h})={lateness_curve(h)} + 10), H {'from timetable' if derived else 'default'}",
           train=connector)


def summarise(charges: Iterable[Charge]) -> CostBreakdown:
    sums = dict.fromkeys(COMPONENTS, 0.0)
    for c in charges:
        sums[c.component] += c.amount
    sums = {k: _r(v) for k, v in sums.items()}
    return CostBreakdown(**sums, total=_r(sum(sums.values())))


def cost(
    timetable: Timetable,
    events: EventLog | Iterable[Event],
    *,
    types: Mapping[str, str] | None = None,
    port_cutoffs: Mapping[str, int] | None = None,
) -> CostBreakdown:
    """The cost of a shift so far. The reward is -cost(...).total."""
    return summarise(itemise(timetable, events, types=types, port_cutoffs=port_cutoffs))
