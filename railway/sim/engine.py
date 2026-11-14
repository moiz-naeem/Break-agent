from __future__ import annotations

import copy

from pydantic import ValidationError

from .constants import (
    CREW_LIMIT,
    CREW_WARNING,
    GRIDLOCK_COST,
    RELIEF_TICKS,
    RESTART_TICKS,
    TIME_CAP,
    TIMEOUT_BLOCK,
    TIMEOUT_COST,
    TRAIN_TYPES,
    TURNAROUND,
    UNPLANNED_STOP_COST,
)
from .cost import (
    CostBreakdown,
    freight_cost,
    missed_call_cost,
    pax_cost,
    platform_change_cost,
    stranded_cost,
    unfinished_cost,
    update_total,
)
from .events import due_events, overlaps_window
from .graph import AMBIGUOUS, Graph
from .model import (
    Action,
    ActionAdapter,
    Authorize,
    BreakdownEvent,
    CancelAuthority,
    Hold,
    LateEntryEvent,
    PossessionEvent,
    RelieveCrew,
    Scenario,
    SpeedRestrictionEvent,
)
from .observe import (
    ActionResult,
    AdvanceResult,
    Arrival,
    Authority,
    Change,
    Reason,
    State,
    StaticInfo,
    StopView,
    TrainState,
)
from .rules import RuleId
from .timing import project, travel_ticks

PAX_TYPES = {"express", "local"}
COUNTERS = ("misroutes", "wrong_terminals", "missed_calls", "missed_connections", "unfinished_trains")


def _copy_train(t: dict) -> dict:
    copied = {
        k: copy.deepcopy(v) if isinstance(v, (dict, list, set, tuple)) else v
        for k, v in t.items()
        if k not in ("arrivals", "auth")
    }
    copied["arrivals"] = list(t["arrivals"])  # append-only; the arrival records themselves are never mutated
    copied["auth"] = t["auth"] and {**t["auth"], "path": list(t["auth"]["path"])}  # atomic but for its path
    return copied


class Simulator:
    def __init__(self, scenario: Scenario):
        self._scenario = scenario
        self.tick = 0
        self.done = False
        self._last_dp = 0
        self._cost = CostBreakdown()
        self._cost_finalized = False
        self._trains = {}
        self._holders = {}
        self._releases = {}  # terminal track -> (tick it frees, train that holds it)
        self._entered = set()
        self._crew_enabled = "F10" in scenario.factors
        self._events = {event.id: event for event in scenario.events}
        self._revealed_events = set()
        self._restrictions = []
        self._possessions = []
        self._pending_changes = []
        self._gridlock_charged = set()
        self._pending_reasons = []
        self.counters = dict.fromkeys(COUNTERS, 0)
        self._rotation_predecessor = {t.next_service: t.id for t in scenario.trains if t.next_service}
        self._graph = Graph(scenario.network)
        self._stations = {s.id: s for s in scenario.network.stations}
        self._section_by_id = {s.id: s for s in scenario.network.sections}
        self._train_defs = {t.id: t for t in scenario.trains}
        for t in scenario.trains:
            self._trains[t.id] = {
                "id": t.id,
                "type": t.type,
                "pax": t.pax,
                "phase": "not_entered",
                "station": None,
                "track": None,
                "section": None,
                "section_from": None,
                "section_enter": None,
                "section_exit": None,
                "came_by": None,
                "position": t.start,
                "auth": None,
                "entry_tick": t.entry_tick,
                "original_entry_tick": t.entry_tick,
                "ready_tick": t.entry_tick,
                "ready_reported_tick": None,
                "hold_until": None,
                "dwell_until": None,
                "stop_idx": 0,
                "served": [],
                "arrivals": [],
                "departures": {},
                "finished_tick": None,
                "outcome": None,
                "off_route": False,
                "entry_blocked": False,
                "restart_pending": False,
                "crew_base": t.crew_used_at_entry,
                "crew_since_tick": t.entry_tick,
                "crew_warning_fired": False,
                "crew_blocked": False,
                "relief_until": None,
                "breakdown_until": None,
                "blocked_until": None,
                "waits_for": None,
                "active_breakdown": None,
            }
        self._gridlock = []
        for tid, predecessor in self._rotation_predecessor.items():
            if "F6" in scenario.factors:
                self._trains[tid]["waits_for"] = predecessor
        for tid in sorted(self._trains):
            t = self._trains[tid]
            if t["entry_tick"] == 0 and not t["waits_for"]:
                self._try_enter(t, announce=False)
        self._process_reveals(0, initial=True)

    # -- network helpers ---------------------------------------------------------
    def _is_place(self, node: str) -> bool:
        return node in self._stations or node in self._graph.terminals

    def _tracked(self, node: str) -> bool:
        """A node whose tracks are resources: every station and every terminal with tracks."""
        return node in self._stations or (node in self._graph.terminals and not self._graph.unlimited(node))

    def _tracks_at(self, node: str) -> dict:
        return self._graph.tracks.get(node, {})

    def _fits(self, tid: str, track) -> bool:
        kind = self._trains[tid]["type"]
        return TRAIN_TYPES[kind]["length_m"] <= track.length_m and not (track.kind == "goods" and kind in PAX_TYPES)

    def _target(self, t) -> str:
        """Where the train is heading next: its next unserved scheduled stop, else its destination."""
        definition = self._train_defs[t["id"]]
        if t["stop_idx"] < len(definition.stops):
            return definition.stops[t["stop_idx"]].station
        return definition.end

    def _heading(self, t) -> tuple[str, str | None]:
        """(node, arriving section) where the train will stand once its current authority is used up."""
        if t["auth"]:
            sections = [r for r in t["auth"]["path"] if r in self._section_by_id]
            came = sections[-1] if sections else (t["section"] or t["came_by"])
            return t["auth"]["dest_station"], came
        return t["station"] or t["position"], t["came_by"]

    def _route(self, start: str, came: str | None, dest: str, track: str, via=(), strict=True):
        """Resource path start -> dest (sections, intermediate mains, the destination track), None if dest can't
        be reached without reversing, or AMBIGUOUS when two shortest routes exist and `via` doesn't pick one."""
        hops = self._graph.walk(start, dest, came, via=via, strict=strict)
        if hops is None or hops == AMBIGUOUS:
            return hops
        out = []
        for section, node in hops[:-1]:
            out.extend([section, f"{node}/main"])
        out.append(hops[-1][0])
        out.append(f"{dest}/{track}" if self._tracked(dest) else dest)
        return out

    def _process_reveals(self, tick: int, initial: bool = False):
        reasons = []
        for event in due_events(self._scenario.events, self._revealed_events, tick):
            self._revealed_events.add(event.id)
            if isinstance(event, BreakdownEvent):
                train = self._trains[event.train]
                if train["phase"] in {"not_entered", "finished"}:
                    continue
                self._apply_breakdown(train, event, tick)
                reasons.append(
                    Reason(kind="BREAKDOWN", train=event.train, detail=f"until t={train['breakdown_until']}")
                )
            elif isinstance(event, PossessionEvent):
                self._possessions.append(
                    {"id": event.id, "section": event.section, "start": event.start, "end": event.end}
                )
                self._pending_changes.extend(self._cut_authorities(event))
                if not initial:
                    reasons.append(Reason(kind="BULLETIN"))
            elif isinstance(event, SpeedRestrictionEvent):
                self._restrictions.append(
                    {
                        "id": event.id,
                        "sections": list(event.sections),
                        "speed_kmh": event.speed_kmh,
                        "start": event.start,
                        "end": event.end,
                    }
                )
                if not initial:
                    reasons.append(Reason(kind="BULLETIN"))
            elif isinstance(event, LateEntryEvent):
                self._trains[event.train]["entry_tick"] = max(event.new_entry_tick, self._rotation_floor(event.train))
                if not initial:
                    reasons.append(Reason(kind="BULLETIN"))
        return reasons

    def _rotation_floor(self, tid: str) -> int:
        if "F6" not in self._scenario.factors:
            return 0
        predecessor = self._rotation_predecessor.get(tid)
        if not predecessor:
            return 0
        before = self._trains[predecessor]
        if before["phase"] != "finished":
            return 0
        return before["finished_tick"] + TURNAROUND

    def _apply_breakdown(self, train: dict, event: BreakdownEvent, tick: int):
        train["breakdown_until"] = tick + event.duration
        train["active_breakdown"] = event.id
        train["restart_pending"] = "F5" in self._scenario.factors
        if train["phase"] == "in_section":
            train["section_exit"] += event.duration
        elif train["phase"] == "at_station":
            train["ready_tick"] = train["breakdown_until"]

    def _path_nodes(self, t, path: list[str]) -> list[tuple[str, str]]:
        """(section, node it leads to) for each section of an authority path, from the train's position."""
        here = t["station"] or t["position"]
        if t["phase"] == "in_section":
            here = t["section_from"]
        out = []
        for resource in path:
            if resource in self._section_by_id:
                here = self._graph.other_end(resource, here)
                out.append((resource, here))
        return out

    def _cut_authorities(self, event: PossessionEvent):
        changes = []
        for tid in sorted(self._trains):
            train = self._trains[tid]
            auth = train["auth"]
            if not auth or train["phase"] == "finished":
                continue
            if train["section"] == event.section:
                continue
            intervals = self._project_occupation(tid, auth["path"], max(self.tick, train["entry_tick"]))
            if not overlaps_window(intervals, event.section, event.start, event.end):
                continue
            hops = self._path_nodes(train, auth["path"])
            index = next((i for i, (sec, _) in enumerate(hops) if sec == event.section), None)
            if index is None:
                continue
            prior = hops[index - 1][1] if index else (train["station"] or train["position"])
            if prior not in self._stations:
                continue
            path = auth["path"]
            prior_main = f"{prior}/main"
            try:
                cut = path.index(prior_main) + 1
            except ValueError:
                cut = next((i for i, res in enumerate(path) if res == event.section), len(path))
            for resource in path[cut:]:
                if self._holders.get(resource) == tid:
                    self._holders.pop(resource, None)
            auth["dest_station"], auth["dest_track"] = prior, "main"
            auth["path"] = path[:cut] if prior_main in path else path[:cut] + [prior_main]
            changes.append(
                Change(
                    tick=self.tick,
                    kind="AUTHORITY_CUT",
                    train=tid,
                    text=f"AUTHORITY_CUT {tid} -> {prior}/main: possession {event.section} {event.start}-{event.end}",
                )
            )
        return changes

    def _project_occupation(self, tid: str, path: list[str], start_tick: int):
        return project(self, tid, path, start_tick)

    def _restriction_at(self, section: str, tick: int):
        for restriction in self._restrictions:
            if section in restriction["sections"] and restriction["start"] <= tick < restriction["end"]:
                return restriction["speed_kmh"]
        return None

    @classmethod
    def load(cls, scenario: Scenario) -> Simulator:
        return cls(scenario)

    @classmethod
    def from_observation(cls, static: StaticInfo, state: State) -> Simulator:
        """Rebuild the visible world without retaining any hidden scenario events."""
        scenario = Scenario.model_validate(
            {
                "schema_version": 2,
                "rules_version": static.rules_version,
                "scenario_id": "BELIEF",
                "seed": 0,
                "preset": "fixture",
                "shift_ticks": state.shift_ticks,
                "factors": list(static.factors),
                "network": static.network,
                "trains": static.trains,
                "events": [],
                "connections": static.connections,
            }
        )
        sim = cls.load(scenario)
        sim._holders = {}
        sim.tick = state.tick
        sim.done = state.done
        sim._last_dp = state.last_decision_tick
        sim._cost = state.cost.model_copy(deep=True)
        sim._cost_finalized = state.done
        sim._restrictions = [dict(item) for item in state.restrictions]
        sim._possessions = [dict(item) for item in state.possessions]
        sim._revealed_events = set()
        sim._gridlock_charged = set(state.gridlock_charged)
        sim.counters = dict.fromkeys(COUNTERS, 0) | dict(state.counters)
        for observed in state.trains:
            train = sim._trains[observed.id]
            stops = sim._train_defs[observed.id].stops
            arrived = {(a.station, a.sched) for a in observed.arrivals}
            train.update(
                phase=observed.phase,
                station=observed.station,
                track=observed.track,
                section=observed.section,
                section_from=observed.section_from,
                section_enter=observed.section_enter,
                section_exit=observed.section_exit,
                came_by=observed.came_by,
                position=(observed.station or observed.section_from or train["position"]),
                entry_tick=observed.entry_tick,
                ready_tick=observed.ready_tick,
                hold_until=observed.hold_until,
                dwell_until=observed.dwell_until,
                stop_idx=(
                    len(stops)
                    if observed.next_stop is None
                    else next((i for i, stop in enumerate(stops) if stop.station == observed.next_stop.station), 0)
                ),
                served=[i for i, stop in enumerate(stops) if (stop.station, stop.arr) in arrived],
                arrivals=[item.model_dump(mode="json") for item in observed.arrivals],
                finished_tick=observed.finished_tick,
                outcome=observed.outcome,
                off_route=observed.off_route,
                restart_pending=observed.restart_pending,
                relief_until=observed.relief_until,
                breakdown_until=observed.breakdown_until,
                blocked_until=observed.blocked_until,
                waits_for=observed.waits_for,
                ready_reported_tick=state.tick,
                crew_warning_fired=(observed.crew_remaining is not None and observed.crew_remaining <= CREW_WARNING),
                crew_blocked=observed.blocked_until is not None,
            )
            if observed.authority:
                train["auth"] = {
                    "dest_station": observed.authority.dest_station,
                    "dest_track": observed.authority.dest_track,
                    "path": list(observed.authority.path),
                }
            else:
                train["auth"] = None
            if sim._crew_enabled and observed.crew_used is not None:
                train["crew_base"] = observed.crew_used
                train["crew_since_tick"] = state.tick
            if train["auth"]:
                for resource in train["auth"]["path"]:
                    if not sim._graph.unlimited(resource):
                        sim._holders[resource] = observed.id
            if observed.section:
                sim._holders[observed.section] = observed.id
            if observed.phase == "at_station" and observed.station and sim._tracked(observed.station) and observed.track:
                sim._holders[f"{observed.station}/{observed.track}"] = observed.id
        sim._releases = {}
        for held in state.terminal_holds:
            sim._holders[held.resource] = held.train
            if held.until is not None:
                sim._releases[held.resource] = (held.until, held.train)
        sim._gridlock = list(state.gridlock)
        return sim

    # Built from the scenario at load and never mutated: clones share them and copy only the live state.
    _SHARED = (
        "_scenario",
        "_events",
        "_rotation_predecessor",
        "_graph",
        "_section_by_id",
        "_stations",
        "_train_defs",
    )

    def clone(self) -> Simulator:
        new = copy.copy(self)
        for name, value in vars(self).items():
            if name in self._SHARED:
                continue
            if name == "_trains":  # the hot path (check() clones per candidate), so copy by hand
                value = {tid: _copy_train(t) for tid, t in value.items()}
            else:
                value = copy.deepcopy(value)
            setattr(new, name, value)
        return new

    def _resource_holder(self, resource):
        return self._holders.get(resource)

    def _refuse(self, rule: RuleId, reason: str):
        return ActionResult(ok=False, rule=rule.value, message=f"REFUSED {rule.value}: {reason}")

    # -- authority ---------------------------------------------------------------
    def _authorize(self, a: Authorize):
        tid, station, track = a.train, a.station, a.track
        if tid not in self._trains:
            return self._refuse(RuleId.UNKNOWN_TRAIN, tid)
        if not self._is_place(station):
            return self._refuse(RuleId.UNKNOWN_STATION, station)
        if any(not self._is_place(v) for v in a.via):
            return self._refuse(RuleId.UNKNOWN_STATION, ", ".join(v for v in a.via if not self._is_place(v)))
        tracks = self._tracks_at(station)
        if self._tracked(station):
            if track not in tracks and not (station in self._stations and track == "main"):
                return self._refuse(RuleId.UNKNOWN_TRACK, f"{station}/{track}")
        elif track != "main":
            return self._refuse(RuleId.UNKNOWN_TRACK, f"{station}/{track}")
        t = self._trains[tid]
        if t["phase"] == "finished":
            return self._refuse(RuleId.NOT_IN_NETWORK, f"{tid} has finished")
        if t["phase"] == "not_entered" and self.tick < t["entry_tick"]:
            return self._refuse(RuleId.NOT_IN_NETWORK, f"{tid} enters at t={t['entry_tick']}")
        if t["hold_until"] is not None and self.tick < t["hold_until"]:
            return self._refuse(RuleId.HELD, f"{tid} held until t={t['hold_until']}")
        start, came = self._heading(t)
        path = self._route(start, came, station, track, a.via)
        if path is None:
            rule = RuleId.WRONG_DIRECTION if self._graph.chain else RuleId.NO_ROUTE
            return self._refuse(rule, f"{station} is not ahead of {tid}")
        if path == AMBIGUOUS:
            return self._refuse(
                RuleId.ROUTE_AMBIGUOUS, f"more than one route from {start} to {station}: name stations in via"
            )
        old_auth = t["auth"]
        replaced = set(old_auth["path"] if old_auth else [])
        for res in path:
            if self._graph.unlimited(res):
                continue
            holder = self._resource_holder(res)
            if holder and holder != tid and res not in replaced:
                if res in self._section_by_id:
                    rule = RuleId.SECTION_RESERVED
                elif res == path[-1] and station in self._graph.terminals:
                    rule = RuleId.TERMINAL_FULL
                elif res.endswith("/main"):
                    rule = RuleId.MAIN_BLOCKED
                else:
                    rule = RuleId.TRACK_OCCUPIED
                return self._refuse(rule, f"{res} reserved by {holder}")
        destres = path[-1]
        if self._tracked(station):
            loop = tracks.get(track)
            holder = self._resource_holder(destres)
            if holder and holder != tid:
                rule = RuleId.TERMINAL_FULL if station in self._graph.terminals else RuleId.TRACK_OCCUPIED
                return self._refuse(rule, f"{station}/{track} reserved by {holder}")
            length = TRAIN_TYPES[t["type"]]["length_m"]
            if loop and length > loop.length_m:
                return self._refuse(RuleId.LOOP_LEN, f"{tid} is {length} m, {station}/{track} is {loop.length_m} m")
            if loop and loop.kind == "goods" and t["type"] in PAX_TYPES:
                return self._refuse(RuleId.PLATFORM, f"{station}/{track} is a goods loop")
        if "F7" in self._scenario.factors:
            for possession in self._possessions:
                conflict = overlaps_window(
                    self._project_occupation(tid, path, max(self.tick, t["entry_tick"])),
                    possession["section"],
                    possession["start"],
                    possession["end"],
                )
                if conflict:
                    return self._refuse(
                        RuleId.POSSESSION,
                        f"{possession['section']} reserved {possession['start']}-{possession['end']}",
                    )
        # Extension releases the previous destination track and holds its main line as pass-through.
        if old_auth:
            old_path = old_auth["path"]
            old_end = old_auth["dest_station"]
            self._holders.pop(old_path[-1], None)
            path = old_path[:-1] + ([f"{old_end}/main"] if old_end in self._stations else []) + path
        for res in path:
            if not self._graph.unlimited(res):
                self._holders[res] = tid
        t["auth"] = {"dest_station": station, "dest_track": track, "path": path}
        arrival = self._project_arrival(tid, path)
        warning = self._route_warning(t)
        return ActionResult(ok=True, message=f"GRANTED {tid} -> {station}/{track} (arrive t={arrival}){warning}")

    def _route_warning(self, t) -> str:
        """After a grant: which scheduled stops, or the destination, the train can no longer reach."""
        definition = self._train_defs[t["id"]]
        stops = definition.stops
        nodes = [node for _, node in self._path_nodes(t, t["auth"]["path"])]
        lost, idx = [], t["stop_idx"]
        for node in nodes:  # stops the authority passes are served in path order; any it jumps over are lost
            later = next((i for i in range(idx, len(stops)) if stops[i].station == node), None)
            if later is not None:
                lost.extend(stops[i].station for i in range(idx, later))
                idx = later + 1
        frontier = {self._heading(t)}
        for i in range(idx, len(stops)):
            found = self._graph.reach(frontier, stops[i].station)
            if found:
                frontier = found
            else:
                lost.append(stops[i].station)
        lost_destination = definition.end in lost
        if not lost:
            return ""
        if lost_destination and not t["off_route"]:
            t["off_route"] = True
            self.counters["misroutes"] += 1
        what = [s for s in lost if s != definition.end]
        parts = []
        if what:
            parts.append(f"misses scheduled stop{'s' if len(what) > 1 else ''} {', '.join(what)}")
        if lost_destination:
            parts.append(f"can no longer reach {definition.end}")
        return f" ({RuleId.OFF_ROUTE.value}: {t['id']} {' and '.join(parts)})"

    def _project_arrival(self, tid, path):
        intervals = self._project_occupation(tid, path, max(self.tick, self._trains[tid]["entry_tick"]))
        if intervals:
            return intervals[-1][2]
        return max(self.tick, self._trains[tid]["entry_tick"])

    def _delay(self, tid: str) -> int:
        train = self._trains[tid]
        definition = self._train_defs[tid]
        stop = definition.stops[train["stop_idx"]] if train["stop_idx"] < len(definition.stops) else None
        if stop is None or train["phase"] == "finished":
            return 0
        if train["phase"] == "in_section":
            far = self._graph.other_end(train["section"], train["section_from"])
            path = [train["section"]]
            if far != stop.station:
                rest = self._route(far, train["section"], stop.station, "main", strict=False)
                if rest is None:
                    return 0
                path.extend(rest)
        else:
            start = train["station"] or train["position"]
            path = (
                []
                if start == stop.station
                else self._route(start, train["came_by"], stop.station, "main", strict=False)
            )
            if path is None:
                return 0
        projected = self._project_occupation(tid, path, max(self.tick, train["entry_tick"]))
        arrival = projected[-1][2] if projected else max(self.tick, train["entry_tick"])
        return max(0, arrival - stop.arr)

    def _apply_mutating(self, action: Action):
        try:
            a = ActionAdapter.validate_python(action)
        except ValidationError:
            return self._refuse(RuleId.BAD_ARGS, "invalid action arguments")
        if self.done:
            return self._refuse(RuleId.SHIFT_ENDED, "shift has ended")
        if isinstance(a, Authorize):
            return self._authorize(a)
        if a.train not in self._trains:
            return self._refuse(RuleId.UNKNOWN_TRAIN, a.train)
        t = self._trains[a.train]
        if isinstance(a, Hold):
            if a.until_tick <= self.tick:
                return self._refuse(RuleId.HOLD_PAST, f"t={a.until_tick} is not in the future")
            t["hold_until"] = a.until_tick
            if not t["auth"] and t["phase"] == "at_station":
                t["ready_tick"] = max(t["ready_tick"] or 0, a.until_tick)
            return ActionResult(ok=True, message=f"OK {a.train} held until t={a.until_tick}")
        if isinstance(a, CancelAuthority):
            if not t["auth"]:
                return self._refuse(RuleId.NO_AUTHORITY, f"{a.train} has no authority")
            if t["section"]:
                auth = t["auth"]
                # Keep the in-flight section and stop at its far station.
                far = self._graph.other_end(t["section"], t["section_from"])
                if far in self._graph.terminals:  # a section into a terminal: nothing left to cut
                    return ActionResult(ok=True, message=f"OK {a.train} authority cancelled")
                keep = []
                for res in auth["path"][1:]:
                    if res.endswith("/main"):
                        keep.append(res)
                        break
                for res in auth["path"]:
                    if res not in keep:
                        self._holders.pop(res, None)
                auth["path"] = keep
                auth["dest_station"] = far
                auth["dest_track"] = "main"
            else:
                for res in t["auth"]["path"]:
                    if self._holders.get(res) == a.train:
                        self._holders.pop(res, None)
                t["auth"] = None
            return ActionResult(ok=True, message=f"OK {a.train} authority cancelled")
        if isinstance(a, RelieveCrew):
            return self._relieve_crew(a.train)
        return self._refuse(RuleId.BAD_ARGS, "unknown action")

    def _relieve_crew(self, tid: str):
        if not self._crew_enabled:
            return self._refuse(RuleId.NOT_CREW_STATION, "F10 is not enabled")
        train = self._trains[tid]
        if train["phase"] != "at_station":
            return self._refuse(RuleId.NOT_STATIONARY, f"{tid} is not stationary")
        station = self._stations.get(train["station"])
        if station is None or not station.crew_station:
            return self._refuse(RuleId.NOT_CREW_STATION, f"{train['station']} is not a crew station")
        train["crew_base"] = 0
        train["crew_since_tick"] = self.tick
        train["crew_warning_fired"] = False
        train["relief_until"] = self.tick + RELIEF_TICKS
        train["ready_tick"] = train["relief_until"]
        return ActionResult(ok=True, message=f"OK {tid} crew relieved until t={train['relief_until']}")

    def _crew_used_at(self, train: dict, tick: int | None = None):
        if not self._crew_enabled:
            return None
        now = self.tick if tick is None else tick
        if train["phase"] == "not_entered":
            return train["crew_base"]
        stop_at = train["finished_tick"] if train["finished_tick"] is not None else now
        if train["blocked_until"] is not None:
            stop_at = min(stop_at, train["blocked_until"])
        return train["crew_base"] + max(0, stop_at - train["crew_since_tick"])

    def _update_rotation_successor(self, tid: str):
        if "F6" not in self._scenario.factors:
            return
        successor = self._train_defs[tid].next_service
        if not successor:
            return
        previous = self._trains[tid]
        next_train = self._trains[successor]
        next_train["waits_for"] = None
        next_train["entry_tick"] = max(
            next_train["entry_tick"], next_train["original_entry_tick"], previous["finished_tick"] + TURNAROUND
        )
        if self._crew_enabled:
            used_at_finish = self._crew_used_at(previous, previous["finished_tick"])
            next_train["crew_base"] = used_at_finish + next_train["entry_tick"] - previous["finished_tick"]
            next_train["crew_since_tick"] = next_train["entry_tick"]

    def apply(self, action: Action) -> ActionResult:
        return self._apply_mutating(action)

    def check(self, action: Action) -> ActionResult:
        return self.clone()._apply_mutating(action)

    # -- movement ------------------------------------------------------------------
    def _try_enter(self, t, announce: bool = True, ready: bool = False) -> bool:
        """Put a train on its origin. At a terminal with tracks it needs one: its rotation predecessor's (the
        same stock), else the first free fitting track; with none free it waits outside and tries next tick."""
        tid, origin = t["id"], self._train_defs[t["id"]].start
        track = "main"
        if self._tracked(origin):
            predecessor = self._rotation_predecessor.get(tid) if "F6" in self._scenario.factors else None
            mine = [
                r for r, holder in sorted(self._holders.items()) if holder == predecessor and r.startswith(f"{origin}/")
            ]
            if predecessor and mine:
                track = mine[0].split("/", 1)[1]
                self._releases.pop(mine[0], None)
            else:
                free = [
                    k
                    for k, spec in sorted(self._tracks_at(origin).items())
                    if not self._holders.get(f"{origin}/{k}") and self._fits(tid, spec)
                ]
                if not free:
                    t["entry_tick"] = self.tick + 1
                    t["ready_tick"] = t["entry_tick"]
                    if not t["entry_blocked"]:
                        t["entry_blocked"] = True
                        self._pending_reasons.append(Reason(kind="ENTRY_BLOCKED", train=tid, detail=f"{origin} full"))
                    return False
                track = free[0]
            self._holders[f"{origin}/{track}"] = tid
        t["phase"] = "at_station"
        t["station"], t["track"], t["position"] = origin, track, origin
        if ready:
            t["ready_tick"] = self.tick
        elif not announce:
            t["ready_tick"] = None  # in the network at load: tick 0 is a decision point anyway
        self._entered.add(tid)
        if announce:
            self._pending_changes.append(
                Change(tick=self.tick, kind="ENTRY", train=tid, text=f"{tid} enters at {origin}")
            )
        return True

    def _start_move(self, t):
        if t["phase"] == "in_section":
            return
        if t["phase"] == "not_entered":
            if self.tick < t["entry_tick"] or t["waits_for"]:
                return
            if not self._try_enter(t):
                return
        if not t["auth"]:
            return
        ready = max(
            t["entry_tick"],
            t["dwell_until"] or 0,
            t["relief_until"] or 0,
            t["breakdown_until"] or 0,
            t["blocked_until"] or 0,
        )
        if t["station"] in self._stations:
            stop = self._stop_at(t, t["station"])
            if stop and stop.dep is not None:
                ready = max(ready, stop.dep, t["dwell_until"] or 0)
        if self.tick < ready:
            t["ready_tick"] = ready
            return
        t["ready_tick"] = None
        if self._tracked(t["station"]):
            main = f"{t['station']}/main"
            if t["auth"]["path"] and t["auth"]["path"][0] == main:
                t["auth"]["path"].pop(0)
                self._holders.pop(main, None)
            elif t["track"]:
                self._holders.pop(f"{t['station']}/{t['track']}", None)
        path = t["auth"]["path"]
        secid = next((r for r in path if r in self._section_by_id), None)
        if not secid:
            return
        sec = self._section_by_id[secid]
        restriction = self._restriction_at(secid, self.tick)
        duration = travel_ticks(sec.length_m, sec.line_speed_kmh, TRAIN_TYPES[t["type"]]["max_speed_kmh"], restriction)
        if t["restart_pending"] and "F5" in self._scenario.factors:
            duration += RESTART_TICKS[t["type"]]
            t["restart_pending"] = False
        t["departures"].setdefault(t["station"], self.tick)
        t["phase"] = "in_section"
        t["section"] = secid
        t["section_from"] = t["station"]
        t["section_enter"] = self.tick
        t["section_exit"] = self.tick + duration
        t["station"] = None
        t["track"] = None
        self._pending_changes.append(
            Change(tick=self.tick, kind="ENTER", train=t["id"], text=f"{t['id']} enters {secid}")
        )

    def _stop_index(self, t, station):
        """Index of the next not-yet-passed scheduled stop at `station`, or None."""
        stops = self._train_defs[t["id"]].stops
        return next((i for i in range(t["stop_idx"], len(stops)) if stops[i].station == station), None)

    def _stop_at(self, t, station):
        index = self._stop_index(t, station)
        return None if index is None else self._train_defs[t["id"]].stops[index]

    def _serve(self, t, index: int, track: str):
        """Record and charge an arrival at scheduled stop `index`."""
        definition = self._train_defs[t["id"]]
        stop = definition.stops[index]
        t["stop_idx"] = index + 1
        t["served"].append(index)
        t["arrivals"].append({"station": stop.station, "sched": stop.arr, "actual": self.tick})
        late = self.tick - stop.arr
        if t["type"] in PAX_TYPES:
            alight, board = definition.flow(index)
            self._cost.pax_lateness += pax_cost(t["type"], alight, late)
            if stop.platform is not None and track != stop.platform:
                self._cost.operations += platform_change_cost(t["type"], alight, board)
        elif t["type"] == "freight" and index == len(definition.stops) - 1:
            self._cost.freight_lateness += freight_cost(late)
        t["restart_pending"] = False

    def _finish(self, t, node: str, track: str):
        """The train reached a terminal: its own (finished) or another one (wrong terminal)."""
        tid = t["id"]
        definition = self._train_defs[tid]
        t["phase"] = "finished"
        t["finished_tick"] = self.tick
        t["station"] = t["track"] = None
        if self._tracked(node):
            self._holders[f"{node}/{track}"] = tid
            successor = definition.next_service if "F6" in self._scenario.factors else None
            if node != definition.end or not successor:
                self._releases[f"{node}/{track}"] = (self.tick + TURNAROUND, tid)
        if node == definition.end:
            t["outcome"] = "arrived"
            self._update_rotation_successor(tid)
        else:
            t["outcome"] = "wrong_terminal"
            self.counters["wrong_terminals"] += 1

    def _arrive(self, t, station, track):
        tid = t["id"]
        arrived_section = t["section"]
        auth = t["auth"]
        terminal = station in self._graph.terminals
        self._pending_changes.append(
            Change(tick=self.tick, kind="ARRIVE", train=tid, text=f"{tid} at {station}/{track}")
        )
        t["phase"] = "at_station"
        t["station"], t["track"], t["position"] = station, track, station
        t["came_by"] = arrived_section
        t["section"] = t["section_from"] = t["section_enter"] = t["section_exit"] = None
        index = self._stop_index(t, station)
        stop = None if index is None else self._train_defs[tid].stops[index]
        is_end = auth and station == auth["dest_station"]
        if index is not None:
            self._serve(t, index, track if is_end else "main")
        # Release the section just left; authority path holds the remaining reserved resources.
        if arrived_section:
            self._holders.pop(arrived_section, None)
            if auth and auth["path"] and auth["path"][0] == arrived_section:
                auth["path"].pop(0)
        if is_end:
            occupied_resource = f"{station}/{track}" if self._tracked(station) else None
            for res in auth["path"]:
                if res != occupied_resource and self._holders.get(res) == tid:
                    self._holders.pop(res, None)
            t["auth"] = None
            if terminal:
                self._finish(t, station, track)
            elif stop:
                t["dwell_until"] = max(self.tick + stop.min_dwell, stop.dep or self.tick)
                t["ready_tick"] = t["dwell_until"]
            else:
                t["ready_tick"] = self.tick
                if "F5" in self._scenario.factors:
                    t["restart_pending"] = True
                    self._cost.operations += UNPLANNED_STOP_COST[t["type"]]
            return "AUTHORITY_END"
        # Consume the intermediate main resource. It remains occupied while stopped.
        main = f"{station}/main"
        if auth and main in auth["path"]:
            while auth["path"] and auth["path"][0] != main:
                auth["path"].pop(0)
            if auth["path"] and not stop:
                self._holders.pop(auth["path"].pop(0), None)
            elif auth["path"] and stop:
                t["track"] = "main"
                t["dwell_until"] = max(self.tick + stop.min_dwell, stop.dep or self.tick)
        return None

    def _process_possession_windows(self, tick: int):
        for possession in self._possessions:
            if possession["start"] != tick:
                continue
            exits = [
                train["section_exit"]
                for train in self._trains.values()
                if train["phase"] == "in_section" and train["section"] == possession["section"]
            ]
            if exits:
                possession["start"] = max(exits)

    def _process_releases(self, tick: int):
        for resource in sorted(self._releases):
            until, tid = self._releases[resource]
            if until <= tick:
                del self._releases[resource]
                if self._holders.get(resource) == tid:
                    self._holders.pop(resource, None)

    def _process_timers(self, tick: int):
        reasons = []
        for tid in sorted(self._trains):
            train = self._trains[tid]
            if train["breakdown_until"] == tick:
                train["breakdown_until"] = None
                train["active_breakdown"] = None
                if train["phase"] == "in_section" and train["restart_pending"] and "F5" in self._scenario.factors:
                    train["section_exit"] += RESTART_TICKS[train["type"]]
                    train["restart_pending"] = False
                if train["phase"] == "at_station" and not train["auth"]:
                    reasons.append(Reason(kind="READY", train=tid))
                    train["ready_reported_tick"] = tick
            if train["blocked_until"] == tick:
                train["blocked_until"] = None
                if train["crew_blocked"]:
                    train["crew_base"] = 0
                    train["crew_since_tick"] = tick
                    train["crew_blocked"] = False
                    train["crew_warning_fired"] = False
                if train["phase"] == "at_station" and not train["auth"]:
                    reasons.append(Reason(kind="READY", train=tid))
                    train["ready_reported_tick"] = tick
            if train["relief_until"] == tick:
                train["relief_until"] = None
                if train["phase"] == "at_station" and not train["auth"]:
                    reasons.append(Reason(kind="READY", train=tid))
                    train["ready_reported_tick"] = tick
            if (
                train["phase"] == "at_station"
                and not train["auth"]
                and train["ready_tick"] == tick
                and train["ready_reported_tick"] != tick
            ):
                reasons.append(Reason(kind="READY", train=tid))
                train["ready_reported_tick"] = tick
            if not self._crew_enabled or train["phase"] in {"not_entered", "finished"}:
                continue
            used = self._crew_used_at(train, tick)
            if used is None:
                continue
            remaining = CREW_LIMIT - used
            if remaining <= CREW_WARNING and not train["crew_warning_fired"]:
                train["crew_warning_fired"] = True
                reasons.append(Reason(kind="CREW_WARNING", train=tid, detail=f"crew_remaining={remaining}"))
            if remaining <= 0 and not train["blocked_until"]:
                train["crew_warning_fired"] = True
                train["crew_blocked"] = True
                train["blocked_until"] = tick + TIMEOUT_BLOCK
                train["restart_pending"] = "F5" in self._scenario.factors
                if train["phase"] == "in_section":
                    train["section_exit"] += TIMEOUT_BLOCK
                self._cost.crew_timeout += TIMEOUT_COST
                reasons.append(Reason(kind="CREW_TIMEOUT", train=tid))
        return reasons

    # -- gridlock ----------------------------------------------------------------
    def _wait_graph(self):
        """For each train standing at a station or tracked terminal without authority: its ways forward (any
        neighbour but the one it came from), each as (section holder, [occupant of each fitting track there]).
        None means free. A train with an empty list has no way forward at all."""
        standing = {
            (other["station"], other["track"]): oid
            for oid, other in self._trains.items()
            if other["phase"] == "at_station" and other["station"] is not None
        }
        options = {}
        for tid in sorted(self._trains):
            t = self._trains[tid]
            if t["phase"] != "at_station" or t["auth"] or not self._tracked(t["station"]):
                continue
            ways = []
            for section, nxt in self._graph.adj[t["station"]]:
                if section == t["came_by"]:
                    continue
                holder = self._holders.get(section)
                holder = None if holder == tid else holder
                if not self._tracked(nxt):
                    ways.append((holder, [None]))
                    continue
                candidates = (["main"] if nxt in self._stations else []) + [
                    k for k, spec in sorted(self._tracks_at(nxt).items()) if self._fits(tid, spec)
                ]
                occupants = []
                for track in candidates:
                    occupant = self._holders.get(f"{nxt}/{track}") or standing.get((nxt, track))
                    occupants.append(None if occupant == tid else occupant)
                ways.append((holder, occupants))
            options[tid] = ways
        return options

    def _stuck(self) -> set[str]:
        """Trains that can never move again: standing without authority, and every way forward is held by a
        train that is itself stuck. Any other train (moving, under authority, finished, or standing with a way
        that is free or held by a train that can still move) eventually frees what it holds."""
        options = self._wait_graph()
        stuck = set(options)
        changed = True
        while changed:
            changed = False
            for tid in sorted(stuck):
                for holder, occupants in options[tid]:
                    if holder in stuck:
                        continue
                    if any(o is None or o not in stuck for o in occupants):
                        stuck.discard(tid)
                        changed = True
                        break
        return stuck

    def _update_gridlock(self):
        """Gridlock = the stuck trains. Each new deadlock (a group of stuck trains linked by who blocks whom,
        sharing no train with one already charged) costs GRIDLOCK_COST once; with no reversal it is permanent."""
        stuck = self._stuck()
        if not stuck:
            return []
        options = self._wait_graph()
        parent = {tid: tid for tid in stuck}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for tid in sorted(stuck):
            for holder, occupants in options[tid]:
                for other in [holder, *occupants]:
                    if other in stuck:
                        parent[find(other)] = find(tid)
        groups = {}
        for tid in sorted(stuck):
            groups.setdefault(find(tid), set()).add(tid)
        for group in sorted(groups.values(), key=sorted):
            if not self._gridlock_charged & group:
                self._cost.gridlock += GRIDLOCK_COST
            self._gridlock_charged |= group
        return sorted(stuck)

    # -- time ------------------------------------------------------------------------
    def advance(self, max_tick: int | None = None) -> AdvanceResult:
        if self.done:
            return AdvanceResult(tick=self.tick, reasons=[], changes=[], done=True)
        limit = min(max_tick if max_tick is not None else self._scenario.shift_ticks, self._scenario.shift_ticks)
        reasons, changes = [], []
        # Actions at the last returned tick take effect in this phase before time advances.
        for tid in sorted(self._trains):
            t = self._trains[tid]
            before = t["phase"]
            self._start_move(t)
            if before == "not_entered" and t["phase"] != "not_entered" and not t["auth"]:
                reasons.append(Reason(kind="READY", train=tid))
        reasons.extend(self._process_timers(self.tick))
        reasons.extend(self._pending_reasons)
        self._pending_reasons.clear()
        if reasons:
            self._last_dp = self.tick
            return AdvanceResult(tick=self.tick, reasons=reasons, changes=changes, done=False)
        while self.tick < limit:
            future = [limit, min(self._last_dp + TIME_CAP, self._scenario.shift_ticks)]
            future.extend(event.reveal_tick for event in self._scenario.events if event.id not in self._revealed_events)
            future.extend(possession["start"] for possession in self._possessions if possession["start"] > self.tick)
            future.extend(until for until, _ in self._releases.values())
            for t in self._trains.values():
                if t["phase"] == "not_entered" and not t["waits_for"]:
                    future.append(t["entry_tick"])
                elif t["phase"] == "in_section":
                    future.append(t["section_exit"])
                elif t["phase"] == "at_station" and t["auth"]:
                    future.append(
                        max(
                            t["entry_tick"],
                            t["dwell_until"] or 0,
                            t["relief_until"] or 0,
                            t["breakdown_until"] or 0,
                            t["blocked_until"] or 0,
                        )
                    )
                elif t["phase"] == "at_station" and not t["auth"] and t["ready_tick"] is not None:
                    future.append(t["ready_tick"])
                if self._crew_enabled and t["phase"] not in {"not_entered", "finished"}:
                    if not t["crew_warning_fired"]:
                        future.append(t["crew_since_tick"] + CREW_LIMIT - CREW_WARNING - t["crew_base"])
                    if not t["crew_blocked"]:
                        future.append(t["crew_since_tick"] + CREW_LIMIT - t["crew_base"])
                for timer in (t["relief_until"], t["breakdown_until"], t["blocked_until"]):
                    if timer is not None:
                        future.append(timer)
            future = [x for x in future if x > self.tick]
            self.tick = min(future) if future else limit
            reasons.extend(self._process_reveals(self.tick))
            changes.extend(self._pending_changes)
            self._pending_changes.clear()
            self._process_possession_windows(self.tick)
            self._process_releases(self.tick)
            reasons.extend(self._process_timers(self.tick))
            # Section exits and arrivals precede new entries and departures.
            for tid in sorted(self._trains):
                t = self._trains[tid]
                if t["phase"] == "in_section" and t["section_exit"] == self.tick:
                    station = self._graph.other_end(t["section"], t["section_from"])
                    target_track = (
                        t["auth"]["dest_track"] if t["auth"] and station == t["auth"]["dest_station"] else "main"
                    )
                    end = self._arrive(t, station, target_track)
                    if end:
                        reasons.append(Reason(kind=end, train=tid))
            # Entries happen after arrivals, so a rotation released at this tick is visible here.
            for tid in sorted(self._trains):
                t = self._trains[tid]
                if t["phase"] == "not_entered" and t["entry_tick"] == self.tick and not t["waits_for"]:
                    if self._try_enter(t, ready=True) and not t["auth"]:
                        reasons.append(Reason(kind="READY", train=tid))
                        t["ready_reported_tick"] = self.tick
            reasons.extend(self._process_timers(self.tick))
            for tid in sorted(self._trains):
                t = self._trains[tid]
                if t["phase"] == "at_station" and t["auth"]:
                    self._start_move(t)
            reasons.extend(self._pending_reasons)
            self._pending_reasons.clear()
            new_gridlock = self._update_gridlock()
            if new_gridlock and new_gridlock != self._gridlock:
                reasons.append(Reason(kind="GRIDLOCK"))
            self._gridlock = new_gridlock
            changes.extend(self._pending_changes)
            self._pending_changes.clear()
            if self.tick >= self._scenario.shift_ticks:
                reasons.append(Reason(kind="SHIFT_END"))
                self.done = True
                self._finalize_cost()
                break
            if self.tick >= self._last_dp + TIME_CAP:
                reasons.append(Reason(kind="TIME_CAP"))
            if reasons or self.tick >= limit:
                break
        self._last_dp = self.tick
        return AdvanceResult(tick=self.tick, reasons=reasons, changes=changes, done=self.done)

    def end_shift(self) -> AdvanceResult:
        while not self.done:
            self.advance(max_tick=self._scenario.shift_ticks)
        self._finalize_cost()
        return AdvanceResult(tick=self.tick, reasons=[Reason(kind="SHIFT_END")], changes=[], done=True)

    def _finalize_cost(self):
        """Shift end: every scheduled passenger stop not served is a missed call; every train that did not
        reach its own terminal is unfinished; F11 connections whose connector left first are missed."""
        if self._cost_finalized:
            return
        for definition in self._scenario.trains:
            t = self._trains[definition.id]
            if t["outcome"] != "arrived":
                self._cost.unfinished += unfinished_cost(definition, self._scenario.shift_ticks)
                self.counters["unfinished_trains"] += 1
            if definition.type not in PAX_TYPES:
                continue
            for index, stop in enumerate(definition.stops):
                if index in t["served"]:
                    continue
                alight, board = definition.flow(index)
                self._cost.missed_calls += missed_call_cost(definition.type, alight, board, stop.headway)
                self.counters["missed_calls"] += 1
        if "F11" in self._scenario.factors:
            for c in self._scenario.connections:
                feeder, connector = self._trains[c.feeder], self._trains[c.connector]
                left = connector["departures"].get(c.station)
                if left is None:
                    continue  # the connector never served it: its own missed call covers these passengers
                came = next((a["actual"] for a in feeder["arrivals"] if a["station"] == c.station), None)
                if came is None or left < came + c.min_transfer:
                    stop = self._stop_at({"id": c.connector, "stop_idx": 0}, c.station)
                    self._cost.missed_calls += stranded_cost(connector["type"], c.pax, stop.headway)
                    self.counters["missed_connections"] += 1
        update_total(self._cost)
        self._cost_finalized = True

    # -- views ---------------------------------------------------------------------
    def state(self) -> State:
        trains = []
        for tid in sorted(self._trains):
            t = self._trains[tid]
            td = self._train_defs[tid]
            next_stop = td.stops[t["stop_idx"]] if t["stop_idx"] < len(td.stops) else None
            auth = None
            if t["auth"]:
                auth = Authority(
                    dest_station=t["auth"]["dest_station"],
                    dest_track=t["auth"]["dest_track"],
                    path=list(t["auth"]["path"]),
                )
            trains.append(
                TrainState(
                    id=tid,
                    type=t["type"],
                    direction=td.direction,
                    origin=td.start,
                    destination=td.end,
                    pax=t["pax"],
                    phase=t["phase"],
                    station=t["station"],
                    track=t["track"],
                    section=t["section"],
                    section_from=t["section_from"],
                    section_enter=t["section_enter"],
                    section_exit=t["section_exit"],
                    came_by=t["came_by"],
                    authority=auth,
                    entry_tick=t["entry_tick"],
                    ready_tick=t["ready_tick"],
                    hold_until=t["hold_until"],
                    dwell_until=t["dwell_until"],
                    next_stop=(
                        StopView(station=next_stop.station, arr=next_stop.arr, dep=next_stop.dep) if next_stop else None
                    ),
                    arrivals=[Arrival(**a) for a in t["arrivals"]],
                    delay=self._delay(tid),
                    finished_tick=t["finished_tick"],
                    outcome=t["outcome"],
                    off_route=t["off_route"],
                    restart_pending=t["restart_pending"],
                    crew_used=self._crew_used_at(t) if self._crew_enabled else None,
                    crew_remaining=(CREW_LIMIT - self._crew_used_at(t)) if self._crew_enabled else None,
                    relief_until=t["relief_until"],
                    breakdown_until=t["breakdown_until"],
                    blocked_until=t["blocked_until"],
                    waits_for=t["waits_for"],
                )
            )
        finished = {tid for tid, t in self._trains.items() if t["phase"] == "finished"}
        holds = [
            {"resource": r, "train": h, "until": self._releases.get(r, (None,))[0]}
            for r, h in sorted(self._holders.items())
            if h in finished
        ]
        return State(
            tick=self.tick,
            shift_ticks=self._scenario.shift_ticks,
            done=self.done,
            gridlock=list(self._gridlock),
            gridlock_charged=sorted(self._gridlock_charged),
            trains=trains,
            restrictions=[dict(x) for x in self._restrictions],
            possessions=[dict(x) for x in self._possessions],
            terminal_holds=holds,
            counters=dict(self.counters),
            cost=self.cost(),
            last_decision_tick=self._last_dp,
        )

    def static_info(self) -> StaticInfo:
        return StaticInfo(
            network=self._scenario.network.model_dump(mode="json"),
            trains=[t.model_dump(mode="json") for t in self._scenario.trains],
            shift_ticks=self._scenario.shift_ticks,
            factors=list(self._scenario.factors),
            rules_version=self._scenario.rules_version,
            connections=[c.model_dump(mode="json") for c in self._scenario.connections],
        )

    def cost(self) -> CostBreakdown:
        out = self._cost.model_copy(deep=True)
        update_total(out)
        return out
