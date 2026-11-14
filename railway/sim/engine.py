from __future__ import annotations

import copy

from pydantic import ValidationError

from .constants import (
    CREW_LIMIT,
    CREW_WARNING,
    RELIEF_TICKS,
    RESTART_TICKS,
    REVERSE_COST,
    REVERSE_TICKS,
    TIME_CAP,
    TIMEOUT_BLOCK,
    TIMEOUT_COST,
    TRAIN_TYPES,
    TURNAROUND,
)
from .cost import CostBreakdown, freight_cost, pax_cost, unfinished_cost, update_total
from .events import due_events, overlaps_window
from .gridlock import find_gridlock
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
    Reverse,
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
        self._entered = set()
        self._crew_enabled = "F10" in scenario.factors
        self._events = {event.id: event for event in scenario.events}
        self._revealed_events = set()
        self._restrictions = []
        self._possessions = []
        self._pending_changes = []
        self._rotation_predecessor = {t.next_service: t.id for t in scenario.trains if t.next_service}
        for t in scenario.trains:
            origin = "YW" if t.direction == "east" else "YE"
            entered = t.entry_tick == 0
            if entered:
                self._entered.add(t.id)
            self._trains[t.id] = {
                "id": t.id,
                "type": t.type,
                "direction": t.direction,
                "pax": t.pax,
                "phase": "at_station" if entered else "not_entered",
                "station": origin if entered else None,
                "track": "main" if entered else None,
                "section": None,
                "section_enter": None,
                "section_exit": None,
                "position": origin,
                "auth": None,
                "entry_tick": t.entry_tick,
                "original_entry_tick": t.entry_tick,
                "ready_tick": t.entry_tick if not entered else None,
                "ready_reported_tick": None,
                "hold_until": None,
                "dwell_until": None,
                "stop_idx": 0,
                "arrivals": [],
                "finished_tick": None,
                "reversing_to": None,
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
        self._station_ids = ["YW", *[s.id for s in scenario.network.stations], "YE"]
        self._station_idx = {s: i for i, s in enumerate(self._station_ids)}
        self._section_by_pair = {(s.west, s.east): s for s in scenario.network.sections}
        self._section_by_id = {s.id: s for s in scenario.network.sections}
        self._stations = {s.id: s for s in scenario.network.stations}
        self._train_defs = {t.id: t for t in scenario.trains}
        self._gridlock = []
        for tid, predecessor in self._rotation_predecessor.items():
            if "F6" in scenario.factors and self._trains[predecessor]["phase"] != "finished":
                self._trains[tid]["waits_for"] = predecessor
        self._process_reveals(0, initial=True)

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

    def _cut_authorities(self, event: PossessionEvent):
        changes = []
        section = self._section_by_id[event.section]
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
            prior = section.west if train["direction"] == "east" else section.east
            path = auth["path"]
            if prior not in self._station_idx:
                continue
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
                "schema_version": 1,
                "rules_version": static.rules_version,
                "scenario_id": "BELIEF",
                "seed": 0,
                "preset": "fixture",
                "shift_ticks": state.shift_ticks,
                "factors": list(static.factors),
                "network": static.network,
                "trains": static.trains,
                "events": [],
            }
        )
        sim = cls.load(scenario)
        sim.tick = state.tick
        sim.done = state.done
        sim._last_dp = state.last_decision_tick
        sim._cost = state.cost.model_copy(deep=True)
        sim._cost_finalized = state.done
        sim._restrictions = [dict(item) for item in state.restrictions]
        sim._possessions = [dict(item) for item in state.possessions]
        sim._revealed_events = set()
        for observed in state.trains:
            train = sim._trains[observed.id]
            train.update(
                phase=observed.phase,
                station=observed.station,
                track=observed.track,
                section=observed.section,
                section_enter=observed.section_enter,
                section_exit=observed.section_exit,
                position=(observed.station or train["position"]),
                entry_tick=observed.entry_tick,
                ready_tick=observed.ready_tick,
                hold_until=observed.hold_until,
                dwell_until=observed.dwell_until,
                stop_idx=(
                    len(sim._train_defs[observed.id].stops)
                    if observed.next_stop is None
                    else next(
                        (
                            i
                            for i, stop in enumerate(sim._train_defs[observed.id].stops)
                            if stop.station == observed.next_stop.station
                        ),
                        0,
                    )
                ),
                arrivals=[item.model_dump(mode="json") for item in observed.arrivals],
                finished_tick=observed.finished_tick,
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
                    if resource not in {"YW", "YE"}:
                        sim._holders[resource] = observed.id
            if observed.section:
                sim._holders[observed.section] = observed.id
            if observed.phase == "at_station" and observed.station in sim._stations and observed.track:
                sim._holders[f"{observed.station}/{observed.track}"] = observed.id
        sim._gridlock = list(state.gridlock)
        return sim

    # Built from the scenario at load and never mutated: clones share them and copy only the live state.
    _SHARED = (
        "_scenario",
        "_events",
        "_rotation_predecessor",
        "_station_ids",
        "_station_idx",
        "_section_by_pair",
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

    def _route(self, start: str, dest: str, track: str):
        i, j = self._station_idx[start], self._station_idx[dest]
        step = 1 if j > i else -1
        out = []
        for k in range(i, j, step):
            a, b = self._station_ids[k], self._station_ids[k + step]
            section = self._section_by_pair.get((a, b)) or self._section_by_pair.get((b, a))
            out.append(section.id)
            if k + step != j and b in self._stations:
                out.append(f"{b}/main")
        out.append(f"{dest}/{track}" if dest in self._stations else dest)
        return out

    def _resource_holder(self, resource):
        return self._holders.get(resource)

    def _refuse(self, rule: RuleId, reason: str):
        return ActionResult(ok=False, rule=rule.value, message=f"REFUSED {rule.value}: {reason}")

    def _authorize(self, a: Authorize):
        tid, station, track = a.train, a.station, a.track
        if tid not in self._trains:
            return self._refuse(RuleId.UNKNOWN_TRAIN, tid)
        if station not in self._station_idx:
            return self._refuse(RuleId.UNKNOWN_STATION, station)
        if station in self._stations:
            loops = {x.id: x for x in self._stations[station].loops}
            if track != "main" and track not in loops:
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
        start = t["auth"]["dest_station"] if t["auth"] else (t["station"] or t["position"])
        direction = 1 if t["direction"] == "east" else -1
        if direction * (self._station_idx[station] - self._station_idx[start]) <= 0:
            return self._refuse(RuleId.WRONG_DIRECTION, f"{station} is not ahead of {tid}")
        path = self._route(start, station, track)
        old_auth = t["auth"]
        replaced = set(old_auth["path"] if old_auth else [])
        for res in path:
            if res in {"YW", "YE"}:
                continue
            holder = self._resource_holder(res)
            if holder and holder != tid and res not in replaced:
                if res in self._section_by_id:
                    rule = RuleId.SECTION_RESERVED
                elif res.endswith("/main"):
                    rule = RuleId.MAIN_BLOCKED
                else:
                    rule = RuleId.TRACK_OCCUPIED
                label = res.replace("/", "/")
                return self._refuse(rule, f"{label} reserved by {holder}")
        destres = path[-1]
        if station in self._stations:
            loop = next((x for x in self._stations[station].loops if x.id == track), None)
            holder = self._resource_holder(destres)
            if holder and holder != tid:
                return self._refuse(RuleId.TRACK_OCCUPIED, f"{station}/{track} reserved by {holder}")
            length = TRAIN_TYPES[t["type"]]["length_m"]
            if loop and length > loop.length_m:
                return self._refuse(RuleId.LOOP_LEN, f"{tid} is {length} m, {station}/{track} is {loop.length_m} m")
            if loop and loop.kind == "goods" and t["type"] in {"express", "local"}:
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
            if res not in {"YW", "YE"}:
                self._holders[res] = tid
        t["auth"] = {"dest_station": station, "dest_track": track, "path": path}
        arrival = self._project_arrival(tid, path)
        return ActionResult(ok=True, message=f"GRANTED {tid} -> {station}/{track} (arrive t={arrival})")

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
            section = self._section_by_id[train["section"]]
            next_station = section.east if train["direction"] == "east" else section.west
            path = [train["section"]]
            if next_station != stop.station:
                path.extend(self._route(next_station, stop.station, "main"))
        elif train["phase"] == "reversing":
            station = train["reversing_to"][0]
            path = self._route(station, stop.station, "main")
        else:
            station = train["station"] or train["position"]
            path = self._route(station, stop.station, "main")
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
                keep = []
                for res in auth["path"][1:]:
                    if res.endswith("/main"):
                        keep.append(res)
                        break
                for res in auth["path"]:
                    if res not in keep:
                        self._holders.pop(res, None)
                auth["path"] = keep
                section = self._section_by_id[t["section"]]
                auth["dest_station"] = section.east if t["direction"] == "east" else section.west
                auth["dest_track"] = "main"
            else:
                for res in t["auth"]["path"]:
                    self._holders.pop(res, None)
                t["auth"] = None
            return ActionResult(ok=True, message=f"OK {a.train} authority cancelled")
        if isinstance(a, RelieveCrew):
            return self._relieve_crew(a.train)
        if isinstance(a, Reverse):
            return self._reverse(a.train)
        return self._refuse(RuleId.NOT_STATIONARY, "crew relief is not available in T02")

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

    def _start_move(self, t):
        if t["phase"] == "in_section" or t["phase"] == "reversing":
            return
        if t["phase"] == "not_entered":
            if self.tick < t["entry_tick"] or t["waits_for"]:
                return
            t["phase"] = "at_station"
            t["station"] = t["position"]
            t["track"] = "main"
            self._entered.add(t["id"])
            self._pending_changes.append(
                Change(tick=self.tick, kind="ENTRY", train=t["id"], text=f"{t['id']} enters at {t['station']}")
            )
        if not t["auth"]:
            return
        ready = max(
            t["entry_tick"],
            t["dwell_until"] or 0,
            t["relief_until"] or 0,
            t["breakdown_until"] or 0,
            t["blocked_until"] or 0,
        )
        if t["station"] not in {"YW", "YE"}:
            stop = self._stop_at(t, t["station"])
            if stop and stop.dep is not None:
                ready = max(ready, stop.dep, t["dwell_until"] or 0)
        if self.tick < ready:
            t["ready_tick"] = ready
            return
        t["ready_tick"] = None
        if t["station"] in self._stations:
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
        t["phase"] = "in_section"
        t["section"] = secid
        t["section_enter"] = self.tick
        t["section_exit"] = self.tick + duration
        t["station"] = None
        t["track"] = None
        self._pending_changes.append(
            Change(tick=self.tick, kind="ENTER", train=t["id"], text=f"{t['id']} enters {secid}")
        )

    def _stop_at(self, t, station):
        return next((s for s in self._train_defs[t["id"]].stops if s.station == station), None)

    def _arrive(self, t, station, track):
        tid = t["id"]
        arrived_section = t["section"]
        self._pending_changes.append(
            Change(tick=self.tick, kind="ARRIVE", train=tid, text=f"{tid} at {station}/{track}")
        )
        t["phase"] = "at_station"
        t["station"], t["track"], t["position"] = station, track, station
        t["section"] = t["section_enter"] = t["section_exit"] = None
        stop = self._stop_at(t, station)
        if stop:
            t["stop_idx"] = next(
                (i + 1 for i, x in enumerate(self._train_defs[tid].stops) if x.station == station),
                t["stop_idx"],
            )
            t["arrivals"].append({"station": station, "sched": stop.arr, "actual": self.tick})
            if t["type"] in {"express", "local"}:
                self._cost.pax_lateness += pax_cost(t["type"], t["pax"], self.tick - stop.arr)
            elif t["type"] == "freight" and station in {"YW", "YE"}:
                self._cost.freight_lateness += freight_cost(self.tick - stop.arr)
            t["restart_pending"] = False
        auth = t["auth"]
        # Release the section just left; authority path holds the remaining reserved resources.
        if arrived_section:
            self._holders.pop(arrived_section, None)
            if auth and auth["path"] and auth["path"][0] == arrived_section:
                auth["path"].pop(0)
        if station == auth["dest_station"]:
            occupied_resource = f"{station}/{track}" if station in self._stations else None
            for res in auth["path"]:
                if res != occupied_resource:
                    self._holders.pop(res, None)
            t["auth"] = None
            if stop and station not in {"YW", "YE"}:
                t["dwell_until"] = max(self.tick + stop.min_dwell, stop.dep or self.tick)
                t["ready_tick"] = t["dwell_until"]
            elif station in self._stations:
                t["ready_tick"] = self.tick
                if "F5" in self._scenario.factors:
                    t["restart_pending"] = True
            if station in {"YW", "YE"}:
                t["phase"] = "finished"
                t["finished_tick"] = self.tick
                t["station"] = t["track"] = None
                self._update_rotation_successor(tid)
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

    def _wait_graph(self):
        graph = {}
        for tid, t in self._trains.items():
            if t["phase"] != "at_station" or t["auth"] or t["station"] not in self._stations:
                continue
            pos = self._station_idx[t["station"]]
            nxt = self._station_ids[pos + (1 if t["direction"] == "east" else -1)]
            section = self._section_by_pair.get((t["station"], nxt)) or self._section_by_pair.get((nxt, t["station"]))
            section_holder = self._holders.get(section.id) if section else None
            if section_holder and section_holder != tid:
                graph[tid] = [section_holder]
                continue
            candidates = ["main"]
            if nxt in self._stations:
                station = self._stations[nxt]
                candidates += [
                    loop.id
                    for loop in station.loops
                    if TRAIN_TYPES[t["type"]]["length_m"] <= loop.length_m
                    and (loop.kind == "platform" or t["type"] not in {"express", "local"})
                ]
            blocked_by = []
            for track in candidates:
                resource = nxt if nxt not in self._stations else f"{nxt}/{track}"
                holder = self._holders.get(resource)
                if holder and holder != tid:
                    blocked_by.append(holder)
                    continue
                # A train standing on main blocks through movement and use of that main track.
                occupant = next(
                    (
                        oid
                        for oid, other in self._trains.items()
                        if oid != tid
                        and other["phase"] == "at_station"
                        and other["station"] == nxt
                        and other["track"] == track
                    ),
                    None,
                )
                if occupant:
                    blocked_by.append(occupant)
                else:
                    break
            else:
                blockers = sorted(set(blocked_by))
                if blockers:
                    graph[tid] = blockers
        return graph

    def _update_gridlock(self):
        components = find_gridlock(self._wait_graph())
        return sorted({x for comp in components for x in comp})

    def _reverse(self, tid):
        if tid not in self._gridlock:
            return self._refuse(RuleId.NO_GRIDLOCK, f"{tid} is not in a gridlock")
        t = self._trains[tid]
        idx = self._station_idx[t["station"]]
        prev = self._station_ids[idx - (1 if t["direction"] == "east" else -1)]
        if prev in self._stations:
            free = [
                x
                for x in self._stations[prev].loops
                if not self._holders.get(f"{prev}/{x.id}")
                and TRAIN_TYPES[t["type"]]["length_m"] <= x.length_m
                and (x.kind == "platform" or t["type"] not in {"express", "local"})
            ]
            if not free:
                return self._refuse(RuleId.NO_REVERSE_PATH, f"no free loop at {prev}")
            track = sorted(free, key=lambda x: x.id)[0].id
        else:
            track = "main"
        sec = self._section_by_pair.get((prev, t["station"])) or self._section_by_pair.get((t["station"], prev))
        if self._holders.get(sec.id) not in {None, tid}:
            return self._refuse(RuleId.NO_REVERSE_PATH, f"no free reverse section behind {tid}")
        self._holders.pop(f"{t['station']}/{t['track']}", None)  # only after the checks: a refusal changes nothing
        self._pending_changes.append(
            Change(tick=self.tick, kind="ENTER", train=tid, text=f"{tid} reversing into {sec.id}")
        )
        self._cost.reversal += REVERSE_COST
        t.update(
            phase="reversing",
            section=sec.id,
            section_enter=self.tick,
            section_exit=self.tick + REVERSE_TICKS,
            reversing_to=(prev, track),
            auth=None,
            station=None,
            track=None,
        )
        self._holders[sec.id] = tid
        self._gridlock = self._update_gridlock()
        return ActionResult(ok=True, message=f"OK {tid} reversing to {prev}/{track}")

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
        if reasons:
            self._last_dp = self.tick
            return AdvanceResult(tick=self.tick, reasons=reasons, changes=changes, done=False)
        while self.tick < limit:
            future = [limit, min(self._last_dp + TIME_CAP, self._scenario.shift_ticks)]
            future.extend(event.reveal_tick for event in self._scenario.events if event.id not in self._revealed_events)
            future.extend(possession["start"] for possession in self._possessions if possession["start"] > self.tick)
            for t in self._trains.values():
                if t["phase"] == "not_entered" and not t["waits_for"]:
                    future.append(t["entry_tick"])
                elif t["phase"] in {"in_section", "reversing"}:
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
            if not future:
                self.tick = limit
            else:
                self.tick = min(future)
            reasons.extend(self._process_reveals(self.tick))
            changes.extend(self._pending_changes)
            self._pending_changes.clear()
            self._process_possession_windows(self.tick)
            reasons.extend(self._process_timers(self.tick))
            # Section exits and arrivals precede new entries and departures.
            for tid in sorted(self._trains):
                t = self._trains[tid]
                if t["phase"] == "reversing" and t["section_exit"] == self.tick:
                    self._holders.pop(t["section"], None)
                    station, track = t["reversing_to"]
                    t["phase"] = "at_station"
                    t["station"], t["track"], t["position"] = station, track, station
                    t["ready_tick"] = self.tick
                    if station in self._stations:
                        self._holders[f"{station}/{track}"] = tid
                    t["section"] = t["section_exit"] = t["section_enter"] = None
                    t["reversing_to"] = None
                    self._pending_changes.append(
                        Change(tick=self.tick, kind="ARRIVE", train=tid, text=f"{tid} at {station}/{track}")
                    )
                    reasons.append(Reason(kind="READY", train=tid))
                elif t["phase"] == "in_section" and t["section_exit"] == self.tick:
                    section = self._section_by_id[t["section"]]
                    station = section.east if t["direction"] == "east" else section.west
                    if station not in self._stations and t["auth"] and station == t["auth"]["dest_station"]:
                        self._pending_changes.append(
                            Change(tick=self.tick, kind="ARRIVE", train=tid, text=f"{tid} at {station}/main")
                        )
                        self._holders.pop(t["section"], None)
                        t["phase"] = "at_station"
                        t["station"], t["track"], t["position"] = station, "main", station
                        t["section"] = t["section_exit"] = t["section_enter"] = None
                        auth = t["auth"]
                        for res in auth["path"]:
                            self._holders.pop(res, None)
                        t["auth"] = None
                        t["phase"] = "finished"
                        t["finished_tick"] = self.tick
                        t["station"] = t["track"] = None
                        if t["type"] == "freight":
                            stop = self._stop_at(t, station)
                            if stop:
                                t["stop_idx"] = len(self._train_defs[tid].stops)
                                t["arrivals"].append({"station": station, "sched": stop.arr, "actual": self.tick})
                                self._cost.freight_lateness += freight_cost(self.tick - stop.arr)
                        elif t["type"] in {"express", "local"}:
                            stop = self._stop_at(t, station)
                            if stop:
                                t["stop_idx"] = len(self._train_defs[tid].stops)
                                t["arrivals"].append({"station": station, "sched": stop.arr, "actual": self.tick})
                                self._cost.pax_lateness += pax_cost(t["type"], t["pax"], self.tick - stop.arr)
                        self._update_rotation_successor(tid)
                        reasons.append(Reason(kind="AUTHORITY_END", train=tid))
                    else:
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
                    t["phase"] = "at_station"
                    t["station"] = t["position"]
                    t["track"] = "main"
                    t["ready_tick"] = self.tick
                    self._entered.add(tid)
                    self._pending_changes.append(
                        Change(tick=self.tick, kind="ENTRY", train=tid, text=f"{tid} enters at {t['station']}")
                    )
                    if not t["auth"]:
                        reasons.append(Reason(kind="READY", train=tid))
                        t["ready_reported_tick"] = self.tick
            reasons.extend(self._process_timers(self.tick))
            for tid in sorted(self._trains):
                t = self._trains[tid]
                if t["phase"] == "at_station" and t["auth"]:
                    self._start_move(t)
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
        if self._cost_finalized:
            return
        for t in self._scenario.trains:
            if self._trains[t.id]["phase"] != "finished":
                self._cost.unfinished += unfinished_cost(t, self._scenario.shift_ticks)
        update_total(self._cost)
        self._cost_finalized = True

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
                    direction=t["direction"],
                    pax=t["pax"],
                    phase=t["phase"],
                    station=t["station"],
                    track=t["track"],
                    section=t["section"],
                    section_enter=t["section_enter"],
                    section_exit=t["section_exit"],
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
                    restart_pending=t["restart_pending"],
                    crew_used=self._crew_used_at(t) if self._crew_enabled else None,
                    crew_remaining=(CREW_LIMIT - self._crew_used_at(t)) if self._crew_enabled else None,
                    relief_until=t["relief_until"],
                    breakdown_until=t["breakdown_until"],
                    blocked_until=t["blocked_until"],
                    waits_for=t["waits_for"],
                )
            )
        return State(
            tick=self.tick,
            shift_ticks=self._scenario.shift_ticks,
            done=self.done,
            gridlock=list(self._gridlock),
            trains=trains,
            restrictions=[dict(x) for x in self._restrictions],
            possessions=[dict(x) for x in self._possessions],
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
        )

    def cost(self) -> CostBreakdown:
        out = self._cost.model_copy(deep=True)
        update_total(out)
        return out
