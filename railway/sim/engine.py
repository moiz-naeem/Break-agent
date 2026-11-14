from __future__ import annotations

import copy

from pydantic import ValidationError

from .constants import REVERSE_COST, REVERSE_TICKS, TIME_CAP, TRAIN_TYPES
from .cost import CostBreakdown, freight_cost, pax_cost, unfinished_cost, update_total
from .gridlock import find_gridlock
from .model import Action, ActionAdapter, Authorize, CancelAuthority, Hold, Reverse, Scenario
from .observe import (
    ActionResult,
    AdvanceResult,
    Arrival,
    Authority,
    Reason,
    State,
    StaticInfo,
    StopView,
    TrainState,
)
from .rules import RuleId
from .timing import travel_ticks


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
        for t in scenario.trains:
            origin = "YW" if t.direction == "east" else "YE"
            entered = t.entry_tick == 0
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
                "ready_tick": t.entry_tick if not entered else None,
                "hold_until": None,
                "dwell_until": None,
                "stop_idx": 0,
                "arrivals": [],
                "finished_tick": None,
                "reversing_to": None,
            }
        self._station_ids = ["YW", *[s.id for s in scenario.network.stations], "YE"]
        self._station_idx = {s: i for i, s in enumerate(self._station_ids)}
        self._section_by_pair = {(s.west, s.east): s for s in scenario.network.sections}
        self._section_by_id = {s.id: s for s in scenario.network.sections}
        self._stations = {s.id: s for s in scenario.network.stations}
        self._train_defs = {t.id: t for t in scenario.trains}
        self._gridlock = []

    @classmethod
    def load(cls, scenario: Scenario) -> Simulator:
        if scenario.events:
            raise NotImplementedError("scenario events are implemented in T03")
        if any(f not in {"F1", "F2", "F3", "F4"} for f in scenario.factors):
            raise NotImplementedError("scenario factors beyond F1-F4 are implemented in T03")
        return cls(scenario)

    def clone(self) -> Simulator:
        return copy.deepcopy(self)

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
        tdef = self._train_defs[tid]
        tick = max(self.tick, tdef.entry_tick)
        for res in path:
            if res in self._section_by_id:
                sec = self._section_by_id[res]
                if self._trains[tid]["section"] == res and self._trains[tid]["section_exit"] is not None:
                    tick += max(0, self._trains[tid]["section_exit"] - self.tick)
                else:
                    tick += travel_ticks(sec.length_m, sec.line_speed_kmh, TRAIN_TYPES[tdef.type]["max_speed_kmh"])
            elif res != path[-1] and res.endswith("/main"):
                stop = next((s for s in tdef.stops if s.station == res.split("/")[0]), None)
                if stop:
                    tick = max(tick + stop.min_dwell, stop.dep or tick)
        return tick

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
        if isinstance(a, Reverse):
            return self._reverse(a.train)
        return self._refuse(RuleId.NOT_STATIONARY, "crew relief is not available in T02")

    def apply(self, action: Action) -> ActionResult:
        return self._apply_mutating(action)

    def check(self, action: Action) -> ActionResult:
        return self.clone()._apply_mutating(action)

    def _start_move(self, t):
        if t["phase"] == "in_section" or t["phase"] == "reversing":
            return
        if t["phase"] == "not_entered":
            if self.tick < t["entry_tick"]:
                return
            t["phase"] = "at_station"
            t["station"] = t["position"]
            t["track"] = "main"
            self._entered.add(t["id"])
        if not t["auth"]:
            return
        ready = max(t["entry_tick"], t["dwell_until"] or 0)
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
        duration = travel_ticks(sec.length_m, sec.line_speed_kmh, TRAIN_TYPES[t["type"]]["max_speed_kmh"])
        t["phase"] = "in_section"
        t["section"] = secid
        t["section_enter"] = self.tick
        t["section_exit"] = self.tick + duration
        t["station"] = None
        t["track"] = None

    def _stop_at(self, t, station):
        return next((s for s in self._train_defs[t["id"]].stops if s.station == station), None)

    def _arrive(self, t, station, track):
        tid = t["id"]
        arrived_section = t["section"]
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
                self._cost.pax_lateness += pax_cost(t["type"], t["pax"], self.tick - stop.arr, station in {"YW", "YE"})
            elif t["type"] == "freight" and station in {"YW", "YE"}:
                self._cost.freight_lateness += freight_cost(self.tick - stop.arr)
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
            if station in {"YW", "YE"}:
                t["phase"] = "finished"
                t["finished_tick"] = self.tick
                t["station"] = t["track"] = None
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
        self._holders.pop(f"{t['station']}/{t['track']}", None)
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
        if reasons:
            self._last_dp = self.tick
            return AdvanceResult(tick=self.tick, reasons=reasons, changes=changes, done=False)
        while self.tick < limit:
            future = [limit, min(self._last_dp + TIME_CAP, self._scenario.shift_ticks)]
            for t in self._trains.values():
                if t["phase"] == "not_entered":
                    future.append(t["entry_tick"])
                elif t["phase"] in {"in_section", "reversing"}:
                    future.append(t["section_exit"])
                elif t["phase"] == "at_station" and t["auth"]:
                    future.append(max(t["entry_tick"], t["dwell_until"] or 0, t["hold_until"] or 0))
                elif t["phase"] == "at_station" and not t["auth"] and t["ready_tick"] is not None:
                    future.append(t["ready_tick"])
            future = [x for x in future if x > self.tick]
            if not future:
                self.tick = limit
            else:
                self.tick = min(future)
            # Completions/releases before departures/entries.
            for tid in sorted(self._trains):
                t = self._trains[tid]
                if t["phase"] == "not_entered" and t["entry_tick"] == self.tick:
                    t["phase"] = "at_station"
                    t["station"] = t["position"]
                    t["track"] = "main"
                    t["ready_tick"] = self.tick
                    if not t["auth"]:
                        reasons.append(Reason(kind="READY", train=tid))
                elif t["phase"] == "reversing" and t["section_exit"] == self.tick:
                    self._holders.pop(t["section"], None)
                    station, track = t["reversing_to"]
                    t["phase"] = "at_station"
                    t["station"], t["track"], t["position"] = station, track, station
                    t["ready_tick"] = self.tick
                    if station in self._stations:
                        self._holders[f"{station}/{track}"] = tid
                    t["section"] = t["section_exit"] = t["section_enter"] = None
                    t["reversing_to"] = None
                    reasons.append(Reason(kind="READY", train=tid))
                elif t["phase"] == "in_section" and t["section_exit"] == self.tick:
                    section = self._section_by_id[t["section"]]
                    station = section.east if t["direction"] == "east" else section.west
                    if station not in self._stations and t["auth"] and station == t["auth"]["dest_station"]:
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
                                self._cost.pax_lateness += pax_cost(t["type"], t["pax"], self.tick - stop.arr, True)
                        reasons.append(Reason(kind="AUTHORITY_END", train=tid))
                    else:
                        target_track = (
                            t["auth"]["dest_track"] if t["auth"] and station == t["auth"]["dest_station"] else "main"
                        )
                        end = self._arrive(t, station, target_track)
                        if end:
                            reasons.append(Reason(kind=end, train=tid))
                elif t["phase"] == "at_station" and not t["auth"] and t["ready_tick"] == self.tick:
                    reasons.append(Reason(kind="READY", train=tid))
            for tid in sorted(self._trains):
                t = self._trains[tid]
                if t["phase"] == "at_station" and t["auth"]:
                    self._start_move(t)
            new_gridlock = self._update_gridlock()
            if new_gridlock and new_gridlock != self._gridlock:
                reasons.append(Reason(kind="GRIDLOCK"))
            self._gridlock = new_gridlock
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
                    delay=0,
                    finished_tick=t["finished_tick"],
                )
            )
        return State(
            tick=self.tick,
            shift_ticks=self._scenario.shift_ticks,
            done=self.done,
            gridlock=list(self._gridlock),
            trains=trains,
            cost=self.cost(),
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
