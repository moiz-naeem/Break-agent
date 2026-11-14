"""Deterministic belief-state dispatcher used as the scripted reference."""

from __future__ import annotations

from railway.policies.baselines import EpisodeOver, Guard, ToolCall
from railway.policies.routes import call_args
from railway.sim.constants import VALUE, W_FREIGHT
from railway.sim.engine import Simulator
from railway.sim.model import Authorize, CancelAuthority, Hold, RelieveCrew
from railway.sim.observe import State, StaticInfo
from railway.tools.observe import Observer

from .candidates import WAIT, candidates
from .rollout import crew_target_arrival, rollout


class Reference:
    def __init__(self, horizon: int = 90, max_candidates: int = 5, lookahead_stations: int = 4):
        self.horizon = horizon
        self.max_candidates = max_candidates
        self.lookahead_stations = lookahead_stations

    def belief(self, static: StaticInfo, state: State) -> Simulator:
        """What the reference plans on: what an agent could see. The anchor search overrides this."""
        return Simulator.from_observation(static, state)

    def _priority(self, train):
        if train.type in {"express", "local"}:
            return (0, -(VALUE[train.type] * train.pax / 100), train.id)
        if train.type == "freight":
            return (1, -W_FREIGHT, train.id)
        return (2, 0, train.id)

    def decide(self, static: StaticInfo, state: State) -> list[ToolCall]:
        belief = self.belief(static, state)
        calls: list[ToolCall] = []

        if "F10" in static.factors:
            crew_stations = {
                station["id"] for station in static.network.get("stations", []) if station.get("crew_station")
            }
            for observed in state.trains:
                remaining = getattr(observed, "crew_remaining", None)
                relieving_until = getattr(observed, "relief_until", None)
                if (
                    remaining is None
                    or observed.phase != "at_station"
                    or observed.station not in crew_stations
                    or (relieving_until is not None and state.tick < relieving_until)
                ):
                    continue
                arrival = crew_target_arrival(belief, observed.id, observed.station)
                if arrival is not None and state.tick + remaining < arrival + 10:
                    action = RelieveCrew(train=observed.id)
                    if belief.apply(action).ok:
                        calls.append(ToolCall("relieve_crew", {"train": observed.id}))

        actionable = []
        for observed in state.trains:
            internal = belief._trains[observed.id]
            if observed.phase == "finished" or observed.phase == "not_entered":
                continue
            if internal["hold_until"] is not None and state.tick < internal["hold_until"]:
                continue
            ordinary = observed.phase == "at_station" and observed.authority is None
            extension = (
                "F5" in static.factors
                and observed.type == "freight"
                and observed.authority is not None
                and observed.authority.dest_station
                not in {stop.station for stop in belief._train_defs[observed.id].stops}
            )
            if ordinary or extension:
                actionable.append(observed)

        for observed in sorted(actionable, key=self._priority):
            options = candidates(
                belief,
                observed.id,
                lookahead_stations=self.lookahead_stations,
                max_candidates=self.max_candidates,
            )
            values = []
            for option in options:
                branch = belief.clone()
                if option is not WAIT:
                    branch.apply(option)
                values.append((rollout(branch, self.horizon), option))
            # Python's stable min preserves candidate ordering on equal values.
            _, winner = min(values, key=lambda pair: pair[0])
            if winner is not WAIT:
                belief.apply(winner)
                calls.append(ToolCall("authorize", call_args(winner)))
        calls.append(ToolCall("advance_to_next_event", {}))
        return calls

    def evaluate(self, static: StaticInfo, state: State, call: ToolCall) -> float:
        sim = Simulator.from_observation(static, state)
        if call.tool == "advance_to_next_event":
            sim.advance()
        else:
            action_types = {
                "authorize": Authorize,
                "hold": Hold,
                "cancel_authority": CancelAuthority,
                "relieve_crew": RelieveCrew,
            }
            action_type = action_types.get(call.tool)
            if action_type is not None:
                try:
                    branch = sim.clone()
                    if branch.apply(action_type(**call.args)).ok:
                        sim = branch
                except (TypeError, ValueError):
                    pass
        # Read-only, notes, unknown tools, and refused calls leave the belief unchanged.
        return rollout(sim, self.horizon)

    def run(self, client) -> None:
        """Observer: network + timetable once; then status -> decide -> execute, until the shift ends."""
        try:
            guarded = Guard(client)
            observer = Observer(guarded)
            while True:
                static, state = observer.observe()
                for call in self.decide(static, state):
                    guarded.call(call.tool, call.args)
        except EpisodeOver:
            pass
