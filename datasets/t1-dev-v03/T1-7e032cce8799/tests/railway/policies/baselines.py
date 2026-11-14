"""Baseline policies (T05 §2.6). They see the game only through a Client, never the scenario or the session."""

from __future__ import annotations

import random
from dataclasses import dataclass

from railway.sim.engine import Simulator
from railway.tools.observe import Observer

from .routes import (
    YARDS,
    ahead,
    authorize,
    call_args,
    is_graph_static,
    is_terminal,
    route_of,
    static_tracks,
    via,
)

__all__ = ["YARDS"]
ENDED = ("SHIFT ENDED", "BUDGET EXHAUSTED", "R-SHIFT-ENDED")


@dataclass(frozen=True)
class ToolCall:
    tool: str
    args: dict


class EpisodeOver(Exception):
    """The shift ended (at the shift end, by end_shift or by the budget): stop playing."""


class Guard:
    """Wraps a client and raises EpisodeOver as soon as an output says the episode is over."""

    def __init__(self, client):
        self.client = client

    def call(self, tool: str, args: dict) -> str:
        out = self.client.call(tool, args)
        if any(marker in out for marker in ENDED):
            raise EpisodeOver
        return out


def stations_ahead(static, train) -> list[str]:
    """Stations after the train's current one along its timetabled route, up to and including its destination."""
    return ahead(route_of(static, train.id), train.station)


def loops(static, station: str) -> list[str]:
    return sorted(loop["id"] for s in static.network["stations"] if s["id"] == station for loop in s["loops"])


def grant(static, train, station: str, track: str):
    """An authorize along the train's timetabled route (`via` on a graph)."""
    route = route_of(static, train.id)
    return authorize(train.id, station, track, via(route, train.station, station, is_graph_static(static)))


def waiting_trains(state):
    """In the network, unfinished, at a station, no authority, not held."""
    return [
        t
        for t in state.trains
        if t.phase == "at_station" and t.authority is None and (t.hold_until is None or state.tick >= t.hold_until)
    ]


class Policy:
    def run(self, client) -> None:
        try:
            self.play(Guard(client))
        except EpisodeOver:
            pass

    def play(self, client) -> None:
        raise NotImplementedError


class DoNothing(Policy):
    def play(self, client) -> None:
        pass


class Greedy(Policy):
    """The score anchor (D15): each waiting train goes to the nearest station ahead with a grantable loop."""

    def play(self, client) -> None:
        observer = Observer(client)
        while True:
            for call in self.decide(*observer.observe()):
                client.call(call.tool, call.args)

    def decide(self, static, state) -> list[ToolCall]:
        """Pure (static, state) -> this decision point's calls (T08 §2.5): grants, then advance."""
        if self.stop(state):
            return [ToolCall("end_shift", {})]
        belief = Simulator.from_observation(static, state)
        grants = [  # choose() applies each grant to the belief, so later trains see it
            ToolCall("authorize", call_args(a))
            for a in self.choose(static, state, belief)
        ]
        return [*grants, ToolCall("advance_to_next_event", {})]

    def stop(self, state) -> bool:
        return False

    def choose(self, static, state, belief):
        for train in sorted(waiting_trains(state), key=lambda t: (t.entry_tick, t.id)):
            action = self._first_granted(static, belief, train)
            if action is not None:
                belief.apply(action)  # later trains see this grant
                yield action

    @staticmethod
    def _first_granted(static, belief, train):
        for station in stations_ahead(static, train):
            terminal = is_terminal(static, station)
            for track in static_tracks(static, station, loops_only=not terminal):  # never a station's main
                action = grant(static, train, station, track)
                if belief.check(action).ok:
                    return action
        return None


class StopEarly(Greedy):
    def stop(self, state) -> bool:
        return state.tick >= 30


class RandomLegal(Greedy):
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def choose(self, static, state, belief):
        for train in sorted(waiting_trains(state), key=lambda t: t.id):
            if self.rng.random() < 0.3:
                continue
            nodes = stations_ahead(static, train)
            stations = dict.fromkeys([*nodes[:3], *[s for s in nodes[-1:] if is_terminal(static, s)]])
            options = [
                a
                for station in stations
                for k in static_tracks(static, station)
                if belief.check(a := grant(static, train, station, k)).ok
            ]
            if options:
                choice = self.rng.choice(options)
                belief.apply(choice)
                yield choice
