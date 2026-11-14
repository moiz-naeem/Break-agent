"""Baseline policies (T05 §2.6). They see the game only through a Client, never the scenario or the session."""

from __future__ import annotations

import random
from dataclasses import dataclass

from railway.sim.engine import Simulator
from railway.sim.model import Authorize
from railway.tools.observe import Observer

YARDS = ("YW", "YE")
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
    """Stations after the train's current one, in its direction, up to and including its destination."""
    places = ["YW", *[s["id"] for s in static.network["stations"]], "YE"]
    if train.direction == "west":
        places.reverse()
    destination = next(t for t in static.trains if t["id"] == train.id)["stops"][-1]["station"]
    here = places.index(train.station)
    return places[here + 1 : places.index(destination) + 1]


def loops(static, station: str) -> list[str]:
    return sorted(loop["id"] for s in static.network["stations"] if s["id"] == station for loop in s["loops"])


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
            ToolCall("authorize", {"train": a.train, "station": a.station, "track": a.track})
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
            for track in ["main"] if station in YARDS else loops(static, station):  # never a station's main
                action = Authorize(train=train.id, station=station, track=track)
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
            ahead = stations_ahead(static, train)
            stations = dict.fromkeys([*ahead[:3], *[s for s in ahead[-1:] if s in YARDS]])
            options = [
                a
                for station in stations
                for k in ["main", *loops(static, station)]
                if belief.check(a := Authorize(train=train.id, station=station, track=k)).ok
            ]
            if options:
                choice = self.rng.choice(options)
                belief.apply(choice)
                yield choice
