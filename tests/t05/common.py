"""Shared T05 helpers: sessions on mini3 variants and the T02 Episode A call sequence."""

import json

from railway.sim.model import Scenario
from railway.tools.session import Session
from tests.t03.common import vf

# T02 Episode A (test_T02_26) as tool calls, with the status each call must log.
EPISODE_A = [
    ("authorize", {"train": "E1", "station": "S2", "track": "L1"}, "GRANTED"),
    ("advance_to_next_event", {}, "INFO"),  # t=2
    ("authorize", {"train": "L5", "station": "S2", "track": "main"}, "GRANTED"),
    ("advance_to_next_event", {}, "INFO"),  # t=6
    ("authorize", {"train": "E1", "station": "YE", "track": "main"}, "REFUSED"),
    ("advance_to_next_event", {}, "INFO"),  # t=7
    ("advance_to_next_event", {}, "INFO"),  # t=11
    ("authorize", {"train": "E1", "station": "YE", "track": "main"}, "GRANTED"),
    ("end_shift", {}, "INFO"),
]


def session(*factors, budget=None, **kw) -> Session:
    scenario = Scenario.model_validate(vf(*factors, **kw))
    return Session(scenario) if budget is None else Session(scenario, budget=budget)


def play(s: Session, calls) -> list[str]:
    return [s.call(tool, args) for tool, args, *_ in calls]


def kpis(s: Session) -> dict:
    return json.loads(s.call("kpis", {}))
