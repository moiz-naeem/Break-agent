"""T05-23..25: replaying an episode log through a fresh Session reproduces the live result."""

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from railway.sim.constants import TOOL_BUDGET
from railway.sim.model import Scenario
from railway.tools.replay import ReplayMismatch, replay
from railway.tools.session import TOOLS, Session
from tests.t03.common import STATIONS, TRACKS, TRAINS, all_factor_scenario
from tests.t05.common import EPISODE_A, play, session

train = st.sampled_from([*TRAINS, "X9"])
call = st.one_of(
    st.sampled_from([(name, {}) for name in TOOLS]),
    st.sampled_from([("teleport", {}), ("authorize", {}), ("hold", {"train": "E1", "until_tick": "x"})]),
    st.builds(lambda c: ("status", {"compact": c}), st.booleans()),
    st.builds(lambda a, w: ("stringline", {"from_tick": a, "to_tick": a + w}), st.integers(-2, 90), st.integers(0, 70)),
    st.builds(
        lambda t, s, k: ("authorize", {"train": t, "station": s, "track": k}),
        train,
        st.sampled_from(STATIONS),
        st.sampled_from(TRACKS),
    ),
    st.builds(lambda t, u: ("hold", {"train": t, "until_tick": u}), train, st.integers(0, 90)),
    st.builds(
        lambda tool, t: (tool, {"train": t}), st.sampled_from(["cancel_authority", "relieve_crew", "reverse"]), train
    ),
    st.builds(lambda n: ("notes", {"text": "n" * n}), st.sampled_from([0, 5, 1500, 1501])),
)


def episode(scenario, calls, budget):
    s = Session(scenario, budget=budget)
    play(s, calls)
    s.close()
    return s


@settings(max_examples=60)
@given(st.lists(call, max_size=40), st.sampled_from([TOOL_BUDGET, 25]))
def test_T05_23_replay_equals_live(calls, budget):
    scenario = Scenario.model_validate(all_factor_scenario())
    live = episode(scenario, calls, budget)
    log = json.loads(json.dumps(live.episode_log()))  # through JSON, as the verifier receives it
    assert replay(scenario, log, budget=budget) == live.final


def test_T05_24_replay_detects_tamper():
    live = session()
    play(live, EPISODE_A)
    log = live.episode_log()
    assert replay(live.scenario, log) == live.final
    flipped = json.loads(json.dumps(log))
    flipped["entries"][0]["status"] = "REFUSED"
    with pytest.raises(ReplayMismatch):
        replay(live.scenario, flipped)
    moved = json.loads(json.dumps(log))
    moved["entries"][0]["args"]["track"] = "L9"  # the granted authorize now gets refused
    with pytest.raises(ReplayMismatch):
        replay(live.scenario, moved)


def test_T05_25_replay_empty_log():
    s = session()
    log = s.episode_log() | {"entries": []}
    assert replay(s.scenario, log).cost.total == 1235.0
