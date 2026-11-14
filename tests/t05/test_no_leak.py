"""T05-16: T03-36 extended to tool output strings. Nothing a tool prints may depend on an unrevealed event."""

from hypothesis import example, given, settings
from hypothesis import strategies as st

from railway.sim.model import Scenario
from railway.tools.session import Session
from tests.t03.common import ALL_FACTORS, F22, L6, STATIONS, TRACKS, TRAINS, vf
from tests.t03.test_no_leak import hidden_event

train = st.sampled_from(TRAINS)
calls = st.lists(
    st.one_of(
        st.sampled_from(
            [
                ("advance_to_next_event", {}),
                ("status", {}),
                ("status", {"compact": True}),
                ("bulletins", {}),
                ("kpis", {}),
                ("timetable", {}),
                ("network", {}),
            ]
        ),
        st.builds(
            lambda a, w: ("stringline", {"from_tick": a, "to_tick": a + w}), st.integers(0, 60), st.integers(0, 60)
        ),
        st.builds(
            lambda t, s, k: ("authorize", {"train": t, "station": s, "track": k}),
            train,
            st.sampled_from(STATIONS),
            st.sampled_from(TRACKS),
        ),
        st.builds(lambda t, u: ("hold", {"train": t, "until_tick": u}), train, st.integers(0, 60)),
        st.builds(
            lambda tool, t: (tool, {"train": t}),
            st.sampled_from(["cancel_authority", "relieve_crew", "reverse"]),
            train,
        ),
    ),
    max_size=30,
)


def sessions(event):
    common = {"extra": (L6, F22), "shift": 90, "L5": {"next_service": "L6"}}
    x = vf(*ALL_FACTORS, events=[event], **common)
    y = vf(*ALL_FACTORS, **common)
    return Session(Scenario.model_validate(x)), Session(Scenario.model_validate(y))


@settings(max_examples=100)
@given(hidden_event(), calls)
@example(  # E1 breaks down at 5: projected stringline rows must not foreshadow it
    {"id": "EV1", "kind": "breakdown", "reveal_tick": 5, "train": "E1", "tick": 5, "duration": 10},
    [
        ("authorize", {"train": "E1", "station": "S2", "track": "L1"}),
        ("stringline", {"from_tick": 0, "to_tick": 30}),
        ("advance_to_next_event", {}),
        ("stringline", {"from_tick": 0, "to_tick": 30}),
        ("status", {"compact": True}),
    ],
)
def test_T05_16_tool_outputs_no_leak(event, plan):
    x, y = sessions(event)
    reveal = event["reveal_tick"]
    for tool, args in plan:
        if x.sim.tick >= reveal or x.closed:
            break
        out_x, out_y = x.call(tool, args), y.call(tool, args)
        if max(x.sim.tick, y.sim.tick) < reveal:  # an advance that reaches the reveal may differ, as intended
            assert out_x == out_y, (tool, args)
