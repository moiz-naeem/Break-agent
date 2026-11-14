import pytest
from pydantic import ValidationError

from railway.sim.model import Scenario
from tests.t03.common import F22, vf


def pos(**kw):
    return {"id": "EV1", "kind": "possession", "reveal_tick": 0, "section": "S2-S3", "start": 10, "end": 20, **kw}


def res(**kw):
    return {
        "id": "EV1",
        "kind": "speed_restriction",
        "reveal_tick": 0,
        "sections": ["S1-S2"],
        "speed_kmh": 40,
        "start": 10,
        "end": 20,
        **kw,
    }


def late(**kw):
    return {"id": "EV1", "kind": "late_entry", "reveal_tick": 5, "train": "F22", "new_entry_tick": 45, **kw}


def brk(**kw):
    return {"id": "EV1", "kind": "breakdown", "reveal_tick": 5, "train": "E1", "tick": 5, "duration": 3, **kw}


ALL = ("F7", "F8", "F15", "F16")


@pytest.mark.parametrize(
    ("factors", "event"),
    [
        pytest.param(ALL, pos(reveal_tick=11), id="possession-reveal-after-start"),
        pytest.param(ALL, pos(start=20), id="possession-start-ge-end"),
        pytest.param(ALL, pos(end=61), id="possession-past-shift"),
        pytest.param(ALL, pos(section="S9-S8"), id="possession-unknown-section"),
        pytest.param(ALL, res(speed_kmh=0), id="restriction-speed-0"),
        pytest.param(ALL, res(speed_kmh=140), id="restriction-speed-140"),
        pytest.param(ALL, res(sections=["S1-S2", "S1-S2"]), id="restriction-duplicate-sections"),
        pytest.param(ALL, res(sections=["S9-S8"]), id="restriction-unknown-section"),
        pytest.param(ALL, res(reveal_tick=11), id="restriction-reveal-after-start"),
        pytest.param(ALL, res(start=20), id="restriction-start-ge-end"),
        pytest.param(ALL, late(new_entry_tick=30), id="late-entry-not-later"),
        pytest.param(ALL, late(reveal_tick=30), id="late-entry-reveal-ge-entry"),
        pytest.param(ALL, late(train="X9"), id="late-entry-unknown-train"),
        pytest.param(ALL, brk(reveal_tick=4), id="breakdown-reveal-ne-tick"),
        pytest.param(ALL, brk(duration=0), id="breakdown-duration-0"),
        pytest.param(("F8", "F15", "F16"), pos(), id="f7-event-with-f7-off"),
        pytest.param((*ALL, "F9"), None, id="factor-F9"),
    ],
)
def test_T03_31_event_validation(factors, event):
    with pytest.raises(ValidationError):
        Scenario.model_validate(vf(*factors, extra=(F22,), events=[event] if event else []))


def test_T03_31_valid_events_accepted():
    Scenario.model_validate(vf(*ALL, extra=(F22,), events=[pos(), res(id="EV2"), late(id="EV3"), brk(id="EV4")]))
