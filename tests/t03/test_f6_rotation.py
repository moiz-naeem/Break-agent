import pytest
from pydantic import ValidationError

from railway.sim.model import Authorize, Scenario
from tests.t03.common import L6, kinds, load, tr, vf


def rotation(*factors, **patches):
    return load(vf("F6", *factors, extra=(L6,), L5={"next_service": "L6"}, **patches))


def test_T03_05_successor_waits():
    s = rotation()
    assert s.apply(Authorize(train="L5", station="YW", track="main")).message.endswith("(arrive t=19)")
    a = s.advance()
    assert (a.tick, kinds(a)) == (2, [("READY", "F21")])
    assert tr(s, "L6").waits_for == "L5"
    a = s.advance()
    assert a.tick == 19 and ("AUTHORITY_END", "L5") in kinds(a)
    assert tr(s, "L6").waits_for is None
    assert tr(s, "L6").entry_tick == 29
    a = s.advance()
    assert a.tick == 29 and ("READY", "L6") in kinds(a)


def test_T03_06_f6_off():
    s = load(vf(extra=(L6,), L5={"next_service": "L6"}))
    s.apply(Authorize(train="L5", station="YW", track="main"))
    assert tr(s, "L6").waits_for is None
    while True:
        a = s.advance()
        if ("READY", "L6") in kinds(a):
            break
        assert not a.done
    assert a.tick == 20


def test_T03_07_predecessor_unfinished():
    base = load(vf("F6"))
    base.end_shift()
    s = rotation()
    s.end_shift()
    assert tr(s, "L6").phase == "not_entered"
    # L6 unfinished
    assert s.cost().unfinished - base.cost().unfinished == 200.0  # v0.3: the flat charge; its stops are missed calls


@pytest.mark.parametrize("patch", [{"E1": {"next_service": "L6"}}, {"L5": {"next_service": "X9"}}])
def test_T03_08_validation(patch):
    with pytest.raises(ValidationError):
        Scenario.model_validate(vf("F6", extra=(L6,), **patch))
