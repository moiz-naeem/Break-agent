from railway.sim.model import Authorize
from tests.t03.common import F22, L6, kinds, load, tr, vf


def late(train, new, reveal):
    return {"id": "EV1", "kind": "late_entry", "reveal_tick": reveal, "train": train, "new_entry_tick": new}


def test_T03_29_late_entry():
    s = load(vf("F16", extra=(F22,), shift=90, events=[late("F22", 45, 10)]))
    while s.tick < 12:
        assert tr(s, "F22").entry_tick == (30 if s.tick < 10 else 45), s.tick
        s.advance(max_tick=s.tick + 1)
    s = load(vf("F16", extra=(F22,), shift=90, events=[late("F22", 45, 10)]))
    for tick, reason in [(2, ("READY", "F21")), (10, ("BULLETIN", None)), (45, ("READY", "F22"))]:
        a = s.advance()
        assert a.tick == tick and reason in kinds(a), (tick, kinds(a))


def test_T03_30_late_entry_with_rotation():
    s = load(vf("F6", "F16", extra=(L6,), L5={"next_service": "L6"}, events=[late("L6", 25, 0)]))
    s.apply(Authorize(train="L5", station="YW", track="main"))
    while True:
        a = s.advance()
        if ("READY", "L6") in kinds(a):
            break
        assert not a.done
    assert a.tick == 29 == tr(s, "L6").entry_tick  # max(25, 19 + 10)
