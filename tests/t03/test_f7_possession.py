from hypothesis import example, given, settings

from railway.sim.model import Authorize
from tests.t03.common import dump, kinds, load, steps, tr, vf


def possession(reveal, start, end, **patches):
    event = {"id": "EV1", "kind": "possession", "reveal_tick": reveal, "section": "S2-S3", "start": start, "end": end}
    return load(vf("F7", events=[event], **patches))


LATE_F21 = {"F21": {"entry_tick": 3, "stops": [{"station": "YE", "arr": 27, "dep": None, "min_dwell": 0}]}}


def test_T03_09_boundary_grant():
    s = possession(0, 20, 40)
    assert [(p["section"], p["start"], p["end"]) for p in s.state().possessions] == [("S2-S3", 20, 40)]
    assert s.advance().tick == 2
    assert s.apply(Authorize(train="F21", station="YE", track="main")).ok  # exits S2-S3 at 20, not > 20


def test_T03_10_overlap_refused():
    s = possession(0, 20, 40, **LATE_F21)
    assert s.advance().tick == 3
    before = dump(s.state())
    r = s.apply(Authorize(train="F21", station="YE", track="main"))
    assert r.rule.value == "R-POSSESSION" and "S2-S3" in r.message and "20-40" in r.message
    assert dump(s.state()) == before
    assert s.apply(Authorize(train="F21", station="S2", track="L2")).ok


def test_T03_11_authority_cut():
    s = possession(5, 12, 30)
    assert s.advance().tick == 2
    assert s.apply(Authorize(train="F21", station="YE", track="main")).ok
    a = s.advance()
    assert a.tick == 5 and "BULLETIN" in [k for k, _ in kinds(a)]
    assert "AUTHORITY_CUT F21 -> S2/main: possession S2-S3 12-30" in [c.text for c in a.changes]
    f21 = tr(s, "F21")
    assert (f21.authority.dest_station, f21.authority.dest_track) == ("S2", "main")
    held_by_f21 = {res for res, tid in s._holders.items() if tid == "F21"}
    assert not held_by_f21 & {"S2-S3", "S3/main", "S3-YE"}
    a = s.advance()
    assert a.tick == 14 and ("AUTHORITY_END", "F21") in kinds(a)
    assert s.apply(Authorize(train="F21", station="YE", track="main")).rule.value == "R-POSSESSION"


@settings(max_examples=60)
@given(steps)
# Pin a run that tries to push trains through S2-S3 during the window.
@example(
    [
        ([], None),
        ([Authorize(train="F21", station="YE", track="main"), Authorize(train="E1", station="YE", track="main")], None),
        ([Authorize(train="L5", station="YW", track="main")], 8),
        ([Authorize(train="F21", station="YE", track="main")], None),
    ]
    * 2
)
def test_T03_12_no_entry_during_window(plan):
    s = possession(5, 12, 30)
    for acts, jump in plan:
        if s.done:
            break
        before = s.tick
        for a in acts:
            s.apply(a)
        # Step one tick at a time so a pass-through entry cannot be skipped over.
        target = s.tick + (jump or 6)
        while s.tick < target and not s.done:
            s.advance(max_tick=s.tick + 1)
            for t in s.state().trains:
                if t.section == "S2-S3":
                    assert not 12 <= t.section_enter < 30, (t.id, t.section_enter, before)


def test_T03_13_unrevealed_not_checked():
    s = possession(15, 20, 40, **LATE_F21)
    assert s.advance().tick == 3
    assert s.apply(Authorize(train="F21", station="YE", track="main")).ok  # no leak through refusals
    a = s.advance()
    assert a.tick == 15
    assert "AUTHORITY_CUT F21 -> S2/main: possession S2-S3 20-40" in [c.text for c in a.changes]
    # F21 exits S1-S2 at 15 too, so the cut authority ends on arrival in the same tick.
    f21 = tr(s, "F21")
    assert (f21.station, f21.track) == ("S2", "main")
