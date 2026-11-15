"""Red-team regressions for the network track (docs/redteam-net.md).

Each `xfail(strict=True)` is an open finding. It turns into a hard failure (XPASS) the day the hole is closed,
so whoever fixes it must remove the marker. That stops the fix from being undone later without anyone noticing.
"""

from __future__ import annotations

import pytest

from railway.net import exploits as X
from railway.net.session import Session
from railway.redteam.net_battery import check_seed_battery

from .conftest import build_t1

F1 = "F1: platform() accepts a platform at another station, and the call there counts as served"


def _regulate(s: Session) -> None:
    for _ in range(X._probe_budget(s.timetable)):
        if s.closed:
            break
        X._relieve_warned_crews(s)
        X._rescue_stranded(s)
        X._order_open_conflicts(s)
        s.call("advance_to_next_event", {})
    s.close()


def _first_teleport(tt):
    """A passenger train whose next call can be re-platformed to a platform at a different station at t=0."""
    net = tt.net
    for trip in sorted(tt.trips, key=lambda t: t.train.id):
        if not trip.train.passenger:
            continue
        booked_node = net.element(trip.calls[0].platform).node
        for p in net.platforms:
            if p.node == booked_node or p.kind == "siding" or p.length_m < trip.train.length_m:
                continue
            s = Session(tt)
            if s.call("platform", {"train": trip.train.id, "station": p.node, "n": p.id}).startswith("OK"):
                return trip.train.id, booked_node, p
    return None


@pytest.mark.xfail(strict=True, reason=F1)
def test_platform_refuses_a_platform_at_another_station():
    assert _first_teleport(build_t1(1)) is None


@pytest.mark.xfail(strict=True, reason=F1)
def test_a_call_made_at_another_station_is_not_served():
    tt = build_t1(1)
    found = _first_teleport(tt)
    assert found is not None  # holds while F1 is open; the test above covers the fixed case
    train, booked_node, p = found
    s = Session(tt, budget=X._probe_budget(tt))
    s.call("platform", {"train": train, "station": p.node, "n": p.id})
    _regulate(s)
    served_elsewhere = [
        e
        for e in s.engine.log.events
        if e.kind == "ARRIVE" and e.train == train and e.call == 0 and e.node != booked_node
    ]
    missed = [e for e in s.engine.log.events if e.kind == "CALL_MISSED" and e.train == train and e.call == 0]
    # The fixed engine either refuses the move or bills call 0 as missed; today it bills it as made, on time.
    assert not served_elsewhere or missed


def test_battery_replay_and_classification_invariants():
    row = check_seed_battery(build_t1(1))
    assert row["class_mismatches"] == []
    assert row["replay_exact"] is True
    assert row["forged_rejected"] is True
    assert row["policies"]["advance_only"] > row["policies"]["booked_order"]
    assert row["policies"]["do_nothing"] > row["policies"]["booked_order"]
