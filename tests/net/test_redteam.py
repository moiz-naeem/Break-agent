"""Red-team regressions for the network track (docs/redteam-net.md).

Pins Finding F1 (closed): `platform()` must refuse platforms at any station other than the next booked call's
station, so cross-station re-platforming can never serve a stop at the wrong station without `CALL_MISSED`.
"""

from __future__ import annotations

from railway.net.rules import NetRule
from railway.net.session import Session
from railway.redteam.net_battery import check_seed_battery
from railway.redteam.net_teleport import run_arm

from .conftest import build_t1


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


def test_platform_refuses_a_platform_at_another_station():
    tt = build_t1(1)
    assert _first_teleport(tt) is None
    s = Session(tt)
    out = s.call("platform", {"train": "IC01", "station": "B", "n": "B1"})
    assert out.startswith("REFUSED R-NO-ROUTE") and "not at A" in out
    refusal = s.engine.platform("IC01", "B1")
    assert refusal is not None and refusal.rule == NetRule.NO_ROUTE


def test_paired_teleport_probe_matches_same_station_control_after_fix():
    tt = build_t1(1)
    ctrl = run_arm(tt, False, k=1, horizon=30, include_entries=False)
    expl = run_arm(tt, True, k=1, horizon=30, include_entries=False)
    assert expl["served_elsewhere"] == []
    assert expl["cost"]["total"] == ctrl["cost"]["total"]


def test_battery_replay_and_classification_invariants():
    row = check_seed_battery(build_t1(1))
    assert row["class_mismatches"] == []
    assert row["replay_exact"] is True
    assert row["forged_rejected"] is True
    assert row["policies"]["advance_only"] > row["policies"]["booked_order"]
    assert row["policies"]["do_nothing"] > row["policies"]["booked_order"]
