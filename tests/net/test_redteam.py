"""Red-team regressions for the network track (docs/redteam-net.md).

Pins Finding F1 (closed): `platform()` must refuse platforms at any station other than the next booked call's
station, so cross-station re-platforming can never serve a stop at the wrong station without `CALL_MISSED`.
"""

from __future__ import annotations

import copy

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


# ---------------------------------------------------------------- action-tool loophole audit (net_actions_audit)


def test_route_and_short_turn_conserve_and_bill_every_dropped_call():
    """Every granted route/short_turn/skip/cancel bills exactly the calls it drops; diversions keep physics."""
    from railway.redteam import net_actions_audit as A

    tt = build_t1(1)
    audit = A.Audit()
    base = Session(tt, budget=A.BIG_BUDGET)
    playouts: list = []
    A.probe_route(tt, base, audit, "start", playouts)
    A.probe_short_turn(tt, base, audit, "start")
    for tid in A._live_ids(base):
        A.act_checked(copy.deepcopy(base), audit, "cancel", {"train": tid}, tid)
    for label, tid, eid, s in playouts:
        A.regulate(s)
        s.close()
        A.check_episode(tt, s, audit, f"{label}:{tid}:{eid}")
    assert audit.violations == []
    c = audit.counts
    assert c["route:granted"] > 0 and c["route:pax_diversions_dropping_calls"] > 0 and c["route:freight_granted"] > 0
    assert c["short_turn:no_return_attempts"] > 0 and c["cancel:granted"] > 0
    assert playouts and c["pax_calls_missed"] > 0


def test_short_turn_bills_the_skipped_stop():
    from railway.net import exploits as X

    tt = X.build_timetable("T1", 1)
    s = Session(tt)
    n = len(s.engine.log.events)
    assert s.call("short_turn", {"train": "RE01", "station": "B"}).startswith("OK")
    missed = [e for e in s.engine.log.events[n:] if e.kind == "CALL_MISSED"]
    assert [(e.train, e.call, e.node, e.detail["why"]) for e in missed] == [("RE01", 2, "C", "short_turn")]
    assert s.call("short_turn", {"train": "SE01", "station": "N"}).startswith("REFUSED R-NO-RETURN")


def test_cancelling_freight_or_late_passenger_trains_loses_to_dispatching():
    from railway.net import exploits as X
    from railway.redteam import net_actions_audit as A

    tt = X.build_timetable("T1", 3)
    audit = A.Audit()
    with X._cached_routing():
        dispatch = min(A.play(tt, pax_first=p)[0].cost.total for p in (False, True))
        for name in ("cancel_all_freight", "cancel_delayed_pax"):
            res, s = A.play(tt, A.POLICIES[name](), pax_first=True)
            A.check_episode(tt, s, audit, name)
            assert res.cost.total > dispatch, name
    assert audit.violations == []
