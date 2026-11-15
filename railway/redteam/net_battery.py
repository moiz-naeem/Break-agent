"""Baseline, accounting and replay-integrity battery for the network track (docs/redteam-net.md).

Checks every seed in a split for:
  * degenerate / non-dispatching policy costs (`do_nothing`, `stop_early`, `cancel_all`, `hold_all_freight`,
    `advance_only`, `no_relief`) against honest `booked_order` conflict regulation;
  * train-type reward classification consistency (`cost.classify` vs `formats.train_type_code` and `.passenger`);
  * missed-connection billing magnitudes under `booked_order`;
  * replay integrity (`replay` reproduces `booked_order` cost and rejects a forged `final_cost: 0.0`).

    python -m railway.redteam.net_battery --tier T1 --seeds 1,3,4,5,7
"""

from __future__ import annotations

import argparse
import json

from railway.net import exploits as X
from railway.net.cost import classify, itemise
from railway.net.formats import train_type_code
from railway.net.model import Timetable
from railway.net.replay import ReplayMismatch, replay
from railway.net.session import FinalResult, Session


def advance_only(tt: Timetable) -> FinalResult:
    """Only call `advance_to_next_event`; never resolve conflicts, rescue stranded trains or relieve crews."""
    s = Session(tt, budget=X._probe_budget(tt))
    for _ in range(3000):
        if s.closed:
            break
        s.call("advance_to_next_event", {})
    return s.close()


def no_relief(tt: Timetable) -> FinalResult:
    """Run booked-order conflict regulation while ignoring all crew warnings."""
    s = Session(tt, budget=X._probe_budget(tt))
    all_trains = frozenset(t.train.id for t in tt.trips)
    for _ in range(X._probe_budget(tt)):
        if s.closed:
            break
        X._relieve_warned_crews(s, skip_trains=all_trains)
        X._rescue_stranded(s)
        X._order_open_conflicts(s)
        s.call("advance_to_next_event", {})
    return s.close()


def check_seed_battery(tt: Timetable) -> dict:
    with X._cached_routing():
        s_booked = Session(tt, budget=X._probe_budget(tt))
        for _ in range(X._probe_budget(tt)):
            if s_booked.closed:
                break
            X._relieve_warned_crews(s_booked)
            X._rescue_stranded(s_booked)
            X._order_open_conflicts(s_booked)
            s_booked.call("advance_to_next_event", {})
        booked = s_booked.close()

        dn = X.do_nothing(tt)
        se = X.stop_early(tt)
        ca = X.cancel_all(tt)
        hf = X.hold_all_freight(tt)
        ao = advance_only(tt)
        nr = no_relief(tt)

    # Train classification consistency.
    mismatches = []
    for trip in tt.trips:
        cl = classify(trip)
        code = train_type_code(trip.train)
        if cl.passenger != trip.train.passenger or cl.code != code:
            mismatches.append({"train": trip.train.id, "code": code, "classified": cl.code, "reward": cl.reward})

    # Connection charges in booked_order.
    charges = itemise(tt, s_booked.engine.log)
    conn_charges = [c.amount for c in charges if c.event.kind == "CONNECTION" and c.amount > 0]

    # Replay determinism + forged-cost rejection.
    replayed = replay(tt, {"entries": s_booked.log, "final_cost": booked.cost.total}, budget=X._probe_budget(tt))
    replay_exact = abs(replayed.cost.total - booked.cost.total) < 1e-6
    forged_rejected = False
    try:
        replay(tt, {"entries": s_booked.log, "final_cost": 0.0}, budget=X._probe_budget(tt))
    except ReplayMismatch:
        forged_rejected = True

    policies = {
        "booked_order": booked.cost.total,
        "no_relief": nr.cost.total,
        "hold_all_freight": hf.cost.total,
        "advance_only": ao.cost.total,
        "stop_early": se.cost.total,
        "cancel_all": ca.cost.total,
        "do_nothing": dn.cost.total,
    }
    non_dispatch_min = min(dn.cost.total, se.cost.total, ca.cost.total, ao.cost.total)
    return {
        "tier": tt.tier,
        "seed": tt.seed,
        "policies": policies,
        "non_dispatch_ratio": round(non_dispatch_min / booked.cost.total, 2) if booked.cost.total > 0 else None,
        "booked_counts": booked.counts,
        "no_relief_timeouts": nr.counts["crew_timeouts"],
        "class_mismatches": mismatches,
        "connections_total": len(tt.connections),
        "missed_connection_charges": conn_charges,
        "replay_exact": replay_exact,
        "forged_rejected": forged_rejected,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tier", default="T1")
    ap.add_argument("--seeds", default="1,3,4,5,7")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = []
    for seed in (int(x) for x in a.seeds.split(",")):
        r = check_seed_battery(X.build_timetable(a.tier, seed))
        rows.append(r)
        p = r["policies"]
        print(
            f"{a.tier}-{seed}: booked {p['booked_order']:9.2f} | no_relief {p['no_relief']:9.2f} | "
            f"hold_frt {p['hold_all_freight']:9.2f} | adv_only {p['advance_only']:10.2f} | "
            f"do_nothing {p['do_nothing']:10.2f} ({r['non_dispatch_ratio']}x)",
            flush=True,
        )
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
