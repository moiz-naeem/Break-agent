"""Drive one engine through a fixed action script and print its trajectory as JSON (cost left out).

Run as a subprocess with PYTHONPATH pointing at the package to test, so the v0.2 engine (from git) and the
v0.3 engine can be compared on the same chain scenario:  python drive_engine.py scenario.json script.json
"""

import json
import re
import sys

from railway.sim.engine import Simulator
from railway.sim.model import ActionAdapter, Scenario

FIELDS = (
    "id phase station track section section_enter section_exit entry_tick ready_tick hold_until dwell_until "
    "delay finished_tick restart_pending crew_used crew_remaining relief_until breakdown_until blocked_until "
    "waits_for"
).split()
COST_TEXT = re.compile(r"(Final cost|cost) [-0-9.]+")


def snapshot(sim):
    state = sim.state()
    trains = []
    for t in state.trains:
        row = {k: getattr(t, k) for k in FIELDS}
        row["authority"] = t.authority and [t.authority.dest_station, t.authority.dest_track, t.authority.path]
        row["next_stop"] = t.next_stop and t.next_stop.station
        row["arrivals"] = [[a.station, a.sched, a.actual] for a in t.arrivals]
        trains.append(row)
    return {"tick": state.tick, "done": state.done, "gridlock": state.gridlock, "trains": trains}


def main(scenario_path, script_path):
    raw = json.load(open(scenario_path, encoding="utf-8"))
    from railway.sim.constants import RULES_VERSION

    raw["rules_version"] = RULES_VERSION
    sim = Simulator.load(Scenario.model_validate(raw))
    out = [snapshot(sim)]
    for step in json.load(open(script_path, encoding="utf-8")):
        if step == "advance":
            r = sim.advance()
            out.append(
                {
                    "advance": [r.tick, [(x.kind, x.train) for x in r.reasons], [c.text for c in r.changes], r.done],
                    "state": snapshot(sim),
                }
            )
        else:
            r = sim.apply(ActionAdapter.validate_python(step))
            out.append({"apply": [r.ok, r.rule and str(r.rule), COST_TEXT.sub("", r.message)]})
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:])
