"""Write random action scripts for chain scenarios, played on the current engine (seeded, reproducible)."""

import json
import random
import sys
from pathlib import Path

from railway.gen.generator import generate
from railway.gen.presets import PRESETS
from railway.sim.engine import Simulator


def script_for(scenario, rng, steps=120):
    sim = Simulator.load(scenario)
    places = ["YW", *[s.id for s in scenario.network.stations], "YE"]
    loops = {s.id: ["main", *[x.id for x in s.loops]] for s in scenario.network.stations}
    script = []
    for _ in range(steps):
        if sim.done:
            break
        state = sim.state()
        live = [t for t in state.trains if t.phase not in {"finished"}]
        for _ in range(rng.randint(0, 4)):
            if not live:
                break
            t = rng.choice(live)
            kind = rng.random()
            if kind < 0.7:
                here = t.station or (t.authority and t.authority.dest_station) or t.origin
                i = places.index(here) if here in places else 0
                step = 1 if t.direction == "east" else -1
                ahead = places[i + step :: step] if 0 <= i + step < len(places) else []
                station = rng.choice(ahead or places)
                track = rng.choice(loops.get(station, ["main"]))
                action = {"kind": "authorize", "train": t.id, "station": station, "track": track}
            elif kind < 0.8:
                action = {"kind": "hold", "train": t.id, "until_tick": state.tick + rng.randint(-1, 20)}
            elif kind < 0.9:
                action = {"kind": "cancel_authority", "train": t.id}
            else:
                action = {"kind": "relieve_crew", "train": t.id}
            from railway.sim.model import ActionAdapter

            sim.apply(ActionAdapter.validate_python(action))
            script.append(action)
        sim.advance()
        script.append("advance")
    return script


def main(out_dir, count=12):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for preset in ("T1", "T2"):
        for seed in range(int(count)):
            sc = generate(PRESETS[preset], seed)
            rng = random.Random(f"{preset}-{seed}")
            (out / f"{preset}-{seed}.scenario.json").write_text(json.dumps(sc.model_dump(mode="json")))
            (out / f"{preset}-{seed}.script.json").write_text(json.dumps(script_for(sc, rng)))
            n += 1
    print(n, "scripts")


if __name__ == "__main__":
    main(*sys.argv[1:])
