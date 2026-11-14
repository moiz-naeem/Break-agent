"""Measurements of the v0.3 net engine on Modal: one CPU container per seed. Needs `modal setup` once.

    modal run scripts/modal_jobs.py::main                          # T1, seeds 1-8, its default 180 min shift
    modal run scripts/modal_jobs.py::main --tier T3                # T3, seeds 1-8, its default 360 min shift
    modal run scripts/modal_jobs.py::main --seeds 1,2,3 --shifts 180,360
    modal run scripts/modal_jobs.py::profile --tier T2 --seed 1    # cProfile of one generator call

For each seed and shift length, `measure_seed` generates the tier's timetable (templates.TIERS: its layout and its
service groups) and runs the engine over it twice:

- **none**: nobody answers. `Engine.advance` is called until SHIFT_END and every decision is left unanswered. A
  CONFLICT that is not answered is raised again at the same minute, so the run stalls there; it is stopped after
  the same decision comes back STALL_REPEATS times and reported as stalled at that minute.
- **planner**: every conflict is answered in timetable order, as `timetable.plan_run` does. This is the run the
  timetable was booked from, so it is the no-disruption baseline an agent's run is compared with.

For both it reports wall time, decisions by kind and how many trains finished their work (phase "done"), and for
the planner run how many ran late. A shift is clean when the planner run finishes every train with none late.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
# The engine needs only pydantic; the rest of the project is left out of the image. Kept separate from
# modal_tests.py's image on purpose, so the container does not need that script to import this one.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pydantic==2.14.3", "pyyaml==6.0.3", "numpy==2.5.4")
    .env({"PYTHONHASHSEED": "0"})
    .add_local_python_source("railway")
)
app = modal.App("railway-net-jobs", image=image)

KINDS = ("CONFLICT", "UNROUTABLE", "BLOCKED", "IDLE")
STALL_REPEATS = 3
MAX_DECISIONS = 20_000


def _run(net, trips, shift: int, answer: bool) -> dict:
    from railway.net.engine import Engine

    engine = Engine(net, list(trips), shift)
    kinds: Counter[str] = Counter()
    last, repeats, stalled_at = None, 0, None
    started = time.perf_counter()
    for _ in range(MAX_DECISIONS):
        decision = engine.advance()
        if decision.kind == "SHIFT_END":
            break
        kinds[decision.kind] += 1
        same = (decision.tick, decision.kind, decision.conflicts, decision.trains)
        repeats = repeats + 1 if same == last else 0
        last = same
        if repeats >= STALL_REPEATS:
            stalled_at = decision.tick
            break
        if answer:
            for conflict in decision.conflicts:
                order = sorted(conflict.trains, key=lambda t: engine.trains[t].booked())
                engine.order(conflict.key, order)
    else:
        stalled_at = engine.tick
    seconds = time.perf_counter() - started
    done = sum(mv.phase == "done" for mv in engine.trains.values())
    late = sum(mv.worst_late > 0 for mv in engine.trains.values())
    return {
        "seconds": round(seconds, 3),
        "kinds": {k: kinds.get(k, 0) for k in KINDS},
        "done": done,
        "late": late,
        "stalled_at": stalled_at,
        "end_tick": engine.tick,
    }


@app.function(cpu=1.0, timeout=3600)
def measure_seed(seed: int, shifts: list[int], tier: str = "T1") -> list[dict]:
    from railway.net.templates import TIERS
    from railway.net.timetable import generate, peak_utilisation

    preset = TIERS[tier]
    net = preset.network()
    core, fill = preset.services()
    rows = []
    for shift in shifts:
        row = {"seed": seed, "shift": shift}
        started = time.perf_counter()
        try:
            timetable = generate(net, core, fill, shift, seed, tier)
        except ValueError as exc:
            rows.append({**row, "gen_s": round(time.perf_counter() - started, 2), "rejected": str(exc)})
            continue
        row["gen_s"] = round(time.perf_counter() - started, 2)
        trips = timetable.trips
        row["trains"] = len(trips)
        row["freight"] = sum(not t.train.passenger for t in trips)
        row["connections"] = len(timetable.connections)
        row["peak"] = round(peak_utilisation(net, trips, shift), 3)
        row["none"] = _run(net, trips, shift, answer=False)
        row["planner"] = _run(net, trips, shift, answer=True)
        rows.append(row)
    return rows


def _fmt(run: dict) -> str:
    k = run["kinds"]
    stall = f" stalled@{run['stalled_at']}" if run["stalled_at"] is not None else ""
    return (
        f"{run['seconds']:7.3f}s  C{k['CONFLICT']:4d} U{k['UNROUTABLE']:2d} B{k['BLOCKED']:2d} I{k['IDLE']:2d}"
        f"  done {run['done']:3d}{stall}"
    )


@app.local_entrypoint()
def main(tier: str = "T1", seeds: str = "1,2,3,4,5,6,7,8", shifts: str = ""):
    from railway.net.templates import TIERS

    seed_list = [int(s) for s in seeds.split(",")]
    shift_list = [int(s) for s in shifts.split(",")] if shifts else [TIERS[tier].shift_min]
    kwargs = {"shifts": shift_list, "tier": tier}
    rows = [row for result in measure_seed.map(seed_list, kwargs=kwargs) for row in result]
    print(f"\n{tier} measurements (C/U/B/I = CONFLICT/UNROUTABLE/BLOCKED/IDLE decisions; done = trains that finished)")
    header = f"{'seed':>4} {'shift':>5} {'gen s':>7} {'trains':>6} {'frt':>3} {'conn':>4} {'peak':>5}"
    print(f"{header}   {'no answers':<46} {'timetable-order answers'}")
    for row in sorted(rows, key=lambda r: (r["shift"], r["seed"])):
        lead = f"{row['seed']:>4} {row['shift']:>5} {row['gen_s']:>7.2f}"
        if "rejected" in row:
            print(f"{lead}   rejected: {row['rejected']}")
            continue
        body = f"{row['trains']:>6} {row['freight']:>3} {row['connections']:>4} {row['peak']:>5.2f}"
        print(f"{lead} {body}   {_fmt(row['none']):<46} {_fmt(row['planner'])}  late {row['planner']['late']}")
    good = [r for r in rows if "rejected" not in r]
    if good:
        clean = sum(r["planner"]["done"] == r["trains"] and r["planner"]["late"] == 0 for r in good)
        trains = [r["trains"] for r in good]
        peaks = [r["peak"] for r in good]
        print(
            f"\n{tier}: {len(good)}/{len(rows)} generated, {clean} clean (every train done, none late); "
            f"trains {min(trains)}-{max(trains)}, peak {min(peaks):.2f}-{max(peaks):.2f}, "
            f"generation {min(r['gen_s'] for r in good):.1f}-{max(r['gen_s'] for r in good):.1f} s"
        )


@app.function(cpu=1.0, timeout=3600)
def profile_generate(seed: int, shift: int, top: int, tier: str = "T1") -> str:
    import cProfile
    import io
    import pstats

    from railway.net.templates import TIERS
    from railway.net.timetable import generate

    preset = TIERS[tier]
    net = preset.network()
    core, fill = preset.services()
    shift = shift or preset.shift_min
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    generate(net, core, fill, shift, seed, tier)
    profiler.disable()
    out = io.StringIO()
    took = time.perf_counter() - started
    out.write(f"generate({tier}, seed={seed}, shift={shift}) took {took:.1f}s under cProfile\n")
    stats = pstats.Stats(profiler, stream=out).strip_dirs()
    stats.sort_stats("tottime").print_stats(top)
    stats.sort_stats("cumulative").print_stats(top)
    stats.print_callers("plan_run|_set_routes|_conflicts|_wanting")
    return out.getvalue()


@app.local_entrypoint()
def profile(tier: str = "T1", seed: int = 1, shift: int = 0, top: int = 30):
    print(profile_generate.remote(seed, shift, top, tier))
