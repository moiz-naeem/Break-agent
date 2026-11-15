"""Network red-team probes on Modal: baseline battery + paired F1 teleport search over T1/T2/T3 dev seeds.

Each `(tier, seed, arm)` runs in its own multi-core container, loading cached scenarios from `railway-scenarios`
when available and falling back to `build_timetable(tier, seed)`. T1 and T2 run the full `k=3, horizon=None`
search; T3 runs the lighter `k=1, horizon=30` variant so every container finishes well inside the timeout.

    modal run infra/modal/net_redteam.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

IGNORE = [".git", ".venv", "jobs", "results", "runs", "artifacts", "datasets", "**/__pycache__", ".pytest_cache"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("httpx>=0.28.1", "mcp>=2.3.0", "numpy>=2.4.6", "pydantic>=2.14.3", "pyyaml>=6.0.3")
    .env({"PYTHONPATH": "/src", "PYTHONHASHSEED": "0"})
)
if modal.is_local():
    REPO = Path(__file__).resolve().parents[2]
    image = image.add_local_dir(str(REPO), "/src", ignore=IGNORE)
else:
    REPO = Path("/src")
app = modal.App("railway-net-redteam", image=image)
scenarios = modal.Volume.from_name("railway-scenarios", create_if_missing=True)
WORKERS = 16


def _load_tt(tier: str, seed: int):
    import sys

    if "/src" not in sys.path:
        sys.path.insert(0, "/src")
    from railway.net.exploits import build_timetable
    from railway.net.model import Timetable

    cached = Path(f"/scenarios/net/{tier}/{seed}.json")
    if cached.is_file():
        return Timetable.model_validate_json(cached.read_text(encoding="utf-8"))
    return build_timetable(tier, seed)


@app.function(cpu=2, memory=4096, timeout=1800, max_containers=30, volumes={"/scenarios": scenarios})
def battery_seed(tier: str, seed: int) -> dict:
    tt = _load_tt(tier, seed)
    from railway.redteam.net_battery import check_seed_battery

    return check_seed_battery(tt)


@app.function(cpu=WORKERS, memory=16384, timeout=6 * 3600, max_containers=40, volumes={"/scenarios": scenarios})
def teleport_arm(tier: str, seed: int, cross: bool, k: int, horizon: int | None) -> dict:
    tt = _load_tt(tier, seed)
    from railway.net import exploits as X
    from railway.redteam.net_teleport import run_arm
    booked = X.booked_order(tt).cost.total
    arm = run_arm(tt, cross, k=k, horizon=horizon, workers=WORKERS, include_entries=False)
    return {"booked": booked, **arm}


def _dev_seeds(tier: str) -> list[int]:
    data = json.loads((REPO / "seeds" / f"net-{tier}-dev.json").read_text(encoding="utf-8"))
    return [int(row["seed"]) for row in data["accepted"]]


@app.local_entrypoint()
def main(tiers: str = "T1,T2,T3", out_dir: str = "results/redteam-net"):
    tier_list = [t.strip() for t in tiers.split(",") if t.strip()]
    seed_pairs = [(t, s) for t in tier_list for s in _dev_seeds(t)]

    arm_jobs = []
    for t, s in seed_pairs:
        k, horizon = (1, 30) if t == "T3" else (3, None)
        arm_jobs.append((t, s, False, k, horizon))
        arm_jobs.append((t, s, True, k, horizon))

    # Launch both maps concurrently across Modal containers via spawn/get.
    bat_handles = [(t, s, battery_seed.spawn(t, s)) for t, s in seed_pairs]
    tp_handles = [(t, s, cross, teleport_arm.spawn(t, s, cross, k, horizon)) for t, s, cross, k, horizon in arm_jobs]

    bat_rows = [h.get() for _, _, h in bat_handles]
    bat_rows.sort(key=lambda r: (r["tier"], r["seed"]))

    by_seed: dict[tuple[str, int], dict[str, dict]] = {}
    for t, s, cross, h in tp_handles:
        res = h.get()
        by_seed.setdefault((t, s), {})["exploit" if cross else "control"] = res

    tp_rows = []
    for t, s in sorted(by_seed):
        ctrl = by_seed[(t, s)]["control"]
        expl = by_seed[(t, s)]["exploit"]
        booked = ctrl["booked"]
        ctrl_clean = {k: v for k, v in ctrl.items() if k != "booked"}
        expl_clean = {k: v for k, v in expl.items() if k != "booked"}
        row = {
            "tier": t,
            "seed": s,
            "booked": booked,
            "control": ctrl_clean,
            "exploit": expl_clean,
            "gain": round(ctrl_clean["cost"]["total"] - expl_clean["cost"]["total"], 6),
        }
        tp_rows.append(row)
        print(
            f"{t}-{s}: booked {booked:10.2f} | control {ctrl_clean['cost']['total']:10.2f} "
            f"({ctrl_clean['secs']}s) | exploit {expl_clean['cost']['total']:10.2f} "
            f"({expl_clean['secs']}s) | gain {row['gain']:9.2f} | "
            f"served_elsewhere {len(expl_clean['served_elsewhere'])}",
            flush=True,
        )

    dest = REPO / out_dir
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "battery-dev.json").write_text(json.dumps(bat_rows, indent=2) + "\n", encoding="utf-8", newline="\n")
    (dest / "teleport-dev.json").write_text(json.dumps(tp_rows, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {dest / 'battery-dev.json'} and {dest / 'teleport-dev.json'}")
