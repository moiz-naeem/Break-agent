"""v0.3 network seed generation on Modal: disjoint `dev` / `test` / `train` sets per tier, checked in parallel.

One CPU container per candidate `(tier, seed)`: it generates the timetable (with disruptions when the
generator supports them), runs `railway.net.exploits.check_seed` inline, and caches the accepted scenario as
JSON on the Modal volume `railway-scenarios` (`/net/<tier>/<seed>.json`), so benchmark containers load it in
milliseconds. The local entrypoint accepts the first `count` passing seeds in seed order, so the result is
deterministic whatever the batch size, and writes `seeds/net-<tier>-<split>.json`.

Seed ranges come from `railway.net.exploits.SPLITS` (dev 1..9999, test 10000..19999, train 20000..99999), so
the splits never overlap (docs/decisions.md #13). Nothing greedy or reference-based is computed or stored.

    modal run infra/modal/net_seeds.py --tiers T1 --splits dev,test --counts dev=5,test=30
    modal run infra/modal/net_seeds.py --tiers all --splits dev,test,train --counts dev=5,test=30,train=200

`--tiers all` expands to every tier `railway.net.templates` provides (T1 now; T2, T3, T2-X once `TIERS`
merges). The code sent is the committed HEAD of this checkout (`git archive`), so commit before running.
"""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

import modal

SCENARIOS_VOLUME = "railway-scenarios"
SCENARIOS_MOUNT = "/scenarios"
MIN_DECISIONS = 10

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "httpx>=0.28.1", "mcp>=2.3.0", "numpy>=2.4.6", "pydantic>=2.14.3", "pyyaml>=6.0.3",
)  # fmt: skip
if modal.is_local():
    REPO = Path(__file__).resolve().parents[2]
    source = Path(tempfile.mkdtemp(prefix="railway-net-seeds-"))
    archive = subprocess.run(["git", "-C", str(REPO), "archive", "HEAD", "railway"], capture_output=True, check=True)
    tarfile.open(fileobj=io.BytesIO(archive.stdout)).extractall(source, filter="data")
    image = image.add_local_dir(source, "/src")

app = modal.App("railway-net-seeds", image=image)
scenarios = modal.Volume.from_name(SCENARIOS_VOLUME, create_if_missing=True)


def _setup() -> None:
    import os
    import sys

    os.chdir("/src")
    if "/src" not in sys.path:
        sys.path.insert(0, "/src")


def scenario_path(root: str | Path, tier: str, seed: int) -> Path:
    return Path(root) / "net" / tier / f"{seed}.json"


def describe(tt) -> dict:
    """Seed-list metadata for one timetable: identity and size, never a cost anchor."""
    from railway.net.exploits import scenario_id, split_of
    from railway.util import content_hash

    return {
        "seed": tt.seed,
        "split": split_of(tt.seed),
        "scenario_id": scenario_id(tt.tier, tt.seed),
        "scenario_hash": content_hash(tt.model_dump(mode="json")),
        "shift_min": tt.shift_min,
        "trains": len(tt.trips),
        "passenger_trains": sum(t.train.passenger for t in tt.trips),
        "freight_trains": sum(not t.train.passenger for t in tt.trips),
        "disruptions": len(tt.disruptions),
    }


@app.function(cpu=1, memory=2048, timeout=1800, max_containers=200, volumes={SCENARIOS_MOUNT: scenarios})
def run_candidate(tier: str, seed: int, min_decisions: int = MIN_DECISIONS) -> dict:
    """Generate and check one candidate; cache it on the volume when it passes."""
    _setup()
    import time

    from railway.net.exploits import build_timetable, check_seed

    t0 = time.monotonic()
    try:
        tt = build_timetable(tier, seed)
        chk = check_seed(tt, min_decisions=min_decisions)
    except Exception as exc:  # noqa: BLE001 - a broken candidate is rejected, never fatal for the sweep
        return {"tier": tier, "seed": seed, "ok": False, "reasons": [f"error:{type(exc).__name__}:{exc}"]}
    row = {
        "tier": tier,
        **describe(tt),
        "decision_points": chk.decision_points,
        "ok": chk.ok,
        "reasons": list(chk.reasons),
        "wall_s": round(time.monotonic() - t0, 3),
    }
    if chk.ok:
        path = scenario_path(SCENARIOS_MOUNT, tier, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tt.model_dump_json(), encoding="utf-8")
        scenarios.commit()
    return row


def _parse_counts(spec: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        name, _, n = part.partition("=")
        out[name.strip()] = int(n)
    return out


def _git_head() -> str:
    proc = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def accept_split(tier: str, split: str, count: int, *, batch: int, min_decisions: int) -> tuple[list[dict], dict]:
    """Sweep the split's range in seed order until `count` seeds pass; returns (accepted, rejection tally)."""
    from railway.net.exploits import SPLITS

    lo, hi = SPLITS[split]
    accepted: list[dict] = []
    rejected: dict[str, int] = {}
    next_seed = lo
    while len(accepted) < count and next_seed < hi:
        size = min(batch, max(10, 2 * (count - len(accepted))))  # don't spin up 100 containers for 5 dev seeds
        seeds = list(range(next_seed, min(hi, next_seed + size)))
        next_seed = seeds[-1] + 1
        rows = sorted(
            run_candidate.map([tier] * len(seeds), seeds, [min_decisions] * len(seeds)), key=lambda r: r["seed"]
        )
        for row in rows:
            if row["ok"] and len(accepted) < count:
                accepted.append({k: v for k, v in row.items() if k not in ("ok", "reasons", "tier", "wall_s")})
            elif not row["ok"]:
                for reason in row["reasons"]:
                    key = reason.split(":", 1)[0]
                    rejected[key] = rejected.get(key, 0) + 1
        print(f"  {tier}-{split}: scanned up to {next_seed - 1}, accepted {len(accepted)}/{count}", flush=True)
    return accepted, rejected


@app.local_entrypoint()
def main(
    tiers: str = "T1",
    splits: str = "dev,test",
    counts: str = "dev=5,test=30,train=200",
    batch: int = 100,
    min_decisions: int = MIN_DECISIONS,
    out_dir: str = "seeds",
):
    import sys

    sys.path.insert(0, str(REPO))
    from railway.net.exploits import DISRUPTION_DENSITY, SPLITS, available_tiers

    tier_list = available_tiers() if tiers.strip() == "all" else [t.strip() for t in tiers.split(",") if t.strip()]
    count_map = _parse_counts(counts)
    head = _git_head()
    out = REPO / out_dir
    out.mkdir(parents=True, exist_ok=True)
    for tier in tier_list:
        for split in [s.strip() for s in splits.split(",") if s.strip()]:
            count = count_map.get(split, 0)
            if count <= 0:
                continue
            accepted, rejected = accept_split(tier, split, count, batch=batch, min_decisions=min_decisions)
            doc = {
                "header": {
                    "tier": tier,
                    "split": split,
                    "seed_range": list(SPLITS[split]),
                    "requested": count,
                    "accepted": len(accepted),
                    "min_decisions": min_decisions,
                    "disruption_density": DISRUPTION_DENSITY,
                    "checks": ["decision_points>=min_decisions", "exploits_strictly_worse_than_booked_order"],
                    "rejected": dict(sorted(rejected.items())),
                    "git_head": head,
                    "scenario_cache": f"{SCENARIOS_VOLUME}:/net/{tier}/<seed>.json",
                },
                "accepted": accepted,
            }
            path = out / f"net-{tier}-{split}.json"
            path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
            print(f"{tier}-{split}: {len(accepted)}/{count} accepted -> {path.relative_to(REPO)}")
