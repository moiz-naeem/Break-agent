"""Red-team battery breadth on Modal: one container per accepted seed.

Local runs cover a handful of seeds because a T2 episode costs minutes; this runs the whole accepted
list at once. Each container runs `railway.redteam.strategic.run_seed`, which is the same code the
local battery and the panel use, so a Modal row and a local row are identical.

    modal run infra/modal/v03_redteam.py::battery --preset T2 --set-name dev
    modal run infra/modal/v03_redteam.py::battery --preset T1 --slow true

The code sent is the committed HEAD of this checkout (`git archive`), so commit before running.
Plan-based attackers and the stop-early curve need the seed's best plan: T1 stores it inline in the
seed list, while T2 takes it from an exported task under `datasets/`, so on T2 they run only for the
seeds that have a dataset task.
"""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

import modal

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "httpx>=0.28.1", "matplotlib>=3.11.2", "mcp>=2.3.0", "numpy>=2.4.6", "pydantic>=2.14.3", "pyyaml>=6.0.3",
    "ortools>=9.12",
)  # fmt: skip
if modal.is_local():
    REPO = Path(__file__).resolve().parents[2]
    source = Path(tempfile.mkdtemp(prefix="railway-v03-redteam-"))
    archive = subprocess.run(
        ["git", "-C", str(REPO), "archive", "HEAD", "railway", "seeds", "datasets"], capture_output=True, check=True
    )
    tarfile.open(fileobj=io.BytesIO(archive.stdout)).extractall(source)
    image = image.add_local_dir(source, "/src")

app = modal.App("railway-v03-redteam", image=image)


def _setup():
    import os
    import sys

    os.chdir("/src")
    if "/src" not in sys.path:
        sys.path.insert(0, "/src")


@app.function(cpu=4, memory=8192, timeout=7200, max_containers=100)
def seed_battery(preset: str, entry: dict, threshold: float, slow: bool) -> dict:
    _setup()
    from dataclasses import asdict

    from railway.redteam.strategic import run_seed

    return asdict(run_seed(preset, entry, threshold, slow=slow))


@app.local_entrypoint()
def battery(preset: str = "T2", set_name: str = "dev", threshold: float = 0.02, slow: bool = False):
    import sys

    sys.path.insert(0, str(REPO))
    from railway.redteam.strategic import SeedReport, render_markdown

    entries = json.loads((REPO / "seeds" / f"{preset}-{set_name}.json").read_text(encoding="utf-8"))["accepted"]
    jobs = [(preset, e, threshold, slow) for e in entries]
    rows = sorted(seed_battery.starmap(jobs, return_exceptions=False), key=lambda r: r["seed"])
    reports = [SeedReport(**r) for r in rows]

    out = REPO / "results" / "redteam"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"strategic-{preset}-{set_name}-modal.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md = render_markdown(preset, reports, threshold)
    (out / f"strategic-{preset}-{set_name}-modal.md").write_text(md, encoding="utf-8")
    print(md)
    failures = [(r.seed, f) for r in reports for f in r.failures]
    for seed, f in failures:
        print(f"FAIL seed {seed}: {f}")
    print(f"{len(failures)} failure(s) over {len(reports)} seeds; wrote {out}")
