"""v0.3 seed acceptance and calibration on Modal: one container per candidate seed, many at once.

Each call evaluates one scenario exactly as `railway.ref.seed_filter.evaluate_scenario` does (greedy, the
reference, every exploit, and the exact solver where it applies). The local `accepted_seeds` keeps the order
and the cache file; Modal only does the work, so the result is identical to a local run.

    modal run infra/modal/v03_seeds.py --presets T1,T2 --set-name dev --count 20
    modal run infra/modal/v03_seeds.py::compare --preset T2 --seeds 40      (greedy vs reference only)

The code sent is the committed HEAD of this checkout (`git archive`), so commit before running.
"""

from __future__ import annotations

import io
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
    source = Path(tempfile.mkdtemp(prefix="railway-v03-"))
    archive = subprocess.run(["git", "-C", str(REPO), "archive", "HEAD", "railway", "seeds"], capture_output=True, check=True)
    tarfile.open(fileobj=io.BytesIO(archive.stdout)).extractall(source)
    image = image.add_local_dir(source, "/src")

app = modal.App("railway-v03-seeds", image=image)


def _setup():
    import os
    import sys

    os.chdir("/src")
    if "/src" not in sys.path:
        sys.path.insert(0, "/src")


@app.function(cpu=4, memory=4096, timeout=3600, max_containers=200)
def evaluate(preset: str, seed: int):
    _setup()
    from railway.ref.seed_filter import _evaluate_seed

    return _evaluate_seed(preset, seed)


@app.function(cpu=2, memory=2048, timeout=1800, max_containers=200)
def greedy_vs_reference(preset: str, seed: int) -> dict:
    _setup()
    from railway.gen.generator import generate
    from railway.gen.presets import PRESETS
    from railway.policies.baselines import Greedy
    from railway.policies.runner import run_policy
    from railway.ref.reference import Reference

    sc = generate(PRESETS[preset], seed)
    row = {"seed": seed}
    for name, policy in (("greedy", Greedy()), ("reference", Reference())):
        r = run_policy(policy, sc)
        row[name] = r.final.cost.model_dump()
        row[name + "_counts"] = {k: v for k, v in r.final.counts.items() if v}
    return row


@app.local_entrypoint()
def main(presets: str = "T1,T2", set_name: str = "dev", count: int = 20, batch: int = 100):
    import sys

    sys.path.insert(0, str(REPO))
    from railway.ref.seed_filter import accepted_seeds

    def mapper(names, seeds):
        return list(evaluate.map(list(names), list(seeds)))

    for preset in presets.split(","):
        accepted = accepted_seeds(preset, set_name, count, output_dir=str(REPO / "seeds"), workers=batch, mapper=mapper)
        print(f"{preset}-{set_name}: {len(accepted)} accepted -> seeds/{preset}-{set_name}.json")


@app.local_entrypoint()
def compare(preset: str = "T2", seeds: int = 40):
    rows = list(greedy_vs_reference.map([preset] * seeds, list(range(seeds))))
    better = 0
    for r in rows:
        g, ref = r["greedy"]["total"], r["reference"]["total"]
        better += ref < g
        print(f"{preset}-{r['seed']:3d} greedy {g:9.1f}  reference {ref:9.1f}  {r['reference_counts']}")
    print(f"reference beats greedy on {better}/{len(rows)}")
