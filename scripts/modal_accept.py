"""`gen accept` on Modal: each candidate seed is evaluated in its own CPU container.

The scan order, acceptance rule and output file are exactly `accepted_seeds`'s; only the per-seed
evaluation runs remotely, on an image built from this repo's `uv.lock`. Needs `modal setup` once.

    uv run --with modal modal run scripts/modal_accept.py --preset T1 --set-name dev --count 20
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]

image = modal.Image.debian_slim(python_version="3.12").uv_sync(str(ROOT)).add_local_python_source("railway")
app = modal.App("railway-seed-filter", image=image)


@app.function(cpu=1.0, timeout=3600, retries=1)
def evaluate(preset: str, seed: int):
    from railway.ref.seed_filter import _evaluate_seed

    return _evaluate_seed(preset, seed)


@app.local_entrypoint()
def main(preset: str = "T1", set_name: str = "dev", count: int = 20, batch: int = 200):
    from railway.ref.seed_filter import accepted_seeds

    entries = accepted_seeds(preset, set_name, count, output_dir=ROOT / "seeds", workers=batch, mapper=evaluate.map)
    print(f"accepted {len(entries)} -> seeds/{preset}-{set_name}.json")
