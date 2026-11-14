from __future__ import annotations

import json
import random
from pathlib import Path

_RANGES_PATH = Path(__file__).resolve().parents[2] / "seeds" / "ranges.json"


def ranges() -> dict[str, tuple[int, int]]:
    raw = json.loads(_RANGES_PATH.read_text(encoding="utf-8"))
    return {name: (int(bounds[0]), int(bounds[1])) for name, bounds in raw.items()}


def candidate_seeds(set_name: str, count: int | None = None):
    if set_name not in ranges():
        raise ValueError(f"unknown seed set {set_name!r}; choose dev, train, or heldout")
    start, end = ranges()[set_name]
    stop = end if count is None else min(end, start + count)
    yield from range(start, stop)


def fresh_seeds(count: int, seed: int | None = None):
    rng = random.Random(seed)
    for _ in range(count):
        yield rng.randrange(2_000_000, 2**31)
