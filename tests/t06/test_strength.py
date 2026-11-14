"""T06-22..23 (slow): strength check against a full-knowledge offline search."""

import json
import re
from pathlib import Path

import pytest

from railway.gen.generator import generate
from railway.gen.presets import PRESETS
from railway.ref.offline_search import strength_rows

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def rows():
    seeds = [e["seed"] for e in json.loads((ROOT / "seeds" / "T1-dev.json").read_text(encoding="utf-8"))["accepted"]]
    return strength_rows([generate(PRESETS["T1"], s) for s in seeds[:5]])


def test_T06_22_offline_not_worse(rows):
    assert len(rows) == 5
    assert all(r["c_offline"] <= r["c_ref"] for r in rows)


def test_T06_23_report_written(rows):
    text = (ROOT / "docs" / "strength-check.md").read_text(encoding="utf-8")
    table = [line for line in text.splitlines() if re.match(r"^\| *\d", line)]
    assert len(table) == 5
    header = next(line for line in text.splitlines() if line.startswith("| seed"))
    assert [c.strip() for c in header.strip("|").split("|")] == ["seed", "c_greedy", "c_ref", "c_offline", "gap_pct"]
    for line, row in zip(table, rows, strict=True):
        cells = [c.strip() for c in line.strip("|").split("|")]
        assert cells == [str(row["seed"]), *(f"{row[k]:.1f}" for k in ("c_greedy", "c_ref", "c_offline", "gap_pct"))]
    interpretation = text.split("## Interpretation", 1)[1].strip()
    assert len(re.findall(r"[.!?](\s|$)", interpretation)) >= 3
