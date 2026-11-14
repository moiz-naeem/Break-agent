"""T09 §5.2 freeze and re-export. The freeze itself waits for the pilot (real models)."""

from __future__ import annotations

import json

import pytest

from railway.eval.pilot import constants_dump
from railway.sim import constants
from railway.util import canonical_json
from tests.helpers import REPO


def test_T09_03_constants_dump_shape():
    """T09-03 (mechanism): the dump is canonical JSON of every public constant, tuples as lists."""
    text = constants_dump()
    data = json.loads(text)
    assert text == canonical_json(data) + "\n"
    public = {n for n in vars(constants) if n.isupper() and not n.startswith("_")}
    assert set(data) == public
    assert data["RULES_VERSION"] == constants.RULES_VERSION
    assert data["POSSESSION_LOOKAHEAD"] == list(constants.POSSESSION_LOOKAHEAD)
    assert data["TRAIN_TYPES"]["freight"] == {"length_m": 600, "max_speed_kmh": 70}


def test_T09_03_constants_frozen():
    """T09-03: RULES_VERSION == "1.0" and the dump equals docs/frozen-constants.json."""
    if constants.RULES_VERSION != "1.0":
        pytest.skip(f"waiting for the pilot freeze (RULES_VERSION is {constants.RULES_VERSION!r})")
    frozen = REPO / "docs" / "frozen-constants.json"
    assert frozen.read_text(encoding="utf-8") == constants_dump()


def _manifest(name: str) -> dict:
    path = REPO / "datasets" / name / "manifest.json"
    if not path.exists():
        pytest.skip(f"waiting for the post-freeze export of datasets/{name} (T07 export + pilot freeze)")
    return json.loads(path.read_text(encoding="utf-8"))


def test_T09_04_heldout_dataset():
    """T09-04: datasets/t2-heldout has 20 tasks at rules_version 1.0."""
    manifest = _manifest("t2-heldout")
    assert manifest["rules_version"] == "1.0"
    assert len([p for p in (REPO / "datasets" / "t2-heldout").iterdir() if p.is_dir()]) == 20
    # Seed ranges, scenario_hash regeneration, norm.json and train/dev overlap need T04/T06/T07 APIs.
    pytest.importorskip("railway.gen.seeds", reason="waiting for T04 (seed ranges) to finish T09-04")
    pytest.fail("T09-04 seed-range / regeneration checks not written yet (need T04 + T07 APIs)")


def test_T09_05_s14_dataset():
    """T09-05: datasets/t2s14-heldout has 10 tasks, 14 stations each, no scenario_hash shared with t2-heldout."""
    _manifest("t2s14-heldout")
    hashes = {}
    for name in ("t2-heldout", "t2s14-heldout"):
        hashes[name] = set()
        for task in sorted(p for p in (REPO / "datasets" / name).iterdir() if p.is_dir()):
            sc = json.loads((task / "tests" / "scenario.json").read_text(encoding="utf-8"))
            if name == "t2s14-heldout":
                assert len(sc["network"]["stations"]) == 14
            hashes[name].add(canonical_json(sc))
    assert len(hashes["t2s14-heldout"]) == 10
    assert not hashes["t2-heldout"] & hashes["t2s14-heldout"]


def test_T09_06_ci_dataset_current():
    """T09-06: datasets/t1-dev-mini manifest rules_version == "1.0"."""
    assert _manifest("t1-dev-mini")["rules_version"] == "1.0"
