from __future__ import annotations

import json
from pathlib import Path

import pytest

from railway.sim.model import Scenario
from railway.util import canonical_json, content_hash
from tests.helpers import load_fixture, variant

from .fakes import FakeSession

INSTRUCTION = "# Railway dispatcher shift\n\nTEST INSTRUCTION TEXT."


@pytest.fixture
def mini3() -> Scenario:
    return Scenario.model_validate(load_fixture("mini3"))


def real_session_cls():
    mod = pytest.importorskip("railway.tools.session", reason="waits for T05 (railway.tools.session)")
    if not hasattr(mod.Session, "peek"):
        pytest.skip("waits for Session.peek (T08 addition to T05's session.py)")
    return mod.Session


@pytest.fixture(params=["fake", "real"])
def session_cls(request):
    """FakeSession now; the real T05 Session (with peek) switches on automatically once it lands."""
    if request.param == "fake":
        return FakeSession
    return real_session_cls()


def make_dataset(root: Path, n: int, c_greedy: float = 1300.0, c_best: float = 100.0) -> Path:
    """A loop-harness dataset of n mini3 variants: <root>/<scenario_id>/tests/{scenario.json,norm.json}."""
    root.mkdir(parents=True, exist_ok=True)
    base = load_fixture("mini3")
    for i in range(n):
        sc = variant(base, scenario_id=f"FIX-mini3-{i}", seed=i)
        tests = root / sc["scenario_id"] / "tests"
        tests.mkdir(parents=True)
        (tests / "scenario.json").write_text(canonical_json(sc) + "\n", encoding="utf-8")
        norm = {
            "scenario_id": sc["scenario_id"],
            "scenario_hash": content_hash(sc),
            "rules_version": sc["rules_version"],
            "c_greedy": c_greedy,
            "c_best": c_best,
        }
        (tests / "norm.json").write_text(json.dumps(norm), encoding="utf-8")
    return root
