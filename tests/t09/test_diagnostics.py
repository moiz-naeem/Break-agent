"""T09 §5.4 diagnostics. Rows are SYNTHETIC; state sampling runs the real railway.sim Simulator.

T09-12..T09-16 need the tools layer (T05), the reference (T06) or the presets (T04) and
skip until those modules exist in this checkout.
"""

from __future__ import annotations

import json

import pytest

from railway.eval.diagnostics import (
    ablation_gaps,
    failure_breakdown,
    group_rows,
    mean_regret,
    regret,
    sample_decision_states,
    score_by_preset,
)
from railway.sim.engine import Simulator
from railway.sim.model import Authorize, Scenario
from tests.helpers import load_fixture
from tests.t09.synthetic import make_row, model_rows, ros_test_only


def _mini3() -> Simulator:
    return Simulator.load(Scenario.model_validate(load_fixture("mini3")))


# ---- failure breakdown (§2.6 item 1) ---------------------------------------------------


def test_T09_diag_failure_breakdown():
    """§2.6 item 1: mean cost by component and mean failure counts per episode, ok rows only."""
    rows = [
        make_row(cost_total=400.0, gridlocks=1, crew_timeouts=0, invalid_calls=4),
        make_row(seed=2, cost_total=800.0, gridlocks=0, crew_timeouts=2, invalid_calls=0),
        make_row(seed=3, cost_total=10_000.0, status="error"),
    ]
    fb = failure_breakdown(rows)
    assert fb["episodes"] == 2
    assert fb["pax_lateness"] == pytest.approx((300 + 600) / 2)
    assert fb["unfinished"] == pytest.approx((100 + 200) / 2)
    assert fb["freight_lateness"] == 0.0
    assert fb["total"] == pytest.approx(600.0)
    assert (fb["gridlocks"], fb["crew_timeouts"], fb["invalid_calls"]) == (0.5, 1.0, 2.0)


def test_T09_diag_failure_breakdown_no_ok_rows():
    """No ok rows → no breakdown (never a made-up zero)."""
    assert failure_breakdown([make_row(status="error")]) is None


def test_T09_diag_group_rows_keeps_display_order():
    """Rows group by (model, harness, mode) in first-appearance order."""
    rows = [
        make_row(model="big"),
        make_row(model="small"),
        make_row(model="big", harness="loop", mode="windowed"),
        make_row(model="big", seed=2),
    ]
    groups = group_rows(rows)
    assert list(groups) == ["big|harbor|", "small|harbor|", "big|loop|windowed"]
    assert len(groups["big|harbor|"]) == 2


# ---- factor ablation and score vs length (§2.6 items 2 and 4) ---------------------------


def test_T09_diag_ablation_gaps():
    """§2.6 item 2: per factor, small and big score_ros on `T2-only-<F>` and the gap."""
    rows = []
    for f, (s, b) in {"F5": (0.1, 0.5), "F16": (0.3, 0.4)}.items():
        rows += model_rows("small", s, seeds=range(1, 6), preset=f"T2-only-{f}")
        rows += model_rows("big", b, seeds=range(1, 6), preset=f"T2-only-{f}")
    rows += model_rows("big", 0.9, seeds=range(1, 6), preset="T2")  # not an ablation preset: ignored
    gaps = ablation_gaps(rows, score_fn=ros_test_only)
    assert list(gaps) == ["F5", "F16"]  # factor order, not string order
    assert gaps["F5"]["small"] == pytest.approx(0.1)
    assert gaps["F5"]["big"] == pytest.approx(0.5)
    assert gaps["F5"]["gap"] == pytest.approx(0.4)
    assert gaps["F16"]["gap"] == pytest.approx(0.1)


def test_T09_diag_ablation_gap_missing_model():
    """A factor run for only one model has no gap (None), not a fake one."""
    rows = model_rows("big", 0.5, seeds=range(1, 6), preset="T2-only-F7")
    gaps = ablation_gaps(rows, score_fn=ros_test_only)
    assert gaps["F7"] == {"small": None, "big": pytest.approx(0.5), "gap": None}


def test_T09_diag_score_by_preset():
    """§2.6 item 4: score vs episode length, per (model, preset), presets in T1, T2, T3 order."""
    rows = []
    for preset, s in (("T3", 0.1), ("T1", 0.5), ("T2", 0.3)):
        rows += model_rows("small", s, seeds=range(1, 6), preset=preset)
    out = score_by_preset(rows, score_fn=ros_test_only)
    assert list(out) == [("small", "T1"), ("small", "T2"), ("small", "T3")]
    assert out[("small", "T2")] == pytest.approx(0.3)


# ---- same-state regret (§2.6 item 3) ----------------------------------------------------


@pytest.mark.parametrize(
    ("action", "candidates", "expected"),
    [
        (10.0, [10.0, 12.0, 15.0], 0.0),  # the action is the argmin
        (12.0, [10.0, 12.0, 15.0], 2.0),
        (9.0, [10.0, 12.0], 0.0),  # the action beats every candidate: min over candidates ∪ {action}
        (5.0, [], 0.0),
    ],
)
def test_T09_15_regret_formula(action, candidates, expected):
    """T09-15: regret = v(action) − min(v(c) for c in candidates ∪ {action}) ≥ 0."""
    assert regret(action, candidates) == expected


def test_T09_15_regret_rejects_nan():
    with pytest.raises(ValueError):
        regret(float("nan"), [1.0])


def test_T09_15_mean_regret_per_model():
    records = [{"model": "big", "regret": 0.0}, {"model": "small", "regret": 4.0}, {"model": "big", "regret": 2.0}]
    assert mean_regret(records) == {"big": 1.0, "small": 4.0}


def _authorize_e1_once(sim: Simulator) -> None:
    if sim.tick == 0:
        sim.apply(Authorize(kind="authorize", train="E1", station="S2", track="L1"))


def test_T09_15_sample_states_real_sim():
    """T09-15 (sampling): every k-th decision point of a trajectory, from the real Simulator.

    mini3 with E1 → S2/L1 at t=0: decision points t = 0, 2 (READY F21), 6 (AUTHORITY_END E1),
    7 (READY E1). k=2 → indices 0 and 2 → ticks 0 and 6.
    """
    states = sample_decision_states(_mini3(), _authorize_e1_once, every_k=2, limit=10)
    assert [(s.index, s.tick) for s in states] == [(0, 0), (2, 6)]
    assert states[1].state.tick == 6
    assert states[0].static.rules_version == "0.1-provisional"


def test_T09_15_sample_states_limit_and_determinism():
    """T09-15: the sample is capped at `limit` and identical across runs."""

    def dump(states):
        return [(s.index, s.tick, json.dumps(s.state.model_dump(mode="json"), sort_keys=True)) for s in states]

    a = sample_decision_states(_mini3(), _authorize_e1_once, every_k=1, limit=3)
    b = sample_decision_states(_mini3(), _authorize_e1_once, every_k=1, limit=3)
    assert [s.tick for s in a] == [0, 2, 6]
    assert dump(a) == dump(b)


def test_T09_15_sample_states_are_snapshots():
    """A sampled state is not changed by the rest of the trajectory."""
    states = sample_decision_states(_mini3(), _authorize_e1_once, every_k=1, limit=10)
    assert states[0].state.tick == 0
    assert states[-1].tick < 60  # the shift end is not a decision to ask about


def test_T09_15_sample_states_bad_k():
    with pytest.raises(ValueError):
        sample_decision_states(_mini3(), _authorize_e1_once, every_k=0, limit=1)


# ---- need other tickets ------------------------------------------------------------------


def test_T09_12_diagnostic_tools_refused_officially():
    """T09-12: official session → conflicts/what_if refused with R-DIAGNOSTIC-ONLY."""
    session_mod = pytest.importorskip("railway.tools.session", reason="waiting for T05 (tools layer)")
    s = session_mod.Session(Scenario.model_validate(load_fixture("mini3")))
    assert s.call("conflicts", {}).startswith("REFUSED R-DIAGNOSTIC-ONLY")
    assert s.call("what_if", {"actions": []}).startswith("REFUSED R-DIAGNOSTIC-ONLY")


def test_T09_13_conflicts():
    """T09-13: mini3 at t=0 with diagnostics → conflicts() includes the opposing pair E1/L5."""
    session_mod = pytest.importorskip("railway.tools.session", reason="waiting for T05 (tools layer)")
    s = session_mod.Session(Scenario.model_validate(load_fixture("mini3")), diagnostics=True)
    entries = json.loads(s.call("conflicts", {}))
    assert any(e["trains"] == ["E1", "L5"] for e in entries)


def test_T09_14_what_if_pure():
    """T09-14: what_if returns Reference.evaluate's value; state unchanged; budget +1."""
    session_mod = pytest.importorskip("railway.tools.session", reason="waiting for T05 (tools layer)")
    ref_mod = pytest.importorskip("railway.ref.reference", reason="waiting for T06 (reference)")
    scenario = Scenario.model_validate(load_fixture("mini3"))
    s = session_mod.Session(scenario, diagnostics=True)
    before = s.peek("status", {})
    used = json.loads(s.peek("kpis", {}))["budget_used"]
    call = {"tool": "authorize", "args": {"train": "E1", "station": "S2", "track": "L1"}}
    out = json.loads(s.call("what_if", {"actions": [call]}))
    sim = Simulator.load(scenario)
    expected = ref_mod.Reference().evaluate(sim.static_info(), sim.state(), ref_mod.ToolCall(**call))
    assert out["projected_cost"] == expected
    assert s.peek("status", {}) == before
    assert json.loads(s.peek("kpis", {}))["budget_used"] == used + 1


def test_T09_15_regret():
    """T09-15: on 5 sampled reference states, regret ≥ 0 and the reference's own first action has regret 0."""
    pytest.importorskip("railway.ref.reference", reason="waiting for T06 (reference)")
    pytest.importorskip("railway.ref.candidates", reason="waiting for T06 (candidates)")
    # Deliberately red once T06 lands: the regret runner (reference trajectories → sample_decision_states →
    # Reference.evaluate over candidates(B, train) ∪ {advance}) is not built yet. See docs/gates/T09.md.
    pytest.fail("T09-15 regret runner not built yet (needs T06 Reference/candidates and the T08 loop prompts)")


def test_T09_16_ablation_presets():
    """T09-16: T2-only-F5 … T2-only-F16 exist; their scenarios have factors F1..F4 + that one."""
    presets = pytest.importorskip("railway.gen.presets", reason="waiting for T04 (presets)")
    generator = pytest.importorskip("railway.gen.generator", reason="waiting for T04 (generator)")
    for f in ("F5", "F6", "F7", "F8", "F10", "F15", "F16"):
        name = f"T2-only-{f}"
        assert name in presets.PRESETS, f"{name} missing (T09 adds it to T04's presets)"
        sc = generator.generate(presets.PRESETS[name], 1)
        assert sorted(sc.factors) == sorted(["F1", "F2", "F3", "F4", f])
