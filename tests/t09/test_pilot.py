"""T09 §5.1 pilot gate tests. All rows here are SYNTHETIC (tests/t09/synthetic.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from railway.eval.pilot import GateResult, PilotLog, evaluate_gate, next_actions, parse_pilot_md, render_pilot_md
from tests.helpers import REPO
from tests.t09.synthetic import model_rows, ros_test_only


def _gate(small, big, ratio=0.6, **kw):
    return evaluate_gate(
        model_rows("small", small, ratio=ratio, **kw), model_rows("big", big, ratio=ratio), score_fn=ros_test_only
    )


@pytest.mark.parametrize(
    ("small", "big", "ratio", "failed"),
    [
        (0.2, 0.6, 0.6, ()),
        (0.2, 0.85, 0.6, ("b",)),
        (0.4, 0.6, 0.6, ("a",)),
        (0.2, 0.6, 0.75, ("c",)),
        (0.7, 0.9, 0.8, ("a", "b", "c")),
    ],
)
def test_T09_01_gate_logic(small, big, ratio, failed):
    """T09-01: PASS / b fails / a fails / c fails; GateResult names the failing conditions."""
    g = _gate(small, big, ratio)
    assert isinstance(g, GateResult)
    assert g.failed == failed
    assert g.passed == (failed == ())
    assert g.small == pytest.approx(small)
    assert g.big == pytest.approx(big)
    assert g.gap == pytest.approx(big - small)
    assert g.ref_ratio == pytest.approx(ratio)
    assert (g.a, g.b, g.c) == ("a" not in failed, "b" not in failed, "c" not in failed)
    assert g.seeds == tuple(f"T1-synth{s:07d}" for s in range(1, 11))


def test_T09_01_gate_boundaries_inclusive():
    """T09-01: the thresholds are inclusive (gap == 0.3, big == 0.8, ratio == 0.7 all pass)."""
    assert _gate(0.5, 0.8, 0.7).failed == ()


def test_T09_01_gate_sums_over_seeds_not_mean_of_ratios():
    """T09-01 (c): Σc_ref ≤ 0.7 × Σc_greedy over the seeds, each seed counted once (not per row)."""
    from tests.t09.synthetic import make_row

    # Seed 1: cg 1000, cr 500. Seed 2: cg 100, cr 90. Σcr/Σcg = 590/1100 ≈ 0.536 → c holds,
    # although the mean of the per-seed ratios (0.5 + 0.9)/2 = 0.7 is at the line.
    def rows(model, score):
        return [make_row(model=model, seed=1, c_greedy=1000, c_ref=500, score=score, rep=rep) for rep in (0, 1)] + [
            make_row(model=model, seed=2, c_greedy=100, c_ref=90, score=score)
        ]

    g = evaluate_gate(rows("small", 0.1), rows("big", 0.5), score_fn=ros_test_only)
    assert g.sum_c_ref == pytest.approx(590)
    assert g.sum_c_greedy == pytest.approx(1100)
    assert g.c


def test_T09_01_gate_rejects_mismatched_seeds():
    """T09-01: small and big must be scored on the same seeds."""
    with pytest.raises(ValueError, match="same seeds"):
        evaluate_gate(
            model_rows("small", 0.2, seeds=range(1, 11)),
            model_rows("big", 0.6, seeds=range(2, 12)),
            score_fn=ros_test_only,
        )


def test_T09_01_gate_excludes_error_rows():
    """T09-01: error rows are infra failures, not scores; they are excluded and counted."""
    small = model_rows("small", 0.2) + model_rows("small", 0.0, seeds=[1], status="error", rep=1)
    g = evaluate_gate(small, model_rows("big", 0.6), score_fn=ros_test_only)
    assert g.small == pytest.approx(0.2)
    assert g.errors == {"small": 1, "big": 0}


@pytest.mark.parametrize(
    ("small", "big", "ratio", "invalid", "expected"),
    [
        (0.2, 0.6, 0.6, 0, []),
        (0.2, 0.85, 0.6, 0, ["raise_difficulty"]),
        (0.4, 0.6, 0.6, 5, ["raise_difficulty"]),  # 5% of tool calls invalid: not a usability artifact
        (0.4, 0.6, 0.6, 11, ["fix_tool_usability"]),  # 11% > 10%
        (0.2, 0.6, 0.75, 0, ["investigate_reference"]),
        (0.7, 0.9, 0.8, 0, ["raise_difficulty", "investigate_reference"]),
    ],
)
def test_T09_01_if_it_fails(small, big, ratio, invalid, expected):
    """T09-01 / §2.1 "If it fails": the next step per failing condition."""
    small_rows = model_rows("small", small, ratio=ratio, invalid_calls=invalid)
    g = evaluate_gate(small_rows, model_rows("big", big, ratio=ratio), score_fn=ros_test_only)
    assert next_actions(g, small_rows) == expected


def test_T09_01_if_it_fails_counts_format_errors():
    """T09-01: loop rows' format_errors count toward the small model's usability rate."""
    small_rows = model_rows("small", 0.4, harness="loop", invalid_calls=6, format_errors=5)
    g = evaluate_gate(small_rows, model_rows("big", 0.6), score_fn=ros_test_only)
    assert next_actions(g, small_rows) == ["fix_tool_usability"]


def test_T09_01_pilot_log_escalates_after_three_failures():
    """T09-01 / §2.1: at most 3 iterations, each recorded; after 3 failures, stop and escalate."""
    log = PilotLog()
    fail = _gate(0.4, 0.6)
    assert log.status == "not_run"
    log.record(fail, "initial pilot (T1)")
    assert log.status == "iterate"
    log.record(fail, "T1+ = T1 with F5, F7, F8")
    log.record(fail, "T1+ with F6 and F10")
    assert log.status == "escalate"
    with pytest.raises(RuntimeError, match="escalate"):
        log.record(fail, "a fourth try")
    assert [it.change for it in log.iterations] == [
        "initial pilot (T1)",
        "T1+ = T1 with F5, F7, F8",
        "T1+ with F6 and F10",
    ]


def test_T09_01_pilot_log_pass_stops():
    """T09-01: a PASS ends the pilot; nothing more may be recorded."""
    log = PilotLog()
    log.record(_gate(0.4, 0.6), "initial pilot (T1)")
    log.record(_gate(0.2, 0.6), "T1+ = T1 with F5, F7, F8")
    assert log.status == "pass"
    assert log.verdict == "PASS"
    with pytest.raises(RuntimeError):
        log.record(_gate(0.2, 0.6), "again")


def _synthetic_record():
    small, big = model_rows("small", 0.2), model_rows("big", 0.6)
    log = PilotLog()
    log.record(_gate(0.4, 0.6), "initial pilot (T1)")
    log.record(evaluate_gate(small, big, score_fn=ros_test_only), "T1+ = T1 with F5, F7, F8")
    md = render_pilot_md(
        models={"small": "synthetic-small", "big": "synthetic-big"},
        date="2026-01-01",
        small_rows=small,
        big_rows=big,
        log=log,
        score_fn=ros_test_only,
    )
    return md, log


def test_T09_02_pilot_record_roundtrip_synthetic():
    """T09-02 (synthetic): the rendered record has the per-seed table, the three numbers,
    every iteration and a final Verdict line, and parses back to the gate's numbers."""
    md, log = _synthetic_record()
    parsed = parse_pilot_md(md)
    final = log.iterations[-1].gate
    assert parsed["verdict"] == "PASS"
    assert parsed["gap"] == pytest.approx(final.gap, abs=5e-4)
    assert parsed["big"] == pytest.approx(final.big, abs=5e-4)
    assert parsed["ref_ratio"] == pytest.approx(final.ref_ratio, abs=5e-4)
    assert parsed["iterations"] == ["initial pilot (T1)", "T1+ = T1 with F5, F7, F8"]
    assert md.rstrip().splitlines()[-1] == "Verdict: PASS"
    for s in range(1, 11):
        assert f"T1-synth{s:07d}" in md
    assert "@" not in md  # names only, no keys or emails


def test_T09_02_pilot_record_deterministic():
    """T09-02: same inputs → same text (no wall clock)."""
    assert _synthetic_record()[0] == _synthetic_record()[0]


def test_T09_02_pilot_record():
    """T09-02: docs/pilot.md says PASS and its numbers equal evaluate_gate on the real pilot results."""
    pilot_md = REPO / "docs" / "pilot.md"
    small_f, big_f = REPO / "results" / "pilot-small.jsonl", REPO / "results" / "pilot-big.jsonl"
    if not (pilot_md.exists() and small_f.exists() and big_f.exists()):
        pytest.skip("waiting for the real pilot runs (blocking input: models, PRD §9)")
    pytest.importorskip("railway.eval.stats")

    def load(p: Path):
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    parsed = parse_pilot_md(pilot_md.read_text(encoding="utf-8"))
    g = evaluate_gate(load(small_f), load(big_f))
    assert parsed["verdict"] == "PASS"
    assert parsed["gap"] == pytest.approx(g.gap, abs=5e-4)
    assert parsed["big"] == pytest.approx(g.big, abs=5e-4)
    assert parsed["ref_ratio"] == pytest.approx(g.ref_ratio, abs=5e-4)
    assert len(parsed["iterations"]) >= 1


def test_T09_01_default_score_uses_stats():
    """T09-01: without score_fn, evaluate_gate scores with T08's stats.py (D16)."""
    pytest.importorskip("railway.eval.stats")
    small, big = model_rows("small", 0.2), model_rows("big", 0.6)
    assert evaluate_gate(small, big).gap == pytest.approx(evaluate_gate(small, big, score_fn=ros_test_only).gap)
