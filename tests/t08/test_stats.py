"""T08-22..29: aggregate statistics (DESIGN.md §6.2, PRD D16)."""

from __future__ import annotations

import json
import random

import pytest

from railway.eval import stats
from railway.eval.run import main


def row(sid, cg, cr, ca, rep, status="ok", **extra):
    r = {
        "scenario_id": sid,
        "rep": rep,
        "c_greedy": cg,
        "c_best": cr,
        "cost_total": ca,
        "status": status,
        "gridlocks": 0,
        "crew_timeouts": 0,
        "invalid_calls": 0,
        "format_errors": 0,
        "tool_calls": 10,
        "decision_points": 5,
        "tokens_in": 100,
        "tokens_out": 10,
        "error": None,
    }
    if status != "ok":
        r.update(cost_total=None, error="boom")
    r.update(extra)
    return r


def fixture_rows(a=(600, 600), b=(300, 300)):
    rows_a = [row("A", 1000, 500, ca, i) for i, ca in enumerate(a)]
    return rows_a + [row("B", 200, 100, ca, i) for i, ca in enumerate(b)]


def test_T08_22_ros():
    assert stats.score_ros(fixture_rows()) == pytest.approx(0.5)
    assert stats.summarize(fixture_rows())["score_ros"] == pytest.approx(0.5)


def test_T08_23_reward_mean_and_unclipped():
    rows = fixture_rows()
    assert stats.reward_mean(rows) == pytest.approx(0.4)
    assert stats.unclipped_mean(rows) == pytest.approx(-0.1)


def test_T08_24_cost_ratio():
    assert stats.cost_ratio(fixture_rows()) == pytest.approx(1.5)


def test_T08_25_rep_averaging():
    assert stats.score_ros(fixture_rows(a=(500, 700))) == pytest.approx(0.5)


def test_T08_26_flagged():
    rows = fixture_rows() + [row("C", 100, 50, 40, 0), row("C", 100, 50, 60, 1)]
    assert stats.flagged(rows) == ["C"]
    assert stats.flagged(fixture_rows()) == []
    # v0.3: reward is -cost; normalised score has no clip at 1.0 (seed C reps: 1.2 and 0.8 -> mean 1.0).
    assert stats.normalized_mean(rows) == pytest.approx(0.6)
    assert stats.neg_cost_mean(fixture_rows()) == pytest.approx(-450.0)


def _synthetic(n=20, seed=1):
    rng = random.Random(seed)
    rows = []
    for s in range(n):
        cg = rng.randint(800, 1200)
        cr = cg - rng.randint(200, 500)
        for rep in range(3):
            rows.append(row(f"S{s:02d}", cg, cr, rng.randint(cr - 50, cg + 100), rep))
    return rows


def test_T08_27_bootstrap_ci():
    rows = _synthetic()
    for name in ("score_ros", "reward_mean", "unclipped_mean", "cost_ratio"):
        point = getattr(stats, name)(rows)
        lo, hi = stats.bootstrap_ci(rows, name)
        assert lo <= point <= hi, name
        assert lo < hi
        assert stats.bootstrap_ci(rows, name) == (lo, hi)  # rng seed 0: identical
    assert stats.bootstrap_ci(rows, "score_ros", seed=1) != stats.bootstrap_ci(rows, "score_ros")
    same = [row(f"S{s}", 1000, 500, 700, 0) for s in range(20)]
    lo, hi = stats.bootstrap_ci(same, "score_ros")
    assert lo == hi == pytest.approx(0.6)


def test_T08_27_resamples_seeds_not_rows():
    """Reps of one seed move together: a seed with many reps has the weight of one seed."""
    rows = [row("A", 1000, 500, 600, r) for r in range(10)] + [row("B", 200, 100, 300, 0)]
    assert stats.reward_mean(rows) == pytest.approx(0.4)  # (0.8 + 0.0) / 2, not weighted by reps
    lo, hi = stats.bootstrap_ci(rows, "reward_mean", n_boot=2000)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(0.8)


def test_T08_28_errors_excluded():
    rows = fixture_rows()[:3] + [row("B", 200, 100, None, 1, status="error")]
    rows.append(row("C", 100, 50, 60, 0))
    s = stats.summarize(rows)
    assert s["errors"] == 1 and s["planned"] == 5 and s["episodes"] == 4
    assert s["incomplete"] is True
    ok_only = stats.summarize([r for r in rows if r["status"] == "ok"])
    assert ok_only["incomplete"] is False and ok_only["errors"] == 0
    assert s["score_ros"] == ok_only["score_ros"]


def test_T08_28_summary_contents():
    rows = fixture_rows()
    rows[0].update(gridlocks=2, format_errors=4, tokens_in=300, tokens_out=None)
    s = stats.summarize(rows)
    for key in (
        "score_ros",
        "reward_mean",
        "unclipped_mean",
        "cost_ratio",
        "flagged",
        "ci",
        "episodes",
        "errors",
        "planned",
        "incomplete",
        "seeds",
    ):
        assert key in s, key
    assert set(s["ci"]) == {"score_ros", "reward_mean", "unclipped_mean", "cost_ratio"}
    means = s["means"]
    assert means["gridlocks"] == pytest.approx(0.5) and means["format_errors"] == pytest.approx(1.0)
    assert means["tool_calls"] == 10 and means["decision_points"] == 5
    assert means["tokens_in"] == pytest.approx(150.0) and means["tokens_out"] == pytest.approx(10.0)
    assert s["seeds"] == 2
    json.dumps(s)  # serializable for --json


def test_T08_28_duplicate_rows_prefer_ok():
    rows = fixture_rows() + [row("A", 1000, 500, None, 0, status="error")]
    s = stats.summarize(rows)
    assert s["errors"] == 0 and s["episodes"] == 4 and s["planned"] == 4


def test_T08_29_paired_diff():
    a = fixture_rows()
    b = fixture_rows(a=(500, 500), b=(200, 200))
    assert stats.score_ros(b) == pytest.approx(500 / 600)
    res = stats.paired_diff(a, b, "score_ros")
    assert res["diff"] == pytest.approx(1 / 3)
    assert res["ci"][0] > 0
    assert res["ci"][0] == pytest.approx(0.2) and res["ci"][1] == pytest.approx(1.0)
    assert res["common_seeds"] == 2


def test_T08_29_paired_diff_common_seeds_only():
    a = fixture_rows()
    b = fixture_rows(a=(500, 500), b=(200, 200)) + [row("Z", 100, 50, 0, 0)]
    assert stats.paired_diff(a, b, "score_ros")["diff"] == pytest.approx(1 / 3)


def test_T08_29_compare_cli(tmp_path, capsys):
    pa, pb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    pa.write_text("".join(json.dumps(r) + "\n" for r in fixture_rows()))
    pb.write_text("".join(json.dumps(r) + "\n" for r in fixture_rows(a=(500, 500), b=(200, 200))))
    assert main(["compare", str(pa), str(pb), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["score_ros"]["diff"] == pytest.approx(1 / 3)
    assert main(["compare", str(pa), str(pb)]) == 0
    assert "score_ros" in capsys.readouterr().out
