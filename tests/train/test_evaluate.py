"""Unit tests for `railway.train.evaluate`: seed splits, episode loop rows, per-seed weighting, paired CIs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from railway.train.env import is_net_train_seed, is_val_seed
from railway.train.evaluate import (
    DROPPED_FIELDS,
    EvalCandidate,
    dedupe_rows,
    main,
    paired_compare,
    parse_candidates,
    play_eval_job,
    select_eval_seeds,
    summarize_rows,
)


class _ScriptedClient:
    provider = "vllm"

    def __init__(self, model: str = "qwen38-nothink"):
        self.model = model
        self._turn = 0

    def chat(self, messages, tools):
        self._turn += 1
        name = "end_shift" if self._turn >= 2 else "advance_to_next_event"
        return {
            "tool_calls": [{"id": f"call_{self._turn}", "name": name, "arguments_raw": "{}"}],
            "content": None,
            "reasoning": None,
            "finish_reason": "tool_calls",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 12,
                "input_tokens": 100,
                "output_tokens": 12,
                "reasoning_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
                "cost_usd": 0.001,
            },
        }

    def close(self) -> None:
        pass


class _FailingClient:
    provider = "vllm"
    model = "qwen38-nothink"

    def chat(self, messages, tools):
        raise RuntimeError("boom")

    def close(self) -> None:
        pass


def test_parse_candidates_variants():
    cands = parse_candidates(["base", "sft=/adapters/run1/sft", "/adapters/run2/dagger1"], thinking=False)
    assert cands == [
        EvalCandidate("qwen38-nothink", None),
        EvalCandidate("sft", "/adapters/run1/sft"),
        EvalCandidate("run2-dagger1", "/adapters/run2/dagger1"),
    ]
    assert parse_candidates(["base"], thinking=True) == [EvalCandidate("qwen38-think", None)]
    with pytest.raises(ValueError, match="duplicate"):
        parse_candidates(["sft=/adapters/a/sft", "sft=/adapters/b/sft"])


def test_select_eval_seeds_strict_disjointness():
    for tier in ("T1", "T2", "T3"):
        val = {e["seed"] for e in select_eval_seeds(tier, "val")}
        train = {e["seed"] for e in select_eval_seeds(tier, "train")}
        dev = {e["seed"] for e in select_eval_seeds(tier, "dev")}
        test = {e["seed"] for e in select_eval_seeds(tier, "test")}
        assert val and train and dev and test
        assert all(is_val_seed(s) for s in val)
        assert all(is_net_train_seed(s) for s in train)
        assert val.isdisjoint(train)
        assert val.isdisjoint(dev)
        assert val.isdisjoint(test)
        assert train.isdisjoint(dev)
        assert train.isdisjoint(test)
        assert dev.isdisjoint(test)
    assert len(select_eval_seeds("T1", "val", limit=3)) == 3


def test_play_eval_job_emits_net_bench_compatible_row(tmp_path: Path):
    entry = select_eval_seeds("T1", "val", limit=1)[0]
    job = {**entry, "run": "unit-eval", "model": "qwen38-nothink", "rep": 0}
    row = play_eval_job(job, _ScriptedClient, scenarios_dir=None, transcript_root=tmp_path / "transcripts")
    assert row["status"] == "ok"
    assert row["tier"] == "T1" and row["preset"] == "T1"
    assert row["reward"] == -row["cost_total"]
    assert set(row["components"]) == {"C1", "C2", "C3", "C4", "C5"}
    assert "on_time_pct" in row["kpis"] and "clean_shift" in row["kpis"]
    assert "gridlocks" in row["counts"]
    assert all(k not in row for k in DROPPED_FIELDS)
    assert row["tokens_in"] == 200 and row["tokens_out"] == 24

    err_row = play_eval_job(job, _FailingClient, scenarios_dir=None, transcript_root=tmp_path / "transcripts_err")
    assert err_row["status"] == "error" and "boom" in err_row["error"]


def _row(
    model: str,
    tier: str,
    seed: int,
    cost: float | None,
    *,
    rep: int = 0,
    status: str = "ok",
    clean: bool = True,
    invalid: int = 0,
    calls: int = 50,
    only_preset: bool = False,
) -> dict:
    out: dict = {
        "model": model,
        "preset": tier,
        "seed": seed,
        "rep": rep,
        "scenario_id": f"{tier}-{seed}",
        "status": status,
        "tool_calls": calls,
        "invalid_calls": invalid,
        "refusals": 1,
        "format_errors": 0,
        "tokens_in": 1000,
        "tokens_out": 100,
        "tokens_reasoning": 0,
        "tokens_cached": 500,
        "cost_usd": 0.02,
        "wall_s": 12.0,
    }
    if not only_preset:
        out["tier"] = tier
    if status == "ok" and cost is not None:
        out |= {
            "cost_total": cost,
            "reward": -cost,
            "cost_by_component": {
                "pax_lateness": cost * 0.5,
                "missed_calls": cost * 0.5,
                "unfinished": 0.0,
                "freight_lateness": 0.0,
                "crew_timeout": 0.0,
                "gridlock": 0.0,
                "operations": 0.0,
                "total": cost,
            },
            "kpis": {
                "on_time_pct": 85.0 if clean else 60.0,
                "passenger_delay_minutes": cost * 0.2,
                "freight_delivered_pct": 100.0,
                "clean_shift": clean,
            },
            "counts": {"gridlocks": 0 if clean else 1, "crew_timeouts": 0, "unfinished": 0},
        }
    return out


def test_dedupe_and_equal_per_seed_weighting():
    rows = [
        _row("m", "T1", 10, None, rep=0, status="error"),
        _row("m", "T1", 10, 100.0, rep=0, status="ok"),  # supersedes earlier error on rep 0
        _row("m", "T1", 10, 200.0, rep=1, status="ok"),  # seed 10 mean cost = 150
        _row("m", "T1", 20, 50.0, rep=0, status="ok"),   # seed 20 mean cost = 50 (only 1 rep)
    ]
    deduped = dedupe_rows(rows)
    assert len(deduped) == 3
    summary = summarize_rows(rows, n_boot=500)["m|T1"]
    assert summary["seeds"] == 2 and summary["ok"] == 3 and summary["infra_errors"] == 0
    # Equal per-seed weight: (150 + 50) / 2 = 100, NOT episode average (100+200+50)/3 = 116.67
    assert summary["mean_cost"] == pytest.approx(100.0)
    assert summary["mean_reward"] == pytest.approx(-100.0)


def test_paired_compare_verdicts_and_cli_report(tmp_path: Path):
    base_rows = [
        _row("qwen38-nothink", "T1", s, 200.0 + s, clean=False, invalid=5, only_preset=True) for s in range(1, 13)
    ]
    good_rows = [_row("sft-v1", "T1", s, 100.0 + s, clean=True, invalid=1) for s in range(1, 13)]
    noisy_rows = [
        _row("sft-noisy", "T1", s, (150.0 + s) if s % 2 == 0 else (255.0 + s), clean=False)
        for s in range(1, 13)
    ]
    # >10% error rate on incomplete candidate must never claim significant_improvement
    inc_rows = [
        _row("sft-inc", "T1", s, 50.0 if s <= 9 else None, status="ok" if s <= 9 else "error")
        for s in range(1, 13)
    ]

    all_rows = base_rows + good_rows + noisy_rows + inc_rows
    p_good = paired_compare(all_rows, base_model="qwen38-nothink", cand_model="sft-v1", n_boot=2000)["ALL"]
    assert p_good["common_seeds"] == 12
    assert p_good["wins"] == 12 and p_good["losses"] == 0
    assert p_good["reward_delta"] == pytest.approx(100.0)
    assert p_good["cost_delta"] == pytest.approx(-100.0)
    assert p_good["significant_improvement"] is True
    assert p_good["verdict"].startswith("IMPROVED")
    assert p_good["kpi_deltas"]["clean_shift_rate"] == pytest.approx(1.0)
    assert p_good["failure_deltas"]["invalid_rate"] == pytest.approx((1 - 5) / 50)

    p_noisy = paired_compare(all_rows, base_model="qwen38-nothink", cand_model="sft-noisy", n_boot=2000)["ALL"]
    assert p_noisy["significant_improvement"] is False
    assert p_noisy["verdict"].startswith("INCONCLUSIVE")

    p_inc = paired_compare(all_rows, base_model="qwen38-nothink", cand_model="sft-inc", n_boot=2000)["ALL"]
    assert p_inc["cand_incomplete"] is True
    assert p_inc["significant_improvement"] is False
    assert p_inc["verdict"].startswith("INCOMPLETE")

    f_base = tmp_path / "base.jsonl"
    f_cands = tmp_path / "cands.jsonl"
    f_base.write_text("\n".join(json.dumps(r) for r in base_rows) + "\n", encoding="utf-8")
    f_cands.write_text("\n".join(json.dumps(r) for r in (good_rows + noisy_rows)) + "\n", encoding="utf-8")
    out_dir = tmp_path / "report_out"
    rc = main(
        [
            "report",
            "--rows",
            str(f_base),
            str(f_cands),
            "--base",
            "qwen38-nothink",
            "--out",
            str(out_dir),
            "--n-boot",
            "500",
        ]
    )
    assert rc == 0
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "paired.json").is_file()
    md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "IMPROVED (95% paired CI > 0)" in md
    assert "sft-v1" in md
