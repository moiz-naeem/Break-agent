"""T06-18..21: the seed filter's four conditions, acceptance rates and the accepted-list file."""

import concurrent.futures
import functools
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from railway.gen.generator import generate
from railway.gen.presets import PRESETS
from railway.gen.seeds import candidate_seeds
from railway.ref import seed_filter
from railway.ref.seed_filter import POLICY_NAMES, evaluate_scenario
from railway.sim.constants import RULES_VERSION
from railway.util import content_hash

BASE = {name: 150.0 for name in POLICY_NAMES} | {"greedy": 100.0, "reference": 60.0}


def stub(costs, gridlocks=0, unfinished=0):
    def run(name, _scenario):
        counts = {"gridlocks": gridlocks if name == "reference" else 0}
        final = SimpleNamespace(cost=SimpleNamespace(total=costs[name]), counts=counts)
        return SimpleNamespace(final=final, unfinished=unfinished if name == "greedy" else 0, log=[name])

    return run


def fake_best(c_best, seen=None):
    def best(_scenario, reference_log):
        if seen is not None:
            seen.append(reference_log)
        plan = [{"tool": "end_shift", "args": {}}]
        return {"c_best": c_best, "c_best_kind": "proven", "c_best_source": "solver", "solver": {"status": "optimal"},
                "plan": plan}  # fmt: skip

    return best


@pytest.mark.parametrize(
    ("costs", "c_best", "gridlocks", "unfinished", "reason"),
    [
        (BASE, 70.0, 0, 0, None),  # C_best exactly 30% below greedy is enough (v0.3 §2.6)
        (BASE, 70.5, 0, 0, "best_not_30pct_better"),
        (BASE | {"reference": 95.0}, 50.0, 0, 0, None),  # the reference itself no longer needs the headroom
        (BASE, 50.0, 1, 0, "ref_gridlock"),
        (BASE | {"Malformed": 99.0}, 50.0, 0, 0, "exploit_beats_greedy"),
        (BASE | {"StopEarly": 100.0}, 50.0, 0, 0, None),  # an exploit equal to greedy still scores 0
        (BASE, 50.0, 0, 1, "greedy_unfinished"),
        (BASE, 50.0, 0, 0, None),
    ],
)
def test_T06_18_filter_conditions(costs, c_best, gridlocks, unfinished, reason):
    seen = []
    report = evaluate_scenario(None, runner=stub(costs, gridlocks, unfinished), best=fake_best(c_best, seen))
    assert report["accepted"] is (reason is None)
    assert report["reasons"] == ([] if reason is None else [reason])
    assert (report["c_greedy"], report["c_ref"]) == (costs["greedy"], costs["reference"])
    assert set(report["exploit_costs"]) == set(POLICY_NAMES) - {"greedy", "reference"}
    if reason in {"ref_gridlock", "exploit_beats_greedy", "greedy_unfinished"}:
        assert seen == [] and report["c_best"] is None and report["plan"] is None  # C_best is not computed
    else:
        assert seen == [["reference"]]  # C_best gets the reference's own episode log
        assert (report["c_best"], report["c_best_kind"], report["solver_status"]) == (c_best, "proven", "optimal")


def _accepted(preset, seed):
    return evaluate_scenario(generate(PRESETS[preset], seed))["accepted"]


@pytest.mark.slow
def test_T06_19_acceptance_rate():
    t1 = [("T1", s) for s in candidate_seeds("dev", count=30)]
    t2 = [("T2", s) for s in candidate_seeds("dev", count=20)]
    with concurrent.futures.ProcessPoolExecutor(os.cpu_count()) as pool:
        verdicts = list(pool.map(_accepted, *zip(*(t1 + t2), strict=True)))
    rate1, rate2 = sum(verdicts[:30]), sum(verdicts[30:])
    print(f"acceptance: T1 {rate1}/30, T2 {rate2}/20")
    assert rate1 >= 0.3 * 30 and rate2 >= 0.3 * 20


@pytest.mark.slow
def test_T06_20_accepted_file(tmp_path):
    cmd = [sys.executable, "-m", "railway.gen.cli", "accept", "--config", "T1", "--set", "dev", "--count", "5"]
    subprocess.run(cmd, cwd=tmp_path, check=True, env=os.environ | {"PYTHONPATH": os.getcwd()})
    data = json.loads((tmp_path / "seeds" / "T1-dev.json").read_text(encoding="utf-8"))
    assert data["header"]["rules_version"] == RULES_VERSION
    assert data["header"]["generator_version"] == seed_filter.generator_version()
    entries = data["accepted"]
    assert len(entries) == 5 and [e["seed"] for e in entries] == sorted(e["seed"] for e in entries)
    scenarios = [generate(PRESETS["T1"], e["seed"]) for e in entries]
    with concurrent.futures.ProcessPoolExecutor(5) as pool:
        reports = list(pool.map(evaluate_scenario, scenarios))
    for entry, scenario, report in zip(entries, scenarios, reports, strict=True):
        assert content_hash(scenario.model_dump(mode="json")) == entry["scenario_hash"]
        assert scenario.scenario_id == entry["scenario_id"]
        assert (report["c_greedy"], report["c_ref"]) == (entry["c_greedy"], entry["c_ref"])


def accepted_report(seed, kind="proven", status="optimal"):
    plan = [{"tool": "hold", "args": {"train": "L1", "until_tick": seed + 1}}]
    return {"accepted": True, "c_greedy": 2.0, "c_ref": 1.5, "c_best": 1.0, "c_best_kind": kind,
            "c_best_source": "solver", "solver_status": status, "plan": plan, "reasons": []}  # fmt: skip


def test_solver_timeouts_are_listed(tmp_path, monkeypatch):
    """A seed whose exact solve timed out stays best-known and is listed in the header."""
    def fake(scenario, runner=None):
        return accepted_report(scenario.seed, *(("best-known", "timeout") if scenario.seed % 2 else ()))

    monkeypatch.setattr(seed_filter, "evaluate_scenario", fake)
    entries = seed_filter.accepted_seeds("T1", "dev", 4, output_dir=tmp_path, workers=1)
    header = json.loads((tmp_path / "T1-dev.json").read_text(encoding="utf-8"))["header"]
    assert header["solver_timeouts"] == [e["seed"] for e in entries if e["seed"] % 2]
    assert {e["c_best_kind"] for e in entries if e["seed"] % 2} == {"best-known"}


def test_generator_version_ignores_line_endings(tmp_path, monkeypatch):
    """A CRLF checkout must hash like an LF one, or every Windows run treats the committed lists as stale."""
    gen = tmp_path / "railway" / "gen"
    gen.mkdir(parents=True)
    (gen / "generator.py").write_bytes(b"a = 1\nb = 2\n")
    monkeypatch.setattr(seed_filter, "__file__", str(tmp_path / "railway" / "ref" / "seed_filter.py"))
    lf = seed_filter.generator_version()
    (gen / "generator.py").write_bytes(b"a = 1\r\nb = 2\r\n")
    assert seed_filter.generator_version() == lf


def test_T06_21_cache_invalidation(tmp_path, monkeypatch):
    calls = []

    def fake(scenario, runner=None):
        calls.append(scenario.seed)
        return accepted_report(scenario.seed)

    monkeypatch.setattr(seed_filter, "evaluate_scenario", fake)
    first = seed_filter.accepted_seeds("T1", "dev", 3, output_dir=tmp_path, workers=1)
    assert len(first) == 3 and len(calls) == 3
    for entry in first:  # each accepted entry's plan is stored next to the list
        stored = seed_filter.load_plan("T1", "dev", entry["scenario_id"], tmp_path)
        assert stored == accepted_report(entry["seed"])["plan"]
    assert seed_filter.accepted_seeds("T1", "dev", 3, output_dir=tmp_path, workers=1) == first
    assert len(calls) == 3  # reused
    seed_filter.plan_path("T1", "dev", first[0]["scenario_id"], tmp_path).unlink()
    assert seed_filter.accepted_seeds("T1", "dev", 3, output_dir=tmp_path, workers=1) == first
    assert len(calls) == 6  # a missing plan file makes the list stale
    calls.clear()
    monkeypatch.setattr(seed_filter, "generator_version", lambda: "changed")
    assert seed_filter.accepted_seeds("T1", "dev", 3, output_dir=tmp_path, workers=1) == first
    assert len(calls) == 3  # recomputed


def test_T06_21b_batch_paths_agree(tmp_path, monkeypatch):
    """Serial, pooled (threads stand in for processes, so the fake applies) and `mapper` give one list."""
    accept = {0, 2, 3, 5}

    def fake(scenario, runner=None):
        ok = scenario.seed in accept
        return accepted_report(scenario.seed) if ok else {"accepted": False, "reasons": ["ref_gridlock"]}

    monkeypatch.setattr(seed_filter, "evaluate_scenario", fake)
    monkeypatch.setattr(seed_filter, "candidate_seeds", lambda _set: iter(range(100)), raising=False)
    monkeypatch.setattr("railway.gen.seeds.candidate_seeds", lambda _set: iter(range(100)))
    monkeypatch.setattr(seed_filter.concurrent.futures, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor)
    runs = [
        seed_filter.accepted_seeds("T1", "dev", 3, output_dir=tmp_path / "serial", workers=1),
        seed_filter.accepted_seeds("T1", "dev", 3, output_dir=tmp_path / "pool", workers=2),
        seed_filter.accepted_seeds(
            "T1",
            "dev",
            3,
            output_dir=tmp_path / "mapper",
            workers=4,
            mapper=functools.partial(map, seed_filter._evaluate_seed),
        ),
    ]
    assert [e["seed"] for e in runs[0]] == [0, 2, 3]
    assert runs[1] == runs[0] and runs[2] == runs[0]
