"""T08-17..21: run manifests, the runner (loop + Harbor harnesses), resume, concurrency."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from railway.eval import run as run_mod
from railway.eval.manifest import load_manifest
from railway.eval.run import LoopDeps, main, run_manifest
from railway.loop.fake_model import FakeModelServer, policy
from railway.loop.model_client import ModelClient
from railway.sim.engine import Simulator

from .conftest import INSTRUCTION, make_dataset, real_session_cls
from .fakes import FakeSession, fake_decide, fake_reward_from, fake_static

REPO = Path(__file__).resolve().parents[2]

BASE = {
    "name": "tmp",
    "harness": "loop",
    "mode": "windowed",
    "model": {"id": "fake", "name": "fake-model", "base_url": "http://127.0.0.1:1/v1"},
    "temperature": None,
    "max_tokens": 1024,
    "scenarios": {"preset": "T1", "set": "dev", "count": 3},
    "reps": 2,
    "concurrency": 1,
    "tools_disabled": [],
    "diagnostics": False,
}


def _write(tmp_path: Path, data: dict, stem: str | None = None) -> Path:
    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    path = runs / f"{stem or data['name']}.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _bad_cases():
    both = copy.deepcopy(BASE) | {"dataset": "datasets/x"}
    neither = {k: v for k, v in BASE.items() if k != "scenarios"}
    harbor_no_model = {k: v for k, v in BASE.items() if k != "scenarios"} | {"harness": "harbor", "dataset": "d"}
    harbor_scenarios = copy.deepcopy(BASE) | {"harness": "harbor"}
    harbor_scenarios["model"] = dict(BASE["model"], harbor_model="openrouter/x")
    unknown = copy.deepcopy(BASE) | {"temprature": 0.5}
    unknown_model_key = copy.deepcopy(BASE)
    unknown_model_key["model"] = dict(BASE["model"], provider="x")
    bad_mode = copy.deepcopy(BASE) | {"mode": "sideways"}
    loop_no_url = copy.deepcopy(BASE)
    loop_no_url["model"] = {"id": "fake", "name": "fake-model"}
    harbor_ok = {k: v for k, v in BASE.items() if k != "scenarios"} | {"harness": "harbor", "dataset": "d"}
    harbor_ok["model"] = dict(BASE["model"], harbor_model="openrouter/x")
    return {
        "harbor_tools_disabled": (harbor_ok | {"tools_disabled": ["stringline"]}, None),
        "disabled_unknown_tool": (copy.deepcopy(BASE) | {"tools_disabled": ["stringlines"]}, None),
        "disabled_peekable_tool": (copy.deepcopy(BASE) | {"tools_disabled": ["status"]}, None),
        "harbor_diagnostics": (harbor_ok | {"diagnostics": True}, None),
        "both_dataset_and_scenarios": (both, None),
        "neither": (neither, None),
        "harbor_without_harbor_model": (harbor_no_model, None),
        "harbor_with_scenarios": (harbor_scenarios, None),
        "unknown_key": (unknown, None),
        "unknown_model_key": (unknown_model_key, None),
        "bad_mode": (bad_mode, None),
        "loop_without_base_url": (loop_no_url, None),
        "name_not_stem": (copy.deepcopy(BASE), "other-name"),
    }


@pytest.mark.parametrize("case", list(_bad_cases()))
def test_T08_17_manifest_validation(tmp_path, case):
    data, stem = _bad_cases()[case]
    with pytest.raises(ValueError):
        load_manifest(_write(tmp_path, data, stem))


def test_T08_17_manifest_valid_and_defaults(tmp_path):
    m = load_manifest(_write(tmp_path, BASE))
    assert m.name == "tmp" and m.reps == 2 and m.temperature is None
    minimal = {
        "name": "mini",
        "harness": "loop",
        "model": {"id": "s", "name": "n", "base_url": "http://x/v1"},
        "dataset": "datasets/x",
    }
    m2 = load_manifest(_write(tmp_path, minimal))
    assert m2.reps == 3 and m2.mode == "windowed" and m2.max_tokens == 1024 and m2.tools_disabled == []
    assert m2.diagnostics is False


@pytest.mark.parametrize("stem", ["example-loop", "example-harbor"])
def test_T08_17_example_manifests_validate(stem):
    m = load_manifest(REPO / "runs" / f"{stem}.yaml")
    assert m.harness == stem.split("-")[1]


# --- loop harness with the fake session -----------------------------------------------------------


class _Clients:
    """Starts one fake model server per episode, driven by `decide` against that episode's session."""

    def __init__(self, decide):
        self.decide = decide
        self.servers: list[FakeModelServer] = []

    def __call__(self, session, episode):
        srv = FakeModelServer(policy(self.decide, session, Simulator.load(episode.scenario).static_info())).start()
        self.servers.append(srv)
        return ModelClient(srv.base_url, "fake", None)

    def stop(self):
        for srv in self.servers:
            srv.stop()


@pytest.fixture
def fake_clients():
    clients = _Clients(fake_decide)
    yield clients
    clients.stop()


class _CountingSession(FakeSession):
    created = 0

    def __init__(self, *a, **kw):
        type(self).created += 1
        super().__init__(*a, **kw)


def _fake_decide_cost(mini3) -> float:
    session = FakeSession(mini3)
    with FakeModelServer(policy(fake_decide, session, fake_static(mini3))) as srv:
        from railway.loop.runner import play_episode

        return play_episode(
            mini3, ModelClient(srv.base_url, "fake", None), "windowed", session=session, instruction=INSTRUCTION
        ).cost_total


def _loop_manifest(tmp_path, dataset, **over):
    data = {k: v for k, v in BASE.items() if k != "scenarios"} | {"dataset": str(dataset)} | over
    return load_manifest(_write(tmp_path, data))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_T08_18_loop_run_writes_rows_fake(tmp_path, fake_clients, mini3, capsys):
    cost = _fake_decide_cost(mini3)
    dataset = make_dataset(tmp_path / "ds", 3, c_greedy=cost, c_ref=cost - 10)
    manifest = _loop_manifest(tmp_path, dataset, concurrency=2)
    deps = LoopDeps(
        session_cls=FakeSession, score_fn=fake_reward_from, client_factory=fake_clients, instruction=INSTRUCTION
    )
    results = tmp_path / "results"
    report = run_manifest(manifest, results_dir=results, deps=deps)
    rows = _rows(results / "tmp.jsonl")
    assert report["executed"] == 6 and len(rows) == 6
    assert sorted((r["scenario_id"], r["rep"]) for r in rows) == sorted(
        (f"FIX-mini3-{i}", rep) for i in range(3) for rep in range(2)
    )
    for r in rows:
        assert r["status"] == "ok" and r["run"] == "tmp" and r["model"] == "fake" and r["harness"] == "loop"
        assert r["score_unclipped"] == 0.0 and r["score_clipped"] == 0.0
        assert r["cost_ratio"] == pytest.approx(cost / (cost - 10))
        assert r["c_greedy"] == cost and r["seed"] in (0, 1, 2)
    for i in range(3):
        for rep in range(2):
            assert (results / "tmp" / "transcripts" / f"FIX-mini3-{i}-r{rep}.jsonl").exists()
    assert main(["summarize", str(results / "tmp.jsonl"), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["score_ros"] == 0.0 and summary["episodes"] == 6 and summary["errors"] == 0
    assert main(["summarize", str(results / "tmp.jsonl")]) == 0
    assert "score_ros" in capsys.readouterr().out


def test_T08_19_resume_fake(tmp_path, fake_clients, mini3):
    dataset = make_dataset(tmp_path / "ds", 3)
    manifest = _loop_manifest(tmp_path, dataset)
    _CountingSession.created = 0
    deps = LoopDeps(
        session_cls=_CountingSession, score_fn=fake_reward_from, client_factory=fake_clients, instruction=INSTRUCTION
    )
    results = tmp_path / "results"
    run_manifest(manifest, results_dir=results, deps=deps)
    assert _CountingSession.created == 6
    path = results / "tmp.jsonl"
    rows = _rows(path)
    kept = rows[2:]
    kept[0] = dict(kept[0], status="error", error="ModelError: boom")
    path.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
    _CountingSession.created = 0
    report = run_manifest(manifest, results_dir=results, deps=deps)
    assert _CountingSession.created == 3 and report["executed"] == 3
    final = _rows(path)
    ok = [r for r in final if r["status"] == "ok"]
    assert len(ok) == 6 and len({(r["scenario_id"], r["rep"]) for r in ok}) == 6
    # summarize sees the superseded error row as gone
    from railway.eval.stats import summarize

    s = summarize(final)
    assert s["errors"] == 0 and s["episodes"] == 6


def test_T08_21_concurrency_safe(tmp_path, fake_clients):
    dataset = make_dataset(tmp_path / "ds", 4)
    manifest = _loop_manifest(tmp_path, dataset, concurrency=4)
    deps = LoopDeps(
        session_cls=FakeSession, score_fn=fake_reward_from, client_factory=fake_clients, instruction=INSTRUCTION
    )
    run_manifest(manifest, results_dir=tmp_path / "results", deps=deps)
    text = (tmp_path / "results" / "tmp.jsonl").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert text.endswith("\n") and len(lines) == 8
    rows = [json.loads(line) for line in lines]  # every line is one valid JSON object
    assert len({(r["scenario_id"], r["rep"]) for r in rows}) == 8
    assert all(r["status"] == "ok" for r in rows)


def test_T08_21_infra_error_becomes_error_row(tmp_path):
    """A crash inside one episode (not a score) becomes an `error` row; the others still run."""
    dataset = make_dataset(tmp_path / "ds", 2)
    manifest = _loop_manifest(tmp_path, dataset, reps=1)

    def broken_client(session, episode):
        if episode.scenario.scenario_id.endswith("-0"):
            raise RuntimeError("cannot reach endpoint")
        return _Clients(fake_decide)(session, episode)

    deps = LoopDeps(
        session_cls=FakeSession, score_fn=fake_reward_from, client_factory=broken_client, instruction=INSTRUCTION
    )
    run_manifest(manifest, results_dir=tmp_path / "results", deps=deps)
    rows = {r["scenario_id"]: r for r in _rows(tmp_path / "results" / "tmp.jsonl")}
    assert rows["FIX-mini3-0"]["status"] == "error" and "cannot reach" in rows["FIX-mini3-0"]["error"]
    assert rows["FIX-mini3-0"]["cost_total"] is None
    assert rows["FIX-mini3-1"]["status"] == "ok"


# --- loop harness with the real modules (switch on once T04-T07 land) ------------------------------


def _real_greedy_clients():
    real_session_cls()
    greedy = pytest.importorskip("railway.policies.baselines", reason="waits for T05 policies")
    if not hasattr(greedy.Greedy, "decide"):
        pytest.skip("waits for Greedy.decide (T08 refactor of T05's greedy.py)")
    pytest.importorskip("railway.harbor.verify", reason="waits for T07 reward_from")
    pytest.importorskip("railway.gen.generator", reason="waits for T04 generator")
    pytest.importorskip("railway.ref.seed_filter", reason="waits for T06 accepted lists")
    return _Clients(greedy.Greedy().decide)


def test_T08_18_loop_run_writes_rows(tmp_path, capsys):
    clients = _real_greedy_clients()
    try:
        manifest = load_manifest(_write(tmp_path, BASE))
        run_manifest(manifest, results_dir=tmp_path / "results", deps=LoopDeps(client_factory=clients))
    finally:
        clients.stop()
    rows = _rows(tmp_path / "results" / "tmp.jsonl")
    assert len(rows) == 6
    assert all(r["status"] == "ok" and r["score_unclipped"] == 0.0 and r["score_clipped"] == 0.0 for r in rows)
    assert main(["summarize", str(tmp_path / "results" / "tmp.jsonl")]) == 0
    print(capsys.readouterr().out)  # pasted into the gate record


def test_T08_19_resume(tmp_path):
    clients = _real_greedy_clients()
    try:
        manifest = load_manifest(_write(tmp_path, BASE))
        results = tmp_path / "results"
        run_manifest(manifest, results_dir=results, deps=LoopDeps(client_factory=clients))
        path = results / "tmp.jsonl"
        rows = _rows(path)[2:]
        rows[0] = dict(rows[0], status="error", error="x")
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        report = run_manifest(manifest, results_dir=results, deps=LoopDeps(client_factory=clients))
    finally:
        clients.stop()
    assert report["executed"] == 3
    assert len([r for r in _rows(path) if r["status"] == "ok"]) == 6


# --- Harbor harness --------------------------------------------------------------------------------


def _details(sid, cost):
    return {
        "scenario_id": sid,
        "rules_version": "0.1-provisional",
        "cost_total": cost,
        "cost_by_component": {
            "pax_lateness": cost,
            "freight_lateness": 0.0,
            "crew_timeout": 0.0,
            "unfinished": 0.0,
            "total": cost,
        },
        "c_greedy": 1300.0,
        "c_ref": 100.0,
        "score_unclipped": (1300.0 - cost) / 1200.0,
        "reward": min(1.0, max(0.0, (1300.0 - cost) / 1200.0)),
        "above_reference": False,
        "counts": {
            "gridlocks": 1,
            "crew_timeouts": 0,
            "refusals": 3,
            "invalid_calls": 4,
            "decision_points": 40,
        },
        "tool_calls": 150,
        "final_tick": 60,
    }


def test_T08_20_harbor_harness_parsing(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path / "ds", 3)
    data = {k: v for k, v in BASE.items() if k != "scenarios"} | {
        "name": "hb",
        "harness": "harbor",
        "dataset": str(dataset),
        "reps": 1,
        "model": {"id": "big", "name": "Big", "harbor_model": "openrouter/big"},
    }
    manifest = load_manifest(_write(tmp_path, data))
    seen = {}

    def fake_harbor(cmd, env):
        seen["cmd"] = cmd
        jobs = Path(cmd[cmd.index("-o") + 1])
        job = jobs / "2026-11-14__job"
        job.mkdir(parents=True)
        (job / "result.json").write_text("{}")
        for i, sid in enumerate(sorted(p.name for p in dataset.iterdir())):
            trial = job / f"{sid[:20]}__t{i}"
            (trial / "verifier").mkdir(parents=True)
            (trial / "config.json").write_text(json.dumps({"task": {"path": str(dataset / sid)}}))
            # 0: clean; 1: the agent timed out but was graded (scored, as Harbor does); 2: infra error after grading
            exc = [None, {"exception_type": "AgentTimeoutError"}, {"exception_type": "Boom"}][i]
            (trial / "result.json").write_text(json.dumps({"exception_info": exc}))
            (trial / "verifier" / "details.json").write_text(json.dumps(_details(sid, 400.0 + 100 * i)))
            (trial / "verifier" / "reward.txt").write_text("0.750000\n")
        return 0

    monkeypatch.setattr(run_mod, "harbor_subprocess", fake_harbor)
    run_manifest(manifest, results_dir=tmp_path / "results")
    cmd = seen["cmd"]
    assert cmd[:2] == ["harbor", "run"]
    assert cmd[cmd.index("-p") + 1] == str(dataset) and cmd[cmd.index("-a") + 1] == "opencode"
    assert cmd[cmd.index("-m") + 1] == "openrouter/big" and cmd[cmd.index("-k") + 1] == "1"
    rows = {r["scenario_id"]: r for r in _rows(tmp_path / "results" / "hb.jsonl")}
    assert len(rows) == 3
    ok = [rows["FIX-mini3-0"], rows["FIX-mini3-1"]]
    for i, r in enumerate(ok):
        assert r["status"] == "ok" and r["harness"] == "harbor" and r["mode"] is None and r["model"] == "big"
        assert r["cost_total"] == 400.0 + 100 * i and r["c_greedy"] == 1300.0 and r["c_ref"] == 100.0
        assert r["score_unclipped"] == pytest.approx((900.0 - 100 * i) / 1200.0)
        assert r["score_clipped"] == pytest.approx((900.0 - 100 * i) / 1200.0)
        assert r["cost_ratio"] == pytest.approx((400.0 + 100 * i) / 100.0)
        assert (r["gridlocks"], r["refusals"], r["invalid_calls"]) == (1, 3, 4)
        assert r["decision_points"] == 40 and r["tool_calls"] == 150
        assert r["tokens_in"] is None and r["tokens_out"] is None
        assert r["seed"] == i and r["preset"] == "fixture" and r["scenario_hash"]
        assert r["rep"] == 0
    err = rows["FIX-mini3-2"]
    assert err["status"] == "error" and err["error"] and err["cost_total"] is None
