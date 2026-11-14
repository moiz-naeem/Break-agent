"""T07 §5.2: the verifier (`railway/harbor/verify.py`, run as the task's vendored `tests/verify.py`).

Two flavours:
- `*_fake_sim` / unit tests run today: a fixture task (mini3) exported by `export.py`, a stdlib fake of the
  sim's `/episode_log` route, logs made by `tests/t07/fake_replay.play` on the real Simulator, and the fake
  replay vendored into the task copy if T05's `railway.tools.replay` is missing.
- the ticket's T07-11..14 end-to-end tests run `railway-sim` (T05) with real policies (T05/T06); they skip
  until those modules exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from railway.sim.constants import RULES_VERSION
from railway.sim.cost import COMPONENTS
from tests.helpers import free_port, load_fixture, start_server
from tests.t07.conftest import REAL_EXPORT_MODULES, fixture_anchors, require_modules, run_verify, task_dirs
from tests.t07.fake_replay import play
from tests.t07.fakes import fake_sim, scripted_plan, use_fake_replay

DETAIL_KEYS = {
    "scenario_id",
    "rules_version",
    "cost_total",
    "cost_by_component",
    "c_greedy",
    "c_best",
    "best_source",
    "best_proven",
    "norm_version",
    "score_unclipped",
    "reward",
    "above_best",
    "counts",
    "kpis",
    "tool_calls",
    "final_tick",
}
COUNT_KEYS = {
    "gridlocks", "crew_timeouts", "refusals", "invalid_calls", "decision_points",
    "misroutes", "wrong_terminals", "missed_calls", "missed_connections", "unfinished_trains",
}  # fmt: skip
OPENCODE_LOG = Path("opencode/xdg-data/opencode/log/2026-11-14T120000.log")


class Task:
    def __init__(self, root: Path, logs: Path):
        self.root = root
        self.logs = logs
        self.scenario = json.loads((root / "tests" / "scenario.json").read_text(encoding="utf-8"))
        self.norm = json.loads((root / "tests" / "norm.json").read_text(encoding="utf-8"))
        self.token = (root / "tests" / "token").read_text(encoding="utf-8").strip()

    @property
    def reward_file(self) -> Path:
        return self.logs / "reward.txt"

    def details(self) -> dict:
        return json.loads((self.logs / "details.json").read_text(encoding="utf-8"))

    def verify(self, log: dict | bytes, *extra: str, agent_logs: Path | None = None):
        with fake_sim(self.token, log) as server:
            return run_verify(self.root, self.logs, server.url, *extra, agent_logs=agent_logs)

    def expect_no_reward(self, res, message: str) -> None:
        assert res.returncode != 0, res.stdout
        assert message in res.stdout, res.stdout + res.stderr
        assert not self.reward_file.exists()
        assert not (self.logs / "reward.json").exists()
        assert not (self.logs / "details.json").exists()

    def expect_reward(self, res, text: str) -> dict:
        assert res.returncode == 0, res.stdout + res.stderr
        assert self.reward_file.read_text(encoding="utf-8") == text
        assert not (self.logs / "reward.json").exists()
        return self.details()


@pytest.fixture(scope="session")
def mini3_export(tmp_path_factory) -> Path:
    from railway.harbor.export import ExportOptions, export_scenarios

    scenario = load_fixture("mini3")
    out = tmp_path_factory.mktemp("verify") / "ds"
    export_scenarios([(scenario, fixture_anchors(scenario))], out, ExportOptions(), meta={"preset": "fixture"})
    (task,) = task_dirs(out)
    use_fake_replay(task)
    return task


@pytest.fixture
def task(mini3_export, tmp_path) -> Task:
    logs = tmp_path / "verifier"
    logs.mkdir()
    return Task(mini3_export, logs)


@pytest.fixture(scope="session")
def plan_log() -> dict:
    scenario = load_fixture("mini3")
    return play(scenario, scripted_plan(scenario))


# ---------------------------------------------------------------- scoring through the fake sim


def test_T07_11f_anchor_plan_scores_one_fake_sim(task, plan_log):
    d = task.expect_reward(task.verify(plan_log), "1.000000\n")
    assert d["score_unclipped"] == 1.0 and d["reward"] == 1.0
    assert d["above_best"] is False


def test_T07_12f_partial_plan_scores_between_fake_sim(task, plan_log):
    calls = [(e["tool"], e["args"]) for e in plan_log["entries"]]
    log = play(task.scenario, [*calls[:2], ("end_shift", {})])
    c_greedy, c_best, c_agent = task.norm["c_greedy"], task.norm["c_best"], log["final_cost"]
    u = (c_greedy - c_agent) / (c_greedy - c_best)
    assert 0.0 < u < 1.0, "fixture plan prefix should land strictly between the anchors"
    d = task.expect_reward(task.verify(log), f"{u:.6f}\n")
    assert d["score_unclipped"] == pytest.approx(u, abs=1e-12)
    assert d["cost_total"] == c_agent


@pytest.mark.parametrize("calls", [[], [("end_shift", {})]], ids=["no-calls", "end-shift-at-once"])
def test_T07_13f_do_nothing_and_stop_early_zero_fake_sim(task, calls):
    d = task.expect_reward(task.verify(play(task.scenario, calls)), "0.000000\n")
    assert d["score_unclipped"] == 0.0
    assert d["tool_calls"] == len(calls)


def test_T07_14f_details_shape_fake_sim(task, plan_log):
    d = task.expect_reward(task.verify(plan_log), "1.000000\n")
    assert set(d) == DETAIL_KEYS
    assert set(d["counts"]) == COUNT_KEYS
    assert set(d["cost_by_component"]) == set(COMPONENTS)
    assert set(d["kpis"]) == {"on_time_pct", "passenger_delay_minutes", "freight_delivered_pct", "red_flags", "clean_shift"}
    assert d["scenario_id"] == task.scenario["scenario_id"] and d["rules_version"] == RULES_VERSION
    assert (d["c_greedy"], d["c_best"]) == (task.norm["c_greedy"], task.norm["c_best"])
    assert d["cost_total"] == plan_log["final_cost"] and d["final_tick"] == plan_log["final_tick"]
    assert d["tool_calls"] == len(plan_log["entries"])
    assert json.loads((task.logs / "action_log.json").read_text(encoding="utf-8")) == plan_log
    assert sorted(p.name for p in task.logs.iterdir()) == ["action_log.json", "details.json", "reward.txt"]


def test_T07_14g_calls_after_shift_end_do_not_count(task, plan_log):
    """Calls refused with R-SHIFT-ENDED are logged but use no budget, so they are not tool calls."""
    log = play(task.scenario, [("end_shift", {}), ("status", {}), ("kpis", {})])
    assert [e["rule"] for e in log["entries"]] == [None, "R-SHIFT-ENDED", "R-SHIFT-ENDED"]
    d = task.expect_reward(task.verify(log), "0.000000\n")
    assert d["tool_calls"] == 1


def test_T07_15_reward_math():
    from railway.harbor.verify import reward_from

    s = reward_from(1000, 500, 400)  # v0.3: no clip at the top; beating the best known plan is flagged
    assert s.reward == pytest.approx(1.2) and s.above_best is True and s.score_unclipped == pytest.approx(1.2)
    s = reward_from(1000, 500, 1200)
    assert s.reward == 0.0 and s.score_unclipped == pytest.approx(-0.4) and s.above_best is False
    s = reward_from(1000, 500, 750)
    assert s.reward == 0.5 and s.score_unclipped == 0.5
    assert reward_from(1000, 500, 500).reward == 1.0 and reward_from(1000, 500, 500).above_best is False
    assert reward_from(1000, 500, 1000).reward == 0.0
    for bad in ((500, 500, 1), (400, 500, 1)):
        with pytest.raises(ValueError):
            reward_from(*bad)


# ---------------------------------------------------------------- infrastructure failures: no reward


def test_T07_16_sim_unreachable(task):
    res = run_verify(task.root, task.logs, f"http://127.0.0.1:{free_port()}")
    task.expect_no_reward(res, "INFRA: sim unreachable")


def _corrupt_cases(log: dict) -> dict[str, bytes]:
    def dump(**patch) -> bytes:
        return json.dumps({**log, **patch}).encode()

    without_entries = {k: v for k, v in log.items() if k != "entries"}
    return {
        "non-json": b"<html>502 Bad Gateway</html>",
        "wrong-hash": dump(scenario_hash="0" * 64),
        "wrong-rules-version": dump(rules_version="9.9"),
        "wrong-scenario-id": dump(scenario_id="T1-000000000000"),
        "missing-entries": json.dumps(without_entries).encode(),
        "not-closed": dump(closed=False),
        "bad-entry": dump(entries=[{"seq": 1, "tool": "status"}]),
        "bad-status": dump(entries=[{**log["entries"][0], "status": "MAYBE"}]),
        "json-list": b"[]",
    }


@pytest.mark.parametrize("case", list(_corrupt_cases({"entries": [{}]})))
def test_T07_17_corrupt_log(task, plan_log, case):
    task.expect_no_reward(task.verify(_corrupt_cases(plan_log)[case]), "INFRA: corrupt or foreign episode log")


def test_T07_18_replay_mismatch_cost(task, plan_log):
    task.expect_no_reward(task.verify({**plan_log, "final_cost": plan_log["final_cost"] + 10}), "REPLAY MISMATCH: cost")


def test_T07_18b_replay_mismatch_component(task, plan_log):
    """Same total, components shifted: still a cost mismatch."""
    comp = dict(plan_log["final_cost_by_component"])
    comp["pax_lateness"] += 5
    comp["unfinished"] -= 5
    task.expect_no_reward(task.verify({**plan_log, "final_cost_by_component": comp}), "REPLAY MISMATCH: cost")


def test_T07_18c_replay_mismatch_status(task, plan_log):
    entries = [dict(e) for e in plan_log["entries"]]
    first = next(i for i, e in enumerate(entries) if e["status"] == "GRANTED")
    entries[first].update(status="REFUSED", rule="R-SECTION-RESERVED")
    task.expect_no_reward(task.verify({**plan_log, "entries": entries}), "REPLAY MISMATCH")


def test_T07_18d_replay_mismatch_final_tick(task, plan_log):
    task.expect_no_reward(task.verify({**plan_log, "final_tick": plan_log["final_tick"] + 1}), "REPLAY MISMATCH")


def test_T07_19_wrong_token_is_infra(task, plan_log):
    with fake_sim("f" * 64, plan_log) as server:  # the server holds a different token → 403
        res = run_verify(task.root, task.logs, server.url)
        assert server.log_requests >= 1
    task.expect_no_reward(res, "INFRA")


def test_T07_21_stale_reward_files_removed(task, plan_log):
    """An agent that pre-writes reward.txt / reward.json must gain nothing: both are removed first."""
    for name in ("reward.txt", "reward.json"):
        (task.logs / name).write_text("1.0\n", encoding="utf-8")
    task.expect_no_reward(run_verify(task.root, task.logs, f"http://127.0.0.1:{free_port()}"), "INFRA")
    (task.logs / "reward.json").write_text('{"reward": 1.0}', encoding="utf-8")
    task.expect_reward(task.verify(play(task.scenario, [])), "0.000000\n")


@pytest.mark.parametrize("patch", [{"scenario_hash": "0" * 64}, {"rules_version": "9.9"}, {"c_best": 1e9}, {"best_source": "x"}])
def test_T07_22_bad_norm_is_infra(tmp_path, mini3_export, plan_log, patch):
    import shutil

    root = tmp_path / "task"
    shutil.copytree(mini3_export, root)
    norm_path = root / "tests" / "norm.json"
    norm_path.write_text(json.dumps({**json.loads(norm_path.read_text(encoding="utf-8")), **patch}), encoding="utf-8")
    logs = tmp_path / "verifier"
    logs.mkdir()
    t = Task(root, logs)
    t.expect_no_reward(t.verify(plan_log), "INFRA: bad norm.json")


# ---------------------------------------------------------------- MCP startup failure vs a genuine do-nothing


def _agent_logs(tmp_path: Path, text: str | bytes | None) -> Path:
    root = tmp_path / "agent"
    root.mkdir()
    if text is not None:
        path = root / OPENCODE_LOG
        path.parent.mkdir(parents=True)
        path.write_bytes(text if isinstance(text, bytes) else text.encode())
    return root


SPIKE_FAILURE = (
    "INFO  2026-11-14T12:00:01 +2ms service=mcp key=railway type=remote server unavailable\n"
    "INFO  2026-11-14T12:00:01 +0ms service=session tools=bash,read,write,edit\n"
)


@pytest.mark.parametrize(
    "agent_log",
    [
        SPIKE_FAILURE,
        "ERROR 2026-11-14T12:00:01 service=mcp key=railway error=ECONNREFUSED failed to connect\n",
        "WARN service=mcp server=railway connection timed out\n",
    ],
    ids=["spike-line", "connect-failed", "timed-out"],
)
def test_T07_20_mcp_startup_failure_is_trial_error(task, tmp_path, agent_log):
    res = task.verify(play(task.scenario, []), agent_logs=_agent_logs(tmp_path, agent_log))
    task.expect_no_reward(res, "INFRA: agent started without MCP tools")


@pytest.mark.parametrize(
    "agent_log",
    [
        None,  # no agent logs at all (e.g. the built-in `nop` agent)
        "",
        "INFO service=mcp key=railway type=remote connected tools=15\n",
        "ModuleNotFoundError: No module named 'railway'\n",  # a probe line, not an MCP failure
        "INFO service=mcp key=toy server unavailable\n",  # another server's failure
        b"\xff\xfe\x00binary junk\x00 key=railway\n",
        # The task prompt logged on one line names the `railway` MCP server; "crew timeouts" is not "timeout".
        'INFO service=session prompt="...tools of the `railway` MCP server; ... crew timeouts, gridlock"\n',
    ],
    ids=["no-logs", "empty", "mcp-healthy", "import-error", "other-server", "binary", "prompt-line"],
)
def test_T07_20b_genuine_do_nothing_scores_zero(task, tmp_path, agent_log):
    res = task.verify(play(task.scenario, []), agent_logs=_agent_logs(tmp_path, agent_log))
    task.expect_reward(res, "0.000000\n")


def test_T07_20c_failure_line_ignored_when_tools_were_used(task, tmp_path, plan_log):
    """A logged MCP hiccup doesn't void an episode that did reach the server: it is scored normally."""
    res = task.verify(plan_log, agent_logs=_agent_logs(tmp_path, SPIKE_FAILURE))
    task.expect_reward(res, "1.000000\n")


def test_T07_20d_rule_unit(tmp_path):
    from railway.harbor.verify import mcp_startup_failure

    assert mcp_startup_failure(tmp_path / "missing") is None
    root = _agent_logs(tmp_path, SPIKE_FAILURE)
    hit = mcp_startup_failure(root)
    assert hit is not None and "server unavailable" in hit


# ---------------------------------------------------------------- probe


def test_T07_23_probe_copied_into_details(task, tmp_path, plan_log):
    agent = _agent_logs(tmp_path, None)
    probe = {"uid": "1000", "github": "unreachable", "sim_health": "ok"}
    (agent / "probe.json").write_text(json.dumps(probe), encoding="utf-8")
    d = task.expect_reward(task.verify(plan_log, "--probe", agent_logs=agent), "1.000000\n")
    assert d["probe"] == probe
    assert set(d) == DETAIL_KEYS | {"probe"}


def test_T07_23b_probe_ignored_without_flag(task, tmp_path, plan_log):
    agent = _agent_logs(tmp_path, None)
    (agent / "probe.json").write_text('{"uid": "0"}', encoding="utf-8")
    d = task.expect_reward(task.verify(plan_log, agent_logs=agent), "1.000000\n")
    assert set(d) == DETAIL_KEYS


# ---------------------------------------------------------------- end-to-end with the real sim (T05/T06)


@pytest.fixture(scope="module")
def real_task(tmp_path_factory) -> Path:
    require_modules(*REAL_EXPORT_MODULES, "railway.server", "railway.tools.client", "railway.tools.replay")
    from railway.harbor.export import ExportOptions, export_dataset

    out = tmp_path_factory.mktemp("real") / "ds"
    export_dataset("T1", "dev", 1, out, ExportOptions())
    return task_dirs(out)[0]


def _run_real(real_task: Path, tmp_path: Path, policy) -> Task:
    from railway.tools.client import McpClient

    logs = tmp_path / "verifier"
    logs.mkdir(parents=True)
    sim = real_task / "environment" / "sim"
    args = ["-m", "railway.server", "--scenario", str(sim / "scenario.json"), "--token-file", str(sim / "token")]
    with start_server(args) as base:
        if policy is not None:
            client = McpClient(f"{base}/mcp")
            try:
                policy.run(client)
            finally:
                getattr(client, "close", lambda: None)()
        res = run_verify(real_task, logs, base)
    t = Task(real_task, logs)
    t.last = res
    return t


def test_T07_11_oracle_scores_one(real_task, tmp_path):
    require_modules("railway.ref.reference")
    from railway.ref.reference import Reference

    t = _run_real(real_task, tmp_path, Reference())
    d = t.expect_reward(t.last, "1.000000\n")
    assert d["score_unclipped"] == 1.0


def test_T07_12_greedy_scores_zero(real_task, tmp_path):
    require_modules("railway.policies.baselines")
    from railway.policies.baselines import Greedy

    t = _run_real(real_task, tmp_path, Greedy())
    d = t.expect_reward(t.last, "0.000000\n")
    assert d["score_unclipped"] == 0.0


def test_T07_13_do_nothing_and_stop_early_zero(real_task, tmp_path):
    require_modules("railway.policies.baselines")
    from railway.policies.baselines import StopEarly

    t = _run_real(real_task, tmp_path / "nothing", None)
    t.expect_reward(t.last, "0.000000\n")
    t = _run_real(real_task, tmp_path / "stop", StopEarly())
    t.expect_reward(t.last, "0.000000\n")


def test_T07_14_details_shape(real_task, tmp_path):
    require_modules("railway.policies.baselines")
    from railway.policies.baselines import Greedy

    t = _run_real(real_task, tmp_path, Greedy())
    d = t.expect_reward(t.last, "0.000000\n")
    assert set(d) == DETAIL_KEYS and set(d["counts"]) == COUNT_KEYS
    log = json.loads((t.logs / "action_log.json").read_text(encoding="utf-8"))
    assert log["closed"] is True and log["final_cost"] == d["cost_total"]
    assert not (t.logs / "reward.json").exists()
