"""Offline unit and local-MCP integration tests for `railway/harbor/net_export.py`, `net_verify.py`,
and `infra/modal/harbor_runner.py` configuration (Lane H, decisions.md #2, #4, #8, #10).
"""

from __future__ import annotations

import contextlib
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import tomllib
from pathlib import Path

import pytest
import yaml

from infra.modal.harbor_runner import (
    CUSTOM_PROVIDERS,
    MODEL_PRESETS,
    OPENCODE_VERSION,
    VOLUME_NAME,
    build_agent_config,
    build_opencode_config,
)
from railway.harbor.net_export import (
    DEFAULT_PROVIDER_HOSTS,
    ExportOptions,
    export_task,
    export_timetables,
    scenario_id_for,
)
from railway.harbor.net_verify import agent_cli_failure, mcp_startup_failure
from railway.net.cost import COMPONENTS, cost
from railway.net.events import Event, EventLog
from railway.net.formats import NET_RULES_VERSION
from railway.net.manual import TOOL_BUDGET
from railway.net.server import spawn
from railway.net.session import COUNT_KEYS, Session
from railway.util import canonical_json, content_hash

REPO = Path(__file__).resolve().parents[2]

EXPECTED_TASK_PATHS = (
    "instruction.md",
    "task.toml",
    "environment/Dockerfile",
    "environment/docker-compose.yaml",
    "environment/sim/Dockerfile",
    "environment/sim/requirements.txt",
    "environment/sim/timetable.json",
    "environment/sim/token",
    "environment/sim/railway/__init__.py",
    "environment/sim/railway/util.py",
    "environment/sim/railway/net/__init__.py",
    "tests/test.sh",
    "tests/verify.py",
    "tests/timetable.json",
    "tests/token",
    "tests/railway/__init__.py",
    "tests/railway/util.py",
    "tests/railway/net/__init__.py",
    "solution/solve.sh",
    "solution/solve.py",
)

EXPECTED_DETAIL_KEYS = {
    "scenario_id",
    "scenario_hash",
    "tier",
    "seed",
    "rules_version",
    "reward",
    "cost_total",
    "cost_by_component",
    "kpis",
    "counts",
    "tool_calls",
    "final_tick",
}

FORBIDDEN_LEGACY_KEYS = {
    "c_greedy",
    "c_ref",
    "c_best",
    "best_source",
    "best_proven",
    "norm_version",
    "score",
    "score_unclipped",
    "score_normalized",
    "above_reference",
    "above_best",
}

FORBIDDEN_DOC_WORDS = ("single-track", "reversal", "greedy", "reference", "c_greedy", "c_ref", "c_best", "norm.json")
OPENCODE_LOG_REL = Path("opencode/xdg-data/opencode/log/2026-11-14T120000.log")


# ---------------------------------------------------------------- helpers


@contextlib.contextmanager
def _fake_sim(expected_token: str, payload: dict | bytes, status_code: int = 200):
    raw = payload if isinstance(payload, bytes) else canonical_json(payload).encode("utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
                return
            if self.path == "/episode_log":
                if self.headers.get("Authorization") != f"Bearer {expected_token}":
                    self.send_response(403)
                    self.end_headers()
                    self.wfile.write(b"forbidden")
                    return
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(raw)
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, *_args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _run_verify_cli(
    task_dir: Path,
    logs_dir: Path,
    sim_url: str,
    *extra: str,
    agent_logs_dir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    verify_py = task_dir / "tests" / "verify.py"
    cmd = [
        sys.executable,
        str(verify_py),
        "--logs-dir",
        str(logs_dir),
        "--sim-url",
        sim_url,
        "--retries",
        "2",
        "--retry-wait",
        "0.05",
        *extra,
    ]
    if agent_logs_dir is not None:
        cmd += ["--agent-logs-dir", str(agent_logs_dir)]
    # Isolate sys.path to `<task>/tests` via PYTHONSAFEPATH=1 to prove the vendored subset is self-sufficient.
    env = {
        **{k: v for k, v in os.environ.items() if k in ("SYSTEMROOT", "PATH", "HOME", "TMP", "TEMP")},
        "PYTHONSAFEPATH": "1",
        "PYTHONPATH": str(task_dir / "tests"),
        "PYTHONIOENCODING": "utf-8",
    }
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=30)


def _play_inproc(timetable, calls: list[tuple[str, dict]]) -> dict:
    s = Session(timetable)
    for tool_name, args in calls:
        s.call(tool_name, args)
    return json.loads(canonical_json(s.episode_log()))


def _agent_logs(tmp_path: Path, name: str, content: str | bytes | None) -> Path:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    if content is not None:
        p = root / OPENCODE_LOG_REL
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return root


@pytest.fixture(scope="module")
def exported_ds(tmp_path_factory, sample_timetable, stranded_timetable) -> Path:
    out = tmp_path_factory.mktemp("net_harbor") / "ds"
    tt_a = sample_timetable.model_copy(update={"seed": 101, "tier": "T1"})
    tt_b = stranded_timetable.model_copy(update={"seed": 202, "tier": "T1"})
    export_timetables(
        [(scenario_id_for(tt_a), tt_a), (scenario_id_for(tt_b), tt_b)],
        out,
        ExportOptions(),
        meta={"tier": "T1", "set": "unit"},
    )
    return out


# ---------------------------------------------------------------- 1. Export layout & isolation


def test_export_layout_manifest_and_zero_greedy_or_ref(exported_ds):
    manifest = json.loads((exported_ds / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rules_version"] == NET_RULES_VERSION
    assert manifest["count"] == 2
    assert len(manifest["tasks"]) == 2
    assert not (set(manifest) & FORBIDDEN_LEGACY_KEYS)

    readme = (exported_ds / "README.md").read_text(encoding="utf-8")
    for bad in FORBIDDEN_DOC_WORDS:
        assert bad not in readme.lower(), f"README.md mentions forbidden legacy term {bad!r}"

    tokens = set()
    for row in manifest["tasks"]:
        assert not (set(row) & FORBIDDEN_LEGACY_KEYS)
        assert {"task", "scenario_id", "scenario_hash", "tier", "seed", "shift_min"} <= set(row)
        task_dir = exported_ds / row["task"]
        for rel in EXPECTED_TASK_PATHS:
            assert (task_dir / rel).is_file(), f"{row['task']} missing {rel}"

        # No norm.json, no plan.json, no scenario.json anywhere.
        assert not (task_dir / "tests" / "norm.json").exists()
        assert not (task_dir / "tests" / "plan.json").exists()
        assert not list(task_dir.rglob("norm.json"))

        # Byte-identical timetable.json, token, verify.py, and vendored railway/net.
        sim_dir = task_dir / "environment" / "sim"
        tests_dir = task_dir / "tests"
        assert (sim_dir / "timetable.json").read_bytes() == (tests_dir / "timetable.json").read_bytes()
        assert (sim_dir / "token").read_bytes() == (tests_dir / "token").read_bytes()
        tok = (tests_dir / "token").read_text(encoding="utf-8").strip()
        assert re.fullmatch(r"[0-9a-f]{64}", tok)
        tokens.add(tok)
        assert (tests_dir / "verify.py").read_bytes() == (REPO / "railway" / "harbor" / "net_verify.py").read_bytes()

        tt_dict = json.loads((tests_dir / "timetable.json").read_text(encoding="utf-8"))
        assert content_hash(tt_dict) == row["scenario_hash"]

        for sub in (sim_dir / "railway", tests_dir / "railway"):
            pkgs = {p.name for p in sub.iterdir() if p.is_dir() and p.name != "__pycache__"}
            assert pkgs == {"net"}
            for f in sub.rglob("*"):
                if f.is_file():
                    assert "__pycache__" not in f.parts and f.suffix != ".pyc"
                    assert f.read_bytes() == (REPO / "railway" / f.relative_to(sub)).read_bytes()

    assert len(tokens) == 2


def test_instruction_task_toml_compose_and_strict_agent_isolation(exported_ds, sample_timetable, tmp_path):
    manifest = json.loads((exported_ds / "manifest.json").read_text(encoding="utf-8"))
    for row in manifest["tasks"]:
        task_dir = exported_ds / row["task"]
        token = (task_dir / "tests" / "token").read_text(encoding="utf-8").strip()

        # Instruction checks: ARS + -(total cost), zero legacy terms, zero scenario/seed leak.
        instr = (task_dir / "instruction.md").read_text(encoding="utf-8")
        assert f"{TOOL_BUDGET} calls" in instr
        assert "-(total cost)" in instr
        assert "advance_to_next_event()" in instr
        for bad in (*FORBIDDEN_DOC_WORDS, "authorize"):
            assert bad not in instr.lower(), f"instruction.md mentions {bad!r}"
        assert row["scenario_id"] not in instr and row["scenario_hash"] not in instr

        # task.toml checks (Harbor 0.26 schema_version = "1.4").
        cfg = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
        assert cfg["schema_version"] == "1.4"
        assert cfg["agent"]["user"] == "agent"
        assert cfg["agent"]["timeout_sec"] == 3600
        assert cfg["verifier"]["timeout_sec"] == 600
        env_cfg = cfg["environment"]
        assert env_cfg["network_mode"] == "allowlist"
        assert set(env_cfg["allowed_hosts"]) >= {"sim", *DEFAULT_PROVIDER_HOSTS}
        assert env_cfg["mcp_servers"] == [
            {"name": "railway", "transport": "streamable-http", "url": "http://sim:8000/mcp"}
        ]

        # Agent image isolation: zero COPY/ADD, zero timetable/token/railway outside environment/sim, zero in solution/.
        env_dir = task_dir / "environment"
        for p in env_dir.rglob("*"):
            if not p.is_file() or (env_dir / "sim") in p.parents:
                continue
            assert p.name not in ("timetable.json", "scenario.json", "token")
            txt = p.read_text(encoding="utf-8", errors="replace")
            for secret in (row["scenario_id"], row["scenario_hash"], token):
                assert secret not in txt
        dockerfile = (env_dir / "Dockerfile").read_text(encoding="utf-8")
        assert "opencode-ai@1.19.5" in dockerfile
        assert "useradd -m -u 1000 agent" in dockerfile
        for line in dockerfile.splitlines():
            assert not re.match(r"\s*(COPY|ADD)\b", line, re.I), f"unexpected COPY/ADD in agent Dockerfile: {line}"

        sol_files = {
            p.relative_to(task_dir / "solution").as_posix() for p in (task_dir / "solution").rglob("*") if p.is_file()
        }
        assert sol_files == {"solve.sh", "solve.py"}

        # Compose & sim Dockerfile checks.
        compose = yaml.safe_load((env_dir / "docker-compose.yaml").read_text(encoding="utf-8"))
        assert compose["services"]["sim"]["networks"] == ["default"]
        assert compose["services"]["main"]["depends_on"]["sim"]["condition"] == "service_healthy"
        sim_df = (env_dir / "sim" / "Dockerfile").read_text(encoding="utf-8")
        assert "railway.net.server" in sim_df and "/srv/timetable.json" in sim_df and "--token-file" in sim_df
        for sh_path in (task_dir / "tests" / "test.sh", task_dir / "solution" / "solve.sh"):
            assert b"\r" not in sh_path.read_bytes()

    # Probe export option check.
    probe_out = tmp_path / "probe_ds"
    tt = sample_timetable.model_copy(update={"seed": 7, "tier": "T1"})
    m_probe = export_timetables([tt], probe_out, ExportOptions(probe=True, default_hosts=False, agent_user="root"))
    assert len(m_probe["tasks"]) == 2
    probe_task = probe_out / "T1-7-probe"
    assert "--probe" in (probe_task / "tests" / "test.sh").read_text(encoding="utf-8")
    assert "/logs/agent/probe.json" in (probe_task / "solution" / "solve.sh").read_text(encoding="utf-8")
    root_cfg = tomllib.loads((probe_out / "T1-7" / "task.toml").read_text(encoding="utf-8"))
    assert "user" not in root_cfg["agent"]
    assert root_cfg["environment"]["allowed_hosts"] == ["sim"]


# ---------------------------------------------------------------- 2. Verifier happy paths + live MCP oracle


def test_oracle_solve_py_and_verifier_end_to_end_over_live_mcp(exported_ds, t1_timetable, tmp_path):
    # Test on exported tasks (`T1-101` conflicts + `T1-202` UNROUTABLE re-platforming) AND real 180-min T1.
    t1_dir = tmp_path / "T1-1"
    export_task("T1-1", t1_timetable, t1_dir, ExportOptions())

    for idx, src_task_dir in enumerate([exported_ds / "T1-101", exported_ds / "T1-202", t1_dir]):
        scratch = tmp_path / f"run_{idx}"
        task_dir = scratch / "task"
        shutil.copytree(src_task_dir, task_dir)
        sim_dir = task_dir / "environment" / "sim"
        tt_obj_json = (sim_dir / "timetable.json").read_text(encoding="utf-8")
        from railway.net.model import Timetable

        tt_obj = Timetable.model_validate_json(tt_obj_json)
        with spawn(tt_obj) as srv:
            (task_dir / "tests" / "token").write_text(srv.token, encoding="utf-8")
            (sim_dir / "token").write_text(srv.token, encoding="utf-8")

            solve_res = subprocess.run(
                [sys.executable, str(task_dir / "solution" / "solve.py"), "--url", srv.url],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert solve_res.returncode == 0, solve_res.stdout + solve_res.stderr

            logs_dir = scratch / "verifier"
            vres = _run_verify_cli(task_dir, logs_dir, srv.base)
            assert vres.returncode == 0, vres.stdout + vres.stderr

        assert sorted(p.name for p in logs_dir.iterdir()) == [
            "action_log.json",
            "details.json",
            "events.jsonl",
            "reward.txt",
        ]
        details = json.loads((logs_dir / "details.json").read_text(encoding="utf-8"))
        assert set(details) == EXPECTED_DETAIL_KEYS
        assert not (set(details) & FORBIDDEN_LEGACY_KEYS)
        assert set(details["cost_by_component"]) == set(COMPONENTS)
        assert set(details["counts"]) == set(COUNT_KEYS)
        assert details["rules_version"] == NET_RULES_VERSION
        assert details["reward"] == round(-details["cost_total"], 6)
        assert (logs_dir / "reward.txt").read_text(encoding="utf-8") == f"{details['reward']}\n"
        assert float((logs_dir / "reward.txt").read_text(encoding="utf-8")) == -details["cost_total"]
        assert details["counts"]["invalid_calls"] == 0
        assert details["counts"]["refusals"] == 0
        assert details["counts"]["unfinished"] == 0
        assert (logs_dir / "events.jsonl").read_text(encoding="utf-8").strip(), "replayed EventLog is empty"


def test_real_t1_do_nothing_is_penalised_and_oracle_beats_it(t1_timetable, tmp_path):
    """With the real engine EventLog, doing nothing must cost > 0 and the booked-order oracle must do better."""
    task_dir = tmp_path / "T1-1"
    export_task("T1-1", t1_timetable, task_dir, ExportOptions())
    token = (task_dir / "tests" / "token").read_text(encoding="utf-8").strip()

    logs_dir = tmp_path / "v_nop"
    with _fake_sim(token, _play_inproc(t1_timetable, [])) as sim_url:
        res = _run_verify_cli(task_dir, logs_dir, sim_url)
    assert res.returncode == 0, res.stdout + res.stderr
    nop = json.loads((logs_dir / "details.json").read_text(encoding="utf-8"))
    assert nop["cost_total"] > 0 and nop["reward"] < 0
    assert nop["counts"]["unfinished"] > 0
    assert float((logs_dir / "reward.txt").read_text(encoding="utf-8")) == nop["reward"]
    events = [json.loads(line) for line in (logs_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events and any(e["kind"] == "SHIFT_END" for e in events)

    sim_dir = task_dir / "environment" / "sim"
    with spawn(t1_timetable) as srv:
        (task_dir / "tests" / "token").write_text(srv.token, encoding="utf-8")
        (sim_dir / "token").write_text(srv.token, encoding="utf-8")
        solve_res = subprocess.run(
            [sys.executable, str(task_dir / "solution" / "solve.py"), "--url", srv.url],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert solve_res.returncode == 0, solve_res.stdout + solve_res.stderr
        oracle_logs = tmp_path / "v_oracle"
        vres = _run_verify_cli(task_dir, oracle_logs, srv.base)
        assert vres.returncode == 0, vres.stdout + vres.stderr
    oracle = json.loads((oracle_logs / "details.json").read_text(encoding="utf-8"))
    assert oracle["reward"] > nop["reward"]


def test_verifier_nonzero_cost_and_events_jsonl_with_populated_event_log(sample_timetable, tmp_path):
    """Verify `reward == -cost_total < 0` and `events.jsonl` serialization when `Engine.log` emits Events."""
    tt = sample_timetable.model_copy(update={"seed": 303, "tier": "T1"})
    task_dir = tmp_path / "T1-303"
    export_task("T1-303", tt, task_dir, ExportOptions())
    token = (task_dir / "tests" / "token").read_text(encoding="utf-8").strip()

    # Patch the vendored `tests/railway/net/session.py` inside this isolated temp task so `Session.close()`
    # attaches a deterministic EventLog with real late-arrival, signal-stop, and unfinished charges.
    hook = (
        "\n_ORIG_FINISH = Session._finish_shift\n"
        "def _instrumented_finish(self):\n"
        "    _ORIG_FINISH(self)\n"
        "    ev = EventLog()\n"
        "    ev.emit(35, 'ARRIVE', train='IC1', element='A1', node='A', call=0, booked=10)\n"
        "    ev.emit(40, 'SIGNAL_STOP', train='IC1', element='A-JA.u', detail={'restart_min': 1})\n"
        "    ev.emit(self.engine.tick, 'SHIFT_END')\n"
        "    ev.emit(self.engine.tick, 'UNFINISHED', train='L4', detail={'freight': False})\n"
        "    self.engine.log = ev\n"
        "Session._finish_shift = _instrumented_finish\n"
    )
    vendored_session_py = task_dir / "tests" / "railway" / "net" / "session.py"
    vendored_session_py.write_text(vendored_session_py.read_text(encoding="utf-8") + hook, encoding="utf-8")

    # Build the matching live episode log using the exact same EventLog.
    live = Session(tt)
    live.call("notes", {"text": "checking events"})
    live.call("end_shift", {})
    # Recompute live log with those events attached.
    live2 = Session(tt)
    live2.call("notes", {"text": "checking events"})
    live2._finish_shift()
    ev_log = EventLog()
    ev_log.emit(35, "ARRIVE", train="IC1", element="A1", node="A", call=0, booked=10)
    ev_log.emit(40, "SIGNAL_STOP", train="IC1", element="A-JA.u", detail={"restart_min": 1})
    ev_log.emit(live2.engine.tick, "SHIFT_END")
    ev_log.emit(live2.engine.tick, "UNFINISHED", train="L4", detail={"freight": False})
    live2.engine.log = ev_log  # type: ignore[attr-defined]
    ep_log = live2.episode_log()

    expected_breakdown = cost(tt, ev_log)
    assert expected_breakdown.total > 0
    expected_reward = round(-expected_breakdown.total, 6)
    assert expected_reward < 0

    logs_dir = tmp_path / "verifier_logs"
    with _fake_sim(token, ep_log) as sim_url:
        res = _run_verify_cli(task_dir, logs_dir, sim_url)
    assert res.returncode == 0, res.stdout + res.stderr

    reward_txt = (logs_dir / "reward.txt").read_text(encoding="utf-8")
    assert reward_txt == f"{expected_reward}\n"
    assert float(reward_txt) == pytest.approx(-expected_breakdown.total)

    details = json.loads((logs_dir / "details.json").read_text(encoding="utf-8"))
    assert details["cost_total"] == pytest.approx(expected_breakdown.total)
    assert details["reward"] == pytest.approx(expected_reward)
    assert details["tool_calls"] == 1

    event_lines = (logs_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(event_lines) == 4
    parsed_events = [Event.model_validate_json(line) for line in event_lines]
    assert [e.kind for e in parsed_events] == ["ARRIVE", "SIGNAL_STOP", "SHIFT_END", "UNFINISHED"]


def test_post_shift_calls_excluded_stale_rewards_cleared_and_probe_flag(exported_ds, sample_timetable, tmp_path):
    task_dir = exported_ds / "T1-101"
    token = (task_dir / "tests" / "token").read_text(encoding="utf-8").strip()
    tt = sample_timetable.model_copy(update={"seed": 101, "tier": "T1"})
    ep_log = _play_inproc(tt, [("end_shift", {}), ("status", {}), ("kpis", {})])
    assert [e["rule"] for e in ep_log["entries"]] == [None, "R-SHIFT-ENDED", "R-SHIFT-ENDED"]

    logs_dir = tmp_path / "verifier"
    logs_dir.mkdir()
    (logs_dir / "reward.txt").write_text("999.0\n", encoding="utf-8")
    (logs_dir / "reward.json").write_text('{"reward": 999.0}', encoding="utf-8")

    # On failure, pre-existing reward files must be deleted and stay absent.
    bad_res = _run_verify_cli(task_dir, logs_dir, "http://127.0.0.1:1")
    assert bad_res.returncode == 1 and "INFRA: sim unreachable" in bad_res.stdout
    assert not (logs_dir / "reward.txt").exists() and not (logs_dir / "reward.json").exists()

    # On clean run with --probe, probe.json is copied into details.json and tool_calls == 1.
    agent_dir = _agent_logs(tmp_path, "agent_probe", None)
    probe_payload = {"uid": "1000", "github": "unreachable", "sim_health": "ok"}
    (agent_dir / "probe.json").write_text(json.dumps(probe_payload), encoding="utf-8")
    (logs_dir / "reward.json").write_text('{"reward": 999.0}', encoding="utf-8")

    with _fake_sim(token, ep_log) as sim_url:
        ok_res = _run_verify_cli(task_dir, logs_dir, sim_url, "--probe", agent_logs_dir=agent_dir)
    assert ok_res.returncode == 0, ok_res.stdout + ok_res.stderr
    assert not (logs_dir / "reward.json").exists()
    details = json.loads((logs_dir / "details.json").read_text(encoding="utf-8"))
    assert details["tool_calls"] == 1
    assert details["probe"] == probe_payload


# ---------------------------------------------------------------- 3. Tamper rejection (REPLAY MISMATCH)


def test_verifier_rejects_tampered_logs_with_replay_mismatch(exported_ds, sample_timetable, tmp_path):
    task_dir = exported_ds / "T1-101"
    token = (task_dir / "tests" / "token").read_text(encoding="utf-8").strip()
    tt = sample_timetable.model_copy(update={"seed": 101, "tier": "T1"})
    base_log = _play_inproc(
        tt,
        [
            ("advance_to_next_event", {}),
            ("order", {"resource": "A-JA.u:0", "trains": ["IC1", "L4"]}),
            ("order", {"resource": "A.e1/A.e3", "trains": ["IC1", "L4"]}),
            ("end_shift", {}),
        ],
    )

    shifted_comp = dict(base_log["final_cost_by_component"])
    shifted_comp["pax_lateness"] += 10.0
    shifted_comp["unfinished"] -= 10.0

    bad_status_entries = [dict(e) for e in base_log["entries"]]
    bad_status_entries[1] = {**bad_status_entries[1], "status": "REFUSED", "rule": "R-UNKNOWN-TRAIN"}

    bad_args_entries = [dict(e) for e in base_log["entries"]]
    bad_args_entries[1] = {**bad_args_entries[1], "args": {"resource": "A-JA.u:0", "trains": ["GHOST"]}}

    bad_tick_entries = [dict(e) for e in base_log["entries"]]
    bad_tick_entries[1] = {**bad_tick_entries[1], "tick": bad_tick_entries[1]["tick"] + 3}

    tampered_cases = {
        "final_cost": {**base_log, "final_cost": base_log["final_cost"] - 25.0},
        "component_shift": {**base_log, "final_cost_by_component": shifted_comp},
        "entry_status": {**base_log, "entries": bad_status_entries},
        "entry_args": {**base_log, "entries": bad_args_entries},
        "entry_tick": {**base_log, "entries": bad_tick_entries},
        "final_tick": {**base_log, "final_tick": base_log["final_tick"] + 1},
        "final_state": {**base_log, "final_state": []},
        "counts": {**base_log, "counts": {**base_log["counts"], "gridlocks": 99}},
    }

    for name, bad_log in tampered_cases.items():
        logs_dir = tmp_path / f"tamper_{name}"
        with _fake_sim(token, bad_log) as sim_url:
            res = _run_verify_cli(task_dir, logs_dir, sim_url)
        assert res.returncode != 0, f"{name} unexpectedly succeeded"
        assert "REPLAY MISMATCH:" in res.stdout, f"{name}: {res.stdout}"
        assert not (logs_dir / "reward.txt").exists()
        assert not (logs_dir / "details.json").exists()


# ---------------------------------------------------------------- 4. Corrupt / foreign logs & auth (INFRA)


def test_verifier_rejects_corrupt_logs_and_bad_auth_with_infra(exported_ds, sample_timetable, tmp_path):
    task_dir = exported_ds / "T1-101"
    token = (task_dir / "tests" / "token").read_text(encoding="utf-8").strip()
    tt = sample_timetable.model_copy(update={"seed": 101, "tier": "T1"})
    valid_log = _play_inproc(tt, [("end_shift", {})])

    # Wrong bearer token -> HTTP 403 -> INFRA
    with _fake_sim("f" * 64, valid_log) as sim_url:
        res_403 = _run_verify_cli(task_dir, tmp_path / "auth_403", sim_url)
    assert res_403.returncode != 0 and "INFRA: episode_log refused (HTTP 403)" in res_403.stdout
    assert not (tmp_path / "auth_403" / "reward.txt").exists()

    corrupt_payloads: dict[str, dict | bytes] = {
        "html": b"<html>502 Bad Gateway</html>",
        "json_list": b"[]",
        "wrong_hash": {**valid_log, "scenario_hash": "0" * 64},
        "wrong_id": {**valid_log, "scenario_id": "T1-999999"},
        "wrong_rules": {**valid_log, "rules_version": "0.1-old"},
        "not_closed": {**valid_log, "closed": False},
        "no_entries": {k: v for k, v in valid_log.items() if k != "entries"},
        "bad_entry": {**valid_log, "entries": [{"seq": 1, "tool": "status"}]},
        "bad_status": {**valid_log, "entries": [{**valid_log["entries"][0], "status": "BOGUS"}]},
        "bad_seq": {**valid_log, "entries": [{**valid_log["entries"][0], "seq": 2}]},
    }
    for name, payload in corrupt_payloads.items():
        logs_dir = tmp_path / f"corrupt_{name}"
        with _fake_sim(token, payload) as sim_url:
            res = _run_verify_cli(task_dir, logs_dir, sim_url)
        assert res.returncode != 0, name
        assert "INFRA: corrupt or foreign episode log" in res.stdout, f"{name}: {res.stdout}"
        assert not (logs_dir / "reward.txt").exists()


# ---------------------------------------------------------------- 5. MCP startup-race guard vs genuine do-nothing


@pytest.mark.parametrize(
    "log_text",
    [
        "INFO  2026-11-14T12:00:01 service=mcp key=railway type=remote server unavailable\n",
        "ERROR 2026-11-14T12:00:01 service=mcp key=railway error=ECONNREFUSED failed to connect\n",
        "WARN  service=mcp server=railway connection timed out\n",
        "ERROR mcp client railway endpoint unreachable\n",
    ],
)
def test_mcp_unavailable_zero_calls_marked_infra(exported_ds, sample_timetable, tmp_path, log_text):
    task_dir = exported_ds / "T1-101"
    token = (task_dir / "tests" / "token").read_text(encoding="utf-8").strip()
    tt = sample_timetable.model_copy(update={"seed": 101, "tier": "T1"})
    zero_log = _play_inproc(tt, [])
    agent_dir = _agent_logs(tmp_path, "mcp_fail", log_text)
    assert mcp_startup_failure(agent_dir) is not None

    logs_dir = tmp_path / "v_fail"
    with _fake_sim(token, zero_log) as sim_url:
        res = _run_verify_cli(task_dir, logs_dir, sim_url, agent_logs_dir=agent_dir)
    assert res.returncode != 0
    assert "INFRA: agent started without MCP tools" in res.stdout
    assert not (logs_dir / "reward.txt").exists()


@pytest.mark.parametrize(
    ("case_id", "log_text"),
    [
        ("no_logs", None),
        ("empty", ""),
        ("healthy", "INFO service=mcp key=railway type=remote connected tools=14\n"),
        ("probe_import", "ModuleNotFoundError: No module named 'railway'\n"),
        ("other_server", "INFO service=mcp key=toy server unavailable\n"),
        ("prompt_line", 'INFO session prompt="...tools of the `railway` MCP server; crew timeouts..."\n'),
    ],
)
def test_genuine_do_nothing_and_nonzero_calls_scored_cleanly(
    exported_ds, sample_timetable, tmp_path, case_id, log_text
):
    task_dir = exported_ds / "T1-101"
    token = (task_dir / "tests" / "token").read_text(encoding="utf-8").strip()
    tt = sample_timetable.model_copy(update={"seed": 101, "tier": "T1"})
    zero_log = _play_inproc(tt, [])
    agent_dir = _agent_logs(tmp_path, f"agent_{case_id}", log_text)

    logs_dir = tmp_path / f"v_{case_id}"
    with _fake_sim(token, zero_log) as sim_url:
        res = _run_verify_cli(task_dir, logs_dir, sim_url, agent_logs_dir=agent_dir)
    assert res.returncode == 0, res.stdout + res.stderr
    assert (logs_dir / "reward.txt").is_file()

    # Even when a failure line is present, an episode with >=1 budgeted tool call is scored normally.
    if case_id == "no_logs":
        one_call = _play_inproc(tt, [("end_shift", {})])
        fail_agent = _agent_logs(
            tmp_path,
            "agent_transient",
            "INFO service=mcp key=railway type=remote server unavailable\n",
        )
        logs_ok = tmp_path / "v_transient"
        with _fake_sim(token, one_call) as sim_url:
            res_ok = _run_verify_cli(task_dir, logs_ok, sim_url, agent_logs_dir=fail_agent)
        assert res_ok.returncode == 0
        assert (logs_ok / "reward.txt").is_file()


# ---------------------------------------------------------------- 5b. Crashed agent CLI vs genuine do-nothing

FATAL_CLI_LINES = [
    'ERROR 2026-11-14T12:00:02 service=llm status=401 {"error":{"type":"authentication_error"}}\n',
    "Error: 402 Payment Required: insufficient credits on this account\n",
    "APIError: HTTP 403 Forbidden\n",
    'ERROR service=session error="429 Too Many Requests" giving up\n',
    '{"error": {"code": "insufficient_quota", "message": "You exceeded your current quota"}}\n',
    "anthropic.BadRequestError: Your credit balance is too low to access the Anthropic API.\n",
    "openai.AuthenticationError: Incorrect API key provided\n",
    "Traceback (most recent call last):\n  File \"agent.py\", line 3, in <module>\nKeyError: 'x'\n",
]

BENIGN_LINES = [
    "",
    "INFO service=mcp key=railway type=remote connected tools=14\n",
    'INFO session prompt="...tools of the `railway` MCP server; crew timeouts..."\n',
    "INFO agent finished: nothing to do, stopping without tool calls\n",
    "ModuleNotFoundError: No module named 'railway'\n",
    "INFO exit code 0 after 4012 ms\n",
]


@pytest.mark.parametrize("line", FATAL_CLI_LINES)
def test_agent_cli_failure_detects_fatal_lines(tmp_path, line):
    assert agent_cli_failure(_agent_logs(tmp_path, "fatal", line)) is not None


@pytest.mark.parametrize("line", BENIGN_LINES)
def test_agent_cli_failure_ignores_benign_lines(tmp_path, line):
    assert agent_cli_failure(_agent_logs(tmp_path, "benign", line)) is None
    assert agent_cli_failure(tmp_path / "missing_dir") is None


@pytest.mark.parametrize(
    "line", [FATAL_CLI_LINES[i] for i in (0, 1, 5, 7)], ids=["http401", "credits402", "balance", "traceback"]
)
def test_crashed_agent_cli_with_zero_calls_is_infra_not_do_nothing(exported_ds, sample_timetable, tmp_path, line):
    task_dir = exported_ds / "T1-101"
    token = (task_dir / "tests" / "token").read_text(encoding="utf-8").strip()
    tt = sample_timetable.model_copy(update={"seed": 101, "tier": "T1"})
    zero_log = _play_inproc(tt, [])
    assert zero_log["entries"] == []
    agent_dir = _agent_logs(tmp_path, "crashed_cli", line)

    logs_dir = tmp_path / "v_crashed"
    with _fake_sim(token, zero_log) as sim_url:
        res = _run_verify_cli(task_dir, logs_dir, sim_url, agent_logs_dir=agent_dir)
    assert res.returncode != 0
    assert "INFRA: agent CLI failed before any tool call" in res.stdout
    assert not (logs_dir / "reward.txt").exists()


def test_nop_and_stop_early_still_scored_as_do_nothing(exported_ds, sample_timetable, tmp_path):
    """`harbor run -a nop` (no agent logs) and a stop-early agent keep a valid do-nothing reward."""
    task_dir = exported_ds / "T1-101"
    token = (task_dir / "tests" / "token").read_text(encoding="utf-8").strip()
    tt = sample_timetable.model_copy(update={"seed": 101, "tier": "T1"})
    zero_log = _play_inproc(tt, [])
    stop_early = _play_inproc(tt, [("end_shift", {})])

    cases = {
        "nop_no_logs": (zero_log, _agent_logs(tmp_path, "nop", None)),
        "nop_empty_log": (zero_log, _agent_logs(tmp_path, "nop_empty", "")),
        # >= 1 logged MCP call: the agent played, so a later API error never voids its score.
        "stop_early_then_429": (stop_early, _agent_logs(tmp_path, "stop_early", FATAL_CLI_LINES[3])),
    }
    for name, (log, agent_dir) in cases.items():
        logs_dir = tmp_path / f"v_{name}"
        with _fake_sim(token, log) as sim_url:
            res = _run_verify_cli(task_dir, logs_dir, sim_url, agent_logs_dir=agent_dir)
        assert res.returncode == 0, f"{name}: {res.stdout}{res.stderr}"
        reward = float((logs_dir / "reward.txt").read_text(encoding="utf-8"))
        assert reward == pytest.approx(-log["final_cost"]) and reward < 0, name


# ---------------------------------------------------------------- 6. Modal Harbor runner configuration


def test_harbor_runner_benchmark_presets_and_opencode_config():
    assert OPENCODE_VERSION == "1.19.5"
    assert VOLUME_NAME == "railway-harbor-logs"
    assert {"nop", "oracle", "fable-high", "qwen38-think", "gemini-carbon-high", "nemotron-super"} <= set(MODEL_PRESETS)

    cfg = build_opencode_config("http://sim:8000/mcp")
    assert cfg["permission"] == {"*": "deny", "railway_*": "allow"}
    assert cfg["mcp"]["railway"]["url"] == "http://sim:8000/mcp"
    assert set(CUSTOM_PROVIDERS) == {"anthropic", "google", "modal", "verda"}
    assert "claude-fable-5-2" in CUSTOM_PROVIDERS["anthropic"]["models"]
    assert "gemini-4-carbon" in CUSTOM_PROVIDERS["google"]["models"]
    assert "qwen38" in CUSTOM_PROVIDERS["modal"]["models"]
    assert "nemotron-3.5-super-vl" in CUSTOM_PROVIDERS["verda"]["models"]

    # Ensure API keys use `{env:...}` placeholders and never embed literal keys.
    serialized = json.dumps(cfg)
    assert "{env:VLLM_API_KEY}" in serialized
    assert "{env:VERDA_INFERENCE_KEY}" in serialized

    fable_cfg, fable_key, fable_warm = build_agent_config(preset="fable-high")
    assert fable_cfg["name"] == "opencode"
    assert fable_cfg["model_name"] == "anthropic/claude-fable-5-2"
    assert fable_cfg["kwargs"]["reasoning_effort"] == "high"
    assert fable_key == "ANTHROPIC_API_KEY"
    assert fable_warm is None

    qwen_cfg, qwen_key, qwen_warm = build_agent_config(preset="qwen38-think")
    assert qwen_cfg["model_name"] == "modal/qwen38"
    assert qwen_key == "VLLM_API_KEY"
    assert qwen_warm is not None and ".modal.direct" in qwen_warm

    gem_cfg, gem_key, _ = build_agent_config(preset="gemini-carbon-high")
    assert gem_cfg["model_name"] == "google/gemini-4-carbon"
    assert gem_cfg["kwargs"]["reasoning_effort"] == "high"
    assert gem_key == "GOOGLE_GENERATIVE_AI_API_KEY"

    nemo_cfg, nemo_key, _ = build_agent_config(preset="nemotron-super")
    assert nemo_cfg["model_name"] == "verda/nemotron-3.5-super-vl"
    assert nemo_key == "VERDA_INFERENCE_KEY"
