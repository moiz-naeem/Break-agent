"""T05-17..22: the MCP server in a subprocess on mini3 (and a T2 scenario for the stdout check)."""

import json
import subprocess
import sys

import httpx

from railway.gen.generator import generate
from railway.gen.presets import PRESETS
from railway.gen.seeds import candidate_seeds
from railway.server.spawn import spawn
from railway.tools.client import InProcClient, McpClient
from railway.tools.session import TOOLS
from tests.t05.common import EPISODE_A, session


def test_T05_17_lists_15_tools():
    with spawn(session().scenario) as srv, McpClient(srv.url) as client:
        names = client.tool_names()
    assert sorted(names) == sorted(TOOLS) and len(names) == 15
    assert not {"conflicts", "what_if"} & set(names)


def test_T05_18_parity_with_inproc():
    inproc = InProcClient(session())
    with spawn(session().scenario) as srv, McpClient(srv.url) as client:
        for tool, args, _ in [*EPISODE_A[:-1], ("status", {"compact": True}, None), *EPISODE_A[-1:]]:
            assert client.call(tool, args) == inproc.call(tool, args), tool


def test_T05_19_episode_log_token():
    with spawn(session().scenario) as srv, McpClient(srv.url) as client:
        for tool, args, _ in EPISODE_A:
            client.call(tool, args)
        assert httpx.get(srv.base + "/episode_log").status_code == 403
        bad = httpx.get(srv.base + "/episode_log", headers={"Authorization": "Bearer nope"})
        assert bad.status_code == 403
        ok = httpx.get(srv.base + "/episode_log", headers={"Authorization": f"Bearer {srv.token}"})
        assert ok.status_code == 200
        log = ok.json()
        assert set(log) == {
            "scenario_id", "scenario_hash", "rules_version", "entries", "closed",
            "final_tick", "final_cost", "final_cost_by_component",
        }  # fmt: skip
        assert log["closed"] is True and log["final_cost"] == 718.0
        assert [e["status"] for e in log["entries"]] == [status for *_, status in EPISODE_A]
        assert set(log["entries"][0]) == {"seq", "tick", "tool", "args", "status", "rule"}
        assert client.call("status", {}).startswith("REFUSED R-SHIFT-ENDED")


def test_T05_19b_episode_log_closes_open_session():
    with spawn(session().scenario) as srv, McpClient(srv.url) as client:
        client.call("kpis", {})
        log = httpx.get(srv.base + "/episode_log", headers={"Authorization": f"Bearer {srv.token}"}).json()
        assert log["closed"] is True and log["final_cost"] == 1176.0  # do-nothing cost on mini3
        assert client.call("kpis", {}).startswith("REFUSED R-SHIFT-ENDED")


def test_T05_20_routes():
    with spawn(session().scenario) as srv:
        assert httpx.get(srv.base + "/health").text == "ok"
        for path in ("/", "/scenario", "/state"):
            assert httpx.get(srv.base + path).status_code == 404, path
        assert httpx.post(srv.base + "/episode_log").status_code == 405


def test_T05_21_rules_version_mismatch(tmp_path):
    data = session().scenario.model_dump(mode="json") | {"rules_version": "9.9-other"}
    (tmp_path / "s.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "token").write_text("t0ken", encoding="utf-8")
    cmd = [sys.executable, "-m", "railway.server", "--scenario", str(tmp_path / "s.json")]
    proc = subprocess.run(
        [*cmd, "--token-file", str(tmp_path / "token"), "--port", "0"], capture_output=True, timeout=60
    )
    assert proc.returncode == 2


def test_T05_22_nothing_secret_on_stdout():
    scenario = generate(PRESETS["T2"], next(iter(candidate_seeds("dev", count=1))))
    event_trains = {e.train for e in scenario.events if getattr(e, "train", None)}
    assert event_trains, "need a scenario with train events"
    with spawn(scenario) as srv, McpClient(srv.url) as client:
        for _ in range(10):
            client.call("advance_to_next_event", {})
            client.call("status", {"compact": True})
        httpx.get(srv.base + "/episode_log", headers={"Authorization": f"Bearer {srv.token}"})
    output = srv.output()
    assert srv.token not in output
    for train in event_trains:
        assert train not in output, train
