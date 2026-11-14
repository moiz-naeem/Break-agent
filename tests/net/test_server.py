"""Tests for `railway/net/server.py`: streamable HTTP MCP server, auth, parity and a full T1 run."""

from __future__ import annotations

import json
import re
import subprocess
import sys

import httpx

from railway.net.replay import replay
from railway.net.server import spawn
from railway.net.session import TOOLS, Session
from railway.tools.client import InProcClient, McpClient


def test_mcp_server_lists_all_tools_and_health_routes(sample_timetable):
    with spawn(sample_timetable) as srv, McpClient(srv.url) as client:
        names = client.tool_names()
        assert sorted(names) == sorted(TOOLS) and len(names) == 14
        assert httpx.get(srv.base + "/health").text == "ok"
        for path in ("/", "/timetable", "/state"):
            assert httpx.get(srv.base + path).status_code == 404, path
        assert httpx.post(srv.base + "/episode_log").status_code == 405


def test_mcp_server_parity_with_inproc_and_episode_log_auth(sample_timetable):
    script = [
        ("network", {"area": "JA"}),
        ("timetable", {"train": "IC1"}),
        ("advance_to_next_event", {}),
        ("conflicts", {}),
        ("notes", {"text": "IC1 first"}),
        ("status", {"compact": True}),
        ("order", {"resource": "A-JA.u:0", "trains": ["IC1", "L4"]}),
        ("order", {"resource": "A.e1/A.e3", "trains": ["IC1", "L4"]}),
        ("platform", {"train": "IC1", "station": "C", "n": 2}),
        ("hold", {"train": "L4", "until": 14}),
        ("kpis", {}),
        ("end_shift", {}),
    ]
    inproc_sess = Session(sample_timetable)
    inproc = InProcClient(inproc_sess)

    with spawn(sample_timetable) as srv, McpClient(srv.url) as client:
        for tool_name, args in script:
            assert client.call(tool_name, args) == inproc.call(tool_name, args), tool_name

        # Bearer token check on /episode_log.
        assert httpx.get(srv.base + "/episode_log").status_code == 403
        assert httpx.get(srv.base + "/episode_log", headers={"Authorization": "Bearer wrong"}).status_code == 403
        ok = httpx.get(srv.base + "/episode_log", headers={"Authorization": f"Bearer {srv.token}"})
        assert ok.status_code == 200
        log = ok.json()
        assert log["closed"] is True
        assert [e["tool"] for e in log["entries"]] == [t for t, _ in script]
        assert set(log["entries"][0]) == {"seq", "tick", "tool", "args", "status", "rule"}
        assert client.call("status", {}).startswith("REFUSED R-SHIFT-ENDED")
    assert srv.token not in srv.output()
    assert replay(sample_timetable, log) == inproc_sess.close()


def test_mcp_server_rules_version_mismatch_exits_cleanly(sample_timetable, tmp_path):
    bad = sample_timetable.model_dump(mode="json") | {"rules_version": "9.9-wrong"}
    (tmp_path / "tt.json").write_text(json.dumps(bad), encoding="utf-8")
    (tmp_path / "tok").write_text("secret123", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "railway.net.server",
            "--timetable",
            str(tmp_path / "tt.json"),
            "--token-file",
            str(tmp_path / "tok"),
            "--port",
            "0",
        ],
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 2


def test_t1_episode_played_by_scripted_policy_over_spawned_mcp(t1_timetable):
    """Play a full 180-min T1 shift over HTTP MCP with a trivial conflict-ordering policy, then replay."""
    tt_data: dict[str, int] = {}
    with spawn(t1_timetable) as srv, McpClient(srv.url) as client:
        tt_json = json.loads(client.call("timetable", {}))
        for train in tt_json["trains"]:
            tt_data[train["id"]] = train["entry_tick"]

        steps = 0
        while steps < 250:
            steps += 1
            adv = client.call("advance_to_next_event", {})
            if "stop: SHIFT_END" in adv or "SHIFT ENDED" in adv:
                break
            conf_out = client.call("conflicts", {})
            for m in re.finditer(r"^C\d+\s+(\S+)\s+\([^)]+\):\s+(.+)$", conf_out, re.M):
                resource = m.group(1)
                trains = [part.strip().split()[0] for part in m.group(2).split("|")]
                trains.sort(key=lambda t: (tt_data.get(t, 0), t))
                res = client.call("order", {"resource": resource, "trains": trains})
                assert res.startswith("GRANTED"), res

        resp = httpx.get(srv.base + "/episode_log", headers={"Authorization": f"Bearer {srv.token}"})
        assert resp.status_code == 200
        log = resp.json()

    assert log["closed"] is True
    assert log["final_tick"] == 180
    assert log["counts"]["invalid_calls"] == 0
    assert log["counts"]["refusals"] == 0
    assert log["counts"]["unfinished"] == 0
    assert log["kpis"]["on_time_pct"] == 100.0
    assert log["kpis"]["freight_delivered_pct"] == 100.0

    # Replaying the HTTP episode log in-process reproduces the exact same final state.
    replayed = replay(t1_timetable, log)
    assert [list(r) for r in replayed.state] == log["final_state"]
    assert all(phase == "done" for _, phase, _, _ in replayed.state)
