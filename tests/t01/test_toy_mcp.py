import asyncio
import json

import httpx
import pytest
from mcp import Client

from tests.helpers import REPO, start_server

SERVER = REPO / "archive" / "spike" / "toy-task" / "environment" / "sim" / "server.py"
TOKEN = "t" * 64


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    tok = tmp_path_factory.mktemp("tok") / "token"
    tok.write_text(TOKEN)
    with start_server([str(SERVER), "--token-file", str(tok)]) as base:
        yield base


def _call(base, coro_fn):
    async def run():
        async with Client(f"{base}/mcp") as c:
            return await coro_fn(c)
    return asyncio.run(run())


def test_T01_11_health(toy):
    r = httpx.get(f"{toy}/health")
    assert r.status_code == 200 and r.text == "ok"


def test_T01_12_list_tools(toy):
    tools = _call(toy, lambda c: c.list_tools())
    assert [t.name for t in tools.tools] == ["ping"]


def test_T01_13_ping_roundtrip(toy):
    res = _call(toy, lambda c: c.call_tool("ping", {"text": "7"}))
    assert res.content[0].text == "pong:7"


def test_T01_14_episode_log_requires_token(toy):
    _call(toy, lambda c: c.call_tool("ping", {"text": "7"}))
    assert httpx.get(f"{toy}/episode_log").status_code == 403
    bad = httpx.get(f"{toy}/episode_log", headers={"Authorization": "Bearer wrong"})
    assert bad.status_code == 403
    ok = httpx.get(f"{toy}/episode_log", headers={"Authorization": f"Bearer {TOKEN}"})
    assert ok.status_code == 200
    calls = json.loads(ok.text)
    assert {"tool": "ping", "text": "7"} in calls


def test_T01_15_no_other_routes(toy):
    assert httpx.get(f"{toy}/").status_code == 404
    assert httpx.get(f"{toy}/scenario").status_code == 404
    assert httpx.post(f"{toy}/episode_log").status_code == 405
