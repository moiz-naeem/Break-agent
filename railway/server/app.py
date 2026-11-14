"""MCP server (T05 §2.4): one Session per process behind the 15 tools, plus /health and /episode_log."""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from railway.sim.constants import RULES_VERSION
from railway.sim.model import Scenario
from railway.tools.session import Session
from railway.util import canonical_json


def build(scenario: Scenario, token: str) -> MCPServer:
    if not token:
        raise ValueError("empty token would accept 'Bearer '")
    session = Session(scenario)
    lock = asyncio.Lock()
    # WARNING keeps uvicorn's per-request lines out of the output; nothing here prints scenario data.
    server = MCPServer("railway-sim", log_level="WARNING")

    async def call(tool: str, args: dict) -> str:
        async with lock:
            return session.call(tool, args)

    # Only arguments that differ from their default are forwarded, so the log matches an in-process
    # call that omits them. Calls the MCP schema rejects never get here: no budget, no log entry.
    @server.tool()
    async def manual() -> str:
        """The dispatcher manual: rules, costs, tools. See manual()."""
        return await call("manual", {})

    @server.tool()
    async def network() -> str:
        """Static network as JSON. See manual()."""
        return await call("network", {})

    @server.tool()
    async def timetable(train: str | None = None) -> str:
        """Timetable as JSON, for all trains or one. See manual()."""
        return await call("timetable", {} if train is None else {"train": train})

    @server.tool()
    async def status(compact: bool = False) -> str:
        """Current state, as JSON or as compact text. See manual()."""
        return await call("status", {"compact": True} if compact else {})

    @server.tool()
    async def bulletins() -> str:
        """Revealed events as JSON. See manual()."""
        return await call("bulletins", {})

    @server.tool()
    async def kpis() -> str:
        """Cost so far, counters and budget as JSON. See manual()."""
        return await call("kpis", {})

    @server.tool()
    async def stringline(from_tick: int, to_tick: int) -> str:
        """ASCII time-distance diagram for a tick window. See manual()."""
        return await call("stringline", {"from_tick": from_tick, "to_tick": to_tick})

    @server.tool()
    async def notes(text: str) -> str:
        """Replace your notes scratchpad. See manual()."""
        return await call("notes", {"text": text})

    @server.tool()
    async def authorize(train: str, station: str, track: str) -> str:
        """Grant a train movement authority to a station track. See manual()."""
        return await call("authorize", {"train": train, "station": station, "track": track})

    @server.tool()
    async def cancel_authority(train: str) -> str:
        """Cancel or truncate a train's authority. See manual()."""
        return await call("cancel_authority", {"train": train})

    @server.tool()
    async def hold(train: str, until_tick: int) -> str:
        """Hold a train at its station until a tick. See manual()."""
        return await call("hold", {"train": train, "until_tick": until_tick})

    @server.tool()
    async def relieve_crew(train: str) -> str:
        """Start a crew relief at a crew station. See manual()."""
        return await call("relieve_crew", {"train": train})

    @server.tool()
    async def reverse(train: str) -> str:
        """Reverse a gridlocked train. See manual()."""
        return await call("reverse", {"train": train})

    @server.tool()
    async def advance_to_next_event() -> str:
        """Advance time to the next decision point. See manual()."""
        return await call("advance_to_next_event", {})

    @server.tool()
    async def end_shift() -> str:
        """Run to the end of the shift and report the final cost. See manual()."""
        return await call("end_shift", {})

    @server.custom_route("/health", methods=["GET"])
    async def health(_: Request) -> Response:
        return PlainTextResponse("ok")

    @server.custom_route("/episode_log", methods=["GET"])
    async def episode_log(request: Request) -> Response:
        # Compare bytes: compare_digest raises TypeError on non-ASCII str, which would become a 500.
        if not hmac.compare_digest(request.headers.get("authorization", "").encode(), f"Bearer {token}".encode()):
            return PlainTextResponse("forbidden", status_code=403)
        async with lock:
            log = session.episode_log()
        return Response(canonical_json(log), media_type="application/json")

    return server


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="railway-sim")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    data = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    if data.get("rules_version") != RULES_VERSION:  # checked before validation, which would echo the scenario
        print("railway-sim: scenario rules_version does not match this simulator", file=sys.stderr)
        return 2
    scenario = Scenario.model_validate(data)
    server = build(scenario, Path(args.token_file).read_text(encoding="utf-8").strip())
    # The Host header is "sim:8000" inside compose; the sidecar is only reachable on the task network.
    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    server.run("streamable-http", host=args.host, port=args.port, transport_security=security)
    return 0
