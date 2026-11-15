"""MCP server over streamable HTTP for the v0.3 railway network, plus subprocess spawn helper."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hmac
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from railway.util import canonical_json

from .formats import NET_RULES_VERSION
from .manual import TOOL_BUDGET
from .model import Timetable
from .session import Session, describe


def build(tt: Timetable, token: str, *, budget: int = TOOL_BUDGET) -> MCPServer:
    if not token:
        raise ValueError("empty token would accept 'Bearer '")
    session = Session(tt, budget=budget)
    lock = asyncio.Lock()
    server = MCPServer("railway-net", log_level="WARNING")

    async def call(tool_name: str, args: dict) -> str:
        async with lock:
            return session.call(tool_name, args)

    # Forward only non-default arguments so logged entries match in-process calls that omit defaults.
    @server.tool(description=describe("manual"))
    async def manual() -> str:
        return await call("manual", {})

    @server.tool(description=describe("network"))
    async def network(area: str | None = None) -> str:
        return await call("network", {} if area is None else {"area": area})

    @server.tool(description=describe("status"))
    async def status(area: str | None = None, compact: bool = False) -> str:
        args: dict = {}
        if area is not None:
            args["area"] = area
        if compact:
            args["compact"] = True
        return await call("status", args)

    @server.tool(description=describe("conflicts"))
    async def conflicts() -> str:
        return await call("conflicts", {})

    @server.tool(description=describe("timetable"))
    async def timetable(train: str | None = None) -> str:
        return await call("timetable", {} if train is None else {"train": train})

    @server.tool(description=describe("bulletins"))
    async def bulletins() -> str:
        return await call("bulletins", {})

    @server.tool(description=describe("kpis"))
    async def kpis() -> str:
        return await call("kpis", {})

    @server.tool(description=describe("stringline"))
    async def stringline(corridor: str, from_tick: int, to_tick: int) -> str:
        return await call("stringline", {"corridor": corridor, "from_tick": from_tick, "to_tick": to_tick})

    @server.tool(description=describe("notes"))
    async def notes(text: str) -> str:
        return await call("notes", {"text": text})

    @server.tool(description=describe("order"))
    async def order(resource: str, trains: list[str]) -> str:
        return await call("order", {"resource": resource, "trains": trains})

    @server.tool(description=describe("hold"))
    async def hold(train: str, until: int) -> str:
        return await call("hold", {"train": train, "until": until})

    @server.tool(description=describe("platform"))
    async def platform(train: str, station: str, n: int | str) -> str:
        return await call("platform", {"train": train, "station": station, "n": n})

    @server.tool(description=describe("route"))
    async def route(train: str, via: list[str]) -> str:
        return await call("route", {"train": train, "via": via})

    @server.tool(description=describe("skip"))
    async def skip(train: str, station: str) -> str:
        return await call("skip", {"train": train, "station": station})

    @server.tool(description=describe("short_turn"))
    async def short_turn(train: str, station: str) -> str:
        return await call("short_turn", {"train": train, "station": station})

    @server.tool(description=describe("cancel"))
    async def cancel(train: str) -> str:
        return await call("cancel", {"train": train})

    @server.tool(description=describe("single_line"))
    async def single_line(segment: str, on: bool) -> str:
        return await call("single_line", {"segment": segment, "on": on})

    @server.tool(description=describe("relieve_crew"))
    async def relieve_crew(train: str) -> str:
        return await call("relieve_crew", {"train": train})

    @server.tool(description=describe("advance_to_next_event"))
    async def advance_to_next_event() -> str:
        return await call("advance_to_next_event", {})

    @server.tool(description=describe("end_shift"))
    async def end_shift() -> str:
        return await call("end_shift", {})

    @server.custom_route("/health", methods=["GET"])
    async def health(_: Request) -> Response:
        return PlainTextResponse("ok")

    @server.custom_route("/episode_log", methods=["GET"])
    async def episode_log(request: Request) -> Response:
        if not hmac.compare_digest(request.headers.get("authorization", "").encode(), f"Bearer {token}".encode()):
            return PlainTextResponse("forbidden", status_code=403)
        async with lock:
            log = session.episode_log()
        return Response(canonical_json(log), media_type="application/json")

    return server


# ------------------------------------------------------------------ subprocess spawn helper


@dataclass
class Spawned:
    base: str
    token: str
    proc: subprocess.Popen
    log_path: Path
    captured: str | None = None

    @property
    def url(self) -> str:
        return self.base + "/mcp"

    def output(self) -> str:
        if self.captured is not None:
            return self.captured
        return self.log_path.read_text(encoding="utf-8", errors="replace")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stop(proc: subprocess.Popen, timeout: float) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.terminate()
    proc.wait(timeout)


@contextlib.contextmanager
def spawn(timetable: Timetable, *, budget: int = TOOL_BUDGET, timeout: float = 60):
    root = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else root

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        (tmp / "timetable.json").write_text(canonical_json(timetable.model_dump(mode="json")), encoding="utf-8")
        token = secrets.token_hex(16)
        (tmp / "token").write_text(token, encoding="utf-8")
        port = _free_port()
        cmd = [
            sys.executable,
            "-m",
            "railway.net.server",
            "--timetable",
            str(tmp / "timetable.json"),
            "--token-file",
            str(tmp / "token"),
            "--budget",
            str(budget),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        with open(tmp / "output.log", "wb") as out:
            proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)
            spawned = Spawned(f"http://127.0.0.1:{port}", token, proc, tmp / "output.log")
            try:
                deadline = time.monotonic() + timeout
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"railway-net exited with {proc.returncode}: {spawned.output()}")
                    with contextlib.suppress(httpx.HTTPError):
                        if httpx.get(spawned.base + "/health", timeout=1).text == "ok":
                            break
                    if time.monotonic() > deadline:
                        raise TimeoutError("railway-net did not become healthy")
                    time.sleep(0.1)
                yield spawned
            finally:
                _stop(proc, timeout)
        spawned.captured = spawned.output()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="railway-net")
    parser.add_argument("--timetable", "--scenario", dest="timetable", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--budget", type=int, default=TOOL_BUDGET)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    data = json.loads(Path(args.timetable).read_text(encoding="utf-8"))
    if "rules_version" in data and data.get("rules_version") != NET_RULES_VERSION:
        print("railway-net: timetable rules_version does not match this simulator", file=sys.stderr)
        return 2
    data.pop("rules_version", None)
    timetable = Timetable.model_validate(data)
    token = Path(args.token_file).read_text(encoding="utf-8").strip()
    server = build(timetable, token, budget=args.budget)
    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    server.run("streamable-http", host=args.host, port=args.port, transport_security=security)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
