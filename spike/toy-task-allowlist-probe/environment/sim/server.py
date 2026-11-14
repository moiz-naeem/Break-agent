"""Toy MCP sidecar for the day-1 Harbor spike (T01). One tool, a health check, a token-protected log."""
import argparse
import hmac
import json
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

CALLS: list[dict] = []
TOKEN = ""
server = MCPServer("toy")


@server.tool()
def ping(text: str) -> str:
    """Echo `text` back as pong:<text>."""
    CALLS.append({"tool": "ping", "text": text})
    return "pong:" + text


@server.custom_route("/health", methods=["GET"])
async def health(_: Request) -> Response:
    return PlainTextResponse("ok")


@server.custom_route("/episode_log", methods=["GET"])
async def episode_log(request: Request) -> Response:
    auth = request.headers.get("authorization", "")
    # Compare bytes: compare_digest raises TypeError on non-ASCII str, which would turn into a 500.
    if not hmac.compare_digest(auth.encode(), f"Bearer {TOKEN}".encode()):
        return PlainTextResponse("forbidden", status_code=403)
    return Response(json.dumps(CALLS), media_type="application/json")


def main() -> None:
    global TOKEN
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--token-file", default="/srv/token")
    a = p.parse_args()
    TOKEN = Path(a.token_file).read_text().strip()
    assert TOKEN, "empty token file would accept 'Bearer '"
    # Host header is "sim:8000" inside compose; the sidecar is only reachable on the task network.
    server.run("streamable-http", host=a.host, port=a.port,
               transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))


if __name__ == "__main__":
    main()
