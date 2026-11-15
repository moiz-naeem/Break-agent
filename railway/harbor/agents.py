"""Harbor baseline agents (T07 §2.9).

- `railway.harbor.agents:NetStopEarlyAgent` (v0.3 network tasks, `datasets/net-*`): runs the oracle client
  `net_templates/solve.py --stop-at 30` in the main container: it dispatches in booked order over MCP for the
  first 30 minutes, then ends the shift.
- `railway.harbor.agents:StopEarlyAgent` (v0.2 single-track tasks, `archive/datasets/t1-dev-mini`): uploads a vendored
  `railway` (sim, tools, policies) into the main container for this trial only (it is never part of the image)
  and plays `StopEarly` through the task's MCP server, like any agent would.
"""

from __future__ import annotations

import shlex
import tempfile
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from railway.harbor.export import vendor

TARGET = "/tmp/railway-baseline"
MCP_URL = "http://sim:8000/mcp"
PLAY = (
    "from railway.policies.baselines import StopEarly; from railway.tools.client import McpClient\n"
    f"with McpClient({MCP_URL!r}) as client: StopEarly().run(client)"
)


class StopEarlyAgent(BaseAgent):
    capabilities = AgentCapabilities(mcp_servers=True)  # Harbor refuses agents without it on MCP tasks

    @staticmethod
    def name() -> str:
        return "railway-stop-early"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vendor(Path(tmp) / "railway", ("sim", "tools", "policies"))
            await environment.upload_dir(Path(tmp) / "railway", f"{TARGET}/railway")

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        result = await environment.exec(f"PYTHONPATH={TARGET} python3 -c {shlex.quote(PLAY)}", timeout_sec=600)
        if result.return_code != 0:  # a crash must be a trial error, never a silent 0
            raise RuntimeError(f"StopEarly failed ({result.return_code}): {result.stderr or result.stdout}")


NET_SOLVE = Path(__file__).resolve().parent / "net_templates" / "solve.py"
NET_STOP_AT_MIN = 30


class NetStopEarlyAgent(BaseAgent):
    """v0.3 network baseline: booked-order dispatching to minute 30, then `end_shift` (needs only `mcp`)."""

    capabilities = AgentCapabilities(mcp_servers=True)

    @staticmethod
    def name() -> str:
        return "railway-net-stop-early"

    def version(self) -> str:
        return "0.3.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        script = NET_SOLVE.read_text(encoding="utf-8")
        cmd = f"python3 -c {shlex.quote(script)} --url {MCP_URL} --stop-at {NET_STOP_AT_MIN}"
        result = await environment.exec(cmd, timeout_sec=600)
        if result.return_code != 0:  # a crash must be a trial error, never a silent 0
            raise RuntimeError(f"NetStopEarly failed ({result.return_code}): {result.stderr or result.stdout}")
