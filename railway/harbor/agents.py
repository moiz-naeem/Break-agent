"""Harbor agents for score-0 baselines (T07 §2.9). Load with `-a railway.harbor.agents:StopEarlyAgent`.

The agent uploads a vendored `railway` (sim, tools, policies) into the main container for this trial only
(it is never part of the image) and plays `StopEarly` through the task's MCP server, like any agent would.
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
