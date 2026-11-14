"""Custom Harbor agent for the T01 spike (fact F-CUSTOM): proves Harbor can load our agents."""
from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class NoOpAgent(BaseAgent):
    # Our tasks declare MCP servers; Harbor refuses agents that don't claim MCP support.
    capabilities = AgentCapabilities(mcp_servers=True)

    @staticmethod
    def name() -> str:
        return "railway-noop"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        pass
