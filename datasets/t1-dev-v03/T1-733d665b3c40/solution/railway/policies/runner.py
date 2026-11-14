"""run_policy (T05 §2.6): play one episode in-process or over MCP and return what happened."""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from railway.sim.model import Scenario
from railway.tools import formats
from railway.tools.client import InProcClient, McpClient
from railway.tools.replay import replay
from railway.tools.session import FinalResult, Session


@dataclass
class EpisodeResult:
    final: FinalResult
    log: list[dict]
    decision_points: int
    tool_calls: int
    wall_s: float
    output_chars: int = 0  # in-process only: advance outputs + one compact status per decision point (T05-33)
    unfinished: int | None = None  # in-process only: trains not finished at the end (T06 seed filter)


class _Measured(InProcClient):
    def __init__(self, session: Session):
        super().__init__(session)
        self.output_chars = 0

    def call(self, tool: str, args: dict) -> str:
        out = super().call(tool, args)
        if tool == "advance_to_next_event":
            self.output_chars += len(out) + len(formats.compact_status(self.session.state(), self.session.scenario))
        return out


def run_policy(policy, scenario: Scenario, *, transport: str = "inproc") -> EpisodeResult:
    if transport == "inproc":
        session = Session(scenario)
        client = _Measured(session)
        start = time.perf_counter()
        policy.run(client)
        wall = time.perf_counter() - start
        final, log, chars = session.close(), session.log, client.output_chars
        unfinished = sum(t.phase != "finished" for t in session.sim.state().trains)
    elif transport == "mcp":
        from railway.server.spawn import spawn

        with spawn(scenario) as server, McpClient(server.url) as client:
            start = time.perf_counter()
            policy.run(client)
            wall = time.perf_counter() - start
            auth = {"Authorization": f"Bearer {server.token}"}
            episode = httpx.get(server.base + "/episode_log", headers=auth, timeout=60).json()
        final, log, chars, unfinished = replay(scenario, episode), episode["entries"], 0, None  # replay: counts
        if final.cost.total != episode["final_cost"]:
            raise RuntimeError(f"replay cost {final.cost.total} != live cost {episode['final_cost']}")
    else:
        raise ValueError(f"unknown transport {transport!r}")
    return EpisodeResult(final, log, final.counts["decision_points"], len(log), wall, chars, unfinished)
