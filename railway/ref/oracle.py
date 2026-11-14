"""Command-line entry point for the Harbor oracle.

v0.3 (design-changes §2.7): the oracle replays the task's stored C_best plan (`solution/plan.json`, the tool
calls `railway.ref.best` re-played to exactly C_best) through the MCP server, so it scores exactly 1.0. The
simulator is deterministic, so the same calls give the same episode. Without `--plan` it runs the scripted
reference dispatcher instead (a baseline, which now scores at most 1.0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway.sim.rules import RuleId

from .reference import Reference


class PlanReplay:
    """Plays a fixed list of tool calls ({"tool", "args"}) and stops if the shift has already ended."""

    def __init__(self, calls: list[dict]):
        self.calls = list(calls)

    def run(self, client) -> None:
        for call in self.calls:
            if client.call(call["tool"], call["args"]).startswith(f"REFUSED {RuleId.SHIFT_ENDED.value}"):
                return


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the railway Harbor oracle")
    parser.add_argument("--url", required=True, help="MCP endpoint URL")
    parser.add_argument("--plan", help="plan.json to replay (the task's C_best plan); default: the reference")
    args = parser.parse_args(argv)
    from railway.tools.client import McpClient

    policy = PlanReplay(json.loads(Path(args.plan).read_text(encoding="utf-8"))) if args.plan else Reference()
    policy.run(McpClient(args.url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
