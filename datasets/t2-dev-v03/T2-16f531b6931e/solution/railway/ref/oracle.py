"""Command-line entry point for the oracle: replay the task's best plan (or run the reference) over MCP."""

from __future__ import annotations

import argparse

from .reference import Reference


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the railway reference dispatcher")
    parser.add_argument("--url", required=True, help="MCP endpoint URL")
    parser.add_argument("--plan", help="plan.json: the calls that reach C_best; without it, the reference plays")
    args = parser.parse_args(argv)
    from railway.tools.client import McpClient

    client = McpClient(args.url)
    if args.plan is None:
        Reference().run(client)
        return 0
    import json
    from pathlib import Path

    from railway.policies.baselines import ENDED

    for call in json.loads(Path(args.plan).read_text(encoding="utf-8")):
        out = client.call(call["tool"], call["args"])
        if any(marker in out for marker in ENDED):
            return 0
    client.call("end_shift", {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
