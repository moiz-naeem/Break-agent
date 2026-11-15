"""Command-line entry point for the MCP reference dispatcher."""

from __future__ import annotations

import argparse

from .reference import Reference


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the railway reference dispatcher")
    parser.add_argument("--url", required=True, help="MCP endpoint URL")
    args = parser.parse_args(argv)
    from railway.tools.client import McpClient

    Reference().run(McpClient(args.url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
