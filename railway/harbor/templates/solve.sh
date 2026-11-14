#!/usr/bin/env bash
# Oracle: the reference dispatcher, playing through the MCP server exactly like an agent.
set -euo pipefail
PYTHONPATH="$(dirname "$0")" python3 -m railway.ref.oracle --url http://sim:8000/mcp
