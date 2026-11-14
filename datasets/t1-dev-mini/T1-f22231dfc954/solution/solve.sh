#!/usr/bin/env bash
# Oracle: replays this task's stored C_best plan through the MCP server, exactly like an agent (scores 1.0).
set -euo pipefail
PYTHONPATH="$(dirname "$0")" python3 -m railway.ref.oracle --url http://sim:8000/mcp --plan "$(dirname "$0")/plan.json"
