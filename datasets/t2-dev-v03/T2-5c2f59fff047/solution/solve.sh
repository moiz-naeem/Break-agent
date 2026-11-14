#!/usr/bin/env bash
# Oracle: replays the best known plan for this task through the MCP server, exactly like an agent would.
set -euo pipefail
HERE="$(dirname "$0")"
PYTHONPATH="$HERE" python3 -m railway.ref.oracle --url http://sim:8000/mcp --plan "$HERE/plan.json"
