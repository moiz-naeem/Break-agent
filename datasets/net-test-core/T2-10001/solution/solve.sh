#!/usr/bin/env bash
# Oracle: booked-order dispatcher playing strictly through the MCP server.
set -euo pipefail
python3 "$(dirname "$0")/solve.py" --url http://sim:8000/mcp
