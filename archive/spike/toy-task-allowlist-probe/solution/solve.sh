#!/usr/bin/env bash
# Toy oracle: one ping over MCP with this container's uid.
set -euo pipefail
python3 - <<'EOF'
import asyncio
import os

from mcp import Client


async def main():
    async with Client("http://sim:8000/mcp") as c:
        r = await c.call_tool("ping", {"text": str(os.getuid())})
        print(r.content[0].text)


asyncio.run(main())
EOF
