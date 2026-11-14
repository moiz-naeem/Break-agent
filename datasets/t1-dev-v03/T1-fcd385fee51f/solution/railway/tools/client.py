"""Clients (T05 §2.3). Policies talk only to a `Client`, so they run the same in-process and over MCP."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Protocol

from mcp import Client as _McpSdkClient


class Client(Protocol):
    def call(self, tool: str, args: dict) -> str: ...


class InProcClient:
    def __init__(self, session):
        self.session = session

    def call(self, tool: str, args: dict) -> str:
        return self.session.call(tool, args)


class McpClient:
    """Streamable HTTP to `http://host:port/mcp` over one persistent connection, behind a sync API.

    The SDK client is async and its context must be entered and left in one task, so a background
    thread owns an event loop that holds the connection and serves calls from a queue.
    """

    def __init__(self, url: str, timeout: float = 120):
        self.url, self.timeout = url, timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None
        self._ready = concurrent.futures.Future()
        self._thread = threading.Thread(target=lambda: asyncio.run(self._serve()), daemon=True)
        self._thread.start()
        self._ready.result(timeout)  # re-raises a connection failure here

    async def _serve(self):
        try:
            self._loop, self._queue = asyncio.get_running_loop(), asyncio.Queue()
            async with _McpSdkClient(self.url) as client:
                self._ready.set_result(None)
                while (item := await self._queue.get()) is not None:
                    fn, future = item
                    try:
                        future.set_result(await fn(client))
                    except Exception as e:  # handed to the calling thread
                        future.set_exception(e)
        except Exception as e:
            if not self._ready.done():
                self._ready.set_exception(e)
            else:
                raise

    def _run(self, fn):
        future = concurrent.futures.Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (fn, future))
        return future.result(self.timeout)

    def call(self, tool: str, args: dict) -> str:
        async def go(client):
            result = await client.call_tool(tool, args)
            return "".join(getattr(c, "text", "") for c in result.content)

        return self._run(go)

    def tool_names(self) -> list[str]:
        async def go(client):
            return [t.name for t in (await client.list_tools()).tools]

        return self._run(go)

    def close(self):
        if self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
            self._thread.join(self.timeout)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
