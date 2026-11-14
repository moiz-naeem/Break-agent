"""A local fake OpenAI-compatible server for tests (T08 §2.5). No real model, key or network.

    with FakeModelServer(scripted([...])) as srv:
        client = ModelClient(srv.base_url, "fake", None)

Behaviours decide each reply: `scripted`, `policy`, `prose`, `double`, `flaky`.
A scripted call is a ToolCall-like object (`.tool`, `.args`), a dict `{"name", "arguments"}` (a dict,
JSON-encoded on the wire), a dict `{"name", "arguments_raw"}` (sent verbatim, e.g. broken JSON), or
`None` (a prose reply with no tool call).
"""

from __future__ import annotations

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

END_SHIFT = {"name": "end_shift", "arguments": {}}


def _wire(call) -> tuple[str, str]:
    """(name, arguments_raw) for one scripted call."""
    if hasattr(call, "tool"):
        return call.tool, json.dumps(call.args)
    if "arguments_raw" in call:
        return call["name"], call["arguments_raw"]
    return call["name"], json.dumps(call.get("arguments") or {})


def _completion(body: dict, calls: list, content: str | None, n: int) -> dict:
    tool_calls = []
    for k, call in enumerate(calls):
        name, raw = _wire(call)
        tool_calls.append({"id": f"fake_{n}_{k}", "type": "function", "function": {"name": name, "arguments": raw}})
    prompt_chars = len(json.dumps(body.get("messages", []), sort_keys=True))
    completion_chars = sum(len(tc["function"]["arguments"]) + 10 for tc in tool_calls) + len(content or "")
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"fake-{n}",
        "object": "chat.completion",
        "model": body.get("model", "fake"),
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": prompt_chars // 4 + 1, "completion_tokens": completion_chars // 4 + 1},
    }


class Behaviour:
    """Returns (http_status, json_body) for each request. Calls are serialized by the server."""

    def respond(self, body: dict, n: int) -> tuple[int, dict]:
        raise NotImplementedError


class Scripted(Behaviour):
    def __init__(self, calls):
        self.queue = deque(calls)

    def next_call(self):
        return self.queue.popleft() if self.queue else END_SHIFT

    def respond(self, body, n):
        call = self.next_call()
        if call is None:
            return 200, _completion(body, [], "Let me think about the trains first.", n)
        return 200, _completion(body, [call], None, n)


class Double(Scripted):
    """Two tool calls per reply: the next scripted call, then a `status` call."""

    def respond(self, body, n):
        call = self.next_call()
        second = END_SHIFT if call is END_SHIFT else {"name": "status", "arguments": {}}
        return 200, _completion(body, [call, second], None, n)


class Prose(Behaviour):
    def respond(self, body, n):
        return 200, _completion(body, [], "I would hold the freight at S2 and let the express pass.", n)


class Flaky(Behaviour):
    def __init__(self, n_fail: int, calls, status: int = 500):
        self.n_fail, self.status, self.inner = n_fail, status, Scripted(calls)

    def respond(self, body, n):
        if self.n_fail > 0:
            self.n_fail -= 1
            return self.status, {"error": {"message": f"fake HTTP {self.status}"}}
        return self.inner.respond(body, n)


class Policy(Behaviour):
    """Test-only side channel: replays `decide_fn(static, state)` one call per reply.

    When the queue is empty it reads the live state with `session.peek("status", {})` (free, unlogged)
    and refills it. `decide_fn` must be pure.
    """

    def __init__(self, decide_fn, session, static):
        from railway.sim.observe import State

        self._state_cls = State
        self.decide_fn, self.session, self.static = decide_fn, session, static
        self.queue: deque = deque()

    def respond(self, body, n):
        if getattr(self.session, "closed", False):
            return 200, _completion(body, [END_SHIFT], None, n)
        if not self.queue:
            state = self._state_cls.model_validate_json(self.session.peek("status", {}))
            self.queue.extend(self.decide_fn(self.static, state) or [END_SHIFT])
        return 200, _completion(body, [self.queue.popleft()], None, n)


def scripted(calls) -> Behaviour:
    return Scripted(calls)


def policy(decide_fn, session, static=None) -> Behaviour:
    if static is None:
        from railway.sim.engine import Simulator

        static = Simulator.load(session.scenario).static_info()
    return Policy(decide_fn, session, static)


def prose() -> Behaviour:
    return Prose()


def double(calls=()) -> Behaviour:
    return Double(calls)


def flaky(n: int, calls=(), status: int = 500) -> Behaviour:
    return Flaky(n, calls, status)


class FakeModelServer:
    """`/v1/chat/completions` on 127.0.0.1 and a free port. Records request bodies and headers."""

    def __init__(self, behaviour: Behaviour, host: str = "127.0.0.1", port: int = 0):
        self.behaviour = behaviour
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - http.server API
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    self._send(404, {"error": {"message": "not found"}})
                    return
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                with outer._lock:
                    outer.requests.append(body)
                    outer.headers.append({k.lower(): v for k, v in self.headers.items()})
                    status, payload = outer.behaviour.respond(body, len(outer.requests))
                self._send(status, payload)

            def _send(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):  # keep test output quiet
                pass

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def start(self) -> FakeModelServer:
        self._thread = threading.Thread(target=self._server.serve_forever, args=(0.05,), daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> FakeModelServer:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
