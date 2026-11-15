"""Wake a served endpoint and check that it returns a well-formed tool call.

Usage:  python infra/modal/check_endpoint.py <base-url> <model-name>
The key is read from VLLM_API_KEY, or from the git-ignored .env at the repo root.
"""

import json
import os
import sys
import time
from pathlib import Path

import httpx


def key() -> str:
    if os.environ.get("VLLM_API_KEY"):
        return os.environ["VLLM_API_KEY"]
    for line in (Path(__file__).resolve().parents[2] / ".env").read_text().splitlines():
        if line.startswith("VLLM_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("VLLM_API_KEY not found")


def main(base: str, model: str) -> None:
    headers = {"Authorization": f"Bearer {key()}"}
    start = time.monotonic()
    while True:  # the first request starts a container and loads 55 GB of weights
        try:
            if httpx.get(f"{base}/health", timeout=60).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if time.monotonic() - start > 40 * 60:
            raise SystemExit("endpoint did not become healthy in 40 minutes")
        time.sleep(10)
    print(f"healthy after {time.monotonic() - start:.0f} s")

    unauth = httpx.get(f"{base}/v1/models", timeout=60).status_code
    print("request without the key ->", unauth, "(expected 401)")

    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Authorize train E1 to station S2 on track L1."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "authorize",
                    "description": "Give a train a movement authority to a station track.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "train": {"type": "string"},
                            "station": {"type": "string"},
                            "track": {"type": "string"},
                        },
                        "required": ["train", "station", "track"],
                    },
                },
            }
        ],
        "max_tokens": 2048,
    }
    t = time.monotonic()
    reply = httpx.post(f"{base}/v1/chat/completions", headers=headers, json=body, timeout=600).json()
    if "choices" not in reply:
        raise SystemExit(f"error reply: {json.dumps(reply)[:800]}")
    choice = reply["choices"][0]
    message = choice["message"]
    calls = [(c["function"]["name"], c["function"]["arguments"]) for c in message.get("tool_calls") or []]
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    print("finish_reason:", choice["finish_reason"])
    print("tool calls:", calls)
    print("reasoning characters:", len(reasoning))
    print("usage:", reply.get("usage"), f"| {time.monotonic() - t:.1f} s")
    ok = choice["finish_reason"] == "tool_calls" and calls and calls[0][0] == "authorize"
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"), sys.argv[2])
