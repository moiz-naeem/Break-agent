"""Modal backend for oc_bench: the simulator and OpenCode run in two separate Modal sandboxes.

The agent sandbox holds OpenCode, an empty working directory and only the API key its provider
needs. It has no repository, no scenario and no simulator code; it reaches the simulator only
through the MCP URL of the sim sandbox's tunnel, which is what the Harbor sidecar layout gives.
The episode log is fetched from this machine with a token the agent never sees.
"""

from __future__ import annotations

import io
import json
import secrets
import subprocess
import tarfile
import tempfile
import threading
import time
from pathlib import Path

import httpx
import modal
import oc_bench as ob

OPENCODE_VERSION = "2.3.0"
APP_NAME = "railway-ocbench"
# Which keys from .env each provider's agent sandbox receives; nothing else is passed in.
PROVIDER_KEYS = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "google": ["GOOGLE_GENERATIVE_AI_API_KEY"],
    "nvidia": ["NVIDIA_API_KEY"],
    "verda": ["VERDA_INFERENCE_KEY"],
    "modal": ["VLLM_API_KEY"],
    "opencode": [],
}

_lock = threading.Lock()
_cache: dict = {}


def _source_dir() -> Path:
    """The committed railway package and seed lists (git archive HEAD), as the sim image's /src."""
    src = Path(tempfile.mkdtemp(prefix="railway-src-"))
    archive = subprocess.run(["git", "-C", str(ob.ROOT), "archive", "HEAD", "railway", "seeds"],
                             capture_output=True, check=True)  # fmt: skip
    tarfile.open(fileobj=io.BytesIO(archive.stdout)).extractall(src, filter="data")
    return src


def _setup():
    with _lock:
        if not _cache:
            app = modal.App.lookup(APP_NAME, create_if_missing=True)
            sim = (
                modal.Image.debian_slim(python_version="3.12")
                .uv_pip_install("httpx>=0.28.1", "mcp>=2.3.0", "numpy>=2.4.6", "pydantic>=2.14.3", "pyyaml>=6.0.3",
                                "matplotlib>=3.11.2")  # fmt: skip
                .add_local_dir(_source_dir(), "/src")
            )
            agent = modal.Image.from_registry("node:22-bookworm-slim").run_commands(
                f"npm install -g @opencode/cli@{OPENCODE_VERSION}", "opencode --version"
            )
            _cache.update(app=app, sim=sim, agent=agent)
        return _cache["app"], _cache["sim"], _cache["agent"]


SIM_CMD = (
    "cd /src && python -c \"import sys; from railway.gen.generator import generate; "
    "from railway.gen.presets import PRESETS; from railway.util import canonical_json; "
    "open('/tmp/scenario.json','w').write(canonical_json(generate(PRESETS[sys.argv[1]], int(sys.argv[2]))"
    ".model_dump(mode='json')))\" \"$PRESET\" \"$SEED\" && printf '%s' \"$SIM_TOKEN\" > /tmp/token && "
    "exec python -m railway.server --scenario /tmp/scenario.json --token-file /tmp/token --host 0.0.0.0 --port 8000"
)

AGENT_CMD = (
    "mkdir -p /root/.config/opencode /work && printf '%s' \"$OC_CFG\" > /root/.config/opencode/opencode.json && "
    # stdin must be closed: `opencode run` waits for piped input to end before it starts.
    "cd /work && exec opencode run --standalone --format json --auto -m \"$OC_MODEL\" \"$OC_PROMPT\" < /dev/null"
)


def run_job(job: dict, out: Path, env_base: dict, spend) -> dict:
    prepared = ob.prepare(job, out)
    if isinstance(prepared, dict):
        return prepared
    jdir, row = prepared
    app, sim_image, agent_image = _setup()
    token = secrets.token_hex(16)
    t0 = time.monotonic()
    sim = agent = None
    killed = None
    log = None
    try:
        sim = modal.Sandbox.create(
            "sh", "-c", SIM_CMD, app=app, image=sim_image, encrypted_ports=[8000], cpu=1.0, memory=2048,
            timeout=int(ob.SHIFT_TIMEOUT_S + 1800),
            secrets=[modal.Secret.from_dict({"PRESET": job["preset"], "SEED": str(job["seed"]), "SIM_TOKEN": token})],
        )  # fmt: skip
        base = sim.tunnels()[8000].url.rstrip("/")
        deadline = time.monotonic() + 300
        while True:
            try:
                if httpx.get(base + "/health", timeout=10).text == "ok":
                    break
            except httpx.HTTPError:
                pass
            if sim.poll() is not None:
                raise RuntimeError(f"sim sandbox exited: {sim.stderr.read()[-2000:]}")
            if time.monotonic() > deadline:
                raise TimeoutError("sim sandbox not healthy after 300 s")
            time.sleep(2)
        row["sim_ready_s"] = round(time.monotonic() - t0)

        cfg = ob.opencode_config(base + "/mcp")
        (jdir / "opencode.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        env = {"OC_CFG": json.dumps(cfg), "OC_MODEL": row["model_arg"], "OC_PROMPT": ob.INSTRUCTION}
        for key in PROVIDER_KEYS.get(job["provider"], []):
            if key not in env_base:
                raise RuntimeError(f"{key} missing from .env")
            env[key] = env_base[key]
        agent = modal.Sandbox.create(
            "sh", "-c", AGENT_CMD, app=app, image=agent_image, cpu=1.0, memory=2048,
            timeout=int(ob.SHIFT_TIMEOUT_S + 600), secrets=[modal.Secret.from_dict(env)],
        )  # fmt: skip
        t_agent = time.monotonic()
        events = jdir / "events.jsonl"
        # Read stdout on this thread only: polling the sandbox from a second thread stalls the stream.
        last_check = 0.0
        with events.open("w", encoding="utf-8") as fo:
            for chunk in agent.stdout:
                fo.write(chunk)
                fo.flush()
                now = time.monotonic()
                if now - last_check < 2:
                    continue
                last_check = now
                live = ob.parse_events(events)
                if live["forbidden"]:
                    killed = "sandbox_violation"
                elif now - t_agent > ob.SHIFT_TIMEOUT_S:
                    killed = "timeout"
                elif spend is not None:
                    spend.set_live(live["usd"])
                    if spend.total() >= ob.ANTHROPIC_CAP_USD:
                        killed = "anthropic_cap"
                if killed:
                    agent.terminate()
                    break
        if not killed:
            agent.wait()
        try:
            (jdir / "stderr.log").write_text(agent.stderr.read(), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        row["exit"] = agent.returncode
        row["wall_s"] = round(time.monotonic() - t_agent)
        log = ob.fetch_log(base, token, row)
    except Exception as e:  # noqa: BLE001
        row["infra_error"] = f"{type(e).__name__}: {e}"
    finally:
        for sb in (agent, sim):
            if sb is not None:
                try:
                    sb.terminate()
                except Exception:  # noqa: BLE001
                    pass
    if not (jdir / "events.jsonl").exists():
        (jdir / "events.jsonl").write_text("", encoding="utf-8")
    return ob.finalize(job, out, jdir, row, log, killed, spend)
