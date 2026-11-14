"""Two-Sandbox Modal environment plugin for Harbor 0.26 sidecar (`sim` + `main`) tasks.

Loaded by `harbor run` inside the Modal orchestrator container via:
`environment.import_path = "infra.modal._harbor_sidecar_env:ModalSidecarEnvironment"`.

Instead of Docker-in-Docker (`_ModalDinD`), each trial runs two hardware-isolated Modal Sandboxes:
1. `sim` Sandbox: pre-cached Python 3.11 + `requirements.txt` image, `/srv` uploaded from
   `<task>/environment/sim` (`railway/`, `timetable.json`, `token`), serving `railway.net.server`
   on encrypted tunnel port 8000.
2. `main` Sandbox: pre-cached image built from `<task>/environment/Dockerfile` in an empty build
   context (zero simulator code, zero `timetable.json`, zero `token`), with outbound domain
   allowlist restricted to the task's `allowed_hosts` plus this trial's `sim` tunnel host, and a
   root-only local streaming HTTP->HTTPS proxy at `sim:8000` (`127.0.0.1:8000`).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shlex
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, override
from uuid import uuid4

import httpx
from harbor.environments.base import ExecResult
from harbor.environments.definition import parse_dockerfile_workdir
from harbor.environments.modal import ModalEnvironment, _ModalDirect
from harbor.environments.tar_transfer import pack_dir_to_file, remote_unpack_command
from harbor.models.task.config import NetworkMode, NetworkPolicy
from modal import App, Image, Sandbox

_IMAGE_LOCK = threading.Lock()
_SIM_IMAGES: dict[str, Image] = {}
_MAIN_IMAGES: dict[str, Image] = {}

_PROXY_SCRIPT = """\
import contextlib
import sys
import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.routing import Route

TARGET = sys.argv[1].rstrip("/")
STRIP_REQ = {"host", "connection", "keep-alive", "transfer-encoding", "content-length", "upgrade"}
STRIP_RESP = {"connection", "keep-alive", "transfer-encoding", "content-length", "content-encoding"}
CLIENT: httpx.AsyncClient | None = None


@contextlib.asynccontextmanager
async def lifespan(_app):
    global CLIENT
    CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0), follow_redirects=True)
    try:
        yield
    finally:
        await CLIENT.aclose()


async def forward(request: Request) -> StreamingResponse:
    assert CLIENT is not None
    qs = f"?{request.url.query}" if request.url.query else ""
    url = f"{TARGET}{request.url.path}{qs}"
    headers = [(k, v) for k, v in request.headers.raw if k.decode("latin1").lower() not in STRIP_REQ]
    body = await request.body()
    req = CLIENT.build_request(request.method, url, headers=headers, content=body)
    resp = await CLIENT.send(req, stream=True)
    out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in STRIP_RESP}

    async def chunks():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(chunks(), status_code=resp.status_code, headers=out_headers)


app = Starlette(
    routes=[Route("/{path:path}", forward, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])],
    lifespan=lifespan,
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
"""


def _sim_image_for(sim_dir: Path) -> Image:
    req_bytes = (sim_dir / "requirements.txt").read_bytes()
    digest = hashlib.sha256(req_bytes).hexdigest()[:16]
    with _IMAGE_LOCK:
        if digest not in _SIM_IMAGES:
            pins = [line.strip() for line in req_bytes.decode("utf-8").splitlines() if line.strip()]
            _SIM_IMAGES[digest] = Image.from_registry("python:3.11-slim").pip_install(*pins)
        return _SIM_IMAGES[digest]


def _main_image_for(dockerfile_path: Path) -> Image:
    df_bytes = dockerfile_path.read_bytes()
    digest = hashlib.sha256(df_bytes).hexdigest()[:16]
    with _IMAGE_LOCK:
        if digest not in _MAIN_IMAGES:
            ctx = Path("/tmp") / f"harbor_main_ctx_{digest}"
            ctx.mkdir(parents=True, exist_ok=True)
            clean_df = ctx / "Dockerfile"
            clean_df.write_bytes(df_bytes)
            _MAIN_IMAGES[digest] = Image.from_dockerfile(clean_df, context_dir=ctx)
        return _MAIN_IMAGES[digest]


def _parse_sim_cmd(sim_dockerfile: Path) -> list[str]:
    text = sim_dockerfile.read_text(encoding="utf-8")
    for line in reversed(text.splitlines()):
        m = re.match(r"^\s*CMD\s+(\[.*\])\s*$", line)
        if m:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return parsed
    return [
        "python",
        "-m",
        "railway.net.server",
        "--timetable",
        "/srv/timetable.json",
        "--token-file",
        "/srv/token",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]


class _ModalSidecarStrategy(_ModalDirect):
    """Spawns a `sim` Modal Sandbox + a separate `main` Modal Sandbox bridged at `http://sim:8000`."""

    def __init__(self, env: ModalSidecarEnvironment):
        super().__init__(env)
        self._sidecar_env = env
        self._sim_sandbox: Sandbox | None = None
        self._sim_proc: Any = None
        self._proxy_proc: Any = None

    async def _upload_sim_payload(self, sim_sb: Sandbox, sim_dir: Path) -> None:
        remote_archive = f"/tmp/sim_upload_{uuid4().hex}.tar.gz"
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = Path(tmp_dir) / "sim.tar.gz"
            pack_dir_to_file(sim_dir, archive_path)
            await sim_sb.filesystem.copy_from_local.aio(archive_path, remote_archive)
        proc = await sim_sb.exec.aio(
            "sh",
            "-c",
            f"{remote_unpack_command(remote_archive, '/srv')} && rm -f {shlex.quote(remote_archive)}",
            timeout=120,
        )
        await proc.wait.aio()
        if proc.returncode != 0:
            err = await proc.stderr.read.aio()
            raise RuntimeError(f"failed to unpack /srv into sim sandbox: {err}")

    @override
    async def start(self, force_build: bool) -> None:
        env = self._sidecar_env
        sim_dir = env.environment_dir / "sim"
        sim_image = _sim_image_for(sim_dir)
        env._image = _main_image_for(env._environment_definition_path)
        env._app = await App.lookup.aio(name=env._app_name, create_if_missing=True)

        # 1. Start isolated `sim` Sandbox and wait for /health == "ok".
        self._sim_sandbox = await Sandbox.create.aio(
            "sh",
            "-c",
            "sleep infinity",
            app=env._app,
            image=sim_image,
            encrypted_ports=[8000],
            cpu=1.0,
            memory=2048,
            timeout=env._sandbox_timeout,
        )
        try:
            await self._upload_sim_payload(self._sim_sandbox, sim_dir)
            sim_cmd = _parse_sim_cmd(sim_dir / "Dockerfile")
            self._sim_proc = await self._sim_sandbox.exec.aio(
                *sim_cmd,
                workdir="/srv",
                timeout=env._sandbox_timeout,
            )
            tunnels = await self._sim_sandbox.tunnels.aio()
            sim_base = tunnels[8000].url.rstrip("/")
            sim_host = urllib.parse.urlsplit(sim_base).hostname
            if not sim_host:
                raise RuntimeError(f"invalid sim tunnel URL: {sim_base}")

            deadline = time.monotonic() + 120.0
            async with httpx.AsyncClient(timeout=5.0) as client:
                while True:
                    if await self._sim_proc.poll.aio() is not None:
                        stderr = await self._sim_proc.stderr.read.aio()
                        stdout = await self._sim_proc.stdout.read.aio()
                        raise RuntimeError(f"sim server exited early: {stdout} {stderr}")
                    try:
                        resp = await client.get(f"{sim_base}/health")
                        if resp.status_code == 200 and resp.text.strip() == "ok":
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("sim sandbox /health did not become ready within 120s")
                    await asyncio.sleep(0.5)

            # 2. Start isolated `main` Sandbox with allowlist including `sim_host`.
            env._sim_tunnel_host = sim_host
            env._sandbox = await env._create_sandbox(entrypoint=env.options.keepalive)

            workdir = env.task_env_config.workdir or env._workdir
            if workdir:
                await env._sdk_exec(f"mkdir -p {shlex.quote(workdir)}")
            await env.ensure_dirs(env._mount_targets(writable_only=True))

            # 3. Install root-only `sim:8000` reverse proxy inside `main` (contains zero secrets/scenarios).
            setup_cmd = (
                "chmod 700 /root && "
                "grep -qE '(^|\\s)sim(\\s|$)' /etc/hosts || echo '127.0.0.1 sim' >> /etc/hosts && "
                f"cat > /root/.sim_proxy.py << 'PYEOF'\n{_PROXY_SCRIPT}PYEOF\n"
                "chmod 700 /root/.sim_proxy.py"
            )
            setup_res: ExecResult = await env._sdk_exec(setup_cmd, timeout_sec=30)
            if setup_res.return_code != 0:
                raise RuntimeError(f"failed setting up sim proxy in main sandbox: {setup_res.stderr}")

            self._proxy_proc = await env._sandbox.exec.aio(
                "python3",
                "/root/.sim_proxy.py",
                sim_base,
                workdir="/root",
                timeout=env._sandbox_timeout,
            )

            proxy_deadline = time.monotonic() + 30.0
            while True:
                if await self._proxy_proc.poll.aio() is not None:
                    perr = await self._proxy_proc.stderr.read.aio()
                    raise RuntimeError(f"main sim_proxy exited early: {perr}")
                probe = await env._sdk_exec("curl -sf --max-time 3 http://sim:8000/health", timeout_sec=10)
                if probe.return_code == 0 and (probe.stdout or "").strip() == "ok":
                    break
                if time.monotonic() > proxy_deadline:
                    raise TimeoutError("http://sim:8000/health inside main sandbox not ready after 30s")
                await asyncio.sleep(0.3)
        except BaseException:
            await self.stop(delete=True)
            raise

    @override
    async def stop(self, delete: bool) -> None:
        try:
            await super().stop(delete)
        finally:
            if self._sim_sandbox is not None:
                try:
                    await self._sim_sandbox.terminate.aio()
                    await self._sim_sandbox.wait.aio(raise_on_termination=False)
                except Exception as exc:  # noqa: BLE001
                    self._sidecar_env.logger.warning(f"Error terminating sim Modal sandbox: {exc}")
                finally:
                    self._sim_sandbox = None
                    self._sim_proc = None
                    self._proxy_proc = None


class ModalSidecarEnvironment(ModalEnvironment):
    """Harbor `BaseEnvironment` running `sim` and `main` in two separate Modal Sandboxes."""

    @classmethod
    @override
    def preflight(cls, kwargs: dict[str, Any] | None = None) -> None:
        cls.parse_options(kwargs)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("modal_vm_runtime", False)
        self._sim_tunnel_host: str | None = None
        super().__init__(*args, **kwargs)
        if (self.environment_dir / "sim").is_dir() and (self.environment_dir / "Dockerfile").is_file():
            self._compose_mode = False
            self._workdir = parse_dockerfile_workdir(self.environment_dir / "Dockerfile")
            self._strategy = _ModalSidecarStrategy(self)

    def _with_sim_tunnel(self, spec: dict[str, Any]) -> dict[str, Any]:
        domains = list(spec.get("outbound_domain_allowlist") or ())
        if self._sim_tunnel_host and self._sim_tunnel_host not in domains and "*" not in domains:
            domains.append(self._sim_tunnel_host)
        return {
            "outbound_domain_allowlist": domains,
            "outbound_cidr_allowlist": list(spec.get("outbound_cidr_allowlist") or ()),
        }

    @override
    def _allowlist_network_kwargs(  # type: ignore[override]
        self,
        network_policy: NetworkPolicy,
        *,
        compose_mode: bool = False,
    ) -> dict[str, Any]:
        # Pass compose_mode=True so unqualified service names like `sim` are filtered out of external DNS rules.
        base = dict(ModalEnvironment._allowlist_network_kwargs(network_policy, compose_mode=True))
        return self._with_sim_tunnel(base)

    @override
    def _dynamic_network_kwargs(  # type: ignore[override]
        self,
        network_policy: NetworkPolicy,
        *,
        compose_mode: bool = False,
    ) -> dict[str, Any]:
        if network_policy.network_mode == NetworkMode.PUBLIC:
            return {"outbound_domain_allowlist": ["*"], "outbound_cidr_allowlist": ["0.0.0.0/0"]}
        if network_policy.network_mode == NetworkMode.ALLOWLIST:
            return self._allowlist_network_kwargs(network_policy, compose_mode=True)
        # Even in NO_NETWORK mode, `sim` is an in-compose sidecar on the local Docker network in Harbor.
        domains = [self._sim_tunnel_host] if self._sim_tunnel_host else []
        return {"outbound_domain_allowlist": domains, "outbound_cidr_allowlist": []}
