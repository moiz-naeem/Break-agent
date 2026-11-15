"""Harbor 0.26 agent: Factory Droid over ACP, pre-installed in the task's main image (teacher runs).

Harbor's generic `AcpAgent` (`acp:factory-droid`) works, but its install step runs `apt-get`, the uv
installer and an unpinned `pip install agent-client-protocol` inside the trial. That fails behind the
task's network allowlist, and `agent-client-protocol` 1.0 no longer exports `text_block`, which Harbor
0.26's `acp_runner.py` imports. So the teacher image (`net_export.ExportOptions(droid=True)`) bakes in
`droid` and `/opt/harbor-acp-venv` with `agent-client-protocol==0.12.1`, and this subclass skips the
install when both are there.

Keeping Droid on the railway tools: in ACP mode Droid does not apply `--only-tools` (the MCP server is
only attached at `session/new`, so `MCP:railway` is unknown at startup) and still offers its built-in
tools after `--remove-tools`. Two levers instead:
- every railway call goes through an ACP permission request, so the uploaded runner is Harbor's own
  `acp_runner.py` with one change: with `ACP_RUNNER_TOOL_PREFIX` set, a permission request for a tool
  whose title does not start with that prefix (case-insensitive) is refused;
- a short appended system note (`--append-system-prompt`) says to use only the railway tools.
The training data never depends on this: samples come only from the simulator's action log.

The model and reasoning effort go on Droid's command line (`droid exec --output-format acp-daemon -m ...
-r ...`); Harbor's ACP `set_model` resolves to the same id. `FACTORY_API_KEY` comes from the
orchestrator's environment (the Modal secret `factory-key`) and is passed to the agent process only.

    agent config: {"import_path": "infra.modal._harbor_droid_agent:DroidAcpAgent",
                   "model_name": "factory/claude-opus-5-5", "kwargs": {"effort": "high"}}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from harbor.agents.installed.acp import AcpAgent, AcpOptions
from pydantic import Field

DROID_ENTRY_ID = "factory-droid"
ACP_CLIENT_VERSION = "0.12.1"
KEY_ENV = "FACTORY_API_KEY"
# Droid titles MCP calls `railway___order` (0.244 on Windows) or `railway: Order` (1.12.1 in the image).
TOOL_PREFIX = "railway"
TEACHER_NOTE = (
    "Use only the railway MCP tools (names starting with railway___). Do not read files, run commands, "
    "search the web, plan with todo lists or start sub-agents: none of them can help with this task."
)

_PERMISSION_ANCHOR = 'if self._permission_mode == "deny":'
_PERMISSION_PATCH = (
    'if self._permission_mode == "deny" or ('
    'os.environ.get("ACP_RUNNER_TOOL_PREFIX") and not str(getattr(tool_call, "title", "") or "").lower()'
    '.startswith(os.environ["ACP_RUNNER_TOOL_PREFIX"].lower())):'
)


def droid_registry_entry(model: str, effort: str | None) -> dict[str, Any]:
    args = ["exec", "--output-format", "acp-daemon", "-m", model]
    if effort:
        args += ["-r", effort]
    args += ["--append-system-prompt", TEACHER_NOTE]
    return {
        "id": DROID_ENTRY_ID,
        "name": "Factory Droid (pre-installed)",
        "version": "preinstalled",
        "description": "Factory Droid CLI over ACP, baked into the teacher image",
        "distribution": {
            "local": {
                "cmd": "droid",
                "args": args,
                "env": {"DROID_DISABLE_AUTO_UPDATE": "true", "FACTORY_DROID_AUTO_UPDATE_ENABLED": "false"},
            }
        },
    }


def patched_runner_source() -> str:
    import harbor.agents.installed.acp as acp_mod

    src = Path(acp_mod.__file__).with_name("acp_runner.py").read_text(encoding="utf-8")
    if src.count(_PERMISSION_ANCHOR) != 1 or "\nimport os\n" not in src:
        raise RuntimeError("Harbor's acp_runner.py changed; update the railway-only permission patch")
    return src.replace(_PERMISSION_ANCHOR, _PERMISSION_PATCH)


class DroidOptions(AcpOptions):
    effort: str | None = Field(default="high", description="Droid reasoning effort (droid exec -r).")


class DroidAcpAgent(AcpAgent):
    options_model = DroidOptions
    options: DroidOptions

    def __init__(self, *args, effort: str | None = "high", **kwargs):
        model_name = kwargs.get("model_name") or "factory/claude-opus-5-5"
        droid_model = model_name.split("/", 1)[1] if "/" in model_name else model_name
        kwargs.setdefault("registry_entry", droid_registry_entry(droid_model, effort))
        kwargs.setdefault("distribution_preference", "local")
        kwargs.setdefault("auth_policy", "disabled")
        super().__init__(*args, **kwargs)
        self._droid_model = droid_model
        self._droid_effort = effort

    @staticmethod
    def name() -> str:
        return "droid-acp"

    async def install(self, environment) -> None:
        check = (
            "command -v droid >/dev/null 2>&1 && "
            f"{self._RUNNER_VENV_PATH}/bin/python -c 'from acp import text_block, spawn_agent_process' >/dev/null 2>&1"
        )
        result = await environment.exec(command=check, user="root")
        if getattr(result, "return_code", 1) == 0:
            return
        await super().install(environment)

    async def configure(self, environment) -> None:
        await super().configure(environment)
        runner = self.logs_dir / "acp_runner_railway.py"
        runner.write_text(patched_runner_source(), encoding="utf-8")
        await environment.upload_file(source_path=runner, target_path=self._RUNNER_REMOTE_PATH)
        await environment.exec(command=f"chmod +x {self._RUNNER_REMOTE_PATH}", user="root")

    async def exec_as_agent(self, environment, command: str, env: dict[str, str] | None = None, **kwargs):
        env = {**(env or {}), "ACP_RUNNER_TOOL_PREFIX": TOOL_PREFIX}
        key = os.environ.get(KEY_ENV)
        if key:
            env[KEY_ENV] = key
        return await super().exec_as_agent(environment, command=command, env=env, **kwargs)
