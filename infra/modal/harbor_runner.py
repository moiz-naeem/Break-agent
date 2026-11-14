"""Run Harbor 0.26 railway network tasks on Modal and persist complete Harbor `<job>/<trial>/` logs.

Neither Docker nor `harbor` is needed on the laptop:
- An orchestrator Modal Function (`harbor==0.26.0` + `modal`) mounts the `railway-harbor-logs` Modal Volume
  at `/jobs`.
- Each trial spawns two isolated Modal Sandboxes via `ModalSidecarEnvironment` (`_harbor_sidecar_env.py`):
  1) `sim` Sandbox running `python -m railway.net.server` on encrypted tunnel port 8000 (`timetable.json` + `token`),
  2) `main` Sandbox running the agent (`nop`, `oracle`, or `opencode-ai@1.19.5` with direct MCP tools) with zero
     `timetable.json`, zero `token`, and zero simulator source, followed by Harbor's verifier (`tests/verify.py`).
- Finished `<job>/<trial>/` bundles (`config.json`, `lock.json`, `result.json`, `trial.log`,
  `verifier/{reward.txt,details.json,action_log.json,events.jsonl,test-stdout.txt}`, and `agent/...`) are
  committed to `railway-harbor-logs` and downloaded locally into `jobs/harbor/<job>/`.

Usage (with `PYTHONIOENCODING=utf-8`):
    modal run infra/modal/harbor_runner.py::smoke --seed 1 --shift-min 180
    modal run infra/modal/harbor_runner.py::run --preset oracle --tier T1 --seeds 1 --job-name t1-oracle
    modal run infra/modal/harbor_runner.py::run --preset fable-high --task-path datasets/t1-net-dev
    modal run infra/modal/harbor_runner.py::download --job-name t1-oracle --out jobs/harbor
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import modal

# Inside Modal containers this file is mounted as `/root/harbor_runner.py`; the repo lives at `/repo`.
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/repo")
DEFAULT_ENV_PATHS = (
    ROOT / ".env",
    ROOT.parent / "Afterquery" / ".env",
)

APP_NAME = "railway-harbor-runner"
VOLUME_NAME = "railway-harbor-logs"
OPENCODE_VERSION = "1.19.5"
SIDECAR_ENV_IMPORT = "infra.modal._harbor_sidecar_env:ModalSidecarEnvironment"

QWEN_THINK_BASE_URL = "https://hoaianh517--railway-qwen-think.us-east.modal.direct/v1"
VERDA_BASE_URL = "https://containers.datacrunch.io/nemotron-3-5-super-vl-b300/v1"

# OpenCode 1.19.5 provider registrations for all 4 v0.3 benchmark models (decisions.md #8):
# - Exposes the 14 `railway_*` MCP tools directly as native JSON tool calls (no Code Mode JS wrapper).
# - Registers `claude-fable-5-2` (adaptive thinking + effort) and `gemini-4-carbon` (thinkingLevel),
#   plus custom OpenAI-compatible vLLM endpoints for Modal `qwen38` and Verda `nemotron-3.5-super-vl`.
CUSTOM_PROVIDERS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "models": {
            "claude-fable-5-2": {
                "name": "Claude Fable 5.2",
                "family": "claude-fable",
                "tool_call": True,
                "reasoning": True,
                "temperature": False,
                "limit": {"context": 1000000, "output": 128000},
                "cost": {"input": 8, "output": 40, "cache": {"read": 0.2, "write": 10}},
                "variants": {
                    "low": {"thinking": {"type": "adaptive", "display": "summarized"}, "effort": "low"},
                    "medium": {"thinking": {"type": "adaptive", "display": "summarized"}, "effort": "medium"},
                    "high": {"thinking": {"type": "adaptive", "display": "summarized"}, "effort": "high"},
                    "xhigh": {"thinking": {"type": "adaptive", "display": "summarized"}, "effort": "xhigh"},
                    "max": {"thinking": {"type": "adaptive", "display": "summarized"}, "effort": "max"},
                },
            },
        },
    },
    "google": {
        "models": {
            "gemini-4-carbon": {
                "name": "Gemini 4 Carbon",
                "family": "gemini",
                "tool_call": True,
                "reasoning": True,
                "temperature": True,
                "limit": {"context": 1048576, "output": 32768},
                "cost": {"input": 2, "output": 10, "cache": {"read": 0.1, "write": 0}},
                "variants": {
                    "low": {"thinkingConfig": {"includeThoughts": True, "thinkingLevel": "low"}},
                    "medium": {"thinkingConfig": {"includeThoughts": True, "thinkingLevel": "medium"}},
                    "high": {"thinkingConfig": {"includeThoughts": True, "thinkingLevel": "high"}},
                },
            },
        },
    },
    "modal": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Modal vLLM",
        "options": {
            "baseURL": QWEN_THINK_BASE_URL,
            "apiKey": "{env:VLLM_API_KEY}",
        },
        "models": {
            "qwen38": {
                "name": "Qwen 3.8 27B (thinking)",
                "tool_call": True,
                "reasoning": True,
                "limit": {"context": 131072, "output": 32768},
            },
        },
    },
    "verda": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Verda Nemotron 3.5 Super VL",
        "options": {
            "baseURL": VERDA_BASE_URL,
            "apiKey": "{env:VERDA_INFERENCE_KEY}",
        },
        "models": {
            "nemotron-3.5-super-vl": {
                "name": "Nemotron 3.5 Super VL",
                "tool_call": True,
                "reasoning": True,
                "limit": {"context": 1048576, "output": 32768},
            },
        },
    },
}


@dataclass(frozen=True)
class RunnerPreset:
    agent: str
    model: str | None = None
    effort: str | None = None
    api_key_env: str | None = None
    warm_url: str | None = None


MODEL_PRESETS: dict[str, RunnerPreset] = {
    "nop": RunnerPreset(agent="nop"),
    "oracle": RunnerPreset(agent="oracle"),
    "fable-high": RunnerPreset(
        agent="opencode",
        model="anthropic/claude-fable-5-2",
        effort="high",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "qwen38-think": RunnerPreset(
        agent="opencode",
        model="modal/qwen38",
        effort=None,
        api_key_env="VLLM_API_KEY",
        warm_url=QWEN_THINK_BASE_URL,
    ),
    "gemini-carbon-high": RunnerPreset(
        agent="opencode",
        model="google/gemini-4-carbon",
        effort="high",
        api_key_env="GOOGLE_GENERATIVE_AI_API_KEY",
    ),
    "nemotron-super": RunnerPreset(
        agent="opencode",
        model="verda/nemotron-3.5-super-vl",
        effort=None,
        api_key_env="VERDA_INFERENCE_KEY",
    ),
}

PROVIDER_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_GENERATIVE_AI_API_KEY",
    "gemini": "GOOGLE_GENERATIVE_AI_API_KEY",
    "modal": "VLLM_API_KEY",
    "verda": "VERDA_INFERENCE_KEY",
}


def build_opencode_config(mcp_url: str | None = None) -> dict[str, Any]:
    """Build the OpenCode 1.19.5 configuration overlay for Harbor (`railway_*` tools only)."""
    cfg: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "plugin": [],
        "autoupdate": False,
        "share": "disabled",
        "provider": CUSTOM_PROVIDERS,
        # Deny built-in file/shell/web/task tools so the agent interacts solely with `railway` MCP tools.
        "permission": {"*": "deny", "railway_*": "allow"},
    }
    if mcp_url is not None:
        cfg["mcp"] = {"railway": {"type": "remote", "url": mcp_url, "oauth": False}}
    return cfg


def build_agent_config(
    *,
    preset: str | None = None,
    agent: str = "opencode",
    model: str | None = None,
    effort: str | None = None,
    api_key_env: str | None = None,
) -> tuple[dict[str, Any], str | None, str | None]:
    """Return `(harbor_agent_dict, required_api_key_env_name, warm_url)`."""
    warm_url: str | None = None
    if preset:
        if preset not in MODEL_PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {sorted(MODEL_PRESETS)}")
        spec = MODEL_PRESETS[preset]
        agent, model, effort, api_key_env, warm_url = (
            spec.agent,
            spec.model,
            spec.effort,
            spec.api_key_env,
            spec.warm_url,
        )

    if agent in ("nop", "oracle"):
        return {"name": agent}, None, None

    if not model or "/" not in model:
        raise ValueError(f"agent {agent!r} requires --model <provider/model>")
    provider = model.split("/", 1)[0]
    key_name = api_key_env or PROVIDER_KEY_ENV.get(provider)
    kwargs: dict[str, Any] = {}
    if agent == "opencode":
        kwargs["opencode_config"] = build_opencode_config()
        if effort:
            kwargs["reasoning_effort"] = effort
    elif effort:
        kwargs["reasoning_effort"] = effort

    agent_cfg: dict[str, Any] = {
        "name": agent,
        "model_name": model,
        "kwargs": kwargs,
    }
    if key_name:
        agent_cfg["model_api_key_name"] = key_name
    if provider == "modal" and warm_url is None:
        warm_url = QWEN_THINK_BASE_URL
    return agent_cfg, key_name, warm_url


def load_secret_env(key_names: list[str], env_paths: tuple[Path, ...] = DEFAULT_ENV_PATHS) -> dict[str, str]:
    """Read ONLY the requested secret variable names from `os.environ` or `.env` (never logged/printed)."""
    if not key_names:
        return {}
    pool: dict[str, str] = {}
    for env_path in env_paths:
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line.strip())
                if m:
                    pool.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))
    for k, v in os.environ.items():
        if v:
            pool[k] = v
    if "GEMINI_API_KEY" in pool and "GOOGLE_GENERATIVE_AI_API_KEY" not in pool:
        pool["GOOGLE_GENERATIVE_AI_API_KEY"] = pool["GEMINI_API_KEY"]
    if "CLAUDE_API_KEY" in pool and "ANTHROPIC_API_KEY" not in pool:
        pool["ANTHROPIC_API_KEY"] = pool["CLAUDE_API_KEY"]

    selected: dict[str, str] = {}
    for name in key_names:
        val = pool.get(name)
        if not val:
            raise RuntimeError(f"required API key variable {name} is not set in environment or .env")
        selected[name] = val
    return selected


def warm_modal_if_needed(url: str | None, timeout_s: float = 900.0) -> None:
    """Wake scale-to-zero Modal vLLM endpoints before starting Harbor agent timers."""
    if not url or ".modal.direct" not in url:
        return
    health = re.sub(r"/v1/?$", "", url.rstrip("/")) + "/health"
    t0 = time.monotonic()
    announced = False
    while True:
        try:
            if httpx.get(health, timeout=30.0).status_code == 200:
                if announced:
                    print(f"WARM {health} ready after {time.monotonic() - t0:.0f}s", flush=True)
                return
        except httpx.HTTPError:
            pass
        if not announced:
            print(f"WARM waiting for Modal endpoint cold start at {health} ...", flush=True)
            announced = True
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError(f"Modal endpoint {health} not healthy after {timeout_s:.0f}s")
        time.sleep(10.0)


def _pack_dir(src: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(src, arcname=src.name)
    return buf.getvalue()


def _unpack_bytes(archive_bytes: bytes, dest_parent: Path) -> Path:
    dest_parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tf:
        top_names = {m.name.split("/", 1)[0] for m in tf.getmembers() if m.name}
        for name in top_names:
            target = dest_parent / name
            if target.exists():
                shutil.rmtree(target)
        tf.extractall(dest_parent, filter="data")
    if len(top_names) == 1:
        return dest_parent / next(iter(top_names))
    return dest_parent


# ---------------------------------------------------------------- Modal App & Volume


IGNORE = [
    ".git",
    ".venv",
    "jobs",
    "datasets",
    "results",
    "**/__pycache__",
    ".hypothesis",
    ".pytest_cache",
    ".ruff_cache",
]

runner_image = (
    modal.Image.debian_slim(python_version="3.12")
    # `dockerfile-parse` is an optional Harbor dependency needed by `parse_dockerfile_workdir`.
    .pip_install("harbor==0.26.0", "modal>=1.6.0", "numpy>=2.4.6", "dockerfile-parse>=2.0")
    .env({"PYTHONPATH": "/repo", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    .add_local_dir(str(ROOT / "railway"), "/repo/railway", ignore=IGNORE)
    .add_local_dir(str(ROOT / "infra"), "/repo/infra", ignore=IGNORE)
    .add_local_file(str(ROOT / "uv.lock"), "/repo/uv.lock")
)

harbor_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _summarize_job_dir(job_dir: Path) -> dict[str, Any]:
    trials_summary: list[dict[str, Any]] = []
    for res_path in sorted(job_dir.glob("*/result.json")):
        tdir = res_path.parent
        res = json.loads(res_path.read_text(encoding="utf-8"))
        vdir = tdir / "verifier"
        reward_file = vdir / "reward.txt"
        details_file = vdir / "details.json"
        reward = float(reward_file.read_text(encoding="utf-8").strip()) if reward_file.is_file() else None
        details = json.loads(details_file.read_text(encoding="utf-8")) if details_file.is_file() else None
        trials_summary.append(
            {
                "trial_name": tdir.name,
                "task_name": res.get("task_name"),
                "agent": (res.get("agent_info") or {}).get("name"),
                "model": ((res.get("agent_info") or {}).get("model_info") or {}).get("name"),
                "reward": reward,
                "cost_total": details.get("cost_total") if details else None,
                "tool_calls": details.get("tool_calls") if details else None,
                "exception_info": res.get("exception_info"),
                "files": sorted(p.relative_to(tdir).as_posix() for p in tdir.rglob("*") if p.is_file()),
                "has_agent_dir": (tdir / "agent").is_dir(),
            }
        )
    return {
        "job_name": job_dir.name,
        "job_result_exists": (job_dir / "result.json").is_file(),
        "trials": trials_summary,
    }


@app.function(
    image=runner_image,
    volumes={"/jobs": harbor_volume},
    cpu=2.0,
    memory=4096,
    timeout=4 * 3600,
)
def execute_harbor_job(
    task_archive: bytes | None,
    export_spec: dict[str, Any] | None,
    job_name: str,
    agent_cfg: dict[str, Any],
    n_concurrent: int = 4,
    n_attempts: int = 1,
    agent_timeout_sec: int | None = None,
    secret_Holder: modal.Secret | None = None,  # unused placeholder; secrets injected via Sandbox/Function
) -> tuple[dict[str, Any], bytes]:
    del secret_Holder
    with tempfile.TemporaryDirectory(prefix="harbor-task-") as tmp_str:
        tmp = Path(tmp_str)
        if task_archive is not None:
            staged = _unpack_bytes(task_archive, tmp / "staged")
        elif export_spec is not None:
            from railway.harbor.net_export import ExportOptions, export_timetables, generate_timetable, scenario_id_for

            tier = export_spec.get("tier", "T1")
            seeds = export_spec.get("seeds", [1])
            shift_min = int(export_spec.get("shift_min", 180))
            probe = bool(export_spec.get("probe", False))
            items = [(scenario_id_for(tt := generate_timetable(tier, int(s), shift_min)), tt) for s in seeds]
            staged = tmp / f"{tier.lower()}-modal-export"
            export_timetables(items, staged, ExportOptions(probe=probe), meta={"tier": tier, "set": "modal"})
        else:
            raise ValueError("either task_archive or export_spec must be provided")

        job_dir = Path("/jobs") / job_name
        if job_dir.exists():
            shutil.rmtree(job_dir)

        if agent_timeout_sec is not None:
            agent_cfg = {**agent_cfg, "override_timeout_sec": float(agent_timeout_sec)}

        is_single_task = (staged / "task.toml").is_file()
        job_config: dict[str, Any] = {
            "job_name": job_name,
            "jobs_dir": "/jobs",
            "n_attempts": n_attempts,
            "n_concurrent_trials": n_concurrent,
            "environment": {
                "import_path": SIDECAR_ENV_IMPORT,
                "kwargs": {"app_name": APP_NAME},
            },
            "agents": [agent_cfg],
        }
        if is_single_task:
            job_config["tasks"] = [{"path": str(staged)}]
        else:
            job_config["datasets"] = [{"path": str(staged)}]

        cfg_path = tmp / "job_config.json"
        cfg_path.write_text(json.dumps(job_config, indent=2), encoding="utf-8")

        env = {**os.environ, "PYTHONPATH": "/repo", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.run(
            ["harbor", "run", "-y", "-q", "-c", str(cfg_path)],
            capture_output=True,
            text=True,
            env=env,
        )
        harbor_volume.commit()

        if not job_dir.is_dir():
            tail = (proc.stdout + "\n" + proc.stderr)[-4000:]
            raise RuntimeError(f"harbor run exited {proc.returncode} without creating {job_dir}:\n{tail}")

        summary = _summarize_job_dir(job_dir)
        summary["harbor_returncode"] = proc.returncode
        if proc.returncode != 0:
            summary["harbor_output_tail"] = (proc.stdout + "\n" + proc.stderr)[-3000:]
        packed_job = _pack_dir(job_dir)
        return summary, packed_job


@app.function(image=runner_image, volumes={"/jobs": harbor_volume}, timeout=600)
def list_volume_jobs() -> list[str]:
    harbor_volume.reload()
    root = Path("/jobs")
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


@app.function(image=runner_image, volumes={"/jobs": harbor_volume}, timeout=1200)
def fetch_volume_job(job_name: str) -> bytes:
    harbor_volume.reload()
    job_dir = Path("/jobs") / job_name
    if not job_dir.is_dir():
        raise FileNotFoundError(f"job {job_name!r} not found on volume {VOLUME_NAME}")
    return _pack_dir(job_dir)


def run_job_on_modal(
    *,
    job_name: str,
    preset: str | None = None,
    agent: str = "opencode",
    model: str | None = None,
    effort: str | None = None,
    api_key_env: str | None = None,
    task_path: Path | None = None,
    tier: str = "T1",
    seeds: list[int] | None = None,
    shift_min: int = 180,
    probe: bool = False,
    n_concurrent: int = 4,
    n_attempts: int = 1,
    agent_timeout_sec: int | None = None,
    download_dir: Path | None = None,
) -> tuple[dict[str, Any], Path | None]:
    """Execute a Harbor job on Modal, commit to `railway-harbor-logs`, and optionally download locally."""
    agent_cfg, needed_key, warm_url = build_agent_config(
        preset=preset,
        agent=agent,
        model=model,
        effort=effort,
        api_key_env=api_key_env,
    )
    secrets_dict = load_secret_env([needed_key] if needed_key else [])
    warm_modal_if_needed(warm_url)

    task_archive = _pack_dir(task_path.resolve()) if task_path is not None else None
    export_spec = (
        None
        if task_path is not None
        else {
            "tier": tier,
            "seeds": seeds or [1],
            "shift_min": shift_min,
            "probe": probe,
        }
    )

    fn = execute_harbor_job.with_options(
        secrets=[modal.Secret.from_dict(secrets_dict)] if secrets_dict else [],
    )
    summary, archive_bytes = fn.remote(
        task_archive=task_archive,
        export_spec=export_spec,
        job_name=job_name,
        agent_cfg=agent_cfg,
        n_concurrent=n_concurrent,
        n_attempts=n_attempts,
        agent_timeout_sec=agent_timeout_sec,
    )
    local_job_dir: Path | None = None
    if download_dir is not None:
        local_job_dir = _unpack_bytes(archive_bytes, download_dir)
    return summary, local_job_dir


# ---------------------------------------------------------------- Modal CLI entrypoints


@app.local_entrypoint()
def run(
    preset: str = "",
    agent: str = "opencode",
    model: str = "",
    effort: str = "",
    api_key_env: str = "",
    task_path: str = "",
    tier: str = "T1",
    seeds: str = "1",
    shift_min: int = 180,
    probe: bool = False,
    job_name: str = "",
    n_concurrent: int = 4,
    n_attempts: int = 1,
    agent_timeout_sec: int = 0,
    download_dir: str = "jobs/harbor",
) -> None:
    resolved_job = job_name or f"harbor-{preset or agent}-{int(time.time())}"
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    dl_path = (ROOT / download_dir) if download_dir else None
    summary, local_dir = run_job_on_modal(
        job_name=resolved_job,
        preset=preset or None,
        agent=agent,
        model=model or None,
        effort=effort or None,
        api_key_env=api_key_env or None,
        task_path=Path(task_path) if task_path else None,
        tier=tier,
        seeds=seed_list,
        shift_min=shift_min,
        probe=probe,
        n_concurrent=n_concurrent,
        n_attempts=n_attempts,
        agent_timeout_sec=agent_timeout_sec or None,
        download_dir=dl_path,
    )
    print(json.dumps(summary, indent=2))
    if local_dir is not None:
        print(f"Downloaded Harbor job logs -> {local_dir}")


@app.local_entrypoint()
def list_jobs() -> None:
    for name in list_volume_jobs.remote():
        print(name)


@app.local_entrypoint()
def download(job_name: str, out: str = "jobs/harbor") -> None:
    archive = fetch_volume_job.remote(job_name)
    dest = _unpack_bytes(archive, ROOT / out)
    print(f"Downloaded {job_name} -> {dest}")


@app.local_entrypoint()
def smoke(seed: int = 1, shift_min: int = 180, download_dir: str = "jobs/harbor") -> None:
    """Export 1 T1 net task, run `nop` and `oracle` on Modal, and verify `<job>/<trial>/` on volume & disk."""
    from railway.harbor.net_export import ExportOptions, export_timetables, generate_timetable, scenario_id_for

    with tempfile.TemporaryDirectory(prefix="t1-smoke-ds-") as tmp_ds:
        ds_dir = Path(tmp_ds) / "t1-net-smoke"
        tt = generate_timetable("T1", seed, shift_min)
        export_timetables([(scenario_id_for(tt), tt)], ds_dir, ExportOptions(), meta={"tier": "T1", "set": "smoke"})

        dl_root = ROOT / download_dir
        stamp = int(time.time())
        rewards: dict[str, float] = {}
        for agent_name in ("nop", "oracle"):
            jname = f"smoke-t1-s{seed}-{agent_name}-{stamp}"
            summary, local_dir = run_job_on_modal(
                job_name=jname,
                preset=agent_name,
                task_path=ds_dir,
                n_concurrent=1,
                download_dir=dl_root,
            )
            print(f"\n=== {agent_name.upper()} ({jname}) ===")
            print(json.dumps(summary, indent=2))
            assert local_dir is not None and (local_dir / "result.json").is_file()
            assert len(summary["trials"]) == 1
            trial_info = summary["trials"][0]
            assert trial_info["exception_info"] is None, trial_info["exception_info"]
            assert trial_info["reward"] is not None
            tdir = local_dir / trial_info["trial_name"]
            for rel in (
                "config.json",
                "result.json",
                "verifier/reward.txt",
                "verifier/details.json",
                "verifier/action_log.json",
                "verifier/events.jsonl",
                "verifier/test-stdout.txt",
            ):
                assert (tdir / rel).is_file(), f"missing {rel} in {tdir}"
            assert (tdir / "agent").is_dir(), f"missing agent/ in {tdir}"
            assert (tdir / "verifier/events.jsonl").read_text(encoding="utf-8").strip(), "empty events.jsonl"
            rewards[agent_name] = trial_info["reward"]

        # Doing nothing must be penalised and the booked-order oracle must beat it.
        assert rewards["nop"] < 0, rewards
        assert rewards["oracle"] > rewards["nop"], rewards
        print(f"\nRewards: {rewards}")

        vol_jobs = list_volume_jobs.remote()
        print(f"\nVerified jobs on Modal Volume '{VOLUME_NAME}': {vol_jobs[-4:]}")
