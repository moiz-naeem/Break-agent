"""Export v0.3 railway network (`Timetable`) scenarios as Harbor 0.26 tasks (decisions.md #2, #4, #10).

    python -m railway.harbor.net_export --tier T1 --seeds 1,2,3 --shift-min 180 --out datasets/t1-net-dev
    python -m railway.harbor.net_export --timetable-file path/to/timetable.json --out datasets/custom

No greedy or reference anchors are computed or stored: `norm.json` is dropped completely, `reward.txt` is
`-(total cost)`, and `solution/solve.sh` runs a booked-order MCP client (`solve.py`) strictly over HTTP MCP.
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import shutil
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from railway.net.formats import NET_RULES_VERSION
from railway.net.manual import TOOL_BUDGET
from railway.net.model import Timetable
from railway.util import canonical_json, content_hash

HARBOR_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = HARBOR_DIR.parent  # .../railway
REPO_ROOT = PACKAGE_ROOT.parent
NET_TEMPLATES = HARBOR_DIR / "net_templates"

# Model-provider hosts allowed by default (including Modal Qwen3.8 and Verda Nemotron 3.5 Super VL).
DEFAULT_PROVIDER_HOSTS = (
    "api.openai.com",
    "api.anthropic.com",
    "openrouter.ai",
    "generativelanguage.googleapis.com",
    "containers.datacrunch.io",
    "hoaianh517--railway-qwen-think.us-east.modal.direct",
    "hoaianh517--railway-qwen-nothink.us-east.modal.direct",
    "api.together.xyz",
    "api.fireworks.ai",
    "api.groq.com",
    "api.mistral.ai",
    "api.deepseek.com",
)

# Teacher image only (`ExportOptions(droid=True)`): the hosts `droid exec` contacts with FACTORY_API_KEY auth
# (measured through a logging proxy on Modal: api, prem, telemetry), plus app and relay.
FACTORY_HOSTS = ("api.factory.ai", "prem.factory.ai", "app.factory.ai", "relay.factory.ai", "telemetry.factory.ai")
# Harbor 0.26's acp_runner.py imports `acp.text_block`, which agent-client-protocol 1.0 removed.
ACP_CLIENT_PIN = "agent-client-protocol==0.12.1"
# Baked into the teacher image so Harbor's ACP agent needs no install step behind the network allowlist
# (`infra/modal/_harbor_droid_agent.py` skips it). Droid comes from Factory's installer; the runner venv
# is Python 3.12 at the paths Harbor's AcpAgent uses.
DROID_DOCKERFILE_BLOCK = f"""# Teacher runs: Factory Droid over ACP (Harbor AcpAgent paths), installed at build time.
ENV DROID_DISABLE_AUTO_UPDATE=true FACTORY_DROID_AUTO_UPDATE_ENABLED=false
RUN curl -fsSL https://app.factory.ai/cli | sh \\
 && install -m 0755 /root/.local/bin/droid /usr/local/bin/droid && droid --version
RUN pip install --no-cache-dir uv \\
 && UV_PYTHON_INSTALL_DIR=/opt/harbor-acp-python uv python install 3.12 \\
 && UV_PYTHON_INSTALL_DIR=/opt/harbor-acp-python uv venv --python 3.12 /opt/harbor-acp-venv \\
 && uv pip install --python /opt/harbor-acp-venv/bin/python "{ACP_CLIENT_PIN}" \\
 && chmod -R a+rX /opt/harbor-acp-python /opt/harbor-acp-venv \\
 && /opt/harbor-acp-venv/bin/python -c "from acp import text_block, spawn_agent_process"
"""

VENDOR_MODULES = ("__init__.py", "util.py")
SIM_PACKAGES = ("net",)
TESTS_PACKAGES = ("net",)

REQUIREMENT_ROOTS = ("pydantic", "mcp", "httpx")
TARGET_ENV = {
    "sys_platform": "linux",
    "platform_system": "Linux",
    "os_name": "posix",
    "platform_python_implementation": "CPython",
    "implementation_name": "cpython",
    "python_version": "3.11",
    "python_full_version": "3.11.0",
}


@dataclass(frozen=True)
class ExportOptions:
    allow_hosts: tuple[str, ...] = ()
    default_hosts: bool = True
    agent_user: str = "agent"
    agent_timeout: int = 3600
    verifier_timeout: int = 600
    probe: bool = False
    droid: bool = False  # teacher image: Droid + ACP runner baked in, Factory hosts allowed

    def __post_init__(self) -> None:
        if self.agent_user not in ("agent", "root"):
            raise ValueError("agent_user must be 'agent' or 'root'")


# ---------------------------------------------------------------- requirements from uv.lock


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v))


def marker_applies(marker: str | None, env: dict[str, str] = TARGET_ENV) -> bool:
    if not marker:
        return True
    if "(" in marker:
        raise ValueError(f"unsupported marker: {marker}")
    return any(all(_compare(term.strip(), env) for term in clause.split(" and ")) for clause in marker.split(" or "))


def _compare(term: str, env: dict[str, str]) -> bool:
    m = re.fullmatch(r"(\w+)\s*(==|!=|<=|>=|<|>)\s*'([^']*)'", term)
    if not m or m.group(1) not in env:
        raise ValueError(f"unsupported marker term: {term}")
    name, op, value = m.groups()
    left: object = env[name]
    right: object = value
    if name.startswith("python"):
        left, right = _version_tuple(env[name]), _version_tuple(value)
    return {
        "==": left == right,
        "!=": left != right,
        "<": left < right,
        "<=": left <= right,
        ">": left > right,
        ">=": left >= right,
    }[op]


def _pick(candidates: list[dict]) -> dict:
    if len(candidates) > 1:
        candidates = [c for c in candidates if any(marker_applies(m) for m in c.get("resolution-markers", []))]
    if len(candidates) != 1:
        raise ValueError("uv.lock: cannot pick one version for the target platform")
    return candidates[0]


def pinned_requirements(lock_path: Path | None = None, roots: tuple[str, ...] = REQUIREMENT_ROOTS) -> list[str]:
    """`name==version` for the transitive closure of `roots` in uv.lock, for Linux/3.11 containers."""
    lock = tomllib.loads((lock_path or REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    by_name: dict[str, list[dict]] = {}
    for p in lock["package"]:
        by_name.setdefault(p["name"], []).append(p)
    packages: dict[str, dict] = {}
    seen: set[str] = set()
    stack = [(name, ()) for name in roots]
    while stack:
        name, extras = stack.pop()
        if name not in packages:
            packages[name] = _pick(by_name[name])
        pkg = packages[name]
        deps = list(pkg.get("dependencies", []))
        for extra in extras:
            deps += pkg.get("optional-dependencies", {}).get(extra, [])
        if name in seen and not extras:
            continue
        seen.add(name)
        for dep in deps:
            if marker_applies(dep.get("marker")):
                stack.append((dep["name"], tuple(dep.get("extra", ()))))
    return [f"{name}=={packages[name]['version']}" for name in sorted(seen)]


# ---------------------------------------------------------------- rendering


def _template(name: str) -> str:
    return (NET_TEMPLATES / name).read_text(encoding="utf-8")


def _fill(text: str, **values: object) -> str:
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    if "{{" in text:
        raise ValueError(f"unfilled placeholder in template: {text[text.index('{{') :][:40]}")
    return text


def render_instruction() -> str:
    return _template("instruction.md").replace("{TOOL_BUDGET}", str(TOOL_BUDGET))


def allowed_hosts(opts: ExportOptions) -> list[str]:
    hosts = ["sim", *(DEFAULT_PROVIDER_HOSTS if opts.default_hosts else ()), *opts.allow_hosts]
    if opts.droid:
        hosts += FACTORY_HOSTS
    return list(dict.fromkeys(hosts))


def render_task_toml(opts: ExportOptions) -> str:
    for host in opts.allow_hosts:
        if not re.fullmatch(r"[A-Za-z0-9.\-]+(:\d+)?", host):
            raise ValueError(f"invalid host: {host!r}")
    return _fill(
        _template("task.toml"),
        AGENT_TIMEOUT=int(opts.agent_timeout),
        AGENT_USER='user = "agent"\n' if opts.agent_user == "agent" else "",
        VERIFIER_TIMEOUT=int(opts.verifier_timeout),
        ALLOWED_HOSTS=json.dumps(allowed_hosts(opts)),
    )


def render_main_dockerfile(pins: list[str], *, droid: bool = False) -> str:
    return _fill(
        _template("main.Dockerfile"),
        PINS=" \\\n".join(f"    {p}" for p in pins),
        TEACHER=DROID_DOCKERFILE_BLOCK if droid else "",
    )


# ---------------------------------------------------------------- writing


def _write(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))  # LF line endings for Linux containers
    if executable:
        path.chmod(0o755)


def vendor_net(dest: Path, packages: tuple[str, ...]) -> None:
    """Copy (never symlink) `railway/{__init__,util}.py` and the given subpackages byte for byte."""
    dest.mkdir(parents=True, exist_ok=True)
    for module in VENDOR_MODULES:
        shutil.copyfile(PACKAGE_ROOT / module, dest / module)
    for pkg in packages:
        src = PACKAGE_ROOT / pkg
        for path in sorted(src.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            target = dest / path.relative_to(PACKAGE_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def scenario_id_for(timetable: Timetable) -> str:
    return f"{timetable.tier}-{timetable.seed}"


def _validate_task_slug(slug: str) -> None:
    if not slug or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._\-]*", slug):
        raise ValueError(f"invalid task name / scenario_id: {slug!r}")


def export_task(
    scenario_id: str,
    timetable: Timetable,
    task_dir: Path,
    opts: ExportOptions = ExportOptions(),
    *,
    probe: bool = False,
    pins: list[str] | None = None,
) -> dict:
    """Export one `(scenario_id, timetable)` Harbor 0.26 task directory with fresh bearer token."""
    _validate_task_slug(scenario_id)
    if not isinstance(timetable, Timetable):
        timetable = Timetable.model_validate(timetable)

    pins = pins if pins is not None else pinned_requirements()
    token = secrets.token_hex(32)
    tt_data = timetable.model_dump(mode="json")
    tt_json = canonical_json(tt_data)
    canonical_id = scenario_id_for(timetable)
    tt_hash = content_hash(tt_data)

    task_dir = Path(task_dir)
    if task_dir.exists():
        shutil.rmtree(task_dir)

    _write(task_dir / "instruction.md", render_instruction())
    _write(task_dir / "task.toml", render_task_toml(opts))

    env = task_dir / "environment"
    _write(env / "Dockerfile", render_main_dockerfile(pins, droid=opts.droid))
    _write(env / "docker-compose.yaml", _template("docker-compose.yaml"))
    _write(env / "sim" / "Dockerfile", _template("sim.Dockerfile"))
    _write(env / "sim" / "requirements.txt", "\n".join(pins) + "\n")
    _write(env / "sim" / "timetable.json", tt_json)
    _write(env / "sim" / "token", token)
    vendor_net(env / "sim" / "railway", SIM_PACKAGES)

    tests = task_dir / "tests"
    _write(tests / "test.sh", _fill(_template("test.sh"), VERIFY_EXTRA=" --probe" if probe else ""), executable=True)
    shutil.copyfile(HARBOR_DIR / "net_verify.py", tests / "verify.py")
    _write(tests / "timetable.json", tt_json)
    _write(tests / "token", token)
    vendor_net(tests / "railway", TESTS_PACKAGES)

    solution = task_dir / "solution"
    _write(solution / "solve.sh", _template("solve_probe.sh" if probe else "solve.sh"), executable=True)
    _write(solution / "solve.py", _template("solve.py"))

    row: dict = {
        "task": scenario_id,
        "scenario_id": canonical_id,
        "scenario_hash": tt_hash,
        "tier": timetable.tier,
        "seed": timetable.seed,
        "shift_min": timetable.shift_min,
    }
    if probe:
        row["probe"] = True
    return row


def export_timetables(
    items: Sequence[tuple[str, Timetable] | Timetable],
    out: Path,
    opts: ExportOptions = ExportOptions(),
    *,
    meta: dict | None = None,
) -> dict:
    """Export `(scenario_id, timetable)` pairs (or bare `Timetable`s) into `out/<task>/` + manifest + README."""
    out = Path(out)
    meta = dict(meta or {})
    normalized: list[tuple[str, Timetable]] = []
    seen_slugs: set[str] = set()
    for item in items:
        if isinstance(item, tuple):
            slug, tt = item
            if not isinstance(tt, Timetable):
                tt = Timetable.model_validate(tt)
        else:
            tt = item if isinstance(item, Timetable) else Timetable.model_validate(item)
            slug = scenario_id_for(tt)
        _validate_task_slug(slug)
        if slug in seen_slugs:
            raise ValueError(f"duplicate task name in export: {slug!r}")
        seen_slugs.add(slug)
        normalized.append((slug, tt))

    out.mkdir(parents=True, exist_ok=True)
    pins = pinned_requirements()
    tasks: list[dict] = []
    for slug, tt in normalized:
        tasks.append(export_task(slug, tt, out / slug, opts, pins=pins))
    if opts.probe and normalized:
        slug0, tt0 = normalized[0]
        probe_slug = f"{slug0}-probe"
        tasks.append(export_task(probe_slug, tt0, out / probe_slug, opts, probe=True, pins=pins))

    tier_label = str(meta.get("tier") or (normalized[0][1].tier if normalized else "?"))
    set_label = str(meta.get("set", "custom"))
    manifest = {
        "rules_version": NET_RULES_VERSION,
        **meta,
        "count": len(normalized),
        "tasks": tasks,
    }
    _write(out / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write(
        out / "README.md",
        _fill(
            _template("README.md"),
            NAME=out.name,
            COUNT=len(normalized),
            TIER=tier_label,
            SET=set_label,
            RULES_VERSION=NET_RULES_VERSION,
            DATASET=f"datasets/{out.name}",
        ),
    )
    return manifest


# ---------------------------------------------------------------- generator convenience + CLI


def generate_timetable(tier: str, seed: int, shift_min: int = 180) -> Timetable:
    """Generate a `Timetable` for `tier` (`T1`, plus `T2`/`T3` once template builders are present)."""
    from railway.net import templates as net_templates
    from railway.net.timetable import generate

    tier_u = tier.upper()
    if tier_u == "T1":
        net = net_templates.trunk_and_branches()
        core, fill = net_templates.t1_services()
        return generate(net, core, fill, shift_min, seed, "T1")
    net_fn = getattr(net_templates, f"{tier_u.lower()}_network", None)
    svc_fn = getattr(net_templates, f"{tier_u.lower()}_services", None)
    if callable(net_fn) and callable(svc_fn):
        net = net_fn()
        core, fill = svc_fn()
        return generate(net, core, fill, shift_min, seed, tier_u)
    raise ValueError(f"unsupported tier {tier!r} (no template builder in railway.net.templates)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export v0.3 railway network Timetables as Harbor 0.26 tasks.")
    p.add_argument("--tier", default="T1", help="network tier, e.g. T1")
    p.add_argument("--seeds", default="1", help="comma-separated integer seeds (ignored if --timetable-file is set)")
    p.add_argument("--shift-min", type=int, default=180)
    p.add_argument("--set", dest="set_name", default="dev")
    p.add_argument("--timetable-file", action="append", default=[], help="existing Timetable JSON file(s) to export")
    p.add_argument("--out", required=True, help="output dataset directory")
    p.add_argument("--allow-host", action="append", default=[])
    p.add_argument("--no-default-hosts", action="store_true")
    p.add_argument("--agent-user", choices=("agent", "root"), default="agent")
    p.add_argument("--agent-timeout", type=int, default=3600)
    p.add_argument("--verifier-timeout", type=int, default=600)
    p.add_argument("--probe", action="store_true", help="add an isolation-probe task")
    a = p.parse_args(argv)

    opts = ExportOptions(
        allow_hosts=tuple(a.allow_host),
        default_hosts=not a.no_default_hosts,
        agent_user=a.agent_user,
        agent_timeout=a.agent_timeout,
        verifier_timeout=a.verifier_timeout,
        probe=a.probe,
    )
    items: list[tuple[str, Timetable]] = []
    if a.timetable_file:
        for raw_path in a.timetable_file:
            data = json.loads(Path(raw_path).read_text(encoding="utf-8"))
            data.pop("rules_version", None)
            tt = Timetable.model_validate(data)
            items.append((scenario_id_for(tt), tt))
        meta = {"tier": items[0][1].tier if items else a.tier, "set": a.set_name}
    else:
        seed_list = [int(s.strip()) for s in a.seeds.split(",") if s.strip()]
        for seed in seed_list:
            tt = generate_timetable(a.tier, seed, a.shift_min)
            items.append((scenario_id_for(tt), tt))
        meta = {"tier": a.tier.upper(), "set": a.set_name, "shift_min": a.shift_min}

    manifest = export_timetables(items, Path(a.out), opts, meta=meta)
    print(f"exported {len(manifest['tasks'])} task(s) to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
