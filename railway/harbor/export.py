"""Export accepted scenarios as Harbor tasks (ticket T07 §2.2-§2.7; DESIGN.md §8).

    python -m railway.harbor.export --config T1 --set dev --count 3 --out datasets/t1-dev-mini
        [--fresh N] [--allow-host HOST ...] [--no-default-hosts]
        [--agent-user agent|root] [--agent-timeout SEC] [--probe]

`gen export` (the T04 CLI) should delegate to `main()` here.

Layers:
- `export_scenarios(items, out, opts, meta)` writes tasks for given (scenario, accepted entry) pairs; it
  needs only `railway.sim` and the files it vendors.
- `export_dataset(preset, set, count, out, opts, fresh)` draws the pairs from T06's accepted lists and
  T04's generator (imported lazily).
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import shutil
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from railway.sim.constants import RULES_VERSION, TOOL_BUDGET
from railway.util import canonical_json, content_hash

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../railway
REPO_ROOT = PACKAGE_ROOT.parent
TEMPLATES = Path(__file__).resolve().parent / "templates"

# PRD D18: model-provider hosts allowed by default.
D18_HOSTS = (
    "api.openai.com",
    "api.anthropic.com",
    "openrouter.ai",
    "generativelanguage.googleapis.com",
    "api.together.xyz",
    "api.fireworks.ai",
    "api.groq.com",
    "api.mistral.ai",
    "api.deepseek.com",
)
# docs/spike-report.md "Minimal allowlist": nothing beyond `sim` once OpenCode is pre-installed in the image.
SPIKE_HOSTS: tuple[str, ...] = ()

# Vendored subsets of `railway/` (ticket §2.3): package-level modules + subpackages.
VENDOR_MODULES = ("__init__.py", "util.py")
SIM_PACKAGES = ("sim", "tools", "server")
TESTS_PACKAGES = ("sim", "tools")
SOLUTION_PACKAGES = ("sim", "tools", "ref", "policies")

# Roots of the image requirement closure (uv.lock); uvicorn/starlette arrive through mcp.
REQUIREMENT_ROOTS = ("pydantic", "mcp", "httpx")
# Target platform of both images (python:3.11-slim, CPython on Linux) for uv.lock environment markers.
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
    agent_user: str = "agent"  # spike F-NONROOT == PASS; "root" omits `[agent] user`
    agent_timeout: int = 3600  # provisional; set from the pilot (T09)
    verifier_timeout: int = 600
    probe: bool = False

    def __post_init__(self):
        if self.agent_user not in ("agent", "root"):
            raise ValueError("agent_user must be 'agent' or 'root'")


# ---------------------------------------------------------------- requirements from uv.lock


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v))


def marker_applies(marker: str | None, env: dict[str, str] = TARGET_ENV) -> bool:
    """Evaluate the simple PEP 508 markers uv writes into uv.lock (`and`/`or` of comparisons, no parens)."""
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
    """One lock entry per name; several versions (per resolution marker) are narrowed to the target's."""
    if len(candidates) > 1:
        candidates = [c for c in candidates if any(marker_applies(m) for m in c.get("resolution-markers", []))]
    if len(candidates) != 1:
        raise ValueError("uv.lock: cannot pick one version for the target platform")
    return candidates[0]


def pinned_requirements(lock_path: Path | None = None, roots: tuple[str, ...] = REQUIREMENT_ROOTS) -> list[str]:
    """`name==version` for the transitive closure of `roots` in uv.lock, for the Linux/3.11 images."""
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
    return (TEMPLATES / name).read_text(encoding="utf-8")


def _fill(text: str, **values) -> str:
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    if "{{" in text:
        raise ValueError(f"unfilled placeholder in template: {text[text.index('{{') :][:40]}")
    return text


def render_instruction() -> str:
    return _template("instruction.md").replace("{TOOL_BUDGET}", str(TOOL_BUDGET))


def allowed_hosts(opts: ExportOptions) -> list[str]:
    hosts = ["sim", *SPIKE_HOSTS, *(D18_HOSTS if opts.default_hosts else ()), *opts.allow_hosts]
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


def render_main_dockerfile(pins: list[str]) -> str:
    return _fill(_template("main.Dockerfile"), PINS=" \\\n".join(f"    {p}" for p in pins))


# ---------------------------------------------------------------- writing


def _write(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))  # LF line endings on every OS: these run in Linux containers
    if executable:
        path.chmod(0o755)


def vendor(dest: Path, packages: tuple[str, ...]) -> None:
    """Copy (never symlink) `railway/{__init__,util}.py` and the given subpackages, byte for byte."""
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


def norm_for(scenario: dict, entry: dict) -> dict:
    scenario_hash = content_hash(scenario)
    if entry["scenario_hash"] != scenario_hash or entry["scenario_id"] != scenario["scenario_id"]:
        raise ValueError(f"stale accepted entry for {entry['scenario_id']}: scenario_hash does not match")
    if scenario["rules_version"] != RULES_VERSION:
        raise ValueError(f"{entry['scenario_id']}: rules_version {scenario['rules_version']} != {RULES_VERSION}")
    if not entry["c_greedy"] > entry["c_best"]:
        raise ValueError(f"{entry['scenario_id']}: needs c_greedy > c_best")
    if entry["c_best_kind"] not in ("proven", "best-known"):
        raise ValueError(f"{entry['scenario_id']}: c_best_kind {entry['c_best_kind']!r}")
    if not isinstance(entry.get("plan"), list) or not entry["plan"]:
        raise ValueError(f"{entry['scenario_id']}: needs the C_best plan the oracle replays")
    return {
        "scenario_id": scenario["scenario_id"],
        "scenario_hash": scenario_hash,
        "rules_version": RULES_VERSION,
        "c_greedy": entry["c_greedy"],
        "c_best": entry["c_best"],
        "c_best_kind": entry["c_best_kind"],
        "c_ref": entry["c_ref"],  # the scripted reference: a baseline row (v0.3 §2.4), not an anchor
    }


def export_task(
    scenario: dict,
    entry: dict,
    task_dir: Path,
    opts: ExportOptions,
    *,
    probe: bool = False,
    pins: list[str] | None = None,
) -> None:
    """Write one task directory (ticket §2.3). A fresh token per task, per export; never printed."""
    norm = norm_for(scenario, entry)
    pins = pins if pins is not None else pinned_requirements()
    token = secrets.token_hex(32)
    scenario_json = canonical_json(scenario)
    if task_dir.exists():
        shutil.rmtree(task_dir)

    _write(task_dir / "instruction.md", render_instruction())
    _write(task_dir / "task.toml", render_task_toml(opts))

    env = task_dir / "environment"
    _write(env / "Dockerfile", render_main_dockerfile(pins))
    _write(env / "docker-compose.yaml", _template("docker-compose.yaml"))
    _write(env / "sim" / "Dockerfile", _template("sim.Dockerfile"))
    _write(env / "sim" / "requirements.txt", "\n".join(pins) + "\n")
    _write(env / "sim" / "scenario.json", scenario_json)
    _write(env / "sim" / "token", token)
    vendor(env / "sim" / "railway", SIM_PACKAGES)

    tests = task_dir / "tests"
    _write(tests / "test.sh", _fill(_template("test.sh"), VERIFY_EXTRA=" --probe" if probe else ""), executable=True)
    shutil.copyfile(Path(__file__).resolve().parent / "verify.py", tests / "verify.py")
    _write(tests / "scenario.json", scenario_json)
    _write(tests / "norm.json", canonical_json(norm))
    _write(tests / "token", token)
    vendor(tests / "railway", TESTS_PACKAGES)

    solution = task_dir / "solution"
    _write(solution / "solve.sh", _template("solve_probe.sh" if probe else "solve.sh"), executable=True)
    _write(solution / "plan.json", canonical_json(entry["plan"]))  # the oracle replays it (v0.3 §2.7)
    vendor(solution / "railway", SOLUTION_PACKAGES)


def export_scenarios(items: list[tuple[dict, dict]], out: Path, opts: ExportOptions, *, meta: dict) -> dict:
    """Export (scenario, accepted entry) pairs into `out/<scenario_id>/`, plus manifest.json and README.md."""
    out = Path(out)
    tasks = []
    for scenario, entry in items:
        norm_for(scenario, entry)  # validate every entry before writing anything
    out.mkdir(parents=True, exist_ok=True)
    pins = pinned_requirements()
    for scenario, entry in items:
        export_task(scenario, entry, out / scenario["scenario_id"], opts, pins=pins)
        tasks.append(_manifest_row(scenario, entry, scenario["scenario_id"]))
    if opts.probe and items:
        scenario, entry = items[0]
        name = f"{scenario['scenario_id']}-probe"
        export_task(scenario, entry, out / name, opts, probe=True, pins=pins)
        tasks.append({**_manifest_row(scenario, entry, name), "probe": True})
    manifest = {"rules_version": RULES_VERSION, **meta, "count": len(items), "tasks": tasks}
    _write(out / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write(
        out / "README.md",
        _fill(
            _template("README.md"),
            NAME=out.name,
            COUNT=len(items),
            PRESET=meta.get("preset", "?"),
            SET=meta.get("set", "?"),
            RULES_VERSION=RULES_VERSION,
            DATASET=f"datasets/{out.name}",
        ),
    )
    return manifest


def _manifest_row(scenario: dict, entry: dict, task: str) -> dict:
    return {
        "task": task,
        "scenario_id": scenario["scenario_id"],
        "scenario_hash": content_hash(scenario),
        "seed": scenario["seed"],
        "c_greedy": entry["c_greedy"],
        "c_best": entry["c_best"],
        "c_best_kind": entry["c_best_kind"],
        "c_ref": entry["c_ref"],
    }


# ---------------------------------------------------------------- datasets from T04/T06


def fresh_dir(out: Path) -> Path:
    """`<out>-fresh`, which `.gitignore` covers (`datasets/*-fresh/`)."""
    return out.with_name(out.name + "-fresh")


def _entry_dict(entry) -> dict:
    keys = ("seed", "scenario_id", "scenario_hash", "c_greedy", "c_ref", "c_best", "c_best_kind")
    return {k: entry[k] if isinstance(entry, dict) else getattr(entry, k) for k in keys}


def _scenario_dict(scenario) -> dict:
    return scenario.model_dump(mode="json") if hasattr(scenario, "model_dump") else dict(scenario)


def accepted_items(preset: str, set_name: str, count: int) -> list[tuple[dict, dict]]:
    """T06 accepted list → (regenerated scenario, entry) pairs. Recomputes the list if stale (T06 does)."""
    from railway.gen.generator import generate
    from railway.gen.presets import PRESETS
    from railway.ref.seed_filter import accepted_seeds, load_plan

    items = []
    for raw in accepted_seeds(preset, set_name, count):
        entry = _entry_dict(raw)
        entry["plan"] = load_plan(preset, set_name, entry["scenario_id"])
        items.append((_scenario_dict(generate(PRESETS[preset], entry["seed"])), entry))
    return items


def fresh_items(preset: str, count: int) -> list[tuple[dict, dict]]:
    """N accepted scenarios from the `fresh` range, filtered here so no seeds/ file is written for them."""
    from railway.gen.generator import generate
    from railway.gen.presets import PRESETS
    from railway.gen.seeds import candidate_seeds
    from railway.ref.seed_filter import evaluate_scenario

    items = []
    for seed in candidate_seeds("fresh"):
        if len(items) == count:
            break
        scenario = generate(PRESETS[preset], seed)
        report = evaluate_scenario(scenario)
        report = report if isinstance(report, dict) else vars(report)
        if not report["accepted"]:
            continue
        data = _scenario_dict(scenario)
        entry = {
            "seed": seed,
            "scenario_id": data["scenario_id"],
            "scenario_hash": content_hash(data),
            **{k: report[k] for k in ("c_greedy", "c_ref", "c_best", "c_best_kind", "plan")},
        }
        items.append((data, entry))
    return items


def export_dataset(
    preset: str, set_name: str, count: int, out: Path, opts: ExportOptions, fresh: int | None = None
) -> dict:
    out = Path(out)
    manifest = export_scenarios(
        accepted_items(preset, set_name, count), out, opts, meta={"preset": preset, "set": set_name}
    )
    if fresh:
        fresh_out = fresh_dir(out)
        items = fresh_items(preset, fresh)
        export_scenarios(items, fresh_out, opts, meta={"preset": preset, "set": "fresh"})
        print(f"fresh seeds ({fresh_out}): {' '.join(str(e['seed']) for _, e in items)}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gen export", description="Export accepted scenarios as Harbor tasks.")
    p.add_argument("--config", required=True, help="preset, e.g. T1")
    p.add_argument("--set", dest="set_name", required=True, help="seed set, e.g. dev or heldout")
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fresh", type=int, default=None)
    p.add_argument("--allow-host", action="append", default=[])
    p.add_argument("--no-default-hosts", action="store_true")
    p.add_argument("--agent-user", choices=("agent", "root"), default="agent")
    p.add_argument("--agent-timeout", type=int, default=3600)
    p.add_argument("--probe", action="store_true", help="add an isolation-probe task (never for judge datasets)")
    a = p.parse_args(argv)
    opts = ExportOptions(
        allow_hosts=tuple(a.allow_host),
        default_hosts=not a.no_default_hosts,
        agent_user=a.agent_user,
        agent_timeout=a.agent_timeout,
        probe=a.probe,
    )
    manifest = export_dataset(a.config, a.set_name, a.count, Path(a.out), opts, fresh=a.fresh)
    print(f"exported {len(manifest['tasks'])} task(s) to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
