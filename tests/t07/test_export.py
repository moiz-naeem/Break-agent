"""T07 §5.1: Harbor task export. Each test runs on the fixture export and, once T04/T06 land, on 2 T1 dev tasks."""

from __future__ import annotations

import json
import re
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

from railway.sim.constants import RULES_VERSION, TOOL_BUDGET
from railway.util import content_hash
from tests.helpers import REPO
from tests.t07.conftest import REAL_EXPORT_MODULES, require_modules, task_dirs

TASK_PATHS = [
    "instruction.md",
    "task.toml",
    "environment/Dockerfile",
    "environment/docker-compose.yaml",
    "environment/sim/Dockerfile",
    "environment/sim/railway/__init__.py",
    "environment/sim/requirements.txt",
    "environment/sim/scenario.json",
    "environment/sim/token",
    "tests/test.sh",
    "tests/verify.py",
    "tests/railway/__init__.py",
    "tests/scenario.json",
    "tests/norm.json",
    "tests/token",
    "solution/solve.sh",
    "solution/plan.json",
    "solution/railway/__init__.py",
]

D18_HOSTS = {
    "api.openai.com",
    "api.anthropic.com",
    "openrouter.ai",
    "generativelanguage.googleapis.com",
    "api.together.xyz",
    "api.fireworks.ai",
    "api.groq.com",
    "api.mistral.ai",
    "api.deepseek.com",
}
# docs/spike-report.md "Minimal allowlist": nothing beyond `sim` once OpenCode is pre-installed.
SPIKE_HOSTS: set[str] = set()

# Ticket §2.4, verbatim; only {TOOL_BUDGET} is interpolated.
INSTRUCTION = """# Railway dispatcher shift

You are the dispatcher of a single-track railway for one shift. You control every train
through the tools of the `railway` MCP server; nothing moves without your authority.

1. Call `manual()` first. It contains every rule, cost formula and tool format.
2. Then read `network()`, `timetable()` and `status()`.
3. Time moves only when you call `advance_to_next_event()`. Give your orders
   (`authorize`, `hold`, `relieve_crew`, ...) and then advance.
4. Goal: the lowest total cost at the end of the shift (passenger and freight lateness,
   crew timeouts, unfinished trains; a gridlock is permanent).

Scoring: your total cost is compared with a simple greedy dispatcher (score 0) and the
best plan known for this shift (score 1). Beating that plan scores above 1.

Ending: the shift ends when you call `end_shift()`, when you use up the tool budget
({TOOL_BUDGET} calls), or when you stop. Stopping without calling end_shift() counts as
end_shift(): the rest of the shift runs with no new authorities and unfinished trains
are charged.

Everything you need is in the tools. There are no files to read.
"""


def manifest(out: Path) -> dict:
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def files_under(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def lock_versions() -> dict[str, str]:
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    return {p["name"]: p["version"] for p in lock["package"]}


@pytest.fixture
def exported(exporter, tmp_path):
    out = tmp_path / "ds"
    exporter(out)
    return out


def test_T07_01_layout(exported):
    m = manifest(exported)
    assert len(m["tasks"]) == 2
    assert m["rules_version"] == RULES_VERSION
    for t in m["tasks"]:
        assert {"scenario_id", "scenario_hash", "c_greedy", "c_best", "c_best_kind", "c_ref"} <= set(t)
    assert (exported / "README.md").is_file()
    for task in task_dirs(exported):
        assert task.name == json.loads((task / "tests" / "scenario.json").read_text(encoding="utf-8"))["scenario_id"]
        for rel in TASK_PATHS:
            assert (task / rel).is_file(), f"{task.name}: missing {rel}"
    readme = (exported / "README.md").read_text(encoding="utf-8")
    # DoD: the "Run the judges' checks" block with the exact harbor commands.
    assert "Run the judges' checks" in readme
    for agent in ("-a oracle", "-a nop", "railway.harbor.agents:StopEarlyAgent"):
        assert agent in readme


def test_T07_02_norm_values(exporter, exported):
    m = manifest(exported)
    for entry, task in zip(m["tasks"], task_dirs(exported), strict=True):
        norm = json.loads((task / "tests" / "norm.json").read_text(encoding="utf-8"))
        assert norm == {
            "scenario_id": entry["scenario_id"],
            "scenario_hash": entry["scenario_hash"],
            "rules_version": RULES_VERSION,
            "c_greedy": entry["c_greedy"],
            "c_best": entry["c_best"],
            "c_best_kind": entry["c_best_kind"],
            "c_ref": entry["c_ref"],
        }
        scenario = json.loads((task / "tests" / "scenario.json").read_text(encoding="utf-8"))
        assert content_hash(scenario) == norm["scenario_hash"]


def test_T07_02_norm_values_recomputed(tmp_path):
    """Real path only: norm.json == accepted-list entry == fresh in-process greedy and reference."""
    require_modules(*REAL_EXPORT_MODULES, "railway.tools.session", "railway.tools.client")
    require_modules("railway.policies.baselines", "railway.ref.reference")
    from railway.harbor.export import ExportOptions, export_dataset
    from railway.policies.baselines import Greedy
    from railway.ref.reference import Reference
    from railway.ref.seed_filter import accepted_seeds
    from railway.sim.model import Scenario
    from railway.tools.client import InProcClient
    from railway.tools.session import Session

    out = tmp_path / "ds"
    export_dataset("T1", "dev", 2, out, ExportOptions())

    def field(obj, key):
        return obj[key] if isinstance(obj, dict) else getattr(obj, key)

    accepted = {field(a, "scenario_id"): a for a in accepted_seeds("T1", "dev", 2)}
    for task in task_dirs(out):
        norm = json.loads((task / "tests" / "norm.json").read_text(encoding="utf-8"))
        entry = accepted[norm["scenario_id"]]
        keys = ("c_greedy", "c_ref", "c_best", "c_best_kind")
        assert tuple(norm[k] for k in keys) == tuple(field(entry, k) for k in keys)
        assert norm["c_best_kind"] in ("proven", "best-known") and norm["c_best"] <= norm["c_ref"]
        scenario = Scenario.model_validate_json((task / "tests" / "scenario.json").read_text(encoding="utf-8"))
        costs = []
        for policy in (Greedy(), Reference()):
            session = Session(scenario)
            policy.run(InProcClient(session))
            costs.append(session.close().cost.total)
        assert costs == [norm["c_greedy"], norm["c_ref"]]


def test_T07_03_main_context_clean(exported):
    m = manifest(exported)
    for entry, task in zip(m["tasks"], task_dirs(exported), strict=True):
        token = (task / "tests" / "token").read_text(encoding="utf-8").strip()
        env = task / "environment"
        for path in files_under(env):
            if (env / "sim") in path.parents:
                continue
            assert not path.name.startswith("scenario"), path
            text = path.read_text(encoding="utf-8", errors="replace")
            for secret in (entry["scenario_id"], entry["scenario_hash"], token):
                assert secret not in text, f"{path} leaks {secret[:8]}..."
        dockerfile = (env / "Dockerfile").read_text(encoding="utf-8")
        for line in dockerfile.splitlines():
            if re.match(r"\s*(COPY|ADD)\b", line, re.I):
                assert not re.search(r"\b(sim|tests|solution|railway|token)\b", line), line


def test_T07_04_sim_and_tests_consistent(exported):
    tokens = []
    for task in task_dirs(exported):
        sim, tests = task / "environment" / "sim", task / "tests"
        assert (sim / "scenario.json").read_bytes() == (tests / "scenario.json").read_bytes()
        assert (sim / "token").read_bytes() == (tests / "token").read_bytes()
        token = (tests / "token").read_text(encoding="utf-8").strip()
        assert re.fullmatch(r"[0-9a-f]{64}", token)
        tokens.append(token)
    assert len(set(tokens)) == len(tokens)


def _subpackages(root: Path) -> set[str]:
    return {p.name for p in root.iterdir() if p.is_dir() and p.name != "__pycache__"}


def _modules(root: Path) -> set[str]:
    return {p.name for p in root.iterdir() if p.is_file()}


@pytest.mark.parametrize(
    "where, packages",
    [
        ("environment/sim/railway", {"sim", "tools", "server"}),
        ("tests/railway", {"sim", "tools"}),
        ("solution/railway", {"sim", "tools", "ref", "policies"}),
    ],
)
def test_T07_05_vendored_subsets(exported, where, packages):
    for task in task_dirs(exported):
        root = task / where
        assert _subpackages(root) == packages
        assert _modules(root) == {"__init__.py", "util.py"}
        vendored = files_under(root)
        assert vendored
        for path in vendored:
            assert "__pycache__" not in path.parts and path.suffix != ".pyc"
            rel = path.relative_to(root)
            assert path.read_bytes() == (REPO / "railway" / rel).read_bytes(), rel
        # And nothing in the repo subset was left out.
        for pkg in packages:
            for src in (REPO / "railway" / pkg).rglob("*"):
                if src.is_file() and "__pycache__" not in src.parts and src.suffix != ".pyc":
                    assert (root / src.relative_to(REPO / "railway")).is_file(), src


def test_T07_06_instruction(exported):
    expected = INSTRUCTION.replace("{TOOL_BUDGET}", str(TOOL_BUDGET))
    for entry, task in zip(manifest(exported)["tasks"], task_dirs(exported), strict=True):
        text = (task / "instruction.md").read_text(encoding="utf-8")
        assert text == expected
        assert len(text) <= 2500
        assert entry["scenario_id"] not in text and entry["scenario_hash"] not in text
        if entry["seed"] >= 1000:  # fixture seeds (0) are too short to test as substrings
            assert str(entry["seed"]) not in text


def _toml(task: Path) -> dict:
    return tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))


def test_T07_07_task_toml(exporter, exported, tmp_path):
    for task in task_dirs(exported):
        raw = (task / "task.toml").read_text(encoding="utf-8")
        assert "@" not in raw
        cfg = _toml(task)
        assert cfg["schema_version"] == "1.4"  # the key Harbor 0.26.0 validated in the spike
        assert cfg["metadata"]["difficulty"] == "hard"
        assert cfg["agent"]["user"] == "agent"  # F-NONROOT == PASS
        assert cfg["agent"]["timeout_sec"] == 3600
        assert cfg["verifier"]["timeout_sec"] == 600
        env = cfg["environment"]
        assert env["network_mode"] == "allowlist"
        assert set(env["allowed_hosts"]) >= {"sim"} | D18_HOSTS | SPIKE_HOSTS
        (mcp,) = env["mcp_servers"]
        assert mcp == {"name": "railway", "transport": "streamable-http", "url": "http://sim:8000/mcp"}

    extra = tmp_path / "extra"
    exporter(extra, allow_hosts=("x.example",))
    for task in task_dirs(extra):
        assert "x.example" in _toml(task)["environment"]["allowed_hosts"]

    bare = tmp_path / "bare"
    exporter(bare, default_hosts=False, agent_user="root", agent_timeout=1800)
    for task in task_dirs(bare):
        cfg = _toml(task)
        assert set(cfg["environment"]["allowed_hosts"]) == {"sim"} | SPIKE_HOSTS
        assert "user" not in cfg["agent"]
        assert cfg["agent"]["timeout_sec"] == 1800


def test_T07_08_requirements_pinned(exported):
    lock = lock_versions()
    for task in task_dirs(exported):
        lines = (task / "environment" / "sim" / "requirements.txt").read_text(encoding="utf-8").splitlines()
        names = set()
        for line in lines:
            m = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.\-+]+)", line)
            assert m, line
            assert lock[m.group(1)] == m.group(2), line
            names.add(m.group(1))
        assert {"pydantic", "pydantic-core", "mcp", "httpx", "uvicorn", "starlette"} <= names
        assert "pywin32" not in names  # Windows-only marker; the images are Linux


def test_T07_08b_main_image_pins_match_sim(exported):
    """The verifier replays in the main image: its pins must equal the sim image's (no version drift)."""
    for task in task_dirs(exported):
        sim_pins = set((task / "environment" / "sim" / "requirements.txt").read_text(encoding="utf-8").split())
        dockerfile = (task / "environment" / "Dockerfile").read_text(encoding="utf-8")
        main_pins = set(re.findall(r"[A-Za-z0-9_.\-]+==[A-Za-z0-9_.\-+]+", dockerfile))
        assert main_pins == sim_pins
        assert "FROM python:3.11-slim" in dockerfile
        assert "opencode-ai@1.19.5" in dockerfile  # spike consequence 1
        assert "useradd -m -u 1000 agent" in dockerfile


def test_T07_08c_compose_and_scripts(exported):
    for task in task_dirs(exported):
        compose = yaml.safe_load((task / "environment" / "docker-compose.yaml").read_text(encoding="utf-8"))
        sim = compose["services"]["sim"]
        assert sim["build"]["context"] == "./sim"
        assert sim["networks"] == ["default"]  # spike: keeps sim out of the egress namespace
        assert "/health" in " ".join(sim["healthcheck"]["test"])
        assert compose["services"]["main"]["depends_on"]["sim"]["condition"] == "service_healthy"
        sim_docker = (task / "environment" / "sim" / "Dockerfile").read_text(encoding="utf-8")
        assert "FROM python:3.11-slim" in sim_docker and "--token-file" in sim_docker
        test_sh = (task / "tests" / "test.sh").read_bytes()
        assert b"\r" not in test_sh and test_sh.startswith(b"#!/usr/bin/env bash\n")
        assert b"set -euo pipefail" in test_sh
        assert b'python3 "$(dirname "$0")/verify.py" --logs-dir /logs/verifier' in test_sh
        solve = (task / "solution" / "solve.sh").read_bytes()
        assert b"\r" not in solve and b"set -euo pipefail" in solve
        assert (
            b'PYTHONPATH="$(dirname "$0")" python3 -m railway.ref.oracle --url http://sim:8000/mcp'
            b' --plan "$(dirname "$0")/plan.json"' in solve
        )
        assert (task / "tests" / "verify.py").read_bytes() == (REPO / "railway" / "harbor" / "verify.py").read_bytes()


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in files_under(root)}


def test_T07_09_export_deterministic(exporter, tmp_path):
    # Same dataset folder name in both runs: the README quotes it in the judges' commands.
    exporter(tmp_path / "a" / "ds")
    exporter(tmp_path / "b" / "ds")
    a, b = _tree(tmp_path / "a" / "ds"), _tree(tmp_path / "b" / "ds")
    assert a.keys() == b.keys()
    differing = {k for k in a if a[k] != b[k]}
    assert differing and all(Path(k).name == "token" for k in differing)


def test_T07_09b_refuses_stale_entry(fixture_items, tmp_path):
    """An accepted-list entry whose hash doesn't match the scenario is stale: export must refuse it."""
    from railway.harbor.export import ExportOptions, export_scenarios

    scenario, entry = fixture_items[0]
    with pytest.raises(ValueError, match="scenario_hash"):
        export_scenarios([(scenario, {**entry, "scenario_hash": "0" * 64})], tmp_path / "x", ExportOptions(), meta={})


def test_T07_10_fresh_dir_is_git_ignored():
    from railway.harbor.export import fresh_dir

    out = fresh_dir(Path("datasets/t1-dev-mini"))
    assert out.as_posix() == "datasets/t1-dev-mini-fresh"
    res = subprocess.run(["git", "check-ignore", "-q", "--no-index", f"{out.as_posix()}/"], cwd=REPO)
    assert res.returncode == 0


def test_T07_10_fresh(tmp_path):
    require_modules(*REAL_EXPORT_MODULES)
    from railway.harbor.export import ExportOptions, export_dataset, fresh_dir

    out = REPO / "datasets" / f"t07-test-{tmp_path.name}"
    fresh = fresh_dir(out)
    try:
        export_dataset("T1", "dev", 1, out, ExportOptions(), fresh=2)
        m = manifest(fresh)
        assert len(m["tasks"]) == 2
        assert all(t["seed"] >= 2_000_000 for t in m["tasks"])
        res = subprocess.run(["git", "check-ignore", "-q", str(fresh.relative_to(REPO).as_posix())], cwd=REPO)
        assert res.returncode == 0
    finally:
        import shutil

        shutil.rmtree(out, ignore_errors=True)
        shutil.rmtree(fresh, ignore_errors=True)


def test_T07_24_probe_task(fixture_items, tmp_path):
    """`--probe` adds one diagnostics task: isolation checks then the oracle in solve.sh; `verify.py --probe`."""
    from railway.harbor.export import ExportOptions, export_scenarios

    out = tmp_path / "ds"
    m = export_scenarios(fixture_items, out, ExportOptions(probe=True), meta={"preset": "fixture", "set": "fixture"})
    probes = [t for t in m["tasks"] if t.get("probe")]
    assert len(probes) == 1 and len(m["tasks"]) == 3
    task = out / probes[0]["task"]
    solve = (task / "solution" / "solve.sh").read_text(encoding="utf-8")
    for check in ("id -u", "https://github.com", "http://sim:8000/health", "Bearer wrong", "scenario*.json"):
        assert check in solve
    assert solve.index("import railway") < solve.index("PYTHONPATH=")  # the import probe runs before PYTHONPATH
    assert "/logs/agent/probe.json" in solve
    assert "--probe" in (task / "tests" / "test.sh").read_text(encoding="utf-8")
    plain = [t for t in m["tasks"] if not t.get("probe")]
    for t in plain:
        assert "--probe" not in (out / t["scenario_id"] / "tests" / "test.sh").read_text(encoding="utf-8")
