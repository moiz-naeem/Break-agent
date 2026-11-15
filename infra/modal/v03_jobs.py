"""Heavy v0.3 jobs on Modal, so the laptop only edits and commits (Anh, 2026-11-14).

    modal run infra/modal/v03_jobs.py::tests                         # full fast suite, sharded
    modal run infra/modal/v03_jobs.py::tests --paths tests/t07 --shards 8
    modal run infra/modal/v03_jobs.py::export --preset T1 --count 5 --out datasets/t1-dev-v03

The working tree is uploaded as it is (uncommitted edits included); `datasets/` and caches are left out. Each
container has its own copy, so tests that write seed lists never race with each other or with the laptop.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
MARKERS = "not harbor and not model and not gpu"
IGNORE = [".venv", "jobs", "datasets", "results", "**/__pycache__", ".hypothesis", ".pytest_cache", ".ruff_cache"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands("git config --global --add safe.directory '*'")
    .uv_pip_install(
        "httpx>=0.28.1", "matplotlib>=3.11.2", "mcp>=2.3.0", "numpy>=2.4.6", "pydantic>=2.14.3", "pyyaml>=6.0.3",
        "ortools>=9.12", "pytest>=9.2.0", "hypothesis>=6.170.15",
    )
    .add_local_dir(str(ROOT), "/repo", ignore=IGNORE)
)  # fmt: skip
app = modal.App("railway-v03-jobs", image=image)


@app.function(cpu=2.0, memory=4096, timeout=3 * 3600, max_containers=100)
def run_shard(test_ids: list[str]) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", MARKERS, *test_ids]
    proc = subprocess.run(cmd, cwd="/repo", capture_output=True, text=True, env={"PYTHONPATH": "/repo", "PATH": "/usr/local/bin:/usr/bin:/bin"})
    return proc.returncode, proc.stdout[-8000:] + proc.stderr[-2000:]


@app.local_entrypoint()
def tests(paths: str = "tests", shards: int = 48, k: str = ""):
    collect = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "-m", MARKERS]
    collect += paths.split() + (["-k", k] if k else [])
    out = subprocess.run(collect, cwd=ROOT, capture_output=True, text=True, env={"PYTHONPATH": str(ROOT), "SYSTEMROOT": "C:\\Windows", "PATH": ""}).stdout
    ids = [line for line in out.splitlines() if "::" in line]
    if not ids:
        sys.exit(f"no tests collected:\n{out[-2000:]}")
    groups = [ids[i::shards] for i in range(min(shards, len(ids)))]
    failed_lines = []
    passed = failed = 0
    for code, report in run_shard.map(groups):
        for line in report.splitlines():
            if line.startswith(("FAILED", "ERROR")):
                failed_lines.append(line)
        last = [x for x in report.splitlines() if " passed" in x or " failed" in x or " error" in x]
        summary = last[-1] if last else ""
        for part in summary.replace(",", " ").split(" in ")[0].split("  "):
            pass
        import re

        passed += sum(int(n) for n in re.findall(r"(\d+) passed", summary))
        failed += sum(int(n) for n in re.findall(r"(\d+) (?:failed|errors?)", summary))
        if code not in (0, 5) and not failed_lines:
            print(report[-3000:])
    for line in failed_lines:
        print(line)
    print(f"{len(ids)} tests in {len(groups)} shards: {passed} passed, {failed} failed or errored")
    if failed:
        sys.exit(1)


@app.function(cpu=8.0, memory=8192, timeout=4 * 3600)
def export_remote(preset: str, set_name: str, count: int, out_name: str) -> bytes:
    import os

    os.chdir("/repo")
    sys.path.insert(0, "/repo")
    from railway.harbor.export import ExportOptions, export_dataset

    out = Path("/tmp") / out_name
    export_dataset(preset, set_name, count, out, ExportOptions())
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out, arcname=out_name)
    return buf.getvalue()


@app.local_entrypoint()
def export(preset: str = "T1", set_name: str = "dev", count: int = 5, out: str = "datasets/t1-dev-v03"):
    target = ROOT / out
    data = export_remote.remote(preset, set_name, count, target.name)
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(target.parent, filter="data")
    print(f"exported {preset}-{set_name} x{count} -> {target}")
