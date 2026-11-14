"""Run the pytest suite on Modal, sharded across CPU containers (the full suite takes ~48 min on one machine).

Test ids are collected locally and dealt round-robin into shards; each shard runs in its own container on an image
built from this repo's `uv.lock`, the same way as `modal_accept.py`. Needs `modal setup` once.

    uv run --with modal modal run scripts/modal_tests.py
    uv run --with modal modal run scripts/modal_tests.py --paths tests/net --shards 4
"""

import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
MARKERS = "not harbor and not model and not gpu"
IGNORE = [".venv", "jobs", "**/__pycache__", ".hypothesis", ".pytest_cache", ".ruff_cache"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")  # export tests stamp the git commit; .git is uploaded for them
    .run_commands("git config --global --add safe.directory '*'")
    .uv_sync(str(ROOT), groups=["dev"])
    .add_local_dir(str(ROOT), "/repo", ignore=IGNORE)
)
app = modal.App("railway-tests", image=image)


@app.function(cpu=2.0, timeout=3600)
def run_shard(test_ids: list[str]) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", MARKERS, *test_ids]
    proc = subprocess.run(cmd, cwd="/repo", capture_output=True, text=True)
    return proc.returncode, proc.stdout[-6000:] + proc.stderr[-2000:]


@app.local_entrypoint()
def main(paths: str = "tests", shards: int = 24, k: str = ""):
    collect = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "-m", MARKERS]
    collect += paths.split() + (["-k", k] if k else [])
    out = subprocess.run(collect, cwd=ROOT, capture_output=True, text=True).stdout
    ids = [line for line in out.splitlines() if "::" in line]
    if not ids:
        sys.exit(f"no tests collected:\n{out}")
    groups = [ids[i::shards] for i in range(min(shards, len(ids)))]
    failed = 0
    for code, report in run_shard.map(groups):
        lines = report.strip().splitlines()
        print(lines[-1] if lines else f"exit {code}, no output")
        if code != 0:
            failed += 1
            print(report)
    print(f"{len(ids)} tests in {len(groups)} shards; {failed} shard(s) failed")
    if failed:
        sys.exit(1)
