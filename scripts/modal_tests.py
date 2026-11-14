"""Run the pytest suite on Modal, sharded across CPU containers (the full suite takes ~48 min on one machine).

Test ids are collected locally and dealt round-robin into shards; each shard runs in its own container. The image
is built from pip installs of the project's dependencies (pyproject.toml plus the dev group and ortools), so the
only thing needed locally is a Python with `modal` and `pytest` and a `modal setup` once; uv is not.

    modal run scripts/modal_tests.py
    modal run scripts/modal_tests.py --paths tests/net --shards 4
    modal run scripts/modal_tests.py --paths "tests/net tests/t06" -k engine

The summary at the end gives the totals over every shard and each FAILED / ERROR line pytest reported.
"""

import re
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
MARKERS = "not harbor and not model and not gpu"
# .git is left out: in a worktree it is a pointer to a Windows path. The container makes its own empty repo,
# which is all the tests that ask git about .gitignore need.
IGNORE = [".git", ".venv", "jobs", "results", "**/__pycache__", ".hypothesis", ".pytest_cache", ".ruff_cache"]
PACKAGES = [  # pyproject.toml's dependencies and dev group, pinned to what the local venv runs, plus ortools
    "httpx==0.28.1",
    "matplotlib==3.11.2",
    "mcp==2.3.0",
    "numpy==2.5.4",
    "pydantic==2.14.3",
    "pyyaml==6.0.3",
    "ortools==9.15.6755",
    "pytest==9.2.0",
    "hypothesis==6.170.15",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands("git config --global --add safe.directory '*'")
    .pip_install(*PACKAGES)
    .env({"PYTHONPATH": "/repo", "PYTHONHASHSEED": "0"})
    .add_local_dir(str(ROOT), "/repo", ignore=IGNORE)
)
app = modal.App("railway-tests", image=image)

COUNT = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)")


def _counts(text: str) -> dict[str, int]:
    """The totals on pytest's last summary line, e.g. '3 failed, 40 passed, 1 error in 12.3s'."""
    lines = [line for line in text.splitlines() if re.search(r" in [\d.]+s", line)]
    out: dict[str, int] = {}
    for number, word in COUNT.findall(lines[-1] if lines else ""):
        out["error" if word.startswith("error") else word] = int(number)
    return out


@app.function(cpu=2.0, timeout=3600)
def run_shard(test_ids: list[str]) -> dict:
    if subprocess.run(["git", "rev-parse"], cwd="/repo", capture_output=True).returncode != 0:
        subprocess.run(["git", "init", "-q"], cwd="/repo", check=True)
    cmd = [sys.executable, "-m", "pytest", "-q", "-rfE", "-p", "no:cacheprovider", "-m", MARKERS, *test_ids]
    proc = subprocess.run(cmd, cwd="/repo", capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    bad = [line for line in text.splitlines() if line.startswith(("FAILED ", "ERROR "))]
    return {"code": proc.returncode, "counts": _counts(proc.stdout), "bad": bad, "tail": text[-4000:]}


@app.local_entrypoint()
def main(paths: str = "tests", shards: int = 24, k: str = ""):
    collect = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "-m", MARKERS]
    collect += paths.split() + (["-k", k] if k else [])
    out = subprocess.run(collect, cwd=ROOT, capture_output=True, text=True).stdout
    ids = [line for line in out.splitlines() if "::" in line]
    if not ids:
        sys.exit(f"no tests collected:\n{out}")
    groups = [ids[i::shards] for i in range(min(shards, len(ids)))]
    totals: dict[str, int] = {}
    bad: list[str] = []
    broken = 0
    for i, shard in enumerate(run_shard.map(groups)):
        for word, n in shard["counts"].items():
            totals[word] = totals.get(word, 0) + n
        bad += shard["bad"]
        print(f"shard {i:2d}: exit {shard['code']}  " + ", ".join(f"{n} {w}" for w, n in shard["counts"].items()))
        if shard["code"] not in (0, 1) or (shard["code"] == 1 and not shard["bad"]):
            broken += 1  # pytest itself fell over (collection error, crash): show what it said
            print(shard["tail"])
    print("\n" + "=" * 100)
    print(f"{len(ids)} tests collected in {len(groups)} shards")
    print("TOTAL: " + ", ".join(f"{totals[w]} {w}" for w in sorted(totals)))
    for line in sorted(set(bad)):
        print(line)
    if broken:
        print(f"{broken} shard(s) ended without a usable pytest report")
    if bad or broken or totals.get("passed", 0) == 0:
        sys.exit(1)
