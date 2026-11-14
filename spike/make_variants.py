"""Regenerate the spike task variants from spike/toy-task (run: uv run python spike/make_variants.py)."""
import shutil
from pathlib import Path

SPIKE = Path(__file__).resolve().parent
BASE = SPIKE / "toy-task"
ALLOWLIST = {"environment": 'network_mode = "allowlist"\nallowed_hosts = ["sim"]\n'}


def make(name: str, sections: dict[str, str] | None = None, drop_compose: bool = False,
         instruction: str | None = None) -> None:
    """Copy the base task, inserting `lines` right under each existing `[section]` header."""
    dst = SPIKE / name
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(BASE, dst)
    toml = (dst / "task.toml").read_text()
    for section, lines in (sections or {}).items():
        header = f"[{section}]\n"
        assert header in toml, f"{header!r} missing from base task.toml"  # else the variant silently loses its setting
        toml = toml.replace(header, header + lines, 1)
    (dst / "task.toml").write_text(toml, newline="\n")
    if drop_compose:
        (dst / "environment" / "docker-compose.yaml").unlink()
    if instruction:
        (dst / "instruction.md").write_text(instruction, newline="\n")


if __name__ == "__main__":
    # Provider and install hosts are added at run time with --allow-environment-host.
    make("toy-task-allowlist", ALLOWLIST)
    make("toy-task-nonroot", {"agent": 'user = "agent"\n'})
    make("toy-task-nosim", drop_compose=True)
    # Diagnostic kept for re-running: does plain HTTP from the agent phase reach sim under the allowlist?
    make("toy-task-allowlist-probe", ALLOWLIST,
         instruction="Run these two shell commands and report their full output verbatim, then stop:\n"
                     "1. curl -sv --max-time 10 http://sim:8000/health\n"
                     "2. curl -sv --max-time 10 -X POST -H 'Content-Type: application/json' "
                     "-H 'Accept: application/json, text/event-stream' "
                     "-d '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}' http://sim:8000/mcp\n")
