"""Run v0.3 loop episodes and test suites on Modal (keeps the local laptop completely free).

    modal run infra/modal/v03_bench.py::pytest_suite --targets "tests/t08 tests/t07/test_verifier.py"
    modal run infra/modal/v03_bench.py::run_probe --only "greedy,reference" --datasets "datasets/t1-dev-v03,datasets/t2-dev-v03"

Ships `git archive HEAD` (`railway`, `seeds`, `datasets`, `infra`, `tests`, `runs`, `pyproject.toml`),
so commit on your branch before invoking `modal run`.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

import modal

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "httpx>=0.28.1",
    "matplotlib>=3.11.2",
    "mcp>=2.3.0",
    "numpy>=2.4.6",
    "pydantic>=2.14.3",
    "pyyaml>=6.0.3",
    "ortools>=9.12",
    "pytest>=8.0",
    "hypothesis>=6.100",
    "anyio>=4.0",
)

if modal.is_local():
    REPO = Path(__file__).resolve().parents[2]
    source = Path(tempfile.mkdtemp(prefix="railway-bench-v03-"))
    archive = subprocess.run(
        ["git", "-C", str(REPO), "archive", "HEAD", "railway", "seeds", "datasets", "infra", "tests", "runs", "pyproject.toml"],
        capture_output=True,
        check=True,
    )
    tarfile.open(fileobj=io.BytesIO(archive.stdout)).extractall(source, filter="data")
    image = image.add_local_dir(source, "/src")

app = modal.App("railway-v03-bench", image=image)


def _setup():
    import sys

    os.chdir("/src")
    if "/src" not in sys.path:
        sys.path.insert(0, "/src")


@app.function(cpu=4, memory=4096, timeout=1800)
def run_pytest_remote(targets: list[str]) -> tuple[int, str]:
    _setup()
    proc = subprocess.run(["python", "-m", "pytest", *targets, "-q"], capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


@app.function(cpu=2, memory=2048, timeout=7200, max_containers=50)
def run_episode_remote(job: dict) -> dict:
    """Run one `(config_id, dataset, task_dir, rep)` episode inside a Modal container and return all artifacts."""
    _setup()
    from infra.bench.loop_probe import CONFIG_BY_ID, Spend, enrich_c_ref, make_client, warm_modal_if_needed
    from railway.eval.run import episodes_from_dataset, reward_from_t07
    from railway.loop.prompts import load_instruction
    from railway.loop.runner import make_session, play_episode

    for k, v in (job.get("secrets") or {}).items():
        os.environ[k] = v

    cfg = CONFIG_BY_ID[job["config_id"]]
    eps = enrich_c_ref(episodes_from_dataset(job["dataset"]))
    ep = next(e for e in eps if e.scenario.scenario_id == job["scenario_id"])
    warm_modal_if_needed(cfg[1])

    tmp = Path(tempfile.mkdtemp(prefix="ep-"))
    spend = Spend(tmp / "spend.json")
    session = make_session(ep.scenario)
    client = make_client(cfg, spend, session=session, ep=ep)
    try:
        rec = play_episode(
            ep.scenario,
            client,
            "windowed",
            session=session,
            instruction=load_instruction(),
            transcript_path=tmp / "transcript.jsonl",
            trajectory_path=tmp / "trajectory.json",
            run=job["run_name"],
            model=cfg[0],
            rep=job["rep"],
            c_greedy=ep.c_greedy,
            c_best=ep.c_best,
            c_ref=ep.c_ref,
            best_source=ep.best_source,
            best_proven=ep.best_proven,
            score_fn=reward_from_t07,
        )
    finally:
        if hasattr(client, "close"):
            client.close()
    return {
        "row": rec.to_row(),
        "transcript": (tmp / "transcript.jsonl").read_text(encoding="utf-8") if (tmp / "transcript.jsonl").exists() else "",
        "trajectory": (tmp / "trajectory.json").read_text(encoding="utf-8") if (tmp / "trajectory.json").exists() else "",
    }


def _local_secrets(key_env: str | None) -> dict[str, str]:
    if not key_env:
        return {}
    env = dict(os.environ)
    for candidate in (REPO / ".env", REPO.parent / "Afterquery" / ".env"):
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line.strip())
                if m:
                    env.setdefault(m.group(1), m.group(2).strip().strip('"'))
            break
    return {key_env: env[key_env]} if key_env in env else {}


@app.local_entrypoint()
def pytest_suite(targets: str = "tests/t08 tests/t07/test_verifier.py"):
    code, out = run_pytest_remote.remote(targets.split())
    print(out)
    if code != 0:
        raise SystemExit(code)


@app.local_entrypoint()
def run_probe(
    only: str = "greedy,reference",
    datasets: str = "datasets/t1-dev-v03,datasets/t2-dev-v03",
    reps: int = 1,
    results_dir: str = "results/v03_dryrun",
):
    import sys

    sys.path.insert(0, str(REPO))
    from infra.bench.loop_probe import CONFIG_BY_ID, _dataset_tag
    from railway.eval.run import _done_pairs, episodes_from_dataset

    out_root = REPO / results_dir
    only_ids = [x.strip() for x in only.split(",") if x.strip()]
    jobs = []
    for ds in [x.strip() for x in datasets.split(",") if x.strip()]:
        ds_path = REPO / ds
        tag = _dataset_tag(ds_path)
        eps = episodes_from_dataset(ds_path)
        for cid in only_ids:
            cfg = CONFIG_BY_ID[cid]
            run_name = f"{tag}-{cid}"
            done = _done_pairs(out_root / f"{run_name}.jsonl")
            sec = _local_secrets(cfg[3])
            for ep in eps:
                for rep in range(reps):
                    if (ep.scenario.scenario_id, rep) not in done:
                        jobs.append(
                            {
                                "config_id": cid,
                                "dataset": ds,
                                "scenario_id": ep.scenario.scenario_id,
                                "rep": rep,
                                "run_name": run_name,
                                "secrets": sec,
                            }
                        )
    print(f"Dispatching {len(jobs)} episodes on Modal...")
    for res in run_episode_remote.map(jobs):
        row = res["row"]
        rname, sid, rep = row["run"], row["scenario_id"], row["rep"]
        out_root.mkdir(parents=True, exist_ok=True)
        with (out_root / f"{rname}.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        if res["transcript"]:
            tp = out_root / rname / "transcripts" / f"{sid}-r{rep}.jsonl"
            tp.parent.mkdir(parents=True, exist_ok=True)
            tp.write_text(res["transcript"], encoding="utf-8")
        if res["trajectory"]:
            jp = out_root / rname / "trajectories" / f"{sid}-r{rep}.json"
            jp.parent.mkdir(parents=True, exist_ok=True)
            jp.write_text(res["trajectory"], encoding="utf-8")
        print(f"DONE {rname} {sid} r{rep}: reward={row.get('reward')} norm={row.get('score_normalized')}")
