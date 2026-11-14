"""`railway-eval`: run manifests, summarize and compare results (T08 §2.7).

    railway-eval run runs/<name>.yaml            # appends rows to results/<name>.jsonl (resumable)
    railway-eval summarize results/<name>.jsonl [--json]
    railway-eval compare results/a.jsonl results/b.jsonl [--json]

Resuming skips (scenario_id, rep) pairs that already have an `ok` row and re-runs error rows.
Rows are appended by one writer thread fed by a queue, so concurrent episodes never interleave.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from railway.eval import stats
from railway.eval.manifest import Manifest, load_manifest
from railway.loop.runner import EpisodeRecord, make_session, play_episode, scenario_meta
from railway.sim.model import Scenario
from railway.util import content_hash


@dataclass
class EpisodeSpec:
    scenario: Scenario
    c_greedy: float
    c_ref: float


@dataclass
class LoopDeps:
    """What the loop harness plugs together. Defaults are the real modules; tests swap in fakes."""

    session_cls: type | None = None  # default: railway.tools.session.Session (T05)
    score_fn: Callable | None = None  # default: railway.harbor.verify.reward_from (T07)
    client_factory: Callable | None = None  # (session, EpisodeSpec) -> chat client; default: ModelClient
    instruction: str | None = None  # default: T07's instruction.md


# Harbor 0.26 error_policy.py: these failures are the agent's own and keep a valid, graded reward (dropping them
# would re-run only the hard episodes and inflate scores). Every other exception is infra: an error row, re-run.
AGENT_FAILURES = frozenset(
    {
        "NonZeroAgentExitCodeError",
        "AgentTimeoutError",
        "ContextWindowExceededError",
        "OutputTokenExceededError",
        "ContextLengthExceededError",
        "OutputLengthExceededError",
        "AgentSafetyRefusalError",
        "ContentPolicyViolationError",
    }
)


def reward_from_t07(c_greedy: float, c_ref: float, c_agent: float):
    """The verifier's own reward function, so loop and Harbor rows score identically."""
    from railway.harbor.verify import reward_from

    return reward_from(c_greedy, c_ref, c_agent)


# --- scenario sources ------------------------------------------------------------------------------


def episodes_from_dataset(dataset: str | Path) -> list[EpisodeSpec]:
    """One episode per task: `<task>/tests/scenario.json` with `c_greedy`/`c_ref` from `norm.json`."""
    out = []
    for task in sorted(Path(dataset).iterdir()):
        tests = task / "tests"
        if not (tests / "scenario.json").exists():
            continue
        raw = json.loads((tests / "scenario.json").read_text(encoding="utf-8"))
        norm = json.loads((tests / "norm.json").read_text(encoding="utf-8"))
        if "scenario_hash" in norm and norm["scenario_hash"] != content_hash(raw):
            raise ValueError(f"{task}: norm.json scenario_hash does not match scenario.json")
        out.append(EpisodeSpec(Scenario.model_validate(raw), float(norm["c_greedy"]), float(norm["c_ref"])))
    if not out:
        raise ValueError(f"{dataset}: no tasks with tests/scenario.json")
    return out


def episodes_from_scenarios(preset: str, set_name: str, count: int) -> list[EpisodeSpec]:
    """Accepted seeds (T06) regenerated with the T04 generator; their hash must match the accepted list."""
    from railway.gen.generator import generate
    from railway.gen.presets import PRESETS
    from railway.ref.seed_filter import accepted_seeds

    out = []
    for acc in accepted_seeds(preset, set_name, count):
        acc = acc if isinstance(acc, dict) else vars(acc)
        scenario = generate(PRESETS[preset], acc["seed"])
        if content_hash(scenario.model_dump(mode="json")) != acc["scenario_hash"]:
            raise ValueError(f"{preset}/{set_name} seed {acc['seed']}: accepted list is stale (hash mismatch)")
        out.append(EpisodeSpec(scenario, float(acc["c_greedy"]), float(acc["c_ref"])))
    return out


# --- results file ----------------------------------------------------------------------------------


def read_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _done_pairs(path: Path) -> set[tuple[str, int]]:
    return {(r["scenario_id"], r["rep"]) for r in read_rows(path) if r.get("status") == "ok"}


class _Writer:
    """The only thread that appends to the results file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.queue: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            while (row := self.queue.get()) is not None:
                fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                fh.flush()

    def put(self, row: dict) -> None:
        self.queue.put(row)

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()


# --- loop harness ----------------------------------------------------------------------------------


def _default_client_factory(manifest: Manifest):
    from railway.loop.model_client import ModelClient

    def factory(session, episode):
        m = manifest.model
        return ModelClient(
            m.base_url, m.name, m.api_key_env, temperature=manifest.temperature, max_tokens=manifest.max_tokens
        )

    return factory


def _run_loop(manifest: Manifest, out: Path, results_dir: Path, deps: LoopDeps | None) -> dict:
    deps = deps or LoopDeps()
    if manifest.dataset is not None:
        episodes = episodes_from_dataset(manifest.dataset)
    else:
        sc = manifest.scenarios
        episodes = episodes_from_scenarios(sc.preset, sc.set, sc.count)
    session_cls = deps.session_cls
    if session_cls is None:
        from railway.tools.session import Session as session_cls
    score_fn = deps.score_fn or reward_from_t07
    if deps.score_fn is None:
        from railway.harbor.verify import reward_from  # noqa: F401 - fail now, not inside every episode
    instruction = deps.instruction
    if instruction is None:
        from railway.loop.prompts import load_instruction

        instruction = load_instruction()
    client_factory = deps.client_factory or _default_client_factory(manifest)

    done = _done_pairs(out)
    todo = [(ep, rep) for ep in episodes for rep in range(manifest.reps) if (ep.scenario.scenario_id, rep) not in done]
    transcripts = results_dir / manifest.name / "transcripts"
    writer = _Writer(out)

    def one(item):
        ep, rep = item
        sid = ep.scenario.scenario_id
        session = client = None
        try:
            session = make_session(
                ep.scenario, disabled=manifest.tools_disabled, diagnostics=manifest.diagnostics, session_cls=session_cls
            )
            client = client_factory(session, ep)
            rec = play_episode(
                ep.scenario,
                client,
                manifest.mode,
                disabled=manifest.tools_disabled,
                session=session,
                instruction=instruction,
                transcript_path=transcripts / f"{sid}-r{rep}.jsonl",
                run=manifest.name,
                model=manifest.model.id,
                rep=rep,
                c_greedy=ep.c_greedy,
                c_ref=ep.c_ref,
                score_fn=score_fn,
            )
        except Exception as exc:  # noqa: BLE001 - infra failure: an error row, never a score
            if session is not None and not session.closed:
                try:
                    session.close()
                except Exception:  # noqa: BLE001, S110
                    pass
            rec = EpisodeRecord(
                run=manifest.name,
                mode=manifest.mode,
                model=manifest.model.id,
                rep=rep,
                c_greedy=ep.c_greedy,
                c_ref=ep.c_ref,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                tool_calls=None,
                tokens_in=None,
                tokens_out=None,
                format_errors=None,
                **scenario_meta(ep.scenario),
            )
        finally:
            if client is not None and hasattr(client, "close"):
                client.close()
        writer.put(rec.to_row())

    try:
        with ThreadPoolExecutor(max_workers=manifest.concurrency) as pool:
            list(pool.map(one, todo))
    finally:
        writer.close()
    return {"executed": len(todo), "skipped": len(episodes) * manifest.reps - len(todo), "results": str(out)}


# --- Harbor harness --------------------------------------------------------------------------------


def harbor_subprocess(cmd: list[str], env: dict) -> int:
    """Runs the Harbor CLI. Tests monkeypatch this."""
    return subprocess.run(cmd, env=env, check=False).returncode


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj and obj[key] is not None:
            return obj[key]
        for v in obj.values():
            if (found := _find_key(v, key)) is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            if (found := _find_key(v, key)) is not None:
                return found
    return None


def _trial_task(trial: Path, tasks: dict[str, Path]) -> str | None:
    """Which task a trial ran: a task path in its config.json, else the `<task>__<id>` dir-name prefix."""
    config = trial / "config.json"
    if config.exists():
        for s in _strings(json.loads(config.read_text(encoding="utf-8"))):
            name = Path(s.rstrip("/\\")).name
            if name in tasks:
                return name
    prefix = trial.name.split("__")[0]
    matches = [t for t in tasks if t.startswith(prefix)]
    return matches[0] if len(matches) == 1 else None


def parse_harbor_jobs(jobs: Path, tasks: dict[str, Path], manifest: Manifest) -> list[EpisodeRecord]:
    """One record per trial (`<jobs>/<job>/<trial>/result.json`), from the verifier's details.json."""
    records = []
    reps: dict[str, int] = defaultdict(int)
    for trial in sorted(p.parent for p in jobs.glob("*/*/result.json")):
        sid = _trial_task(trial, tasks)
        if sid is None:
            continue
        scenario = Scenario.model_validate_json((tasks[sid] / "tests" / "scenario.json").read_text(encoding="utf-8"))
        norm = json.loads((tasks[sid] / "tests" / "norm.json").read_text(encoding="utf-8"))
        result = json.loads((trial / "result.json").read_text(encoding="utf-8"))
        details_path = trial / "verifier" / "details.json"
        details = json.loads(details_path.read_text(encoding="utf-8")) if details_path.exists() else None
        rec = EpisodeRecord(
            run=manifest.name,
            harness="harbor",
            mode=None,
            model=manifest.model.id,
            rep=reps[sid],
            c_greedy=float(norm["c_greedy"]),
            c_ref=float(norm["c_ref"]),
            tool_calls=None,
            tokens_in=_find_key(result, "n_input_tokens"),
            tokens_out=_find_key(result, "n_output_tokens"),
            format_errors=None,
            **scenario_meta(scenario),
        )
        reps[sid] += 1
        exc = result.get("exception_info")
        exc_type = exc.get("exception_type") if isinstance(exc, dict) else None
        if details is None or (exc and exc_type not in AGENT_FAILURES):
            rec.status = "error"
            rec.error = (
                f"{exc.get('exception_type', 'exception')}: {exc.get('exception_message', '')}".strip()
                if isinstance(exc, dict)
                else (str(exc) if exc else "no verifier details.json")
            )
        else:
            counts = details.get("counts", {})
            rec.cost_total = details["cost_total"]
            rec.cost_by_component = details["cost_by_component"]
            rec.c_greedy, rec.c_ref = float(details["c_greedy"]), float(details["c_ref"])
            rec.score_clipped, rec.score_unclipped = details["reward"], details["score_unclipped"]
            rec.cost_ratio = rec.cost_total / rec.c_ref
            for key in ("gridlocks", "crew_timeouts", "refusals", "invalid_calls", "decision_points"):
                setattr(rec, key, counts.get(key))
            rec.unfinished = counts.get("unfinished")
            rec.tool_calls = details.get("tool_calls")
        records.append(rec)
    return records


def _run_harbor(manifest: Manifest, out: Path, results_dir: Path, env: dict | None) -> dict:
    dataset = Path(manifest.dataset)
    tasks = {p.name: p for p in sorted(dataset.iterdir()) if (p / "tests" / "scenario.json").exists()}
    done = _done_pairs(out)
    if all((sid, rep) in done for sid in tasks for rep in range(manifest.reps)):
        return {"executed": 0, "skipped": len(tasks) * manifest.reps, "results": str(out)}
    base = results_dir / manifest.name / "harbor-jobs"
    attempt = len(list(base.glob("attempt-*"))) if base.exists() else 0
    jobs = base / f"attempt-{attempt}"
    jobs.mkdir(parents=True, exist_ok=True)
    cmd = [
        "harbor",
        "run",
        "-y",
        "-p",
        str(dataset),
        "-a",
        "opencode",
        "-m",
        manifest.model.harbor_model,
        "-o",
        str(jobs),
        "-k",
        str(manifest.reps),
    ]
    env = dict(os.environ if env is None else env)
    # Keys stay in the env file; Harbor maps the named variable to the provider key (as in tests/helpers.py).
    if env.get("HARBOR_ENV_FILE"):
        cmd += ["--env-file", env["HARBOR_ENV_FILE"]]
    if env.get("HARBOR_API_KEY_NAME"):
        cmd += ["--api-key-name", env["HARBOR_API_KEY_NAME"]]
    harbor_subprocess(cmd, env)
    writer = _Writer(out)
    executed = 0
    try:
        for rec in parse_harbor_jobs(jobs, tasks, manifest):
            if (rec.scenario_id, rec.rep) not in done:
                writer.put(rec.to_row())
                executed += 1
    finally:
        writer.close()
    return {"executed": executed, "skipped": len(done), "results": str(out)}


def run_manifest(
    manifest: Manifest, *, results_dir: str | Path = "results", deps: LoopDeps | None = None, env: dict | None = None
) -> dict:
    results_dir = Path(results_dir)
    out = results_dir / f"{manifest.name}.jsonl"
    if manifest.harness == "loop":
        return _run_loop(manifest, out, results_dir, deps)
    return _run_harbor(manifest, out, results_dir, env)


# --- CLI -------------------------------------------------------------------------------------------


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="railway-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="run a manifest; append rows to results/<name>.jsonl")
    p_run.add_argument("manifest")
    p_run.add_argument("--results-dir", default="results")
    p_sum = sub.add_parser("summarize", help="print the stats table")
    p_sum.add_argument("results")
    p_sum.add_argument("--json", action="store_true")
    p_cmp = sub.add_parser("compare", help="paired difference b - a on common seeds, with CIs")
    p_cmp.add_argument("a")
    p_cmp.add_argument("b")
    p_cmp.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.cmd == "run":
        report = run_manifest(load_manifest(args.manifest), results_dir=args.results_dir)
        _print_json(report)
    elif args.cmd == "summarize":
        summary = stats.summarize(read_rows(args.results))
        _print_json(summary) if args.json else print(stats.format_summary(summary))
    else:
        a, b = read_rows(args.a), read_rows(args.b)
        diffs = {name: stats.paired_diff(a, b, name) for name in stats.STATS}
        if args.json:
            _print_json(diffs)
        else:
            print(f"{'statistic':<16}{'b - a':>10}{'95% CI':>24}  seeds")
            for name, d in diffs.items():
                lo, hi = d["ci"]
                print(f"{name:<16}{d['diff']:>10.4f}{f'[{lo:.4f}, {hi:.4f}]':>24}  {d['common_seeds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
