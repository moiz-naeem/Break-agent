"""Play railway shifts through headless OpenCode, one isolated OpenCode + sim server per job.

Each job is one (model, effort variant, preset, seed, repeat). The agent sees only the 15 railway
MCP tools: every built-in OpenCode tool is denied, global config and plugins are replaced by a
per-job config, and the working directory is empty.

Usage:
  uv run python infra/bench/oc_bench.py matrix --out jobs/ocbench/probe1        # write the probe job list
  uv run python infra/bench/oc_bench.py run --out jobs/ocbench/probe1 [--only REGEX] [--lanes ...]
  uv run python infra/bench/oc_bench.py report --out jobs/ocbench/probe1
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from railway.gen.generator import generate  # noqa: E402
from railway.gen.presets import PRESETS  # noqa: E402
from railway.server.spawn import spawn  # noqa: E402
from railway.sim.constants import TOOL_BUDGET  # noqa: E402

CATALOG = Path.home() / ".cache" / "opencode" / "models.json"
SHIFT_TIMEOUT_S = 4 * 3600

INSTRUCTION = f"""# Railway dispatcher shift

You are the dispatcher of a single-track railway for one shift. You control every train
through the tools of the `railway` MCP server; nothing moves without your authority.

1. Call `manual()` first. It contains every rule, cost formula and tool format.
2. Then read `network()`, `timetable()` and `status()`.
3. Time moves only when you call `advance_to_next_event()`. Give your orders
   (`authorize`, `hold`, `relieve_crew`, ...) and then advance.
4. Goal: the lowest total cost at the end of the shift (passenger and freight lateness,
   crew timeouts, gridlock reversals, unfinished trains).

Scoring: your total cost is compared with a simple greedy dispatcher (score 0) and a
strong reference dispatcher (score 1).

Ending: the shift ends when you call `end_shift()`, when you use up the tool budget
({TOOL_BUDGET} calls), or when you stop. Stopping without calling end_shift() counts as
end_shift(): the rest of the shift runs with no new authorities and unfinished trains
are charged.

Everything you need is in the tools. There are no files to read. Play the whole shift
without asking for confirmation; nobody will answer questions.
"""

# Endpoints outside OpenCode's catalog: OpenAI-compatible vLLM servers.
CUSTOM_PROVIDERS = {
    "verda": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Verda Nemotron 3.5 Super VL",
        "options": {"baseURL": "https://containers.datacrunch.io/nemotron-3-5-super-vl-b300/v1",
                    "apiKey": "{env:VERDA_INFERENCE_KEY}"},  # fmt: skip
        "models": {"nemotron-3.5-super-vl": {"name": "Nemotron 3.5 Super VL", "tool_call": True, "reasoning": True,
                                             "limit": {"context": 1048576, "output": 32768}}},  # fmt: skip
    },
    "modal": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Modal vLLM",
        "options": {"baseURL": "https://hoaianh517--railway-qwen-think.us-east.modal.direct/v1",
                    "apiKey": "{env:VLLM_API_KEY}"},  # fmt: skip
        "models": {"qwen38": {"name": "Qwen 3.8 27B (thinking)", "tool_call": True, "reasoning": True,
                              "limit": {"context": 131072, "output": 32768}}},  # fmt: skip
    },
}

# Lane = which jobs may run at the same time. Anthropic is one lane so the spending cap is exact.
LANE_OF_PROVIDER = {"anthropic": "anthropic", "openai": "openai", "google": "google", "nvidia": "nvidia",
                    "opencode": "opencode", "verda": "verda", "modal": "modal"}  # fmt: skip
DEFAULT_LANES = {"anthropic": 1, "openai": 4, "google": 2, "nvidia": 2, "opencode": 2, "verda": 2, "modal": 2}
ANTHROPIC_CAP_USD = 10.0
# Any other tool (shell, read, ...) means the sandbox leaked: the run is killed and never counted.
ALLOWED_TOOLS = {"execute"}

PROBE_MODELS = [
    # (provider, model, variants or None for "every variant OpenCode offers")
    ("anthropic", "claude-fable-5-2", ["high"]),
    ("anthropic", "claude-haiku-5-5", ["high"]),
    ("google", "gemini-4-hydrogen", None),
    ("google", "gemini-4-carbon", None),
    ("verda", "nemotron-3.5-super-vl", [None]),
    ("modal", "qwen38", [None]),
]
PROBE_SCENARIOS = [("T1", 5), ("T2", 0)]


def opencode_exe() -> str:
    # The npm .cmd/.ps1 wrappers mangle multi-line arguments on Windows; call the binary.
    if sys.platform == "win32":
        exe = Path(os.environ["APPDATA"]) / "npm/node_modules/@opencode/cli/bin/opencode.exe"
        if exe.exists():
            return str(exe)
    return shutil.which("opencode") or "opencode"


def load_env() -> dict[str, str]:
    env = dict(os.environ)
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2)
    if "GEMINI_API_KEY" in env:
        env.setdefault("GOOGLE_GENERATIVE_AI_API_KEY", env["GEMINI_API_KEY"])
    return env


def variants_for(provider: str, model: str) -> list[str | None]:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    entry = catalog.get(provider, {}).get("models", {}).get(model)
    if not entry:
        return [None]
    for opt in entry.get("reasoning_options") or []:
        if opt.get("type") == "effort":
            return list(opt["values"])
    return [None]


def norms() -> dict[tuple[str, int], dict]:
    out = {}
    for preset in ("T1", "T2"):
        path = ROOT / "seeds" / f"{preset}-dev.json"
        if path.exists():
            for a in json.loads(path.read_text(encoding="utf-8"))["accepted"]:
                out[(preset, a["seed"])] = a
    return out


def job_id(job: dict) -> str:
    model = job["model"].replace("/", "_")
    variant = job["variant"] or "default"
    return f"{job['preset']}-s{job['seed']}-{job['provider']}_{model}_{variant}-r{job['rep']}"


def cmd_matrix(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for preset, seed in PROBE_SCENARIOS:
        for provider, model, variants in PROBE_MODELS:
            for variant in variants if variants is not None else variants_for(provider, model):
                jobs.append({"provider": provider, "model": model, "variant": variant,
                             "preset": preset, "seed": seed, "rep": 0})  # fmt: skip
    # Anthropic runs T1 before T2 so the cap spends on the cheap tier first.
    jobs.sort(key=lambda j: (j["provider"] != "anthropic", j["preset"]))
    (out / "jobs.json").write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    print(f"{len(jobs)} jobs -> {out / 'jobs.json'}")
    for j in jobs:
        print(" ", job_id(j))


def opencode_config(mcp_url: str) -> dict:
    return {
        "$schema": "https://opencode.ai/config.json",
        "plugin": [],
        "autoupdate": False,
        "share": "disabled",
        "provider": CUSTOM_PROVIDERS,
        "mcp": {"railway": {"type": "remote", "url": mcp_url, "enabled": True}},
        # OpenCode 2.x exposes MCP tools inside its code-mode `execute` tool, so execute stays allowed;
        # everything else (shell, files, web, subagents, opencode.* helpers) is denied.
        "permission": {"*": "deny", "execute": "allow", "railway_*": "allow", "railway.*": "allow"},
    }


class Spend:
    """Running Anthropic spend, summed from OpenCode's per-step cost events."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.done = json.loads(path.read_text())["usd"] if path.exists() else 0.0
        self.live = 0.0

    def total(self) -> float:
        with self.lock:
            return self.done + self.live

    def set_live(self, usd: float):
        with self.lock:
            self.live = usd

    def commit(self, usd: float):
        with self.lock:
            self.done += usd
            self.live = 0.0
            self.path.write_text(json.dumps({"usd": round(self.done, 4)}))


def parse_events(path: Path) -> dict:
    """Sum tokens and cost from `opencode run --format json` output."""
    tokens = {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0}
    cost = 0.0
    steps = tool_calls = 0
    errors = []
    forbidden = []
    if not path.exists():
        return {"tokens": tokens, "usd": 0.0, "steps": 0, "tool_events": 0, "errors": [], "forbidden": []}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = ev.get("part") or {}
        kind = ev.get("type") or ""
        if kind == "step_finish" or part.get("type") == "step-finish":
            steps += 1
            cost += float(part.get("cost") or 0)
            t = part.get("tokens") or {}
            tokens["input"] += int(t.get("input") or 0)
            tokens["output"] += int(t.get("output") or 0)
            tokens["reasoning"] += int(t.get("reasoning") or 0)
            cache = t.get("cache") or {}
            tokens["cache_read"] += int(cache.get("read") or 0)
            tokens["cache_write"] += int(cache.get("write") or 0)
        elif kind == "tool_use" or part.get("type") == "tool":
            tool_calls += 1
            code = json.dumps((part.get("state") or {}).get("input") or {})
            if part.get("tool") not in ALLOWED_TOOLS or "tools.opencode" in code:
                forbidden.append(f"{part.get('tool')}: {code[:200]}")
        elif kind == "error":
            errors.append(json.dumps(ev.get("error") or ev)[:500])
    return {"tokens": tokens, "usd": round(cost, 4), "steps": steps, "tool_events": tool_calls, "errors": errors,
            "forbidden": forbidden}  # fmt: skip


def prepare(job: dict, out: Path) -> tuple[Path, dict] | dict:
    """Fresh job directory and the result row; returns the stored summary instead if the job is done."""
    jdir = out / "runs" / job_id(job)
    summary_path = jdir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if jdir.exists():
        shutil.rmtree(jdir)
    jdir.mkdir(parents=True)
    model_arg = f"{job['provider']}/{job['model']}" + (f"#{job['variant']}" if job["variant"] else "")
    return jdir, {**job, "id": job_id(job), "model_arg": model_arg}


def watch(poll, kill, events: Path, flush, t0: float, spend: Spend | None) -> str | None:
    """Wait for the agent; kill it on a sandbox violation, the timeout or the Anthropic cap."""
    while poll() is None:
        time.sleep(1)
        flush()
        live = parse_events(events)
        killed = None
        if live["forbidden"]:
            killed = "sandbox_violation"
        elif time.monotonic() - t0 > SHIFT_TIMEOUT_S:
            killed = "timeout"
        elif spend is not None:
            spend.set_live(live["usd"])
            if spend.total() >= ANTHROPIC_CAP_USD:
                killed = "anthropic_cap"
        if killed:
            kill()
            return killed
    return None


def fetch_log(base: str, token: str, row: dict) -> dict | None:
    try:
        r = httpx.get(base + "/episode_log", headers={"Authorization": f"Bearer {token}"}, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001
        row["infra_error"] = f"episode_log: {e}"
        return None


def finalize(job: dict, out: Path, jdir: Path, row: dict, log: dict | None, killed: str | None,
             spend: Spend | None) -> dict:  # fmt: skip
    row["killed"] = killed
    norm = norms().get((job["preset"], job["seed"]), {})
    if log is not None:
        (jdir / "episode.json").write_text(json.dumps(log), encoding="utf-8")
        entries = log["entries"]
        row.update({
            "final_cost": log["final_cost"], "by_component": log["final_cost_by_component"],
            "final_tick": log["final_tick"], "tool_calls": len(entries),
            "refused": sum(e["status"] == "REFUSED" for e in entries),
            "invalid": sum(e["status"] == "INVALID" for e in entries),
            "ended_by": "end_shift" if any(e["tool"] == "end_shift" for e in entries) else "stopped_or_budget",
        })  # fmt: skip
        if norm:
            g, ref = norm["c_greedy"], norm["c_ref"]
            raw = (g - log["final_cost"]) / (g - ref)
            row.update({"c_greedy": g, "c_ref": ref, "score_unclipped": round(raw, 4),
                        "score": round(min(1.0, max(0.0, raw)), 4)})  # fmt: skip
    usage = parse_events(jdir / "events.jsonl")
    row.update({"usd": usage["usd"], "tokens": usage["tokens"], "llm_steps": usage["steps"],
                "oc_errors": usage["errors"][:5], "forbidden_tools": usage["forbidden"][:10]})  # fmt: skip
    if usage["forbidden"]:
        killed = row["killed"] = "sandbox_violation"
    if spend is not None:
        spend.commit(usage["usd"])
    # Only a finished, clean episode is final; anything else reruns next time.
    if log is not None and not killed and not row.get("infra_error") and row.get("tool_calls", 0) > 3:
        (jdir / "summary.json").write_text(json.dumps(row, indent=1), encoding="utf-8")
    (jdir / "attempt.json").write_text(json.dumps(row, indent=1), encoding="utf-8")
    with (out / "results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def run_job(job: dict, out: Path, env_base: dict, spend: Spend | None) -> dict:
    """Local backend: OpenCode on this machine. Not a sandbox; see the forbidden-tool guard."""
    prepared = prepare(job, out)
    if isinstance(prepared, dict):
        return prepared
    jdir, row = prepared
    (jdir / "agent").mkdir()
    for d in ("config/opencode", "data", "state"):
        (jdir / "oc" / d).mkdir(parents=True)
    scenario = generate(PRESETS[job["preset"]], job["seed"])
    t0 = time.monotonic()
    with spawn(scenario) as sim:
        cfg = opencode_config(sim.url)
        (jdir / "oc/config/opencode/opencode.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        env = dict(env_base)
        env.update({
            "XDG_CONFIG_HOME": str(jdir / "oc/config"), "XDG_DATA_HOME": str(jdir / "oc/data"),
            "XDG_STATE_HOME": str(jdir / "oc/state"), "OPENCODE_CONFIG_DIR": str(jdir / "oc/config/opencode"),
            "OPENCODE_CONFIG": str(jdir / "oc/config/opencode/opencode.json"),
        })  # fmt: skip
        cmd = [opencode_exe(), "run", "--standalone", "--format", "json", "--auto", "--archived",
               "-m", row["model_arg"], INSTRUCTION]  # fmt: skip
        events = jdir / "events.jsonl"
        with events.open("w", encoding="utf-8") as fo, (jdir / "stderr.log").open("w", encoding="utf-8") as fe:
            proc = subprocess.Popen(cmd, cwd=jdir / "agent", env=env, stdin=subprocess.DEVNULL, stdout=fo, stderr=fe)

            def kill():
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
                else:
                    proc.kill()
                proc.wait(60)

            killed = watch(proc.poll, kill, events, fo.flush, t0, spend)
        row["exit"] = proc.returncode
        row["wall_s"] = round(time.monotonic() - t0)
        log = fetch_log(sim.base, sim.token, row)
    return finalize(job, out, jdir, row, log, killed, spend)

def cmd_run(args):
    out = Path(args.out)
    jobs = json.loads((out / "jobs.json").read_text(encoding="utf-8"))
    if args.only:
        jobs = [j for j in jobs if re.search(args.only, job_id(j))]
    lanes = dict(DEFAULT_LANES)
    for spec in args.lanes or []:
        k, v = spec.split("=")
        lanes[k] = int(v)
    env = load_env()
    spend = Spend(out / "anthropic_spend.json")
    runner = run_job
    if args.backend == "modal":
        import modal_bench

        runner = modal_bench.run_job
    by_lane: dict[str, list] = {}
    for j in jobs:
        by_lane.setdefault(LANE_OF_PROVIDER.get(j["provider"], j["provider"]), []).append(j)
    print_lock = threading.Lock()

    def lane_worker(lane: str, queue: list, width: int):
        def one(job):
            if lane == "anthropic" and spend.total() >= ANTHROPIC_CAP_USD:
                with print_lock:
                    print(f"SKIP {job_id(job)}: Anthropic cap ${ANTHROPIC_CAP_USD} reached", flush=True)
                return
            with print_lock:
                print(f"START {job_id(job)}", flush=True)
            try:
                row = runner(job, out, env, spend if lane == "anthropic" else None)
                msg = (f"DONE {row['id']} cost={row.get('final_cost')} score={row.get('score')} "
                       f"calls={row.get('tool_calls')} usd={row.get('usd')} wall={row.get('wall_s')}s "
                       f"killed={row.get('killed')} "
                       f"err={row.get('infra_error') or row.get('oc_errors') or ''}")  # fmt: skip
            except Exception as e:  # noqa: BLE001
                msg = f"FAIL {job_id(job)}: {type(e).__name__}: {e}"
            with print_lock:
                print(msg, flush=True)

        with cf.ThreadPoolExecutor(max_workers=width) as pool:
            list(pool.map(one, queue))

    threads = [threading.Thread(target=lane_worker, args=(lane, q, lanes.get(lane, 1))) for lane, q in by_lane.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"ALL DONE. Anthropic spend ${spend.total():.2f}", flush=True)


def cmd_report(args):
    out = Path(args.out)
    rows = []
    for p in sorted((out / "runs").glob("*/summary.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    cols = ["id", "final_cost", "score", "score_unclipped", "tool_calls", "refused", "usd", "wall_s", "ended_by"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r.get(c, "")) for c in cols))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("matrix")
    m.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--only")
    r.add_argument("--lanes", nargs="*")
    r.add_argument("--backend", choices=("modal", "local"), default="modal")
    rep = sub.add_parser("report")
    rep.add_argument("--out", required=True)
    a = p.parse_args()
    {"matrix": cmd_matrix, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
