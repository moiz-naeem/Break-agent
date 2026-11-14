# T01: Foundations and Harbor spike

| | |
|---|---|
| Depends on | nothing (first ticket) |
| Unblocks | T02 (repo + test harness), T07 (architecture decision: sidecar or fallback) |
| DESIGN.md | §8 (Harbor packaging), §8.4 (facts to test), §11 (hours 0–3), §13 |
| PRD | §5 (process), §7.1 (layout), §7.12 (determinism), §9 (open items) |
| Required markers | default + `harbor`; `model` for facts F-ALLOW, F-NONROOT, F-SMALL |
| Suggested split | A: repo scaffold + util + test harness. B: toy Harbor task + harbor runs + spike report. C: toy MCP server + its unit tests |

## 1. Goal

There are two outputs:

1. **A working repo skeleton.** It has the uv project, the package layout, the test
   harness with markers, and the determinism helpers. It is something every later ticket
   can build on.
2. **A decision, with evidence, on the Harbor architecture.** Everything in T05 and T07
   assumes a separate `sim` container reachable from the agent and the verifier. This
   ticket proves or disproves that with a toy task before any real code depends on it.

## 2. Context the implementer needs

- **Harbor** (https://github.com/laude-institute/harbor) runs a *task*:
  - `instruction.md` — what the agent reads;
  - `task.toml` — config;
  - `environment/Dockerfile` — the agent's "main" container;
  - optional `environment/docker-compose.yaml` — extra services ("sidecars"), reachable
    by service name;
  - `tests/test.sh` — the verifier. It runs after the agent finishes, and Harbor uploads
    `tests/` only then;
  - `solution/solve.sh` — used only by the oracle agent `-a oracle`.
- The verifier writes one float to `/logs/verifier/reward.txt`.
  - `reward.json` takes priority if present; we must never write it.
  - Harbor parses `reward.txt` with `float()` and no range check.
- **Commands:**
  - Oracle: `harbor run -p <task-or-dataset> -a oracle`.
  - With OpenCode: `harbor run -p <task> -a opencode -m <provider/model>`.
- **Confirmed earlier from Harbor source** (DESIGN.md §8.4):
  - sidecars in compose;
  - `[[environment.mcp_servers]]` with `name`, `transport`
    (`stdio | sse | streamable-http`) and `url`, passed to OpenCode as a remote MCP
    server;
  - `[environment] network_mode = "allowlist"` with `allowed_hosts`;
  - `[agent] user` to run as non-root (the default is root);
  - verifier "shared" mode (the default) keeps sidecars; "separate" mode drops them;
  - sidecars work on docker, podman, ec2, daytona and modal (DinD), but not on e2b or
    runloop.
- **Still unconfirmed.** Those are the facts this ticket must establish:

| Fact id | Question | How to establish |
|---|---|---|
| F-REACH | Can the verifier (main container, after the agent) reach the `sim` sidecar over HTTP? | Oracle run: `solve.sh` calls the toy MCP tool; `test.sh` fetches `/episode_log` from `sim` and writes 1.0 only if the call is there |
| F-ALLOW | With `network_mode = "allowlist"`, does OpenCode still install/start and reach the model provider? | OpenCode run with the allowlist on |
| F-NONROOT | Does OpenCode work with `[agent] user = "agent"`? | OpenCode run with a non-root user; the instruction makes the agent pass `id -u` to the `ping` tool |
| F-SMALL | Can OpenCode drive a small model behind an OpenAI-compatible endpoint (vLLM, Ollama, OpenRouter...)? | OpenCode run with `-m <provider>/<small-model>`; reward 1.0 |
| F-CUSTOM | Can Harbor load a custom agent from our Python package (for the scripted stop-early baseline)? | `harbor run -a <import path>` with a do-nothing custom agent |
| F-NOP | Is there a built-in do-nothing agent (and what is it called)? | `harbor --help` / source; then a run gives reward 0.0 |
| F-SECRET | Is the sidecar's filesystem invisible to the main container? | `test.sh` (runs in main) checks that `/srv/secret.txt` (present only in the sim image) does not exist |
| F-LAYOUT | Where does Harbor write trial results (reward, verifier logs), and what is the exact compose/`task.toml` shape it accepts? | Read from a real run; recorded for the test helpers |

## 3. Scope

**In scope:**

1. **Repo scaffold** per PRD §7.1:
   - `pyproject.toml` (uv, Python ≥ 3.11);
   - the `railway/` package with every subpackage as an empty `__init__.py`;
   - `tests/conftest.py`, `tests/helpers.py`, `tests/fixtures/`;
   - ruff config;
   - a `Makefile` with `test`, `lint`, `ci` aliases;
   - `.gitignore` (`.venv/`, `results/*.jsonl` except committed headline files,
     `reports/*.html`, `jobs/`, `__pycache__/`, `datasets/*-fresh/`).
2. **`railway/util.py`:**
   - `canonical_json(obj) -> str` (PRD §7.12);
   - `content_hash(obj) -> str` (sha256 hex of the canonical JSON).
3. **Test harness:**
   - markers `slow`, `harbor`, `model` and `gpu` registered, `--strict-markers` on;
   - a Hypothesis profile `ci` with `derandomize=True`, `max_examples=200`, loaded by
     default;
   - in `tests/helpers.py`:
     - `load_fixture(name) -> dict`;
     - `variant(base: dict, **patches) -> dict` (deep-copies, then applies dotted-path
       patches, e.g. `variant(d, **{"shift_ticks": 90})`);
     - `harbor_run(...)` (below).
4. **Toy Harbor task** `spike/toy-task/`:
   - `instruction.md`: "Call the `ping` tool of the `toy` MCP server once with the
     output of `id -u` as `text`. Then stop."
   - `task.toml`:
     - an MCP server `toy`, `streamable-http`, `http://sim:8000/mcp`;
     - variants for the allowlist and non-root runs, as separate copies
       `spike/toy-task-allowlist/` and `spike/toy-task-nonroot/`, or a templating
       script — keep it simple.
   - `environment/Dockerfile` (main): a Debian/Ubuntu base with `python3` (≥ 3.11),
     `curl`, and pip-installed `mcp`, `httpx` and `pydantic`. These are needed later by
     the oracle and the verifier. **Nothing from `environment/sim/` is copied.**
   - `environment/docker-compose.yaml`: adds a service `sim` built from `./sim`, with
     port 8000 and a healthcheck on `/health`.
   - `environment/sim/Dockerfile` + `environment/sim/server.py`: a FastMCP server with
     - one tool, `ping(text: str) -> str`, returning `"pong:" + text`, which records each
       call;
     - `GET /health` → `ok`;
     - `GET /episode_log` → a JSON list of calls. It requires `Authorization: Bearer
       <token>`, with the token read from `/srv/token`.
     - The image also contains `/srv/secret.txt`.
   - `tests/test.sh` (the verifier):
     - reads the token from `tests/token`;
     - curls `http://sim:8000/episode_log`;
     - if the sim is unreachable after 5 retries, it exits 1 **without** writing a
       reward;
     - writes `1.0` if there is at least one `ping` call **and** `/srv/secret.txt` does
       not exist in this (main) container; otherwise `0.0`;
     - also writes `/logs/verifier/details.json` with the calls (this shows the
       `id -u` value for F-NONROOT).
   - `solution/solve.sh`: a Python MCP client (`mcp.client.streamable_http`) calls
     `ping` with `$(id -u)`.
5. **Custom agent** for F-CUSTOM: `railway/harbor/spike_agents.py:NoOpAgent`. It is
   implemented against Harbor's agent base class and does nothing. Read Harbor's
   `BaseAgent` interface at the pinned version.
6. **`tests/helpers.py::harbor_run(task_path, agent, model=None, extra_args=()) ->
   list[TrialResult]`:**
   - runs `harbor run` as a subprocess and parses the trial results (reward or error)
     from Harbor's output directory (fact F-LAYOUT);
   - `TrialResult = {reward: float | None, errored: bool, verifier_details: dict |
     None, raw_dir: Path}`.
7. **`docs/spike-report.md`** with:
   - the Harbor version (pip version + git sha);
   - the host OS and Docker version;
   - for **each fact id** in §2: `PASS | FAIL | BLOCKED`, the exact command, and an
     evidence excerpt;
   - the exact working `task.toml` and `docker-compose.yaml` shapes;
   - the trial output layout;
   - the name of the built-in no-op agent;
   - the custom-agent import syntax;
   - one line: `Decision: SIDECAR` or `Decision: FALLBACK (simulator inside main
     container)`, with the reason;
   - one line per FAIL saying what T07 must do differently.

**Out of scope:**

- The real simulator and the real MCP tools (T02–T05).
- The real export (T07).
- Building the single-container fallback (only in T07, and only if
  `Decision: FALLBACK`).

## 4. Implementation notes and pitfalls

- **Pin versions.** `uv add mcp pydantic httpx pyyaml numpy matplotlib`, plus dev
  dependencies `uv add --dev pytest hypothesis ruff`. Record the Harbor version in the
  spike report. Install Harbor per its README (`uv tool install harbor` or `pip install
  harbor`).
- **FastMCP:**
  - `from mcp.server.fastmcp import FastMCP`;
  - `mcp = FastMCP("toy", host="0.0.0.0", port=8000)`;
  - tools are declared with `@mcp.tool()`;
  - extra HTTP routes are declared with `@mcp.custom_route("/health", methods=["GET"])`
    (Starlette request/response);
  - `mcp.run(transport="streamable-http")` serves MCP at `/mcp`.
  - Check these names against the installed `mcp` version. If the API differs, follow
    the installed version and note it in the spike report.
- **Harbor needs Docker.** On Windows, run the `harbor` tests inside WSL2 with Docker
  Desktop's WSL integration. Unit tests run anywhere.
- **Model provider for F-ALLOW / F-SMALL / F-NONROOT.** Models are an open item (PRD
  §9). Use any small model reachable from the team (e.g. an Ollama or vLLM endpoint, or
  OpenRouter) and note which one. If no model access exists yet, mark those facts
  `BLOCKED`. The ticket can still pass (see §7), but T07 then re-runs them before its
  gate.
- **Allowlist host list.** For the OpenCode run, include the provider host and whatever
  OpenCode's install step fetches (find out from the failure: npm registry, GitHub
  releases, ...). Record the minimal list. T07 uses it.
- **Do not put the token or secret in the main image.** The token for the toy verifier
  lives in the task's `tests/token` (uploaded after the agent runs) and in the sim image.

## 5. Tests (write these first)

`tests/t01/test_scaffold.py`

| Id | Test | Assertion |
|---|---|---|
| T01-01 | `test_T01_01_package_layout` | Each of `railway.sim, gen, tools, server, policies, ref, harbor, loop, eval, train` is importable |
| T01-02 | `test_T01_02_python_and_deps` | `sys.version_info >= (3, 11)`; `importlib.metadata.version` works for `pydantic` (major 2), `mcp`, `httpx`, `hypothesis`, `pytest` |
| T01-03 | `test_T01_03_markers_registered` | The pytest config (read via `pytestconfig.getini("markers")`) contains `slow`, `harbor`, `model`, `gpu`; `--strict-markers` is in `addopts` |
| T01-04 | `test_T01_04_hypothesis_profile` | `hypothesis.settings.default.derandomize is True` |

`tests/t01/test_util.py`

| Id | Test | Assertion |
|---|---|---|
| T01-05 | `test_T01_05_canonical_json_key_order` | `canonical_json({"b":1,"a":[2,{"d":1,"c":2}]}) == '{"a":[2,{"c":2,"d":1}],"b":1}'` |
| T01-06 | `test_T01_06_canonical_json_unicode_and_floats` | `canonical_json({"x":"é","y":0.1})` contains `"é"` (not `é`) and `0.1` |
| T01-07 | `test_T01_07_content_hash_stable` | Two dicts with equal content but different insertion order → equal `content_hash`; hash is 64 hex chars; changing one value changes the hash |
| T01-08 | `test_T01_08_hash_across_processes` | Run `content_hash` of a fixed object in 2 subprocesses with `PYTHONHASHSEED=1` and `=2` → identical output |

`tests/t01/test_helpers.py`

| Id | Test | Assertion |
|---|---|---|
| T01-09 | `test_T01_09_variant_deep_copies` | `variant(d, **{"a.b": 5})` returns a new dict with `a.b == 5`; the original `d` is unchanged |
| T01-10 | `test_T01_10_variant_list_index` | `variant(d, **{"trains.0.pax": 7})` patches list elements by index |

`tests/t01/test_toy_mcp.py` (starts `spike/toy-task/environment/sim/server.py` in a
subprocess on a free port, with a temp token file)

| Id | Test | Assertion |
|---|---|---|
| T01-11 | `test_T01_11_health` | `GET /health` → 200, body `ok` |
| T01-12 | `test_T01_12_list_tools` | An MCP client over streamable-http lists exactly one tool, `ping` |
| T01-13 | `test_T01_13_ping_roundtrip` | `ping(text="7")` → `"pong:7"` |
| T01-14 | `test_T01_14_episode_log_requires_token` | No header → 403; wrong token → 403; correct token → 200 with a JSON list containing the ping call with `text == "7"` |
| T01-15 | `test_T01_15_no_other_routes` | `GET /` and `GET /scenario` → 404; `POST /episode_log` → 405 |

`tests/t01/test_toy_task_files.py` (static checks, no Docker)

| Id | Test | Assertion |
|---|---|---|
| T01-16 | `test_T01_16_task_layout` | `spike/toy-task/` has `instruction.md`, `task.toml`, `environment/Dockerfile`, `environment/docker-compose.yaml`, `environment/sim/Dockerfile`, `environment/sim/server.py`, `tests/test.sh`, `solution/solve.sh` |
| T01-17 | `test_T01_17_task_toml_mcp` | `tomllib` parses `task.toml`; it declares an MCP server `toy` with transport `streamable-http` and url `http://sim:8000/mcp` |
| T01-18 | `test_T01_18_main_dockerfile_has_no_sim` | `environment/Dockerfile` has no `COPY`/`ADD` referencing `sim`, `secret` or `token` |
| T01-19 | `test_T01_19_verifier_never_writes_reward_json` | `tests/test.sh` text does not contain `reward.json` |

`tests/t01/test_spike_harbor.py` (all marked `harbor`; the last three also `model`)

| Id | Test | Assertion |
|---|---|---|
| T01-H1 | `test_T01_H1_oracle_reaches_sim` (F-REACH, F-SECRET) | `harbor_run("spike/toy-task", "oracle")` → 1 trial, not errored, reward == 1.0 |
| T01-H2 | `test_T01_H2_nop_scores_zero` (F-NOP) | Built-in no-op agent → reward == 0.0, not errored |
| T01-H3 | `test_T01_H3_custom_agent_loads` (F-CUSTOM) | `harbor_run(..., agent="railway.harbor.spike_agents:NoOpAgent")` (syntax per Harbor) → not errored, reward 0.0 |
| T01-H4 | `test_T01_H4_sim_down_is_trial_error` | A copy of the toy task whose compose omits `sim` → the trial is errored (or reward is None), **not** reward 0.0 |
| T01-H5 | `test_T01_H5_allowlist_opencode` (F-ALLOW, `model`) | `spike/toy-task-allowlist` with OpenCode + the chosen model → reward 1.0 |
| T01-H6 | `test_T01_H6_nonroot_opencode` (F-NONROOT, `model`) | `spike/toy-task-nonroot` with OpenCode → reward 1.0, and `verifier_details` shows a ping `text` that is not `"0"` |
| T01-H7 | `test_T01_H7_small_model` (F-SMALL, `model`) | OpenCode with a small model (≤ 8B) behind an OpenAI-compatible endpoint → reward 1.0 |

`tests/t01/test_spike_report.py`

| Id | Test | Assertion |
|---|---|---|
| T01-20 | `test_T01_20_report_complete` | `docs/spike-report.md` exists; contains every fact id from §2 followed by `PASS`, `FAIL` or `BLOCKED`; contains a line matching `^Decision: (SIDECAR\|FALLBACK)`; contains `Harbor version:` |

## 6. Gate commands

```bash
uv run pytest tests/t01 -m "not harbor and not model and not gpu"
```
```bash
uv run pytest tests/t01 -m "harbor and not model"
```
```bash
uv run pytest tests/t01 -m "harbor and model"
```

The third run is required unless the model facts are `BLOCKED` (see §7).

## 7. Definition of done

- [ ] All non-marker tests pass; T01-H1 to T01-H4 pass.
- [ ] T01-H5 to T01-H7 pass, **or** they are marked `BLOCKED` in the spike report
      because no model access exists yet. A `BLOCKED` fact moves to T07's gate as a
      required re-run. A `FAIL` does not block T01, but the report must state the T07
      consequence.
- [ ] `docs/spike-report.md` is complete with a `Decision:` line. If `FALLBACK`, tell the
      user immediately: it changes T05 (no HTTP sidecar) and T07.
- [ ] The minimal allowlist host list is recorded.
- [ ] Gate record `docs/gates/T01.md` is written (PRD §5.4) and the reviewers ran.
- [ ] Hand over the changed-files list, a `git add` line and a draft commit message. Do
      not commit.
