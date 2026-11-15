# T07: Harbor task export, verifier, oracle, isolation

| | |
|---|---|
| Depends on | T06 (reference/oracle, seed filter, accepted lists); T01 spike report (exact Harbor shapes) |
| Unblocks | T08 (Harbor harness runs), T09 (held-out dataset, judge-facing runs) |
| DESIGN.md | §6.2 (reward), §6.5 (exploits), §8 (all), §9.3 (datasets), §13 (architecture, isolation, verifier, infra failures) |
| PRD | §3 G2, G3, G7; §7.9 (episode log); §8 D11, D12, D14, D18; §9 (backend, provider host) |
| Required markers | default + `harbor`; `model` for T07-H8; re-run any T01 fact marked `BLOCKED` |
| Suggested split | A: `export.py` + templates + dataset manifest. B: `verify.py` + verifier unit tests + Harbor runs. C: isolation probes (allowlist, token, non-root, image contents) + `make ci` |

## 1. Goal

Turn accepted scenarios into Harbor tasks the judges can run with no surprises:

- the oracle scores exactly **1.0**;
- do-nothing and stop-early score **0.0**;
- an infrastructure failure is a **trial error**, never a 0;
- the agent's container holds **nothing secret** and reaches only the model API and
  `sim`;
- the verifier **re-scores by replaying** the action log against a scenario that
  arrives only after the agent is done.

## 2. Context the implementer needs

### 2.1 Read first

`docs/spike-report.md` (T01). It has:

- the exact `task.toml` keys;
- the compose shape;
- the trial-output layout;
- the no-op agent name;
- the custom-agent syntax;
- the minimal allowlist;
- the `Decision:` line.

**If `Decision: FALLBACK`**, build §2.8 instead of the sidecar layout and tell the user
first. Everything else in this ticket still applies.

### 2.2 Export command (extends the `gen` CLI)

```
gen export --config T1 --set dev --count 3 --out datasets/t1-dev-mini
           [--fresh N] [--allow-host HOST ...] [--no-default-hosts]
           [--agent-user agent|root] [--agent-timeout SEC] [--probe]
```

- Uses `accepted_seeds(preset, set, count)` from T06. Recomputes if stale.
- `--fresh N`:
  - draws N accepted seeds from the `fresh` range;
  - writes to `<out>-fresh/`, which `.gitignore` covers;
  - prints the seeds.
- Writes one task directory per scenario, named `<scenario_id>`, plus
  `<out>/manifest.json` and `<out>/README.md`.
- `--probe` adds a diagnostics task (§2.7). Never use it for judge datasets.

### 2.3 Task layout

DESIGN.md §8.1, made exact:

```
<scenario_id>/
  instruction.md
  task.toml
  environment/
    Dockerfile                 main image: python3.11 + pinned pydantic/mcp/httpx; user "agent"; NOTHING from sim/
    docker-compose.yaml        adds service "sim" (build ./sim, port 8000, healthcheck GET /health)
    sim/
      Dockerfile               python:3.11-slim; COPY railway/ + scenario.json + token; CMD railway-sim ...
      railway/                 vendored: __init__, util, sim/, tools/, server/   (no gen/ref/policies/harbor/loop/eval/train)
      requirements.txt         pinned from uv.lock (pydantic, mcp, httpx, uvicorn/starlette as needed)
      scenario.json            canonical JSON
      token                    32 random bytes hex (secrets.token_hex(32)), fresh per export
  tests/
    test.sh                    `python3 "$(dirname "$0")/verify.py" --logs-dir /logs/verifier`
    verify.py                  copy of railway/harbor/verify.py
    railway/                   vendored: __init__, util, sim/, tools/ (for replay)
    scenario.json  norm.json  token
  solution/
    solve.sh                   `PYTHONPATH="$(dirname "$0")" python3 -m railway.ref.oracle --url http://sim:8000/mcp`
    railway/                   vendored: __init__, util, sim/, tools/, ref/, policies/ (no gen)
```

`norm.json` holds `{scenario_id, scenario_hash, rules_version, c_greedy, c_ref}`, where
`c_*` come from T06.

### 2.4 `instruction.md` (template; exact text, with the budget interpolated)

```markdown
# Railway dispatcher shift

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

Everything you need is in the tools. There are no files to read.
```

### 2.5 `task.toml` (template; key names as confirmed in the spike report)

```toml
version = "1.0"

[metadata]
difficulty = "hard"
category = "planning"
tags = ["railway", "dispatching", "long-horizon", "mcp", "deterministic"]

[agent]
timeout_sec = 3600            # provisional; set from the pilot (T09)
user = "agent"                # only if spike F-NONROOT == PASS; else omit

[verifier]
timeout_sec = 600

[environment]
network_mode = "allowlist"
allowed_hosts = ["sim", ...]  # D18 defaults + spike's OpenCode install hosts + --allow-host

[[environment.mcp_servers]]
name = "railway"
transport = "streamable-http"
url = "http://sim:8000/mcp"
```

- **D18 default hosts:**
  - `api.openai.com`
  - `api.anthropic.com`
  - `openrouter.ai`
  - `generativelanguage.googleapis.com`
  - `api.together.xyz`
  - `api.fireworks.ai`
  - `api.groq.com`
  - `api.mistral.ai`
  - `api.deepseek.com`
- **No author email or personal data** in any template.

### 2.6 Verifier (`railway/harbor/verify.py`, vendored into `tests/`)

```
verify.py --logs-dir DIR [--sim-url http://sim:8000] [--retries 5] [--retry-wait 2]
```

1. **Fetch the log.**
   - Read `tests/token`.
   - `GET {sim}/episode_log` with `Authorization: Bearer <token>`, with retries.
   - Unreachable after the retries → print `INFRA: sim unreachable`, **exit 1**, write
     nothing.
2. **Parse and validate.** It must match PRD §7.9:
   - `scenario_hash == content_hash(tests/scenario.json)`;
   - `rules_version` == the vendored code's.
   - Otherwise print `INFRA: corrupt or foreign episode log`, exit 1, write nothing.
3. **Replay.**
   - `replay(scenario, log)`. A `ReplayMismatch` → print `REPLAY MISMATCH: ...`, exit 1,
     write nothing.
   - Compare the replayed cost with `final_cost` and `final_cost_by_component` (absolute
     tolerance 1e-6). A difference → `REPLAY MISMATCH: cost`, exit 1, write nothing.
4. **Score:**
   ```
   u = (c_greedy − c_agent) / (c_greedy − c_ref)
   reward = min(1.0, max(0.0, u))
   ```
5. **Write outputs:**
   - `reward.txt` = `f"{reward:.6f}\n"`;
   - `details.json` =
     ```
     {scenario_id, rules_version, cost_total, cost_by_component, c_greedy, c_ref,
      score_unclipped: u, reward, above_reference: u > 1.0,
      counts: {gridlocks, crew_timeouts, reversals, refusals, invalid_calls,
               decision_points},
      tool_calls, final_tick}
     ```
   - `action_log.json` (the raw log).
   - **Never** write `reward.json`.
6. Exit 0.

### 2.7 Probe task (`--probe`; for isolation tests only)

A copy of one task whose `solve.sh` runs checks inside the main container and writes
them to `/logs/agent/probe.json`. The verifier copies them into `details.json`. The
checks:

- `id -u`;
- `curl --max-time 5 https://github.com` (must fail);
- `curl http://sim:8000/health` (must be `ok`);
- `curl http://sim:8000/episode_log` with no token and with `Bearer wrong` (must be 403);
- `find / -xdev -name 'scenario*.json'` (must be empty);
- `python3 -c "import railway"` (must fail, before `PYTHONPATH` is set).

If Harbor's layout doesn't allow `/logs/agent`, use what the spike found.

### 2.8 Fallback (only if `Decision: FALLBACK` or the judges' backend is e2b/runloop)

Export with `--single-container`:

- the simulator and the scenario go into the main image under `/opt/sim`, owned by root,
  mode `0700`;
- an entrypoint starts `railway-sim` as root on `127.0.0.1:8000`, then deletes
  `/opt/sim/scenario.json` after load;
- the agent runs as the non-root `agent` user;
- the MCP url is `http://127.0.0.1:8000/mcp`.

It is weaker isolation (DESIGN.md §8.4). Say so in the README. Its tests are T07-F1 to
T07-F3, and they replace the sidecar-specific probes.

### 2.9 Stop-early as a Harbor agent

Only if spike F-CUSTOM == PASS. `railway/harbor/agents.py:StopEarlyAgent` (and
`DoNothingAgent` if there is no built-in no-op):

- it uploads a vendored `railway` (sim, tools, policies) into the container at runtime;
- it runs `StopEarly` via `McpClient`.

The uploaded code is only for this baseline trial, not part of the image.

### 2.10 `make ci`

1. The default unit gate.
2. `harbor run -p datasets/t1-dev-mini -a oracle`.
3. A check that every trial's reward is exactly 1.0.

`datasets/t1-dev-mini/` (3 T1 dev tasks) is committed for CI.

## 3. Scope

**In scope:**

- `railway/harbor/` (`export.py`, `verify.py`, `templates/`, `agents.py`);
- `gen export`;
- `datasets/t1-dev-mini/`;
- `make ci`;
- probe tasks;
- re-running any `BLOCKED` T01 facts.

**Out of scope:**

- The held-out export. It happens in T09, after the weight freeze, because `norm.json`
  depends on the weights.
- Model runs beyond the T07-H8 smoke test.

## 4. Implementation notes and pitfalls

- **Vendoring.**
  - Copy files; don't symlink (Docker build contexts don't follow symlinks out of the
    context).
  - The vendored copies must be byte-identical to the repo's. A test hashes them.
  - Keep the vendored subsets minimal, per §2.3.
- **Pin dependencies in the images** to the exact `uv.lock` versions (export generates
  `requirements.txt` from the lock). A version drift between the sim image and the
  verifier would break the replay equality.
- **Token handling.**
  - Generate the token with `secrets.token_hex(32)` per task, per export.
  - Never print it.
  - It lives only in `sim/token` (sim image) and `tests/token` (uploaded after the
    agent).
- **Never let the verifier write a reward on any path that isn't a clean replay.** A
  crash must be a trial error. Use `set -euo pipefail` in `test.sh`, and have
  `verify.py` write `reward.txt` as its last action.
- **Reward exactly 1.0 for the oracle.** This needs the oracle's MCP run to produce the
  same cost as `c_ref`, which was computed in-process. T06-13 parity proves it in
  general; T07-H1 proves it per task.

## 5. Tests (write these first)

### 5.1 `tests/t07/test_export.py` (exports 2 T1 dev tasks into `tmp_path`)

| Id | Test | Assertion |
|---|---|---|
| T07-01 | `test_T07_01_layout` | Every path in §2.3 exists for each task; `manifest.json` lists both tasks with `scenario_id`, `scenario_hash`, `c_greedy`, `c_ref` |
| T07-02 | `test_T07_02_norm_values` | `norm.json` equals the accepted-list entry and a fresh in-process recomputation (greedy and reference via `InProcClient`) |
| T07-03 | `test_T07_03_main_context_clean` | Under `environment/` **excluding** `sim/`: no file contains the scenario id, the scenario hash or the token; there is no `scenario*.json`; the `Dockerfile` has no `COPY`/`ADD` of `sim`, `tests`, `solution`, `railway` or `token` |
| T07-04 | `test_T07_04_sim_and_tests_consistent` | `sim/scenario.json` == `tests/scenario.json` (bytes); `sim/token` == `tests/token`; the token is 64 hex characters and differs between the 2 tasks |
| T07-05 | `test_T07_05_vendored_subsets` | `sim/railway` has exactly the subpackages of §2.3 (no `gen`, `ref`, `policies`, `harbor`, `loop`, `eval`, `train`); `tests/railway` has `util`, `sim`, `tools` only; `solution/railway` has no `gen`; every vendored file is byte-identical to the repo file |
| T07-06 | `test_T07_06_instruction` | `instruction.md` equals the §2.4 template with `TOOL_BUDGET` interpolated; ≤ 2,500 characters; contains no scenario id, hash or seed |
| T07-07 | `test_T07_07_task_toml` | `tomllib` parses it; MCP server `railway`, `streamable-http`, `http://sim:8000/mcp`; `network_mode == "allowlist"`; `allowed_hosts` ⊇ {`sim`} ∪ D18 defaults ∪ spike hosts; `--allow-host x.example` adds `x.example`; `--no-default-hosts` drops the D18 list; no `@` character anywhere (no emails) |
| T07-08 | `test_T07_08_requirements_pinned` | Every line of `sim/requirements.txt` is `name==version` and matches `uv.lock` |
| T07-09 | `test_T07_09_export_deterministic` | Exporting twice → identical trees except the `token` files |
| T07-10 | `test_T07_10_fresh` | `--fresh 2` writes `<out>-fresh/` with 2 tasks whose seeds (in the manifest) are ≥ 2,000,000; `git check-ignore` reports the directory as ignored |

### 5.2 `tests/t07/test_verifier.py`

These tests run a `railway-sim` subprocess with a task's `sim/` files, play a policy via
`McpClient`, then run `verify.py --sim-url http://127.0.0.1:<port> --logs-dir tmp` with
the task's `tests/` folder.

| Id | Test | Assertion |
|---|---|---|
| T07-11 | `test_T07_11_oracle_scores_one` | Reference → `reward.txt == "1.000000\n"`; `details.score_unclipped == 1.0` |
| T07-12 | `test_T07_12_greedy_scores_zero` | Greedy → `"0.000000\n"`; `score_unclipped == 0.0` |
| T07-13 | `test_T07_13_do_nothing_and_stop_early_zero` | No calls at all → `"0.000000\n"`; StopEarly → `"0.000000\n"` |
| T07-14 | `test_T07_14_details_shape` | `details.json` keys exactly as in §2.6; `action_log.json` equals the fetched log; **no** `reward.json` exists |
| T07-15 | `test_T07_15_reward_math` (unit) | `reward_from(1000,500,400) == 1.0` with `above_reference True` and `score_unclipped == 1.2`; `(1000,500,1200) → 0.0`; `(1000,500,750) → 0.5` |
| T07-16 | `test_T07_16_sim_unreachable` | No server → exit code ≠ 0; stdout has `INFRA: sim unreachable`; `reward.txt` does not exist |
| T07-17 | `test_T07_17_corrupt_log` | A fake server returns non-JSON, then valid JSON with the wrong `scenario_hash`, then the wrong `rules_version` → each: exit ≠ 0, `INFRA: corrupt or foreign episode log`, no `reward.txt` |
| T07-18 | `test_T07_18_replay_mismatch` | A fake server returns a real log with `final_cost` + 10 → exit ≠ 0, `REPLAY MISMATCH: cost`, no reward; a log with one status flipped → `REPLAY MISMATCH`, no reward |
| T07-19 | `test_T07_19_wrong_token_is_infra` | `tests/token` altered → the server answers 403 → exit ≠ 0, no reward |

### 5.3 `tests/t07/test_harbor_runs.py` (marker `harbor`)

| Id | Test | Assertion |
|---|---|---|
| T07-H1 | `test_T07_H1_oracle_all_tasks` | `harbor_run("datasets/t1-dev-mini", "oracle")` → 3 trials, none errored, every reward == 1.0 |
| T07-H2 | `test_T07_H2_nop_zero` | Built-in no-op agent (name from the spike) → every reward == 0.0, none errored |
| T07-H3 | `test_T07_H3_stop_early_zero` | `StopEarlyAgent` (§2.9) → every reward == 0.0. If F-CUSTOM was FAIL: `pytest.skip` with the spike reference, recorded in the gate |
| T07-H4 | `test_T07_H4_probe_isolation` | Probe task with the oracle → `details.probe` shows: `id -u ≠ 0` (if agent user is set), github unreachable, sim health ok, both `episode_log` attempts 403, no scenario file found, `import railway` fails |
| T07-H5 | `test_T07_H5_image_contents` | Build the main image; `docker run --rm <img> sh -c "find / -xdev \( -name 'scenario*.json' -o -name token \) 2>/dev/null; python3 -c 'import railway'"` → no file paths printed and the import fails; `grep -rl <scenario_id> / --exclude-dir={proc,sys,dev}` finds nothing |
| T07-H6 | `test_T07_H6_sim_down_is_error` | A task copy whose compose has no `sim` → the trial is errored or the reward is missing, **never** 0.0 |
| T07-H7 | `test_T07_H7_t2_oracle` | Export 1 T2 dev task; oracle → 1.0 (proves the T2 runtime fits the timeouts) |
| T07-H8 | `test_T07_H8_opencode_smoke` (also `model`) | OpenCode + a cheap model on 1 T1 task → not errored; 0 ≤ reward ≤ 1; `details.tool_calls > 0` |

### 5.4 Fallback-only tests (`tests/t07/test_fallback.py`; skipped unless `--single-container` is built)

| Id | Test | Assertion |
|---|---|---|
| T07-F1 | `test_T07_F1_oracle` | Single-container oracle → 1.0 on 3 tasks |
| T07-F2 | `test_T07_F2_agent_cannot_read_sim` | The probe as `agent`: `/opt/sim` is unreadable, `scenario.json` is gone after start, the token is unreadable |
| T07-F3 | `test_T07_F3_nop_zero` | No-op → 0.0 |

## 6. Gate commands

```bash
uv run pytest tests/t01 tests/t02 tests/t03 tests/t04 tests/t05 tests/t06 tests/t07 -m "not harbor and not model and not gpu"
```
```bash
uv run pytest tests/t07 -m "harbor and not model"
```
```bash
uv run pytest tests/t07 -m "harbor and model"
```
```bash
make ci
```

## 7. Definition of done

- [ ] All tests pass, including every `harbor` test (T07-H8 needs model access; if it is
      still unavailable, the gate record says so and T09 runs it first).
- [ ] Any T01 facts marked `BLOCKED` are re-run and recorded as PASS or FAIL in
      `docs/spike-report.md`.
- [ ] `datasets/t1-dev-mini/` exists and `make ci` passes.
- [ ] The README draft (finished in T09) has a "Run the judges' checks" block with the
      exact `harbor run` commands for oracle, no-op and stop-early.
- [ ] The user has been told about the two open organizer questions (backend, provider
      host) if they are still unanswered.
- [ ] Gate record, reviewers, handover (no commit).
