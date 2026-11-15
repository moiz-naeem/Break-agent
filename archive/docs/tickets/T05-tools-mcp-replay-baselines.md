# T05: Tool layer, MCP server, episode log, replay, baseline policies

| | |
|---|---|
| Depends on | T04 (manual, generator) |
| Unblocks | T06 (reference plays through the client), T07 (sim container, verifier replay), T08 (model loop) |
| DESIGN.md | §5 (tools, decision points, budgets, compact status), §6.4 (baselines), §6.5 (malformed commands, direct sim access), §8.3 (episode log, replay) |
| PRD | §7.8 (tools), §7.9 (episode log), §8 D11–D13, D15, D19; spike report (T01) for transport details |
| Required markers | default only |
| Suggested split | A: `Session` + formats + stringline. B: MCP server + `McpClient` + `/episode_log` + replay. C: policies (do-nothing, stop-early, greedy, random-legal) + calibration |

## 1. Goal

Expose the simulator as the 15 tools of PRD §7.8, behind one code path used by everyone:

- the MCP server (agents, oracle);
- the in-process client (our loop, training);
- the replay (verifier).

Ship the scripted baselines that anchor the score (greedy) and that the judges will run
(do-nothing, stop-early).

**One code path is the integrity guarantee.** The same `Session.call()` produces every
output, every log entry and every cost, whoever calls it.

## 2. Context the implementer needs

### 2.1 `Session` (`railway/tools/session.py`)

```python
class Session:
    def __init__(self, scenario: Scenario, *, budget: int = TOOL_BUDGET,
                 diagnostics: bool = False): ...
    def call(self, tool: str, args: dict) -> str: ...  # the ONLY entry point; never raises for bad input
    def close(self) -> FinalResult: ...                # runs end_shift if not ended; idempotent
    log: list[LogEntry]                                # PRD §7.9 entry shape
    closed: bool
    final: FinalResult | None                          # {tick, cost: CostBreakdown, counts}
```

- **Tool registry:** a dict of name → (pydantic args model, handler). `extra="forbid"`
  on every args model.
- **Arguments:**
  - unknown tool, missing, extra or wrong-typed args → `REFUSED R-BAD-ARGS: <detail>`,
    logged with status `INVALID`;
  - `R-UNKNOWN-*` refusals are also `INVALID`;
  - other rule refusals are `REFUSED`.
- **Statuses:** mutating successes are `GRANTED` (authorize) or `OK`. Read-only tools,
  `advance_to_next_event` and `end_shift` are `INFO`.
- **Budget:**
  - every call to `call()` counts, before it runs;
  - the call that makes `used == budget` runs normally, then the session runs
    `end_shift` and appends `\nBUDGET EXHAUSTED (R-BUDGET): shift ended. Final cost
    <total>`;
  - later calls return `REFUSED R-SHIFT-ENDED: shift is over. Final cost <total>` and
    are logged (status `REFUSED`) but don't count.
- **Notes:** `notes(text)` replaces the scratchpad. More than `NOTES_MAX` characters is
  refused with `R-NOTES-TOO-LONG`. The scratchpad shows in `status()`, both forms.
- **Diagnostics:** `diagnostics=False` (official) makes `conflicts` and `what_if` (added
  in T09) return `REFUSED R-DIAGNOSTIC-ONLY`. They are not listed by the MCP server in
  official mode.

### 2.2 Output formats (`railway/tools/formats.py`)

- **`network()`** — JSON:
  ```
  {rules_version, shift_ticks, factors, yards,
   stations: [{id, major, crew_station, loops: [{id, length_m, kind}]}],
   sections: [{id, west, east, length_m, line_speed_kmh}],
   crew_stations: [...]}
  ```
- **`timetable(train=None)`** — JSON:
  ```
  {trains: [{id, type, direction, pax, length_m, max_kmh, entry_tick, stops,
             next_service, crew_used_at_entry}]}
  ```
  - With `train`: one train, or `REFUSED R-UNKNOWN-TRAIN`.
  - `crew_used_at_entry` only appears if F10 is on.
- **Rebuilding `StaticInfo`.** Together, `network()` + `timetable()` must rebuild
  `StaticInfo` exactly (`StaticInfo.from_tool_json(network_json, timetable_json)`).
- **`status(compact=False)`** — `State` as JSON (`model_dump(mode="json")`, canonical),
  with `notes`.
- **`status(compact=True)`** — PRD §7.8.1 exactly. Train line grammar:
  `<id padded 4> <type padded 9> <position> auth:<dest|-> delay:<+n|0>[ crew:<remaining>][ next:<station>@<arr>]`.

  | Phase | `<position>` |
  |---|---|
  | `at_station` | `at <station>/<track>-<platform\|goods\|main>` (yards: `at YW`) |
  | `in_section` | `<section> <eastbound\|westbound>` |
  | `reversing` | `<section> reversing` |
  | `not_entered` | `waiting <yard> entry@<tick>` (or `waiting <yard> after <pred>`) |

  - The header is `t=<tick> | gridlock: <none|ids comma-separated> | notes: "<notes>"`.
  - The footer is `restrictions: ...` (possessions then speed restrictions, `;`-joined),
    or `restrictions: none`.
- **`bulletins()`** — JSON list of revealed events (as in the scenario, plus
  `reveal_tick`), sorted by `(reveal_tick, id)`. Breakdowns appear from their tick.
  Skipped breakdowns never appear.
- **`kpis()`** — JSON:
  ```
  {tick,
   cost: {pax_lateness, freight_lateness, crew_timeout, reversal, unfinished, total},
   counts: {gridlocks, crew_timeouts, reversals, refusals, invalid_calls, decision_points},
   budget: {used, left, total}}
  ```
- **`advance_to_next_event()`** — text:
  ```
  t=<tick> stop: <KIND> <train>, <KIND> <train>, ...
  changes since t=<prev>:
    t=<tick> <one change per line, in processing order>
  budget: <used>/<total>
  ```
  - Only changes since the previous advance.
  - `stop: SHIFT_END` with the final cost line when the shift is over.
- **`end_shift()`** — `SHIFT ENDED at t=<shift>. Final cost <total> (pax <..>, freight
  <..>, crew <..>, reversal <..>, unfinished <..>)`.
- **`stringline(from_tick, to_tick)`:**
  - **Window checks:** valid iff `0 ≤ from_tick ≤ to_tick ≤ shift_ticks` and
    `to − from ≤ 60`, otherwise `R-WINDOW`.
  - **Columns** alternate place and section: `YW YW-S1 S1 S1-S2 ... YE`, each cell 4
    characters wide. The header row shortens section names to `-`.
  - **Rows:** one per tick, with `step = ceil((to−from+1)/30)`, increased until the
    output is ≤ 2000 characters.
  - **Cells:** a cell shows the occupying train id. At a station holding more than one
    train it shows the lowest id + `+`. `.` means empty.
  - **Past and present rows** (`≤ tick`) come from recorded history.
  - **Future rows** come from a belief-simulator projection of the **current
    authorities only** (no new actions), so they never use unrevealed information.
  - **Current-tick row** is prefixed `>`.

### 2.3 Clients (`railway/tools/client.py`)

```python
class Client(Protocol):
    def call(self, tool: str, args: dict) -> str: ...
class InProcClient:  # wraps a Session
class McpClient:     # streamable-http to http://host:port/mcp; persistent connection; sync API
```

Policies talk only to a `Client`.

`railway/tools/observe.py::Observer(client)`:

- caches `network()` + `timetable()` (2 calls, once);
- `observe() -> (StaticInfo, State)` calls `status()` once.

The belief simulator is `Simulator.from_observation(static, state)`.

### 2.4 MCP server (`railway/server/`, console script `railway-sim`)

```
railway-sim --scenario scenario.json --token-file token --host 0.0.0.0 --port 8000
```

- One `Session` per process. Tool calls are serialized with an `asyncio.Lock`.
- **Tools.** The 15 tools, with typed parameters:
  - `authorize(train: str, station: str, track: str)`;
  - `hold(train: str, until_tick: int)`;
  - `status(compact: bool = False)`;
  - `stringline(from_tick: int, to_tick: int)`;
  - `notes(text: str)`;
  - `timetable(train: str | None = None)`;
  - `cancel_authority`, `relieve_crew` and `reverse` take `(train: str)`;
  - the rest take no arguments.

  Each docstring is one line ending with "See manual()."
- **Schema rejections.** Calls that MCP schema validation rejects never reach the
  `Session`: they take no budget and leave no log (document this in the manual's Tools
  section).
- **Routes:**
  - `GET /health` → `ok`.
  - `GET /episode_log` with `Authorization: Bearer <token>`:
    - closes the session (if not closed);
    - returns the PRD §7.9 JSON;
    - wrong or missing token → 403.
  - Any other path → 404. Other methods on `/episode_log` → 405.
- **Startup:**
  - refuses to start (exit 2) if the scenario's `rules_version` ≠ the code's;
  - logs nothing to stdout that contains scenario data.

### 2.5 Replay (`railway/tools/replay.py`)

`replay(scenario, log_json) -> FinalResult`:

1. Make a fresh `Session` and `call(entry.tool, entry.args)` for every entry in order.
2. Raise `ReplayMismatch` if any recomputed `status` or `rule` differs from the logged
   one.
3. `close()`.
4. Return the final result.

The verifier (T07) compares this with the live `final_cost`.

### 2.6 Baseline policies (`railway/policies/`)

`Policy.run(client: Client) -> None` plays a whole episode, then returns. The runner
calls `session.close()`.

| Policy | Exact behaviour |
|---|---|
| `DoNothing` | Makes **no calls** |
| `Greedy` (the score anchor, D15) | **Setup:** fetch `network` and `timetable` once. **Each decision point:** (1) `status()`. (2) Candidates: trains in the network, unfinished, `at_station`, no authority, not held. (3) Order by `(entry_tick, id)`. (4) For each candidate, scan stations ahead in order to the destination yard. The first station with a free, fitting, type-allowed **loop** (lowest loop id first) whose authorization the belief simulator `check()` grants gets `authorize`. The destination yard counts as a free loop. **Never:** uses `main` as a destination, holds, relieves crew, reverses or extends authorities. (5) Then call `advance_to_next_event()`. **Stop** when the advance output says `SHIFT_END` |
| `StopEarly` | Greedy until the first decision point with `tick ≥ 30`, then `end_shift()` |
| `RandomLegal(seed)` | Same candidate set, ordered by id. Skip a train with probability 0.3; otherwise pick uniformly among all `check()`-granted `(station within 3 ahead or the destination yard, track ∈ {main} ∪ loops)`. Uses `random.Random(seed)`; never issues a refused call |

Runner: `run_policy(policy, scenario, *, transport="inproc"|"mcp") -> EpisodeResult`
with `{final, log, decision_points, tool_calls, wall_s}`.

## 3. Scope

**In scope:**

- `tools/` (session, formats, stringline, client, observe, replay);
- `server/`;
- `policies/` (the four above);
- the `railway-sim` script;
- `StaticInfo.from_tool_json`.

**Out of scope:**

- The reference and exploits (T06).
- Harbor files (T07).
- Model prompts (T08).
- `conflicts` / `what_if` (T09).

## 4. Implementation notes and pitfalls

- **Policies must use only the client.** They must not import the scenario, the
  simulator internals or the session. A test enforces this by running every policy over
  `McpClient` against a server subprocess.
- **Budget economy.** Greedy should use about 2 + (1 status + k authorizes + 1 advance)
  per decision point. Reading `status()` once per decision point is the convention.
- **History for stringline.** Record movements from `AdvanceResult.changes` in the
  session. Don't add history to `State`: it must stay compact.
- **FastMCP return types.** Return `str` from every tool. JSON outputs are strings of
  canonical JSON, so MCP and in-process outputs are byte-identical.
- **Windows.** `McpClient` tests start the server with `sys.executable -m railway.server`
  on `127.0.0.1` and a free port. Wait for `/health` before connecting.

## 5. Tests (write these first)

### 5.1 `tests/t05/test_session.py`

| Id | Test | Assertion |
|---|---|---|
| T05-01 | `test_T05_01_all_tools_registered` | The registry names are exactly the 15 of PRD §7.8 |
| T05-02 | `test_T05_02_bad_args` (parametrized) | Unknown tool `teleport`; `authorize` missing `track`; extra arg; `hold(until_tick="x")` → output starts `REFUSED R-BAD-ARGS`, status `INVALID`, budget used +1, `state()` unchanged |
| T05-03 | `test_T05_03_status_classification` | Episode A of T02 through `call()`: statuses are `GRANTED`, `INFO`, `REFUSED`, ... in the expected places; `kpis.counts.refusals == 1`, `invalid_calls == 0` |
| T05-04 | `test_T05_04_budget_exhaustion` | `budget=5`: 4 × `kpis`, then `advance_to_next_event` → the 5th output contains `BUDGET EXHAUSTED (R-BUDGET)`; `session.closed`; the 6th call → `REFUSED R-SHIFT-ENDED`; final cost == do-nothing cost after the one advance (1235.0 on mini3) |
| T05-05 | `test_T05_05_close_equals_end_shift` | Two sessions with the same calls; one ends with `end_shift`, the other with `close()` → equal final cost and tick |
| T05-06 | `test_T05_06_notes` | `notes("x"*1500)` OK; `notes("x"*1501)` → `R-NOTES-TOO-LONG`; `notes("a")` then `notes("b")` → status shows `"b"` only (replace, not append); compact header shows `notes: "b"` |
| T05-07 | `test_T05_07_after_close` | After `close()`, any tool → `REFUSED R-SHIFT-ENDED` with the final cost |

### 5.2 `tests/t05/test_formats.py`

| Id | Test | Assertion |
|---|---|---|
| T05-08 | `test_T05_08_compact_status_grammar` | Episode A at t=6: the compact status header matches `^t=6 \| gridlock: none \| notes: ""$`; lines for E1, F21, L5 match the grammar regex; E1 line contains `at S2/L1-platform` and `auth:-`; L5 line contains `at S3/main` (station `main` has no suffix) and `auth:S2/main`; F21 (entered at 2, never authorized) contains `at YW` and `auth:-`; the last line is `restrictions: none` |
| T05-09 | `test_T05_09_full_status_roundtrip` | `State.model_validate_json(status())` equals `sim.state()` plus notes |
| T05-10 | `test_T05_10_static_from_tool_json` | `StaticInfo.from_tool_json(network(), timetable())` == `sim.static_info()` for mini3 and for 5 T2 dev scenarios |
| T05-11 | `test_T05_11_advance_changes_only` | Episode A: the second advance's output contains no change line with `t=` < its `since` tick; each output ends with `budget: n/400` |
| T05-12 | `test_T05_12_kpis_shape` | `kpis()` JSON has exactly the keys of §2.2; cost total equals the sum |
| T05-13 | `test_T05_13_bulletins` | T03-29 scenario: `bulletins()` is `[]` at t=2 and has 1 late-entry event from t=10; a skipped breakdown (T03-25) never appears |
| T05-14 | `test_T05_14_stringline_limits` | `(0,61)` → `R-WINDOW`; `(10,5)` → `R-WINDOW`; `(0,60)` on a T2 scenario ≤ 2000 characters; `(0,10)` has 11 rows (step 1) |
| T05-15 | `test_T05_15_stringline_content` | Episode A after step 9 (t=11, E1 authorized to YE): `stringline(0,20)` has 21 rows (step 1); row `t3` shows `E1` in the `S1-S2` column (E1 enters it at 3); row `t6` shows `E1` in the `S2` column; future row `t15` shows `E1` in the `S3-YE` column (projection of its current authority: S2-S3 11–14, S3-YE 14–17); the `t11` row starts with `>` |
| T05-16 | `test_T05_16_tool_outputs_no_leak` (Hypothesis) | T03-36 extended: for scenario pairs X/Y differing in one unrevealed event, the output **strings** of every tool (incl. stringline, compact and full status, bulletins, kpis, advance, authorize) are identical while `tick < reveal` |

### 5.3 `tests/t05/test_mcp_server.py` (server subprocess on mini3)

| Id | Test | Assertion |
|---|---|---|
| T05-17 | `test_T05_17_lists_15_tools` | The MCP `list_tools` names == the registry names; no `conflicts` / `what_if` |
| T05-18 | `test_T05_18_parity_with_inproc` | Episode A's call sequence via `McpClient` and via `InProcClient` → identical output strings, call by call |
| T05-19 | `test_T05_19_episode_log_token` | No token → 403; wrong token → 403; right token → 200, JSON per PRD §7.9, `closed: true`, `final_cost == 519.0` after episode A; a later MCP `status` call → `R-SHIFT-ENDED` |
| T05-20 | `test_T05_20_routes` | `/health` → 200; `/`, `/scenario`, `/state` → 404; `POST /episode_log` → 405 |
| T05-21 | `test_T05_21_rules_version_mismatch` | A scenario file with a different `rules_version` → the server exits with code 2 |
| T05-22 | `test_T05_22_nothing_secret_on_stdout` | Server stdout/stderr during an episode contain no train id from the scenario's events and not the token |

### 5.4 `tests/t05/test_replay.py`

| Id | Test | Assertion |
|---|---|---|
| T05-23 | `test_T05_23_replay_equals_live` (Hypothesis) | Random call sequences (valid and invalid, all tools) on the all-factor mini3 variant: `replay(scenario, log).cost == session.final.cost` |
| T05-24 | `test_T05_24_replay_detects_tamper` | Flip one logged status `GRANTED → REFUSED` → `ReplayMismatch`; change logged args of a granted authorize so it's refused → `ReplayMismatch` |
| T05-25 | `test_T05_25_replay_empty_log` | An empty log → do-nothing cost |

### 5.5 `tests/t05/test_policies.py`

| Id | Test | Assertion |
|---|---|---|
| T05-26 | `test_T05_26_greedy_mini3_exact` | Greedy on mini3 → `total == 26.0` (hand trace: L5 late 8 at S2 → 6, at S1 → 6, at YW → 12 terminus; F21 late 4 → 2; E1 on time); every train finished; zero refusals |
| T05-27 | `test_T05_27_do_nothing` | DoNothing on mini3 → 0 tool calls; total 1235.0 |
| T05-28 | `test_T05_28_stop_early` | On a T2 dev scenario: the log's last call is `end_shift` at the first decision tick ≥ 30; no call after it |
| T05-29 | `test_T05_29_random_legal` | Same seed → identical logs; different seeds → different logs; `refusals == invalid_calls == 0` |
| T05-30 | `test_T05_30_policies_over_mcp` | Every policy over `McpClient` gives the same final cost as in-process (mini3 + 2 T2 dev scenarios) |
| T05-31 | `test_T05_31_policies_import_hygiene` | AST scan of `railway/policies/*.py`: no import of `railway.tools.session`, `railway.sim.model.Scenario` or `railway.gen` |

### 5.6 `tests/t05/test_calibration.py` (`slow`)

| Id | Test | Assertion |
|---|---|---|
| T05-32 | `test_T05_32_t2_decision_points` | Greedy on 10 T2 dev seeds: mean decision points per episode in [30, 120] (DESIGN target is ~60) |
| T05-33 | `test_T05_33_t2_output_size` | Same runs: mean of Σ(advance output characters + one compact status per decision point) ≤ 250,000 characters (≈ 62k tokens; a proxy for < 100k tokens per episode) |
| T05-34 | `test_T05_34_t2_greedy_speed` | Greedy T2 episode in-process < 3 s; over MCP < 30 s |
| T05-35 | `test_T05_35_t2_greedy_budget` | Greedy uses < 70% of `TOOL_BUDGET` on every T2 dev seed tested |

If T05-32 or T05-33 fails, tune the generator defaults (T04 `GenConfig`). Never loosen
the tests. Keep T04's tests green and record the tuning in the gate record.

## 6. Gate command

```bash
uv run pytest tests/t01 tests/t02 tests/t03 tests/t04 tests/t05 -m "not harbor and not model and not gpu"
```

## 7. Definition of done

- [ ] All tests pass.
- [ ] `railway-sim --scenario <T2 file>` runs and an MCP inspector, or `McpClient` in a
      REPL, can play a few turns.
- [ ] The manual's Tools section matches the implemented formats exactly. Update T04's
      manual text if formats were refined here; T04 tests stay green.
- [ ] Gate record (it includes the calibration numbers), reviewers, handover (no
      commit).
