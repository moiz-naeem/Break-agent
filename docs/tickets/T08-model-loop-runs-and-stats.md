# T08: Model loop, run manifests, results JSONL, statistics

| | |
|---|---|
| Depends on | T07 (datasets, verifier, `reward_from`); T06 (reference `decide`); T05 (Session) |
| Unblocks | T09 (pilot, held-out evaluation, report), T10 (training data uses the same prompts) |
| DESIGN.md | §9.3 (manifests, defaults, JSONL), §9.4 (two ways to play), §9.5 (models), §6.2 (aggregation) |
| PRD | §7.10 (results row), §8 D16; §9 (models are an open item) |
| Required markers | default; `model` is optional (one smoke test) |
| Suggested split | A: `model_client.py` + `fake_model.py` + `prompts.py`. B: `loop/runner.py` (episode loop, strict format, windowed/full). C: `eval/manifest.py`, `eval/run.py` (loop + Harbor harnesses), `eval/stats.py` |

## 1. Goal

Make any model playable and measurable **by config**:

- **Our own loop** for diagnostics and training: identical prompts for every model, a
  strict one-tool-call-per-turn format, windowed (main) and full-history modes.
- **Run manifests** (`runs/<name>.yaml`) for both harnesses (loop and Harbor/OpenCode),
  writing one JSONL row per episode.
- **Statistics**:
  - ratio-of-sums score;
  - mean clipped reward;
  - unclipped mean;
  - cost ratio;
  - bootstrap 95% CI by resampling seeds;
  - flagged seeds.

**No real model is required to pass this ticket.** A fake OpenAI-compatible server
drives the tests.

## 2. Context the implementer needs

### 2.1 Model client (`railway/loop/model_client.py`)

```python
class ModelClient:
    def __init__(self, base_url: str, model: str, api_key_env: str | None,
                 temperature: float | None = None, max_tokens: int = 1024,
                 timeout_s: float = 120, max_retries: int = 5): ...
    def chat(self, messages: list[dict], tools: list[dict]) -> ModelReply: ...
# ModelReply = {tool_calls: [{id, name, arguments_raw: str}], content: str|None,
#               usage: {prompt_tokens, completion_tokens}}
```

- **Protocol:** OpenAI-compatible `POST {base_url}/chat/completions` with `tools` and
  `tool_choice: "auto"`.
- **Temperature:** `None` omits the field (the model's default; DESIGN.md §9.3).
- **Retries:**
  - 429 and 5xx retry on a deterministic backoff of 1, 2, 4, 8 and 16 s;
  - other 4xx raise immediately;
  - a final failure raises `ModelError`, and the runner records the episode as
    `status: "error"` (an infra error, not a score).
- **API keys** come from env vars only. Never log them. Never write them into results.

### 2.2 Tool schemas

`railway/tools/schemas.py::tool_schemas(disabled=()) -> list[dict]` builds OpenAI
function specs from the Session registry's pydantic arg models.

- Names and one-line descriptions are identical to the MCP server's.
- A disabled tool (e.g. `stringline` for the ablation) is omitted from the schemas.
- A disabled tool is also refused by the session (`R-BAD-ARGS: tool disabled`).

### 2.3 Prompts (`railway/loop/prompts.py`)

Messages are rebuilt every turn. The system message is the T07 `instruction.md` text
plus: "Reply with exactly one tool call per message. Do not write prose outside tool
calls."

| Mode | Messages |
|---|---|
| `windowed` (main) | `system`; `user: "MANUAL\n" + manual()`; the **last 5 turns** (each = the assistant tool-call message + its tool-result message); `user: CURRENT` |
| `full` | Same, but **all** turns |

`CURRENT` is:

```
CURRENT STATE
<status(compact=True)>
BULLETINS
<bulletins()>
NOTES
<notes>
Budget used: <used>/<total>. Call exactly one tool.
```

- `CURRENT` is produced with `Session.peek(tool, args)`, a new **read-only** method:
  - it is allowed only for `status`, `bulletins`, `kpis` and `manual`;
  - it doesn't count against the budget;
  - it isn't logged;
  - it raises for mutating tools.
- **The windowed context therefore never grows with episode length**, which is the
  point of DESIGN.md §9.4.
- **Our-loop numbers are diagnostic.** The headline numbers come from Harbor/OpenCode,
  where the agent pays for its own `status` calls. Write that in the report legend
  (T09).

### 2.4 Episode loop (`railway/loop/runner.py`)

`play_episode(scenario, model_client, mode, *, disabled=(), max_turns=TOOL_BUDGET+20) ->
EpisodeRecord`:

- **Each turn:**
  1. Build the messages.
  2. `chat`.
  3. Exactly one tool call → `session.call(name, parsed_args)`.
     - If the args aren't valid JSON, pass `{"__raw__": arguments_raw}`, which the
       session refuses with `R-BAD-ARGS`. That counts against the budget.
     - Record the turn.
  4. **Zero tool calls:**
     - `format_errors += 1`;
     - no session call;
     - the next CURRENT starts with `ERROR: reply with exactly one tool call.`;
     - 3 consecutive zero-call replies → stop.
  5. **More than one tool call:** execute only the first, `format_errors += 1`.
- **The episode ends** when the session is closed (`end_shift`, budget) or after
  `max_turns`, or on the 3-strikes stop. It then calls `session.close()` (stopping
  counts as `end_shift`).
- **The record holds** everything in PRD §7.10:
  - tokens summed from `usage`;
  - `decision_points` from kpis;
  - plus `format_errors`;
  - plus the turn transcript, saved to `results/<run>/transcripts/<scenario_id>-r<rep>.jsonl`.

### 2.5 Fake model (`railway/loop/fake_model.py`)

A local HTTP server implementing `/v1/chat/completions` (tool calls + usage), started in
tests on a free port. Behaviours:

| Behaviour | Replies |
|---|---|
| `scripted(calls)` | Returns the given tool calls in order, then `end_shift` |
| `policy(decide_fn, session)` | Test-only side channel. Keeps a queue; when empty, refills from `decide_fn(static, state)` via `session.peek`; returns one call per reply |
| `prose` | Returns text with no tool call (tests the 3-strikes rule) |
| `double` | Returns two tool calls per reply |
| `flaky(n)` | Returns HTTP 500 for the first n requests, then behaves like `scripted` |

`decide_fn` must be a pure `(static, state) -> list[ToolCall]`.

- `Reference.decide` exists (T06).
- Add `Greedy.decide(static, state)` to the T05 greedy policy as a refactor: `run()`
  then loops observe → `decide` → execute. T05 tests must stay green.

### 2.6 Run manifests (`railway/eval/manifest.py`)

```yaml
name: pilot-t1-small            # must equal the file stem
harness: loop                   # loop | harbor
mode: windowed                  # windowed | full   (loop only)
model:
  id: small                     # label used in tables and charts
  name: Qwen/Qwen2.5-7B-Instruct
  base_url: http://localhost:8001/v1      # loop only
  api_key_env: SMALL_API_KEY              # loop only, optional
  harbor_model: openrouter/qwen/qwen-2.5-7b-instruct   # harbor only (-m value)
temperature: null               # null = model default
max_tokens: 1024
dataset: datasets/t2-heldout    # EITHER dataset (harbor or loop) ...
scenarios: {preset: T1, set: dev, count: 10}   # ... OR scenarios (loop only)
reps: 3                         # default 3 (DESIGN.md §9.3)
concurrency: 4
tools_disabled: []              # e.g. [stringline]
diagnostics: false              # true only for T09 diagnostic runs
```

Validation: pydantic, `extra="forbid"`; exactly one of `dataset`/`scenarios`; harness
`harbor` requires `dataset` and `harbor_model`; `name` == file stem.

### 2.7 Runner (`railway/eval/run.py`; console script `railway-eval`)

| Command | Effect |
|---|---|
| `railway-eval run runs/<name>.yaml` | Runs every (scenario, rep); appends rows to `results/<name>.jsonl`. **Resumable:** skips (scenario_id, rep) pairs already present with `status: ok`, and re-runs error rows |
| `railway-eval summarize results/<name>.jsonl [--json]` | Prints the stats table |
| `railway-eval compare results/a.jsonl results/b.jsonl` | Paired difference with CI (used by T10) |

**Loop harness.**

- Scenarios come from the dataset's `tests/scenario.json` files, or from
  `accepted_seeds(...)`.
- `c_greedy` and `c_ref` come from `norm.json` or the accepted list.
- Reward uses the T07 `reward_from` (the same function the verifier uses).

**Harbor harness.**

- Runs `harbor run -p <dataset> -a opencode -m <harbor_model>` with the repetition flag
  from the spike report.
- Reads each trial's `details.json` (T07) into a row (`harness: "harbor"`, tokens if
  Harbor reports them, else null).
- An errored trial becomes an `error` row.
- The subprocess call goes through a function the tests can monkeypatch.

### 2.8 Statistics (`railway/eval/stats.py`)

Rows with `status != "ok"` are excluded and counted. If more than 10% of the planned
episodes errored, the summary is marked `incomplete: true`.

**Per seed** s: `ca_s` = mean `cost_total` over the ok reps. Then:

```
score_ros      = Σ_s (cg_s − ca_s) / Σ_s (cg_s − cr_s)                          (headline, D16)
reward_mean    = mean_s ( mean_reps clip((cg − ca)/(cg − cr), 0, 1) )            (what Harbor shows)
unclipped_mean = mean_s ( mean_reps (cg − ca)/(cg − cr) )
cost_ratio     = Σ_s ca_s / Σ_s cr_s
flagged        = [s : any rep with unclipped > 1.0]
ci(stat)       = percentile 95% CI from B = 10,000 bootstrap resamples of SEEDS
                 (with replacement), numpy.random.default_rng(0)
paired_diff(a, b, stat) = stat(b) − stat(a) on common seeds, with a seed-resampled CI
```

The summary also includes:

- means of `gridlocks`, `crew_timeouts`, `invalid_calls`, `format_errors`,
  `tool_calls`, `decision_points`;
- mean tokens;
- the episode count and the error count.

## 3. Scope

**In scope:**

- `loop/` (all);
- `tools/schemas.py`;
- `Session.peek`;
- `Greedy.decide`;
- `eval/manifest.py`, `run.py`, `stats.py`;
- the `railway-eval` script;
- example manifests `runs/example-loop.yaml` and `runs/example-harbor.yaml`.

**Out of scope:**

- The real pilot and evaluation runs (T09).
- The report (T09).
- Training (T10).

## 4. Implementation notes and pitfalls

- **Identical prompts across models.** The prompt builder is a pure function of
  `(manual, turns, current, mode)`. A test pins a golden prompt for a scripted
  2-turn episode.
- **Determinism of our loop** depends on the model. With the fake model it is fully
  deterministic, and the tests rely on that.
- **Concurrency.**
  - One `Session` per episode; nothing shared.
  - Results are appended under a file lock (`fcntl`/`msvcrt`, or a single writer
    thread). Keep it simple: one writer thread fed by a queue.
- **Privacy.** Results and transcripts must contain no API keys and no user email.
  Model names are fine.

## 5. Tests (write these first)

### 5.1 `tests/t08/test_model_client.py` (fake server)

| Id | Test | Assertion |
|---|---|---|
| T08-01 | `test_T08_01_tool_call_roundtrip` | `scripted([authorize E1 S2 L1])` → `chat` returns 1 tool call, name `authorize`, `arguments_raw` parses to the args; usage is parsed |
| T08-02 | `test_T08_02_temperature_omitted` | `temperature=None` → the request body the fake server received has no `temperature` key; `0.7` → present |
| T08-03 | `test_T08_03_retries` | `flaky(2)` → success after 2 retries (the backoff sleep is monkeypatched to record calls: `[1, 2]`); `flaky(10)` → `ModelError` after 5 retries |
| T08-04 | `test_T08_04_no_key_leak` | With `api_key_env` set to a var holding `sk-test-SECRET`, the full run's results and transcripts contain no `SECRET` |

### 5.2 `tests/t08/test_prompts.py`

| Id | Test | Assertion |
|---|---|---|
| T08-05 | `test_T08_05_windowed_shape` | After 8 turns, windowed messages = system + manual + 5 turns × 2 messages + CURRENT = 13 messages; full = 2 + 16 + 1 = 19 |
| T08-06 | `test_T08_06_current_block` | CURRENT contains the compact status, bulletins and notes exactly as the tools would return them, and `Budget used: n/400` |
| T08-07 | `test_T08_07_peek_free` | 50 `peek("status", {})` calls → `budget.used` unchanged, log unchanged; `peek("authorize", ...)` raises |
| T08-08 | `test_T08_08_golden_prompt` | A scripted 2-turn episode on mini3 → messages equal `tests/fixtures/golden_prompt_mini3.json` (written by hand in the test-first phase from the format spec, then frozen) |
| T08-09 | `test_T08_09_tool_schemas` | Names == registry; `tools_disabled=["stringline"]` removes it, and a session call to it is refused |

### 5.3 `tests/t08/test_loop.py`

| Id | Test | Assertion |
|---|---|---|
| T08-10 | `test_T08_10_fake_greedy_equals_greedy` | `policy(Greedy.decide)` through `play_episode` on mini3 and 2 T1 dev scenarios → final cost == `Greedy` via `InProcClient` (mini3: 26.0) |
| T08-11 | `test_T08_11_fake_reference_equals_reference` | `policy(Reference.decide)` on 2 T1 dev scenarios → final cost == `c_ref` from the accepted list |
| T08-12 | `test_T08_12_prose_three_strikes` | `prose` → the episode stops after 3 replies; `format_errors == 3`; the session closes; cost == do-nothing cost |
| T08-13 | `test_T08_13_double_calls` | `double` → only the first call per reply executes; `format_errors == number of replies` |
| T08-14 | `test_T08_14_bad_json_args` | A scripted reply with `arguments_raw = "{oops"` → the session logs `INVALID R-BAD-ARGS`; budget +1 |
| T08-15 | `test_T08_15_record_fields` | The `EpisodeRecord` has every PRD §7.10 field plus `format_errors`; the transcript file exists, one JSON line per turn |
| T08-16 | `test_T08_16_windowed_vs_full_same_actions` | With `policy(Greedy.decide)` the actions and final cost are identical in both modes (the mode changes only what the model sees) |

### 5.4 `tests/t08/test_manifest_and_runner.py`

| Id | Test | Assertion |
|---|---|---|
| T08-17 | `test_T08_17_manifest_validation` (parametrized) | Rejected: both `dataset` and `scenarios`; neither; harness `harbor` without `harbor_model`; unknown key; `name` ≠ file stem |
| T08-18 | `test_T08_18_loop_run_writes_rows` | `runs/tmp.yaml` (loop, fake greedy, `scenarios: T1 dev 3`, `reps: 2`) → 6 rows, all `status: ok`, `score_unclipped == 0.0`, `score_clipped == 0.0` |
| T08-19 | `test_T08_19_resume` | Delete 2 rows and mark 1 row `error` → re-run executes exactly 3 episodes; the final file has 6 ok rows (the error row superseded) |
| T08-20 | `test_T08_20_harbor_harness_parsing` | Monkeypatch the Harbor subprocess to create 3 fake trial dirs (2 with `details.json`, 1 errored) → 2 ok rows with fields mapped from `details.json`, 1 error row |
| T08-21 | `test_T08_21_concurrency_safe` | `concurrency: 4`, 8 episodes → 8 valid JSON lines, no interleaving |

### 5.5 `tests/t08/test_stats.py`

**Fixture rows.** Seed A: `cg=1000, cr=500`, reps `ca = [600, 600]`. Seed B:
`cg=200, cr=100`, reps `ca = [300, 300]`.

| Id | Test | Assertion |
|---|---|---|
| T08-22 | `test_T08_22_ros` | `score_ros == (400 + (−100)) / (500 + 100) == 0.5` |
| T08-23 | `test_T08_23_reward_mean_and_unclipped` | `reward_mean == (0.8 + 0.0)/2 == 0.4`; `unclipped_mean == (0.8 − 1.0)/2 == −0.1` |
| T08-24 | `test_T08_24_cost_ratio` | `cost_ratio == 900/600 == 1.5` |
| T08-25 | `test_T08_25_rep_averaging` | Seed A reps `[500, 700]` → same `score_ros` (0.5) as `[600, 600]` |
| T08-26 | `test_T08_26_flagged` | Add seed C `cg=100, cr=50`, reps `[40, 60]` → `flagged == ["C"]` (rep 40 gives 1.2) |
| T08-27 | `test_T08_27_bootstrap_ci` | 20 synthetic seeds → the CI contains the point estimate; same input → identical CI (rng seed 0); with all seeds identical → a zero-width CI |
| T08-28 | `test_T08_28_errors_excluded` | Add 1 error row out of 5 planned → excluded, `errors == 1`, `incomplete == True` (20% > 10%) |
| T08-29 | `test_T08_29_paired_diff` | a = fixture (`score_ros` 0.5); b = fixture with every `ca` reduced by 100 (A: 500, B: 200 → `(500 + 0)/600`) → `paired_diff(score_ros) == approx(1/3)`. Every bootstrap resample (AA → 0.2, BB → 1.0, AB → 1/3) is positive, so the CI lower bound > 0 |

### 5.6 `tests/t08/test_real_model.py` (marker `model`; optional for this gate)

| Id | Test | Assertion |
|---|---|---|
| T08-M1 | `test_T08_M1_real_model_smoke` | `MODEL_SMOKE_BASE_URL`, `MODEL_SMOKE_NAME` and the key from env; 1 T1 dev episode in windowed mode → `status: ok`, `tool_calls ≥ 5`, 0 ≤ `score_clipped` ≤ 1 |

## 6. Gate command

```bash
uv run pytest tests/t01 tests/t02 tests/t03 tests/t04 tests/t05 tests/t06 tests/t07 tests/t08 -m "not harbor and not model and not gpu"
```

Also run T08-M1 if model access exists.

## 7. Definition of done

- [ ] All tests pass.
- [ ] `runs/example-loop.yaml` and `runs/example-harbor.yaml` exist and validate.
- [ ] `railway-eval summarize` output for T08-18 is pasted in the gate record.
- [ ] Gate record, reviewers, handover (no commit).
