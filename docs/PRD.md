# PRD: Single-Track Railway Dispatcher RL Environment

| | |
|---|---|
| Version | 1.0 (derived from DESIGN.md v0.2) |
| Date | 2026-11-14 |
| Source of truth for *why* | [DESIGN.md](../DESIGN.md) |
| Source of truth for *what to build and how to check it* | this file + [tickets/](tickets/) |
| Team | 3 people (A, B, C) plus AI coding agents |
| Deadline | 48 hours from kick-off |

If this PRD and DESIGN.md disagree, this PRD wins for implementation details, and
section 8 lists every place where it adds a decision DESIGN.md left open. Raise any
disagreement before building, not after.

---

## 1. Problem and product

The AfterQuery "Break the Agent" hackathon asks for an RL environment that frontier
agents find hard, scored by judges on: **gap (30), meaningful difficulty (30), reward
integrity (25), craft (15)**. The tiebreaker is a small model that improves after
training on the environment.

The judges rerun:

- the reference run (Harbor oracle);
- 3 agent runs;
- a do-nothing baseline;
- a stop-early baseline.

Then they spend 15 minutes hunting exploits.

**Product:** a deterministic single-track railway dispatching simulator, exposed as MCP
tools, packaged as one Harbor task per seed, with:

- a scripted reference (the oracle);
- greedy-anchored scoring;
- a benchmarking pipeline (manifests, results, report);
- a post-training pipeline for a small model.

## 2. Users

| User | What they need |
|---|---|
| Judges | `harbor run` works first time; oracle = 1.0; do-nothing and stop-early = 0.0; no exploit pays; rules are discoverable via `manual()` |
| Agents under test (OpenCode + a model) | Clear tools, one call to see full state, refusals that explain themselves |
| The team | Config-driven runs, reproducible numbers, one HTML report |
| The training pipeline | Unlimited labelled states (the reference labels any state); raw costs |

## 3. Goals and success metrics

| # | Metric | Target | Checked in |
|---|---|---|---|
| G1 | Determinism | Same scenario + same tool calls → byte-identical outputs and cost, across processes and machines | T02, T05, T07 |
| G2 | Oracle | Reward exactly 1.0 on every exported task | T07 |
| G3 | Exploits | Do-nothing, stop-early, spam-advance, hold-freight, main-line-hold, malformed: reward 0 on every accepted seed | T06, T07 |
| G4 | Gap (pilot gate) | Big − small unclipped score ≥ 0.3; big ≤ 0.8; reference ≥ 30% cheaper than greedy (10 T1 dev seeds) | T09 |
| G5 | Episode size | ~60 decision points and < 100k tokens per T2 episode | T05 (proxy), T09 (measured) |
| G6 | Reference speed | < 2 minutes per T2 episode | T06 |
| G7 | Integrity | Verifier replays the log and cross-checks the live cost; infra failure → trial error, never a 0 | T07 |
| G8 | Tiebreaker | Trained small model beats the untrained small model on held-out T2 (95% CI of the difference excludes 0) | T10 |
| G9 | Craft | One-command evaluation in the README; one self-contained HTML report | T09 |

## 4. Non-goals

- The stretch factors (F9, F11–F14, F17, F18). They are listed in the backlog (§10).
- A GUI. The human-baseline CLI is backlog.
- Physics beyond deterministic section travel times.
- Multi-agent or multi-dispatcher play.
- `what_if` / `conflicts` tools in official runs. They are diagnostics only (T09).

---

## 5. Process rules (apply to every ticket)

### 5.1 The gate

The tickets are **strictly sequential**. Ticket N+1 may not start until ticket N passes
its gate. A ticket passes its gate when all of these are true:

1. Every test listed in the ticket exists, was written **before** the implementation,
   and was seen failing (red) for the expected reason (missing module, not-implemented,
   wrong value). Collection errors because a module doesn't exist yet count as red.
2. Every test in the ticket passes (green), **and** every test from all earlier tickets
   still passes (no regressions).
3. Tests behind the markers the ticket names as required (`harbor`, `model`, `gpu`) were
   run and passed on a machine that has those resources.
4. The ticket's "Definition of done" checklist is fully ticked.
5. The review gate from the global workflow rules ran: `/ponytail-review`, pstack
   (`/blast-radius`, `/correct`, `/show-me-your-work`) and
   `/thermo-nuclear-code-quality-review`. Sensible suggestions were applied and the tests
   re-run. If a reviewer could not run, the gate record says so.
6. A gate record exists at `docs/gates/TNN.md` (template in §5.4).

**Within** a ticket, the three people split the work (each ticket suggests a split).
The sequence costs some parallelism. That is the price of the gate, and it is
deliberate.

### 5.2 Test rules

- Test layout: `tests/tNN/test_*.py` per ticket; shared fixtures in `tests/fixtures/`;
  shared helpers in `tests/helpers.py`; `tests/conftest.py` registers markers.
- Each test function carries its ticket test id in its name or docstring, e.g.
  `def test_T02_07_section_reserved_refusal():`.
- Markers (registered with `--strict-markers`):

  | Marker | Meaning |
  |---|---|
  | `slow` | Over 10 s |
  | `harbor` | Needs Docker and the Harbor CLI (Linux, macOS or WSL2) |
  | `model` | Needs a real model API key or endpoint |
  | `gpu` | Needs a CUDA GPU |

- **Default gate command** (run from the repo root), with NN = the current ticket:
  ```bash
  uv run pytest tests/t01 tests/t02 ... tests/tNN -m "not harbor and not model and not gpu"
  ```
  Each ticket lists any extra marker runs it requires.
- Property tests use Hypothesis with `derandomize=True` (a reproducible profile in
  `conftest.py`).
- **Changing a test after it was written** is allowed only when the test contradicts
  this PRD, its ticket or DESIGN.md. Record the change and the reason in the gate record.
  Never weaken an assertion to make an implementation pass.
- Exact numeric expectations in tickets were computed by hand from §7.11 and the rules
  in §7. If an expectation is wrong because the arithmetic is wrong (not because the
  rule is), fix the test, show the corrected arithmetic in the gate record, and flag
  it to the team.

### 5.3 Repo rules

- **Never run `git add`, `git commit`, `git push` or anything that rewrites history.**
  The user stages and commits. At the end of each ticket, hand over the list of changed
  files, a ready `git add ...` line and a drafted `git commit -s -m "..."` message.
- Python ≥ 3.11, managed by `uv`. Dependencies are pinned in `uv.lock`.
- No new dependency for something the standard library or an installed package already
  does.
- `ruff` for lint and format (configured in `pyproject.toml`), run before the gate.

### 5.4 Gate record template (`docs/gates/TNN.md`)

```markdown
# Gate record TNN: <title>
- Date / people:
- Red run: <command> → <N failed, M errors> (summary of failure reasons)
- Green run: <command> → <N passed>
- Marker runs: harbor: <cmd/result or "not required">; model: ...; gpu: ...
- Regression run (t01..tNN): <result>
- Test changes after writing (with reasons):
- Reviewers: ponytail <ran/not run + summary>; pstack <...>; thermo-nuclear <...>
- Accepted / rejected suggestions:
- Deviations from PRD (and why):
- Residual risks:
```

---

## 6. Ticket map

| Ticket | Title | DESIGN.md sections | Required markers | Rough hours |
|---|---|---|---|---|
| [T01](tickets/T01-foundations-and-harbor-spike.md) | Foundations and Harbor spike | §8, §11 (day-1 test), §13 | `harbor` (+ `model` for 3 facts) | 0–3 |
| [T02](tickets/T02-simulator-core.md) | Simulator core: network, authorities, movement, F1–F4, gridlock, cost, decision points | §2, §3 (F1–F4), §5.1, §6.1 | none | 3–10 |
| [T03](tickets/T03-simulator-factors-and-events.md) | Factors F5–F8, F10, F15, F16; events; belief simulator | §2.3, §3, §4 | none | 10–15 |
| [T04](tickets/T04-manual-and-generator.md) | `manual()`, rule-coverage test, scenario generator, presets, seed sets | §2.6, §3, §9.1, §9.2 | none | 15–18 |
| [T05](tickets/T05-tools-mcp-replay-baselines.md) | Tool layer, MCP server, episode log, replay, baseline policies | §5, §6.4, §8.3 | none | 18–22 |
| [T06](tickets/T06-reference-exploits-seed-filter.md) | Reference dispatcher, exploits, seed filter, strength check | §6.3, §6.5, §7 | none | 22–28 |
| [T07](tickets/T07-harbor-packaging-and-verifier.md) | Harbor task export, verifier, oracle, isolation | §8, §9.3 | `harbor` (+ `model` for one smoke test) | 28–32 |
| [T08](tickets/T08-model-loop-runs-and-stats.md) | Our own model loop, run manifests, results JSONL, statistics | §9.3–§9.5 | none (`model` optional) | 32–35 |
| [T09](tickets/T09-pilot-evaluation-report.md) | Pilot gate, weight freeze, held-out evaluation, report, diagnostics, README | §6.2, §9.6–§9.8, §12 | `model`, `harbor` | 35–41 |
| [T10](tickets/T10-post-training.md) | Post-training: BC → DAgger → GRPO; before/after results | §10 | `gpu`, `model` | 41–48 |

**Hours are a guide, not a promise.** If the team falls behind, cut in this order:
stretch factors (already out), then diagnostics (T09 §diagnostics), then GRPO (T10
keeps BC + DAgger). Never cut the MVP factors, the exploit tests, the oracle check or
the replay verifier (DESIGN.md §11).

**Note on the pilot gate.** DESIGN.md puts it at hour 12. Sequential tickets move it to
T09, after all MVP factors exist. The pilot still runs on T1 (F1–F4), so the question
it answers is unchanged. If it fails, T09 says what to change.

---

## 7. Shared contracts

Every ticket builds against these. Changing one is a cross-ticket change: update this
section first, then every ticket that uses it.

### 7.1 Repository layout

```
junction/
  pyproject.toml  uv.lock  Makefile  README.md  DESIGN.md
  docs/  PRD.md  tickets/  gates/  spike-report.md  strength-check.md  pilot.md
  railway/
    __init__.py
    util.py                canonical_json(), content_hash()            (T01)
    sim/
      constants.py         every tunable number + RULES_VERSION       (T02, T03)
      rules.py             RuleId enum + one-line docs                 (T02, T03)
      model.py             pydantic scenario schema                    (T02)
      timing.py            travel time, free-run projection            (T02)
      engine.py            Simulator                                   (T02, T03)
      cost.py              cost components                             (T02)
      gridlock.py          wait-for graph                              (T02)
      events.py            possessions, restrictions, breakdowns, late entries (T03)
      observe.py           State / visible-state models, from_observation()   (T02, T03)
      manual.py            manual() text                               (T04)
    gen/  config.py presets.py generator.py seeds.py cli.py            (T04; export added in T07)
    tools/  session.py formats.py stringline.py client.py replay.py    (T05)
    server/  app.py __main__.py                                        (T05)
    policies/  base.py do_nothing.py stop_early.py greedy.py random_legal.py exploits.py  (T05, T06)
    ref/  reference.py rollout.py candidates.py offline_search.py oracle.py  (T06)
    harbor/  export.py verify.py templates/                            (T07)
    loop/  prompts.py model_client.py runner.py fake_model.py          (T08)
    eval/  manifest.py run.py stats.py report.py diagnostics.py pilot.py  (T08, T09)
    train/  data.py bc.py dagger.py grpo.py evaluate.py                (T10)
  seeds/      candidate ranges + accepted lists (JSON)                 (T04, T06)
  datasets/   exported Harbor tasks; held-out committed                (T07, T09)
  runs/       run manifests (YAML)                                     (T08)
  results/    JSONL results                                            (T08)
  reports/    HTML reports                                             (T09)
  spike/toy-task/   day-1 Harbor spike task                            (T01)
  tests/  conftest.py  helpers.py  fixtures/  t01/ ... t10/
```

The repo `tests/` folder (our pytest suite) is unrelated to each Harbor task's `tests/`
folder (that task's verifier). Don't confuse them.

**Console scripts** (in `pyproject.toml`, added by the ticket that implements them):
`gen` (T04), `railway-sim` (the MCP server, T05), `railway-eval` (T08), `railway-train`
(T10). The `Makefile` holds thin aliases (`make test`, `make report RUN=...`,
`make ci`) for Linux/macOS/WSL. Windows users call `uv run ...` directly.

### 7.2 Constants (`railway/sim/constants.py`)

All provisional until the pilot freeze in T09. After that, `RULES_VERSION = "1.0"` and
a test pins every value.

| Name | Value | Meaning |
|---|---|---|
| `RULES_VERSION` | `"0.2-provisional"` | Written into every scenario; verifier refuses a mismatch |
| `TICK_MINUTES` | 1 | 1 tick = 1 minute |
| `TOOL_BUDGET` | 600 (calibrated in T05; was 400) | Tool calls per episode, all tools counted |
| `TIME_CAP` | 60 | Max ticks between decision points |
| `PAX_GRACE` | 2 | Ticks of lateness forgiven at each timed stop |
| `TERMINUS_MULT` | 2 | Multiplier at the final timed stop |
| `W_EXPRESS`, `W_LOCAL` | 3.0, 1.0 | Passenger weights (× pax/100 per late tick) |
| `W_FREIGHT` | 0.5 | Per late tick at destination |
| `W_ENGINEERING` | 0.0 | |
| `UNFINISHED_COST` | 200 | Per train not finished at shift end |
| `UNFINISHED_FREIGHT_EXTRA` | 1000 | Per unfinished freight train, × `shift_ticks` / `UNFINISHED_FREIGHT_SHIFT` (calibrated at T06: without it, holding all freight beat greedy on 24/30 T1 seeds; a flat 1000 still lost on T2) |
| `UNFINISHED_FREIGHT_SHIFT` | 240 | Shift length at which the freight extra is exactly `UNFINISHED_FREIGHT_EXTRA` |
| `RESTART_TICKS` | express 1, local 1, freight 4, engineering 1 | F5 |
| `TURNAROUND` | 10 | F6: min ticks between arrival and the next service's entry |
| `CREW_LIMIT` | 240 | F10: duty ticks before timeout |
| `CREW_WARNING` | 30 | F10: warning when remaining ≤ this |
| `RELIEF_TICKS` | 8 | F10 |
| `TIMEOUT_BLOCK` | 45 | F10: ticks a timed-out train stands still |
| `TIMEOUT_COST` | 300 | F10 |
| `REVERSE_TICKS` | 15 | Gridlock reversal duration |
| `REVERSE_COST` | 50 | |
| `POSSESSION_LOOKAHEAD` | 45–60 | F7 reveal range (generator) |
| `RESTRICTION_LOOKAHEAD` | 30–60 | F8 |
| `LATE_ENTRY_LOOKAHEAD` | 20 | F16: reveal = original entry − 20 |
| `NOTES_MAX` | 1500 | chars |
| `STRINGLINE_WINDOW` | 60 | ticks |
| `STRINGLINE_MAX_CHARS` | 2000 | |
| `TRAIN_TYPES` | see DESIGN.md §2.3 | length m, max km/h |

Train types: express 200 m / 140 km/h; local 120 m / 100 km/h; freight 600 m / 70 km/h;
engineering 100 m / 60 km/h.

### 7.3 Scenario schema (`railway/sim/model.py`, pydantic v2, `extra="forbid"`)

```text
Scenario
  schema_version: 1
  rules_version: str                   must equal constants.RULES_VERSION
  scenario_id: str                     opaque: f"{preset}-{content_hash(scenario minus id)[:12]}", e.g. "T2-3f9a1c2b7d4e"
                                       (never derived visibly from the seed; fixtures use "FIX-<name>")
  seed: int; preset: str               "T1" | "T2" | "T3" | "T2-S14" | "fixture"
  shift_ticks: int
  factors: list[str]                   sorted, subset of F1..F8, F10, F15, F16
  network: Network
  trains: list[Train]                  sorted by id
  events: list[Event]                  sorted by (reveal_tick, id)

Network
  yards: {"west": "YW", "east": "YE"}  unlimited capacity, never reserved
  stations: list[Station]              ordered west → east, ids "S1".."SN"
  sections: list[Section]              ordered west → east, N+1 of them

Station:  id, major: bool, crew_station: bool, loops: list[Loop] (0–3)
Loop:     id ("L1".."L3", unique per station), length_m: 150|300|700, kind: "platform"|"goods"
Section:  id "<west>-<east>" (e.g. "S1-S2", "YW-S1"), west, east, length_m: int, line_speed_kmh: int

Train
  id: str                              "E<n>" express, "L<n>" local, "F<n>" freight, "N<n>" engineering
  type: "express"|"local"|"freight"|"engineering"
  direction: "east"|"west"             east trains run YW → YE; west trains YE → YW
  pax: int                             0 for freight/engineering
  entry_tick: int                      scheduled appearance (and departure) at the origin yard
  stops: list[Stop]                    timed stops in route order; the LAST one is the destination yard
  next_service: str | null             F6: id of the train this set forms next (opposite direction, same yard)
  crew_used_at_entry: int              F10: duty ticks already worked (0 if F10 off)

Stop:     station, arr: int, dep: int | null (null only for the destination), min_dwell: int (1–2; 0 for destination)

Event (discriminated on "kind"; every event has id "EV<n>" and reveal_tick)
  possession:         section, start, end                    (F7)  reveal = start − U[45,60]
  speed_restriction:  sections: list[str], speed_kmh, start, end   (F8)  reveal = start − U[30,60]
  late_entry:         train, new_entry_tick                  (F16) reveal = original entry − 20
  breakdown:          train, tick, duration                  (F15) reveal = tick (when it happens)
```

Trains always start at their origin yard and finish on arrival at their destination yard
(DESIGN.md leaves this open; see §8 D1).

**Resource ids:** sections by id (`"S2-S3"`); station tracks `"S2/main"`, `"S2/L1"`;
yards `"YW"`, `"YE"`.

### 7.4 Simulator API (`railway/sim/engine.py`)

This is the day-1 interface from DESIGN.md §11, made exact.

```python
class Simulator:
    @classmethod
    def load(cls, scenario: Scenario) -> "Simulator": ...
    @classmethod
    def from_observation(cls, static: StaticInfo, state: State) -> "Simulator": ...   # T03 (belief sim)

    tick: int                      # current tick
    done: bool                     # shift over

    def apply(self, action: Action) -> ActionResult: ...    # uses no time
    def check(self, action: Action) -> ActionResult: ...    # dry run: same verdict as apply, no state change
    def advance(self, max_tick: int | None = None) -> AdvanceResult: ...
        # runs to the next decision point (§7.7), or max_tick, or shift end, whichever first
    def end_shift(self) -> AdvanceResult: ...                # run to shift end, no new authorities
    def state(self) -> State: ...                            # AGENT-VISIBLE state only (no unrevealed events)
    def static_info(self) -> StaticInfo: ...                 # network + timetable + shift + factors + rules_version
    def cost(self) -> CostBreakdown: ...
    def clone(self) -> "Simulator": ...                      # deep, independent copy
```

- `state()` must never contain anything not yet revealed. That makes "the agent sees
  only what it may see" true by construction.
- `Action` is a pydantic discriminated union on `kind`:
  - `Authorize(train, station, track)` — `track` is `"main"` or a loop id; for a yard it
    must be `"main"`;
  - `CancelAuthority(train)`;
  - `Hold(train, until_tick)`;
  - `RelieveCrew(train)`;
  - `Reverse(train)`.
- `ActionResult = {ok: bool, rule: RuleId | None, message: str}`. Message formats:
  - `GRANTED E1 -> S2/L1 (arrive t=6)`
  - `REFUSED R-SECTION-RESERVED: S2-S3 reserved by L5`
- `AdvanceResult = {tick, reasons: list[Reason], changes: list[Change], done}`.
  - `Reason = {kind, train | None, detail}`.
  - `Change = {tick, kind, train | None, text}`.

### 7.5 Movement rules (normative)

1. **No authority, no movement** (`R-NO-AUTO-MOVE`).
2. **Authority** `(train, station, track)`:
   - The path is every resource from the train's current position (or its current
     authority end, for an extension) up to the destination:
     - the sections;
     - the `main` of every intermediate station passed through;
     - the destination track.
   - A grant reserves the whole path.
   - A section is released when the train leaves it. A station's `main` is released when
     the train leaves the station.
3. **Grant checks, in this order** (the first failure is the refusal reason):
   - **Argument checks:**
     - `R-BAD-ARGS`
     - `R-UNKNOWN-TRAIN`
     - `R-UNKNOWN-STATION`
     - `R-UNKNOWN-TRACK`
     - `R-NOT-IN-NETWORK`
     - `R-HELD`
     - `R-WRONG-DIRECTION` (destination not strictly ahead of the current position or
       current authority end)
   - **Resource checks** (each resource on the path, in path order):
     - `R-SECTION-RESERVED`
     - `R-MAIN-BLOCKED`
   - **Destination track checks:**
     - `R-TRACK-OCCUPIED`
     - `R-LOOP-LEN` (train length > loop length; `main` and yards have no length limit)
     - `R-PLATFORM` (express or local into a `goods` loop)
   - **Timing check:**
     - `R-POSSESSION` (any revealed possession window overlaps the projected occupation
       of a path section; overlap means `entry < end and exit > start`, using free-run
       projected times)
4. **Extension.** Authorizing a train that already has an authority, to a station
   strictly beyond its current end, *extends* it:
   - the old destination track is released;
   - that station's `main` becomes a pass-through resource, and must be free or already
     held by this train;
   - an extension granted before the train arrives means it does not stop there.
5. **Scheduled stops inside an authority** are served automatically on `main`:
   - the train stops;
   - it dwells at least `min_dwell`;
   - it does not depart before `dep` (`R-NO-EARLY-DEP`);
   - then it continues.

   At the destination, it stops on the requested track.
6. **Departure** happens at the first tick when the train has an authority and all of
   these are true:
   - dwell done;
   - `tick ≥ dep` (for passenger trains at a timed stop) or `tick ≥ entry_tick` (at
     origin);
   - not held;
   - not broken down, relieving, timed out or reversing.
7. **Section travel time:**
   ```
   ceil(length_m × 60 / (min(line_speed, train_max, restriction) × 1000))
   ```
   - Use integer arithmetic: `-(-length_m*60 // (v*1000))`.
   - The minimum is 1 tick.
   - The restriction applies if the train **enters** the section at a tick inside the
     restriction window `[start, end)`.
   - Add the restart penalty to the first section after an unscheduled stop (F5 on).
8. **Passing through a station** takes 0 ticks. A train stopped on `main` blocks the
   station for everyone else (`R-STOP-ON-MAIN-BLOCKS`).
9. **Per-tick processing order.** This order is fixed; ties are broken by train id
   ascending.
   1. Reveal events and apply authority cuts (F7).
   2. Breakdown starts and ends, crew timeouts, relief and reversal completions.
   3. Section exits and arrivals (releases).
   4. Departures and section entries.
   5. Crew duty counters (+1 per tick for every train in the network).
   6. Decision triggers.
   7. Gridlock check.

   **Same-tick effect of actions:** when `advance()` returns at tick t and the agent
   acts, the next `advance()` first re-runs phases 4–7 for tick t (so a ready train
   authorized at t departs at t), then continues to t+1. Edge triggers already
   reported at t are not reported again.
10. **Gridlock** (`R-GRIDLOCK`) is a structural test on the wait-for graph.
    - **Blocked train:** unfinished, in the network, stationary at a station, with no
      authority.
    - **Edges:** `T → U` when every feasible next-station move of T (the section ahead
      plus any fitting, type-allowed track at the next station, or the yard) is blocked
      by occupation or reservation, and U is a train causing that block.
    - **Gridlock** = a strongly connected component of size ≥ 2 among blocked trains.
    - Timers (dwell, hold, breakdown) are ignored.
11. **`reverse(train)`:**
    - Allowed only for a train in a gridlock component (`R-NO-GRIDLOCK`).
    - Moves the train back to the previous station: a free, fitting, type-allowed loop
      there, or the yard if the previous station is a yard. Otherwise refused with
      `R-NO-REVERSE-PATH`.
    - Occupies the section behind for `REVERSE_TICKS`, then arrives.
    - Costs `REVERSE_COST`.
12. **`cancel_authority(train)`:**
    - Truncates the authority to the first station the train has not yet passed.
    - If the old path passed through that station, the destination becomes its `main`.
    - If the train is stationary at its authority origin, the whole authority is removed.
    - With no authority it is refused with `R-NO-AUTHORITY`.
13. **`hold(train, until_tick)`:**
    - The train cannot receive a new authority before `until_tick` (`R-HELD`).
    - It is refused with `R-HOLD-PAST` if `until_tick ≤ tick`.
    - An existing authority is not affected.

### 7.6 Rule IDs and cost components

**`RuleId`** (`railway/sim/rules.py`). Every member has a one-line description that
`manual()` prints. A test (T04) fails if code uses an ID that isn't a member, or a member
that is missing from `manual()`.

- **Refusals:**
  - `R-BAD-ARGS`
  - `R-UNKNOWN-TRAIN`
  - `R-UNKNOWN-STATION`
  - `R-UNKNOWN-TRACK`
  - `R-NOT-IN-NETWORK`
  - `R-HELD`
  - `R-WRONG-DIRECTION`
  - `R-SECTION-RESERVED`
  - `R-MAIN-BLOCKED`
  - `R-TRACK-OCCUPIED`
  - `R-LOOP-LEN`
  - `R-PLATFORM`
  - `R-POSSESSION`
  - `R-NO-AUTHORITY`
  - `R-HOLD-PAST`
  - `R-NOT-STATIONARY`
  - `R-NOT-CREW-STATION`
  - `R-NO-GRIDLOCK`
  - `R-NO-REVERSE-PATH`
  - `R-NOTES-TOO-LONG`
  - `R-WINDOW`
  - `R-SHIFT-ENDED`
  - `R-DIAGNOSTIC-ONLY`
- **Enforced behaviour:**
  - `R-NO-AUTO-MOVE`
  - `R-ONE-TRAIN-PER-RESOURCE`
  - `R-STOP-ON-MAIN-BLOCKS`
  - `R-SCHEDULED-STOP`
  - `R-NO-EARLY-DEP`
  - `R-RESTART` (F5)
  - `R-ROTATION` (F6)
  - `R-AUTHORITY-CUT` (F7)
  - `R-SPEED-RESTRICTION` (F8)
  - `R-CREW-LIMIT` (F10)
  - `R-CREW-RELIEF` (F10)
  - `R-BREAKDOWN` (F15)
  - `R-LATE-ENTRY` (F16)
  - `R-GRIDLOCK`
  - `R-REVERSE`
  - `R-BUDGET`
  - `R-STOP-IS-END-SHIFT`

**`CostComponent`** (`railway/sim/cost.py`). All floats, accumulated in event order, and
reported rounded to 6 decimals.

| Component | When charged | Formula |
|---|---|---|
| `pax_lateness` | On arrival at each timed stop (express, local) | `w × pax/100 × max(0, late − PAX_GRACE)`, × `TERMINUS_MULT` at the destination |
| `freight_lateness` | On arrival at the destination (freight) | `W_FREIGHT × max(0, late)` |
| `crew_timeout` | When the timeout happens | `TIMEOUT_COST` (the 45-tick blockage shows up in lateness naturally) |
| `reversal` | When `reverse` is granted | `REVERSE_COST` (time lost shows up in lateness naturally) |
| `unfinished` | At shift end, per unfinished train (including never-entered trains) | `UNFINISHED_COST` (+ `UNFINISHED_FREIGHT_EXTRA` × `shift_ticks` / `UNFINISHED_FREIGHT_SHIFT` for freight) + the destination-arrival lateness formula above, as if arriving at `shift_ticks` |

`late = actual_arrival − scheduled_arr`.

`total = sum of components`.

### 7.7 Decision points

`advance()` stops at the first tick where any trigger fires. Triggers are **edges**:
they fire when something *becomes* true, never repeatedly for a state that stays true.

| Reason kind | Fires when |
|---|---|
| `AUTHORITY_END` | A train arrives at the end of its authority |
| `READY` | A train in the network with no authority becomes able to depart: it enters at its origin yard, or its dwell, hold, relief, breakdown, timeout block or reversal ends |
| `BULLETIN` | An event is revealed (possession, restriction, late entry) |
| `BREAKDOWN` | A breakdown starts (F15) |
| `CREW_WARNING` | A crew's remaining duty becomes ≤ `CREW_WARNING` (F10) |
| `CREW_TIMEOUT` | A crew times out (F10) |
| `GRIDLOCK` | A gridlock component appears (or gains a member) |
| `TIME_CAP` | `TIME_CAP` ticks passed since the previous decision point |
| `SHIFT_END` | The shift is over |

`load()` itself is decision point 0 (tick 0). Triggers at tick 0 are consumed by it.

### 7.8 Tools (`railway/tools/session.py`; MCP names identical)

Every call, including invalid ones and read-only ones, uses 1 unit of `TOOL_BUDGET`.

| Tool | Args | Output |
|---|---|---|
| `manual` | – | Markdown rules text (T04) |
| `network` | – | JSON: `shift_ticks`, `factors`, `rules_version`, yards, stations (loops), sections, crew stations |
| `timetable` | `train?` | JSON: trains with type, direction, pax, entry, stops, `next_service` (original schedule; late entries appear in `bulletins`/`status`) |
| `status` | `compact=false` | Full: JSON of `State` (visible only, includes revealed restrictions and possessions, notes). Compact: text, one line per train (§7.8.1) |
| `bulletins` | – | JSON list of revealed events with `reveal_tick` |
| `kpis` | – | JSON: cost by component and total; counts of gridlocks, crew timeouts, reversals, refusals, invalid calls; budget used and left |
| `stringline` | `from_tick, to_tick` | ASCII time-distance diagram; `to − from ≤ 60` else `R-WINDOW`; ≤ 2000 characters |
| `notes` | `text` | Replaces the scratchpad; > 1500 characters → `R-NOTES-TOO-LONG` |
| `authorize` | `train, station, track` | `GRANTED ...` / `REFUSED R-...: reason` |
| `cancel_authority` | `train` | `OK ...` / `REFUSED ...` |
| `hold` | `train, until_tick` | `OK ...` / `REFUSED ...` |
| `relieve_crew` | `train` | `OK ...` / `REFUSED ...` |
| `reverse` | `train` | `OK ...` / `REFUSED ...` |
| `advance_to_next_event` | – | Text: header `t=<tick> stop: <REASON> <train>, ...`, then only the changes since the previous `advance`, then `budget: <used>/<TOOL_BUDGET>` |
| `end_shift` | – | Runs to shift end; returns the final cost summary |

After the shift ends (`end_shift`, budget exhausted, or session closed), every call
returns `REFUSED R-SHIFT-ENDED: ...` plus the final cost line.

#### 7.8.1 Compact status format

```
t=142 | gridlock: none | notes: "E3 passes F12 at S6; relieve F12 crew at S4"
E3  express  at S5/L1-platform  auth:-            delay:+4  crew:96  next:S8@158
F12 freight  S6-S7 westbound    auth:S6/L2-goods  delay:+11 crew:23
L7  local    at S2/L1-platform  auth:S3/L1        delay:0   crew:140 next:S3@145
N2  engineering waiting YE entry@200
restrictions: possession S9-S10 170-200; speed S3-S4 40km/h 150-210
```

- The header is always first.
- Then one line per train that hasn't finished, sorted by id. Finished trains are
  omitted.
- The `restrictions:` footer is always last (`restrictions: none` if empty).
- `crew:` appears only if F10 is on.
- `delay` = free-run projected arrival at the next timed stop − its `arr`, floored at 0.

### 7.9 Episode log (`GET /episode_log`, verifier only)

```json
{"scenario_id": "...", "scenario_hash": "...", "rules_version": "...",
 "entries": [{"seq": 1, "tick": 0, "tool": "authorize",
              "args": {"train": "E1", "station": "S2", "track": "L1"},
              "status": "GRANTED", "rule": null}],
 "closed": true, "final_tick": 360,
 "final_cost": 769.0, "final_cost_by_component": {"pax_lateness": 24.0, "...": 0.0}}
```

- `status` is one of `GRANTED | REFUSED | OK | INVALID | INFO`.
- The endpoint requires `Authorization: Bearer <token>`. The token is generated per
  exported task and stored in the sim image and in the task's `tests/`.
- Calling the endpoint **closes the episode**: it runs `end_shift` if needed, and every
  later tool call returns `R-SHIFT-ENDED`.

### 7.10 Results row (`results/<run>.jsonl`, one per episode)

```text
run, harness ("harbor"|"loop"), mode ("windowed"|"full"|null), model, rep,
scenario_id, scenario_hash, seed, preset, rules_version,
cost_total, cost_by_component{...}, c_greedy, c_ref,
score_clipped, score_unclipped, cost_ratio (c_agent / c_ref),
gridlocks, crew_timeouts, reversals, unfinished, refusals, invalid_calls,
tool_calls, decision_points, tokens_in, tokens_out, wall_s,
status ("ok"|"error"), error (str|null)
```

### 7.11 Canonical fixture `tests/fixtures/mini3.json`

Ticket tests compute exact numbers on this fixture. Variants are built in tests with
`tests/helpers.py::variant(base, **patches)`.

```json
{
  "schema_version": 1, "rules_version": "0.2-provisional",
  "scenario_id": "FIX-mini3", "seed": 0, "preset": "fixture",
  "shift_ticks": 60, "factors": ["F1", "F2", "F3", "F4"],
  "network": {
    "yards": {"west": "YW", "east": "YE"},
    "stations": [
      {"id": "S1", "major": false, "crew_station": false,
       "loops": [{"id": "L1", "length_m": 300, "kind": "platform"}]},
      {"id": "S2", "major": true, "crew_station": true,
       "loops": [{"id": "L1", "length_m": 300, "kind": "platform"},
                 {"id": "L2", "length_m": 700, "kind": "goods"}]},
      {"id": "S3", "major": false, "crew_station": false,
       "loops": [{"id": "L1", "length_m": 150, "kind": "platform"}]}
    ],
    "sections": [
      {"id": "YW-S1", "west": "YW", "east": "S1", "length_m": 6000, "line_speed_kmh": 120},
      {"id": "S1-S2", "west": "S1", "east": "S2", "length_m": 6000, "line_speed_kmh": 120},
      {"id": "S2-S3", "west": "S2", "east": "S3", "length_m": 6000, "line_speed_kmh": 120},
      {"id": "S3-YE", "west": "S3", "east": "YE", "length_m": 6000, "line_speed_kmh": 120}
    ]
  },
  "trains": [
    {"id": "E1", "type": "express", "direction": "east", "pax": 200, "entry_tick": 0,
     "stops": [{"station": "S2", "arr": 6, "dep": 7, "min_dwell": 1},
               {"station": "YE", "arr": 13, "dep": null, "min_dwell": 0}],
     "next_service": null, "crew_used_at_entry": 0},
    {"id": "F21", "type": "freight", "direction": "east", "pax": 0, "entry_tick": 2,
     "stops": [{"station": "YE", "arr": 26, "dep": null, "min_dwell": 0}],
     "next_service": null, "crew_used_at_entry": 0},
    {"id": "L5", "type": "local", "direction": "west", "pax": 100, "entry_tick": 0,
     "stops": [{"station": "S3", "arr": 4, "dep": 5, "min_dwell": 1},
               {"station": "S2", "arr": 9, "dep": 10, "min_dwell": 1},
               {"station": "S1", "arr": 14, "dep": 15, "min_dwell": 1},
               {"station": "YW", "arr": 19, "dep": null, "min_dwell": 0}],
     "next_service": null, "crew_used_at_entry": 0}
  ],
  "events": []
}
```

**Travel times on mini3** (every section 6000 m at 120 km/h):

| Train | Effective speed | Ticks per section |
|---|---|---|
| Express | min(120, 140) = 120 | 3 |
| Local | min(120, 100) = 100 | ceil(3.6) = 4 |
| Freight | min(120, 70) = 70 | ceil(5.14) = 6 |

**Other fixtures:**

- `mini2_gridlock.json`
  - Network: stations S1 and S2, **no loops**; sections YW-S1, S1-S2, S2-YE, each
    6000 m / 120 km/h.
  - **L1**: local, east, pax 100, entry 0. Stops: S1 4/5, S2 9/10, YE 14.
  - **L2**: local, west, pax 100, entry 0. Stops: S2 4/5, S1 9/10, YW 14.
  - Shift 60, factors F1–F4.
- `mini3_noloops.json`
  - Network: stations S1, S2 and S3, **no loops**; 4 sections, each 6000 m / 120 km/h.
  - **L1**: local, east, pax 100, entry 0. Stops: S1 4/5, S2 9/10, S3 14/15, YE 19.
  - **L2**: local, west, pax 100, entry 0. Stops: S3 4/5, S2 9/10, S1 14/15, YW 19.
  - Shift 60.

### 7.12 Determinism rules

- **No sources of non-determinism:**
  - no wall clock;
  - no global `random`, only `random.Random(seed)` instances that are passed in;
  - no reliance on `hash()` of strings;
  - no iteration over sets without sorting.
- Sort keys everywhere: `(tick, processing-phase, train id)`.
- **Canonical JSON:** `json.dumps(obj, sort_keys=True, separators=(",", ":"),
  ensure_ascii=False)`.
- **Hashing:** `scenario_hash = sha256(canonical_json(scenario))`.
- Floats appear only in costs. They are summed in event order and printed with
  `round(x, 6)`.

---

## 8. Decisions this PRD adds to DESIGN.md (all approved by the user, 2026-11-14)

| # | Decision | Why |
|---|---|---|
| D1 | Trains start and finish at the terminal yards. Rotations (F6) happen at yards. | DESIGN.md doesn't define mid-line origins; yards keep entry and turnaround simple |
| D2 | Timed stops inside an authority are served automatically on `main`; only the destination track is the agent's choice | Halves the tool calls per local train; still forces the pass/no-pass decision |
| D3 | Every station `main` has a platform; passenger trains may not enter goods loops (`R-PLATFORM`) | Makes "platform" vs "goods" a real constraint |
| D4 | Authority extension is allowed (strictly beyond the current end); extending before arrival avoids the stop and the F5 restart penalty | "Keep freight rolling" (F5) needs a way to not stop |
| D5 | A newly revealed possession cuts back granted authorities that would conflict (`R-AUTHORITY-CUT`); if a train is in the section at the start, the possession starts when the section clears (end unchanged) | Possessions can't be enforced on authorities granted before the reveal otherwise |
| D6 | Unfinished passenger trains are charged at the destination only (× terminus multiplier), plus 200, booked under `unfinished` | Unserved intermediate stops would double-count |
| D7 | `CREW_LIMIT = 240`; crews carry over through rotations; yards are not crew stations; a timeout stops the train where it is (mid-section included) for 45 ticks; restart penalty applies after any unscheduled stop, including mid-section | Makes F6 × F10 the long-delayed effect DESIGN.md §3 wants |
| D8 | `TURNAROUND = 10` | Not in DESIGN.md |
| D9 | Engineering trains exist in the schema; the MVP generator doesn't emit them | DESIGN.md gives no penalty for a missed work site; F7 is complete without them |
| D10 | Gridlock = strongly connected component ≥ 2 in the wait-for graph; reversing into a yard is allowed | Gives an exact, testable definition |
| D11 | Every tool call counts toward the budget (read-only and invalid included) | Simplest; stops cheap spam of read-only calls |
| D12 | `/episode_log` is verifier-only (bearer token) and closes the episode; the agent has no log endpoint | A log endpoint that also returns the "end now" cost would leak future breakdowns |
| D13 | `status()` (full) is JSON and is the complete visible state; scripted policies read only tool outputs (network, timetable once; status per decision point) | Guarantees the reference sees what the agent sees, and makes DAgger labelling possible |
| D14 | Weights live in code constants; scenarios carry `rules_version`; the verifier refuses a mismatch | One frozen place; stale `norm.json` can't be scored silently |
| D15 | Greedy FIFO is exactly as defined in T05 (one station ahead, loops only, lowest loop id, oldest train first) | The score anchor must be precise |
| D16 | Aggregates: headline `score_ros = Σ(Cg − Ca) / Σ(Cg − Cr)` (unclipped terms); also `reward_mean` (mean of clipped per-task rewards) and `cost_ratio = ΣCa / ΣCr` | DESIGN.md §6.2, made exact |
| D17 | Late-entry reveal = original entry − 20; the generator only delays trains whose original entry ≥ 20 | Keeps reveal ≥ 0 |
| D18 | The allowlist defaults to common model-provider hosts plus `sim`; overridable at export | The judges' provider is unknown (§9) |
| D19 | Decision triggers are edge-triggered; authorizing a stationary train during its dwell is allowed (it departs when ready) | Avoids zero-time decision loops |
| D20 | Sequential gate moves the pilot to T09 | User's process rule |

---

## 9. Open items and blocking inputs

| Item | Blocks | Owner | Fallback |
|---|---|---|---|
| Small / mid / big models, providers, API keys | T08 `model` smoke (optional), T09, T10 | User | T08 builds and tests against the fake model; T09 can't pass without real models |
| GPU plan and small-model weights | T10 | User | BC on a ≤ 1.5B model on a single consumer GPU |
| Judges' Harbor backend (docker/podman/ec2/daytona/modal vs e2b/runloop) | T07 | User (ask organizers) | Single-container fallback (DESIGN.md §8.4), built only if needed |
| Judges' model provider host (for the allowlist) | T07 | User (ask organizers) | D18 default list |
| Harbor version to pin | T01 | Person B | Latest release on day 1, recorded in the spike report |
| Agent timeout, tool budget, cost weights | T09 freeze | Pilot | Provisional values in §7.2 |

Note: `junction/` currently sits inside a git repository rooted at the user's home
directory. Consider `git init` inside `junction/` before the first commit.

## 10. Backlog (after T10, only if time remains)

- F13 freight deadline (port ship cutoff): 400 per miss. Then F17 special requests.
  Then F9, F11, F12, F14, F18.
- T3 showcase preset content.
- Human-baseline CLI (DESIGN.md §9.7 item 8).
- Single-container fallback packaging, if the judges' backend needs it (otherwise this
  moves into T07).
