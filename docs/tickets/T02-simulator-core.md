# T02: Simulator core (network, authorities, movement, F1–F4, gridlock, cost, decision points)

| | |
|---|---|
| Depends on | T01 (repo, helpers, `canonical_json`) |
| Unblocks | T03 (factors), and transitively everything else |
| DESIGN.md | §2 (world model), §3 F1–F4, §5.1 (decision points), §6.1 (cost) |
| PRD | §7.2–§7.7, §7.11 (fixtures), §7.12 (determinism), §8 D1–D6, D10, D19 |
| Required markers | default only |
| Suggested split | A: `engine.py` movement + authorities. B: `model.py` + `observe.py` (State) + `timing.py` + fixtures. C: `cost.py`, `gridlock.py`, property tests |

## 1. Goal

A deterministic, event-driven simulator for the single-track line with F1–F4: opposing
trains, loop capacity, loop length and train-type priorities through cost weights. It
must:

- grant or refuse movement authorities exactly per PRD §7.5;
- move trains with integer section times;
- stop at decision points (§7.7);
- detect gridlock and allow reversal;
- charge cost per §7.6.

Everything later (tools, reference, verifier, training) sits on this API. Its
correctness and determinism are the whole project's integrity.

## 2. Context the implementer needs

- **Read PRD §7.3–§7.7 completely before writing a test.** They are normative.
- **Time is integer ticks** (1 tick = 1 minute). Time moves only inside `advance()` /
  `end_shift()`. `apply()` uses no time.
- **"Event-driven"** means `advance()` jumps from one interesting tick to the next (the
  next section exit, dwell end, entry or cap). It doesn't loop over every tick. A simple
  per-tick loop is acceptable if it meets the speed target (§6). Correctness first.
- **Same-tick effect of actions** (add this to your mental model; it is part of the
  contract):
  - when `advance()` returns at tick t, the agent may apply actions at t;
  - the next `advance()` first re-runs phases 4–7 of PRD §7.5 rule 9 for tick t, so a
    ready train authorized at t departs at t;
  - only then does it move on to t+1;
  - edge triggers already reported at t are not reported again.
- **Factors in this ticket:**
  - F1 (opposing trains), F2 (loop capacity) and F3 (loop length) emerge from the
    resource rules;
  - F4 (priorities) is the cost weights;
  - a scenario listing any factor outside F1–F4 must raise `NotImplementedError` in
    `Simulator.load` until T03 adds it. No silent ignoring.
- **Yards** (`YW`, `YE`):
  - unlimited capacity, never reserved, no length limit;
  - track argument `"main"` only;
  - a train "enters the network" at its origin yard at `entry_tick`;
  - it is "finished" on arrival at its destination yard, after which it occupies
    nothing.
- **Scheduled stops inside an authority** (PRD §7.5 rule 5) are served on the station
  `main`, automatically. Express trains stop only at their listed (major) stations,
  locals at every station, freight at none (only the destination).
- **The DESIGN.md movement-authority rule maps to PRD §7.5.** Collisions are impossible
  by construction: every resource on a path is reserved at grant time and released only
  behind the train.

### 2.1 State model (`railway/sim/observe.py`)

`State` is the **agent-visible** state (pydantic; JSON via `model_dump(mode="json")`):

```text
State
  tick: int, shift_ticks: int, done: bool
  gridlock: list[str]                      # sorted train ids in gridlock components
  trains: list[TrainState]                 # ALL trains, sorted by id
  restrictions: list[...] = []             # filled in T03
  possessions: list[...] = []              # filled in T03
  cost: CostBreakdown                      # by component + total
  notes: str = ""                          # filled by the tool layer in T05

TrainState
  id, type, direction, pax
  phase: "not_entered" | "at_station" | "in_section" | "reversing" | "finished"
  station: str|null, track: str|null       # when at_station ("YW"/"YE" with track "main" at yards)
  section: str|null, section_enter: int|null, section_exit: int|null   # in_section / reversing
  authority: {dest_station, dest_track, path: list[str]} | null        # path = remaining reserved resource ids
  entry_tick: int                          # effective entry as currently known
  ready_tick: int|null                     # earliest tick it could depart now (dwell, dep, hold), null if not at a station
  hold_until: int|null, dwell_until: int|null
  next_stop: {station, arr, dep}|null      # next timed stop not yet served
  arrivals: list[{station, sched, actual}] # served timed stops, in order
  delay: int                               # PRD §7.8.1 definition
  finished_tick: int|null
```

`StaticInfo` holds the network, the timetable (trains without events), `shift_ticks`,
`factors` and `rules_version`. It has **no seed, no scenario id, no events.**

`CostBreakdown` has `pax_lateness`, `freight_lateness`, `crew_timeout`, `reversal`,
`unfinished` and `total`, all floats.

### 2.2 Scenario id

`scenario_id = f"{preset}-{content_hash(scenario_without_id)[:12]}"`. It is opaque, so
the seed can't be read off a task name. The generator (T04) sets it; fixtures use
`FIX-<name>`.

## 3. Scope

**In scope:**

- `constants.py`: all of PRD §7.2. The F5+ constants exist but are unused here.
- `rules.py`: the full `RuleId` enum from PRD §7.6, each member with a one-line
  description. The T03-only behaviours are present but not yet enforced.
- `model.py`: the scenario schema with validators (§5 tests list them).
- `timing.py`:
  - `travel_ticks(length_m, line_kmh, train_kmh, restriction_kmh=None) -> int`;
  - `project(sim_view, train_id, path, start_tick) -> list[(resource, enter, exit)]`
    (free-run projection with dwell and `dep` constraints).
- `engine.py`: `Simulator` per PRD §7.4. `from_observation` is T03.
- `cost.py`: `pax_cost(...)`, `freight_cost(...)`, `unfinished_cost(...)` and the
  accumulator.
- `gridlock.py`: `find_gridlock(state_like) -> list[list[str]]` (SCCs ≥ 2, each sorted,
  the list sorted).
- Fixtures: `tests/fixtures/mini3.json`, `mini2_gridlock.json` and `mini3_noloops.json`,
  exactly per PRD §7.11.

**Out of scope:**

- F5–F8, F10, F15, F16 (T03).
- Tool text formats (T05). The `message` strings here are the only text.
- `manual()` (T04).

## 4. Implementation notes and pitfalls

- **Path construction for east trains** at station index i (yard YW = 0, S1..SN = 1..N,
  YE = N+1) going to index j > i:
  - sections `(i,i+1) ... (j-1,j)`;
  - `main` of every station strictly between i and j;
  - then the destination track.
  - West trains mirror this.
  - A train **inside a section** (only possible for extensions in T02) builds the
    extension path from its current authority end.
- **Check order matters.** The first failing check in PRD §7.5 rule 3 order is the
  reported rule. Tests pin this order.
- **Refusal message formats** (tests match these exactly):
  - `REFUSED R-SECTION-RESERVED: <section> reserved by <train>`
  - `REFUSED R-MAIN-BLOCKED: <station>/main reserved by <train>`
  - `REFUSED R-TRACK-OCCUPIED: <station>/<track> reserved by <train>`
  - `REFUSED R-LOOP-LEN: <train> is <len> m, <station>/<loop> is <len> m`
  - `REFUSED R-PLATFORM: <station>/<loop> is a goods loop`
  - `REFUSED R-WRONG-DIRECTION: <station> is not ahead of <train>`
  - `REFUSED R-NOT-IN-NETWORK: <train> enters at t=<n>` or `... has finished`
  - `REFUSED R-HELD: <train> held until t=<n>`
  - `REFUSED R-UNKNOWN-TRAIN|STATION|TRACK: <value>`
  - `REFUSED R-NO-AUTHORITY: <train> has no authority`
  - `REFUSED R-HOLD-PAST: t=<n> is not in the future`
  - `REFUSED R-NO-GRIDLOCK: <train> is not in a gridlock`
  - `REFUSED R-NO-REVERSE-PATH: no free loop at <station>`
- **Reserved vs occupied.** One "holder" map per resource is enough: whoever holds the
  reservation is also the only one who can occupy it.
- **`clone()`** must be a deep copy that shares no mutable state. `copy.deepcopy` is
  fine if fast enough; the reference (T06) clones thousands of times, so keep the
  internal state compact (plain dicts and lists, no pydantic objects inside the hot
  loop).
- **Determinism:**
  - sort every iteration over trains by id;
  - never iterate a `set` unsorted;
  - floats only in cost.
- **Unfinished at shift end** (PRD §7.6, D6):
  - every train with `phase != finished`, including `not_entered`, is charged
    `UNFINISHED_COST + lateness(destination, actual = shift_ticks)`;
  - passenger trains use the terminus multiplier;
  - booked under `unfinished`.
- **Reversal** (PRD §7.5 rule 11, D10):
  - the train occupies the section behind for exactly `REVERSE_TICKS`, regardless of
    its length, then arrives at the chosen loop or yard;
  - its authority is cleared;
  - on arrival, `READY` fires;
  - the next timed stop is unchanged (it must run forward again);
  - `reversal` cost is charged at grant.

## 5. Tests (write these first)

Use `load_fixture("mini3")` and `Simulator.load(Scenario.model_validate(d))`.

### 5.1 `tests/t02/test_model.py`

| Id | Test | Assertion |
|---|---|---|
| T02-01 | `test_T02_01_fixtures_load` | All three fixtures validate; `canonical_json(model_dump)` of mini3 equals `canonical_json` of the raw fixture dict |
| T02-02 | `test_T02_02_extra_field_rejected` | Adding `"foo": 1` at top level, in a station, a train or a stop → `ValidationError` |
| T02-03 | `test_T02_03_section_chain` (parametrized) | Rejected: a missing section; sections out of order; a section `S1-S3` skipping S2; the first section not starting at `YW`; the last not ending at `YE` |
| T02-04 | `test_T02_04_loops` (parametrized) | Rejected: loop length 200; duplicate loop id in a station; 4 loops; kind `"siding"` |
| T02-05 | `test_T02_05_trains` (parametrized) | Rejected: id `X1`; id `E1` with type `local`; east train whose stops are out of route order; last stop not the destination yard (`YE` for east); `arr > dep`; `dep` null on a non-final stop; negative `entry_tick`; freight with an intermediate stop; duplicate train id |
| T02-06 | `test_T02_06_rules_version` | `rules_version` ≠ `constants.RULES_VERSION` → `ValidationError` mentioning `rules_version` |
| T02-07 | `test_T02_07_next_service` | Rejected: `next_service` pointing to a missing id; to a train in the same direction |
| T02-08 | `test_T02_08_unimplemented_factor` | `Simulator.load` on mini3 with `factors` += `["F5"]` raises `NotImplementedError` (this test is **deleted** in T03 and replaced by T03's factor tests; record that in T03's gate) |

### 5.2 `tests/t02/test_timing.py`

| Id | Test | Assertion |
|---|---|---|
| T02-09 | `test_T02_09_travel_ticks` (parametrized) | `(6000,120,140)→3`, `(6000,120,100)→4`, `(6000,120,70)→6`, `(6000,120,140,40)→9`, `(1,140,140)→1`, `(4500,80,100)→4`, `(8000,140,140)→4`, `(9500,140,70)→9`; return type is `int` |
| T02-10 | `test_T02_10_free_run_matches_fixture_timetable` | Free-run projection of the full route from entry gives exactly the mini3 timetable: E1 S2 arr 6 / dep 7, YE 13; L5 S3 4/5, S2 9/10, S1 14/15, YW 19; F21 YE 26 |

### 5.3 `tests/t02/test_authority.py` (fresh mini3 at t=0 unless stated)

| Id | Test | Setup → assertion |
|---|---|---|
| T02-11 | `test_T02_11_grant_message_and_path` | `Authorize(E1,S2,L1)` → `ok`, message `GRANTED E1 -> S2/L1 (arrive t=6)`; `state()` E1 `authority.path == ["YW-S1","S1/main","S1-S2","S2/L1"]` |
| T02-12 | `test_T02_12_unknown_args` | `(X9,S2,L1)` → `R-UNKNOWN-TRAIN`; `(E1,S9,main)` → `R-UNKNOWN-STATION`; `(E1,S2,L7)` → `R-UNKNOWN-TRACK`; `(E1,YE,L1)` → `R-UNKNOWN-TRACK` |
| T02-13 | `test_T02_13_not_in_network` | `(F21,S2,L2)` at t=0 → `R-NOT-IN-NETWORK`, message contains `enters at t=2` |
| T02-14 | `test_T02_14_wrong_direction` | `(E1,YW,main)` → `R-WRONG-DIRECTION` |
| T02-15 | `test_T02_15_loop_len` | `(E1,S3,L1)` → `R-LOOP-LEN` with message `REFUSED R-LOOP-LEN: E1 is 200 m, S3/L1 is 150 m`; after `advance()` to t=2: `(F21,S1,L1)` → `R-LOOP-LEN`; `(F21,S2,L2)` → granted |
| T02-16 | `test_T02_16_platform` | `(L5,S2,L2)` → `R-PLATFORM`; `(E1,S2,L2)` → `R-PLATFORM` |
| T02-17 | `test_T02_17_section_reserved` | Grant `(E1,S2,L1)`; then `(L5,S1,main)` → `REFUSED R-SECTION-RESERVED: S1-S2 reserved by E1` |
| T02-18 | `test_T02_18_main_blocked` | Grant `(L5,S2,main)`; then `(E1,YE,main)` → `REFUSED R-MAIN-BLOCKED: S2/main reserved by L5` |
| T02-19 | `test_T02_19_track_occupied` | Grant `(E1,S2,L1)`; then `(L5,S2,L1)` → `REFUSED R-TRACK-OCCUPIED: S2/L1 reserved by E1` |
| T02-20 | `test_T02_20_hold` | `Hold(E1,0)` at t=0 → `R-HOLD-PAST`; `Hold(E1,5)` → ok; `(E1,S2,L1)` → `REFUSED R-HELD: E1 held until t=5`; a hold does not cancel an existing authority (grant first, then hold → still has the authority) |
| T02-21 | `test_T02_21_cancel` | `CancelAuthority(E1)` with none → `R-NO-AUTHORITY`; grant `(E1,S2,L1)` then cancel at t=0 → authority null, all its resources free (L5 can now get `(L5,S1,main)`) |
| T02-22 | `test_T02_22_cancel_while_moving` | Grant `(E1,YE,main)`; `advance(max_tick=1)`; cancel → E1 authority becomes `S1/main` (the first station not yet passed); `S1-S2`, `S2/main`, `S2-S3`, `S3/main`, `S3-YE` are free; E1 then stops on `S1/main` at t=3 |
| T02-23 | `test_T02_23_refusal_no_state_change` | For every refusal in T02-12..T02-20: `canonical_json(state())` before == after |
| T02-24 | `test_T02_24_check_equals_apply` | For every case in T02-11..T02-21: `check(a)` has the same `ok`/`rule`/`message` as `apply(a)` on a clone, and `check` leaves `state()` unchanged |
| T02-25 | `test_T02_25_extension` | Grant `(E1,S2,L1)`; `advance()` (→ t=2, READY F21); `advance(max_tick=4)` (E1 in `S1-S2`); `Authorize(E1,YE,main)` → granted (extension); E1 does **not** stop in `S2/L1`; it stops at S2 on `main` for its timed stop (arr 6, dep 7) and arrives at YE at t=13; `S2/L1` is released at the extension; an extension to `S2` (not strictly beyond) → `R-WRONG-DIRECTION` |

### 5.4 `tests/t02/test_episodes.py` (exact scripted episodes)

**T02-26 `test_T02_26_episode_A_exact`** (mini3). Assert every step:

1. `load` → `tick == 0`.
2. `apply(Authorize(E1,S2,L1))` → ok.
3. `advance()` → `tick == 2`, reasons `[READY F21]`.
4. `apply(Authorize(L5,S2,main))` → ok, message ends `(arrive t=11)`.
5. `advance()` → `tick == 6`, reasons `[AUTHORITY_END E1]`; E1 `arrivals[-1] == {S2, 6, 6}`; L5 is `at_station S3/main`.
6. `apply(Authorize(E1,YE,main))` → `REFUSED R-SECTION-RESERVED: S2-S3 reserved by L5`.
7. `advance()` → `tick == 7`, reasons `[READY E1]`.
8. `advance()` → `tick == 11`, reasons `[AUTHORITY_END L5]`.
9. `apply(Authorize(E1,YE,main))` → ok, `(arrive t=17)`.
10. `end_shift()` → `done`, `tick == 60`. Cost:
    - `pax_lateness == 24.0` (E1 at YE late 4: 3.0×2×(4−2)×2);
    - `freight_lateness == 0.0`;
    - `unfinished == 495.0` (F21: 200 + 0.5×34 = 217; L5: 200 + 1.0×1×39×2 = 278);
    - `crew_timeout == reversal == 0.0`;
    - `total == 519.0`.

| Id | Test | Assertion |
|---|---|---|
| T02-27 | `test_T02_27_episode_B_gridlock_and_reverse` | mini2_gridlock: grant `(L1,S1,main)`, `(L2,S2,main)`; `advance()` → t=4, reasons `[AUTHORITY_END L1, AUTHORITY_END L2, GRIDLOCK]`; `state().gridlock == ["L1","L2"]`; `Reverse(L1)` → ok; `cost().reversal == 50.0`; `state().gridlock == []`; L1 `phase == "reversing"` in section `YW-S1` with `section_exit == 19`; `Reverse(L2)` → `R-NO-GRIDLOCK`; grant `(L2,S1,main)` → ok `(arrive t=9)`; `advance()` → t=9 `[AUTHORITY_END L2]`; `advance()` → t=10 `[READY L2]`; `advance()` → t=19 `[READY L1]` with L1 at `YW` |
| T02-28 | `test_T02_28_episode_C_no_reverse_path` | mini3_noloops: grant `(L1,S2,main)` and `(L2,S3,main)`; `advance()` → t=4 `[AUTHORITY_END L2]`; `advance()` → t=5 `[READY L2]`; `advance()` → t=9 `[AUTHORITY_END L1, GRIDLOCK]`; `Reverse(L1)` → `REFUSED R-NO-REVERSE-PATH: no free loop at S1`; `Reverse(L2)` → ok (to YE) |
| T02-29 | `test_T02_29_time_cap_and_shift_end` | mini3 with `shift_ticks=200`, no actions: `advance()` → t=2 `[READY F21]`; → t=62 `[TIME_CAP]`; → t=122 `[TIME_CAP]`; → t=182 `[TIME_CAP]`; → t=200 `[SHIFT_END]`, `done` |
| T02-30 | `test_T02_30_edge_triggers` | mini3, no actions: after the t=2 `READY F21`, no later result contains `READY F21` |
| T02-31 | `test_T02_31_advance_max_tick` | Grant `(E1,S2,L1)`; `advance(max_tick=5)` → stops early at `tick == 2` (`[READY F21]`: a trigger beats `max_tick`); `advance(max_tick=5)` again → `tick == 5`, `reasons == []`; E1 `in_section S1-S2` with `section_exit == 6` |
| T02-32 | `test_T02_32_do_nothing_cost` | mini3: `end_shift()` at t=0 → `unfinished == 3×200 + lateness-as-if-at-60`: E1 = 200 + 3.0×2×(47−2)×2 = 740; F21 = 217; L5 = 278 → `total == 1235.0` |
| T02-33 | `test_T02_33_actions_after_done` | After `end_shift()`, `apply(anything)` → `ok == False`, rule `R-SHIFT-ENDED`; `advance()` returns `done` with no change |
| T02-34 | `test_T02_34_no_early_departure` | mini3: grant `(L5,S2,main)` at t=0; L5 arrives S3 at 4 and departs at exactly 5 (dwell and `dep` both end at 5); a variant with S3 `dep: 8` → departs at 8, not 5 |

### 5.5 `tests/t02/test_cost.py`

| Id | Test | Assertion |
|---|---|---|
| T02-35 | `test_T02_35_pax_cost` (parametrized) | `pax_cost("express",200,late=7,terminus=False)==30.0`; same with `terminus=True` → `60.0`; `("local",100,2,False)==0.0`; `("local",150,10,True)==24.0`; `("express",100,-5,True)==0.0` |
| T02-36 | `test_T02_36_freight_cost` | late 10 → `5.0`; late −3 → `0.0` |
| T02-37 | `test_T02_37_total_is_sum` | After episode A, `total == sum(components)` exactly |

### 5.6 `tests/t02/test_gridlock.py`

| Id | Test | Assertion |
|---|---|---|
| T02-38 | `test_T02_38_no_false_positive_opposing_with_loop` | mini3: E1 at S2/L1 (no authority), L5 at S3/main (no authority) → `gridlock == []` (L5 can still enter S2/main) |
| T02-39 | `test_T02_39_timers_ignored` | In episode B at t=4 (dwell still running) gridlock is already reported |
| T02-40 | `test_T02_40_scc_only` | mini3_noloops plus L3 (east local, entry 0, same stops as L1). Grant `(L1,S2,main)`, `(L2,S3,main)`; `advance()` ×2 → t=5 (`READY L2`); grant `(L3,S1,main)` (S1/main was released when L1 left at 5); `advance()` → t=9 with `GRIDLOCK`; L3 (at S1/main) waits on L1 but is not in the cycle → `gridlock == ["L1","L2"]` |

### 5.7 `tests/t02/test_properties.py` (Hypothesis)

Strategy: on mini3 (with `shift_ticks` drawn from {60, 120}), a list of 1–40 steps. Each
step is one of:

- `Authorize(random train, random station or yard, random track in {main, L1, L2, L3})`;
- `Hold(random train, tick + 0..10)`;
- `CancelAuthority`;
- `Reverse`;
- `advance()`.

Invalid choices are welcome.

| Id | Test | Invariant checked after every step |
|---|---|---|
| T02-41 | `test_T02_41_one_train_per_resource` | No section or station track has two holders or two occupants |
| T02-42 | `test_T02_42_no_move_without_authority` | A train's position changes between steps only if it had an authority (or was reversing) at the start of the step |
| T02-43 | `test_T02_43_no_early_departure_property` | Every recorded departure from a timed stop is at `tick >= dep` and `>= arrival + min_dwell` |
| T02-44 | `test_T02_44_fit_and_platform` | Every loop occupant's length ≤ loop length; no express/local ever in a goods loop |
| T02-45 | `test_T02_45_cost_monotone` | Every component ≥ 0, non-decreasing between steps; `total == sum` |
| T02-46 | `test_T02_46_refusal_is_noop` | A refused `apply` leaves `canonical_json(state())` unchanged |
| T02-47 | `test_T02_47_replay_identical` | Replaying the same action list on a fresh `load` gives identical `state()` and `cost()` after every step |
| T02-48 | `test_T02_48_clone_independent` | Applying steps to a clone never changes the original's `state()` |
| T02-49 | `test_T02_49_cross_process_determinism` | Episode A run in 2 subprocesses with `PYTHONHASHSEED=1` / `=2` → identical `canonical_json(state())` + cost |

### 5.8 `tests/t02/test_speed.py`

| Id | Test | Assertion |
|---|---|---|
| T02-50 | `test_T02_50_clone_and_advance_speed` | On mini3 with `shift_ticks=360` and trains duplicated to 14 (ids renamed, entries spread): 1,000 × (`clone` + `advance()` to the end with no actions) completes in < 5 s on a laptop (marker `slow`). It is a guard for T06's rollouts; adjust only with a gate-record note |

## 6. Gate command

```bash
uv run pytest tests/t01 tests/t02 -m "not harbor and not model and not gpu"
```

## 7. Definition of done

- [ ] All T02 tests and all T01 non-marker tests pass.
- [ ] Hand-computed numbers in T02-26 and T02-32 match without changing the rules.
- [ ] `Simulator` public API matches PRD §7.4 exactly (names and signatures), except
      `from_observation`, which T03 adds.
- [ ] No unrevealed information in `state()`. In T02 there are no events, but there must
      be no `events`, `seed` or `scenario_id` field in `State` or `StaticInfo`.
- [ ] Gate record, reviewers, handover (no commit).
