# T04: `manual()`, rule-coverage test, scenario generator, presets, seed sets

| | |
|---|---|
| Depends on | T03 (all MVP factors in the simulator) |
| Unblocks | T05 (`manual` tool, T2 scenarios for calibration), T06 (seed filter needs candidate scenarios), T07 (export) |
| DESIGN.md | §1 (fairness: all rules in the manual), §2.6 (rule IDs), §3, §4 (lookaheads), §9.1 (generator), §9.2 (seed sets) |
| PRD | §7.2, §7.3, §7.6, §7.12, §8 D1, D9, D17 |
| Required markers | default only |
| Suggested split | A: `manual.py` + coverage test. B: generator network + trains + timetable. C: events, presets, seed sets, CLI, property tests |

## 1. Goal

Two things that make the environment *fair* and *dynamic*:

1. **`manual()`.** One text that states every rule, rule ID, cost formula, constant,
   lookahead, budget, decision trigger and output format. A test fails if the code
   enforces anything the manual doesn't mention.
2. **A deterministic, parameterized scenario generator.**
   - Named presets: T1, T2, T3, and T2-S14 as the structural holdout.
   - Non-overlapping seed sets: dev, train, held-out, fresh.
   - A `gen` CLI.

## 2. Context the implementer needs

### 2.1 Manual (`railway/sim/manual.py`)

`render_manual() -> str` builds markdown **from the code's own sources of truth**:

- the `RuleId` enum and its descriptions;
- the `CostComponent` formulas;
- `constants.py`;
- the decision-reason kinds.

Hand-written prose glues these together. Do not hard-code values that exist as
constants; interpolate them.

Required sections, in order (tests check the headings):

1. `# Dispatcher manual` — role and goal in 3 sentences.
2. `## Time and turns` — ticks, turn-based play, the same-tick effect.
3. `## Network` — yards, stations, loops (length, platform/goods), sections, main line,
   crew stations.
4. `## Train types` — table: length, max speed, restart ticks, delay weight, stopping
   pattern.
5. `## Movement authority` — PRD §7.5 rules 1–13 in plain words, including extension,
   cancel, hold and reverse.
6. `## Rules` — table: every `RuleId`, its description, refusal or enforced.
7. `## Factors` — one paragraph per enabled-capable factor. It says that `network()`
   lists the factors active in this shift.
8. `## Events and lookaheads` — the table from DESIGN.md §4 with the constants.
9. `## Decision points` — the §7.7 table.
10. `## Cost` — every component, formula and constant.
11. `## Score` — "Your score compares your total cost with a simple greedy dispatcher
    and with a strong reference dispatcher: matching the reference scores 1, doing no
    better than greedy scores 0."
12. `## Budgets and ending` — the tool budget, what happens when it runs out, and that
    stopping without `end_shift()` counts as `end_shift()`.
13. `## Tools` — every tool, its arguments and output format, including the compact
    status line grammar (PRD §7.8.1) and the advance output.

**Size:** ≤ 16,000 characters (it's read once per episode; about 4k tokens).

**No secrets:** the manual must not mention seeds, the generator or scenario files.

### 2.2 Generator (`railway/gen/`)

`GenConfig` (pydantic):

| Field | Meaning |
|---|---|
| `preset` | |
| `stations` | |
| `trains` | |
| `mix` | `{"express": x, "local": y, "freight": z}`, summing to 1 |
| `factors` | |
| `shift_ticks` | |
| `event_density` | Default 1.0 |
| `loops` | Distribution, below |
| `major_fraction` | Default 0.34 |
| `slack` | Timetable slack, default 0.10 |
| `rotation_fraction` | Default 0.4 |
| `tight_crew_fraction` | Default 0.3 |

**Presets** (`railway/gen/presets.py`):

| Preset | Stations | Trains | Shift | Factors | Notes |
|---|---|---|---|---|---|
| T1 | 8 | 8 | 120 | F1–F4 | Pilot, quick checks |
| T2 | 12 | 14 | 360 | All 11 MVP | Judges, headline |
| T2-S14 | 14 | 16 | 360 | All 11 MVP | Structural holdout, never trained on |
| T3 | 16 | 22 | 720 | All 11 MVP (stretch factors later) | Showcase |

`generate(config: GenConfig, seed: int) -> Scenario`. Use one `random.Random(seed)` and
draw everything from it **in a fixed order**:

1. **Network**
   - **Sections:** length uniform from {4000, 4500, …, 9000} m. Line speed from
     {80, 100, 120, 140}.
   - **Loops per station:** 0–3 with weights {0: 0.15, 1: 0.45, 2: 0.30, 3: 0.10}.
     Loop length from {150: 0.25, 300: 0.45, 700: 0.30}. A 700 m loop is `goods` with
     probability 0.5, otherwise `platform`. Shorter loops are `platform`.
   - **Repairs** (deterministic, after drawing):
     - make sure at least `ceil(N/3)` stations have a 700 m loop;
     - make sure no two neighbouring stations both have zero loops;
     - make sure at least one 700 m loop is in each third of the line.
     Repairs add or upgrade loops scanning west → east.
   - **Major stations:** `major_fraction`, always including the first and last station.
   - **Crew stations** (only if F10): 2–3 stations with at least one loop, spread over
     the line.
2. **Trains**
   - **Types:** from `mix` (largest-remainder rounding).
   - **Ids:** `E1..`, `L1..`, `F1..`, numbered per type.
   - **Direction:** alternating after a shuffle, so the split is balanced within ±1.
   - **pax:** express U[150, 400], local U[50, 200], step 10.
3. **Timetable**
   - Free-run times (T02 `timing`) from the origin.
   - Each section's scheduled time is `ceil(free × (1 + slack))`.
   - Dwell is `min_dwell` (local 1–2, express 1).
   - **Entry:** uniform in `[0, shift − scheduled_duration − 15]`, so every train is
     scheduled to finish by `shift − 15`.
4. **Rotations** (only if F6): pair arrivals and departures at the same yard in opposite
   directions where `B.entry ≥ A.sched_arr + TURNAROUND`. Pairing is greedy by time,
   stopping at `rotation_fraction` of trains.
5. **Crew** (only if F10)
   - `tight_crew_fraction` of first services get `crew_used_at_entry` such that the
     crew would time out between 40% and 80% of the way along the scheduled run.
   - The rest get U[0, 100].
   - Successors in a rotation get 0 in the file; the simulator carries the crew over.
6. **Events** (only for enabled factors). The count per kind is
   `round(event_density × shift / 120)`, at least 1.
   - **Possession (F7):** a random section; duration U[15, 40]; start U[60, shift − 40];
     reveal `start − U[45, 60]`, at least 0.
   - **Restriction (F8):** 1–2 adjacent sections; speed from {40, 60}; window length
     U[30, 90]; start U[30, shift − 30]; reveal `start − U[30, 60]`, at least 0.
   - **Breakdown (F15):** a random train; tick inside its scheduled run, between
     `entry + 5` and `sched_arr − 5`; duration U[5, 25].
   - **Late entry (F16):** a random train with original entry ≥ 20 (D17); delay
     U[10, 40]; reveal = original entry − 20. Skip it if no train qualifies.
   - **Event ids:** `EV1..` after sorting by `(reveal_tick, kind, target)`.
7. **Finishing**
   - `scenario_id = f"{preset}-{content_hash(scenario without scenario_id)[:12]}"`.
   - Validate through `Scenario.model_validate`.

**The generator never emits engineering trains (D9).**

### 2.3 Seed sets (`railway/gen/seeds.py` + `seeds/ranges.json`)

| Set | Candidate range | Use |
|---|---|---|
| `dev` | `[0, 10_000)` | Tuning, pilot |
| `train` | `[10_000, 1_000_000)` | Training only |
| `heldout` | `[1_000_000, 1_010_000)` | Reported numbers, judges |
| `fresh` | `[2_000_000, 2**31)` | `--fresh` draws, never committed |

The ranges never overlap. `candidate_seeds(set_name)` yields seeds in ascending order. The
seed filter (T06) turns candidates into accepted lists `seeds/<preset>-<set>.json`.
T2-S14 uses the `heldout` range with its own preset, so its scenarios differ
structurally.

### 2.4 CLI (`gen`, console script → `railway.gen.cli:main`)

| Command | Effect |
|---|---|
| `gen sample --config T1 --seed 5 --out x.json` | Writes one scenario (canonical JSON + newline) |
| `gen describe x.json` | Prints a one-screen summary (stations, loops, trains by type, events by kind) |
| `gen candidates --set dev --count 10` | Prints seeds |

`gen export` is added in T07.

## 3. Scope

**In scope:**

- `manual.py`;
- the rule-coverage test;
- `gen/config.py`, `presets.py`, `generator.py`, `seeds.py`, `cli.py`;
- `seeds/ranges.json`;
- property tests.

**Out of scope:**

- The seed filter and accepted lists (T06).
- Export (T07).
- Tuning the difficulty to pass the pilot (T09). You may tune defaults here only to make
  property tests pass.

## 4. Implementation notes and pitfalls

- **Fixed draw order.** Adding a draw in the middle changes every later scenario. Once
  T06 has accepted lists, any generator change means regenerating them (the gate record
  must say so). Draw per stage from `random.Random(f"{seed}:{stage}")`, a seed string
  hashed via `random.Random`'s own str-seeding, which is stable. Then later stages don't
  shift when an earlier stage changes.
- **`random.Random(str)` seeding is deterministic across processes** (it uses sha512,
  not `hash()`). `hash()` is not.
- The generator must only use T02/T03 timing functions for free-run times. **Don't
  reimplement travel time.**
- **The coverage test scans source text.** Keep rule IDs as `RuleId.X` references or
  `"R-..."` literals so the scan finds them.

## 5. Tests (write these first)

### 5.1 `tests/t04/test_manual.py`

| Id | Test | Assertion |
|---|---|---|
| T04-01 | `test_T04_01_headings_in_order` | `render_manual()` contains the 13 headings of §2.1, in order |
| T04-02 | `test_T04_02_every_rule_id` | Every `RuleId` value (e.g. `R-LOOP-LEN`) appears in the manual |
| T04-03 | `test_T04_03_no_unknown_rule_ids_in_code` | Regex-scanning `railway/**/*.py` for `R-[A-Z][A-Z-]+` and `RuleId\.([A-Z_]+)` → every hit is an enum member |
| T04-04 | `test_T04_04_every_cost_component_and_constant` | Every `CostComponent` name and every public constant in `constants.py` appears with its current value (e.g. `TOOL_BUDGET` → `400`, `CREW_LIMIT` → `240`, `W_EXPRESS` → `3.0`) |
| T04-05 | `test_T04_05_every_reason_kind` | Every decision-reason kind in §7.7 appears |
| T04-06 | `test_T04_06_every_tool` | All 15 tool names of PRD §7.8 appear in the Tools section |
| T04-07 | `test_T04_07_size_and_secrets` | `len(manual) ≤ 16000`; it contains none of `seed`, `generator`, `scenario.json`, `heldout` (case-insensitive) |
| T04-08 | `test_T04_08_deterministic` | Two renders are equal; the text contains no timestamp |
| T04-09 | `test_T04_09_constant_change_propagates` | Monkeypatching `constants.TOOL_BUDGET = 123` → the manual shows `123` (proves interpolation) |

### 5.2 `tests/t04/test_generator.py`

| Id | Test | Assertion |
|---|---|---|
| T04-10 | `test_T04_10_presets_exist` | `PRESETS` has `T1`, `T2`, `T2-S14`, `T3` with the §2.2 table values |
| T04-11 | `test_T04_11_deterministic_same_process` | `generate(T1, 5)` twice → equal canonical JSON |
| T04-12 | `test_T04_12_deterministic_cross_process` | `gen sample` in 2 subprocesses (`PYTHONHASHSEED` 1 and 2) for T2 seed 7 → byte-identical files |
| T04-13 | `test_T04_13_different_seeds_differ` | T2 seeds 1..20 → 20 distinct `scenario_id`s |
| T04-14 | `test_T04_14_scenario_id_opaque` | `scenario_id` matches `^T2-[0-9a-f]{12}$` and does not contain the seed's decimal string |
| T04-15 | `test_T04_15_validates` | Every generated scenario passes `Scenario.model_validate` and `Simulator.load` |

### 5.3 `tests/t04/test_generator_properties.py`

Hypothesis over `seed` in [0, 10^6) and every preset; `max_examples` = 50 per preset.

| Id | Invariant |
|---|---|
| T04-16 | Station, train and section counts match the preset; each type count is within ±1 of `mix × trains`; the east/west split is within ±1 |
| T04-17 | Loops: ≤ 3 per station; at least `ceil(N/3)` stations with a 700 m loop; no two neighbouring stations both loop-less; at least one 700 m loop in each third of the line |
| T04-18 | Every train's free-run schedule fits: each stop's `arr` ≥ free-run arrival; the last stop's `arr ≤ shift − 15` |
| T04-19 | Express trains stop only at major stations; locals at every station; freight at none; `pax` ranges hold |
| T04-20 | Factors off → no corresponding data: no events of that kind; `next_service` all null without F6; `crew_used_at_entry` all 0 and no crew stations without F10 |
| T04-21 | Rotations: opposite direction, same yard, `B.entry ≥ A.last_arr + TURNAROUND`; no train is the successor of two trains |
| T04-22 | Events: lookaheads within the ranges (possession reveal in [start−60, start−45] or 0; restriction in [start−60, start−30] or 0); late entries only for trains with original entry ≥ 20 and reveal = entry − 20; breakdown ticks inside the target's scheduled run |
| T04-23 | No engineering trains |
| T04-24 | `Simulator.load(s).end_shift()` completes (do-nothing cost is finite and > 0) |

### 5.4 `tests/t04/test_seeds.py`

| Id | Test | Assertion |
|---|---|---|
| T04-25 | `test_T04_25_ranges_disjoint` | Ranges in `seeds/ranges.json` are pairwise disjoint and match §2.3 |
| T04-26 | `test_T04_26_candidates_in_range` | The first 100 `candidate_seeds("heldout")` are inside the held-out range, ascending |
| T04-27 | `test_T04_27_fresh_never_overlaps` | 1,000 fresh draws (seeded RNG) are all ≥ 2,000,000 |

### 5.5 `tests/t04/test_cli.py`

| Id | Test | Assertion |
|---|---|---|
| T04-28 | `test_T04_28_sample_writes_canonical` | `gen sample --config T1 --seed 5 --out tmp` → the file equals `canonical_json(generate(T1,5)) + "\n"` |
| T04-29 | `test_T04_29_describe` | `gen describe` output contains the station count, train counts per type and event counts per kind |
| T04-30 | `test_T04_30_bad_config` | `gen sample --config T9` → non-zero exit, message lists the valid presets |

### 5.6 `tests/t04/test_t2_runs.py` (`slow`)

| Id | Test | Assertion |
|---|---|---|
| T04-31 | `test_T04_31_t2_do_nothing_fast` | 20 T2 dev seeds: `load` + `end_shift` each in < 1 s |

## 6. Gate command

```bash
uv run pytest tests/t01 tests/t02 tests/t03 tests/t04 -m "not harbor and not model and not gpu"
```

## 7. Definition of done

- [ ] All tests pass.
- [ ] `gen sample` and `gen describe` work for every preset.
- [ ] `docs/gates/T04.md` includes one `gen describe` output for a T2 seed, so the team
      can eyeball the realism.
- [ ] Gate record, reviewers, handover (no commit).
