# T03: Factors F5–F8, F10, F15, F16; events; belief simulator

| | |
|---|---|
| Depends on | T02 (simulator core, green) |
| Unblocks | T04 (generator emits these factors), T05 (tools show them), T06 (reference uses the belief simulator) |
| DESIGN.md | §2.3 (types, restart, no early departure), §3 (F5–F8, F10, F15, F16), §4 (events and lookaheads) |
| PRD | §7.2 (constants), §7.3 (events), §7.5, §7.6, §7.7, §8 D4, D5, D7, D8, D13, D17 |
| Required markers | default only |
| Suggested split | A: F5, F8, F15 (timing effects). B: F6, F10, F16 (entry and crew). C: F7 (possessions + cuts) + `from_observation` + no-leak property tests |

## 1. Goal

Complete the 11-factor MVP inside the simulator, add revealed events, and add a
**belief simulator**, `Simulator.from_observation(static, state)`. It rebuilds a runnable
simulator from what the agent can see. That belief simulator is how the reference (T06)
plans "seeing only what the agent sees". It is also how DAgger (T10) labels states.

## 2. Context the implementer needs

### 2.1 Factor rules (normative; PRD §7.5 and §8 for the decisions)

| Factor | Rule |
|---|---|
| **F5 restart** | **Unscheduled stop:** a train comes to rest anywhere except (a) its origin yard before first departure, or (b) a timed stop in its `stops`. Rest also covers a mid-section stop by breakdown or crew timeout. **Cost:** after such a stop, its next movement adds `RESTART_TICKS[type]` to the remaining travel of the current section (mid-section), or to the first section it enters (from a station). **Extension:** a granted extension that removes the stop removes the penalty. State: `restart_pending: bool` |
| **F6 rotations** | **Rule:** if `A.next_service == B`, then `B` cannot enter before `A.finished_tick + TURNAROUND`. Its effective `entry_tick = max(B.entry_tick, B.late_entry if revealed, A.finished_tick + TURNAROUND)`. If A never finishes, B never enters (both unfinished at shift end). **State:** while A is unfinished, B shows `waits_for: "A"`. **F6 off:** `next_service` is ignored |
| **F7 possessions** | **Grant check:** after reveal, a grant whose projected occupation of the section overlaps `[start, end)` is refused `R-POSSESSION`. **Cut back:** on reveal, every existing authority whose not-yet-entered path would overlap the window (free-run projection) is cut back to the last station before the possession section that the train hasn't passed. The new destination track is that station's `main` (`R-AUTHORITY-CUT`), and a change line is added: `AUTHORITY_CUT <train> -> <station>/main: possession <section> <start>-<end>`. **No entry:** no train may *enter* the section during the window. **Occupied at start:** if a train is inside the section at `start` (breakdown or timeout), the possession becomes active when that train exits; the end is unchanged. State shows the effective window |
| **F8 speed restrictions** | **Travel time:** a train entering a listed section at tick `e` with `start ≤ e < end` uses `min(line, train, speed_kmh)`. **Already inside:** a train already inside at `start` is unaffected. **Projections** used by grant checks and messages include revealed restrictions |
| **F10 crew** | **Counter:** `crew_used(t) = crew_used_at_entry + (t − entry)` while in the network; it reaches the limit at `entry + CREW_LIMIT − crew_used_at_entry`. **Warning:** `CREW_WARNING` fires when remaining first becomes ≤ 30; already ≤ 30 at entry means it fires at the entry tick. **Timeout:** at remaining = 0 → `CREW_TIMEOUT`, cost `TIMEOUT_COST`. The train stands still where it is, mid-section included, for `TIMEOUT_BLOCK` ticks (`blocked_until`). Its section exit moves out by the same amount. At the block's end a fresh crew starts (used = 0). **Relief:** `relieve_crew` only when stationary (`R-NOT-STATIONARY`) at a `crew_station` (`R-NOT-CREW-STATION`; yards are not crew stations). The crew resets to 0 **at relief start**; the train cannot depart for `RELIEF_TICKS` (`relief_until`). **Rotations:** crews carry over. `B.crew_used at entry = A.crew_used at finish + (B.entry − A.finished_tick)`. **F10 off:** no crew fields (null) and no crew triggers |
| **F15 breakdowns** | **When it applies:** at `tick`, if the train is in the network and unfinished, it stops for `duration` (`breakdown_until`), mid-section included (section exit += duration). It cannot depart a station during it. **Reveal:** `BREAKDOWN` fires with detail `until t=<tick+duration>`. **Otherwise:** if the train isn't in the network at `tick`, the event is skipped silently: never revealed, never shown |
| **F16 late entries** | **Reveal** at `reveal_tick` (= original entry − 20) → `BULLETIN`. **After reveal:** the train's effective entry becomes `new_entry_tick` (still subject to F6). **Before reveal:** state shows the original entry |

### 2.2 Event validation additions (`model.py`)

- **possession:**
  - `reveal_tick ≤ start < end ≤ shift_ticks`;
  - the section exists.
- **speed_restriction:**
  - `reveal_tick ≤ start < end`;
  - sections exist and are distinct;
  - `0 < speed_kmh < 140`.
- **late_entry:**
  - the train exists;
  - `new_entry_tick > original entry_tick`;
  - `reveal_tick < original entry_tick`.
- **breakdown:**
  - `reveal_tick == tick`;
  - `duration ≥ 1`.
- **Factor coupling:**
  - an event kind whose factor is off → `ValidationError`;
  - `factors` outside the MVP set → `ValidationError`;
  - the T02 `NotImplementedError` path is removed, and test T02-08 with it (record this
    in the gate record).

### 2.3 State additions (`observe.py`)

- `TrainState` gets:
  - `restart_pending`;
  - `crew_used` and `crew_remaining` (null if F10 off);
  - `relief_until`, `breakdown_until`, `blocked_until`;
  - `waits_for`.
- `State` gets:
  - `restrictions`: revealed, `{id, sections, speed_kmh, start, end}`;
  - `possessions`: revealed, `{id, section, start (effective), end}`;
  - `last_decision_tick`.

### 2.4 Belief simulator (`Simulator.from_observation(static, state)`)

- **Inputs:** only `StaticInfo` + `State`. Never the scenario.
- **What it builds:** a `Simulator` whose future contains:
  - only the revealed possessions and restrictions;
  - the known effective entries;
  - current breakdowns, blocks and reliefs with their known end ticks;
  - **no** future breakdowns and **no** unrevealed events.
- **When to call it:** only at decision points.
  - The current tick's triggers count as already reported.
  - `last_decision_tick` is taken from the state.
- **Exactness:** when the true scenario has no unrevealed events left, the belief
  simulator and the true one must evolve **identically** under the same actions.

## 3. Scope

**In scope:**

- `events.py`: reveal processing and the authority cut.
- Changes in `engine.py`, `timing.py`, `cost.py`, `observe.py` and `model.py` for the
  rules above.
- `from_observation`.
- Fixture variants built with `variant()` in the tests (no new fixture files needed).

**Out of scope:** text formats (T05), the generator (T04), `manual()` (T04).

## 4. Implementation notes and pitfalls

- **Possession timing.** Keep projections in one function (`timing.project`). It is used
  by:
  - grant checks;
  - grant messages;
  - authority cuts;
  - the `delay` field;
  - the reference.

  Different copies of "projected times" are a classic source of non-determinism bugs.
- **Mid-section stops.** Store `section_exit` as an absolute tick and shift it. Never
  recompute from speed: a restriction applies at entry only.
- **Order inside a tick** (PRD §7.5 rule 9). Reveals (phase 1) come before breakdowns
  (phase 2). A possession revealed at t cuts authorities before any departure at t.
- **Rotation entry.** A successor's `READY` fires at its effective entry. If the
  predecessor finishes late, `READY` is later. There is no separate trigger for "entry
  moved".
- **Crew counter.** Derive it from ticks, not by incrementing on every visited tick.
  `advance()` jumps over ticks, so incrementing per visited tick goes wrong.
- **Event-driven jumps.** Every new timer is a candidate next-event tick:
  - `breakdown_until`, `blocked_until`, `relief_until`;
  - crew warning and timeout ticks;
  - reveal ticks;
  - effective entries;
  - possession start and end.

## 5. Tests (write these first)

The base is mini3 (PRD §7.11). `vf(**patches)` means `variant(mini3, **patches)` plus
the listed factors added to `factors`.

**Extra trains used below:**

- **L6:** local, east, pax 100, entry 20. Stops: S1 24/25, S2 29/30, S3 34/35, YE 39.
- **F22:** freight, east, entry 30. Stops: YE 54.

### 5.1 `tests/t03/test_f5_restart.py` (factors F1–F5)

| Id | Test | Assertion |
|---|---|---|
| T03-01 | `test_T03_01_freight_restart` | t=2 grant `(F21,S2,L2)`; `advance()` → t=14 `[AUTHORITY_END F21]`; F21 `restart_pending`; grant `(F21,YE,main)` → `(arrive t=30)`; `end_shift()` → `freight_lateness == 2.0`. With F5 off: `(arrive t=26)`, `freight_lateness == 0.0` |
| T03-02 | `test_T03_02_extension_avoids_restart` | t=2 grant `(F21,S2,L2)`; `advance(max_tick=8)`; grant `(F21,YE,main)` (extension) → arrives YE at 26; `freight_lateness == 0.0`; never `restart_pending` |
| T03-03 | `test_T03_03_express_restart_one_tick` | t=0 grant `(E1,S1,L1)` (S1 is not an E1 stop) → arrives 3; grant `(E1,S2,L1)` at t=3 → `(arrive t=7)` |
| T03-04 | `test_T03_04_no_penalty_at_timed_stop_or_origin` | Episode A of T02 with F5 on gives the same total, 519.0 (all stops in it are timed stops or origins) |

### 5.2 `tests/t03/test_f6_rotation.py` (F1–F4, F6; add L6; `L5.next_service = "L6"`)

| Id | Test | Assertion |
|---|---|---|
| T03-05 | `test_T03_05_successor_waits` | t=0 grant `(L5,YW,main)` (arrive 19); `advance()` → t=2 `[READY F21]`; → t=19 `[AUTHORITY_END L5]`; → t=29 `[READY L6]`; `state()` before t=19 shows L6 `waits_for == "L5"`, after t=19 shows `entry_tick == 29` |
| T03-06 | `test_T03_06_f6_off` | Same scenario without F6 → `READY L6` at t=20 |
| T03-07 | `test_T03_07_predecessor_unfinished` | No actions; `end_shift()` → L6 `phase == "not_entered"`; `unfinished` includes L6: 200 + 1.0×1×(60−39−2)×2 = 238.0 |
| T03-08 | `test_T03_08_validation` | `next_service` with the same direction → `ValidationError`; pointing to an unknown id → `ValidationError` |

### 5.3 `tests/t03/test_f7_possession.py` (F1–F4, F7)

| Id | Test | Assertion |
|---|---|---|
| T03-09 | `test_T03_09_boundary_grant` | Possession `S2-S3 [20,40)` reveal 0. `state().possessions` has it at t=0; t=2 grant `(F21,YE,main)` → granted (F21 occupies S2-S3 [14,20): exit 20 is not > start 20) |
| T03-10 | `test_T03_10_overlap_refused` | Same, F21 `entry_tick=3`, YE arr 27 → at t=3 `(F21,YE,main)` → `REFUSED R-POSSESSION` and the message names `S2-S3` and `20-40`; `(F21,S2,L2)` → granted |
| T03-11 | `test_T03_11_authority_cut` | Possession `S2-S3 [12,30)` reveal 5. t=2 grant `(F21,YE,main)` → granted (not yet revealed). `advance()` → t=5 `[BULLETIN]` with a change line `AUTHORITY_CUT F21 -> S2/main: possession S2-S3 12-30`; F21 `authority.dest == S2/main`; `S2-S3`, `S3/main`, `S3-YE` free. `advance()` → t=14 `[AUTHORITY_END F21]`; `(F21,YE,main)` at t=14 → `R-POSSESSION` |
| T03-12 | `test_T03_12_no_entry_during_window` | Hypothesis over random action sequences on the T03-11 scenario: no train ever enters `S2-S3` while `12 ≤ tick < 30` |
| T03-13 | `test_T03_13_unrevealed_not_checked` | Possession `S2-S3 [20,40)` reveal 15: at t=3 (F21 entry 3) `(F21,YE,main)` → **granted** (no leak through refusals); at reveal (t=15) the authority is cut back to `S2/main` |

### 5.4 `tests/t03/test_f8_restriction.py` (F1–F4, F8)

| Id | Test | Assertion |
|---|---|---|
| T03-14 | `test_T03_14_restriction_applies_on_entry` | Restriction `["S1-S2"] 40 km/h [3,60)` reveal 0: `(E1,S2,L1)` → `(arrive t=12)`; `end_shift()` → E1 S2 arrival 12; `pax_lateness` includes 3.0×2×(6−2) = 24.0 for that stop |
| T03-15 | `test_T03_15_boundary_not_applied` | Window `[4,60)` → E1 enters S1-S2 at 3 → `(arrive t=6)` |
| T03-16 | `test_T03_16_revealed_after_grant` | Restriction `["S2-S3"] 40 [5,60)` reveal 1; t=0 grant `(E1,YE,main)` (message says arrive 13); the actual E1 YE arrival is 19 (S2-S3 takes 9 ticks from 7 to 16, then 3 ticks to 19) |
| T03-17 | `test_T03_17_state_lists_restriction` | `state().restrictions` empty before the reveal and lists the restriction from the reveal tick on |

### 5.5 `tests/t03/test_f10_crew.py` (F1–F4, F10; S2 is the only crew station)

| Id | Test | Assertion |
|---|---|---|
| T03-18 | `test_T03_18_timeout_mid_section` | E1 `crew_used_at_entry=235`: t=0 grant `(E1,S2,L1)`; `advance()` → t=2; `advance()` → t=5 `[CREW_TIMEOUT E1]`; `crew_timeout == 300.0`; E1 `in_section S1-S2`, `section_exit == 51`, `blocked_until == 50`; S1-S2 is held by E1 until 51 |
| T03-19 | `test_T03_19_warning` | `crew_used_at_entry=200`: grant `(E1,S2,L1)`; advances → t=2 `[READY F21]`, t=6 `[AUTHORITY_END E1]`, t=7 `[READY E1]`, t=10 `[CREW_WARNING E1]` with `crew_remaining == 30` |
| T03-20 | `test_T03_20_relief` | `crew_used_at_entry=200`: at t=6 (E1 at S2/L1) `RelieveCrew(E1)` → ok; `crew_used == 0`, `relief_until == 14`; grant `(E1,YE,main)` at t=6 → E1 departs at 14, arrives YE at 20; no `CREW_WARNING` ever; `pax_lateness` YE = 3.0×2×(7−2)×2 = 60.0 |
| T03-21 | `test_T03_21_relief_refusals` | At S1 → `R-NOT-CREW-STATION`; in a section → `R-NOT-STATIONARY`; at a yard → `R-NOT-CREW-STATION` |
| T03-22 | `test_T03_22_crew_carries_over_rotation` | F6+F10, L6 added, `L5.next_service = "L6"`, `L5.crew_used_at_entry=100`; grant `(L5,YW,main)` at 0 → L6 enters at 29 with `crew_used == 129` |
| T03-23 | `test_T03_23_f10_off_no_crew` | F10 off → `crew_used is None` for all trains; no crew triggers in a 60-tick run |

### 5.6 `tests/t03/test_f15_breakdown.py` (F1–F4, F15)

| Id | Test | Assertion |
|---|---|---|
| T03-24 | `test_T03_24_mid_section` | Breakdown E1 tick 4, duration 10: grant `(E1,S2,L1)`; `advance()` → t=2; `advance()` → t=4 `[BREAKDOWN E1]` with detail `until t=14`; E1 `section_exit == 16`; E1 S2 arrival = 16; S2 stop cost 3.0×2×(10−2) = 48.0 |
| T03-25 | `test_T03_25_not_in_network_skipped` | Breakdown F21 tick 1 (F21 enters at 2) → no `BREAKDOWN` ever; never visible in `state()` at any tick |
| T03-26 | `test_T03_26_at_station` | Breakdown E1 tick 7, duration 5 (tick 6 would hit E1 still in the section: breakdowns run in phase 2, before arrivals in phase 3): grant `(E1,S2,L1)`; E1 at S2/L1 from 6; grant `(E1,YE,main)` at t=6 → E1 departs at 12, not 7 |
| T03-27 | `test_T03_27_with_restart` | T03-24 with F5 on → `section_exit == 17` |
| T03-28 | `test_T03_28_possession_deferred` | F7+F15: possession `S1-S2 [10,20)` reveal 0; breakdown E1 tick 4, duration 10; grant `(E1,S2,L1)` at 0 (granted: projection [3,6) is clear); E1 exits S1-S2 at 16; `state().possessions[0].start == 16` from then on; `end == 20` |

### 5.7 `tests/t03/test_f16_late_entry.py` (F1–F4, F16; add F22; `shift_ticks=90`)

| Id | Test | Assertion |
|---|---|---|
| T03-29 | `test_T03_29_late_entry` | Event `late_entry F22 → 45`, reveal 10; no actions: `advance()` → t=2 `[READY F21]`; → t=10 `[BULLETIN]`; → t=45 `[READY F22]`; F22 `entry_tick` is 30 in `state()` at t<10 and 45 from t=10 |
| T03-30 | `test_T03_30_late_entry_with_rotation` | F6+F16: L6 late-entry → 25 (reveal 0) and `L5.next_service = "L6"`; grant `(L5,YW,main)` at t=0 so L5 finishes at 19 → L6 enters at max(25, 19+10) = 29 |

### 5.8 `tests/t03/test_validation.py`

| Id | Test | Assertion |
|---|---|---|
| T03-31 | `test_T03_31_event_validation` (parametrized) | Each rule in §2.2 rejected with `ValidationError`: possession reveal > start; start ≥ end; unknown section; restriction speed 0; duplicate sections; late entry not later; late entry reveal ≥ entry; breakdown reveal ≠ tick; F7 event with F7 off; factor `"F9"` |

### 5.9 `tests/t03/test_belief.py`

| Id | Test | Assertion |
|---|---|---|
| T03-32 | `test_T03_32_roundtrip` | Hypothesis: on mini3 with factors F1–F8, F10 (+L6 rotation, a possession and a restriction with reveal 0), random action sequences to a decision point; `from_observation(static, state).state()` == `state()` (canonical JSON) |
| T03-33 | `test_T03_33_exact_when_nothing_hidden` | Same scenarios (no breakdowns, all reveals ≤ the current tick): apply the same random future actions to the real simulator and the belief simulator → identical `AdvanceResult` ticks and reasons, identical final `cost()` |
| T03-34 | `test_T03_34_no_hidden_future` | With an unrevealed breakdown (E1 tick 20): the belief simulator built at t=6 has no breakdown in its future; its `end_shift()` cost differs from the true one; neither `state()` nor `static_info()` contains `"breakdown"` before t=20 |
| T03-35 | `test_T03_35_static_info_has_no_events` | `canonical_json(static_info())` has no `events`, `seed`, `scenario_id` or `reveal_tick` |

### 5.10 `tests/t03/test_no_leak.py` (the core fairness property)

| Id | Test | Assertion |
|---|---|---|
| T03-36 | `test_T03_36_unrevealed_event_invisible` (Hypothesis) | For each event kind (possession, restriction, late entry, breakdown) with reveal tick R: build scenario X with the event and Y without it. Under the same random action sequence, `canonical_json(state())` and every `ActionResult` are identical between X and Y at every step while `tick < R` |

### 5.11 `tests/t03/test_properties_all_factors.py`

| Id | Test | Assertion |
|---|---|---|
| T03-37 | `test_T03_37_t02_invariants_with_all_factors` | Re-run invariants T02-41 to T02-48 (import the helpers) on mini3 + L6 + F22 with all 11 factors and one event of each kind |
| T03-38 | `test_T03_38_cross_process_determinism_all_factors` | Same as T02-49 on the all-factor scenario |

## 6. Gate command

```bash
uv run pytest tests/t01 tests/t02 tests/t03 -m "not harbor and not model and not gpu"
```

## 7. Definition of done

- [ ] All tests pass. T02-08 is removed and the removal is recorded.
- [ ] Every T02 test still passes unchanged (F5+ off by default in the fixtures).
- [ ] `from_observation` uses only `StaticInfo` + `State` (enforced by its signature;
      reviewers check that there's no back door to the scenario).
- [ ] Every rule in PRD §7.6 marked (F5)–(F16) is enforced and referenced in code by its
      `RuleId`.
- [ ] Gate record, reviewers, handover (no commit).
