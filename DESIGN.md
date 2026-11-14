# Single-Track Railway Dispatcher — Environment Design (v0.2)

An RL environment for the AfterQuery "Break the Agent" challenge. The agent is the
dispatcher of a single-track railway for one service shift. It decides which train
waits where, so that trains in opposite directions can pass, express trains keep time,
freight keeps moving, crews don't time out and nothing gridlocks.

The environment is **fully deterministic**: a seed fixes the network, timetable and
every disruption. The same actions always produce the same trajectory and score, so any
run (model, human, scripted reference) can be replayed and re-scored exactly.

It ships as **Harbor tasks** (one task per seed), played by the **OpenCode** harness,
with the simulator behind an **MCP server in its own container**.

> v0.2 folds in the design review: a greedy-anchored score, an 11-factor MVP, a decision
> budget, the Harbor architecture, the evaluation and run system, and the team plan.
> Section 13 lists every decision made.

---

## 1. Design goals

| Goal | How the design meets it |
|---|---|
| A measurable gap between models | Many interacting constraints, decisions whose effects land 20–240 ticks later, trade-offs between train types |
| Gap comes from reasoning, not memory | The full current state is always one tool call away; episodes are capped at about 60 decisions; a windowed mode removes history altogether |
| Fair | Every rule is in `manual()` (enforced by a test); scheduled disruptions are announced ahead; surprises arrive with exact durations |
| Exploit-resistant | The scenario lives in a separate container; the network is allowlisted; unsafe commands are refused; the score anchors on greedy, so doing nothing or stopping early scores 0 |
| Benchmarking runs on config | A parameterized generator, run manifests, committed held-out datasets, a fresh-seed mode, and one HTML report |
| Easy to train on | Each decision fits in a short context; the scripted reference can label any state, so training data is unlimited |
| Cheap to build | An event-driven simulator on block sections, with no physics beyond travel times |

---

## 2. World model

### 2.1 Time
- 1 tick = 1 minute. The shift length is a generator parameter (presets: 120 / 360 / 720).
- Time advances only when the agent calls `advance_to_next_event()`. Commands use no
  time (the game is turn-based).

### 2.2 Network
- A linear single-track mainline of N stations between two terminal yards (unlimited
  capacity). N is a generator parameter (presets: 8 / 12 / 16).
- Between neighbouring stations there is one **block section** with a length (km) and a
  line speed. A section holds at most one train.
- Each station has 0–3 **loops** (passing tracks). Each loop has:
  - a length: 150 m / 300 m / 700 m; a train only fits a loop at least as long as itself;
  - a type: **platform** (passengers can board) or **goods** (no platform).
- A train may stop **on the main line** at a station. That blocks the station: no other
  train can pass through until it leaves.
- **Crew stations:** 2–3 stations where crews can be relieved (used by F10).
- Stretch only: one short **double-track segment** (F9) and a **branch junction** to a
  port (F14).

### 2.3 Train types

| Type | Length | Max speed | Restart penalty | Delay weight | Stopping pattern |
|---|---|---|---|---|---|
| Express | 200 m | 140 km/h | 1 tick | 3.0 × pax/100 | Major stations only |
| Local | 120 m | 100 km/h | 1 tick | 1.0 × pax/100 | Every station, min dwell 1–2 ticks |
| Freight | 600 m | 70 km/h | 4 ticks | 0.5 | Origin and destination only |
| Engineering | 100 m | 60 km/h | 1 tick | 0 | Must reach the work site before a possession starts |
| Hazmat freight *(stretch, F12)* | 400 m | 60 km/h | 4 ticks | 0.5 | Origin and destination only; isolation rule |

- Section travel time = `ceil(length_km / min(line_speed, train_speed, restriction) × 60)`.
- **Restart penalty (F5):** stopping where a train has no scheduled stop costs extra
  ticks when it starts again.
- **No early departures:** passenger trains may not leave a scheduled stop before their
  timetabled time.

### 2.4 Movement authority (the core mechanic)
Modelled on real track-warrant dispatching:

- The agent gives a train an **authority**: `(train, destination_station, loop | main)`.
- The interlocking **grants** it only if:
  1. every section on the path is free of other authorities;
  2. the destination loop is free and long enough (or `main` was requested);
  3. no possession on the path is active, or would start before the train clears it;
  4. the train type is allowed at the destination (platform rules; hazmat if built).
- Once granted, the sections are reserved, the train runs at deterministic section
  times, and it releases each section behind it.
- **A train without authority does not move.** There is no auto-pilot.
- Unsafe commands are **refused, with the reason** (e.g. `REFUSED R-SECTION-RESERVED:
  S4-S5 reserved by F12`). Collisions are impossible; the failures the agent can cause
  are **gridlock and delay**.

### 2.5 Gridlock
- Gridlock exists when unfinished trains wait on each other and no forward authority can
  be granted to any of them. `status()` flags it, listing the trains involved.
- **Recovery:** `reverse(train)` sends a train back to the previous station's loop if
  one is free: 15 ticks plus a 50-point inspection penalty.
- If no reversal is possible, the trains stay stuck and are charged as unfinished.

### 2.6 Rule IDs
Every rule the simulator enforces has an ID (`R-LOOP-LEN`, `R-SECTION-RESERVED`,
`R-NO-EARLY-DEP`, ...). Refusals cite the ID. A test fails if any rule ID or cost
component used by the simulator is missing from `manual()`.

---

## 3. Factors that make decisions hard

Every factor is a switch in the generator config. **MVP = the 11 factors built first.**

| # | Factor | Scope | Decision it forces | Likely failure of a weaker model | Ticks until it bites |
|---|---|---|---|---|---|
| F1 | Opposing trains on single track | **MVP** | Where two trains pass each other | Sends both towards each other with no free loop between them, causing gridlock | 10–40 |
| F2 | Loop capacity | **MVP** | Which station can take the waiting train | Fills a station, so the next pass is impossible | 10–40 |
| F3 | Loop length vs. train length | **MVP** | Freight needs a 700 m loop | Plans a pass at a station with short loops; freight stops on the main line and blocks it | 10–60 |
| F4 | Priorities and passenger weighting | **MVP** | Who waits | Holds the express for a local; ignores passenger load | Immediate to 30 |
| F5 | Restart penalty for heavy trains | **MVP** | Hold freight early or keep it rolling | Stops freight repeatedly; delays build up | Accumulates |
| F6 | Rolling-stock rotations | **MVP** | A late arrival delays the train's next service | Ignores the knock-on; later services start late | 60–240 |
| F7 | Possessions (planned maintenance block) | **MVP** | Clear the section before the window, or hold | Strands trains or misses the window | 30–60 |
| F8 | Temporary speed restrictions | **MVP** | Recompute pass points with slower sections | Uses stale travel times, so the planned pass fails | 10–40 |
| F10 | Crew duty limits | **MVP** | Relieve crews in time, only at crew stations | Crew times out mid-section and blocks it for 45 ticks | 60–180 |
| F15 | Breakdowns (unannounced, exact duration) | **MVP** | Re-plan around a stopped train | Keeps the old plan; trains pile up behind | Immediate |
| F16 | Late entries from outside the network | **MVP** | Re-plan pass points 20 ticks before arrival | Ignores the bulletin | 20 |
| F13 | Freight deadline (port ship cutoff) | Stretch 1 | Freight that misses the ship is heavily penalized | Over-prioritizes passengers until the freight misses the ship | 60–200 |
| F17 | Special train requests (optional work) | Stretch 2 | Accept for a bonus (penalty if late) or decline | Over-commits or always declines | 30–120 |
| F9 | Double-track segment | Stretch | Use passes on the move | Never uses it | — |
| F11 | Passenger connections | Stretch | Hold the express for a late local, or not | Always holds, or never holds | 5–30 |
| F12 | Hazmat isolation | Stretch | Hazmat can't wait at a station with a passenger train at the platform | Plans illegal passes | 10–30 |
| F14 | Branch junction to port | Stretch | Shared junction conflicts | Blocks the junction | 10–40 |
| F18 | Signal failure (manual working) | Stretch | The section takes 2× time + 3 ticks | Plans with normal times | 10–30 |

**Why this separates models:** a good decision keeps 5–10 constraints in mind at once
and projects 20–240 ticks ahead. Weaker models tend to handle one or two constraints and
plan greedily. Stronger models miss the long-delayed effects (F6, F10). The scripted
reference handles them by simulating forward.

---

## 4. Events and information

All events are fixed by the seed; what varies is when the agent learns about them. The
manual states each event type's lookahead.

| Event | Revealed | Content |
|---|---|---|
| Possession (F7) | 45–60 ticks ahead, via `bulletins()` | Section, start, end |
| Speed restriction (F8) | 30–60 ticks ahead | Section(s), speed, window |
| Late entry (F16) | 20 ticks ahead | Train, new entry tick |
| Breakdown (F15) | When it happens | Train, location, **exact** duration |
| Special request (F17, stretch) | When created; answer within 10 ticks | Path, deadline, value, penalty |
| Signal failure (F18, stretch) | When it happens | Section, duration |

---

## 5. Agent interface (MCP tools)

| Tool | Uses time? | Returns |
|---|---|---|
| `manual()` | No | All rules (with IDs), train types, cost formulas, event lookaheads, budgets |
| `network()` | No | Stations, loops (length/type), sections, crew stations |
| `timetable(train?)` | No | Scheduled stops and times, rotations |
| `status(compact=false)` | No | Full state; `compact=true` gives one line per train |
| `bulletins()` | No | Every event revealed so far, including future-dated ones |
| `kpis()` | No | Cost so far, by category |
| `stringline(from_tick, to_tick)` | No | ASCII time-distance diagram; **at most a 60-tick window, 2,000 characters** |
| `notes(text)` | No | **Replaces** the scratchpad (max 1,500 characters); its contents are shown in `status()` |
| `authorize(train, station, loop\|main)` | No | GRANTED, or REFUSED with a rule ID and reason |
| `cancel_authority(train)` | No | Only for sections not yet entered |
| `hold(train, until_tick)` | No | The train won't take a new authority before that tick |
| `relieve_crew(train)` | No | At a crew station: 8-tick relief |
| `reverse(train)` | No | Gridlock recovery (costly) |
| `advance_to_next_event()` | **Yes** | **Only what changed** since the last call |
| `end_shift()` | Yes, to the end | The rest of the shift runs with no new authorities |

### 5.1 Decision points
`advance_to_next_event()` stops at the first of:
1. a train reaches the end of its authority;
2. a train becomes ready to move with no authority (scheduled departure, entry, or
   dwell complete);
3. a bulletin is revealed;
4. an unplanned event (breakdown, late entry);
5. a crew is within 30 ticks of its duty limit;
6. gridlock;
7. **60 ticks have passed** with none of the above.

**Target:** about 60 decision points and under 100k tokens per 360-tick episode.

### 5.2 Budgets and episode end
- **Tool-call budget:** 400 per episode (provisional; set from the pilot). Once it's used
  up, the shift ends automatically.
- **Stopping without `end_shift()` counts as `end_shift()`**: the verifier runs the rest
  of the shift with no new authorities. `instruction.md` states this plainly.

### 5.3 Example `status(compact=true)`
```
t=142 | gridlock: none | notes: "E3 passes F12 at S6; relieve F12 crew at S4"
E3  express  at S5/L1-platform  auth:-            delay:+4  crew:96  next:S8@158
F12 freight  S6-S7 westbound    auth:S6/L2-goods  delay:+11 crew:23
L7  local    at S2/L1-platform  auth:S3/L1        delay:0   crew:140 next:S3@145
restrictions: possession S9-S10 170-200
```

---

## 6. Reward

> v0.3 (branch `net`) implements §6.6, which replaces §6.1 and §6.2: the reward is −C, minus the total cost.

### 6.1 Cost (weighted minutes)
Charged as events happen (visible in `kpis()`):

| Component | Formula |
|---|---|
| Passenger lateness | At each timed stop: `w_type × pax/100 × max(0, late − 2)`; ×2 at the terminus |
| Freight lateness | `0.5 × late` at the destination |
| Crew timeout | 300 + the 45-tick blockage (its delays are counted naturally) |
| Gridlock reversal | 50 + the time lost |
| **Unfinished trains** | At shift end: charged as if arriving at the shift end, + 200 each |
| Freight deadline miss *(F13)* | 400 per train |
| Special request *(F17)* | On time: −value; late or not run: +penalty; declined: 0 |

These weights are provisional. **They are frozen at the end of the pilot** and never
changed in response to held-out results.

### 6.2 Score

> **Superseded (v0.2 only).** Since v0.3 the reward is −C, minus the total cost (§6.6). The
> normalised formula below is not used as a reward anywhere.

The v0.2 per-task reward anchored on greedy, so the scale measured dispatching
quality rather than "did the trains move at all":

```
reward = clip( (C_greedy − C_agent) / (C_greedy − C_ref), 0, 1 )      → reward.txt
```

- Do-nothing, stop-early and anything worse than greedy score **0**. The reference
  scores **1**.
- **Aggregate across seeds as a ratio of sums:**
  `Σ(C_greedy − C_agent) / Σ(C_greedy − C_ref)`, which is stable when a seed's
  greedy–reference gap is small.
- **Comparison table:** also the **unclipped** score (so models worse than greedy keep
  their ordering), the **cost ratio** `C_agent / C_ref` and the cost breakdown.
- **Any seed where a model scores above 1.0 unclipped is flagged** for investigation:
  either the reference erred or the gap claim is wrong.
- **Training uses raw cost**, never the clipped reward.

### 6.3 Seed filter (generator)
A seed is accepted only if:
1. the reference costs **at least 30% less** than greedy;
2. the reference never gridlocks;
3. every exploit baseline costs **at least as much as greedy** (official score 0);
4. greedy finishes every train.

### 6.4 Baselines
1. **Do-nothing:** no tool calls / no authorities.
2. **Stop-early:** a sensible start, then `end_shift()` at tick 30.
3. **Greedy FIFO:** authorize each train to the next free loop ahead, oldest first (the
   score anchor).
4. **Random-legal:** random granted authorities.
5. **Reference** (§7), the Harbor oracle.

They ship as scripted policies in our loop. Do-nothing and stop-early must also behave
correctly through OpenCode, since the judges run them. If the day-1 test shows Harbor
accepts custom agents easily, they also ship as Harbor agents.

### 6.5 Exploit analysis (`exploits.py`, in CI)

| Attack | Why it doesn't pay |
|---|---|
| Do nothing / stop early | Costs at least as much as greedy, so reward 0 |
| Hold all freight, run passengers perfectly | Unfinished-train charges outweigh the passenger savings (checked by the seed filter) |
| Spam `advance` | Same as do-nothing |
| Hold trains on the main line | Blocks passes; drives gridlock and delay |
| Read the scenario or future events | They exist only in the `sim` container; the agent's container has no scenario and no simulator code (CI-checked) |
| Download the repo's datasets, or regenerate a seed from the public generator | Network allowlist: only the model API and `sim` are reachable |
| Call `sim` endpoints directly | Only the MCP tools and a read-only log of the agent's own actions are exposed |
| Tamper with the score | The verifier re-scores by replaying the action log against `tests/scenario.json`, which arrives only after the agent finishes |
| Malformed commands | Refused with a reason; no state change; counted as invalid calls |

### 6.6 v0.3 reward (agreed 2026-11-14, implemented on branch `net`)

This replaces §6.1 and §6.2 on the v0.3 network track (`docs/decisions.md` #2–#4,
`railway/net/cost.py`, `docs/net-cost.md`).

```
reward = −C                            → reward.txt
C      = C1 + C2 + C3 + C4 + C5        (1 unit = 100 passenger-minutes late)
```

- **The reward is minus the total cost. Nothing else is in it:** no greedy or reference
  normalisation, no clip, and no bonuses or shaping. Both benchmark evaluation and
  post-training use `−C`.
- **Greedy and reference are removed:** there is no `norm.json`, `C_greedy`, `C_ref`, or
  `C_best`. `details.json` and benchmark reports carry the cost per component (C1–C5),
  operational KPIs, and event/action counts directly alongside `reward = −C`.

**Lateness curve**, per passenger, used by C1 and C2:

```
f(L) = max(0, L − 2) + 5·[L ≥ 15] + 10·[L ≥ 30] + 20·[L ≥ 60]
```

The 2-minute grace stops the agent chasing small jitters. The slope of 1 means each
extra minute always costs more, so no train gets "written off". The steps mirror UK
Delay Repay refunds (25 / 50 / 100 % of the fare at 15 / 30 / 60 min late).

| Term | Charge | Why |
|---|---|---|
| **C1 Passenger lateness** | At each booked passenger stop: `v × alight/100 × f(late)`, where `v` = express 1.5 (`IC`), local 1.0 (`RE`, `S`) | Each passenger is charged once, at their own stop. This replaces `w_type × pax` and the terminus ×2. `v` reflects that long-distance passengers value time more. |
| **C2 Missed station call** | **`500 + v × (alight + board)/100 × (f(H) + 10)`** per scheduled passenger stop not served: skipped, cancelled, short-turned, bypassed by re-routing, or not reached by shift end | It hits both groups: people who wanted to get off, and people who wanted to get on and lose their journey. `H` = timetable headway to the next same-way passenger service (capped at 120, default 60). 10 = the hassle of being stranded. 500 makes skipping a last resort. |
| **C2 Missed connection** | `v × n/100 × (f(H) + 10)` for the `n` transferring passengers (using the connector's `v` and `H`) | The connecting train left before the feeder arrived. |
| **C2 Unfinished train** | `200` per train (including empty stock and engineering) that does not finish its booked work by shift end, plus missed calls for unserved passenger stops; undelivered freight adds **`1500`** plus freight lateness at shift end | Stock and crew are left out of position, and undelivered freight can never look cheaper than delivering late. |
| **C3 Freight** | `0.5 × late`, plus `50` past its delivery window (`late > 30` for `FI`, `late > 60` for `FB`); F13 port cutoff miss: `400` | Customers buy a delivery window, so freight wins conflicts only near its window. |
| **C4 Crew timeout** | **`40` per crew timeout** (down from 300 in v0.2) | Prevents free timeouts when a stranded train happens to block nobody; the 45-minute section blockage still charges knock-on delays naturally through C1–C3. |
| **C5 Gridlock** | **`1000` per gridlock**, charged when it forms | A deadlock means the plan failed. Reversal stays removed, so the trains stay stuck and their unserved stops add missed calls on top. The charge is on the deadlock, not on recovery, so staying stuck can never be cheaper. |
| **C5 Operations** | Unplanned signal stop: freight `4`, passenger `1` (`0` for empty stock/engineering). Platform change at a booked passenger stop: `v × (alight + board)/100 × 2` | Energy and wear from stop-start running, which acts as a tie-breaker. Re-platforming is a cheap conflict tool, but not a free one. |

**What the weights imply** (checked by script):
- Skipping a 300-passenger express stop costs 818. Arriving late there only costs more
  after 149 minutes. At small stops, skipping is never cheaper than arriving late.
- A blocked express that diverts and skips a stop (601) beats waiting only when the
  blockage lasts more than 84 minutes.
- A gridlock (1000) equals 222 minutes of delay to a full express, or two missed calls.
- Per extra minute of waiting: full express 4.5, 150-passenger local 1.5, freight 0.5.

**Scenario data needed:** passengers getting on and off at each stop; connections
(feeder, connector, hub, passengers, minimum transfer time); the headway `H` per
station and destination; freight window and miss cost; the published platform at hubs.

**Unchanged:** freight 0.5 per minute, the F13 cutoff of 400, and weights frozen after
the pilot. Re-run the §6.5 exploit checks once the full network engine and policies
land.

---

## 7. Reference dispatcher (scripted, deterministic)

Runs as the Harbor **oracle** (`solution/solve.sh`), playing **through the same MCP tools
as the agent**. It only sees what the agent could see at that moment. Its rollout
simulator lives in `solution/` only.

1. **Conflict detection:** project trains forward; list opposing pairs and overtakes in
   the next 90 ticks.
2. **Candidates:** up to **5** feasible pass plans per conflict (loop free and long
   enough, rules met, possession-aware).
3. **Rollout:** simulate each candidate for **90 ticks** with a safe greedy base policy,
   using the **event-driven** simulator. Pick the cheapest; break ties by train id.
4. **Deadlock avoidance:** grant an authority only if the train has a reachable free loop
   beyond its destination, or reaches its terminus.
5. **Crew:** relieve at the last crew station before a predicted timeout.

**Targets:** under 2 minutes per 360-tick episode.

**Strength check:** on 5 T1 seeds, compare against an offline exhaustive search that
knows the whole shift in advance, to measure how far the reference is from the best
possible result.

The large model is the expert *model* in the model comparison. The **scripted
reference** is the official reference and the training teacher, because it can label
any state.

---

## 8. Harbor packaging

### 8.1 Task layout (one task per seed)
```
<task>/
  instruction.md              role, goal, "call manual() first", scoring summary, episode-end rule
  task.toml                   timeouts, network allowlist, agent user, MCP server
  environment/
    Dockerfile                agent image ("main"): OpenCode pre-installed, NO scenario, NO simulator
    docker-compose.yaml       adds the "sim" service
    sim/                      sim image: simulator + MCP server + this task's scenario
  tests/
    test.sh                   verifier
    scenario.json             frozen scenario, used for the replay
    norm.json                 C_greedy, C_ref (computed at export)
  solution/
    solve.sh                  reference dispatcher (oracle) + its rollout simulator
```

### 8.2 `task.toml` essentials
- `[[environment.mcp_servers]]`: `name = "railway"`, `transport = "streamable-http"`,
  `url = "http://sim:8000/mcp"`.
- `[environment] network_mode = "allowlist"`, with only the model provider host(s) and
  `sim` allowed.
- `[agent] user = "agent"` (non-root), if the day-1 test confirms OpenCode works that way.
- Agent timeout: set from the pilot.

### 8.3 Verifier (`tests/test.sh`)
1. Fetch the action log from `http://sim:8000/episode_log`.
2. **Replay it** in a fresh simulator with `tests/scenario.json`. Any part of the shift
   the agent didn't play runs to the end with no new authorities.
3. Check the replayed cost equals the live server's cost. A mismatch is a bug: fail
   loudly.
4. Write the clipped reward to `/logs/verifier/reward.txt`. Write `details.json` (cost
   by category, failure counts, decisions, invalid calls, tokens if available) and the
   action log to `/logs/verifier/`. **Never write `reward.json`**, because Harbor would
   use it instead of `reward.txt`.
5. **Infrastructure failure** (`sim` unreachable, corrupt log): exit with an error and
   write no reward. Harbor then marks the trial as an error to rerun, not a 0.

### 8.4 Facts confirmed vs. still to test
- **Confirmed (Harbor source and docs):** sidecar services in compose; MCP servers in
  `task.toml` and how OpenCode receives them; `tests/` uploaded only after the agent
  finishes; `solution/` only for the oracle; `reward.txt` parsed as a single float with
  no range check; `reward.json` takes priority; the agent runs as root by default.
- **To test on day 1 (§11):** whether the verifier can reach `sim`; whether the
  allowlist breaks OpenCode's installation; whether OpenCode works as non-root; whether
  OpenCode can drive a small model behind an OpenAI-compatible endpoint; whether custom
  agents can be loaded.
- **Backend constraint:** sidecars work on docker, podman, ec2, and daytona/modal through
  Docker-in-Docker, but not on e2b or runloop. If the judges use one of those, fall back
  to the simulator inside the agent's container as a root-only process with the scenario
  held in memory (weaker).

---

## 9. Benchmarking system

### 9.1 Scenario generator (parameterized)
Config knobs: stations, trains, train-type mix, enabled factors, event density, shift
length, loop distribution. Tiers are **named presets**:

| Preset | Stations | Trains | Shift | Factors | Use |
|---|---|---|---|---|---|
| T1 | 8 | 8 | 120 | F1–F4 | Pilot, quick checks |
| T2 | 12 | 14 | 360 | All 11 MVP | **Judges' standard runs**, headline numbers |
| T3 | 16 | 22 | 720 | MVP + any stretch built | Showcase only |

### 9.2 Seed sets (never overlapping)
- **Dev:** tuning the generator, weights and pilot.
- **Train:** small-model training only.
- **Held-out:** reported numbers and judges' runs. **Committed** to the repo.
- **Structural holdout:** a 14-station layout never seen in training, to show
  generalization.

### 9.3 Datasets and runs
- `gen export --config T2 --seeds heldout --out datasets/t2-heldout/` writes one Harbor
  task per seed, including `tests/norm.json`. `--fresh` draws new seeds into an
  uncommitted dataset.
- A **run manifest** (`runs/<name>.yaml`) holds the model, harness, dataset, number of
  repetitions and temperature (default: the model's own).
- Defaults: **20 seeds × 3 repetitions**; 95% confidence intervals by **resampling
  seeds**.
- Results: one JSONL row per episode, holding the scenario hash, model, mode, raw cost
  by category, official and unclipped scores, invalid-call rate, decisions and tokens.

### 9.4 Two ways to play
1. **Harbor + OpenCode (headline):** `harbor run -p datasets/t2-heldout -a opencode -m
   <provider/model>`. All headline numbers, including the model-size comparison, come
   from here, because they match what the judges see.
2. **Our own loop (diagnostics and training):** same MCP server, same tools, identical
   prompt for every model, strict tool-call format.
   - **Windowed mode (main):** system prompt + manual + compact status + bulletins +
     last 5 turns + notes.
   - **Full-history mode (secondary):** shows how much the window helps.

### 9.5 Models
Three models give a curve: **small** (7–8B open-weight, the training target), **mid**
(about 30B, ideally the same family as small), and **big** (the frontier model from the
credits). Exact models: see §12.

### 9.6 Report
`make report RUN=<name>` writes one self-contained HTML file containing:
- a table: score (mean ± CI), cost vs. reference, gridlocks and crew timeouts per model;
- 3 charts: score by model, score by factor (ablation), before vs. after training.

### 9.7 Diagnostics
1. **Score vs. episode length** (T1 / T2 / T3).
2. **Factor ablation:** turn on one factor at a time and plot the model gap.
3. **Failure breakdown:** gridlocks, crew timeouts, deadline misses, invalid calls.
4. **Same-state regret:** both models answer from identical states, scored by reference
   rollouts. This is pure decision quality, with no context differences.
5. **Projection vs. choice:** an optional `conflicts()` tool listing upcoming conflicts.
   Diagnostic only.
6. **What-if ablation:** an optional `what_if(actions)` tool. Diagnostic only, **never**
   in official runs.
7. **`stringline` ablation:** one run without it.
8. **Human baseline (optional, after the main evaluation):** 2–3 people play 3 T1 and 2
   T2 seeds through a CLI; their logs replay exactly.

### 9.8 Pilot gate (about hour 12, 10 T1 dev seeds)
Continue only if:
- (a) the big model's unclipped score is **at least 0.3 above** the small model's;
- (b) the big model scores **at most 0.8**;
- (c) the reference beats greedy by **at least 30%**.

If (b) fails, enable more factors. If (a) fails, check the invalid-call rate first.

---

## 10. Post-training plan (tiebreaker)

1. **Behaviour cloning:** reference actions on train-seed states, in the windowed
   format (short contexts).
2. **DAgger:** run the small model, label every state it visits with the reference
   action, retrain.
3. **GRPO:** sample several actions per decision. Reward = negative reference-rollout
   cost after the action (raw cost; the group baseline handles scale).
4. Report held-out results before and after, the structural-holdout result, and the
   failure breakdown before and after.

Setup (LoRA, GPU, tokenizer) depends on the model choice (§12).

---

## 11. Build plan (team of 3, 48 h)

**Day-1 interface:** A and C agree `load(scenario)`, `apply(action)`, `advance()`,
`state()`, `cost()` in hour 0, so B and C can build against a stub.

| Hours | A: simulator | B: Harbor and tooling | C: policies |
|---|---|---|---|
| 0–3 | Event-driven core: sections, loops, authorities | **Day-1 test:** toy MCP container in Harbor + OpenCode; check the 5 unconfirmed facts | Simulator stub, greedy, do-nothing |
| 3–9 | F1–F4, gridlock, cost, rule IDs, property tests | MCP server, decision points, compact/diff outputs | Generator (T1), random-legal, stop-early |
| 9–12 | `manual()` + rule-coverage test | Task export, verifier with replay | Reference (rollout); **pilot gate** |
| 12–20 | F5–F8, F16 | Manifest runner, our own loop (windowed/full) | Seed filter, `exploits.py`, oracle = 1.0 CI check |
| 20–28 | F10, F15; T2 preset | Network allowlist and non-root setup, image-contents test | Reference strength check; **freeze weights** |
| 28–34 | Stretch F13, then F17 | Full held-out evaluation, report | Training data (BC) |
| 34–44 | Bug fixes, determinism checks | Ablations, diagnostics | DAgger → GRPO |
| 44–48 | Demo | README with one-command evaluation, clean-machine check | Before/after results |

If behind schedule, cut stretch factors first, then diagnostics. Never cut the MVP
factors, the exploit tests, the oracle check or the replay verifier.

---

## 12. Open items

1. **Models:** the exact small, mid and big models and providers, and the GPU plan for
   training. This decides the training setup and the windowed-loop context size.
2. **Judges' Harbor backend:** ask the organizers. If it's e2b or runloop, use the §8.4
   fallback.
3. **Provisional numbers set from the pilot:** tool-call budget (400), agent timeout,
   cost weights (then frozen).

---

## 13. Decision log (design review, v0.1 → v0.2)

| Area | Decision |
|---|---|
| Score | Anchor on greedy; clip to [0, 1] in `reward.txt`; ratio-of-sums aggregation; comparison table also shows unclipped score and cost ratio; flag seeds above 1.0; training on raw cost |
| Seed filter | Reference ≥30% cheaper than greedy; no reference gridlock; exploits cost ≥ greedy; greedy finishes every train |
| "Dynamic" | Parameterized generator + run manifests + committed held-out datasets + `--fresh` seeds; tiers become presets |
| Episode size | ~60 decision points (7 stop triggers incl. a 60-tick cap); changes-only output; compact status; <100k tokens |
| Scope | 11-factor MVP (F1–F8, F10, F15, F16); stretch order F13, F17, then the rest |
| Reference | Event-driven sim; 90-tick rollouts; ≤5 candidates; <2 min/episode; exhaustive-search strength check on 5 T1 seeds |
| Fairness | No what-if tool in official runs; rule-coverage test; `what_if` and `conflicts` only as diagnostics |
| Memory | Notes scratchpad, replace-only, 1,500 characters, in every mode; windowed main mode in our loop |
| `stringline` | Kept, 60-tick / 2,000-character cap; one ablation without it |
| Harness | Headline numbers through Harbor + OpenCode; our own loop for training and same-state diagnostics |
| Run defaults | 20 seeds × 3 reps; model's default temperature; seed-resampled 95% CI; JSONL results |
| Pilot gate | big − small ≥ 0.3; big ≤ 0.8; reference beats greedy by ≥30% |
| Judging tier | T2 for judges; T3 showcase |
| Seeds | Dev / train / held-out sets that never overlap + 14-station holdout; weights frozen after the pilot |
| Stack | Python 3.11+, uv, MCP Python SDK, pydantic, pytest (property tests), matplotlib |
| Report | One HTML file: table + 3 charts |
| Datasets | One Harbor task per seed; held-out committed; `--fresh` uncommitted |
| Architecture | Simulator + MCP in a separate `sim` container (streamable-http); nothing secret in the agent's container |
| Isolation | Network allowlist (model API + `sim`); non-root agent user |
| Instruction | Short `instruction.md`; full rules only in `manual()`; stopping = `end_shift()` |
| Oracle | Reference plays through MCP; CI requires oracle = 1.0 on every task |
| Constants | `C_greedy` and `C_ref` in `tests/norm.json`, computed at export |
| Verifier | Fetch log → replay against `tests/scenario.json` → cross-check → `reward.txt` + `details.json`; never `reward.json` |
| Infra failures | Error out with no reward (trial error), never a 0 |
| Baselines | Scripted in our loop; checked through OpenCode; Harbor custom agents if easy |
| Day-1 test | Hours 0–3: verify sidecar reachability, allowlist + install, non-root, small-model endpoint, custom agents |
| Team | A simulator, B Harbor and tooling, C policies and training |
| Human baseline | Optional, after the main evaluation |
