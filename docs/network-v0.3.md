# High-Traffic Network: World Model (v0.3 proposal)

This specifies the network, trains, traffic and agent decisions for v0.3. It turns the
agreed direction in `docs/design-changes-v0.3.md` §4 into mechanics that can be built.
The reward is `DESIGN.md` §6.6. Steps 1 to 3 are implemented (see §8).

**The idea in one line:** model a real control centre. The timetable is feasible when
everything runs on time. Disruptions and small delays break it. The agent is the
signaller and controller who decides, at every conflict, which train goes first, where it
goes and what gets sacrificed.

---

## 1. Design principles

1. **Real mechanics, coarse physics.** Block signalling, junction conflicts, platforms,
   turnarounds and rotations are modelled. Continuous physics is not: each train type has
   a run time per segment, plus a penalty for each stop and start.
2. **The difficulty is congestion, not the map.** The generator fills the bottlenecks to
   75–90% of capacity, as real timetables do. A bigger map without more traffic would be
   easier.
3. **Every decision is a real trade-off.** Mixed speeds, shared junctions, scarce
   platforms and knock-on effects mean no option is free. The reward (§6.6) prices
   every option in the same unit.
4. **Routine is automatic, conflicts are the agent's.** As in real control centres, an
   automatic route-setting system (ARS) runs trains that conflict with nobody. The agent
   decides every conflict. This keeps 100+ trains tractable while each decision point
   carries several real decisions (§5).
5. **Deterministic and fair**, as in v0.2. A seed fixes everything. Announced
   disruptions have lookaheads. Every rule is in `manual()`.

---

## 2. Infrastructure

The network is a graph. Nodes are stations, junctions, terminals, yards and freight
terminals. Edges are track segments.

### 2.1 Track segments

| Kind | What it is | What it forces |
|---|---|---|
| **Double track** | One track per direction, split into signal blocks of 1.5–3 km. Following trains run automatically, one block apart (a 2–4 min headway). A train entering from a portal occupies one block like any other, so a portal is not a capacity limit in itself. | A slow train ahead caps everyone behind it until the next overtaking point. |
| **Bidirectional double track** | Either track usable in either direction (on some segments only) | Overtaking on the move, and single-line working when one track is blocked. |
| **Single track** | One train between passing points (as in v0.2) | Crossing decisions, as today. |
| **Loops / overtaking sidings** | Off the main line, with a length limit | Where a slow train steps aside, if it fits. |

Each segment has a line speed, an **electrified** flag and a length.

### 2.2 Junctions, where lines converge and diverge

| Type | Behaviour | Real-world example |
|---|---|---|
| **Flat junction** | Some routes cross others. A train turning across the opposite line blocks it for its clearing time (2–3 min). | The classic bottleneck on busy networks. |
| **Single-lead junction** | Both directions of the branch share one short piece of track | Cheap to build, notorious for delays. |
| **Grade-separated** (flyover) | No conflicting moves | The fix real railways pay for. Only some junctions get it. |

The **diverging route** has a turnout speed limit (40 or 80 km/h), so taking the
branch costs time.

### 2.3 Stations, terminals and yards

| Node | Model | What it forces |
|---|---|---|
| **Through station** | Numbered platforms, each with a length and the lines that can reach it. A **station throat** of conflicting routes on the approach. | Platform allocation and the order through the throat. |
| **Hub** | A big through station where 3–4 lines converge, with connections between services | Holding for connections, platform juggling. |
| **Terminal** | A finite number of bay platforms. A booked dwell of 8 minutes or more is a turnaround: only then may a train leave a platform the way it came in, because that is how long changing ends takes. | A full terminal makes arrivals queue outside. Turnaround is the main way delay spreads. |
| **Yard / depot** | Finite sidings for empty trains | Empty trains must reach it so the next service can start. |
| **Freight terminal / port** | Finite reception sidings and slot times (F13) | Freight windows. |

---

## 3. Train types and priorities

| Type | Priority | Length | Max speed | Restart penalty | Traction | Stops | Typical load | Reward class (§6.6) |
|---|---|---|---|---|---|---|---|---|
| **Intercity (IC)** | 1 | 260 m | 200 km/h | 1 min | Electric | Hubs and major stations | 400–600 | express (1.5) |
| **Regional express (RE)** | 2 | 160 m | 160 km/h | 1 min | Electric or diesel | Selected | 200–300 | local (1.0) |
| **Commuter (S)** | 2, or 1 in the peak | 120–240 m | 120 km/h | 0.5 min (fast acceleration) | Electric | Every station, every 10–15 min | 300–800 in the peak | local (1.0) |
| **Intermodal freight (FI)** | 3 | 600–740 m | 100 km/h | 4 min | Electric or diesel | Origin and terminal | none | freight, window 30, port 400 |
| **Bulk freight (FB)** | 4 | 740 m | 80 km/h | 5 min | Diesel | Origin and terminal | none | freight, window 60 |
| **Empty stock (ECS)** | 5 | 120–260 m | as its unit | 1 min | as its unit | none | none | none (its lateness reaches the next service through the rotation) |
| **Engineering** | Special | 100 m | 60 km/h | 1 min | Diesel | Work site | none | as v0.2 (must reach its possession) |

- **Priority is published, not enforced.** ARS uses it by default, and the agent may
  override it. The reward decides who should really go first. A peak commuter train
  with 600 people (6.0 per minute) costs almost as much per minute as an intercity with
  450 people (6.75 per minute). Real controllers know this, and a fixed priority list
  doesn't.
- **Type constraints narrow the options.** Electric trains cannot be diverted over
  non-electrified lines. Long freight fits only long loops. A train only fits a platform
  at least as long as itself.
- **Mixed speeds cost capacity.** On a 20 km stretch without overtaking points, a
  200 km/h IC behind an 80 km/h freight train needs a gap of 15 − 6 + 3 = 12 min, not
  the normal 3 min headway. One freight path uses up about four IC paths. Deciding
  where slow trains step aside is the core of mainline dispatching.

---

## 4. Traffic and disruptions

### 4.1 Timetable (generator)

- **Clockface service groups** (for example, IC every 30 min A↔B, commuter every 15 min
  C↔D), a **peak** in the middle third of the shift with longer dwells and fuller trains,
  and **freight paths** in the gaps. The peak does not yet add extra services; the dwell
  bump is capped so it can never turn a stop into a turnaround.
- **Utilisation target:** the generator adds filler paths until the busiest bottleneck
  reaches 75–90% of capacity. Occupancy is measured as real capacity analysis measures
  it: a junction move is unavailable for its own clearing time and for every conflicting
  move's, a one-way track for its critical block and a bidirectional one for the whole
  transit, each plus a two-minute signalling margin (the junction margin in Network Rail's
  Timetable Planning Rules is 2–3 minutes). T1 tops out around 63–69% rather than 75%,
  because its single-track branch runs out of paths first; the band is the target the
  generator fills toward, not a guarantee. A filler path is only taken if the run still
  gets every other train home, so freight never displaces a passenger service.
- **Rotations:** each passenger train forms another service after turning around at a
  terminal, so late arrivals carry over. A rotation is not a separate object: it is one
  trip whose middle call has a turnaround dwell.
- **Connections** at hubs: feeder, connector, transfer passengers, minimum transfer
  time.
- **Passenger counts per stop** (getting on and off), with peak-period profiles.
- **Feasibility check:** booked times come from the planner's own run of the shift, in
  which every conflict is settled in timetable order and nothing goes wrong. The published
  timetable is therefore one the network really delivers, and an undisrupted shift costs
  nothing. Publishing new departures can reorder the next run, since timetable order is
  what settles conflicts, so the generator rebooks until re-running the published book
  changes nothing and rejects the seed if it never settles. A service the run cannot get
  home inside the shift is dropped, latest path first, exactly as a real timetable gives
  up its freight paths first.

### 4.2 What breaks the plan

| Source | Revealed | Effect |
|---|---|---|
| **Entry delays** (every train entering from outside, 0–10 min, set by the seed) | 20 min ahead | Constant small conflicts. This is the main source of decisions. |
| **Dwell overruns** at busy stations in the peak (+1–4 min) | When they happen | Knock-on through throats and junctions |
| Breakdown (F15) | When it happens, with exact duration | Block or junction route closed |
| **Points failure** at a junction | When it happens, with duration | One route through the junction unavailable, so divert or wait |
| Signal failure (F18) | When it happens | Section runs at 2× time |
| **Platform out of use** | 30 min ahead | Re-platforming at a hub |
| Speed restriction (F8), possession (F7) | 30–60 min ahead | As in v0.2 |
| Crew limits (F10) | Warning 30 min ahead | As in v0.2. There is no reward term (§6.6), only knock-on delay. |

---

## 5. How the agent controls the network

### 5.1 Automatic route setting

ARS sets each train's timetabled route as soon as it is free and nobody else wants it.
**Whenever two or more trains want the same resource** (a junction route, a
single-track section, a platform, a terminal track) **within the next 15 minutes**, ARS
does not choose. It raises a **conflict** for the agent.

This replaces v0.2's "a train without authority does not move" (§2.4). With 100+ trains,
manual authorities would mean hundreds of trivial calls per hour. The agent can switch
ARS off for an area (`auto(area, off)`) and drive it fully by hand.

### 5.2 Decision points

`advance()` stops at the first of:

1. a new **conflict** (bundled: every open conflict is listed together);
2. a disruption is revealed or happens;
3. a **connection is at risk** (feeder delay greater than the transfer slack);
4. a train cannot be routed (its platform or terminal track is full);
5. a crew warning;
6. a gridlock (charged 1000, see §6.6);
7. 30 minutes with none of the above.

### 5.3 Actions

| Action | Real-world equivalent | Typical cost in the reward |
|---|---|---|
| `order(resource, [trains])` | Regulation: who goes first through a junction, section or throat | Lateness of whoever waits |
| `hold(train, where, until)` | Hold at a signal or platform (connections, regulation) | Its lateness, an unplanned stop |
| `overtake(slow, loop, fast)` | Put the slow train in a loop until the fast one passes | Slow train's lateness and restart |
| `platform(train, station, n)` | Re-platform | Platform-change charge |
| `route(train, via=[…])` | Diversion over another line | Extra run time, missed calls if it skips stops |
| `skip(train, station)` | Run past a scheduled stop to recover time | **Missed call: 500+** |
| `short_turn(train, station)` | Terminate early and start the return trip there | Missed calls downstream, which restores the rotation |
| `cancel(train)` | Cancel a service | Missed calls for every stop, plus 200 |
| `single_line(segment, on/off)` | Bidirectional working over one track during a blockage | Both directions share one track |
| `relieve_crew(train)` | Crew relief | 8 min |
| `auto(area, on/off)` | Take over an area manually | none |

Read tools stay as in v0.2, scoped by area so context stays small: `network(area)`,
`status(area, compact)`, `conflicts()`, `timetable`, `bulletins`, `kpis`, and
`stringline(corridor)`, one corridor at a time.

### 5.4 One decision point, as the agent sees it

```
t=08:14  3 conflicts | gridlock: none
C1 Ashford Jn (flat): 1A23 IC 450pax +3 | 2K40 S 600pax 0 | 6M50 FI window in 18 min
   1A23 and 2K40 cross; each move blocks the junction 2 min; 6M50 waits for either.
C2 Hub P4: 2K40 and 1C07 RE both booked P4 at 08:21; P6 free (length ok, +1 throat move)
C3 Connection: 1C07 feeds 1A23 at Hub, 35 transfers, slack 2 min, 1C07 now +4
```

Three linked decisions: junction order, a platform swap, and whether 1A23 waits for
its connection. Per minute of delay, 1A23 costs 6.75 and 2K40 costs 6.0, so a peak
commuter train nearly ties the intercity. If 1A23 goes first, holding it for the
connection hurts more. If 6M50 waits more than 18 minutes it misses its window (+50).
A weak model handles one conflict at a time. A strong one sees that C1, C2 and C3 are
one problem.

---

## 6. Sizes (generator presets)

| Tier | Layout template | Trains / shift | Junctions | Shift | Disruptions | Expected decision points |
|---|---|---|---|---|---|---|
| **T1** | Trunk and branches: double-track trunk, one flat and one single-lead junction, a single-track branch, a port spur | 25–35 | 2 + 1 terminal | 3 h | 3–4 | 40–60 |
| **T2** | Hub: 8–10 platform hub, 4 lines converging on its throat, 2 terminals, depot | 60–80 | 5 + hub throat | 4 h | 5–7 | 80–120 |
| **T3** | Two hubs joined by a trunk with a bypass, branches, port | 130–160 | 9+ | 6 h | 8–12 | 150–250 |

- **Layouts** come from these templates with random perturbation: segment lengths
  ±20%, which junctions are flat or grade-separated, number and length of loops,
  platform counts, gaps in electrification. This follows the team record (§4.4).
- **Difficulty knobs:** utilisation target, speed mix, entry-delay spread, disruption
  count and how many junctions are flat.
- **Context budget:** T3 will not fit a full status in context. Area-scoped tools and
  "changes only" output from `advance()` are required, not optional.

---

## 7. Deliberately not modelled

- Continuous train physics, signal aspects and braking curves. Blocks and run times
  per type are enough to make conflicts real.
- Trains that start or finish the shift in a depot. Every service enters and leaves the
  area through a portal, or terminates in a siding that holds it for the rest of the shift.
- Passenger flow simulation, crowding and dynamic demand. Passenger counts per stop
  are fixed by the seed.
- Safety failures. The interlocking makes collisions impossible, as in v0.2.
- Reversal. It stays removed, and gridlock stays permanent (team record §3.2).

---

## 8. Build order and impact

The single-line assumption runs through the engine, generator, reference, gridlock
check, solver, policies and stringline (team record §4.6). Suggested order:

1. Graph infrastructure: segments with blocks, junction conflict tables, platforms,
   terminals. Interlocking as route locking.
2. ARS and conflict detection; decision points; `order`, `hold`, `platform`.
3. Train types and the timetable generator with the utilisation target, rotations,
   connections, per-stop passenger counts and the feasibility check.
4. Disruptions, then `route`, `skip`, `short_turn`, `cancel`, `single_line`.
5. Area-scoped tools, compact status, stringline per corridor.
6. Greedy (ARS in timetable order, the reward's 0 anchor) and the reference on the
   graph; the solver with "best known" as fallback.
7. The §6.6 reward, then re-run the exploit checks.

Each step ends with one runnable tier: T1 after step 4, T2 and T3 after step 6.

**Status:** steps 1 to 3 are built in `railway/net`:
- `model.py`: graph, blocks, platform sides, conflicts derived from shared points plus listed crossings, and the
  `Trip`, `Connection` and `Timetable` records.
- `interlocking.py`: route locking. A train must fit every platform road it runs over, not only the one it stops at.
- `routing.py`: shortest route, and the track a train runs out of the area on.
- `engine.py`: block signalling, ARS, decision points, `order`, `hold`, `platform`.
- `trains.py`: the §3 type table.
- `timetable.py`: run times, utilisation, the generator and the feasibility check.
- `templates.py`: the fixed T1 layout and its service groups.

It has its own tests in `tests/net`. The v0.2 simulator in `railway/sim` is untouched.

Directed reachability, deferred from step 1, is covered in effect: the generator routes every leg before it books
it and reports any train it cannot get home, so a layout with a track nobody can leave is rejected when it is used.

Known cost, to be dealt with before T2 and T3: generating one T1 shift takes 7 s at 180 minutes and about a minute
at 360, because every filler candidate is tried with a full run of the shift. T3's 130–160 trains will need the
per-trip occupancy cached and summed rather than the whole timetable re-measured for each candidate.
