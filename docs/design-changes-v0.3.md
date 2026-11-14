# v0.3 design changes: reconciled version (2026-11-14)

This is the single version to build. It merges the first record in this file (Anh) with
the scoring section Moiz added to `DESIGN.md` §6.6 (commit `c7cce53`), and settles the
places where the two disagreed.

**Short answer for implementers: build `DESIGN.md` §6.6 with one amendment: the
crew-timeout charge stays (§3.5).**

Nothing here is implemented yet. `DESIGN.md` §6.1 and §6.2 still describe the code.

Status labels: **Settled** (build it), **Starting value** (build it as a config default,
calibrate by measurement), **Open** (needs a decision).

---

## 1. The three conflicts and how they are settled

| # | Situation | Anh's earlier version | Moiz's §6.6 | Settled |
|---|---|---|---|---|
| 1 | Train skips a station where it should stop | Passenger-weighted block of 30 minutes plus its lateness | Flat 500 plus the passengers who could not get on or off | **Moiz's, as written** (§3.3). |
| 2 | Train sent down the wrong branch | One unfinished train: 200 plus lateness | 200 plus a missed-call charge for every stop it never served | **Moiz's, as written.** |
| 3 | Crew runs out of hours | Keep the 300 charge | No charge; only the delay it causes | **Anh's.** The 300 stays, as a starting value (§3.5). |

Why Moiz's formulas win on 1 and 2: they use passengers getting on and off at each stop,
which is more precise than one passenger number per train, and they charge both groups
affected by a skipped stop. The numbers Anh's side gave earlier for missed calls and
misroutes (the "30 + lateness" block) are **superseded** and should not be built.

---

## 2. Reward (Settled)

```
reward = max( 0, (C_greedy − C_agent) / (C_greedy − C_best) )        → reward.txt
```

- **`C_best`** is the proven optimum where the exact solver can prove one, otherwise the
  best known cost, labelled as such in `norm.json`.
- **0** means greedy or worse, including do-nothing and stop-early. **1** means the best
  known plan.
- **Above 1** means the agent beat the best known plan. It is flagged and that plan
  becomes the new `C_best`. Because this changes earlier scores, `norm.json` carries a
  version and past results are re-scored when it changes.
- **The reference dispatcher** becomes a baseline row in the table and stays the
  training teacher.
- **Reported beside the score:** on-time percentage, weighted passenger delay minutes,
  freight delivered, gridlocks, crew timeouts, and the clean-shift rate (share of
  episodes with zero red-flag events).

### 2.1 Why the anchor changed

Today's score uses the scripted reference as the top anchor. A full-knowledge search
beats the reference by a wide margin, so "1.0" meant "matched our script":

| Seed | Do-nothing | Greedy | Reference | Offline search | Old score of the offline plan (unclipped) |
|---|---|---|---|---|---|
| 0 | 7571.5 | 1746.7 | 296.7 | 176.6 | 1.08 |
| 5 | 8716.3 | 1408.2 | 208.5 | 208.5 | 1.00 |
| 6 | 10058.9 | 4809.3 | 2620.7 | 1155.2 | 1.67 |
| 7 | 9850.1 | 2106.2 | 1219.3 | 213.9 | 2.13 |
| 10 | 8659.8 | 1026.3 | 447.0 | 22.5 | 1.73 |

The exact solver's proven lower bound on seed 5 is 4.2, far below the 208.5 that both
the reference and the offline search reach.

### 2.2 Why greedy stays as the bottom anchor

An absolute scale, `1 − C_agent / C_do-nothing`, was measured on 10 T1 dev seeds. Greedy
scored 0.52–0.88 and the reference 0.74–0.99. Everything competent lands near the top
and the gap between models disappears. Greedy as the zero point is what gives the scale
its resolution.

### 2.3 Seed filter

- Settled: drop "the reference costs at least 30% less than greedy".
- Starting value: `C_best` at least 30% below greedy, so the denominator stays stable.
- Unchanged: every exploit baseline costs at least as much as greedy; greedy finishes
  every train.

### 2.4 Harbor oracle (Settled)

The oracle replays the stored best plan for the task, so it scores exactly 1.0.

---

## 3. Cost function

```
C = C1 + C2 + C3 + C4 + C5            (1 unit = 100 passenger-minutes late)
```

Lateness curve, per passenger, used by C1 and C2:

```
f(L) = max(0, L − 2) + 5·[L ≥ 15] + 10·[L ≥ 30] + 20·[L ≥ 60]
```

`v` is 1.5 for an express and 1.0 for a local. `H` is the scheduled wait for the next
service at that station, fixed in the scenario from the timetable.

### 3.1 Removed (Settled)

- **Terminus ×2.** Replaced by per-stop alighting counts in C1: each passenger is
  charged once, at their own stop.
- **Reversal.** The tool, the cost component and its rules are removed. Gridlock is
  permanent.

### 3.2 C1 Passenger lateness (Settled)

At each timed stop: `v × alight/100 × f(late)`.

### 3.3 C2 Missed calls and unfinished trains (Settled)

For each scheduled stop not served, whatever the reason (bypassed, misrouted, wrong
terminal, not reached by shift end, or stuck in a gridlock):

```
500 + v × (alight + board)/100 × (f(H) + 10)
```

- **Unfinished train:** 200 per train that does not finish, plus the charge above for
  each unserved stop.
- **Freight:** an unfinished freight train pays 200 and is charged as arriving at shift
  end (C3).
- **Missed connection:** `v × n/100 × (f(H) + 10)` for the `n` transferring passengers.
- **Grant text:** a grant that loses a stop or the destination still succeeds, but says
  so, for example `GRANTED (off route: E3 can no longer reach T2)`.
- **Counting:** a misroute and the wrong terminal it leads to are one red-flag event.

**The principle:** a station that gets no service is unacceptable, however few people
are waiting there. That is why the 500 is flat and applies to every unserved stop.

**An amendment that was considered and dropped.** Anh's side proposed charging the 500
only when a grant makes the stop unreachable, and not for stops a train simply never
reached. The reason was this measurement, on the 60 passenger trains of 10 T1 dev seeds
(assuming passengers get on and off evenly across stops, and a 60-minute headway), of
what a train that never runs costs:

| Type | Trains | Average stops | Average passengers | Today | §6.6 |
|---|---|---|---|---|---|
| Express | 20 | 5.4 | 280 | 2,190 | 3,765 |
| Local | 40 | 9.0 | 120 | 473 | 4,948 |

Under §6.6 a lost local costs more than a lost express, the reverse of today. That is a
consequence of the principle, not a defect: a local that never runs leaves nine stations
without service, an express five. A passenger stranded because the train ran out of
shift is as stranded as one whose stop was bypassed.

**Known consequence: a missed stop is in practice a zero.** Under §6.6 greedy's whole
cost on T1 is 106–488 (10 dev seeds). One unserved stop costs at least 500, so an agent
that misses a single stop is at or below greedy and its reward is 0. Agents that miss
one stop and agents that do nothing get the same reward; only the unclipped score
separates them.

### 3.4 C3 Freight (Settled)

`0.5 × late`, plus 50 if `late > 30`. The F13 port cutoff stays at 400.

### 3.5 C4 Crew timeout (Settled, starting value)

300 per timeout. The section is still blocked for 45 ticks, and the delays that causes
are charged through C1–C3. A timeout is a red-flag event.

Why it stays: without it, a timeout that delays nobody costs nothing, and for a freight
train a timeout costs about 22 against 4 for a relief. Exceeding crew hours is the one
failure here that is a legal matter on a real railway.

**The size needs calibrating.** Passenger charges under §6.6 are roughly eight times
smaller than today, so 300 weighs far more than it used to. Greedy has 4–5 crew timeouts
per T2 episode; at 300 each that is 1,200–1,500 on top of a §6.6 cost of about
700–1,650. See open item 1.

### 3.6 C5 Gridlock and operations (Settled)

- **Gridlock:** 1000 per gridlock, charged when it forms. The trains stay stuck; their
  unserved stops are charged as missed calls (§3.3), on top of the 1000.
- **Unplanned stop:** freight 4, passenger 1.
- **Platform change at a hub:** `v × (alight + board)/100 × 2`.

### 3.7 Safety failures

Collisions and running past the end of an authority stay impossible: the simulator
refuses the command. Service failures are charged and counted, never an automatic zero,
because one slip in about 60 decisions would otherwise erase the episode and put very
different models at the same score.

### 3.8 Exploit check on hold-freight (measured)

§6.6 removes the extra unfinished-freight charge that was calibrated to stop the
"hold all freight" exploit. This was tested by re-scoring real episodes under §6.6:

| Tier | Seeds | Hold-freight beats greedy today | Under §6.6 |
|---|---|---|---|
| T1 | 10 | 0 | 0 |
| T2 | 4 | 0 | 0 |

The exploit does not pay on any tested seed. An earlier claim that it would is
**withdrawn**. The T2 margin is thin (about 5% on one seed), as it is today, so the full
exploit checks in `DESIGN.md` §6.5 must be re-run once the rules are implemented.

### 3.9 Scenario data needed

Passengers getting on and off at each stop; connections (feeder, connector, hub,
passengers, minimum transfer time); headway `H` per station and destination; freight
window and miss cost; the published platform at hubs.

---

## 4. Network

### 4.1 What changes (Settled)

- **A graph replaces the chain.** Nodes are stations and terminals. A train has an
  origin, a destination and a route, instead of a direction.
- **A junction is an ordinary station with three neighbours.** No separate node type,
  maximum degree 3, at least one loop.
- **Several origin–destination pairs.** Train count was never the gap; trains with
  different start and end points were.
- **Irregular layouts** built from a few templates with random perturbation: flat
  junctions, dead-end branches to a port or depot, uneven branch lengths, a stretch with
  few loops. Not fully random graphs, which give cycles everywhere and unsolvable seeds.
- **Traffic scaled to the network.** A bigger map with the same traffic is easier, so
  the generator keeps the busiest section as loaded as today.
- **Terminals have a finite number of tracks.**

### 4.2 Sizes (Starting values)

| | T1 | T2 | T3 |
|---|---|---|---|
| Stations, including junctions, excluding terminals | 10 | 16 | 24 |
| Terminals | 3 | 4 | 6 |
| Branches off the trunk | 1 | 2 | 4 |
| Bypasses | 0 | 1 | 2 |
| Junction stations | 1 | 4 | 8 |
| Trains | 10 | 14 | 28 |
| Directed origin–destination pairs | 4 | 6–8 | 10–12 |
| Shift (ticks) | 240 | 720 | 720 |

- Branch: 2–4 stations, ending in a terminal. Bypass: 2–3 stations, 20–40% longer
  running time than the stretch it avoids, skipping at least one station where some
  train is scheduled to call.
- Sections, speeds and loops per station keep today's generator ranges.
- At least 70% of trains cross the busiest trunk section.
- From T2 up, one branch terminal is a port or depot for freight.

| Terminal | Tracks |
|---|---|
| Trunk end | 3: two 300 m platform, one 700 m goods |
| Branch end | 2: one 300 m platform, one 700 m goods |
| Port or depot | 2: both 700 m goods |

A finished train holds its track for the 10-tick turnaround, or until its next service
departs. An entering train takes a track at its entry tick and waits outside if none is
free. The generator checks that scheduled peak occupancy is at most the track count
minus one.

**These are config defaults, not a specification.** Do not hard-code them. Three have a
clear test: train counts (greedy still finishes every train, and `C_best` stays 30%
below greedy), the 70% trunk load (raise it if seeds are rejected for lack of headroom),
and the missed-call charges (no stop-skipping exploit beats greedy).

### 4.3 What a review should focus on

The numbers are cheap to change. These decisions are not:

1. **Authority on a graph.** `authorize(train, station, track)` is ambiguous once two
   paths reach the same station. Default: one hop per authority, or a multi-hop
   authority with an explicit list of stations.
2. **Who owns the route.** Default: the dispatcher chooses it. Misroutes and missed
   calls only exist in that case.
3. **Gridlock detection.** Gridlock is permanent and costs 1000, so a false positive or
   a miss changes scores directly. Needs property tests on graph layouts.
4. **Misroute detection.** A reachability check at grant time, counted once.
5. **Terminal capacity and entry.** Must not produce seeds no dispatcher can solve.
6. **Chain as a special case.** An old chain scenario expressed as a graph should replay
   to the same trajectory as the old engine.
7. **Determinism and replay.** No ordering may depend on set or dict iteration.
8. **What the agent sees.** `status` and `stringline` assume one line; decide the
   per-corridor format and check the token size on the largest tier.
9. **Manual coverage.** Every new rule and cost component appears in `manual()`.
10. **Whether it is harder.** Measure the gap between greedy and `C_best` on the new
    seeds; do not argue it from the layout.

---

## 5. Order of work

Per `DESIGN.md` §6.6, the new cost function lands with the graph network. The reward
anchor, the terminus change and the removal of reversal do not depend on the graph and
can go in first on the chain, which makes later score changes easier to attribute.

The exact solver (`railway/ref/optimal.py`) covers only event-free scenarios on the
chain and needs `ortools`, which is not a project dependency. It has to be extended to
the graph and to every cost term above; where it cannot prove an optimum, `C_best` is
the best known cost.

---

## 6. Implemented on branch `v03` (2026-11-14): what changed during the build

Everything in §2–§4 is implemented, with these decisions taken while building. Each is a default the team can
revisit.

**Network and rules**

- **Sizes calibrated.** T3 is 28 stations (24 left no room for four branches and two bypasses) and the
  structural holdout T2-X is 24 stations, three branches, two bypasses. Branch and bypass lengths shrink to
  fit the station budget.
- **Legacy chain tiers** keep the old layouts as C1, C2, C2-S14, C3, used for regression only.
- **Routes.** `authorize` takes an optional `via` (stations in order) when two shortest routes exist; without
  it the grant is refused with `R-ROUTE-AMBIGUOUS`. A train never runs back over the section it arrived by,
  but it may turn round on a bypass loop (a real "balloon loop" move).
- **Off-route grants** succeed and say what was lost (`R-OFF-ROUTE`). A misroute is counted once, when a grant
  first makes the destination unreachable. A skipped stop is not a misroute.
- **Terminals.** A train entering a full origin terminal waits outside and fires a new decision point,
  `ENTRY_BLOCKED`, once. Without it, time ran on silently.
- **Gridlock is an exact deadlock check.** v0.2 flagged two trains waiting on each other even when a third,
  free train was about to clear the way. With a permanent 1000 charge that false positive would have cost
  agents real score. Now a standing train is stuck only if every way forward is held by trains that are
  themselves stuck. Stuck trains form one deadlock group; a new group is charged once.
- **Regression check.** On chain scenarios the v0.3 engine moves trains exactly as v0.2 did
  (`tests/v03/test_chain_regression.py`, random action scripts replayed through both engines).

**Anchor and verifier**

- `C_best` = the cheaper of the scripted reference and the CP-SAT solver. The solver covers event-free tiers
  (T1); it models partial runs, terminal track pools and the new cost exactly. It is proven optimal only
  when its lower bound equals its dispatched plan and the network has no loop. On 6 T1 seeds it proved 4
  optima of 1–15 against the reference's 70–186.
- The full-knowledge beam search is far weaker than the reference on graphs and is no longer part of the
  anchor.
- Every task ships its best plan. The oracle replays it (exactly 1.0), and **the verifier re-derives both
  anchors** (runs greedy, replays the plan) instead of trusting `norm.json`. This closes the red team's
  patched-norm finding.
- `details.json` carries the KPIs: on-time % (at most 5 minutes late), passenger delay minutes, freight
  delivered %, red flags, clean shift.

**Still open**

1. The size of the crew-timeout charge (300) on the new cost scale.
2. Whether the beam search should be improved, or the reference strengthened, for the tiers with events,
   where the reference is the only anchor.
3. Re-run the full exploit battery (red team) and calibrate the train counts on accepted seeds.

## 7. Open items (from the discussion)

1. **Size of the crew-timeout charge** on the new cost scale (§3.5). 300 is today's
   number.
2. **Training signal.** §6.6 trains on the clipped reward with the unclipped value as a
   tie-break. Training on raw cost, as in v0.2, is simpler and equivalent below greedy.
3. **Whether "the reference never gridlocks" stays in the seed filter**, now that the
   reference is only a baseline.
4. **Full exploit checks** under the new rules, on the complete seed sets (§3.8).
5. **The pitch deck and animation page** follow §6.6 as written and need the one
   amendment at the top of this file (the crew charge stays).

---

## 8. Feedback

astra feedback here
