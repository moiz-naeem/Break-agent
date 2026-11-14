# Design changes agreed for v0.3 (discussion record, 2026-11-14)

This records what was discussed and agreed about the network, the reward and the cost
function, and the reasoning behind each point. Nothing here is implemented yet.
`DESIGN.md` (v0.2) still describes the code as it is.

Status labels: **Agreed** (decided in the discussion), **Proposed** (my default, not yet
confirmed), **Open** (needs a decision or a measurement).

---

## 1. Summary

| Area | Change | Status |
|---|---|---|
| Reward | Top anchor becomes the best known cost (proven optimum where possible) instead of the scripted reference | Agreed |
| Reward | No clip at the top; scores above 1.0 are reported and flagged | Agreed |
| Reward | Greedy stays as the bottom anchor | Agreed |
| Reward | Reference demoted to a baseline row and the training teacher | Agreed |
| Reward | Real-world KPIs and a clean-shift rate reported beside the score | Agreed |
| Seed filter | No longer requires the reference to beat greedy by 30% | Agreed |
| Seed filter | 30% headroom rule measured against the best known cost instead | Proposed |
| Cost | Terminus multiplier (×2 at the last stop) removed | Agreed |
| Cost | Reversal removed entirely (tool, cost component, rules) | Agreed |
| Cost | New service failures (misroute, missed call, wrong terminal) charged heavily, not an automatic zero | Agreed |
| Cost | Sizes of those new charges | Proposed |
| Network | Chain replaced by a graph with several origins and destinations | Agreed |
| Network | Irregular, realistic layouts; traffic scaled to capacity | Agreed |
| Network | Terminals get a finite number of tracks | Proposed |
| Harbor oracle | Replays the stored best plan so it scores exactly 1.0 | Proposed |
| Order of work | Reward anchor and cost changes first, then the network | Proposed |

---

## 2. Reward

### 2.1 What it is today

```
reward = clip( (C_greedy − C_agent) / (C_greedy − C_ref), 0, 1 )
```

Implemented in `railway/harbor/verify.py` (`reward_from`). `C_greedy` is the cost of the
greedy FIFO script, `C_ref` the cost of the scripted reference dispatcher. Both are
stored per task in `tests/norm.json`.

### 2.2 What is wrong with it

The concern raised was that the reward looks poor and not trustworthy compared with the
cost function. The data supports that, for one specific reason: **the top anchor is
weak**.

The reference is a scripted heuristic. A full-knowledge search beats it by a wide margin
on four of the five seeds in `docs/strength-check.md`:

| Seed | Do-nothing | Greedy | Reference | Offline search | Current score of the offline plan (unclipped) |
|---|---|---|---|---|---|
| 0 | 7571.5 | 1746.7 | 296.7 | 176.6 | 1.08 |
| 5 | 8716.3 | 1408.2 | 208.5 | 208.5 | 1.00 |
| 6 | 10058.9 | 4809.3 | 2620.7 | 1155.2 | 1.67 |
| 7 | 9850.1 | 2106.2 | 1219.3 | 213.9 | 2.13 |
| 10 | 8659.8 | 1026.3 | 447.0 | 22.5 | 1.73 |

Do-nothing, greedy and reference were re-run on 2026-11-14; the offline column is from
`docs/strength-check.md`.

Consequences:

- **"1.0" means "matched our script", not "dispatched well".** An agent twice as good as
  the reference gets the same 1.0, because the clip hides everything above it.
- **Every score depends on the reference's code.** Change the script and every score
  moves.
- **The seed filter is biased.** It keeps only seeds where the reference beats greedy by
  30%, which favours scenarios that suit the script's style.

A later finding makes this sharper: the exact solver's proven lower bound on seed 5 is
4.2 (pinned in `tests/optimal/test_optimal.py`), while the reference and the offline
search both cost 208.5 there. The true optimum is far below what has been called good.

### 2.3 Why the reward is not simply removed or made absolute

The first idea was to remove the reward or replace it with something closer to real
life. Two things argue against dropping the normalisation:

- **Harbor needs one float per task**, and raw cost is not comparable across seeds
  (greedy alone ranges from 1026 to 4809 on the seeds above).
- **An absolute scale compresses the scores.** The obvious real-life version,
  `1 − C_agent / C_do-nothing` (share of service delivered), was measured on the first 10
  T1 dev seeds:

| Seed | Greedy | Reference | Offline search |
|---|---|---|---|
| 0 | 0.77 | 0.96 | 0.98 |
| 5 | 0.84 | 0.98 | 0.98 |
| 6 | 0.52 | 0.74 | 0.89 |
| 7 | 0.79 | 0.88 | 0.98 |
| 10 | 0.88 | 0.95 | 1.00 |
| 13 | 0.85 | 0.93 | – |
| 15 | 0.77 | 0.90 | – |
| 16 | 0.76 | 0.95 | – |
| 17 | 0.80 | 0.99 | – |
| 21 | 0.74 | 0.90 | – |

Greedy already scores 0.52–0.88 and the reference 0.74–0.99. Everything competent lands
in the top part of the scale and the gap between models disappears. Do-nothing is so
catastrophic (7,500–10,500) that any scale with "nothing delivered" as zero has this
problem.

So the greedy anchor is the part worth keeping: it is what gives the scale resolution.

### 2.4 The new reward (Agreed)

```
reward = max( 0, (C_greedy − C_agent) / (C_greedy − C_best) )
```

- **`C_best`** is the proven optimum where a solver can prove one, otherwise the best
  known cost (best of the reference, the offline search and any recorded run), labelled
  as "best known" in `norm.json` and the report.
- **No top clip.** Against a proven optimum a score above 1.0 is a bug signal. Against a
  best-known anchor it means the agent found a better plan, which then becomes the new
  anchor. Either way it is flagged.
- **Bottom stays at 0.** Do-nothing, stop-early and anything worse than greedy score 0.
  The unclipped score is still reported so weak models keep their ordering.
- **The reference** becomes one baseline row in the table and stays the training
  teacher, because it can label any state.

Expect scores against the optimum to be much lower than against the reference.

### 2.5 Reported beside the score (Agreed)

For human readers, the headline table also shows real operations numbers:

- on-time percentage;
- weighted passenger delay minutes;
- freight delivered;
- gridlocks and crew timeouts;
- **clean-shift rate:** the share of episodes with zero red-flag events (misroutes,
  missed calls, gridlocks, crew timeouts).

On-time percentage is not the reward itself, because how it spreads across models has
not been measured.

### 2.6 Seed filter

- **Agreed:** drop "the reference costs at least 30% less than greedy".
- **Proposed:** keep a 30% headroom rule, measured against `C_best`, so the denominator
  of the score stays stable.
- Unchanged: every exploit baseline costs at least as much as greedy; greedy finishes
  every train.
- **Open:** whether "the reference never gridlocks" stays as a condition now that the
  reference is only a baseline.

### 2.7 Harbor oracle (Proposed)

CI requires the oracle to score exactly 1.0 on every task. The reference will no longer
do that. Proposal: the oracle replays the stored best plan for the task.

### 2.8 Limits

- The exact solver (`railway/ref/optimal.py`, CP-SAT) covers only event-free scenarios
  (factors F1–F4, tier T1) on the current chain layout. It needs `ortools`, which is not
  installed in the project environment.
- Once trains have route choices, a proven optimum is harder to compute. This was judged
  not to be a blocker; the solver is to be extended, with "best known" as the fallback on
  the larger tiers.

---

## 3. Cost function

The existing components stay: passenger lateness, freight lateness, crew timeout,
unfinished trains. Two things are removed and one group is added.

### 3.1 Terminus multiplier removed (Agreed)

Today passenger lateness is doubled at the last stop (`TERMINUS_MULT = 2`). New rule:
every timed stop is charged the same.

Reasoning:

- The argument raised was that the last stop matters least, since few passengers are
  left and only the crew is affected, which is already charged. That is not generally
  true in reality: the terminus is often the main city station and the busiest stop.
- The multiplier is still wrong for this simulator. There is one passenger number per
  train, not per stop, so there is no basis for weighting any stop above another.
- It double-counts. A late arrival at the terminus already delays the train's next
  service through the rotation rule, and that lateness is charged there.

Side effect: the charge for an unfinished passenger train uses the same multiplier, so
it shrinks. The exploit checks must be re-run to confirm do-nothing and stop-early still
cannot beat greedy.

Possible later refinement, not planned: per-stop boarding and alighting counts, which is
the realistic way to express passenger density.

### 3.2 Reversal removed (Agreed)

Today `reverse(train)` sends a gridlocked train back to the previous station for 15
ticks and 50 points. It is removed: the tool, the `reversal` cost component and its
rules.

Reasoning:

- The argument raised was that reversing on a railway is unrealistic. It is in fact a
  real move (setting a train back to the previous station), but rare and used only as
  recovery.
- It is not a normal move in the simulator either. It is legal only during gridlock,
  greedy never uses it and the exact solver does not model it. In the three recorded
  model episodes there was no gridlock and no reversal.
- It fits the red-flag view: gridlock is something to prevent, not something to undo for
  50 points.

Consequence, accepted: **gridlock becomes permanent.** The trains involved stay stuck
and are charged as unfinished at shift end. A gridlocked agent almost certainly scores 0
already, because it ends up worse than greedy. On the graph network, trains outside the
jam can still run.

### 3.3 New service failures on the graph network (Agreed in principle)

On the chain there is nowhere wrong to go: wrong direction is refused, scheduled stops
are enforced, passenger trains cannot use goods loops. A graph adds three failures:

| Failure | What it is |
|---|---|
| Misroute | Train sent down a branch that cannot reach its destination |
| Missed call | Train bypasses a station where it is scheduled to stop |
| Wrong terminal | Train ends at an end point that is not its own |
| Circling | Train routed round a loop in the layout without progressing (shows up as lateness) |

Treatment, following how a real railway separates the two kinds:

- **Safety failures** (collision, running past the end of an authority) stay impossible.
  The simulator refuses the command, as an interlocking would.
- **Service failures** are allowed where the route is physically valid, charged heavily,
  and counted separately as red flags in the report.

They are **not an automatic zero**, because:

- one slip in about 60 decisions would erase the whole episode, making the reward nearly
  binary and putting very different models at the same 0;
- a missed station call is, in reality, a reported and penalised incident, not the end
  of the shift;
- training needs a graded signal.

The strictness is carried by the clean-shift rate (§2.5) instead.

The alternative of refusing misroutes outright with a new rule was rejected: it is
simpler, but it removes routing mistakes as something the benchmark can measure.

**Proposed charge sizes (not yet confirmed):**

- misrouted or wrong-terminal train: treated as unfinished (the existing 200 plus the
  lateness charge);
- missed call: a fixed block of passenger-weighted minutes, calibrated so that skipping
  a stop never costs less than arriving late at it.

---

## 4. Network

### 4.1 What it is today

One chain `YW – S1..SN – YE` with two unlimited yards, enforced in
`railway/sim/model.py` (`Network.ordered_chain`). Every train is `east` or `west`.

### 4.2 The goal

A more authentic experience: several places where paths diverge and converge, and many
trains running at once from several starting points to several destinations.

### 4.3 What changes (Agreed)

- **The network becomes a graph.** Nodes are stations, junctions and terminals. A train
  has an origin, a destination and a route, instead of a direction.
- **Several origin–destination pairs.** Train count was not the gap (T1 already runs 8
  trains, T3 runs 22); what was missing is trains with different start and end points.
- **Irregular layouts.** Flat junctions where crossing moves conflict, dead-end branches
  to a port or depot, mixed single and double track, uneven branch lengths, a stretch
  with few passing loops.
- **Alternative routes.** At least one bypass, so routing is a real decision and
  possessions and breakdowns become "go around or wait".

### 4.4 Realism versus difficulty

My first proposal was a tidy shared trunk with symmetric branches, on the grounds that a
bigger network can be easier: if trains spread over branches, conflicts thin out.

The objection was that a tidy trunk is not realistic, and that real networks diverge and
converge in messy ways. That objection is correct, and the recommendation changed:

- The point was about difficulty, not tidiness. Adding track without adding traffic
  makes the task easier.
- Real railways stay congested because timetables are written to fill the capacity.
- So the generator should **scale traffic to the network**: it targets a utilisation
  level on the busiest sections, and the bottleneck sits wherever the layout puts it.
- Layouts come from a few templates with random perturbation, not fully random graphs.
  Random graphs produce things real railways do not have (cycles everywhere) and many
  unsolvable seeds.

### 4.5 Terminals (Proposed)

Terminals get a finite number of tracks. Unlimited yards are the least realistic part of
the current model.

### 4.6 Cost of the change

The chain assumption runs through the engine, the generator, the reference's candidate
plans, the gridlock check, the exact solver, the greedy and exploit policies, and the
stringline diagram (which only makes sense along one corridor). All seeds, norms,
datasets and the frozen prompt have to be regenerated, and the rules version changes.

Context also grows with the map. Greedy through the model loop already uses about 360k
input tokens on a T1 episode, so `status` has to stay compact, per corridor.

---

## 5. Order of work (Proposed)

1. Reward anchor, terminus multiplier and reversal removal, with `ortools` added as a
   dev dependency. Regenerate T1 norms and re-run the exploit checks.
2. Graph network: model, engine, gridlock check, generator with utilisation target,
   policies, solver, tools and manual.
3. New service-failure charges and red-flag counts; KPI and clean-shift reporting.

The reward comes first because the network change invalidates every stored `C_greedy`
and `C_ref` anyway, and new seeds should be normed against the better anchor from the
start.

---

## 6. Open items

1. Sizes of the misroute, wrong-terminal and missed-call charges (§3.3).
2. Whether the 30% headroom rule against `C_best` is the right filter (§2.6).
3. Whether "the reference never gridlocks" stays in the seed filter (§2.6).
4. Oracle replaying the stored best plan (§2.7).
5. Finite terminal tracks (§4.5).
6. How on-time percentage spreads across models, before it gets any weight (§2.5).

---

## 7. Feedback

astra feedback here
