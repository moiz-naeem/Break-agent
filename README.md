# Railway network dispatcher: an RL environment for long-horizon control

An agent works one shift as the train dispatcher of a busy railway network. It talks to a simulator through
the tools of an MCP server (`railway`), decides which train goes first wherever two trains want the same
track, reacts to breakdowns and closures, and keeps crews within their legal hours. When the shift ends, the
reward is minus the real-world cost of everything that went wrong:

```
reward = -(C1 + C2 + C3 + C4 + C5)        # closer to 0 is better; there is no [0, 1] rescaling
```

Everything here is deterministic: the same seed gives the same timetable, the same disruptions and, for the
same actions, the same reward, which the verifier re-checks by replaying the agent's tool calls.

| | |
|---|---|
| Tiers | T1 Trunk & Branches (180 min), T2 Converging Hub (240 min), T3 Two-Hub Corridor (360 min) |
| Harbor datasets | [`datasets/net-dev-mini`](datasets/net-dev-mini) (3 tasks), [`datasets/net-test-core`](datasets/net-test-core) (15 held-out tasks) |
| Seeds | dev 1–9999, held-out test 10000–19999, training 20000+ (`seeds/net-T*-{dev,test,train}.json`, hash-checked) |
| Design | [`docs/network-v0.3.md`](docs/network-v0.3.md), cost function [`docs/net-cost.md`](docs/net-cost.md), changes from v0.2 [`docs/design-changes-v0.3.md`](docs/design-changes-v0.3.md) |

## Why railway network dispatching, and what an agent learns here

Real railways run most traffic on autopilot. Automatic route-setting sends each train along its booked path
as long as nobody is in the way. The dispatcher is only needed when trains compete, so in this environment
**the simulator jumps from one real decision to the next**: every turn the agent sees is a genuine trade-off
between several trains, not busywork.

Those trade-offs are hard because the consequences come later and spread:

- **Mixed speeds.** A 200 km/h intercity and a 740 m, 80 km/h bulk freight train share the same tracks. A
  freight train that is stopped at a red signal needs 5 minutes to get going again. Letting the fast train
  pass looks free now and can strand the freight train an hour later.
- **Flat junctions and single-track branches.** Two routes cross on the level, and branches have one track
  for both directions. One badly ordered pair of trains can lock a whole branch (a gridlock).
- **Terminals and hubs.** Trains turn around at terminals and must leave again on time. Passengers change
  trains at the hubs, so holding a connecting train helps some passengers and delays others.
- **Crews.** A crew may work at most 240 minutes. The agent is warned 30 minutes before the limit and must
  arrange a relief at a crew-relief platform, or the train stops for 45 minutes where it stands.
- **Disruptions.** Breakdowns, slow zones and track closures are announced during the shift, some before
  they start.

A fixed priority rule ("intercity first") fails here: every policy we tried that ignores what will happen
30–90 minutes later pays for it at the end of the shift. To score well, an agent has to plan ahead, keep its
own plan in mind across many turns (it only sees its last 5 tool results, plus a short notepad), and weigh
passenger, freight and crew costs against each other.

## Environment: tiers, tools and reward

### Tiers

| Tier | Network | Shift | Trains | Decisions per shift | What makes it hard |
|---|---|---|---|---|---|
| **T1** Trunk & Branches | double-track trunk, a flat junction, a single-track branch, a port spur | 180 min | 16–19 | ~25–30 | slow freight to the port over one track; branch turnarounds |
| **T2** Converging Hub | four lines converge on one hub's throat; two terminals | 240 min | 56–59 | ~80–95 | hub platform contention and passenger connections |
| **T3** Two-Hub Corridor | two hubs joined by a trunk with a bypass, branches and a port | 360 min | 108–118 | ~160–200 | long horizon, crew limits (shift > 240-min duty), reversible single-line working |

### MCP tools

The agent reaches the simulator only through the `railway` MCP server (a separate container; the agent never
sees the simulator's files or the timetable file).

| Purpose | Tools |
|---|---|
| Look | `manual`, `network`, `timetable`, `status`, `conflicts`, `bulletins`, `kpis`, `stringline` |
| Remember | `notes` (a 1,500-character notepad that stays on screen) |
| Act | `order` (who uses a track first), `hold`, `platform`, `route`, `skip`, `short_turn`, `cancel`, `single_line`, `relieve_crew` |
| Move time | `advance_to_next_event` (runs the clock to the next decision), `end_shift` |

Each shift allows 600 tool calls. Refused commands come back with a rule code (for example `R-NO-ROUTE`) and
change nothing.

### Reward: five real-world cost terms

The reward is the raw cost, in units of "100 passenger-minutes of lateness" (full definitions and a worked
example: [`docs/net-cost.md`](docs/net-cost.md)).

| Term | What it charges |
|---|---|
| **C1** passenger lateness | every late arrival, weighted by how many passengers get off and by a lateness curve that steps up at 15, 30 and 60 minutes |
| **C2** missed calls and unfinished trains | a skipped stop (500 + the passengers stranded), a missed connection, and every train still out at shift end (200, plus 1,500 for undelivered freight) |
| **C3** freight lateness | late freight beyond its delivery window, and missed port cutoffs |
| **C4** crew timeouts | 40 per crew that runs out of hours (plus the 45-minute stop it causes) |
| **C5** gridlock and operations | 1,000 per gridlock, plus small charges for unplanned signal stops and platform changes |

We deliberately do not squash this into [0, 1]: a reward of −150 and one of −3,800 on the same shift are very
different days on a railway, and the raw cost keeps that difference visible.

## One-command reproduction

### Set up and test

```bash
uv sync
uv run pytest tests/net -q        # engine, cost, session, Harbor export/verifier and oracle tests
uv run ruff check railway tests
```

### Run the Harbor checks

Needs Docker and the Harbor CLI (Linux, macOS or WSL2). On
Windows set `PYTHONIOENCODING=utf-8` first. Run from the repository root.

```bash
# 3 tasks, one per tier (T1-1, T2-2, T3-1): a ~1-minute sanity check of all three tiers
harbor run -y -p datasets/net-dev-mini -a oracle
harbor run -y -p datasets/net-dev-mini -a nop
PYTHONPATH=. harbor run -y -p datasets/net-dev-mini -a railway.harbor.agents:NetStopEarlyAgent

# 15 held-out tasks: the first 5 test seeds of each tier
harbor run -y -p datasets/net-test-core -a oracle
harbor run -y -p datasets/net-test-core -a nop
PYTHONPATH=. harbor run -y -p datasets/net-test-core -a railway.harbor.agents:NetStopEarlyAgent

# a model (OpenCode agent; any provider host in task.toml's allowlist)
GOOGLE_GENERATIVE_AI_API_KEY=... harbor run -y -p datasets/net-test-core -a opencode -m google/gemini-4-carbon
```

What to expect:

- **`oracle`** runs [`solution/solve.py`](railway/harbor/net_templates/solve.py): a plain booked-order
  dispatcher that talks to the MCP server like any agent (no simulator imports, no file access). It orders
  every conflict by booked time, moves stranded trains to another platform at the same station, and relieves
  warned crews who would otherwise run out of hours. It always finishes the shift.
- **`nop`** does nothing. Automatic route-setting stops at the first conflict, so most trains never finish;
  the reward is a large negative number (about −25,000 to −33,000 on T1).
- **`NetStopEarlyAgent`** dispatches like the oracle for 30 minutes and then ends the shift. It loses badly.
- Every trial writes `verifier/reward.txt` (minus the replayed cost) and `verifier/details.json` (cost by
  term, KPIs, counts). A trial **without** `reward.txt` is an infrastructure or replay error to rerun, never a
  score.

### Re-export the datasets

```bash
uv run python -m railway.harbor.net_export --seed-file seeds/net-T1-dev.json --seed-file seeds/net-T2-dev.json \
    --seed-file seeds/net-T3-dev.json --first 1 --set dev --out datasets/net-dev-mini
uv run python -m railway.harbor.net_export --seed-file seeds/net-T1-test.json --seed-file seeds/net-T2-test.json \
    --seed-file seeds/net-T3-test.json --first 5 --set test --out datasets/net-test-core
```

The exporter regenerates every scenario and refuses to write it unless its hash matches the seed list.

### Re-run the model benchmark

```bash
uv run --with modal modal run infra/modal/net_bench.py::run --split test --tiers T1,T2,T3 --reps 3 \
    --models gemini-carbon-high,qwen38-think
uv run --with modal modal run infra/modal/net_bench.py::report --run-name <run-name>
```

Rows land in `results/net_bench/<run-name>/rows.jsonl` with a `summary.md` table. The fine-tuned model is
served from its own endpoint as `qwen38-sft`; add it to `--models` once that endpoint is registered in
`railway/loop/providers.py`.

The rows published below came from this same benchmark loop (`railway/loop`; every row says `harness: loop`),
run on rented GPU containers and a university cluster rather than on Modal.

## Benchmark and post-training results

Held-out **test** seeds only; every row's scenario hash matches `seeds/net-T*-test.json`. Each seed is played
up to 3 times; reps are averaged per seed, then seeds are averaged. CI = 95% bootstrap over seeds.

> **Status (snapshot 2026-11-15 09:36 UTC, full table in [`docs/benchmark-results.md`](docs/benchmark-results.md)).**
> The benchmark was still running at this snapshot, so every number below is partial. Finished so far:
> Gemini 4 Carbon 87 of 270 planned episodes (58 T1, 29 T2); base Qwen 18 of 270 planned episodes (17 T1, 1 T2); fine-tuned Qwen 13 of 30 planned episodes (13 T1).
> One episode takes about an hour of model time, which is why the counts are low. Base Qwen's earlier attempts
> were stopped by the serving setup (context-window and "no upstreams" errors, see
> [`docs/failure-analysis.md`](docs/failure-analysis.md)); those are infrastructure errors and are not scored.
> Cells marked "running" are not results.

### T1 held-out test seeds

| Model | Seeds | Episodes | Mean reward [95% CI] | C1 | C2 | C3 | C4 | C5 | On time | Freight delivered | Clean shifts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemini 4 Carbon (`gemini-carbon-high`) | 26 | 58 | **−1,549.9** [−2,407.2, −943.2] | 356.5 | 1,147.5 | 3.3 | 0.0 | 42.6 | 67.6 % | 78.2 % | 23.7 % |
| Qwen 3.8 27B, base (`qwen38-think`) | 14 | 17 | **−2,518.4** [−4,127.5, −1,234.5] | 499.5 | 1,817.0 | 2.6 | 0.0 | 199.4 | 58.9 % | 92.9 % | 17.9 % |
| Qwen 3.8 27B, fine-tuned (`qwen38-sft`) | 13 | 13 | **−1,145.3** [−1,765.2, −632.2] | 482.6 | 631.4 | 6.0 | 0.0 | 25.3 | 66.5 % | 92.3 % | 15.4 % |

The three rows cover different seeds (each model has finished a different subset), so compare models only
through the paired numbers below, not by reading down the mean column.

For scale, on Carbon's 26 seeds: doing nothing scores **−28,618.6** and the booked-order
oracle **−1,439.0**. Carbon beats the booked-order oracle on 10 of 26 seeds but is more than 100 points worse
on 10, and its average is worse.
Carbon against base Qwen, on the 12 T1 seeds both finished: Carbon −1,955.2, base Qwen −1,998.6, a difference of +43.4 (95% CI [−1,515.6, 1,509.6]); Carbon is better on 4, tied on 0, worse on 8 (not yet distinguishable from noise at this sample size).

T2 and T3: Carbon has finished 29 T2 episodes on 21 seeds (mean **−6,557.2**, 95% CI [−8,698.7, −4,721.4]) and base Qwen has finished 1 T2 episode (mean −2,504.9; Carbon scores −1,112.8 on that seed). T3 is still running.

### Post-training

The fine-tuned model is Qwen 3.8 27B after supervised fine-tuning on dispatcher traces from a stronger
teacher, Claude Opus 5.5\*. The teacher traces were collected only on the disjoint **training** seeds (20000+),
so no test seed was seen in training. The comparison is paired: base vs fine-tuned on the exact 7 T1 seeds
finished by all three models so far.

| Tier | Paired seeds | Carbon | Base Qwen | Fine-tuned Qwen | SFT − Base [95% CI] |
|---|---|---|---|---|---|
| T1 | 7 | −1,559.3 | −1,915.5 | **−1,323.7** | **+591.8** [−1,041.8, +2,385.0] |

On those 7 paired T1 test seeds, fine-tuning improved Qwen by **+591.8** points (wins/ties/losses: 4 / 2 / 1), doubled its clean-shift rate (14.3 % → 28.6 %), reduced unfinished trains (2.14 → 1.43), and closed **166.1 %** of the Carbon-to-Qwen gap (moving ahead of Carbon by 235.5 points on that subset, though the 7-seed CI still spans zero).

\* Why Opus 5.5: it is the only model for which we could collect proper reasoning traces, and we could get
them only through the Factory Droid harness. That harness gives the model a different context from the
benchmark loop the three models above run in, so Opus 5.5 is the source of the training data only. It is not
a benchmark entry, it has no row in these tables, and no score of its own should be compared with them.

## How the fine-tuned model was trained

- **Method.** Supervised fine-tuning of a LoRA adapter (rank 16) on Qwen 3.8 27B: 6 optimizer steps of 8 samples
  each, learning rate 5e-5. Each sample is one teacher turn with the exact prompt and tools the benchmark loop
  sends, so training and evaluation see the same input format. Only turns with exactly one tool call are used.
- **Teacher data.** Dispatcher episodes played by Claude Opus 5.5 through Harbor with the Factory Droid agent
  (`infra/modal/_harbor_droid_agent.py`), on the training seeds only (`seeds/net-T*-train.json`, 20000+).
  `railway/train/env.py` refuses other seeds by default.
- **Code.** `railway/train/` holds the SFT trainer (`sft.py`), the teacher-data builder (`teacher.py`,
  `teacher_data.py`), the rollout and evaluation code and a multi-turn GRPO trainer (`grpo.py`). Modal entry
  points are in `infra/modal/train.py` and `infra/modal/serve_qwen.py`; the method is written up in
  [`docs/training.md`](docs/training.md). Tests: `tests/train/`.
- **What the reported model is.** The fine-tuned row above is the SFT adapter only. The GRPO trainer is
  included but no GRPO-trained model is reported. The scored adapter was trained on a single rented H100
  container, not on Modal.
- **Not in this repository.** The adapter weights and the teacher episode files.

## Where agents fall short, and why

The full evidence, with transcripts, is in [`docs/failure-analysis.md`](docs/failure-analysis.md). In short:

1. **Stranding slow freight at shift end (Carbon, T1-10004).** The booked-order oracle finishes every train
   for a cost of 157.8. Carbon cost 3,784–3,930 in all three tries. To give passenger trains a clean run, it
   held both heavy freight trains at the entry for 25 and 36 minutes, after writing their booked times in its
   own notes. It then estimated their running time on an empty railway and missed that the 80 km/h train
   would be stuck behind traffic and lose 5 minutes at every red signal. Both freight trains (and one commuter
   train) were still out at minute 180: 0 % of the freight delivered, 3,608 points for unfinished trains.
   Across all its T1 episodes, Carbon keeps passengers *more* on time than the oracle, but loses that and more
   on trains that never finish (C2).
2. **Self-made gridlock on a single track (Carbon, T1-10017).** The oracle scores −54.0. Carbon moved a
   freight train into the port road where a second freight train was booked, correctly planned to re-platform
   the second train later, and in one try wrote that plan in its notepad (−303.7). In the other two tries it
   did not; the plan scrolled out of its 5-call memory, the second train ran onto the single port track in
   front of the occupied road, and the line locked up (−2,749.5 and −2,738.7). Same model, same seed, a
   9× difference decided by whether it kept its own plan.
3. **Long-horizon overload (Qwen).** Base Qwen re-reads the network and timetable, sends several commands in
   one turn, invents track names, and its thinking often runs to the cap. On the long T2/T3 shifts (hundreds of
   turns) the growing conversation hits the serving window early, so it has not yet finished one; on the one
   short T1 shift it finished, it dispatched cleanly.

These are the skills the environment is built to reward: anticipating knock-on delays of slow trains,
keeping a plan across a long episode, and protecting single-track sections from deadlock.

## Reward integrity and anti-cheating

The reward can only be earned by running the railway well. What we checked (details:
[`docs/redteam-net.md`](docs/redteam-net.md)):

- **Lazy policies lose by a wide margin.** On every one of the 15 dev seeds (T1–T3), doing nothing, quitting
  at minute 30, cancelling every train, or only advancing the clock costs **5.4× to 122.8×** more than basic
  booked-order dispatching. Parking all freight (`hold_all_freight`) loses on all 15 seeds, and ignoring crew
  warnings costs an extra 840–4,894 on every T3 seed.
- **The score is replayed, not trusted.** The verifier fetches the episode log from the simulator container
  (bearer-token auth), checks the scenario hash and rules version, replays every tool call in a fresh
  simulator, recomputes the cost from the replayed events, and only then writes `reward.txt`. It rejected a
  forged log claiming zero cost on 15 of 15 seeds; replays match to within 1e-6.
- **Isolation.** The agent container has no simulator, no timetable file and no token; it runs as a non-root
  user, and any pre-written reward file is deleted before verification.
- **A loophole we found and closed.** Our red-team audit found that `platform()` accepted a platform at a
  *different* station, so a train could "serve" a stop at the wrong station without the missed-call charge,
  cutting cost by up to 54 % on some dev seeds. The engine now refuses any platform that is not at the
  station of the train's next booked call (`R-NO-ROUTE`), with regression tests in `tests/net/test_engine.py`
  and `tests/net/test_session.py`.

## Repository map

| Path | What it holds |
|---|---|
| `railway/net/` | simulator engine, interlocking, routing, cost, timetable generator, MCP session and server |
| `railway/harbor/net_export.py`, `net_verify.py`, `net_templates/` | Harbor task export, replay verifier, oracle client |
| `datasets/net-dev-mini`, `datasets/net-test-core` | exported Harbor tasks (v0.3 network) |
| `infra/modal/` | benchmark and Harbor runners, training and model-serving jobs on Modal |
| `railway/train/`, `docs/training.md` | post-training: SFT and GRPO trainers, teacher data, evaluation |
| `seeds/net-T*-*.json` | accepted dev, test and training seeds with scenario hashes |
| `tests/net/` | test suite for the network track |
| `results/`, `docs/benchmark-results.md` | raw benchmark rows, scoreboard and the full result tables |
| `archive/` | earlier work kept for reference: the day-1 Harbor spike, the v0.2 single-track tasks, internal planning docs |
