# Red-team playbook: playing the judge against our own environment

> **Reward, as of `v03` @f9ae4e2:** `reward.txt` holds **−(total cost)**. The greedy/best normalised score
> (`score_normalized`, `score_unclipped` in `details.json`) is report-only. Every check in this document is
> therefore a cost comparison: the oracle must land on −C_best, a non-dispatching policy must not cost less
> than greedy, and the numbers in the findings below are costs. `railway/redteam/strategic.py` and
> `harbor_protocol.py` judge in cost terms; one line per attack is in `docs/redteam-attack-notes.md`.
The judges spend 15 minutes looking for a way to score well without dispatching trains, and they rerun
the evaluation themselves: our reference, three agent runs, a do-nothing baseline and a stop-early
baseline. This playbook is how we run that session on ourselves first.

Rubric weights, for orientation: the gap 30 points, meaningful difficulty 30, **reward integrity 25**,
craft 15. Reward integrity is the part this document is about. Craft is lost if any of it only works on
one machine, so everything here is a command.

## What exists

| Piece | Command | What it decides |
|---|---|---|
| Strategic battery | `python -m railway.redteam.strategic --preset T1 --seeds 5 --jobs 5` | Can a policy that is not dispatching score? Does partial play score? |
| Verifier tamper suite | `python -m railway.redteam.tamper --preset T1 --seed 0` | Can the reward be written, forged or renormalised instead of earned? |
| Harbor judge protocol | `python -m railway.redteam.harbor_protocol --dataset datasets/t1-dev-mini` | The judges' five runs, plus hack-elicitation trials, end to end in containers |
| Panel | `make redteam` (or `python -m railway.redteam.panel --preset T1 --seeds 5`) | Runs the two local checks and writes one scoreboard |
| Regression tests | `pytest tests/redteam -m slow` | Keeps the findings below from silently changing |
| Attacker briefs | `railway/redteam/prompts/*.md` | What we hand a frontier model to make it hunt exploits |

Output lands in `results/redteam/`: `scoreboard.md` plus the per-check reports.

## The three layers of attack

An exploit has to come from one of three places. Test them separately, because the fixes are different.

### 1. Strategy inside the rules (`strategic.py`, `attackers.py`)

Nothing is tampered with; the agent just finds a cheap policy that is not dispatching. This is the layer
that actually decides reward integrity for us, because our agent has no filesystem to attack: it only
has MCP tools.

The reward is −(total cost), so the battery judges in costs: an attacker that does not dispatch must not
cost less than greedy, and a mutation of the best plan must not cost less than the plan. The normalised
score is printed in brackets for scale only.

Three families run per seed:

| Family | Members | Pass condition (cost terms) |
|---|---|---|
| Non-dispatching policies | do nothing, stop early, spam `advance`, hold every train (`HoldAll`), passengers only (`PassengersOnly`), hold part of the freight (`PartialFreightHold`), misroute everything (`MisrouteAll`), skip the bypass (`BypassSkip`), park on the main line, one very long authority (`LongAuthHold`), a `via` that revisits sections (`BalloonVia`), jam the terminals (`TerminalJam`), farm crew timeouts (`CrewTimeoutFarm`), no relief at all (`NoRelief`, slow) | must not cost less than greedy |
| Mutations of the best plan | `PlanNoRelief`, `PlanDropFreight`, `PlanHalfFreight`, `PlanMainTracks` | must not cost *less* than the plan itself |
| Cost-neutral mutations | `PlanCancelChurn` | must cost *exactly* what the plan costs |

`PlanCancelChurn` is the potential-shaping trap: cancelling and re-granting the plan's own authorities is
free in the rules, so if it ever lowers cost an agent can farm reward by toggling state. The check is an
equality, not an inequality. (v0.2's `ReverseAbuse` is gone with `reverse` itself: v0.3 has no reversal.)

Two invariants are watched on every episode, not just on the attackers built to break them:

- **Section co-occupancy.** Two trains in one block section at the same tick is a safety failure, not a
  cost question. The `Monitor` client checks it after every call in every run in the battery.
- **Stalled under authority.** An episode that ends with trains permanently stood under a granted
  authority while no gridlock was charged is what an attack on the "stuck" check would produce, so it is
  reported as a warning even when the cost looks normal.

The **stop-early curve** answers the question the rubric asks directly: at what point does play start
scoring? The battery replays the task's own best plan (the 1.0 trajectory), cuts the episode after a share
of its orders, and scores the rest as if the agent had stopped. Reward should stay near zero until nearly
every order is given.

Fail rule: a non-dispatching attacker that costs less than greedy; two trains in one section; a cost-neutral
action that changes cost; a plan mutation cheaper than the plan; a prefix that scores as well as the whole
plan while orders remain.
### 2. The scoring path (`tamper.py`)

Seventeen cases against a real task export, mapped to the Harbor cheat-vector taxonomy
([terminal-bench issue #2086](https://github.com/harbor-framework/terminal-bench/issues/2086)). The
honest episode is the task's own `tests/plan.json`, which scores exactly 1.0, and each case is scored by
running the task's **vendored** `tests/verify.py` in a subprocess, so what the suite reports is what
Harbor would see:

| Vector | Our cases |
|---|---|
| V1 overwrite | patched `norm.json`, patched `plan.json`, patched `norm.json` + `plan.json` together, patched `scenario.json`, patched `railway/sim/cost.py`, patched `verify.py`, swapped `token` |
| V3 reward | pre-written `reward.txt`, pre-written `reward.json` |
| V4 fail-open | empty log, no entries, truncated entries, unclosed log |
| V5 sleeper | both reward files written by a surviving process |
| V7 forgery | a log claiming cost 0 while its entries replay to the real cost |
| V8 disclosure | a foreign scenario's log, a stale `rules_version` |

Each case says what the served log **deserves** and whether an agent in a shipped task could reach the
attack at all. Our architecture removes most of them by construction: the episode log is served by the
`sim` sidecar, and `tests/` (scenario, plan, norm, token) is uploaded only after the agent has finished.
The cases are still worth running, because they are what someone with the repository will try, and because
they are a regression test on `verify.py`.

`verify.py` already deletes any pre-existing `reward.txt` and `reward.json` before scoring and writes
`reward.txt` last, only after a clean replay. V3 and V5 are closed by that, not by luck; the suite is what
keeps them closed.

**Closed in v0.3:** v0.2's one win, `patched_norm`, is gone. The verifier now *re-derives* both anchors
from the frozen scenario (`recompute_anchors`: it runs greedy and replays `tests/plan.json`), so
`norm.json` is only a cache. Rewriting either number, the plan, or the vendored cost module now ends in
`INFRA: bad norm.json: anchors do not match the scenario`, which writes no reward and makes Harbor rerun
the trial rather than score it.

**Remaining win:** `patched_verifier`. Replace `tests/verify.py` with a script that writes `1.0` and the
reward is 1.0, because at that point nothing else is in the loop. This is not a bug to fix; it is the
statement of the trust boundary, and it is why the only integrity claim we make is that `tests/` arrives
after the agent and is never writable by it.
### 3. The model (`prompts/` + `harbor_protocol.py`)

Scripted attackers only find what we thought of. A frontier model told to cheat finds the rest. This is
the method terminal-bench-science uses in CI: `/cheat` appends an adversarial brief to the task
instruction and runs the normal agent on it, then judges the trajectory against a
[cheat-analysis rubric](https://github.com/harbor-framework/terminal-bench-science/blob/main/rubrics/cheat-analysis.toml)
with one criterion per *kind* of hole: answer leakage, verifier permissiveness, grader tampering,
constraint evasion. Their own brief is at
[`.github/hack-trial-prompt.md`](https://github.com/harbor-framework/terminal-bench-science/blob/main/.github/hack-trial-prompt.md);
the open-source elicitation prompt behind the
[Terminal Wrench](https://github.com/few-sh/terminal-wrench) dataset of hackable tasks is
[`prompts/hack_prompt_v5.md`](https://github.com/few-sh/terminal-wrench/blob/main/prompts/hack_prompt_v5.md).
Ours are adapted to an MCP task with no filesystem:

- **`blind-probe.md`** — the agent sees only the MCP tools, as in a real trial. Hunts tool-semantics and
  accounting edges. This is the honest simulation of the judges' hunt.
- **`verifier-aware.md`** — the attacker gets the repository and answers seven questions about the
  scoring path. This is the simulation of a judge reading our README and verifier for 15 minutes.
- **`score-maximiser.md`** — no mention of exploits; just "maximise your score, any means allowed". It
  catches the case where the shortcut is so natural that a model finds it without being told to cheat.

Run the first and third through Harbor, because they are agent trials:

```bash
python -m railway.redteam.harbor_protocol --dataset datasets/t1-dev-mini \
    --model anthropic/claude-opus-5-5 --briefs blind-probe score-maximiser
```

That copies the dataset, appends the brief to every `instruction.md`, and runs the agent on the copy, so
the honest dataset is never modified. Run `verifier-aware.md` by hand in a session with the repository
mounted; it produces a report, not a trial.

Read the trajectories, not just the rewards. A cheat trial that scores 0 while the transcript shows the
model finding a hole we have not closed is still a finding.

## Findings, v0.3 run (T1 dev seeds 0–4, T2 dev seeds 1, 2, 3, 7, 8)

Reports: `results/redteam/strategic-T1-dev.md`, `strategic-T2-dev.md`, `tamper-T1-0.md`, `tamper-T2-1.md`.

1. **No non-dispatching policy scores.** Every attacker in the battery is at reward 0.00 on every seed
   except the two `BypassSkip` cases below. Do-nothing, stop-early, spam-`advance` and malformed-call runs
   all land at the same cost (T1 seed 0: 34664.2 against greedy 59.7), because an unplayed shift pays the
   missed-call, unfinished-train and undelivered-freight charges for the whole timetable.
2. **The stop-early curve is flat at zero.** On all ten seeds, cutting the best plan after 10%, 25%, 50%,
   75% or 90% of its orders scores 0.00; only the whole plan reaches 1.00. Partial play earns nothing.
3. **Finding: the greedy anchor is soft on some seeds.** `BypassSkip` ("always authorize the farthest node
   you can get, never stop in between") scored **0.439** on T1 seed 3 (cost 161.03 against greedy 286.24)
   and **0.244** on T2 seed 1 (2493.0 against 3070.22), both with zero missed calls, zero misroutes and
   every train finished. This is not a scoring exploit: the policy really does run the railway, with one
   rule and no planning. It does mean that on those seeds a quarter to half of the scale is reachable
   without reasoning, and it breaks the seed filter's promise that every exploit baseline costs at least
   what greedy costs. Proposals (for the owner to decide, no cost weights involved): extend the seed
   filter to require *every* battery attacker to cost at least greedy, which rejects these two seeds; or
   move the bottom anchor to the cheapest of a small portfolio of one-line heuristics instead of greedy
   FIFO alone.
4. **Finding: `patched_plan_and_norm` wins on T2** (writes 1.2247 where the served episode honestly
   deserves 0.981). The verifier re-derives `C_best` by replaying `tests/plan.json`, so a worse plan plus a
   `norm.json` that agrees with it is internally consistent and passes the anchor check. It is not
   reachable by an agent (`tests/` is uploaded after the agent finishes) and the score it produces is
   above 1.0, which the verifier already flags as `above_best`. Proposal: bind `plan.json` and `norm.json`
   to the committed seed list, by hashing both into the dataset manifest and checking that hash at score
   time, so the anchors cannot be moved consistently either.
5. **Finding: `patched_verifier` wins, and that is the trust boundary.** Replace `tests/verify.py` and it
   writes whatever you tell it. Nothing downstream can defend against that, which is why the only
   integrity claim we make is the one about when `tests/` arrives and who can write it.
6. **Finding: `cancel_authority` is not reversible.** On T2 seed 2, cancelling an authority the
   interlocking had just granted and immediately re-issuing the identical call is refused:
   `REFUSED R-ROUTE-AMBIGUOUS: more than one route from B1 to A2: name stations in via`. The same call is
   grantable in one state and not in the next, with no event in between, so a cancel can strand a train
   and cost thousands. It is not an exploit (it only ever costs more), but an agent cannot predict it from
   the manual. Proposal: say in `manual()` that cancelling releases the resolved route and that a re-issue
   may need an explicit `via`, or make `authorize` resolve the route the same way in both states.
   The battery now reports such a run as **inconclusive** rather than as a cost anomaly, because once a
   re-grant is refused the run has left the plan and its cost says nothing about whether cancelling is
   free.
7. **Latent risk: a `via` that revisits a section releases its first hold.** An authority such as
   `S6 → via S9,S8,S7,S6,S5,X1,X2,S7` contains `S6-S7` twice; when the train passes it the first time the
   holder entry is dropped, so the section reads `holder=None` while the same authority still needs it.
   Verified by trace, but no second train has ever been able to take the section: timing sweeps produced
   no double occupancy, and the battery now checks section co-occupancy after every call in every episode
   (zero occurrences in all ten seeds). Shipping it as a monitored invariant plus this note.
8. **Observation: T1's scoring band is narrower than a single missed stop.** On T1 seed 0 the whole band is
   `C_greedy − C_best = 59.7 − 1.8 ≈ 58`, while one missed call costs 500. Any T1 episode that misses one
   scheduled stop is pinned at 0, so T1 measures only "serve everything, then shave minutes". That is also
   why `patched_plan_and_norm` cannot be expressed on T1: no recordable plan lands between `C_best` and
   `C_greedy`. T2's band (705 → 3070) behaves normally.
## The judges' five runs

```bash
harbor run -y -p datasets/t1-dev-mini -a oracle                                              # expect 1.0
harbor run -y -p datasets/t1-dev-mini -a nop                                                 # expect 0.0
PYTHONPATH=. harbor run -y -p datasets/t1-dev-mini -a railway.harbor.agents:StopEarlyAgent   # expect 0.0
harbor run -y -p datasets/t1-dev-mini -a opencode -m <provider/model> -k 3                   # the gap
```

`harbor_protocol.py` runs all of them, collects every `reward.txt`, and checks each against what it
should be. It also flags two things a human skims past:

- a trial with **no** `reward.txt` is an infrastructure error to rerun, never a 0 (this is why
  `verify.py` writes no reward on infrastructure failure);
- a `reward.json` left anywhere in a trial's verifier directory, because Harbor prefers it over
  `reward.txt`.

On Windows set `PYTHONIOENCODING=utf-8` first; Harbor prints Unicode tables.

## Before the demo: the checklist

1. `make redteam` is green on T1 and T2 dev seeds, and the scoreboard is committed.
2. `python -m railway.redteam.harbor_protocol` passes on the held-out dataset: oracle 1.0 on every trial,
   nop and stop-early 0.0 on every trial, agent below the reference.
3. Both cheat briefs have been run, with at least one frontier model, and every finding is either closed
   or written down with why it is not reachable.
4. The isolation probe has been run (`gen export --probe`, then the oracle on the probe task) and
   `details.json` shows: a non-root agent, no scenario file in the agent's container, `railway` not
   importable there, `/episode_log` refusing both no token and a wrong token, and the public internet
   unreachable.
5. Every seed in the shipped dataset has passed the seed filter, which refuses a scenario where any
   exploit beats greedy.
6. The stop-early curve is in the demo. It is the single clearest answer to "does doing part of the task
   earn part of the score", and it is better to show it than to assert it.

## What we would say if a judge finds something

Separate the three layers out loud. A tool-semantics exploit is a design bug and we fix the cost
function. A scoring-path exploit is an architecture bug and we fix the verifier. A finding that needs the
repository is not reachable by an agent, and the probe output is the evidence. Knowing which one we are
looking at is most of the answer.

## Sources

- [Harbor cheat-vector taxonomy and mitigations (terminal-bench #2086)](https://github.com/harbor-framework/terminal-bench/issues/2086) — V1–V9, the verifier patterns, and the task audit.
- [RewardKit all-pass aggregation bug (harbor #3509)](https://github.com/harbor-framework/harbor/issues/3509) — a stub solution scoring 1.0 because any non-zero dimension counted as a pass. We do not use RewardKit, but the lesson applies to any aggregation: a partial score must not round up to a pass.
- [terminal-bench-science task review automation](https://github.com/harbor-framework/terminal-bench-science/blob/main/TASK_REVIEW_AUTOMATION.md) — `/cheat` trials, the `/fortify` hacker–fixer loop, and the nop gate that requires exactly 0.0.
- [`hack-trial-prompt.md`](https://github.com/harbor-framework/terminal-bench-science/blob/main/.github/hack-trial-prompt.md) and [`cheat-analysis.toml`](https://github.com/harbor-framework/terminal-bench-science/blob/main/rubrics/cheat-analysis.toml) — the brief and the rubric our prompts are modelled on.
- [Terminal Wrench](https://github.com/few-sh/terminal-wrench) — 331 hackable terminal-bench-style tasks and 3,632 exploit trajectories, with the `hack_prompt_v5` elicitation prompt. Useful as a catalogue of what models actually try.
- [Establishing Best Practices for Building Rigorous Agentic Benchmarks](https://arxiv.org/abs/2507.02825) — the ABC checklist; the outcome-validity argument behind grading server-side state rather than self-reports.
- [METR, Recent frontier models are reward hacking](https://metr.org/blog/2025-06-05-recent-reward-hacking/) and [ImpossibleBench](https://arxiv.org/abs/2510.20270) — what frontier models do when the shortcut exists.
