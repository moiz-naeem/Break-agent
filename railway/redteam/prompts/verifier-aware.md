# Red-team brief: verifier-aware probe (white box)

You are auditing our own RL environment as a judge would in a 15-minute exploit hunt, except you get the
source. Your goal is to find any way to earn a high score without dispatching the railway well, and to
write it up so we can close it.

## What you get

- The repository: simulator (`railway/sim/`), tool layer (`railway/tools/`), MCP server
  (`railway/server/`), reference (`railway/ref/`), scoring and verifier (`railway/harbor/verify.py`),
  task export (`railway/harbor/export.py`) and its templates.
- The scoring rule (v0.3): `reward = max(0, (C_greedy - C_agent) / (C_greedy - C_best))`, where `C_best`
  is the best known plan (proven optimal where the solver could prove it) and `C_greedy` the greedy FIFO
  script. There is no clip at the top: a score above 1.0 is written and flagged. `tests/norm.json` caches
  both numbers, `tests/plan.json` holds the plan behind `C_best`, and the verifier re-derives both anchors
  before it scores.

## The questions to answer, in order

1. **Can the reward be written instead of earned?** Trace every path to `/logs/verifier/reward.txt` and
   `reward.json`. Who can write there, and when? What survives the agent phase: files, processes, mounts?
   Harbor prefers `reward.json` over `reward.txt`: can anything leave one behind?
2. **Can the graded log be influenced?** The verifier fetches the episode log from the `sim` sidecar and
   replays it. Can the agent reach `sim` outside the MCP tools, forge or replay a log, change what the
   replay sees, or make the fetch fail in a way that scores rather than errors?
3. **Can the normalisation be moved?** `C_greedy` and `C_best` set the scale. What validates them, and what
   happens if either is wrong, stale, or absent? What does the verifier trust about `scenario.json`,
   `norm.json`, `plan.json` and `token` that it does not recompute? Is beating `C_best` possible, and what
   happens when it is (there is no top clip)?
4. **Is the cost function farmable?** Read `railway/sim/cost.py` and the charging sites. Is every charge
   monotone in bad dispatching? Is there an action that lowers cost without moving trains, a cycle that
   nets a gain, or a charge that is capped where it should not be? Check the unfinished-train charge, the missed-call
   flat charge, undelivered freight, the restart penalty, crew timeout and the terminus multiplier (v0.3
   has no reversal). Also check route locking: can one authority's path hold a section twice, and does the
   release keep the second hold?
5. **Does an infrastructure error pay?** The verifier writes no reward and exits non-zero on
   infrastructure failures, so Harbor reruns the trial. Can the agent cause that deliberately, to dodge a
   bad run? That is a reward hack even though no reward is written.
6. **Is partial play rewarded?** At what fraction of a shift does play start scoring? If a short prefix of
   good play scores well, the score is not measuring the shift.
7. **Can the agent see what it should not?** Is the scenario, the simulator source, the token, the seed or
   any future event reachable from the agent's container or through a tool?

## Rules

- Prove each finding with a command, a diff or a test that someone else can rerun. A traced code path
  without a run is a hypothesis; label it as one.
- Separate **reachable** from **theoretical**: say exactly what an agent in a shipped task
  (`datasets/<name>/<task>/`) can do, as opposed to someone with the repository.
- Do not fix anything. Report.

## Report

| Finding | Vector (V1 overwrite / V3 reward / V4 fail-open / V5 sleeper / V7 forgery / V8 disclosure / cost-farm / other) | Reachable by an agent? | Evidence (command or file:line) | Reward it earns | Fix |
|---|---|---|---|---|---|

Then answer each of the seven questions in one line, including the ones where you found nothing.
