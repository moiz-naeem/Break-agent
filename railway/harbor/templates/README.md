# {{NAME}}: railway dispatcher tasks

{{COUNT}} Harbor tasks (preset `{{PRESET}}`, seed set `{{SET}}`, rules version `{{RULES_VERSION}}`). Each task
is one shift of a railway network (single-track lines with passing loops, junctions and terminals) that the
agent dispatches through the `railway` MCP server, which runs in its own `sim` container. `manifest.json`
lists every task with its scenario hash and two reference costs: `c_greedy` (a simple greedy dispatcher) and
`c_best` (the best plan known for the task, with `best_source`).

## Run the judges' checks

From the repository root (`PYTHONPATH` lets Harbor load the custom stop-early agent):

```bash
harbor run -y -p {{DATASET}} -a oracle                                   # every trial: reward = -c_best
harbor run -y -p {{DATASET}} -a nop                                      # every trial: reward far below -c_greedy
PYTHONPATH=. harbor run -y -p {{DATASET}} -a railway.harbor.agents:StopEarlyAgent   # far below -c_greedy
```

A model run adds `-a opencode -m <provider/model>` and, if the provider is not in `allowed_hosts`,
`--allow-environment-host <provider host>`.

## What a reward means

`reward.txt` is minus the episode's total cost, `-(C1 + C2 + C3 + C4 + C5)`, computed by replaying the episode
log against the scenario, which reaches the verifier only after the agent has finished. The verifier also
re-derives `c_greedy` (by running greedy) and `c_best` (by replaying the stored best plan) and refuses a task
whose `norm.json` disagrees. `details.json` holds the cost breakdown, the KPIs, failure counts and, for
reading results only, where the cost lands between the two reference costs. A trial without `reward.txt` is
an infrastructure error to rerun (simulator unreachable, corrupt log, replay mismatch, or the agent started
without its MCP tools), never a score.
