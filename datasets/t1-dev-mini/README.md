# t1-dev-mini: railway dispatcher tasks

3 Harbor tasks (preset `T1`, seed set `dev`, rules version `0.3-provisional`). Each task
is one shift of a single-track railway that the agent dispatches through the `railway` MCP server, which
runs in its own `sim` container. `manifest.json` lists every task with its scenario hash and the two
anchors of its score (`c_greedy` scores 0, `c_best` scores 1). `c_best` is the lowest cost known for the
scenario: `c_best_kind` is `proven` where the exact solver proved it optimal, otherwise `best-known`. `c_ref`,
the scripted reference dispatcher's cost, is listed as a baseline.

## Run the judges' checks

From the repository root (`PYTHONPATH` lets Harbor load the custom stop-early agent):

```bash
harbor run -y -p datasets/t1-dev-mini -a oracle                                   # every trial: reward 1.0
harbor run -y -p datasets/t1-dev-mini -a nop                                      # every trial: reward 0.0
PYTHONPATH=. harbor run -y -p datasets/t1-dev-mini -a railway.harbor.agents:StopEarlyAgent   # every trial: reward 0.0
```

A model run adds `-a opencode -m <provider/model>` and, if the provider is not in `allowed_hosts`,
`--allow-environment-host <provider host>`.

## What a reward means

`reward.txt` is `max(0, (c_greedy - c_agent) / (c_greedy - c_best))`, with no top clip (above 1 means the
agent beat the best known plan, flagged as `above_best` in `details.json`), computed by replaying the episode
log against the scenario, which reaches the verifier only after the agent has finished. `details.json`
holds the cost breakdown, the unclipped score and failure counts. A trial without `reward.txt` is an
infrastructure error to rerun (simulator unreachable, corrupt log, replay mismatch, or the agent started
without its MCP tools), never a score of 0.
