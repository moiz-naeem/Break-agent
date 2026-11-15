# net-dev-mini: v0.3 railway network dispatcher tasks

3 Harbor tasks (tier `T1,T2,T3`, seed set `dev`, rules version `0.3-net`). Each task
is one shift of a railway network dispatched through the `railway` MCP server running in its own isolated
`sim` sidecar container. `manifest.json` lists every task with its `scenario_id`, `scenario_hash`, `tier`,
`seed`, and `shift_min`.

## Run Harbor checks

```bash
harbor run -y -p datasets/net-dev-mini -a oracle
harbor run -y -p datasets/net-dev-mini -a nop
PYTHONPATH=. harbor run -y -p datasets/net-dev-mini -a railway.harbor.agents:NetStopEarlyAgent
```

`oracle` is a booked-order dispatcher that plays the whole shift over MCP; `nop` does nothing; the stop-early
agent dispatches like the oracle for 30 minutes and then ends the shift.

A model run adds `-a opencode -m <provider/model>` (and `--allow-environment-host <provider host>` when
using an endpoint not already in `allowed_hosts`).

## Reward definition

`reward.txt` is minus the episode's total cost, `-(C1 + C2 + C3 + C4 + C5)`, verified by deterministically
replaying the logged tool calls against `timetable.json` after the agent finishes. `details.json` records
the cost breakdown by component, operational KPIs, and counts alongside `action_log.json` and `events.jsonl`.
A trial without `reward.txt` is an infrastructure or replay error to rerun, never a valid score.
