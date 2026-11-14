# {{NAME}}: v0.3 railway network dispatcher tasks

{{COUNT}} Harbor tasks (tier `{{TIER}}`, seed set `{{SET}}`, rules version `{{RULES_VERSION}}`). Each task
is one shift of a railway network dispatched through the `railway` MCP server running in its own isolated
`sim` sidecar container. `manifest.json` lists every task with its `scenario_id`, `scenario_hash`, `tier`,
`seed`, and `shift_min`.

## Run Harbor checks

```bash
harbor run -y -p {{DATASET}} -a oracle
harbor run -y -p {{DATASET}} -a nop
```

A model run adds `-a opencode -m <provider/model>` (and `--allow-environment-host <provider host>` when
using an endpoint not already in `allowed_hosts`).

## Reward definition

`reward.txt` is minus the episode's total cost, `-(C1 + C2 + C3 + C4 + C5)`, verified by deterministically
replaying the logged tool calls against `timetable.json` after the agent finishes. `details.json` records
the cost breakdown by component, operational KPIs, and counts alongside `action_log.json` and `events.jsonl`.
A trial without `reward.txt` is an infrastructure or replay error to rerun, never a valid score.
