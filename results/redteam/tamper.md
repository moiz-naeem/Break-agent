# Verifier tamper suite

Honest reference reward on this task: 1.000. Each case serves a chosen episode log and names the reward that log honestly deserves (the Deserves column). An attack wins if the verifier writes more than that, or leaves any reward.json (Harbor prefers it over reward.txt).

Reachable says whether an agent in a shipped task could do this at all: the episode log is served by the sim sidecar, and tests/ arrives only after the agent finishes, so most cases are defence-in-depth checks on the verifier rather than live holes.

| Case | Vector | Attack | Deserves | Reachable | Exit | reward.txt | reward.json | Verdict |
|---|---|---|---|---|---|---|---|---|
| `pre_written_reward_txt` | V3 | agent writes reward.txt 1.0 before the verifier runs | 0.000 | yes | 0 | 0.000000 | no | blocked |
| `pre_written_reward_json` | V3 | agent writes reward.json, which Harbor prefers | 0.000 | yes | 0 | 0.000000 | no | blocked |
| `sleeper_reward_files` | V5 | a surviving process writes both reward files | 0.000 | yes | 0 | 0.000000 | no | blocked |
| `forged_final_cost` | V7 | log claims cost 0 while its entries replay to the real cost | 0.000 | no | 1 | - | no | blocked |
| `dropped_entries` | V4 | log keeps the honest cost but has no entries | 0.000 | no | 1 | - | no | blocked |
| `truncated_entries` | V4 | half the entries removed, cost left as the reference's | 0.000 | no | 1 | - | no | blocked |
| `foreign_scenario_log` | V8 | a log from another scenario | 0.000 | no | 1 | - | no | blocked |
| `stale_rules_version` | V8 | a log from an older rules version | 0.000 | no | 1 | - | no | blocked |
| `unclosed_log` | V4 | an episode log that was never closed | 0.000 | no | 1 | - | no | blocked |
| `patched_norm` | V1 | tests/norm.json rewritten so a greedy log looks excellent | 0.000 | no | 0 | 0.995108 | no | **WIN** |
| `patched_scenario` | V1 | tests/scenario.json shortened to one tick | 0.000 | no | 1 | - | no | blocked |
| `swapped_token` | V1 | tests/token replaced, so the fetch should be refused | 1.000 | no | 1 | - | no | blocked |
| `empty_log` | V4 | an empty JSON object | 0.000 | no | 1 | - | no | blocked |

## Notes

- `pre_written_reward_txt`: reachable: the agent shares /logs with the verifier, so it can write these
- `pre_written_reward_json`: reachable: the agent shares /logs with the verifier, so it can write these
- `sleeper_reward_files`: reachable: the agent shares /logs with the verifier, so it can write these
- `forged_final_cost`: not reachable: the episode log is served by the sim sidecar, which the agent cannot write
- `dropped_entries`: not reachable: the episode log is served by the sim sidecar, which the agent cannot write
- `truncated_entries`: not reachable: the episode log is served by the sim sidecar, which the agent cannot write
- `foreign_scenario_log`: not reachable: the episode log is served by the sim sidecar, which the agent cannot write
- `stale_rules_version`: not reachable: the episode log is served by the sim sidecar, which the agent cannot write
- `unclosed_log`: not reachable: the episode log is served by the sim sidecar, which the agent cannot write
- `patched_norm`: not reachable: Harbor uploads tests/ only after the agent has finished
- `patched_scenario`: not reachable: Harbor uploads tests/ only after the agent has finished
- `swapped_token`: not reachable: Harbor uploads tests/ only after the agent has finished
- `empty_log`: not reachable: the episode log is served by the sim sidecar, which the agent cannot write

WARN: 1 winning attack(s), 0 of them reachable by an agent.
