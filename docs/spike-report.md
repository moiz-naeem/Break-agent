# T01 Harbor spike report

Harbor version: 0.26.0 (PyPI `harbor`, installed with `uv tool install harbor==0.26.0`; git sha: not
available from a PyPI release; the version pin is the reproducible reference)
OpenCode version: 1.19.5 (`opencode-ai`, pre-installed in the main image)
MCP Python SDK: 2.3.0 (server: `mcp.server.mcpserver.MCPServer`; client: `mcp.Client`)
Host: Windows 10 Pro 10.0.19045, Docker Desktop 29.7.2 (Linux engine), runs launched from Git Bash
Models used: `anthropic/claude-haiku-4-5` (small) and `anthropic/claude-opus-5-5` (big); key from `.env`
(`CLAUDE_API_KEY`) via `--env-file .env --api-key-name CLAUDE_API_KEY`
Date: 2026-11-14

Decision: SIDECAR — the simulator runs in its own `sim` compose service; every fact the design depends on passed.

## Facts

| Fact | Result | Command | Evidence |
|---|---|---|---|
| F-REACH | PASS | `harbor run -p spike/toy-task -a oracle` | Reward 1.0: the verifier (main container, after the agent) fetched `/episode_log` from `http://sim:8000` with the bearer token and found the oracle's `ping` |
| F-SECRET | PASS | same run | `details.json` has `secret_visible: false`: `/srv/secret.txt` (only in the sim image) does not exist in main |
| F-NOP | PASS | `harbor run -p spike/toy-task -a nop` | Built-in agent is named `nop`; reward 0.0, no exception |
| F-CUSTOM | PASS | `harbor run -p spike/toy-task -a railway.harbor.spike_agents:NoOpAgent` (with `PYTHONPATH=<repo>`) | Reward 0.0, no exception. Harbor is a separate uv tool, so our package is found through `PYTHONPATH`. A custom agent must declare `capabilities = AgentCapabilities(mcp_servers=True)` or Harbor refuses it on tasks with MCP servers |
| F-SMALL | BLOCKED | `harbor run -p spike/toy-task -a opencode -m anthropic/claude-haiku-4-5 --env-file .env --api-key-name CLAUDE_API_KEY` | **As defined (an 8B-or-smaller model behind an OpenAI-compatible endpoint): not tested**, because no open-weight model is chosen yet. What was tested passed: the chosen small model, Claude Haiku 4.5 over the Anthropic API, got reward 1.0, and so did Opus 5.5. The OpenAI-compatible question returns with the T10 open-weight decision (see "Consequences") |
| F-NONROOT | PASS | `spike/toy-task-nonroot` (`[agent] user = "agent"`), OpenCode + Haiku 4.5 | Reward 1.0; the `ping` text (the agent's `id -u`) was `1000`, not `0` |
| F-ALLOW | PASS | `spike/toy-task-allowlist` (`network_mode = "allowlist"`, `allowed_hosts = ["sim"]`) + `--allow-environment-host api.anthropic.com` | Reward 1.0 on 2 of 3 runs once OpenCode was pre-installed. The one failure: OpenCode logged `server unavailable key=toy` at startup and ran without MCP tools (reward 0.0, not an error). A bash `curl` from the agent phase reached `sim` in a separate probe, so the network path works; the failure looks like a startup race. Risk recorded for T07 below |
| F-LAYOUT | PASS | all runs | See "Exact shapes" |

## Exact shapes (use these in T07)

**Trial output.** `<jobs-dir>/<job>/<trial>/` contains `config.json`, `result.json` (with `exception_info`
and `agent_info.version`), `verifier/reward.txt`, `verifier/details.json`, `verifier/test-stdout.txt`,
`agent/…` (OpenCode logs under `agent/opencode/xdg-data/opencode/log/`). The job dir itself also has a
`result.json`; trial dirs are exactly `<jobs-dir>/*/*/result.json`.

**Inside the container.** Tests are uploaded to `/tests`, the solution to `/solution`, logs go to `/logs/{agent,verifier}`.

**`task.toml`** (validated by Harbor 0.26.0):
```toml
schema_version = "1.4"
[agent]
timeout_sec = 600
user = "agent"                 # optional; needs `useradd -m -u 1000 agent` in the main Dockerfile
[verifier]
timeout_sec = 120
[environment]
build_timeout_sec = 900
network_mode = "allowlist"     # optional
allowed_hosts = ["sim"]
[[environment.mcp_servers]]
name = "toy"
transport = "streamable-http"
url = "http://sim:8000/mcp"
```

**`environment/docker-compose.yaml`**: build contexts resolve relative to the task's own compose file. With
the allowlist on, Harbor moves every task service that has no explicit network settings into its
egress-control network namespace, so `sim` declares `networks: [default]`:
```yaml
services:
  sim:
    build: {context: ./sim}
    networks: [default]
    healthcheck: {test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"], interval: 2s, timeout: 5s, retries: 30}
  main:
    depends_on: {sim: {condition: service_healthy}}
```

**Main image.** Pre-install `nodejs`, `npm` and `opencode-ai@1.19.5`. Harbor's OpenCode setup skips
installing when an OpenCode binary is present; without it, setup runs `apt-get install nodejs npm` and
`npm i -g opencode-ai`, which the allowlist blocks (the first allowlist run failed with
`NonZeroAgentExitCodeError` on `apt-get`).

**MCP server (SDK 2.x).** Run with
`TransportSecuritySettings(enable_dns_rebinding_protection=False)`: requests arrive with `Host: sim:8000`.

**CLI.** `harbor run -y -p <task|dataset> -a <agent> -m <provider/model> -o <jobs-dir>`, plus `-k` for
attempts per trial. `--allow-environment-host` and `--allow-agent-host` add allowlist hosts at run time.

## Minimal allowlist

| Host | Why |
|---|---|
| `sim` | The MCP sidecar (in `allowed_hosts`) |
| The model provider host | `api.anthropic.com` for our runs, added with `--allow-environment-host` (the flag verified here). Harbor also has `--allow-agent-host` (agent phase only), which judges could use for their provider; it was not tested |

Nothing else is needed once OpenCode is pre-installed. OpenCode's attempt to fetch
`https://models.opencode.ai/api.json` is blocked and harmless; it logs an error and continues.

## Consequences for later tickets

1. **T07 must pre-install OpenCode 1.19.5** in the main image and add `networks: [default]` to `sim`.
2. **T07 risk: intermittent MCP startup failure under the allowlist** (1 of 3 runs). A missing MCP
   connection gives reward 0 with no trial error, which is a silent unfair 0 for an agent. Candidate fixes:
   - re-test with more runs in T07 to measure the rate;
   - an `[agent]` phase policy identical to the baseline, so Harbor doesn't switch policy just before the
     agent starts;
   - a verifier rule that marks the trial an infrastructure error when the episode log has zero tool calls
     **and** the agent log shows `server unavailable`.

   Decide in T07 with data.
3. **T10 needs an open-weight training target.** Claude Haiku 4.5 is API-only, so it can be the "small"
   model for the gap measurement but cannot be LoRA-trained. T10 needs either a separate open-weight
   model (7–8B) as the trained small model, or a different tiebreaker plan. This is a user decision.
4. **OpenCode's internal "small model".** It used `claude-haiku-5-5` for session titles when running
   Haiku 4.5. This only affects titles, not tool calls, but the key must allow that model.
5. **Custom Harbor agents** (stop-early baseline in T07) need `AgentCapabilities(mcp_servers=True)` and
   `PYTHONPATH` pointing at the repo.
