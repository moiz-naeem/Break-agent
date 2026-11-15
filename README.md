# Railway dispatcher benchmark

A long-horizon planning benchmark: an agent dispatches trains on a single-track railway for one shift
through the tools of the `railway` MCP server. The score is anchored on a greedy dispatcher (0) and a
scripted reference dispatcher (1). Design: `DESIGN.md`; contracts: `docs/PRD.md`; work plan: `docs/tickets/`.

> Draft (T07). T09 finishes this README with results and the held-out dataset. 

## Run the judges' checks

Needs Docker and the Harbor CLI. From the repository root (`PYTHONPATH` lets Harbor load the custom
stop-early agent):

```bash
harbor run -y -p datasets/t1-dev-mini -a oracle                                              # every trial: reward 1.0
harbor run -y -p datasets/t1-dev-mini -a nop                                                 # every trial: reward 0.0
PYTHONPATH=. harbor run -y -p datasets/t1-dev-mini -a railway.harbor.agents:StopEarlyAgent   # every trial: reward 0.0
```

On Windows, set `PYTHONIOENCODING=utf-8` first: Harbor prints Unicode tables and fails on a cp1252 console.
A trial without `verifier/reward.txt` is an infrastructure error to rerun, never a score of 0.

## Development

```bash
uv sync
make ci        # lint, unit tests, then the oracle on datasets/t1-dev-mini (needs Docker + Harbor)
```

Accepted seed lists (`seeds/*.json`) can be regenerated locally with `uv run gen accept --config T1 --set dev
--count 20`, or on Modal (one container per seed, a few minutes) with
`uv run --with modal modal run scripts/modal_accept.py --preset T1 --set-name dev --count 20`.
