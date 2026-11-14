# README draft: evaluation sections (T09 §2.7 items 2, 3, 4, 7)

> **Draft, not the README.** These sections go into the repo-root `README.md` when T09
> finishes. No numbers here: the result table is pasted from `reports/heldout.html`
> after the real held-out runs. Commands that depend on unmerged tickets are marked with
> the ticket that provides them.

## Quickstart (one command)

```bash
uv sync && make ci
```

`make ci` runs lint and every test that needs no Docker, model key or GPU. Windows users
without `make` run the same steps directly:

```bash
uv sync
uv run ruff check railway tests spike
uv run pytest -m "not harbor and not model and not gpu"
```

## Run the judges' checks

All four use the committed held-out dataset (20 T2 tasks). Each prints one reward per
task.

```bash
# Reference (oracle): every task scores 1.0
harbor run -p datasets/t2-heldout -a oracle

# Do-nothing: every task scores 0.0 (no-op agent name from docs/spike-report.md)
harbor run -p datasets/t2-heldout -a nop

# Stop-early: a sensible start, then end_shift at tick 30; every task scores 0.0 (T07)
PYTHONPATH=. harbor run -p datasets/t2-heldout -a railway.harbor.agents:StopEarlyAgent

# A model through OpenCode
harbor run -p datasets/t2-heldout -a opencode -m <provider/model>
```

The structural holdout (14 stations, never seen in training) works the same way with
`-p datasets/t2s14-heldout`.

## How scoring works

Each task has two fixed anchors computed by the generator: `C_greedy`, the cost of a
simple first-come-first-served dispatcher, and `C_ref`, the cost of the scripted
reference dispatcher. An agent's cost `C_agent` becomes a reward:

```
reward = clip((C_greedy − C_agent) / (C_greedy − C_ref), 0, 1)
```

- greedy scores 0 and the reference scores 1;
- doing nothing, stopping early or anything worse than greedy scores 0;
- across tasks the headline is a ratio of sums, `Σ(C_greedy − C_agent) / Σ(C_greedy − C_ref)`
  (`score_ros`), with a 95% confidence interval from resampling tasks;
- the report also shows the unclipped score (so models worse than greedy keep their
  order), the cost ratio `ΣC_agent / ΣC_ref` and the cost by component;
- a task where a model scores above 1.0 unclipped is flagged and investigated.

## Reproduce our numbers

Every number in the report comes from a run manifest in `runs/` and a results file in
`results/` (one JSON line per episode).

```bash
# 1. Run each manifest (resumable: finished episodes are skipped)        (T08)
uv run railway-eval run runs/heldout-small.yaml
uv run railway-eval run runs/heldout-big.yaml

# 2. Scripted baselines on the same tasks                                 (T09)
uv run railway-eval baselines --dataset datasets/t2-heldout --out results/heldout-baselines.jsonl

# 3. Build the report: one self-contained HTML file, opens offline
make report RUN=heldout
# same as: uv run railway-eval report runs/report-heldout.yaml --out reports/heldout.html
# until T08 wires the subcommand:
uv run python -m railway.eval.report runs/report-heldout.yaml --out reports/heldout.html
```

Model endpoints and keys come from environment variables named in each manifest
(`api_key_env`); keys are never written to results or reports.

Harbor + OpenCode numbers are headline. Our-loop numbers are diagnostic: the loop shows
the state for free.
