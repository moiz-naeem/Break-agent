# T09: Pilot gate, weight freeze, held-out evaluation, report, diagnostics, README

| | |
|---|---|
| Depends on | T08 (runner, stats); T07 (export, Harbor runs) |
| Unblocks | T10 (frozen rules, held-out baseline numbers for the "before" side) |
| DESIGN.md | §6.1 (weights frozen after the pilot), §6.2 (score, comparison table), §9.2 (seed sets), §9.5 (models), §9.6 (report), §9.7 (diagnostics), §9.8 (pilot gate), §11 (hours 28–48), §12 |
| PRD | §3 G4, G5, G9; §8 D16, D20; §9 (models, provider host) |
| Required markers | default + `model` + `harbor` |
| **Blocking input** | The user's small / mid / big models, endpoints and keys (PRD §9). Without them this ticket can't pass; build the non-model parts first |
| Suggested split | A: pilot + freeze + held-out export. B: evaluation runs + report. C: diagnostics (`conflicts`, `what_if`, regret, ablation presets) + README + clean-machine check |

## 1. Goal

Prove the gap, freeze the rules, and produce what the judges and the README show:

1. **Pilot gate** (DESIGN.md §9.8) on 10 accepted T1 dev seeds. All three must hold:
   - (a) big − small ≥ 0.3 (unclipped `score_ros`);
   - (b) big ≤ 0.8;
   - (c) the reference ≥ 30% cheaper than greedy.
2. **Set and freeze** the provisional numbers (tool budget, agent timeout, cost weights):
   - `RULES_VERSION = "1.0"`;
   - a test pins every constant.
3. **Export the held-out datasets:**
   - T2 (20 tasks);
   - T2-S14 structural holdout (10 tasks);
   - re-export the CI mini dataset.
4. **Held-out evaluation:**
   - every model × 20 seeds × 3 reps via Harbor + OpenCode;
   - scripted baselines on the same tasks;
   - our-loop runs as diagnostics.
5. **One self-contained HTML report:** a table + 3 charts.
6. **Diagnostics** (DESIGN.md §9.7, cut in the order given in §2.6).
7. **README** with one-command evaluation, plus a clean-machine check.

## 2. Context the implementer needs

### 2.1 Pilot (`railway/eval/pilot.py`, `railway-eval pilot`)

- **Dataset:** export `datasets/t1-dev-pilot/` (the first 10 accepted T1 dev seeds,
  uncommitted).
- **Runs:**
  - `runs/pilot-small.yaml` and `runs/pilot-big.yaml`;
  - optionally `runs/pilot-mid.yaml`;
  - harness `harbor`, reps 1. These match what the judges see.
  - Also run `runs/pilot-small-loop.yaml` and `runs/pilot-big-loop.yaml` (loop,
    windowed, reps 1) as a cheaper cross-check.
- **Gate:** `evaluate_gate(small_rows, big_rows) -> GateResult`:
  - a = `score_ros(big) − score_ros(small) ≥ 0.3`;
  - b = `score_ros(big) ≤ 0.8`;
  - c = `Σ c_ref ≤ 0.7 × Σ c_greedy` over the 10 seeds.
- **If it fails** (DESIGN.md §9.8). At most 3 iterations, each recorded:
  - **b fails** (the big model is too good): enable more factors in the pilot preset —
    `T1+` = T1 with F5, F7 and F8, then F6 and F10 — or raise `event_density`.
    Re-accept the seeds and re-run.
  - **a fails:**
    - first check `invalid_calls` and `format_errors` for the small model. If they
      average > 10% of its tool calls, the gap is a tool-usability artifact: fix the
      manual or tool descriptions, not the difficulty;
    - otherwise raise difficulty as for (b).
  - **c fails** (rare: the seed filter enforces it per seed): investigate the reference.
  - **After 3 failed iterations, stop** and escalate to the user with the numbers. It
    is a product decision (change the preset for the judges, or accept a smaller gap).
- **Output:** `docs/pilot.md` with:
  - the models (names only, no keys);
  - the date;
  - a per-seed table (`c_greedy`, `c_ref`, small, big);
  - the three conditions with numbers;
  - every iteration and what changed;
  - a final line `Verdict: PASS` or `Verdict: FAIL`.

### 2.2 Setting the provisional numbers, then freezing

Run `runs/calib-big-t2.yaml`: big model, Harbor, 3 T2 dev tasks, reps 1. Then:

| Number | Rule |
|---|---|
| `TOOL_BUDGET` | `max(400, round_up_50(1.5 × max(p90 big tool calls on T2, reference max on T2 dev)))` |
| Agent timeout (`task.toml`) | `max(1800, round_up_600(2 × p90 big wall seconds on T2))` |
| Cost weights | Unchanged **unless** one component is > 80% of `c_greedy` on average over the T2 dev accepted seeds. Then lower that component's weight, re-run the filter, and record it |

**Freeze:**

- set `RULES_VERSION = "1.0"`;
- write `docs/frozen-constants.json` (canonical JSON of every public constant);
- add the pin test (T09-03).

After the freeze, **never change a weight in response to held-out results** (DESIGN.md
§6.1).

**Re-export after the freeze.** `rules_version` changed, so caches invalidate:

| Dataset | Command | Committed? |
|---|---|---|
| `datasets/t2-heldout/` | `gen export --config T2 --set heldout --count 20` | yes |
| `datasets/t2s14-heldout/` | `gen export --config T2-S14 --set heldout --count 10` | yes |
| `datasets/t1-dev-mini/` | `gen export --config T1 --set dev --count 3` (CI) | yes |
| `seeds/*.json` | regenerated | yes |

`datasets/*/manifest.json` and the per-task files are what the user commits.

### 2.3 Held-out evaluation

| Manifest | What |
|---|---|
| `runs/heldout-<model>.yaml` | Harbor, `datasets/t2-heldout`, reps 3, for each of small, mid (if available), big |
| `runs/heldout-<model>-loop.yaml` | Loop, windowed, same dataset, reps 3 (diagnostic) |
| `runs/heldout-<model>-s14.yaml` | Harbor, `datasets/t2s14-heldout`, reps 3 (structural holdout; at least small and big) |
| `runs/heldout-big-nostringline.yaml` | Loop, `tools_disabled: [stringline]` (ablation 7) |

**Baselines:**

```
railway-eval baselines --dataset datasets/t2-heldout --out results/heldout-baselines.jsonl
```

- runs greedy, reference, do-nothing and stop-early in-process;
- rows with `harness: "script"`, `model: "greedy"` etc.

### 2.4 Report (`railway/eval/report.py`)

```
railway-eval report runs/report-<name>.yaml --out reports/<name>.html
```

(`make report RUN=<name>` is an alias.) The report manifest lists the result files and
their display order.

**A single HTML file**, self-contained:

- inline CSS;
- charts as inline `<svg>` produced by matplotlib (`format="svg"`,
  `metadata={"Date": None}`, `svg.hashsalt` fixed, so the output is byte-deterministic);
- no external URLs;
- no `<script src>`.

**Contents:**

1. **Header:** rules version, dataset, date of the results files (not of rendering),
   and the harness legend. "Harbor + OpenCode numbers are headline. Our-loop numbers are
   diagnostic: the loop shows the state for free."
2. **Table** (one row per model and harness, then the baselines). Columns:
   - `score_ros` ± 95% CI;
   - `reward_mean`;
   - `unclipped_mean`;
   - `cost_ratio`;
   - gridlocks/episode;
   - crew timeouts/episode;
   - invalid calls/episode;
   - format errors/episode (loop only);
   - mean tokens;
   - episodes / errors;
   - flagged seeds.
3. **Chart 1:** score by model (bars with CI whiskers; reference = 1 and greedy = 0 as
   lines).
4. **Chart 2:** score by factor (ablation), with small and big side by side. If the data
   is missing, show the placeholder text `ablation not run`.
5. **Chart 3:** before vs after training. A placeholder `pending T10` until T10 fills
   it.
6. **Failure breakdown table:** mean cost by component per model.

### 2.5 Diagnostic tools (official runs refuse them)

Both are registered only when the `Session` has `diagnostics=True`. Otherwise they
return `REFUSED R-DIAGNOSTIC-ONLY`, and the MCP server never lists them.

- **`conflicts()`** — JSON list of projected conflicts within 90 ticks:
  - opposing pairs that will need the same section, and same-direction catch-ups;
  - computed from a belief-simulator projection of the current authorities plus
    free-run schedules;
  - each entry: `{trains, section, earliest_tick}`.
- **`what_if(actions: list[{tool, args}])`** — `{projected_cost}`:
  - `Reference.evaluate` applied to the action sequence on a clone;
  - **never mutates** the session;
  - counts 1 budget unit.

### 2.6 Diagnostics (DESIGN.md §9.7), in cut order

If behind schedule, drop from the bottom of this list:

1. **Failure breakdown.** From the held-out rows; already in the report.
2. **Factor ablation.**
   - Presets `T2-only-F5`, `-F6`, `-F7`, `-F8`, `-F10`, `-F15`, `-F16`: T2 with F1–F4
     plus exactly that one factor.
   - 5 accepted dev seeds each; small and big; loop windowed, reps 1.
   - Plot the gap per factor (chart 2).
3. **Same-state regret** (`railway/eval/diagnostics.py`):
   - **States:** 100 decision-point states sampled deterministically from reference
     trajectories on T2 dev accepted seeds (every k-th decision point).
   - **Ask each model:** for each state, the windowed prompt as of that state (history
     = the reference's last 5 turns). Allow up to 3 read-only calls (answered by
     `peek`). The first non-read-only call is "the action".
   - **Score:** `regret = Reference.evaluate(action) − min(Reference.evaluate(c) for c
     in candidates ∪ {action})`. The candidates come from `candidates(B, train)` for
     every actionable train, plus `advance`.
   - **Report** mean regret per model. Lower is better; 0 = matches the best candidate.
4. **Score vs episode length.** T1, T2, T3; 5 accepted dev seeds each; small and big;
   loop windowed.
5. **Projection vs choice:** the regret run again with `conflicts()` enabled; the
   change in regret.
6. **What-if ablation:** the big model with `what_if` enabled on 5 T2 dev seeds (an
   upper bound on how much forward simulation helps).
7. **`stringline` ablation:** `runs/heldout-big-nostringline.yaml` vs
   `heldout-big-loop`.
8. **Human baseline:** backlog (PRD §10).

### 2.7 README (`README.md`)

Sections:

1. **What this is** — 5 lines.
2. **Quickstart (one command):**
   ```
   uv sync && make ci
   ```
3. **Run the judges' checks** — exact commands:
   - `harbor run -p datasets/t2-heldout -a oracle`;
   - the no-op agent;
   - `StopEarlyAgent`;
   - OpenCode with `-m <provider/model>`.
4. **How scoring works** — the greedy anchor, clipping, ratio-of-sums.
5. **Why it's hard** — the factors, delayed effects, and the result table pasted from
   the report.
6. **Exploit analysis** — DESIGN.md §6.5 table plus a pointer to `tests/t06`, `tests/t07`.
7. **Reproduce our numbers** — the manifests, `railway-eval run`, `make report`.
8. **Repo map.**
9. **Limits and honest caveats** — the strength-check gap, the fallback if used, the
   diagnostic-loop caveat.

**No personal email or names beyond what the user approves.**

**Clean-machine check:** a fresh clone in a clean container or VM, then `uv sync`, then
`make ci` passes. Record the exact commands and output in the gate record.

## 3. Scope

**In scope:**

- `eval/pilot.py`, `report.py`, `diagnostics.py`;
- the `baselines` and `report` subcommands;
- the diagnostic tools;
- the ablation presets;
- the freeze, the re-exports, the runs, the report, the README.

**Out of scope:**

- Training (T10).
- The human-baseline CLI (backlog).
- Stretch factors (backlog).

## 4. Implementation notes and pitfalls

- **Order matters:**
  1. pilot;
  2. budget/timeout calibration;
  3. freeze;
  4. re-export;
  5. held-out runs.

  Any held-out run before the freeze is invalid. The verifier refuses a stale
  `rules_version` anyway.
- **Cost control.** The held-out Harbor runs are 20 × 3 × (number of models) episodes.
  Run the big model last, after the small model's run has validated the pipeline.
- **Never tune on held-out.** If a held-out result looks wrong, investigate on dev seeds.
- **Flagged seeds** (unclipped > 1.0): list them in the report and investigate each in
  the gate record. The design says either the reference erred or the gap claim is
  wrong.

## 5. Tests (write these first)

### 5.1 `tests/t09/test_pilot.py`

| Id | Test | Assertion |
|---|---|---|
| T09-01 | `test_T09_01_gate_logic` (parametrized) | Fixture rows: small 0.2 / big 0.6 / ratio 0.6 → PASS; big 0.85 → b fails; small 0.4 / big 0.6 → a fails; Σc_ref/Σc_greedy 0.75 → c fails; `GateResult` names the failing conditions |
| T09-02 | `test_T09_02_pilot_record` | `docs/pilot.md` exists; contains `Verdict: PASS`; its three numbers equal `evaluate_gate` recomputed from `results/pilot-small.jsonl` and `results/pilot-big.jsonl`; it lists every iteration |

### 5.2 `tests/t09/test_freeze.py`

| Id | Test | Assertion |
|---|---|---|
| T09-03 | `test_T09_03_constants_frozen` | `RULES_VERSION == "1.0"`; the canonical dump of `railway.sim.constants` public names == `docs/frozen-constants.json` |
| T09-04 | `test_T09_04_heldout_dataset` | `datasets/t2-heldout/` has 20 tasks; the manifest `rules_version == "1.0"`; every seed in the held-out range; regenerating each seed gives the same `scenario_hash`; `norm.json` == the accepted list; no seed appears in `seeds/*-train.json` or `*-dev.json` |
| T09-05 | `test_T09_05_s14_dataset` | `datasets/t2s14-heldout/` has 10 tasks, each with 14 stations; no `scenario_hash` in common with `t2-heldout` |
| T09-06 | `test_T09_06_ci_dataset_current` | `datasets/t1-dev-mini/` manifest `rules_version == "1.0"` |

### 5.3 `tests/t09/test_report.py` (fixture result files, no models)

| Id | Test | Assertion |
|---|---|---|
| T09-07 | `test_T09_07_self_contained` | Generated HTML has no `http://` or `https://` in `src=` or `href=`; no `<script src`; exactly 3 chart containers; each holds an inline `<svg` or a placeholder text |
| T09-08 | `test_T09_08_table_numbers` | Every numeric cell in the table equals `stats.py` recomputed from the fixture rows (parse the HTML table) |
| T09-09 | `test_T09_09_deterministic` | Two renders of the same input → identical bytes |
| T09-10 | `test_T09_10_placeholders` | Without ablation or training data → the texts `ablation not run` and `pending T10` are present |
| T09-11 | `test_T09_11_legend` | Contains the headline/diagnostic legend sentence of §2.4 |

### 5.4 `tests/t09/test_diagnostics.py`

| Id | Test | Assertion |
|---|---|---|
| T09-12 | `test_T09_12_diagnostic_tools_refused_officially` | Official session: `conflicts` and `what_if` → `REFUSED R-DIAGNOSTIC-ONLY`; MCP `list_tools` (official server) excludes both |
| T09-13 | `test_T09_13_conflicts` | mini3 at t=0 with `diagnostics=True`: `conflicts()` includes an entry with trains `["E1","L5"]` (opposing, both projected to need S1–S3 sections) |
| T09-14 | `test_T09_14_what_if_pure` | `what_if([authorize E1 S2 L1])` returns `projected_cost == Reference.evaluate(...)`; `state()` before == after; budget +1 |
| T09-15 | `test_T09_15_regret` | On 5 sampled states: regret ≥ 0 always; the reference's own first action has regret 0 (it was the argmin); results identical across 2 runs |
| T09-16 | `test_T09_16_ablation_presets` | `T2-only-F5`…`T2-only-F16` exist; generated scenarios have `factors == F1..F4 + [that one]` |

### 5.5 `tests/t09/test_results_present.py` (outcome checks; need the real runs)

| Id | Test | Assertion |
|---|---|---|
| T09-17 | `test_T09_17_heldout_results` | For each model in `runs/heldout-*.yaml` (Harbor): `results/<run>.jsonl` has 60 rows; ok ≥ 90%; `rules_version == "1.0"` on every row |
| T09-18 | `test_T09_18_baselines` | `results/heldout-baselines.jsonl`: for all 20 tasks, greedy `score_unclipped == 0`, reference `== 1`, do-nothing and stop-early `score_clipped == 0` |
| T09-19 | `test_T09_19_report_built` | `reports/heldout.html` exists and was built from the current results files (the report embeds their sha256 in a comment; the test recomputes and compares) |
| T09-20 | `test_T09_20_readme` | `README.md` has the 9 sections of §2.7; contains `uv sync && make ci`, `harbor run -p datasets/t2-heldout -a oracle`, `railway-eval run`, `make report`; contains no `@` email address |

### 5.6 Harbor runs (marker `harbor`)

| Id | Test | Assertion |
|---|---|---|
| T09-H1 | `test_T09_H1_oracle_heldout` | Oracle on `datasets/t2-heldout` → 20/20 rewards == 1.0 |
| T09-H2 | `test_T09_H2_nop_heldout` | No-op → 20/20 == 0.0 |
| T09-H3 | `test_T09_H3_stop_early_heldout` | `StopEarlyAgent` → 20/20 == 0.0 (skip with reason if F-CUSTOM failed) |
| T09-H4 | `test_T09_H4_oracle_s14` | Oracle on `datasets/t2s14-heldout` → 10/10 == 1.0 |

## 6. Gate commands

```bash
uv run pytest tests/t01 ... tests/t09 -m "not harbor and not model and not gpu"
```
```bash
uv run pytest tests/t09 -m "harbor"
```
```bash
make ci
```

The `model`-dependent work is the runs themselves; T09-17 checks their outputs.

## 7. Definition of done

- [ ] Pilot `Verdict: PASS` in `docs/pilot.md`, or the ticket is **blocked** and
      escalated to the user after 3 failed iterations. A blocked ticket does not pass
      the gate.
- [ ] Frozen at `RULES_VERSION = "1.0"`; held-out, S14 and CI datasets re-exported.
- [ ] Held-out Harbor runs are done for every available model; baselines are done; the
      report is built.
- [ ] Diagnostics 1–2 are done, plus as many of 3–7 as time allows. The gate record
      lists which were cut.
- [ ] README is complete; the clean-machine check passed (recorded).
- [ ] Gate record, reviewers, handover (no commit). The handover lists the dataset and
      results files the user should commit.
