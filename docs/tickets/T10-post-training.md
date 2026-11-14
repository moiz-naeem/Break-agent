# T10: Post-training (BC → DAgger → GRPO) and before/after results

| | |
|---|---|
| Depends on | T09 (frozen rules, held-out datasets, the "before" numbers for the small model, the report) |
| Unblocks | Nothing (last ticket); feeds the tiebreaker and the README |
| DESIGN.md | §10 (post-training plan), §7 (the reference as teacher), §9.2 (train seeds only), §9.4 (windowed format), §6.2 (training on raw cost) |
| PRD | §3 G8; §9 (models, GPU plan) |
| Required markers | default + `gpu` + `model` |
| **Blocking input** | The small model choice (7–8B open-weight with a tool-calling chat template), its serving stack (vLLM or another OpenAI-compatible server with LoRA support) and GPU access (PRD §9) |
| Suggested split | A: data builders (BC, DAgger labels) + leakage tests. B: SFT/LoRA trainer + serving the adapter + DAgger loop. C: GRPO reward function + trainer, evaluation and the before/after report |

## 1. Goal

Show that the small model improves from training on this environment, measured on
held-out T2 seeds through the same Harbor + OpenCode path the judges use. That is the
hackathon tiebreaker.

1. **Behaviour cloning (BC).** Reference actions on train-seed states, in the exact
   windowed prompt format of T08.
2. **DAgger** (2 rounds). Run the small model, label every visited state with the
   reference action, aggregate, retrain.
3. **GRPO** (if time allows; DESIGN.md §11 allows cutting it). Per-state groups of
   sampled actions; reward = −(reference-rollout cost after the action), raw cost.
4. **Before/after results:**
   - held-out T2 (Harbor headline, plus loop);
   - the structural holdout T2-S14;
   - failure breakdown before/after;
   - chart 3 of the report.

## 2. Context the implementer needs

### 2.1 Data sources and leakage rules

- **Seeds:** only `seeds/T1-train.json` and `seeds/T2-train.json`. Produce them with
  `gen accept --config T2 --set train --count 300` (T1 for a curriculum: 200).
- **Never** use dev, held-out or any `T2-S14` scenario.
- **Prompts:** the T08 `prompts.build(...)` in **windowed** mode, the exact function the
  evaluation uses. Prompt drift between training and evaluation silently ruins results.
- **Labels:** `Reference.decide(static, state)` (T06). The **first** call of the
  returned list is the label at that state. The reference never needs read-only tools:
  the windowed CURRENT block already holds the state.

### 2.2 BC data builder (`railway/train/data.py`)

`build_bc(seeds, preset) -> Iterator[Example]`:

1. Play each scenario with the fake `policy(Reference.decide)` model through the T08
   loop. Record every turn as an example:
   ```
   {messages: <prompt at that turn>, tools: tool_schemas(),
    target: {name, arguments}, meta: {scenario_hash, seed, preset, tick, turn}}
   ```
2. Write `data/bc/{train,val}.jsonl`. **The split is by seed, 95/5.**
3. **Size target:** about 300 T2 seeds × about 100 turns ≈ 30k examples. T1 examples
   are optional curriculum.

### 2.3 DAgger (`railway/train/dagger.py`)

For each round r (2 rounds):

1. Serve the current adapter (vLLM `--enable-lora`, OpenAI-compatible).
2. Run the small model with the loop on 100 train seeds.
3. At every turn, record the prompt the model saw, plus the label
   `Reference.decide(static, state)[0]` computed via `Session.peek`. The model's own
   action is still executed: that is DAgger.
4. Aggregate: BC ∪ all DAgger rounds. Dedupe on `(sha256(canonical prompt), target)`.
5. Retrain from the **base** model on the aggregate (the standard DAgger choice; record
   it).

### 2.4 SFT / LoRA (`railway/train/bc.py`; `railway-train sft`)

| Setting | Value |
|---|---|
| Stack | `transformers`, `peft`, `trl` (`SFTTrainer`), pinned in an optional extra `[train]` (`uv sync --extra train`) |
| LoRA | r = 16, alpha = 32, dropout 0.05, all linear projection modules |
| Optimizer | lr 1e-4, cosine, warmup 3%, 2 epochs, bf16 |
| Max sequence | Covers the 99th percentile prompt length of the data (measured and logged; expect about 8–12k tokens) |
| Loss | **Assistant tokens only**: the tool-call message. Prompt tokens are masked with `-100` |
| Formatting | `tokenizer.apply_chat_template(messages + [assistant tool_call], tools=tools)`. The target's assistant message uses the model's native tool-call format, which the template produces |
| Output | `artifacts/<run>/adapter/` + `train_log.jsonl` (loss per step) + `config.json` |

The exact numbers above are defaults. The gate record must state the values actually
used and why they changed (e.g. memory limits).

### 2.5 GRPO (`railway/train/grpo.py`; `railway-train grpo`)

- **States:** 2,000 sampled from the DAgger data (train seeds only).
- **Group:** G = 8 completions per state at temperature 1.0.
- **Reward:**
  ```
  r = −Reference.evaluate(static, state, parsed_call)
  ```
  This is raw cost (DESIGN.md §10.3). The group baseline (mean subtraction) handles
  scale.
- **Unparseable completions or no tool call:**
  ```
  r = −(Reference.evaluate(static, state, no-op) + FORMAT_PENALTY)
  ```
  with `FORMAT_PENALTY = 50`. This is a PRD decision: DESIGN.md doesn't specify it, so
  record it.
- **Parallelism:** `Reference.evaluate` is CPU-bound. Evaluate with a process pool; the
  results must equal serial evaluation.
- **Stack:** TRL `GRPOTrainer` with a custom reward function. The KL coefficient is the
  TRL default. 1 epoch over the states.

### 2.6 Evaluation (`railway/train/evaluate.py`; `railway-train eval`)

- **Manifests:**
  - `runs/heldout-small-<stage>.yaml` (Harbor: the adapter served as an
    OpenAI-compatible model reachable from Harbor; loop: same endpoint);
  - `runs/heldout-small-<stage>-s14.yaml`.
  - Stages: `base` (from T09), `bc`, `dagger2`, `grpo`.
- **Comparison:** `railway-eval compare results/heldout-small.jsonl
  results/heldout-small-<best>.jsonl` gives the paired difference of `score_ros` with a
  seed-resampled 95% CI.
- **Report:**
  - writes `reports/data/training.json` for chart 3 (score by stage, with CI);
  - rebuilds `reports/heldout.html`;
  - the "pending T10" placeholder disappears.
- **`docs/training.md`:**
  - models and adapters (names only);
  - data sizes;
  - GPU hours;
  - loss curves (SVG inline);
  - a before/after table for held-out T2 (Harbor, loop) and S14;
  - failure breakdown before/after;
  - DAgger round deltas;
  - the line `G8: PASS` or `G8: FAIL`. PASS means the CI lower bound of the Harbor
    held-out T2 paired difference (best stage − base) is > 0.

**Serving reachability.** For the Harbor runs, the served adapter must be reachable from
Harbor's agent container. Add the serving host to the allowlist at export time
(`--allow-host`) for a **separate, uncommitted** copy of the held-out dataset (e.g.
`datasets/t2-heldout-trained/`). Never change the committed judge dataset's allowlist
for this.

## 3. Scope

**In scope:**

- `railway/train/` (`data.py`, `bc.py`, `dagger.py`, `grpo.py`, `evaluate.py`);
- the `railway-train` script;
- the `[train]` extra;
- train-seed accepted lists;
- evaluation manifests;
- `docs/training.md`;
- the report update;
- the README section "Training results".

**Out of scope:**

- Training the mid or big model.
- New environment features.
- Changing anything frozen in T09.

## 4. Implementation notes and pitfalls

- **Don't commit data or artifacts.** `data/` and `artifacts/` go in `.gitignore`.
  Commit only `docs/training.md`, the report, the manifests and the results JSONL.
- **Chat templates differ** by model family. Test the formatting on the **chosen** model's
  tokenizer (T10-05) before spending GPU time.
- **Time budget.** BC first, and evaluate it. DAgger only if BC is evaluated and time
  remains. GRPO last. A positive, honest BC result beats an unfinished GRPO run.
- **Honesty.** If G8 fails, say so in the README and in `docs/training.md` with the
  numbers. Do not move to a cherry-picked subset.

## 5. Tests (write these first)

### 5.1 `tests/t10/test_data.py` (default markers; CPU only)

| Id | Test | Assertion |
|---|---|---|
| T10-01 | `test_T10_01_bc_examples_match_reference_turns` | `build_bc` on 2 T1 train seeds: example count == the number of turns the fake reference-model episode took; each `target` == the call executed at that turn |
| T10-02 | `test_T10_02_targets_replay_to_c_ref` | For each of those seeds: replaying all targets in order through a fresh `Session` gives exactly the accepted-list `c_ref` |
| T10-03 | `test_T10_03_prompt_identity` | Each example's `messages` == `prompts.build(...)` recomputed for that turn (the same function object as T08 evaluation; a golden check on mini3's first 2 examples) |
| T10-04 | `test_T10_04_no_leakage` | Every example's seed ∈ the train range; none ∈ dev, held-out or fresh; no preset `T2-S14`; train and val seed sets are disjoint |
| T10-05 | `test_T10_05_dagger_labels` | 20 states visited by `RandomLegal(0)` on a T1 train seed: the label == `Reference.decide(static, state)[0]`; deterministic over 2 runs |
| T10-06 | `test_T10_06_aggregate_dedupe` | BC (n) ∪ DAgger (m) with k injected duplicates → size n + m − k |
| T10-07 | `test_T10_07_grpo_reward` | A valid call → `−Reference.evaluate(...)`; unparseable text → `−(evaluate(no-op) + 50)`; pool evaluation == serial evaluation for 16 completions |

### 5.2 `tests/t10/test_formatting.py` (marker `model`: needs the chosen tokenizer files)

| Id | Test | Assertion |
|---|---|---|
| T10-08 | `test_T10_08_assistant_only_loss_mask` | For 3 examples, label ids are `-100` on every prompt token and equal the input ids on the assistant tool-call tokens; decoding the unmasked span gives a tool call that parses back to the example's `target` |
| T10-09 | `test_T10_09_prompt_length_fits` | The 99th percentile of tokenized prompt lengths ≤ the configured max sequence; the value is logged |

### 5.3 `tests/t10/test_training_smoke.py` (marker `gpu`)

| Id | Test | Assertion |
|---|---|---|
| T10-10 | `test_T10_10_sft_smoke` | LoRA SFT, 30 steps on 300 BC examples with the chosen small model (or a ≤ 1.5B model of the same family if memory is short): mean loss of the last 5 steps < the first 5; the adapter dir exists and loads with PEFT |
| T10-11 | `test_T10_11_grpo_smoke` | 5 GRPO steps on 20 states with G = 4: completes; rewards are logged; no NaN |
| T10-12 | `test_T10_12_served_adapter_plays` | The adapter served via vLLM (OpenAI-compatible) plays 1 T1 dev episode through the T08 loop → `status: ok` |

### 5.4 `tests/t10/test_results.py` (outcome checks; need the real runs)

| Id | Test | Assertion |
|---|---|---|
| T10-13 | `test_T10_13_eval_files` | `results/heldout-small-bc.jsonl` (and each later stage that ran) has 60 rows, ≥ 90% ok, `rules_version == "1.0"`; the S14 runs exist for the best stage |
| T10-14 | `test_T10_14_training_doc` | `docs/training.md` has the sections of §2.6; its before/after numbers equal `railway-eval compare` recomputed from the results files; it has exactly one line `G8: PASS` or `G8: FAIL`, consistent with the recomputed CI |
| T10-15 | `test_T10_15_report_updated` | `reports/heldout.html` no longer has `pending T10`; chart 3 contains one bar per evaluated stage |
| T10-16 | `test_T10_16_no_artifacts_committed` | `git check-ignore data/ artifacts/` → both ignored |

## 6. Gate commands

```bash
uv run pytest tests/t01 ... tests/t10 -m "not harbor and not model and not gpu"
```
```bash
uv run pytest tests/t10 -m "model or gpu"
```

Then the outcome checks T10-13 to T10-16 after the evaluation runs.

## 7. Definition of done

- [ ] All T10 tests pass, and every earlier ticket's default suite still passes.
- [ ] At least BC is trained and evaluated on held-out T2 through Harbor. DAgger and
      GRPO done or explicitly cut (the gate record says which).
- [ ] `docs/training.md` and the report are updated. The README "Training results"
      section states the G8 verdict honestly.
- [ ] **G8** (CI lower bound > 0) is the goal, not the completion criterion. The ticket
      is complete when the pipeline is verified by the tests and the outcome is measured
      and reported truthfully, PASS or FAIL.
- [ ] Gate record, reviewers, handover (no commit). The handover lists the result and
      doc files to commit, and confirms that `data/` and `artifacts/` are ignored.
