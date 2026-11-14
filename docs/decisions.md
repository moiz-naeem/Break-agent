# Decision log

Decisions Anh made for the v0.3 network track (Moiz's engine, branch `net`). Newest last. Where a decision
changes `DESIGN.md` or `docs/network-v0.3.md`, this log wins until those documents are updated.

## 2026-11-14

| # | Decision | Detail |
|---|---|---|
| 1 | Run two tracks in parallel, then narrow to one | Both Moiz's network track and our v03 line track at first; later the same day, **Moiz's track only**. The v03 track is frozen. |
| 2 | Reward is minus the total cost | `reward.txt = −(C1 + C2 + C3 + C4 + C5)`. Nothing else is in the reward: no greedy or best-plan normalisation, no clip, no bonuses or shaping. |
| 3 | C4 (crew timeout) is in | 40 per crew timeout (v0.3 scale, down from 300). The timeout also blocks its section for 45 min, and the delays this causes are charged through C1–C3. C1, C2, C3 and C5 are as in `DESIGN.md` §6.6. |
| 4 | Greedy and best plan are report-only | `details.json` carries the cost per term, KPIs and greedy's cost as a reference row. No reference planner, solver or best-plan anchor is needed for the reward, so they are out of scope for now. |
| 5 | T1 runs first; T2 and T3 built in parallel | T1 is ready first so T1 seeds and benchmark start immediately; a parallel lane speeds up the generator and builds the T2 (hub) and T3 (two hubs + bypass) templates so all three tiers ship. |
| 6 | We build everything on Moiz's engine | Disruptions, cost, tools, verifier, seeds, benchmark and fine-tuning. Moiz's pushes to `origin/work` are merged into `net` as they arrive. |
| 7 | All heavy work runs on Modal | Test suites, generation, seed filtering, the benchmark and training. The laptop is for editing, git and short checks only. |
| 8 | Benchmark models | Claude Fable 5.2 (high effort), Qwen 3.8 27B (thinking on), Gemini 4 Carbon (high), and the strongest available Nemotron (to be confirmed against `nemotron-3.5-super`). |
| 9 | No spend cap | The full benchmark runs straight after the dry run, without asking. |
| 10 | Benchmark runs through our own loop | Direct API calls via `railway/loop`, not Harbor. Harbor packaging is still built, but off the critical path. |
| 11 | Benchmark headline | Mean −cost per model on the same test seeds, with greedy as a reference row. |
| 12 | Fine-tuning follows the benchmark | Qwen 3.8 27B (thinking), LoRA, on Modal GPUs: warm start (SFT) on Fable episodes that beat greedy, then GRPO with reward −cost, groups per training seed; final evaluation on the held-out test seeds next to the benchmark models. |
| 13 | Seed sets are disjoint | Dev (20), test (held out) and a large training set. Warm-start data comes from Fable runs on the **training** seeds, never from benchmark episodes on test seeds. |
