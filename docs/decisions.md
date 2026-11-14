# Decision log

Decisions Anh made for the v0.3 network track (Moiz's engine, branch `net`). Newest last. Where a decision
changes `DESIGN.md` or `docs/network-v0.3.md`, this log wins until those documents are updated.

## 2026-11-14

| # | Decision | Detail |
|---|---|---|
| 1 | Run two tracks in parallel, then narrow to one | Both Moiz's network track and our v03 line track at first; later the same day, **Moiz's track only**. The v03 track is frozen. |
| 2 | Reward is minus the total cost | `reward.txt = −(C1 + C2 + C3 + C4 + C5)`. Nothing else is in the reward: no greedy or best-plan normalisation, no clip, no bonuses or shaping. |
| 3 | C4 (crew timeout) is in | 40 per crew timeout (v0.3 scale, down from 300). The timeout also blocks its section for 45 min, and the delays this causes are charged through C1–C3. C1, C2, C3 and C5 are as in `DESIGN.md` §6.6. |
| 4 | Greedy and reference are removed | Neither greedy nor a reference/best-plan anchor is computed, stored (`norm.json` is dropped), or shown in `details.json` or reports. `details.json` carries the cost per term (C1–C5), operational KPIs, and counts. |
| 5 | T1 runs first; T2 and T3 built in parallel | T1 is ready first so T1 seeds and benchmark start immediately; a parallel lane speeds up the generator and builds the T2 (hub) and T3 (two hubs + bypass) templates so all three tiers ship. |
| 6 | We build everything on Moiz's engine | Disruptions, cost, tools, verifier, Harbor packaging, seeds, and benchmark. Moiz's pushes to `origin/work` are merged into `net` as they arrive. |
| 7 | All heavy work runs on Modal | Test suites, generation, seed filtering, Harbor runs, and the benchmark. The laptop is for editing, git and short checks only. |
| 8 | Benchmark models | Claude Fable 5.2 (high effort), Qwen 3.8 27B (thinking on), Gemini 4 Carbon (high), and the strongest available Nemotron (`nemotron-3.5-super-vl`). |
| 9 | No spend cap | The full benchmark runs straight after the dry run, without asking. |
| 10 | Harbor tasks and Harbor trial logs are required deliverables | Exported Harbor tasks (`datasets/...`) and Harbor trial logs (`<job>/<trial>/` with `result.json`, `verifier/reward.txt`, `verifier/details.json`, `agent/...`) are produced alongside full `railway/loop` transcripts on Modal. |
| 11 | Benchmark headline | Mean −cost (`−(C1 + C2 + C3 + C4 + C5)`) per model on the same held-out test seeds across T1, T2, and T3, alongside operational KPIs (no greedy or reference row). |
| 12 | Post-training pipeline | Handled separately on Anh's own post-training plan; Lane E here is cancelled. |
| 13 | Seed sets are disjoint | Disjoint `dev`, `test` (held out), and `train` seed sets per tier; held-out `test` episodes are never used for training. |
