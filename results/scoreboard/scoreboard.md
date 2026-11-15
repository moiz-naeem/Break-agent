# Network benchmark scoreboard (v0.3, live)

_Snapshot: 2026-11-15 05:57 UTC._

### Progress

| model | id | episodes ok / expected | T1 | T2 | T3 | errors |
|---|---|---|---|---|---|---|
| Gemini 4 Carbon | `gemini-carbon-high` | 58 / 270 (21%) | 58 | 0 | 0 | 0 |
| Qwen 3.8 27B (base) | `qwen38-think` | 1 / 270 (0%) | 1 | 0 | 0 | 82 |
| Qwen 3.8 27B (fine-tuned) | `qwen38-think-sft` | 0 / 30 (0%) | 0 | 0 | 0 | 0 |

### Overall score per tier

Reward = −(C1+C2+C3+C4+C5), higher is better; averaged per seed first. CI = 95% seed bootstrap. Rep std = mean within-seed std across repeated runs.

| tier | model | seeds | eps | mean reward [95% CI] | rep std | do-nothing floor | C1 | C2 | C3 | C4 | C5 | on-time % | pax delay min | freight % | clean % | gridlocks | unfinished |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | Gemini 4 Carbon | 26 | 58 | **-1,549.9** [-2,407.2, -943.2] | 312.0 | -28,618.6 | 356.5 | 1,147.5 | 3.3 | 0.0 | 42.6 | 67.6 | 37,174 | 78.2 | 23.7 | 0.03 | 1.74 |
| T1 | Qwen 3.8 27B (base) | 1 | 1 | **-29.1** [-29.1, -29.1] | – | -26,918.7 | 19.1 | 0.0 | 0.0 | 0.0 | 10.0 | 96.7 | 12,415 | 100.0 | 100.0 | 0.00 | 0.00 |
| ALL | Gemini 4 Carbon | 26 | 58 | **-1,549.9** [-2,407.2, -943.2] | 312.0 | -28,618.6 | 356.5 | 1,147.5 | 3.3 | 0.0 | 42.6 | 67.6 | 37,174 | 78.2 | 23.7 | 0.03 | 1.74 |
| ALL | Qwen 3.8 27B (base) | 1 | 1 | **-29.1** [-29.1, -29.1] | – | -26,918.7 | 19.1 | 0.0 | 0.0 | 0.0 | 10.0 | 96.7 | 12,415 | 100.0 | 100.0 | 0.00 | 0.00 |

### Head-to-head: Gemini 4 Carbon vs Qwen 3.8 27B (base), paired seeds

Difference = Carbon − Qwen in mean reward on seeds both finished; positive means Carbon is better.

| tier | paired seeds | Carbon | Qwen base | difference [95% CI] | Carbon wins / ties / Qwen wins |
|---|---|---|---|---|---|
| T1 | 1 | -69.5 | -29.1 | **-40.4** [-40.4, -40.4] | 0 / 0 / 1 |
| ALL | 1 | -69.5 | -29.1 | **-40.4** [-40.4, -40.4] | 0 / 0 / 1 |

### Post-training: what fine-tuning changed (T1 held-out test seeds)

Seeds finished by all three models: **0**.

### Integrity checks

All rows are held-out `test` split seeds and every scenario hash matches `seeds/net-T*-test.json`.
