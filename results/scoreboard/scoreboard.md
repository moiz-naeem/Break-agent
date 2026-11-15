# Network benchmark scoreboard (v0.3, live)

_Snapshot: 2026-11-15 07:54 UTC._

### Progress

| model | id | episodes ok / expected | T1 | T2 | T3 | errors |
|---|---|---|---|---|---|---|
| Gemini 4 Carbon | `gemini-carbon-high` | 65 / 270 (24%) | 58 | 7 | 0 | 0 |
| Qwen 3.8 27B (base) | `qwen38-think` | 5 / 270 (2%) | 5 | 0 | 0 | 0 |
| Qwen 3.8 27B (fine-tuned) | `qwen38-think-sft` | 1 / 30 (3%) | 1 | 0 | 0 | 0 |

### Overall score per tier

Reward = −(C1+C2+C3+C4+C5), higher is better; averaged per seed first. CI = 95% seed bootstrap. Rep std = mean within-seed std across repeated runs.

| tier | model | seeds | eps | mean reward [95% CI] | rep std | do-nothing floor | C1 | C2 | C3 | C4 | C5 | on-time % | pax delay min | freight % | clean % | gridlocks | unfinished |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | Gemini 4 Carbon | 26 | 58 | **-1,549.9** [-2,407.2, -943.2] | 312.0 | -28,618.6 | 356.5 | 1,147.5 | 3.3 | 0.0 | 42.6 | 67.6 | 37,174 | 78.2 | 23.7 | 0.03 | 1.74 |
| T1 | Qwen 3.8 27B (base) | 4 | 5 | **-950.7** [-1,879.8, -21.6] | 594.2 | -28,417.5 | 153.3 | 780.6 | 1.2 | 0.0 | 15.6 | 90.6 | 19,081 | 87.5 | 50.0 | 0.00 | 1.25 |
| T1 | Qwen 3.8 27B (fine-tuned) | 1 | 1 | **-305.4** [-305.4, -305.4] | – | -29,120.2 | 89.5 | 200.0 | 0.0 | 0.0 | 16.0 | 90.3 | 12,523 | 100.0 | 0.0 | 0.00 | 1.00 |
| T2 | Gemini 4 Carbon | 6 | 7 | **-2,952.4** [-4,612.0, -1,519.1] | 480.2 | -112,223.5 | 1,761.3 | 1,026.5 | 3.7 | 0.0 | 161.0 | 71.1 | 202,404 | 100.0 | 8.3 | 0.00 | 2.00 |
| ALL | Gemini 4 Carbon | 32 | 65 | **-1,812.9** [-2,531.4, -1,212.5] | 319.6 | -44,294.5 | 619.9 | 1,124.8 | 3.4 | 0.0 | 64.8 | 68.2 | 68,155 | 82.3 | 20.8 | 0.02 | 1.79 |
| ALL | Qwen 3.8 27B (base) | 4 | 5 | **-950.7** [-1,879.8, -21.6] | 594.2 | -28,417.5 | 153.3 | 780.6 | 1.2 | 0.0 | 15.6 | 90.6 | 19,081 | 87.5 | 50.0 | 0.00 | 1.25 |
| ALL | Qwen 3.8 27B (fine-tuned) | 1 | 1 | **-305.4** [-305.4, -305.4] | – | -29,120.2 | 89.5 | 200.0 | 0.0 | 0.0 | 16.0 | 90.3 | 12,523 | 100.0 | 0.0 | 0.00 | 1.00 |

### Head-to-head: Gemini 4 Carbon vs Qwen 3.8 27B (base), paired seeds

Difference = Carbon − Qwen in mean reward on seeds both finished; positive means Carbon is better.

| tier | paired seeds | Carbon | Qwen base | difference [95% CI] | Carbon wins / ties / Qwen wins |
|---|---|---|---|---|---|
| T1 | 4 | -1,084.8 | -950.7 | **-134.1** [-536.0, 154.8] | 1 / 0 / 3 |
| ALL | 4 | -1,084.8 | -950.7 | **-134.1** [-536.0, 154.8] | 1 / 0 / 3 |

### Post-training: what fine-tuning changed (T1 held-out test seeds)

Seeds finished by all three models: **0**.

### Integrity checks

All rows are held-out `test` split seeds and every scenario hash matches `seeds/net-T*-test.json`.
