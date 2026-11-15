# Network benchmark scoreboard (v0.3, live)

_Snapshot: 2026-11-15 07:40 UTC._

### Progress

| model | id | episodes ok / expected | T1 | T2 | T3 | errors |
|---|---|---|---|---|---|---|
| Gemini 4 Carbon | `gemini-carbon-high` | 60 / 270 (22%) | 58 | 2 | 0 | 0 |
| Qwen 3.8 27B (base) | `qwen38-think` | 2 / 270 (1%) | 2 | 0 | 0 | 0 |
| Qwen 3.8 27B (fine-tuned) | `qwen38-think-sft` | 1 / 30 (3%) | 1 | 0 | 0 | 0 |

### Overall score per tier

Reward = −(C1+C2+C3+C4+C5), higher is better; averaged per seed first. CI = 95% seed bootstrap. Rep std = mean within-seed std across repeated runs.

| tier | model | seeds | eps | mean reward [95% CI] | rep std | do-nothing floor | C1 | C2 | C3 | C4 | C5 | on-time % | pax delay min | freight % | clean % | gridlocks | unfinished |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | Gemini 4 Carbon | 26 | 58 | **-1,549.9** [-2,407.2, -943.2] | 312.0 | -28,618.6 | 356.5 | 1,147.5 | 3.3 | 0.0 | 42.6 | 67.6 | 37,174 | 78.2 | 23.7 | 0.03 | 1.74 |
| T1 | Qwen 3.8 27B (base) | 2 | 2 | **-21.6** [-29.1, -14.0] | – | -26,225.3 | 9.6 | 0.0 | 0.0 | 0.0 | 12.0 | 98.3 | 6,208 | 100.0 | 100.0 | 0.00 | 0.00 |
| T1 | Qwen 3.8 27B (fine-tuned) | 1 | 1 | **-305.4** [-305.4, -305.4] | – | -29,120.2 | 89.5 | 200.0 | 0.0 | 0.0 | 16.0 | 90.3 | 12,523 | 100.0 | 0.0 | 0.00 | 1.00 |
| T2 | Gemini 4 Carbon | 2 | 2 | **-2,469.1** [-3,345.3, -1,593.0] | – | -113,641.8 | 1,086.6 | 1,250.8 | 0.8 | 0.0 | 131.0 | 83.4 | 141,962 | 100.0 | 0.0 | 0.00 | 2.00 |
| ALL | Gemini 4 Carbon | 28 | 60 | **-1,615.6** [-2,381.2, -1,047.8] | 312.0 | -34,691.7 | 408.6 | 1,154.9 | 3.1 | 0.0 | 49.0 | 68.7 | 44,659 | 79.8 | 22.0 | 0.02 | 1.76 |
| ALL | Qwen 3.8 27B (base) | 2 | 2 | **-21.6** [-29.1, -14.0] | – | -26,225.3 | 9.6 | 0.0 | 0.0 | 0.0 | 12.0 | 98.3 | 6,208 | 100.0 | 100.0 | 0.00 | 0.00 |
| ALL | Qwen 3.8 27B (fine-tuned) | 1 | 1 | **-305.4** [-305.4, -305.4] | – | -29,120.2 | 89.5 | 200.0 | 0.0 | 0.0 | 16.0 | 90.3 | 12,523 | 100.0 | 0.0 | 0.00 | 1.00 |

### Head-to-head: Gemini 4 Carbon vs Qwen 3.8 27B (base), paired seeds

Difference = Carbon − Qwen in mean reward on seeds both finished; positive means Carbon is better.

| tier | paired seeds | Carbon | Qwen base | difference [95% CI] | Carbon wins / ties / Qwen wins |
|---|---|---|---|---|---|
| T1 | 2 | -42.8 | -21.6 | **-21.2** [-40.4, -2.0] | 0 / 0 / 2 |
| ALL | 2 | -42.8 | -21.6 | **-21.2** [-40.4, -2.0] | 0 / 0 / 2 |

### Post-training: what fine-tuning changed (T1 held-out test seeds)

Seeds finished by all three models: **0**.

### Integrity checks

All rows are held-out `test` split seeds and every scenario hash matches `seeds/net-T*-test.json`.
