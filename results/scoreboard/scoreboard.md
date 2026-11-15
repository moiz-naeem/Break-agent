# Network benchmark scoreboard (v0.3, live)

_Snapshot: 2026-11-15 09:36 UTC._

### Progress

| model | id | episodes ok / expected | T1 | T2 | T3 | errors |
|---|---|---|---|---|---|---|
| Gemini 4 Carbon | `gemini-carbon-high` | 87 / 270 (32%) | 58 | 29 | 0 | 0 |
| Qwen 3.8 27B (base) | `qwen38-think` | 18 / 270 (7%) | 17 | 1 | 0 | 0 |
| Qwen 3.8 27B (fine-tuned) | `qwen38-think-sft` | 13 / 30 (43%) | 13 | 0 | 0 | 0 |

### Overall score per tier

Reward = −(C1+C2+C3+C4+C5), higher is better; averaged per seed first. CI = 95% seed bootstrap. Rep std = mean within-seed std across repeated runs.

| tier | model | seeds | eps | mean reward [95% CI] | rep std | do-nothing floor | C1 | C2 | C3 | C4 | C5 | on-time % | pax delay min | freight % | clean % | gridlocks | unfinished |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | Gemini 4 Carbon | 26 | 58 | **-1,549.9** [-2,407.2, -943.2] | 312.0 | -28,618.6 | 356.5 | 1,147.5 | 3.3 | 0.0 | 42.6 | 67.6 | 37,174 | 78.2 | 23.7 | 0.03 | 1.74 |
| T1 | Qwen 3.8 27B (base) | 14 | 17 | **-2,518.4** [-4,127.5, -1,234.5] | 1,812.3 | -28,568.9 | 499.5 | 1,817.0 | 2.6 | 0.0 | 199.4 | 58.9 | 45,499 | 92.9 | 17.9 | 0.18 | 2.86 |
| T1 | Qwen 3.8 27B (fine-tuned) | 13 | 13 | **-1,145.3** [-1,765.2, -632.2] | – | -29,062.6 | 482.6 | 631.4 | 6.0 | 0.0 | 25.3 | 66.5 | 43,740 | 92.3 | 15.4 | 0.00 | 1.38 |
| T2 | Gemini 4 Carbon | 21 | 29 | **-6,557.2** [-8,698.7, -4,721.4] | 1,314.6 | -114,127.3 | 2,674.4 | 3,533.5 | 13.3 | 0.0 | 336.0 | 60.6 | 272,631 | 91.3 | 4.8 | 0.14 | 4.09 |
| T2 | Qwen 3.8 27B (base) | 1 | 1 | **-2,504.9** [-2,504.9, -2,504.9] | – | -114,443.7 | 1,833.0 | 483.4 | 8.5 | 0.0 | 180.0 | 81.0 | 192,446 | 100.0 | 0.0 | 0.00 | 2.00 |
| ALL | Gemini 4 Carbon | 47 | 87 | **-3,787.2** [-5,027.8, -2,681.2] | 562.6 | -66,824.6 | 1,392.2 | 2,213.6 | 7.8 | 0.0 | 173.7 | 64.5 | 142,378 | 84.0 | 15.2 | 0.08 | 2.79 |
| ALL | Qwen 3.8 27B (base) | 15 | 18 | **-2,517.5** [-3,998.7, -1,344.1] | 1,812.3 | -34,293.9 | 588.4 | 1,728.1 | 3.0 | 0.0 | 198.1 | 60.4 | 55,296 | 93.3 | 16.7 | 0.17 | 2.80 |
| ALL | Qwen 3.8 27B (fine-tuned) | 13 | 13 | **-1,145.3** [-1,765.2, -632.2] | – | -29,062.6 | 482.6 | 631.4 | 6.0 | 0.0 | 25.3 | 66.5 | 43,740 | 92.3 | 15.4 | 0.00 | 1.38 |

### Head-to-head: Gemini 4 Carbon vs Qwen 3.8 27B (base), paired seeds

Difference = Carbon − Qwen in mean reward on seeds both finished; positive means Carbon is better.

| tier | paired seeds | Carbon | Qwen base | difference [95% CI] | Carbon wins / ties / Qwen wins |
|---|---|---|---|---|---|
| T1 | 12 | -1,955.2 | -1,998.6 | **43.4** [-1,515.6, 1,509.6] | 4 / 0 / 8 |
| T2 | 1 | -1,112.8 | -2,504.9 | **1,392.1** [1,392.1, 1,392.1] | 1 / 0 / 0 |
| ALL | 13 | -1,890.4 | -2,037.5 | **147.1** [-1,356.5, 1,528.2] | 5 / 0 / 8 |

### Post-training: what fine-tuning changed (T1 held-out test seeds)

Seeds finished by all three models: **7**.

| model | mean reward [95% CI] | on-time % | unfinished trains | clean % |
|---|---|---|---|---|
| Gemini 4 Carbon | **-1,559.3** [-2,161.9, -936.7] | 64.2 | 1.95 | 14.3 |
| Qwen 3.8 27B (fine-tuned) | **-1,323.7** [-2,364.2, -467.7] | 62.1 | 1.43 | 28.6 |
| Qwen 3.8 27B (base) | **-1,915.5** [-3,349.7, -828.4] | 59.3 | 2.14 | 14.3 |

| comparison | difference [95% CI] | A wins / ties / B wins |
|---|---|---|
| fine-tuned − base Qwen (what Qwen learned) | **591.8** [-1,041.8, 2,385.0] | 4 / 2 / 1 |
| Carbon − fine-tuned Qwen (gap left) | **-235.5** [-961.1, 460.7] | 3 / 0 / 4 |
| Carbon − base Qwen (gap before training) | **356.2** [-825.5, 2,055.4] | 3 / 0 / 4 |

Share of the Carbon-to-Qwen gap closed by fine-tuning: **166.1%**.

### Integrity checks

All rows are held-out `test` split seeds and every scenario hash matches `seeds/net-T*-test.json`.
