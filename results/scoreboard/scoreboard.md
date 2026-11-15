# Network benchmark scoreboard (v0.3, live)

_Snapshot: 2026-11-15 09:51 UTC._

### Progress

| model | id | episodes ok / expected | T1 | T2 | T3 | errors |
|---|---|---|---|---|---|---|
| Gemini 4 Carbon | `gemini-carbon-high` | 89 / 270 (33%) | 58 | 29 | 2 | 0 |
| Qwen 3.8 27B (base) | `qwen38-think` | 20 / 270 (7%) | 18 | 2 | 0 | 0 |
| Qwen 3.8 27B (fine-tuned) | `qwen38-think-sft` | 13 / 30 (43%) | 13 | 0 | 0 | 0 |

### Overall score per tier

Reward = −(C1+C2+C3+C4+C5), higher is better; averaged per seed first. CI = 95% seed bootstrap. Rep std = mean within-seed std across repeated runs.

| tier | model | seeds | eps | mean reward [95% CI] | rep std | do-nothing floor | C1 | C2 | C3 | C4 | C5 | on-time % | pax delay min | freight % | clean % | gridlocks | unfinished |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | Gemini 4 Carbon | 26 | 58 | **-1,549.9** [-2,407.2, -943.2] | 312.0 | -28,618.6 | 356.5 | 1,147.5 | 3.3 | 0.0 | 42.6 | 67.6 | 37,174 | 78.2 | 23.7 | 0.03 | 1.74 |
| T1 | Qwen 3.8 27B (base) | 15 | 18 | **-2,495.2** [-3,906.1, -1,319.6] | 1,812.3 | -28,698.0 | 575.4 | 1,730.5 | 2.4 | 0.0 | 186.9 | 56.0 | 51,342 | 93.3 | 16.7 | 0.17 | 2.80 |
| T1 | Qwen 3.8 27B (fine-tuned) | 13 | 13 | **-1,145.3** [-1,765.2, -632.2] | – | -29,062.6 | 482.6 | 631.4 | 6.0 | 0.0 | 25.3 | 66.5 | 43,740 | 92.3 | 15.4 | 0.00 | 1.38 |
| T2 | Gemini 4 Carbon | 21 | 29 | **-6,557.2** [-8,698.7, -4,721.4] | 1,314.6 | -114,127.3 | 2,674.4 | 3,533.5 | 13.3 | 0.0 | 336.0 | 60.6 | 272,631 | 91.3 | 4.8 | 0.14 | 4.09 |
| T2 | Qwen 3.8 27B (base) | 2 | 2 | **-3,313.7** [-4,122.6, -2,504.9] | – | -116,655.2 | 2,597.2 | 499.3 | 4.2 | 0.0 | 213.0 | 71.3 | 256,752 | 100.0 | 0.0 | 0.00 | 2.00 |
| T3 | Gemini 4 Carbon | 2 | 2 | **-12,162.2** [-14,705.8, -9,618.7] | – | -302,892.3 | 6,649.3 | 5,158.4 | 52.0 | 20.0 | 282.5 | 62.8 | 665,084 | 100.0 | 0.0 | 0.00 | 6.50 |
| ALL | Gemini 4 Carbon | 49 | 89 | **-4,129.0** [-5,435.9, -2,957.8] | 562.6 | -76,460.0 | 1,606.7 | 2,333.8 | 9.6 | 0.8 | 178.2 | 64.4 | 163,713 | 84.7 | 14.6 | 0.07 | 2.94 |
| ALL | Qwen 3.8 27B (base) | 17 | 20 | **-2,591.5** [-3,851.3, -1,530.2] | 1,812.3 | -39,045.9 | 813.3 | 1,585.6 | 2.6 | 0.0 | 190.0 | 57.8 | 75,508 | 94.1 | 14.7 | 0.15 | 2.71 |
| ALL | Qwen 3.8 27B (fine-tuned) | 13 | 13 | **-1,145.3** [-1,765.2, -632.2] | – | -29,062.6 | 482.6 | 631.4 | 6.0 | 0.0 | 25.3 | 66.5 | 43,740 | 92.3 | 15.4 | 0.00 | 1.38 |

### Head-to-head: Gemini 4 Carbon vs Qwen 3.8 27B (base), paired seeds

Difference = Carbon − Qwen in mean reward on seeds both finished; positive means Carbon is better.

| tier | paired seeds | Carbon | Qwen base | difference [95% CI] | Carbon wins / ties / Qwen wins |
|---|---|---|---|---|---|
| T1 | 13 | -1,829.5 | -2,011.8 | **182.3** [-1,281.4, 1,620.4] | 5 / 0 / 8 |
| T2 | 2 | -2,157.5 | -3,313.7 | **1,156.3** [920.4, 1,392.1] | 2 / 0 / 0 |
| ALL | 15 | -1,873.2 | -2,185.4 | **312.2** [-995.2, 1,546.0] | 7 / 0 / 8 |

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
