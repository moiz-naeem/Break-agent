# Network benchmark scoreboard (v0.3, live)

_Snapshot: 2026-11-15 10:39 UTC._

### Progress

| model | id | episodes ok / expected | T1 | T2 | T3 | errors |
|---|---|---|---|---|---|---|
| Gemini 4 Carbon | `gemini-carbon-high` | 125 / 270 (46%) | 58 | 31 | 36 | 0 |
| Qwen 3.8 27B (base) | `qwen38-think` | 24 / 270 (9%) | 19 | 5 | 0 | 0 |
| Qwen 3.8 27B (fine-tuned) | `qwen38-think-sft` | 17 / 30 (57%) | 17 | 0 | 0 | 0 |

### Overall score per tier

Reward = −(C1+C2+C3+C4+C5), higher is better; averaged per seed first. CI = 95% seed bootstrap. Rep std = mean within-seed std across repeated runs.

| tier | model | seeds | eps | mean reward [95% CI] | rep std | do-nothing floor | C1 | C2 | C3 | C4 | C5 | on-time % | pax delay min | freight % | clean % | gridlocks | unfinished |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | Gemini 4 Carbon | 26 | 58 | **-1,549.9** [-2,407.2, -943.2] | 312.0 | -28,618.6 | 356.5 | 1,147.5 | 3.3 | 0.0 | 42.6 | 67.6 | 37,174 | 78.2 | 23.7 | 0.03 | 1.74 |
| T1 | Qwen 3.8 27B (base) | 15 | 19 | **-2,514.0** [-3,922.9, -1,347.3] | 1,429.4 | -28,698.0 | 587.3 | 1,737.1 | 2.5 | 0.0 | 187.0 | 56.0 | 52,238 | 93.3 | 16.7 | 0.17 | 2.83 |
| T1 | Qwen 3.8 27B (fine-tuned) | 17 | 17 | **-1,135.1** [-1,665.7, -669.0] | – | -28,670.3 | 476.5 | 631.0 | 4.9 | 0.0 | 22.7 | 69.5 | 43,289 | 94.1 | 23.5 | 0.00 | 1.47 |
| T2 | Gemini 4 Carbon | 22 | 31 | **-6,741.1** [-8,834.4, -5,019.8] | 1,556.0 | -114,136.5 | 2,700.7 | 3,654.5 | 13.3 | 0.0 | 372.5 | 60.4 | 275,935 | 91.7 | 4.5 | 0.18 | 4.45 |
| T2 | Qwen 3.8 27B (base) | 3 | 5 | **-5,863.7** [-9,937.0, -3,598.5] | 3,749.5 | -114,848.6 | 2,272.4 | 3,232.5 | 2.1 | 0.0 | 356.7 | 69.0 | 232,274 | 100.0 | 0.0 | 0.17 | 2.67 |
| T3 | Gemini 4 Carbon | 23 | 36 | **-37,328.8** [-46,903.5, -28,274.5] | 8,825.9 | -300,173.2 | 7,328.9 | 29,192.4 | 67.8 | 104.1 | 635.7 | 56.1 | 669,470 | 91.2 | 0.0 | 0.20 | 20.27 |
| ALL | Gemini 4 Carbon | 71 | 125 | **-14,748.8** [-19,682.9, -10,206.7] | 2,902.1 | -143,085.5 | 3,341.5 | 11,009.3 | 27.3 | 33.7 | 337.0 | 61.6 | 315,985 | 86.6 | 10.1 | 0.13 | 8.58 |
| ALL | Qwen 3.8 27B (base) | 18 | 24 | **-3,072.2** [-4,491.1, -1,800.5] | 2,202.8 | -43,056.4 | 868.2 | 1,986.4 | 2.4 | 0.0 | 215.3 | 58.2 | 82,244 | 94.4 | 13.9 | 0.17 | 2.81 |
| ALL | Qwen 3.8 27B (fine-tuned) | 17 | 17 | **-1,135.1** [-1,665.7, -669.0] | – | -28,670.3 | 476.5 | 631.0 | 4.9 | 0.0 | 22.7 | 69.5 | 43,289 | 94.1 | 23.5 | 0.00 | 1.47 |

### Head-to-head: Gemini 4 Carbon vs Qwen 3.8 27B (base), paired seeds

Difference = Carbon − Qwen in mean reward on seeds both finished; positive means Carbon is better.

| tier | paired seeds | Carbon | Qwen base | difference [95% CI] | Carbon wins / ties / Qwen wins |
|---|---|---|---|---|---|
| T1 | 13 | -1,829.5 | -2,033.4 | **203.9** [-1,254.7, 1,659.2] | 5 / 0 / 8 |
| T2 | 3 | -3,906.0 | -5,863.7 | **1,957.7** [-3,804.6, 8,824.2] | 2 / 0 / 1 |
| ALL | 16 | -2,218.8 | -2,751.6 | **532.7** [-1,065.1, 2,246.4] | 7 / 0 / 9 |

### Post-training: what fine-tuning changed (T1 held-out test seeds)

Seeds finished by all three models: **8**.

| model | mean reward [95% CI] | on-time % | unfinished trains | clean % |
|---|---|---|---|---|
| Gemini 4 Carbon | **-1,373.0** [-1,996.9, -721.5] | 67.5 | 1.71 | 25.0 |
| Qwen 3.8 27B (fine-tuned) | **-1,161.9** [-2,135.3, -360.0] | 66.4 | 1.25 | 37.5 |
| Qwen 3.8 27B (base) | **-1,714.8** [-3,028.9, -695.3] | 63.9 | 1.94 | 25.0 |

| comparison | difference [95% CI] | A wins / ties / B wins |
|---|---|---|
| fine-tuned − base Qwen (what Qwen learned) | **552.9** [-798.5, 2,079.2] | 4 / 3 / 1 |
| Carbon − fine-tuned Qwen (gap left) | **-211.2** [-851.8, 374.5] | 3 / 0 / 5 |
| Carbon − base Qwen (gap before training) | **341.8** [-634.5, 1,788.6] | 3 / 0 / 5 |

Share of the Carbon-to-Qwen gap closed by fine-tuning: **161.8%**.

### Integrity checks

All rows are held-out `test` split seeds and every scenario hash matches `seeds/net-T*-test.json`.
