# Network benchmark scoreboard (v0.3, live)

_Snapshot: 2026-11-15 11:09 UTC._

### Progress

| model | id | episodes ok / expected | T1 | T2 | T3 | errors |
|---|---|---|---|---|---|---|
| Gemini 4 Carbon | `gemini-carbon-high` | 127 / 270 (47%) | 58 | 33 | 36 | 0 |
| Qwen 3.8 27B (base) | `qwen38-think` | 25 / 270 (9%) | 20 | 5 | 0 | 0 |
| Qwen 3.8 27B (fine-tuned) | `qwen38-think-sft` | 21 / 30 (70%) | 21 | 0 | 0 | 0 |

### Overall score per tier

Reward = −(C1+C2+C3+C4+C5), higher is better; averaged per seed first. CI = 95% seed bootstrap. Rep std = mean within-seed std across repeated runs.

| tier | model | seeds | eps | mean reward [95% CI] | rep std | do-nothing floor | C1 | C2 | C3 | C4 | C5 | on-time % | pax delay min | freight % | clean % | gridlocks | unfinished |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | Gemini 4 Carbon | 26 | 58 | **-1,549.9** [-2,407.2, -943.2] | 312.0 | -28,618.6 | 356.5 | 1,147.5 | 3.3 | 0.0 | 42.6 | 67.6 | 37,174 | 78.2 | 23.7 | 0.03 | 1.74 |
| T1 | Qwen 3.8 27B (base) | 16 | 20 | **-2,993.4** [-4,590.0, -1,594.0] | 1,429.4 | -28,440.3 | 551.3 | 2,199.2 | 2.4 | 0.0 | 240.5 | 56.9 | 49,089 | 93.8 | 15.6 | 0.22 | 3.34 |
| T1 | Qwen 3.8 27B (fine-tuned) | 21 | 21 | **-1,179.4** [-1,627.6, -788.3] | – | -28,294.1 | 497.5 | 654.5 | 4.3 | 0.0 | 23.1 | 67.0 | 45,390 | 92.9 | 19.0 | 0.00 | 1.52 |
| T2 | Gemini 4 Carbon | 24 | 33 | **-7,611.5** [-9,760.7, -5,635.9] | 1,556.0 | -114,186.3 | 2,769.5 | 4,380.6 | 13.4 | 0.0 | 448.0 | 58.5 | 281,482 | 92.4 | 4.2 | 0.25 | 4.78 |
| T2 | Qwen 3.8 27B (base) | 3 | 5 | **-5,863.7** [-9,937.0, -3,598.5] | 3,749.5 | -114,848.6 | 2,272.4 | 3,232.5 | 2.1 | 0.0 | 356.7 | 69.0 | 232,274 | 100.0 | 0.0 | 0.17 | 2.67 |
| T3 | Gemini 4 Carbon | 23 | 36 | **-37,328.8** [-46,903.5, -28,274.5] | 8,825.9 | -300,173.2 | 7,328.9 | 29,192.4 | 67.8 | 104.1 | 635.7 | 56.1 | 669,470 | 91.2 | 0.0 | 0.20 | 20.27 |
| ALL | Gemini 4 Carbon | 73 | 127 | **-14,815.6** [-19,538.3, -10,434.3] | 2,902.1 | -142,308.8 | 3,346.6 | 11,046.5 | 26.9 | 32.8 | 362.8 | 61.0 | 316,711 | 87.0 | 9.8 | 0.15 | 8.58 |
| ALL | Qwen 3.8 27B (base) | 19 | 25 | **-3,446.6** [-5,028.5, -2,065.4] | 2,202.8 | -42,083.7 | 823.1 | 2,362.4 | 2.3 | 0.0 | 258.8 | 58.8 | 78,013 | 94.7 | 13.2 | 0.21 | 3.24 |
| ALL | Qwen 3.8 27B (fine-tuned) | 21 | 21 | **-1,179.4** [-1,627.6, -788.3] | – | -28,294.1 | 497.5 | 654.5 | 4.3 | 0.0 | 23.1 | 67.0 | 45,390 | 92.9 | 19.0 | 0.00 | 1.52 |

### Head-to-head: Gemini 4 Carbon vs Qwen 3.8 27B (base), paired seeds

Difference = Carbon − Qwen in mean reward on seeds both finished; positive means Carbon is better.

| tier | paired seeds | Carbon | Qwen base | difference [95% CI] | Carbon wins / ties / Qwen wins |
|---|---|---|---|---|---|
| T1 | 13 | -1,829.5 | -2,033.4 | **203.9** [-1,254.7, 1,659.2] | 5 / 0 / 8 |
| T2 | 3 | -3,906.0 | -5,863.7 | **1,957.7** [-3,804.6, 8,824.2] | 2 / 0 / 1 |
| ALL | 16 | -2,218.8 | -2,751.6 | **532.7** [-1,065.1, 2,246.4] | 7 / 0 / 9 |

### Post-training: what fine-tuning changed (T1 held-out test seeds)

Seeds finished by all three models: **9**.

| model | mean reward [95% CI] | on-time % | unfinished trains | clean % |
|---|---|---|---|---|
| Gemini 4 Carbon | **-1,274.6** [-1,882.3, -674.3] | 65.2 | 1.57 | 27.8 |
| Qwen 3.8 27B (fine-tuned) | **-1,086.8** [-1,973.9, -381.2] | 65.0 | 1.22 | 33.3 |
| Qwen 3.8 27B (base) | **-1,575.4** [-2,836.1, -666.3] | 65.4 | 1.83 | 22.2 |

| comparison | difference [95% CI] | A wins / ties / B wins |
|---|---|---|
| fine-tuned − base Qwen (what Qwen learned) | **488.6** [-738.0, 1,882.8] | 4 / 3 / 2 |
| Carbon − fine-tuned Qwen (gap left) | **-187.8** [-733.8, 340.8] | 3 / 0 / 6 |
| Carbon − base Qwen (gap before training) | **300.8** [-591.7, 1,649.8] | 3 / 0 / 6 |

Share of the Carbon-to-Qwen gap closed by fine-tuning: **162.4%**.

Fine-tuned vs base Qwen on all 10 T1 seeds both finished (Carbon not required): 1,221.1 [-389.9, 3,142.2], wins/ties/losses 5/3/2.

### Integrity checks

All rows are held-out `test` split seeds and every scenario hash matches `seeds/net-T*-test.json`.
