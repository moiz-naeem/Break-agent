# Network benchmark scoreboard (v0.3, live)

_Snapshot: 2026-11-15 11:43 UTC._

### Progress

| model | id | episodes ok / expected | T1 | T2 | T3 | errors |
|---|---|---|---|---|---|---|
| Gemini 4 Carbon | `gemini-carbon-high` | 127 / 270 (47%) | 58 | 33 | 36 | 0 |
| Qwen 3.8 27B (base) | `qwen38-think` | 26 / 270 (10%) | 20 | 6 | 0 | 0 |
| Qwen 3.8 27B (fine-tuned) | `qwen38-think-sft` | 22 / 30 (73%) | 22 | 0 | 0 | 0 |

### Overall score per tier

Reward = −(C1+C2+C3+C4+C5), higher is better; averaged per seed first. CI = 95% seed bootstrap. Rep std = mean within-seed std across repeated runs.

| tier | model | seeds | eps | mean reward [95% CI] | rep std | do-nothing floor | C1 | C2 | C3 | C4 | C5 | on-time % | pax delay min | freight % | clean % | gridlocks | unfinished |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | Gemini 4 Carbon | 26 | 58 | **-1,549.9** [-2,407.2, -943.2] | 312.0 | -28,618.6 | 356.5 | 1,147.5 | 3.3 | 0.0 | 42.6 | 67.6 | 37,174 | 78.2 | 23.7 | 0.03 | 1.74 |
| T1 | Qwen 3.8 27B (base) | 16 | 20 | **-2,993.4** [-4,590.0, -1,594.0] | 1,429.4 | -28,440.3 | 551.3 | 2,199.2 | 2.4 | 0.0 | 240.5 | 56.9 | 49,089 | 93.8 | 15.6 | 0.22 | 3.34 |
| T1 | Qwen 3.8 27B (fine-tuned) | 22 | 22 | **-1,225.7** [-1,644.8, -836.1] | – | -28,469.1 | 490.6 | 708.5 | 4.3 | 0.0 | 22.3 | 67.3 | 44,901 | 93.2 | 18.2 | 0.00 | 1.59 |
| T2 | Gemini 4 Carbon | 24 | 33 | **-7,611.5** [-9,760.7, -5,635.9] | 1,556.0 | -114,186.3 | 2,769.5 | 4,380.6 | 13.4 | 0.0 | 448.0 | 58.5 | 281,482 | 92.4 | 4.2 | 0.25 | 4.78 |
| T2 | Qwen 3.8 27B (base) | 4 | 6 | **-4,642.2** [-8,352.3, -1,747.0] | 3,749.5 | -113,899.4 | 1,835.4 | 2,484.7 | 1.6 | 0.0 | 320.4 | 74.7 | 196,249 | 100.0 | 0.0 | 0.12 | 2.25 |
| T3 | Gemini 4 Carbon | 23 | 36 | **-37,328.8** [-46,903.5, -28,274.5] | 8,825.9 | -300,173.2 | 7,328.9 | 29,192.4 | 67.8 | 104.1 | 635.7 | 56.1 | 669,470 | 91.2 | 0.0 | 0.20 | 20.27 |
| ALL | Gemini 4 Carbon | 73 | 127 | **-14,815.6** [-19,538.3, -10,434.3] | 2,902.1 | -142,308.8 | 3,346.6 | 11,046.5 | 26.9 | 32.8 | 362.8 | 61.0 | 316,711 | 87.0 | 9.8 | 0.15 | 8.58 |
| ALL | Qwen 3.8 27B (base) | 20 | 26 | **-3,323.1** [-4,796.0, -2,029.0] | 2,202.8 | -45,532.1 | 808.1 | 2,256.3 | 2.2 | 0.0 | 256.5 | 60.4 | 78,521 | 95.0 | 12.5 | 0.20 | 3.12 |
| ALL | Qwen 3.8 27B (fine-tuned) | 22 | 22 | **-1,225.7** [-1,644.8, -836.1] | – | -28,469.1 | 490.6 | 708.5 | 4.3 | 0.0 | 22.3 | 67.3 | 44,901 | 93.2 | 18.2 | 0.00 | 1.59 |

### Head-to-head: Gemini 4 Carbon vs Qwen 3.8 27B (base), paired seeds

Difference = Carbon − Qwen in mean reward on seeds both finished; positive means Carbon is better.

| tier | paired seeds | Carbon | Qwen base | difference [95% CI] | Carbon wins / ties / Qwen wins |
|---|---|---|---|---|---|
| T1 | 13 | -1,829.5 | -2,033.4 | **203.9** [-1,254.7, 1,659.2] | 5 / 0 / 8 |
| T2 | 4 | -3,348.1 | -4,642.2 | **1,294.1** [-2,640.1, 6,444.0] | 2 / 0 / 2 |
| ALL | 17 | -2,186.8 | -2,647.2 | **460.4** [-1,014.6, 2,070.1] | 7 / 0 / 10 |

### Post-training: what fine-tuning changed (T1 held-out test seeds)

Seeds finished by all three models: **10**.

| model | mean reward [95% CI] | on-time % | unfinished trains | clean % |
|---|---|---|---|---|
| Gemini 4 Carbon | **-1,379.5** [-1,937.8, -785.6] | 66.6 | 1.72 | 25.0 |
| Qwen 3.8 27B (fine-tuned) | **-1,198.1** [-1,978.1, -496.5] | 65.9 | 1.40 | 30.0 |
| Qwen 3.8 27B (base) | **-1,578.8** [-2,660.5, -743.5] | 66.3 | 1.95 | 20.0 |

| comparison | difference [95% CI] | A wins / ties / B wins |
|---|---|---|
| fine-tuned − base Qwen (what Qwen learned) | **380.7** [-729.4, 1,655.2] | 4 / 3 / 3 |
| Carbon − fine-tuned Qwen (gap left) | **-181.4** [-667.6, 295.0] | 3 / 0 / 7 |
| Carbon − base Qwen (gap before training) | **199.3** [-598.8, 1,394.6] | 3 / 0 / 7 |

Share of the Carbon-to-Qwen gap closed by fine-tuning: **191.0%**.

Fine-tuned vs base Qwen on all 11 T1 seeds both finished (Carbon not required): 1,056.4 [-425.4, 2,839.7], wins/ties/losses 5/3/3.

### Integrity checks

All rows are held-out `test` split seeds and every scenario hash matches `seeds/net-T*-test.json`.
