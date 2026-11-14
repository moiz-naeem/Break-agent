# v0.3 Loop Dry-Run Report (`t1-dev-v03` & `t2-dev-v03`)

Date: 2026-11-14  
Branch: `bench-v03` (merged with `v03` @ `f9ae4e2`)  
Rules version: `0.3-provisional`  
Reward definition: **`reward = -(C1 + C2 + C3 + C4 + C5)` (`-cost_total`)** — nothing else.  
Normalised columns (`score_normalized = max(0, u)` and `score_unclipped = u`, where `u = (C_greedy - C_agent) / (C_greedy - C_best)`) are **report-only** secondary indicators so differences across tasks of different size remain readable.  
Artifacts on disk: `results/v03_dryrun/` (`*.jsonl`, `summary.json`, `transcripts/*.jsonl`, `trajectories/*.json` with continuous `[t0, t1]` train position segments).

---

## 1. Headline Aggregate Results (`5` tasks × `1` rep per tier)

| Run | Tier | OK | **Mean Reward (`-cost`)** | **95% CI (`-cost`)** | Norm Score Mean *(report-only)* | `score_ros` *(report-only)* | Unclipped Mean *(report-only)* | On-Time % | Pax Delay (min) | Freight Deliv % | Red Flags | Clean Shift | Input Tok | Output Tok | USD | Wall (s) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `t1-dev-reference` | T1 | 5/5 | **`-28.47`** | `[-40.68, -16.27]` | `0.6918` | `0.7723` | `0.6918` | `93.7%` | `1,939` | `100.0%` | `0.0` | `100%` | `2.29M` *(est)* | `6.0k` | `$0.00` | `266s` |
| `t1-dev-greedy` | T1 | 5/5 | **`-113.21`** | `[-204.78, -44.39]` | `0.0000` | `0.0000` | `0.0000` | `84.1%` | `6,238` | `100.0%` | `0.0` | `100%` | `3.13M` *(est)* | `8.0k` | `$0.00` | `12s` |
| `t1-dev-gemini-flash-low` | T1 | 5/5 | **`-11,433.21`** | `[-22,949.09, -226.87]` | `0.0000` | `-103.18` | `-234.42` | `56.9%` | `5,720` | `60.0%` | `20.6` | `40%` | `4.57M` | `16.0k` | `$2.09` | `1,416s` |
| `t2-dev-reference` | T2 | 5/5 | **`-1,604.52`** | `[-2,390.66, -926.38]` | `0.8251` | `0.7818` | `0.8251` | `41.1%` | `84,398` | `100.0%` | `3.2` | `0%` | `3.62M` *(est)* | `8.9k` | `$0.00` | `395s` |
| `t2-dev-greedy` | T2 | 5/5 | **`-2,853.29`** | `[-3,690.27, -2,119.87]` | `0.0000` | `0.0000` | `0.0000` | `35.3%` | `136,530` | `100.0%` | `5.0` | `0%` | `6.36M` *(est)* | `15.0k` | `$0.00` | `24s` |
| `t2-dev-gemini-flash-low` | T2 | 5/5 | **`-9,130.94`** | `[-18,524.57, -2,488.16]` | `0.2531` | `-3.9302` | `-7.5113` | `36.1%` | `91,430` | `80.0%` | `14.2` | `0%` | `16.90M` | `38.5k` | `$7.97` | `3,974s` |

*(For reference, the stored best-known anchors `-C_best` average **`-3.50`** on `t1-dev-v03` and **`-1,256.01`** on `t2-dev-v03`.)*

Total Gemini 3.8 Flash (`low`) spend across all 10 episodes (T1 + T2): **`$10.06`** (`21.47M` input tokens, of which `9.26M` were implicit-cached at `$0.075/Mtok`, and `54.5k` output+thinking tokens). Zero Anthropic runs were executed.

---

## 2. Per-Episode Breakdown — `gemini-3.8-flash` (`low` effort)

### T1 (`datasets/t1-dev-v03` — 10 stations, 3 terminals, 10 trains, 240 ticks; `-C_best` = proven CP-SAT optimum on all 5 seeds)

| Task (`seed`) | **Reward (`-C_agent`)** | `-C_greedy` | `-C_ref` | `-C_best` (`solver`) | Norm Score *(report-only)* | Unclipped *(report-only)* | On-Time % | Pax Delay | Freight % | Red Flags / Breakdown | Calls / DP | Input (Cached) / Out (Think) | USD | Wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|
| `T1-9ebf2853f0d9` (`s0`) | **`-272.18`** | `-59.73` | `-31.69` | `-1.76` | `0.0000` | `-3.66` | `92.3%` | `7,375` | `100%` | `0` (**clean shift**) | `153 / 38` | `1.25M` (`603k`) / `5.9k` (`3.0k`) | `$0.56` | `389s` |
| `T1-733d665b3c40` (`s1`) | **`-121.67`** | `-54.90` | `-13.28` | `-2.08` | `0.0000` | `-1.26` | `80.0%` | `7,826` | `100%` | `0` (**clean shift**) | `203 / 48` | `1.42M` (`591k`) / `5.0k` (`856`) | `$0.69` | `454s` |
| `T1-7e032cce8799` (`s2`) | **`-346.64`** | `-138.97` | `-41.45` | `-12.14` | `0.0000` | `-1.64` | `77.1%` | `13,401` | `100%` | `1` (`1` misroute) | `217 / 41` | `1.58M` (`725k`) / `4.5k` (`165`) | `$0.72` | `467s` |
| `T1-932c950af50d` (`s3`) | **`-29,023.85`** | `-286.24` | `-44.42` | `-1.00` | `0.0000` | `-100.75` | `12.0%` | `0` | `0%` | `53` (`3`-strikes prose stop @ turn 12; `44` missed calls, `9` unfin) | `9 / 2` | `97k` (`52k`) / `166` (`22`) | `$0.04` | `31s` |
| `T1-fcd385fee51f` (`s4`) | **`-27,401.72`** | `-26.21` | `-11.54` | `-0.50` | `0.0000` | `-1064.78` | `23.1%` | `0` | `0%` | `49` (`3`-strikes prose stop @ turn 27; `40` missed calls, `9` unfin) | `21 / 7` | `210k` (`94k`) / `395` (`36`) | `$0.10` | `74s` |

- **What `-cost` immediately reveals on T1:** Under the old clipped `[0, ∞)` normalisation, all 5 T1 episodes looked identical (`0.0000`), even though `s0` (`-272.18`) and `s1` (`-121.67`) were **100% clean shifts** with zero unfinished trains and zero missed calls, whereas `s3` (`-29,023.85`) and `s4` (`-27,401.72`) stopped early on 3 consecutive prose replies. With `reward = -cost`, clean near-greedy shifts (`-121.67`) and catastrophic stops (`-29,023.85`) are separated by two orders of magnitude.

### T2 (`datasets/t2-dev-v03` — 16 stations, 4 terminals, 14 trains, 720 ticks; `-C_best` = omniscient portfolio search)

| Task (`seed`) | **Reward (`-C_agent`)** | `-C_greedy` | `-C_ref` | `-C_best` (`search`) | Norm Score *(report-only)* | Unclipped *(report-only)* | On-Time % | Pax Delay | Freight % | Red Flags / Breakdown | Calls / DP | Input (Cached) / Out (Think) | USD | Wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|
| `T2-5c2f59fff047` (`s1`) | **`-2,257.03`** | `-3,070.22` | `-749.01` | `-705.03` | `0.3438` | `0.3438` | `37.8%` | `126,962` | `100%` | `4` (`4` crew timeouts; `3` missed conn) | `279 / 83` | `2.50M` (`1.02M`) / `6.5k` (`1.3k`) | `$1.21` | `662s` |
| `T2-d1f50d807960` (`s2`) | **`-11,446.95`** | `-1,710.20` | `-872.05` | `-718.19` | `0.0000` | `-9.8152` | `43.8%` | `38,329` | `66.7%` | `18` (`196` refusals -> budget hit @ `599`; `1` crew, `1` misroute, `12` missed calls, `4` unfin) | `599 / 61` | `6.64M` (`3.00M`) / `12.3k` (`914`) | `$3.01` | `1,287s` |
| `T2-537560106741` (`s3`) | **`-2,282.32`** | `-2,827.73` | `-1,929.83` | `-1,760.51` | `0.5111` | `0.5111` | `33.3%` | `106,462` | `100%` | `4` (`4` crew timeouts; `3` missed conn) | `295 / 86` | `2.61M` (`1.03M`) / `6.3k` (`787`) | `$1.28` | `667s` |
| `T2-e1cb078dbc73` (`s7`) | **`-3,362.11`** | `-4,482.68` | `-3,081.90` | `-1,752.63` | `0.4105` | `0.4105` | `27.0%` | `172,673` | `100%` | `6` (`6` crew timeouts) | `338 / 81` | `2.95M` (`1.20M`) / `7.7k` (`1.1k`) | `$1.43` | `785s` |
| `T2-16f531b6931e` (`s8`) | **`-26,306.29`** | `-2,175.61` | `-1,389.81` | `-1,343.70` | `0.0000` | `-29.0065` | `38.8%` | `12,722` | `33.3%` | `39` (`3`-strikes prose stop @ turn 269; `32` missed calls, `7` unfin) | `264 / 45` | `2.20M` (`942k`) / `5.7k` (`358`) | `$1.03` | `571s` |

- **3/5 T2 tasks beat Greedy directly on `-cost`:** `s1` (`-2,257.03` vs `-3,070.22`), `s3` (`-2,282.32` vs `-2,827.73`), and `s7` (`-3,362.11` vs `-4,482.68`, within `280` cost units of Reference's `-3,081.90`), delivering 100% of passenger and freight trains with zero gridlocks and zero misroutes.

---

## 3. Prompt Size & `status(compact=True)` Analysis on T2

### Key Finding: `status(compact=True)` stayed compact (`+49` tokens on T2 vs T1); T2 token growth comes from `timetable()` / `network()` / full `status()` JSON re-reads inside the 5-turn sliding window

| Component / Metric | T1 (`gemini-flash-low`) | T2 (`gemini-flash-low`) | Change (T1 → T2) |
|---|---:|---:|---:|
| **Turn 1 `prompt_tokens` (empty window, `t=0`)** | **`5,088` tok** | **`5,137` tok** | **`+49` tok (`+1.0%`)** |
| Static `system` + `manual` + `tools` chars | `1,163 + 12,534 + 3,791 = 17,488` chars | `17,488` chars | `0` chars |
| **`status(compact=True)` chars (mean / max)** | **`395` / `549` chars** (`~115–160` tok) | **`548` / `1,015` chars** (`~160–290` tok) | **`+153` chars mean (`<2%` of prompt)** |
| `bulletins` chars (max) | `2` chars | `1,157` chars (`~320` tok) | `+1,155` chars (active disruptions) |
| **5-turn `window` chars (mean / max)** | **`7,179` / `22,285` chars** | **`12,367` / `46,004` chars** | **`+5,188` chars (`+72%`)** |
| **`prompt_tokens` at decision start (mean / max)** | **`7,198` / `12,099` tok** | **`8,215` / `17,108` tok** | **`+1,017` tok (`+14%`)** |
| **`prompt_tokens` per LLM turn (mean / p50 / max)** | **`7,330` / `6,685` / `12,177` tok** | **`9,338` / `8,330` / `20,626` tok** | **`+2,008` tok (`+27%`)** |
| LLM turns per decision point (mean) | `4.5` turns/decision | `5.0` turns/decision (`3.5` on clean runs; `9.8` on `s2`) | `+0.5` turns/decision |
| **Cumulative `tokens_in` billed per decision point** | **`31,501` tok/decision** | **`50,136` tok/decision** (`29.8k–36.0k` on clean runs; `107.2k` on `s2`) | **`+59%` (`+9%` on clean runs)** |

1. **Static topology/timetable eviction every 5 turns:** Because `windowed` mode keeps only the last `5` tool calls, Gemini spent **`542 / 1,810` turns (`29.9%` of all T2 turns)** re-calling `timetable()` (`394` calls; up to `12,089` chars ≈ `3,500` tokens each) and `network()` (`148` calls; up to `4,694` chars ≈ `1,350` tokens each).
2. **Unnecessary full `status()` calls:** Gemini called `status()` without `compact=True` **`155` times** on T2 (`12,527` chars ≈ `3,600` tokens of raw JSON per call), even though `status(compact=True)` is already injected free in `CURRENT STATE`.
