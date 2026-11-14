# Strength check (T06 §2.7)

First 5 accepted T1 dev seeds (`seeds/T1-dev.json`). `c_offline` is a beam search (beam 16, 4
candidates per train) on the true simulator with full knowledge, seeded with the reference's own
result. `gap_pct = (c_ref - c_offline) / c_ref * 100`. Regenerate with
`uv run python -m railway.ref.offline_search`.

| seed | c_greedy | c_ref | c_offline | gap_pct |
|---|---|---|---|---|
| 0 | 1746.7 | 296.7 | 176.6 | 40.5 |
| 5 | 1408.2 | 208.5 | 208.5 | 0.0 |
| 6 | 4809.3 | 2620.7 | 1155.2 | 55.9 |
| 7 | 2106.2 | 1219.3 | 213.9 | 82.5 |
| 10 | 1026.3 | 447.0 | 22.5 | 95.0 |

## Interpretation

The full-knowledge search improves on the reference by 54.8% on average, ranging from 0.0% to 95.0%. Part of any gap is information the reference never has (T1 has no events, so here it is mostly search depth). The largest gap (95.0% on seed 10) is above 25%, a red flag: the reference may be too weak to anchor scores, so this goes to the team in the gate record.
