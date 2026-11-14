# Strength check (T06 §2.7)

First 5 accepted T1 dev seeds (`seeds/T1-dev.json`). `c_offline` is a beam search (beam 16, 4
candidates per train) on the true simulator with full knowledge, seeded with the reference's own
result. `gap_pct = (c_ref - c_offline) / c_ref * 100`. Regenerate with
`uv run python -m railway.ref.offline_search`.

| seed | c_greedy | c_ref | c_offline | gap_pct |
|---|---|---|---|---|
| 0 | 1554.7 | 280.0 | 168.1 | 40.0 |
| 5 | 1150.2 | 180.9 | 180.9 | 0.0 |
| 6 | 4050.9 | 2349.8 | 959.6 | 59.2 |
| 7 | 1802.6 | 1084.5 | 213.9 | 80.3 |
| 10 | 861.9 | 412.2 | 22.5 | 94.5 |

## Interpretation

The full-knowledge search improves on the reference by 54.8% on average, ranging from 0.0% to 94.5%. Part of any gap is information the reference never has (T1 has no events, so here it is mostly search depth). The largest gap (94.5% on seed 10) is above 25%, a red flag: the reference may be too weak to anchor scores, so this goes to the team in the gate record.
