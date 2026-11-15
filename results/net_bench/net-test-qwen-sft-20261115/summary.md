# Network benchmark `net-test-qwen-sft-20261115`

reward = -(C1 + C2 + C3 + C4 + C5); CI = 95% seed bootstrap.

| model | tier | seeds | ok/eps | mean -cost [95% CI] | C1 | C2 | C3 | C4 | C5 | on-time % | pax delay min | freight % | clean % | invalid % | refusal % | tok in | tok out | tok reason | tok cached | USD | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen38-think-sft | ALL | 13 | 13/13 | -1145.3 [-1763.7, -632.2] | 482.6 | 631.4 | 6.0 | 0.0 | 25.3 | 66.5 | 43740.2 | 92.3 | 15.4 | 0.09 | 0.94 | 791335 | 136941 | 132875 | 0 | 0.00 | 7108 |
| qwen38-think-sft | T1 | 13 | 13/13 | -1145.3 [-1763.7, -632.2] | 482.6 | 631.4 | 6.0 | 0.0 | 25.3 | 66.5 | 43740.2 | 92.3 | 15.4 | 0.09 | 0.94 | 791335 | 136941 | 132875 | 0 | 0.00 | 7108 |
