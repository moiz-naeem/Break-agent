# Network benchmark `net-test-qwen-sft-20261115`

reward = -(C1 + C2 + C3 + C4 + C5); CI = 95% seed bootstrap.

| model | tier | seeds | ok/eps | mean -cost [95% CI] | C1 | C2 | C3 | C4 | C5 | on-time % | pax delay min | freight % | clean % | invalid % | refusal % | tok in | tok out | tok reason | tok cached | USD | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen38-think-sft | ALL | 22 | 22/22 | -1225.7 [-1637.0, -832.0] | 490.6 | 708.5 | 4.3 | 0.0 | 22.3 | 67.3 | 44901.0 | 93.2 | 18.2 | 0.17 | 0.80 | 776176 | 137351 | 133513 | 0 | 0.00 | 6978 |
| qwen38-think-sft | T1 | 22 | 22/22 | -1225.7 [-1637.0, -832.0] | 490.6 | 708.5 | 4.3 | 0.0 | 22.3 | 67.3 | 44901.0 | 93.2 | 18.2 | 0.17 | 0.80 | 776176 | 137351 | 133513 | 0 | 0.00 | 6978 |
