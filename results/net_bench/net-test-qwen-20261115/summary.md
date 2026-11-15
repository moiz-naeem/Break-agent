# Network benchmark `net-test-qwen-20261115`

reward = -(C1 + C2 + C3 + C4 + C5); CI = 95% seed bootstrap.

| model | tier | seeds | ok/eps | mean -cost [95% CI] | C1 | C2 | C3 | C4 | C5 | on-time % | pax delay min | freight % | clean % | invalid % | refusal % | tok in | tok out | tok reason | tok cached | USD | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen38-think | ALL | 2 | 2/2 | -21.6 [-29.1, -14.0] | 9.6 | 0.0 | 0.0 | 0.0 | 12.0 | 98.3 | 6207.5 | 100.0 | 100.0 | 0.62 | 0.00 | 800908 | 124230 | 81254 | 0 | 0.00 | 4421 |
| qwen38-think | T1 | 2 | 2/2 | -21.6 [-29.1, -14.0] | 9.6 | 0.0 | 0.0 | 0.0 | 12.0 | 98.3 | 6207.5 | 100.0 | 100.0 | 0.62 | 0.00 | 800908 | 124230 | 81254 | 0 | 0.00 | 4421 |
