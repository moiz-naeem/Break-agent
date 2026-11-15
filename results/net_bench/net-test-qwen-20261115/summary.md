# Network benchmark `net-test-qwen-20261115`

reward = -(C1 + C2 + C3 + C4 + C5); CI = 95% seed bootstrap.

| model | tier | seeds | ok/eps | mean -cost [95% CI] | C1 | C2 | C3 | C4 | C5 | on-time % | pax delay min | freight % | clean % | invalid % | refusal % | tok in | tok out | tok reason | tok cached | USD | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen38-think | ALL | 4 | 5/5 | -950.7 [-1879.8, -21.6] | 196.3 | 868.6 | 2.0 | 0.0 | 15.5 | 87.5 | 22979.4 | 90.0 | 40.0 | 0.24 | 2.17 | 837745 | 158499 | 138990 | 0 | 0.00 | 4891 |
| qwen38-think | T1 | 4 | 5/5 | -950.7 [-1879.8, -21.6] | 196.3 | 868.6 | 2.0 | 0.0 | 15.5 | 87.5 | 22979.4 | 90.0 | 40.0 | 0.24 | 2.17 | 837745 | 158499 | 138990 | 0 | 0.00 | 4891 |
