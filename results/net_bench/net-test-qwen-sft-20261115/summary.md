# Network benchmark `net-test-qwen-sft-20261115`

reward = -(C1 + C2 + C3 + C4 + C5); CI = 95% seed bootstrap.

| model | tier | seeds | ok/eps | mean -cost [95% CI] | C1 | C2 | C3 | C4 | C5 | on-time % | pax delay min | freight % | clean % | invalid % | refusal % | tok in | tok out | tok reason | tok cached | USD | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen38-think-sft | ALL | 21 | 21/21 | -1179.4 [-1639.0, -774.5] | 497.5 | 654.5 | 4.3 | 0.0 | 23.1 | 67.0 | 45390.0 | 92.9 | 19.0 | 0.18 | 0.77 | 781783 | 138165 | 134288 | 0 | 0.00 | 7049 |
| qwen38-think-sft | T1 | 21 | 21/21 | -1179.4 [-1639.0, -774.5] | 497.5 | 654.5 | 4.3 | 0.0 | 23.1 | 67.0 | 45390.0 | 92.9 | 19.0 | 0.18 | 0.77 | 781783 | 138165 | 134288 | 0 | 0.00 | 7049 |
