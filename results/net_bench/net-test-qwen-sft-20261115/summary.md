# Network benchmark `net-test-qwen-sft-20261115`

reward = -(C1 + C2 + C3 + C4 + C5); CI = 95% seed bootstrap.

| model | tier | seeds | ok/eps | mean -cost [95% CI] | C1 | C2 | C3 | C4 | C5 | on-time % | pax delay min | freight % | clean % | invalid % | refusal % | tok in | tok out | tok reason | tok cached | USD | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen38-think-sft | ALL | 17 | 17/17 | -1135.1 [-1679.0, -680.8] | 476.5 | 631.0 | 4.9 | 0.0 | 22.7 | 69.5 | 43289.0 | 94.1 | 23.5 | 0.22 | 0.87 | 782180 | 133859 | 129838 | 0 | 0.00 | 6921 |
| qwen38-think-sft | T1 | 17 | 17/17 | -1135.1 [-1679.0, -680.8] | 476.5 | 631.0 | 4.9 | 0.0 | 22.7 | 69.5 | 43289.0 | 94.1 | 23.5 | 0.22 | 0.87 | 782180 | 133859 | 129838 | 0 | 0.00 | 6921 |
