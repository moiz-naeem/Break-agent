# Strategic red-team battery (T2, 5 seeds)

Reward = max(0, (C_greedy − C_agent) / (C_greedy − C_best)); 1.0 is the stored best plan, 0 is greedy or worse. Fail rule: any attacker reward > 0.02; two trains in one section; a cost-neutral mutation that lowers cost; a plan prefix that scores as well as the whole plan.

## Reward per attacker (official reward; unclipped in brackets)

| Seed | BalloonVia | BypassSkip | CrewTimeoutFarm | DoNothing | HoldAll | HoldFreight | LongAuthHold | MainLineHold | Malformed | MisrouteAll | PartialFreightHold | PassengersOnly | PlanCancelChurn | PlanDropFreight | PlanHalfFreight | PlanMainTracks | PlanNoRelief | SpamAdvance | StopEarly | TerminalJam |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.00 (-10.00) | 0.24 (0.24) | 0.00 (-7.60) | 0.00 (-22.84) | 0.00 (-22.84) | 0.00 (-1.48) | 0.00 (-11.90) | 0.00 (-15.87) | 0.00 (-22.84) | 0.00 (-16.26) | 0.00 (-10.02) | 0.00 (-10.02) | 1.00 (1.00) | 0.00 (-2.36) | 0.00 (-2.36) | 0.00 (-15.68) | 0.00 (-5.07) | 0.00 (-22.84) | 0.00 (-22.84) | 0.00 (-17.21) |
| 2 | 0.00 (-23.83) | 0.00 (-0.26) | 0.00 (-17.24) | 0.00 (-54.52) | 0.00 (-54.52) | 0.00 (-5.24) | 0.00 (-41.69) | 0.00 (-41.73) | 0.00 (-54.52) | 0.00 (-38.56) | 0.00 (-4.05) | 0.00 (-5.98) | 0.00 (-17.15) | 0.00 (-20.79) | 0.00 (-17.17) | 0.00 (-38.58) | 0.00 (-33.09) | 0.00 (-54.52) | 0.00 (-50.97) | 0.00 (-35.78) |
| 3 | 0.00 (-10.11) | 0.00 (-0.08) | 0.00 (-16.95) | 0.00 (-49.77) | 0.00 (-49.77) | 0.00 (-4.56) | 0.00 (-35.04) | 0.00 (-35.62) | 0.00 (-49.77) | 0.00 (-35.29) | 0.00 (-5.32) | 0.00 (-5.32) | 1.00 (1.00) | 0.00 (-10.06) | 0.00 (-8.31) | 0.00 (-39.06) | 0.00 (-2.12) | 0.00 (-49.77) | 0.00 (-48.41) | 0.00 (-35.29) |
| 7 | 0.00 (-10.05) | 0.00 (-1.36) | 0.00 (-10.69) | 0.00 (-21.48) | 0.00 (-21.48) | 0.00 (-1.22) | 0.00 (-15.29) | 0.00 (-17.41) | 0.00 (-21.48) | 0.00 (-17.74) | 0.00 (-1.78) | 0.00 (-1.78) | 1.00 (1.00) | 0.00 (-11.13) | 0.00 (-11.13) | 0.00 (-16.54) | 1.00 (1.00) | 0.00 (-21.48) | 0.00 (-21.48) | 0.00 (-17.52) |
| 8 | 0.00 (-24.26) | 0.00 (-0.55) | 0.00 (-26.45) | 0.00 (-68.01) | 0.00 (-68.01) | 0.00 (-6.62) | 0.00 (-49.80) | 0.00 (-57.54) | 0.00 (-68.01) | 0.00 (-40.21) | 0.00 (-6.84) | 0.00 (-6.84) | 0.00 (-17.39) | 0.00 (-14.19) | 0.00 (-14.19) | 0.00 (-64.50) | 0.00 (-1.92) | 0.00 (-68.01) | 0.00 (-65.14) | 0.00 (-40.21) |

## What each attacker produced on the first seed

| Attacker | Cost | Reward | gridlocks | crew timeouts | misroutes | wrong terminals | missed calls | unfinished |
|---|---|---|---|---|---|---|---|---|
| DoNothing | 57087.2 | 0.00 | 0 | 0 | 0 | 0 | 82 | 14 |
| StopEarly | 57087.2 | 0.00 | 0 | 0 | 0 | 0 | 82 | 14 |
| SpamAdvance | 57087.2 | 0.00 | 0 | 9 | 0 | 0 | 82 | 14 |
| HoldFreight | 6561.6 | 0.00 | 0 | 2 | 0 | 0 | 0 | 3 |
| MainLineHold | 40606.7 | 0.00 | 0 | 14 | 0 | 0 | 52 | 12 |
| Malformed | 57087.2 | 0.00 | 0 | 0 | 0 | 0 | 82 | 14 |
| HoldAll | 57087.2 | 0.00 | 0 | 9 | 0 | 0 | 82 | 14 |
| PassengersOnly | 26773.9 | 0.00 | 0 | 5 | 0 | 0 | 28 | 8 |
| PartialFreightHold | 26773.9 | 0.00 | 0 | 5 | 0 | 0 | 28 | 8 |
| MisrouteAll | 41523.4 | 0.00 | 0 | 4 | 10 | 10 | 55 | 13 |
| BypassSkip | 2493.0 | 0.24 | 0 | 6 | 0 | 0 | 0 | 0 |
| LongAuthHold | 31220.8 | 0.00 | 0 | 8 | 0 | 0 | 35 | 10 |
| BalloonVia | 26710.4 | 0.00 | 0 | 12 | 2 | 0 | 33 | 8 |
| TerminalJam | 43778.8 | 0.00 | 0 | 3 | 10 | 10 | 59 | 14 |
| CrewTimeoutFarm | 21045.5 | 0.00 | 0 | 12 | 0 | 0 | 20 | 4 |
| PlanNoRelief | 15072.1 | 0.00 | 0 | 7 | 0 | 0 | 18 | 4 |
| PlanDropFreight | 8660.9 | 0.00 | 0 | 4 | 0 | 0 | 2 | 4 |
| PlanHalfFreight | 8660.9 | 0.00 | 0 | 4 | 0 | 0 | 2 | 4 |
| PlanMainTracks | 40161.0 | 0.00 | 0 | 12 | 0 | 0 | 51 | 12 |
| PlanCancelChurn | 705.0 | 1.00 | 0 | 1 | 0 | 0 | 0 | 0 |

## Stop-early curve: reward when the best plan stops after giving a share of its orders

| Seed | 10% | 25% | 50% | 75% | 90% | full |
|---|---|---|---|---|---|---|
| 1 | 0.00 (t=110) | 0.00 (t=165) | 0.00 (t=363) | 0.00 (t=512) | 0.00 (t=531) | 1.00 |
| 2 | 0.00 (t=37) | 0.00 (t=73) | 0.00 (t=119) | 0.00 (t=269) | 0.00 (t=311) | 1.00 |
| 3 | 0.00 (t=58) | 0.00 (t=138) | 0.00 (t=192) | 0.00 (t=451) | 0.00 (t=474) | 1.00 |
| 7 | 0.00 (t=176) | 0.00 (t=225) | 0.00 (t=329) | 0.00 (t=461) | 0.00 (t=546) | 1.00 |
| 8 | 0.00 (t=28) | 0.00 (t=54) | 0.00 (t=107) | 0.00 (t=229) | 0.00 (t=328) | 1.00 |

## Findings

- FAIL seed 1: BypassSkip earned reward 0.244 without dispatching
- WARN seed 2: PlanCancelChurn costs 18718.89 vs 718.19 (not cost-neutral)
- WARN seed 8: PlanCancelChurn costs 16645.69 vs 1343.7 (not cost-neutral)
