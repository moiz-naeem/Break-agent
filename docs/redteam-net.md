# Network Track Red-Team Findings (`v0.3` / `net`)

Reward on the network track is **`-(total cost)`** (`railway/net/cost.py`, `railway/harbor/net_verify.py`), with no greedy or reference normaliser (`docs/decisions.md` #2, #4). A reward hack on this track is therefore either:

1. **Tool / accounting mismatch:** a legal MCP sequence that lowers billed cost without actually delivering the railway service the cost function prices; or
2. **Scoring-path tamper:** altering `/logs/verifier/reward.txt` or the episode log outside honest play.

## Reproducing everything here

```bash
# Fast regression tests (F1 fix, battery invariants, action-tool audit)
pytest tests/net/test_redteam.py

# Local probes on individual tiers/seeds
python -m railway.redteam.net_battery --tier T1 --seeds 1,3,4,5,7
python -m railway.redteam.net_teleport --tier T1 --seeds 1,3,4,5,7 --workers 8

# Action-tool loophole audit (route, short_turn, skip, cancel, single_line), all 15 dev seeds, local
python -m railway.redteam.net_actions_audit --out results/redteam-net/actions-audit-dev.json

# Full 15-seed dev suite (T1, T2, T3) on Modal -> results/redteam-net/{battery,teleport}-dev.json
modal run infra/modal/net_redteam.py
```

Committed artifacts from the 15-seed Modal run: `results/redteam-net/battery-dev.json` and `results/redteam-net/teleport-dev.json`.

---

## Finding F1 (closed on `redteam-net`): `platform()` served a booked stop at another station without a missed-call charge

### Mechanism

The dispatcher manual (`railway/net/manual.py:234`) defines `platform(train, station, n)` as re-allocating a train's next booked call to platform `n` at `station`. `Session` resolves `(station, n)` to a platform ID and calls `Engine.platform(train_id, platform_id)` (`railway/net/engine.py:406-425`), which checks:

- the train is live and not currently `running` or `done`;
- `platform_id != mv.position` and `self.net.element(platform_id)` is a `Platform`;
- `self._chain(...)` finds a valid route from `mv.position` (via `mv.via`) to `platform_id`.

It **never checks** that `platform_id` belongs to the station of `mv.plan[0].platform`. `mv.plan[0]` is overwritten with the foreign platform while `mv.plan_calls[0]` keeps the original booked call index. When the train berths at that foreign station:

1. `Engine` emits `ARRIVE` with the original `call` index, `booked` arrival tick, and `detail={"booked_platform": <original>}`.
2. `cost.py` bills `ARRIVE` as a completed passenger stop (lateness `L = tick - booked`, zero if early/on time) plus the small platform-change operational fee `v * ((alight + board) / 100) * 2` (`operations`), **never** `CALL_MISSED` (`500 + v * ((alight + board) / 100) * (f(H) + 10)`).

Unlike `skip()` and `route()`, which explicitly drop unreachable stations and emit `CALL_MISSED`, cross-station `platform()` lets a train skip congested stations, hubs, or possessions altogether while receiving full credit for serving the stop.

### Paired measurement across T1, T2, and T3 dev seeds

`railway/redteam/net_teleport.py` runs the exact same one-step lookahead search under two option sets:

- **`control` (same-station):** candidate platforms for `mv.plan[0]` must satisfy `p.node == booked_node` (intended tool semantics).
- **`exploit` (cross-station):** candidate platforms may be at any station in `net.platforms` (current engine semantics).

At each decision point, each candidate is rolled out to shift end under `booked_order` regulation and kept only if it strictly lowers final total cost. T1 and T2 use `k=3, horizon=None`; T3 uses the lighter `k=1, horizon=30` (1 nearest candidate per train due within 30 minutes). Because the search algorithm and rollout regulator are identical, `gain = control - exploit` isolates the value of cross-station re-platforming.

| Scenario | `booked_order` | `control` (same-station) | `exploit` (any station) | `gain` (`control - exploit`) | Change vs `control` | Calls served at wrong station |
|---|---:|---:|---:|---:|---:|---:|
| `T1-1` | 4591.23 | 3318.41 | **1511.59** | **+1806.82** | **−54.4%** | 2 |
| `T1-3` | 3391.37 | 443.05 | 443.05 | 0.00 | 0.0% | 0 |
| `T1-4` | 180.54 | 180.54 | 180.54 | 0.00 | 0.0% | 0 |
| `T1-5` | 367.59 | 365.33 | 365.33 | 0.00 | 0.0% | 0 |
| `T1-7` | 1446.54 | 648.73 | 648.73 | 0.00 | 0.0% | 0 |
| `T2-2` | 7101.88 | 2792.62 | **2232.44** | **+560.19** | **−20.1%** | 3 |
| `T2-3` | 10779.78 | 5119.35 | **5062.65** | **+56.70** | **−1.1%** | 1 |
| `T2-4` | 7568.75 | 6091.93 | **4337.80** | **+1754.13** | **−28.8%** | 6 |
| `T2-5` | 4086.09 | 1951.04 | **1238.27** | **+712.77** | **−36.5%** | 4 |
| `T2-6` | 2966.84 | 2151.19 | **1836.79** | **+314.40** | **−14.6%** | 3 |
| `T3-1` | 6763.48 | 5860.58 | **4260.62** | **+1599.96** | **−27.3%** | 5 |
| `T3-2` | 15636.22 | 9292.71 | 9469.01 | −176.30 | +1.9% | 10 |
| `T3-3` | 5020.15 | 4404.10 | **4311.39** | **+92.72** | **−2.1%** | 2 |
| `T3-5` | 8390.53 | 6012.78 | **5162.28** | **+850.51** | **−14.1%** | 5 |
| `T3-6` | 16342.50 | 13693.76 | **6729.33** | **+6964.43** | **−50.9%** | 4 |

**Summary by tier:**

- **T1 (`1/5` dev seeds exploited):** On `T1-1`, `SE04` serves its `A` stop at `B1` (`t=116`) and its `N` branch stop at terminal `C1` (`t=139`), never entering the `JA-N` branch; cost drops from `3318.41` to `1511.59` (`−54.4%`, zero missed calls).
- **T2 (`5/5` dev seeds exploited, mean gain `679.64` / `−18.8%`):** Every T2 hub-and-branches seed is exploited. Hub calls booked at `H` (or terminal `TE` / station `W`) are served at adjacent corridor stations `E1`–`E3` or `W1`–`W3` (e.g. on `T2-4`, 6 calls across `IC01`, `IC02`, `IC03`, `IX03`, and `SW11` are served at foreign stations, cutting cost from `6091.93` to `4337.80`, `−28.8%`).
- **T3 (`4/5` dev seeds exploited even with `k=1, horizon=30`, mean gain `1866.26` / `−23.8%`):** T3's bypass station `BYP` and twin hubs `HA`/`HB` make cross-station substitution systematic: stopping services booked at `M` (`SM02`–`SM10`) re-platform their `M` call onto `BYP1`, and intercity/regional services (`IT01`, `ID01`–`ID09`, `IC07`) serve `HA`/`HB` hub calls at `M1`, `EB2`, or `WA1`. On `T3-6`, 4 cross-station calls cut `missed_calls` from `11` (`8251.07`) to `1` (`1216.14`), `unfinished` from `9` to `5`, and total cost from `13693.76` to `6729.33` (`−50.9%`, saving `6964.43`). (`T3-2`'s `−176.30` reflects greedy myopia: an early greedy cross-station move alters downstream traffic; because `control`'s action set is a subset of `exploit`'s, a planning agent is never worse off.)

### Fix applied (`Engine.platform`)

`Engine.platform` (`railway/net/engine.py:418`) now enforces that the target platform belongs to the station of `mv.plan[0].platform`, returning `R-NO-ROUTE` otherwise:

```diff
--- a/railway/net/engine.py
+++ b/railway/net/engine.py
@@ -415,6 +415,10 @@ class Engine:
             return self._act("platform", args, Refusal(NetRule.NO_ROUTE, f"{train_id} already stands in {platform_id}"))
         if not isinstance(self.net.element(platform_id), Platform):
             return self._act("platform", args, Refusal(NetRule.NO_ROUTE, f"{platform_id} is not a platform"))
+        booked_node = self._node(mv.plan[0].platform)
+        if self._node(platform_id) != booked_node:
+            refusal = Refusal(NetRule.NO_ROUTE, f"{platform_id} is not at {booked_node}, where {train_id} calls next")
+            return self._act("platform", args, refusal)
         stops = [(v, False) for v in mv.via] + [(platform_id, False)]
```

The two unit tests that previously moved stranded `IC9` (260 m, booked into 200 m `N1` at `N`) across stations to `C1` now rescue `IC9` onto the 750 m passing loop `N2` at the same station `N` (matching `railway/harbor/net_templates/solve.py`) and assert that `"C1"` is refused with `R-NO-ROUTE`:

- `tests/net/test_engine.py:178` (`test_re_platforming_frees_a_stranded_train`)
- `tests/net/test_session.py:254` (`test_conflicts_order_hold_platform_and_changes_only`)
- `tests/net/test_redteam.py` (F1 regression tests)

**Rulebook aligned (no trick wording).** The tool schema (`describe("platform")`, from the `_platform` docstring in
`railway/net/session.py`) now reads "Re-platform `train`'s next booked call to another platform `n` at that same
station (never across stations)". The manual (`railway/net/manual.py`) says the same and that a platform at any
other station is refused with `R-NO-ROUTE`, and the `R-NO-ROUTE` rule text names the wrong-station case. These are
text changes only: the `scenario_hash` of every T1/T2/T3 dev and test seed in `seeds/net-*.json` is unchanged
(re-derived from `build_timetable` and compared: 15/15 dev and 90/90 test seeds match).

---

## Checked and not exploitable (`results/redteam-net/battery-dev.json`)

Every check below ran across all 15 dev seeds (`T1`: `1,3,4,5,7`; `T2`: `2,3,4,5,6`; `T3`: `1,2,3,5,6`):

| Seed | `booked_order` | `no_relief` | `hold_all_freight` | `advance_only` | `stop_early` (`t=30`) | `cancel_all` | `do_nothing` | Floor / `booked` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `T1-1` | 4591.23 | 4591.23 | 6072.89 | 29750.69 | 24961.02 | 30932.41 | 29750.69 | 5.44× |
| `T1-3` | 3391.37 | 3391.37 | 4816.82 | 30252.63 | 23331.58 | 30252.63 | 30252.63 | 6.88× |
| `T1-4` | 180.54 | 180.54 | 2670.77 | 25713.13 | 22174.73 | 26895.53 | 25713.13 | 122.82× |
| `T1-5` | 367.59 | 367.59 | 3936.13 | 28234.48 | 22633.02 | 29048.96 | 28234.48 | 61.57× |
| `T1-7` | 1446.54 | 1446.54 | 3178.34 | 26354.27 | 21817.47 | 27715.13 | 26354.27 | 15.08× |
| `T2-2` | 7101.88 | 7101.88 | 7893.16 | 117063.30 | 109727.13 | 117063.30 | 117063.30 | 15.45× |
| `T2-3` | 10779.78 | 10779.78 | 11010.49 | 117132.21 | 111035.77 | 120367.77 | 117132.21 | 10.30× |
| `T2-4` | 7568.75 | 7568.75 | 9357.32 | 115405.80 | 109867.89 | 115405.80 | 115405.80 | 14.52× |
| `T2-5` | 4086.09 | 4086.09 | 4801.62 | 110545.69 | 105901.57 | 112645.58 | 110545.69 | 25.92× |
| `T2-6` | 2966.84 | 2966.84 | 5464.46 | 121522.02 | 108774.41 | 121522.02 | 121522.02 | 36.66× |
| `T3-1` | 6763.48 | 9615.82 | 16251.17 | 281598.72 | 272001.12 | 288886.67 | 281598.72 | 40.22× |
| `T3-2` | 15636.22 | 20530.18 | 19580.47 | 296883.12 | 272559.52 | 302557.95 | 296883.12 | 17.43× |
| `T3-3` | 5020.15 | 6105.25 | 14421.12 | 283030.55 | 253678.11 | 286658.73 | 283030.55 | 50.53× |
| `T3-5` | 8390.53 | 9230.51 | 12722.13 | 307709.99 | 289490.90 | 311767.89 | 307709.99 | 34.50× |
| `T3-6` | 16342.50 | 20174.38 | 19280.84 | 281462.48 | 254554.22 | 285064.93 | 281462.48 | 15.58× |

- **Non-dispatching policies (`do_nothing`, `stop_early`, `cancel_all`, `advance_only`):** Cost **5.4× to 122.8×** more than `booked_order` on every seed (`advance_only` stalls as soon as ARS hits an unresolved conflict, paying full missed-call and unfinished-train penalties).
- **Freight parking (`hold_all_freight`) and ignoring crew warnings (`no_relief`):** `hold_all_freight` loses on all 15 seeds (`+1500` undelivered freight + `200` unfinished per held freight train). `no_relief` triggers crew timeouts and `+840` to `+4894` extra cost on all 5 T3 seeds (360-min shift vs 240-min `CREW_LIMIT_MIN`).
- **Train-type classification (`cost.classify` vs `formats.train_type_code`):** `0` mismatches across all trips on all 15 seeds.
- **Connection billing:** Individual missed-connection charges across the 15 `booked_order` runs range from `0.52` to `88.07` (`v_conn * (transfers / 100) * (f(H_conn) + 10)`), never negative and small relative to `500+` missed-call penalties.
- **Verifier & replay tamper resistance (`railway/harbor/net_verify.py`, `railway/net/replay.py`):** The verifier pulls `/episode_log` from the `sim` sidecar over bearer auth, checks `scenario_hash`, `scenario_id`, and `rules_version`, replays every tool call in a fresh `Session`, recomputes `cost(timetable, event_log)` from the replayed `EventLog`, verifies component-level agreement within `1e-6`, unlinks pre-existing `reward.txt` and `reward.json`, and writes `reward.txt` last. Replaying `booked_order` matches to `< 1e-6` and rejects a forged `final_cost: 0.0` on all `15/15` seeds.

---

## Action tools audited and not exploitable (`results/redteam-net/actions-audit-dev.json`)

F1 was a tool that let a booked call disappear without its bill. `railway/redteam/net_actions_audit.py` asks
the same question of every other service-regulation tool (`route`, `short_turn`, `skip`, `cancel`,
`single_line`) on all 15 dev seeds, at shift start and again at mid-shift, run locally. **Result: 0
violations.**

### In plain English: why the action tools cannot be used to cheat

- **Every stop you drop is paid for, at once.** Whenever `route`, `skip`, `short_turn` or `cancel` is granted,
  the booked calls that leave the train's plan are exactly the calls the engine logs `CALL_MISSED` for, at that
  moment (call conservation). A missed passenger call costs `500 + v x (alight + board)/100 x (f(H) + 10)`,
  whatever the reason. Checked on 15,354 granted diversions (20,713 dropped calls), 82 short turns, 433 skips
  and 1,589 cancels: no exceptions.
- **A stop only counts where it is booked.** Over every episode the audit played, no `ARRIVE` for call `j`
  ever happened at a station other than call `j`'s (the F1 property, now checked for all tools).
- **Passengers changing trains are paid for too.** In 994 connections whose feeder call was dropped while the
  connecting train still left, the missed connection was billed every time. In the 578 cases where the
  connecting train also skipped that call, its own missed call bills its boarders.
- **`route` obeys physics and cannot hide freight.** All 128,876 one-element diversions (every live train x
  every track and platform) were tried: 114,785 were refused with `R-NO-ROUTE`, none through an element the
  train cannot use (no wires for an electric train, a road shorter than the train). The planned path of every
  granted diversion respects electrification, road length and track direction. A freight diversion never
  drops the freight's destination (579 granted, 0 dropped). An undelivered freight still costs `1500 + 200` +
  lateness at shift end.
- **`short_turn` only works where a train really turns.** On all 1,360 attempts against trains with no
  turnaround left, it was refused with `R-NO-RETURN`. Granted turns bill every stop between the turn and the
  turnaround.
- **`single_line` cannot be abused.** Each of the 60 reversible-track openings was followed by routing every
  live train over the track (1,263 granted, 602 the wrong way). Every wrong-way path stays physically valid,
  and closing the line again under a wrong-way train is refused with `R-SINGLE-LINE`.
- **Withdrawing service never beats dispatching.** The four avoidance shortcuts below cost more than honest
  dispatch on **every seed where they did anything** (dispatch = the cheapest of three regulators that never
  withdraw a train or a call: `booked_order`; `pax_first`, which orders freight behind passenger trains; and
  `freight_hold_sweep`, which holds freight back at shift start and still delivers it, late).

| Seed | dispatch | `booked_order` | `pax_first` | `freight_hold_sweep` | `cancel_all_freight` | `cancel_delayed_pax` | `short_turn_all_possible` | `skip_all_intermediate` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `T1-1` | **3173.92** | 4591.23 | 4591.23 | 3173.92 | 6487.92 | 7328.45 | 4221.88 | 5184.93 |
| `T1-3` | **1154.74** | 3391.36 | 2077.01 | 1154.74 | 3141.92 | 4866.83 | no-op | no-op |
| `T1-4` | **180.54** | 180.54 | 180.54 | 180.54 | 2670.77 | no-op | no-op | no-op |
| `T1-5` | **367.59** | 367.59 | 2864.43 | 367.59 | 2144.63 | 1650.88 | 738.78 | 965.80 |
| `T1-7` | **1414.34** | 1446.54 | 1417.34 | 1414.34 | 3178.34 | 7028.90 | no-op | no-op |
| `T2-2` | **5803.97** | 7101.88 | 7101.88 | 5803.97 | 7893.15 | 20846.06 | 15329.42 | 19981.35 |
| `T2-3` | **6689.53** | 10779.77 | 10779.77 | 6689.53 | 8162.34 | 20609.51 | 17220.76 | 25155.84 |
| `T2-4` | **6759.37** | 7568.75 | 7638.33 | 6759.37 | 7382.03 | 19790.38 | 12434.01 | 22168.27 |
| `T2-5` | **3104.01** | 4086.09 | 3104.01 | 3104.01 | 4197.77 | 5690.65 | 6544.94 | 17345.21 |
| `T2-6` | **2931.98** | 2966.84 | 2966.84 | 2931.98 | 3960.95 | 9006.97 | 11087.69 | 17800.99 |
| `T3-1` | **6704.40** | 6763.48 | 6704.40 | 6704.40 | 7386.54 | 30757.78 | 30024.44 | 96777.27 |
| `T3-2` | **15146.75** | 15636.22 | 15636.22 | 15146.75 | 17734.67 | 62426.06 | 53163.42 | 109811.49 |
| `T3-3` | **5020.15** | 5020.15 | 5030.83 | 5020.15 | 6718.65 | 34072.00 | 30405.03 | 83900.04 |
| `T3-5` | **8385.53** | 8390.53 | 12909.72 | 8385.53 | 8727.92 | 30931.69 | 53177.27 | 117410.05 |
| `T3-6` | **11182.57** | 16342.50 | 15771.65 | 11182.57 | 16999.78 | 31840.42 | 49128.14 | 95723.84 |

"no-op": the policy found nothing to withdraw on that seed (no train 15+ min late, no turnaround or no skippable
intermediate stop), so it ran as plain dispatch. Each avoidance policy is layered on both `booked_order` and
`pax_first`, and the cheaper of the two is shown.

| Mean over the 15 dev seeds | Mean cost | Mean ratio to dispatch | Worse than dispatch on |
|---|---:|---:|---|
| dispatch (best honest regulator) | 5201.29 | 1.00 | |
| `cancel_all_freight` | 7119.16 | 2.68 | 15/15 seeds (lowest ratio 1.04) |
| `cancel_delayed_pax` (cancel any passenger train 15+ min late) | 19135.14 | 3.57 | 14/14 seeds where it acted |
| `short_turn_all_possible` | 19143.38 | 2.99 | 12/12 seeds where it acted |
| `skip_all_intermediate` | 41060.00 | 6.08 | 12/12 seeds where it acted |

**What a weaker baseline showed.** Compared with plain `booked_order` alone, two shortcuts did come out
cheaper on a few seeds: `cancel_all_freight` on `T1-3`, `T2-3` and `T2-4`, and `short_turn_all_possible` on
`T1-1`. Neither was an accounting hole. Cancelling the freight was billed in full (e.g. `FT01` on `T2-3`:
`200` unfinished + `1500` undelivered + lateness = `1833`). Short-turning `RE01` on `T1-1` traded a billed
`638.02` missed call at `C` for the `798.70` missed call and `200` unfinished it would have paid at shift end
anyway. They won only because `booked_order` sends freight straight into the passenger peak. Holding the same
freight back and delivering it late (`freight_hold_sweep`) beats cancelling it on every seed (`T2-3`: `6689.53`
vs `8162.34`), so withdrawing a service is a priced, legitimate regulation decision that a competent dispatcher
outperforms, not a free shortcut. Policy `skip_all_intermediate` is mostly a no-op on T1, whose services are
largely two-call trips with a turnaround in between.

Regression tests: `tests/net/test_redteam.py` (call conservation, physics and billing on `T1-1`; the `RE01`
short-turn bill; `R-NO-RETURN`; freight and late-passenger cancellation losing to dispatch on `T1-3`).

---

## V5 (closed): post-agent sleeper processes cannot touch `reward.txt`

In Harbor Docker runs the agent runs as UID 1000 (`agent`) in the main container, and `tests/test.sh` runs
afterwards as root **in the same container**. A hostile agent could leave
`(sleep 5 && echo 0.0 > /logs/verifier/reward.txt) &` behind and exit.

`railway/harbor/net_templates/test.sh` now, when it runs as root:

1. kills every process of UID 1000 (`pkill -9 -u agent` when available, `kill(-1, SIGKILL)` issued as UID
   1000 through `setpriv`, and a `/proc/*/status` scan as fallback; `python:3.11-slim` ships no `pkill`),
   repeating until none is left;
2. deletes anything in `/logs/verifier` not owned by root (e.g. a planted `.reward.txt.tmp` symlink that would
   redirect `verify.py`'s atomic write), then makes the directory `root:root 0755`;
3. runs `verify.py`, and on success leaves its outputs `root:root 0644`.

Run as any other user (the local unit tests), it only runs `verify.py`.

**Checked live** in `python:3.11-slim` (the main image's base) with a hostile agent that left a `setsid`
sleeper, a `nohup` rewrite loop and the symlink: after `test.sh`, no UID-1000 process remained,
`/etc/passwd` was untouched, `reward.txt` still held the verifier's value 5 s later, and a later write as
`agent` failed with `Permission denied`.

## V6 (closed): a crashed agent CLI was scored as a do-nothing shift

In live Harbor runs, an agent CLI that died at once (API key, credit, quota or rate-limit error) made zero MCP
calls; `net_verify.py` then closed the untouched session and wrote a do-nothing reward (e.g. `-28240.34`) as if
the model had chosen to do nothing. `net_verify.py` now raises `INFRA:` (no reward, Harbor records a trial error
to rerun) when the sim logged **zero** entries **and** `/logs/agent` shows a fatal CLI/API failure: HTTP
401/402/403/429, `insufficient_quota`, exhausted credits or credit balance, a bad API key,
`AuthenticationError` / `PermissionDeniedError` / `RateLimitError`, an unhandled exception or a Python
traceback. `harbor run -a nop` (no or empty agent logs) and an agent that stops early still get a valid
`-do_nothing_cost` reward, and any episode with at least one logged call is scored normally whatever its logs
say (`tests/net/test_harbor_net.py`, section 5b).
