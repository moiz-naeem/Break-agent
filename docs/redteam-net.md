# Network Track Red-Team Findings (`v0.3` / `net`)

Reward on the network track is **`-(total cost)`** (`railway/net/cost.py`, `railway/harbor/net_verify.py`), with no greedy or reference normaliser (`docs/decisions.md` #2, #4). A reward hack on this track is therefore either:

1. **Tool / accounting mismatch:** a legal MCP sequence that lowers billed cost without actually delivering the railway service the cost function prices; or
2. **Scoring-path tamper:** altering `/logs/verifier/reward.txt` or the episode log outside honest play.

## Reproducing everything here

```bash
# Fast regression tests (3 passing tests pinning the F1 fix + battery invariants)
pytest tests/net/test_redteam.py

# Local probes on individual tiers/seeds
python -m railway.redteam.net_battery --tier T1 --seeds 1,3,4,5,7
python -m railway.redteam.net_teleport --tier T1 --seeds 1,3,4,5,7 --workers 8

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
- `tests/net/test_redteam.py` (all 3 regression tests passing)

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

## Open container check: post-verifier sleeper process (`V5`)

In exported Harbor tasks (`railway/harbor/net_export.py`), the agent container contains no simulator, `timetable.json`, or bearer `token`, runs as non-root `user = "agent"` (UID 1000, `main.Dockerfile`), and `clear_rewards()` removes any pre-written `reward.txt`/`reward.json` before verification starts.

What has **not** been tested end-to-end in a live Docker/Harbor run on this branch is whether a background process spawned by UID 1000 (`nohup sh -c 'sleep ...; echo 0 > /logs/verifier/reward.txt' &`) survives Harbor's transition from agent phase to `tests/test.sh` *and* has write permission on `/logs/verifier` after `verify.py` writes `reward.txt`. Before freezing the exported dataset on Linux/WSL2:

1. Run a live Harbor sleeper probe (`harbor run`) to confirm `/logs/verifier` is root-owned (`0755` or `0700`) or PID-isolated across phases; or
2. Harden `railway/harbor/net_templates/test.sh` defensively with `mkdir -p /logs/verifier && chmod 0700 /logs/verifier` (since `test.sh` runs as root while the agent runs as UID 1000).
