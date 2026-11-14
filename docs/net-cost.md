# v0.3 Network Cost (`railway/net/cost.py`)

The RL reward on the v0.3 network track is **`-cost(timetable, events).total`** (`1 unit = 100 passenger-minutes late`).
`cost()` and `itemise()` are pure functions of `(timetable, events)` over `railway/net/events.py`, with weights in `railway/net/cost_constants.py` (`DESIGN.md` §6.6 as amended by `docs/design-changes-v0.3.md` §3 and §6). Every individual charge is rounded to 6 decimal places; component subtotals and `total` are rounded again to 6 decimals.

## Train classes and auxiliary functions

| Type code (`docs/network-v0.3.md` §3) | Reward class | Value `v` | Freight window `W` | Unplanned stop |
|---|---|---|---|---|
| `IC` (Intercity) | `express` | `1.5` | — | `1` |
| `RE` (Regional express), `S` (Commuter) | `local` | `1.0` | — | `1` |
| `FI` (Intermodal freight) | `freight` | — | `30` min | `4` |
| `FB` (Bulk freight) | `freight` | — | `60` min | `4` |
| `ECS` (Empty stock), `ENG` (Engineering) | `none` | `0` | — | `0` |

- **Passenger lateness curve:** `f(L) = max(0, L - 2) + 5[L >= 15] + 10[L >= 30] + 20[L >= 60]`.
- **Headway `H`:** minutes in the booked timetable until the next passenger service calling at the same node and continuing the same way (sharing an onward stop/portal before a turnaround, or an earlier stop/portal at a terminus), capped at `120`, or `60` when no later service goes that way.

## Components, triggers, and formulas

| Component | §6.6 | Triggering `Event` | Formula |
|---|---|---|---|
| `pax_lateness` | C1 | `ARRIVE` with `call` set on a passenger train (`L = tick - booked`) | `v * (alight / 100) * f(L)` |
| `missed_calls` | C2 | `CALL_MISSED` with `call` set on a passenger train (once per `(train, call)`) | `500 + v * ((alight + board) / 100) * (f(H) + 10)` |
| `missed_calls` | C2 | `CONNECTION` with `detail["made"] == False` and a passenger connector | `v_conn * (transfers / 100) * (f(H_conn) + 10)` |
| `unfinished` | C2 | `UNFINISHED` (every train, including `ECS` and `ENG`) | `200`, plus for undelivered freight `1500 + 0.5 * max(0, L_end) + 50[L_end > W]` |
| `freight_lateness` | C3 | `ARRIVE` at a freight trip's terminating call (no `exit_node`), or `EXIT` for freight with `exit_node` | `0.5 * max(0, L) + 50[L > W]`, plus `400` if a port cutoff `t_cut` applies and is missed (`tick > t_cut`, or `tick >= t_cut` on `UNFINISHED`) |
| `crew_timeout` | C4 | `CREW_TIMEOUT` | `40` |
| `gridlock` | C5 | `GRIDLOCK` | `1000` |
| `operations` | C5 | `SIGNAL_STOP` (passenger or freight); `ARRIVE` on a booked passenger call with a changed `detail["booked_platform"]` | `SIGNAL_STOP`: `1` (passenger) or `4` (freight); platform change: `v * ((alight + board) / 100) * 2` |

`ENTER`, `DEPART`, `PASS`, `CREW_WARNING`, `CREW_RELIEF`, `DISRUPTION_START`, `DISRUPTION_END`, `ACTION`, and `SHIFT_END` incur no direct charge (`DEPART` only records the freight last-call departure delay as a fallback when `EXIT.booked` is omitted).

## Worked example

Consider a 180-minute shift on `trunk_and_branches` with three services:

- **`IC01`** (`IC`, `v = 1.5`): calls at `A1` (`arr=10, dep=12`, `alight=100, board=50`), `B1` (`arr=30, dep=32`, `alight=200, board=40`), `C1` (`arr=50`, `alight=300`).
- **`S01`** (`S`, `v = 1.0`): calls at `A2` (`arr=20, dep=21`), `B2` (`arr=40, dep=41`, `alight=30, board=20`), `C2` (`arr=60`, `alight=80`), with a connection `IC01 -> S01` at `B` (`transfers = 50`). No service follows `S01`, so `H(S01, B) = 60` (`f(60) = 93`).
- **`FB01`** (`FB`, `W = 60`): booked to terminate at `P1` (`PORT`) at `arr = 70`.

Suppose the shift produces these billable events:

1. `t=28`, `SIGNAL_STOP` (`IC01` on `JA-B.u`): passenger unplanned stop -> **`operations += 1.0`**.
2. `t=41`, `CONNECTION` (`node="B"`, `feeder="IC01"`, `connector="S01"`, `transfers=50`, `made=False`): `S01` leaves `B` before delayed `IC01` arrives -> `1.0 * (50 / 100) * (93 + 10) = 51.5` -> **`missed_calls += 51.5`**.
3. `t=50`, `ARRIVE` (`IC01` at `B2`, `call=1`, `booked=30`, `detail={"booked_platform": "B1"}`):
   - `L = 20`, `f(20) = 18 + 5 = 23` -> `1.5 * (200 / 100) * 23 = 69.0` -> **`pax_lateness += 69.0`**.
   - Re-platformed at `B2` -> `1.5 * ((200 + 40) / 100) * 2 = 7.2` -> **`operations += 7.2`**.
4. `t=60`, `CALL_MISSED` (`IC01`, `call=2` at `C`, `why="short_turn"`): `S01` arrives at `C` from `B` at `60`, so `H = 60 - 50 = 10`, `f(10) = 8` -> `500 + 1.5 * (300 / 100) * (8 + 10) = 581.0` -> **`missed_calls += 581.0`**.
5. `t=135`, `ARRIVE` (`FB01` at `P1`, `call=1`, `booked=70`): `L = 65 > 60` -> `0.5 * 65 + 50 = 82.5` -> **`freight_lateness += 82.5`**.
6. `t=180`, `UNFINISHED` (`IC01`, `freight=False`): short-turned before `C1` -> **`unfinished += 200.0`**.

Resulting breakdown: `pax_lateness = 69.0`, `missed_calls = 632.5`, `unfinished = 200.0`, `freight_lateness = 82.5`, `crew_timeout = 0.0`, `gridlock = 0.0`, `operations = 8.2`, **`total = 992.2`**, **reward = `-992.2`**.
