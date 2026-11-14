from __future__ import annotations

from . import constants
from .rules import RULE_DESCRIPTIONS


def _value(value):
    if isinstance(value, dict):
        return repr(value)
    if isinstance(value, tuple):
        return "–".join(str(part) for part in value)
    return str(value)


def render_manual() -> str:
    c = constants
    train_rows = []
    for name, spec in c.TRAIN_TYPES.items():
        value = c.VALUE.get(name, "–")
        stopping = {
            "express": "major stations",
            "local": "every station",
            "freight": "destination only",
            "engineering": "as scheduled",
        }[name]
        train_rows.append(
            f"| {name} | {spec['length_m']} m | {spec['max_speed_kmh']} km/h | {c.RESTART_TICKS[name]} "
            f"| {value} | {stopping} |"
        )
    constants_rows = "\n".join(
        f"| `{name}` | `{_value(v)}` |" for name, v in vars(c).items() if name.isupper() and not name.startswith("__")
    )
    rule_rows = "\n".join(f"| `{rule.value}` | {description} |" for rule, description in RULE_DESCRIPTIONS.items())
    steps = " + ".join(f"{add}·[L ≥ {at}]" for at, add in c.LATE_STEPS)
    at, step = c.FREIGHT_LATE_STEP
    unplanned = ", ".join(f"{k} {v}" for k, v in c.UNPLANNED_STOP_COST.items())
    return f"""# Dispatcher manual

You dispatch trains on a single-track railway network for one shift. Route every train
from its origin to its destination, stop it where the timetable says, keep it on time,
and never let trains block each other for good.

## Time and turns

One tick is {c.TICK_MINUTES} minute. Play is turn based: actions at the current tick take
effect before time advances. `advance_to_next_event` stops at the earliest trigger or
after {c.TIME_CAP} ticks.

## Network

The network is a graph. Terminals are end points; stations have a main track and
optional numbered loops; a station joined to three sections is a junction. Sections
join two nodes and hold one train at a time. `network()` lists everything, including
loop and terminal-track lengths and kinds and the crew stations.

- A terminal has a fixed number of tracks (P1..P4). A train entering the network takes
  a free fitting track at its origin terminal (its rotation predecessor's track, if it
  has one); if none is free it waits outside, and `ENTRY_BLOCKED` is reported once.
- A train that arrives at a terminal leaves the network there. It holds its track for
  {c.TURNAROUND} ticks, or until its next service departs from it.
- On a two-yard line, the yards YW and YE are terminals with unlimited room and the
  single track `main`.
- Each train has an origin, a destination and a timetabled route (`timetable()`). The
  route is the plan; you may send a train another way, but see `R-OFF-ROUTE`.

## Train types

| Type | Length | Maximum speed | Restart ticks | Value of time | Scheduled stops |
|---|---:|---:|---:|---:|---|
{chr(10).join(train_rows)}

## Movement authority

1. A train moves only under authority. `authorize(train, station, track, via)` reserves
   every section, each intermediate station's main, and the requested destination track;
   sections and mains are released as the train leaves them.
2. The route is the shortest one from where the train stands (or its current authority
   ends) to `station`, never back over the section it arrived by (`R-NO-REVERSING`:
   trains do not reverse). If two shortest routes exist, name stations in `via`, in
   order; otherwise `R-ROUTE-AMBIGUOUS`. No route at all is `R-NO-ROUTE`
   (`R-WRONG-DIRECTION` on a two-yard line).
3. Grant checks run in order: arguments and train state; route; resources in path
   order; destination track occupancy (`R-TERMINAL-FULL` at a terminal), length and
   permitted kind (passenger trains never use goods tracks); then revealed possession
   timing using projected section occupancy.
4. A grant that leaves a scheduled stop, or the destination, out of reach still
   succeeds but says so: `R-OFF-ROUTE`. Reaching a terminal that is not the
   destination ends the train's run there (`R-WRONG-TERMINAL`).
5. Extending authority releases its old destination track, reserves that station's main
   as a pass-through, and means the train does not stop at the old destination.
6. Scheduled stops on the way happen automatically on main. A train dwells at least
   `min_dwell`, waits until `dep`, then continues; at the destination it uses the
   requested track.
7. Departure waits for entry time, scheduled departure, dwell, hold, relief, timeout and
   breakdown timers.
8. Section travel is ceil(length × 60 / (minimum of line, train and active restriction
   speeds × 1000)), at least one tick. Restrictions apply when entering within
   `[start, end)`. F5 adds the type's restart ticks after an unscheduled stop.
9. Passing a station takes zero ticks. A stopped train on main blocks that station.
10. Each tick processes reveals and authority cuts; terminal releases; timers; section
    exits and arrivals; entries; departures; crew time; decision triggers; then
    gridlock. Train-id order resolves ties.
11. Gridlock: a train standing without authority whose every way forward is held by
    trains that are themselves stuck. Trains do not reverse, so a gridlock is permanent.
12. `cancel_authority(train)` truncates authority at the first station not yet passed,
    using its main if the path passes through it. If stationary at the authority origin,
    it removes the whole authority. Without authority it is refused.
13. `hold(train, until_tick)` blocks new authority until that tick; it cannot affect an
    existing authority. A hold at or before the current tick is refused.

## Rules

| Rule ID | Meaning |
|---|---|
{rule_rows}

## Factors

Factors enabled for this shift are listed by `network()`. F1 defines the network. F2
defines train types, lengths and speeds. F3 enforces authority, occupancy and scheduled
stops. F4 computes the costs. F5 adds restart time and an unplanned-stop charge after
a stop with no scheduled call. F6 enforces rotations and crew carryover. F7 reveals
possessions, cuts conflicting authorities and blocks entry during the window. F8
applies a revealed speed restriction on entry. F10 tracks crew duty, warns at
{c.CREW_WARNING} minutes remaining, times out at {c.CREW_LIMIT} minutes (the train stops
for {c.TIMEOUT_BLOCK} ticks), and permits {c.RELIEF_TICKS}-tick relief at crew stations.
F11 adds passenger connections (`timetable()`). F15 stops a train for the announced
breakdown duration. F16 reveals a late entry.

## Events and lookaheads

| Event | Effect | Reveal lookahead |
|---|---|---|
| Possession | Blocks a section for a half-open window; conflicting authorities are cut back | {c.POSSESSION_LOOKAHEAD[0]}–{c.POSSESSION_LOOKAHEAD[1]} ticks |
| Speed restriction | Lowers speed on listed sections for trains entering during the window | {c.RESTRICTION_LOOKAHEAD[0]}–{c.RESTRICTION_LOOKAHEAD[1]} ticks |
| Late entry | Moves entry later than the timetable | {c.LATE_ENTRY_LOOKAHEAD} ticks |
| Breakdown | Stops a train for its announced duration | Revealed when it starts |

## Decision points

`advance_to_next_event` stops when a trigger becomes true: `AUTHORITY_END`, `READY`,
`ENTRY_BLOCKED`, `BULLETIN`, `BREAKDOWN`, `CREW_WARNING`, `CREW_TIMEOUT`, `GRIDLOCK`,
`TIME_CAP`, `SHIFT_END`. They fire on the transition into a condition, not repeatedly.
No trigger fires when a reservation clears: check `status()` before advancing far.
The initial state at tick zero is a decision point.

## Cost

One unit is 100 passenger-minutes late. Lateness `L` = actual − scheduled arrival.
Passenger lateness curve: f(L) = max(0, L − {c.PAX_GRACE}) + {steps}. `v` is the value
of time above; `alight` and `board` are the people getting off and on at a stop
(`timetable()`); `H` is the stop's `headway` (default {c.DEFAULT_HEADWAY}).

| Component | When | Charge |
|---|---|---|
| `pax_lateness` | Arrival at a scheduled passenger stop | v × alight/100 × f(L) |
| `missed_calls` | Shift end, per scheduled passenger stop not served (skipped, misrouted, wrong terminal, not reached, gridlocked) | {c.MISSED_CALL_FLAT} + v × (alight + board)/100 × (f(H) + {c.MISSED_CALL_HASSLE}) |
| `missed_calls` | Shift end, per F11 connection whose connector left before feeder arrival + transfer time | v × n/100 × (f(H) + {c.MISSED_CALL_HASSLE}) |
| `unfinished` | Shift end, per train not at its own destination terminal | {c.UNFINISHED_COST}; freight adds {c.UNDELIVERED_FREIGHT} (undelivered cargo) and its lateness as if arriving at shift end |
| `freight_lateness` | Freight arrival at its destination | {c.W_FREIGHT} × L, plus {step} if L > {at} |
| `crew_timeout` | F10 crew timeout | {c.TIMEOUT_COST} |
| `gridlock` | A new gridlock forms | {c.GRIDLOCK_COST} |
| `operations` | F5 unplanned stop: {unplanned}. Stop at a hub on a track other than its published `platform` | as listed; v × (alight + board)/100 × {c.PLATFORM_CHANGE_MINUTES} |

Total is the sum of the seven components, each charge rounded to six decimals. A
station left without service is never cheaper than a late one.

## Score

Your reward is minus your total cost: every unit of cost above lowers it by one. Nothing else
counts. Reports also show where you land between a simple greedy dispatcher and the best plan
known for this shift.

## Budgets and ending

Each episode has a budget of {c.TOOL_BUDGET} calls; every call, including read-only and
invalid calls, uses one. When the budget runs out the episode ends. Stopping without
`end_shift()` counts as calling `end_shift()` and finalizes the shift-end costs.

## Tools

Every tool name below is also its MCP name. Every call uses one budget unit.

| Tool | Arguments | Output |
|---|---|---|
| `manual` | none | This manual |
| `network` | none | Shift, factors, rules version, terminals, stations, loops, sections, crew stations |
| `timetable` | optional `train` | Routes, stops (with alight, board, headway, platform), rotations, connections |
| `status` | `compact=false` | Full visible state; compact mode prints one line per train |
| `bulletins` | none | Revealed events and reveal ticks |
| `kpis` | none | Cost by component, counters (gridlocks, timeouts, misroutes, ...) and budget |
| `stringline` | `from_tick`, `to_tick`, optional `train` | ASCII time-distance diagram along that train's timetabled route (default: the longest); at most {c.STRINGLINE_WINDOW} ticks and {c.STRINGLINE_MAX_CHARS} characters |
| `notes` | `text` | Replaces notes; maximum {c.NOTES_MAX} characters |
| `authorize` | `train`, `station`, `track`, optional `via` | Granted authority (with any `R-OFF-ROUTE` warning) or a rule refusal |
| `cancel_authority` | `train` | Cancels or truncates authority, or a refusal |
| `hold` | `train`, `until_tick` | Sets a hold or a refusal |
| `relieve_crew` | `train` | Starts crew relief or a refusal |
| `advance_to_next_event` | none | Tick, stop reasons, changes since last advance and remaining budget |
| `end_shift` | none | Runs to shift end and returns final cost |

## Public constants

| Constant | Current value |
|---|---|
{constants_rows}
"""
