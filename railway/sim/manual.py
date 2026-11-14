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
    train_rows = []
    for name, spec in constants.TRAIN_TYPES.items():
        restart = constants.RESTART_TICKS[name]
        weight = {
            "express": constants.W_EXPRESS,
            "local": constants.W_LOCAL,
            "freight": constants.W_FREIGHT,
            "engineering": constants.W_ENGINEERING,
        }[name]
        stopping = {
            "express": "major stations",
            "local": "every station",
            "freight": "destination only",
            "engineering": "as scheduled",
        }[name]
        train_rows.append(
            f"| {name} | {spec['length_m']} m | {spec['max_speed_kmh']} km/h | {restart} | {weight} | {stopping} |"
        )

    constants_rows = "\n".join(
        f"| `{name}` | `{_value(value)}` |"
        for name, value in vars(constants).items()
        if name.isupper() and not name.startswith("__")
    )
    rule_rows = "\n".join(f"| `{rule.value}` | {description} |" for rule, description in RULE_DESCRIPTIONS.items())
    cost_rows = (
        f"| `pax_lateness` | On passenger stop arrival | w × pax/100 × max(0, late − {constants.PAX_GRACE}); "
        f"× {constants.TERMINUS_MULT} at terminus |\n"
        f"| `freight_lateness` | Freight destination arrival | {constants.W_FREIGHT} × max(0, late) |\n"
        f"| `crew_timeout` | Crew timeout | {constants.TIMEOUT_COST} |\n"
        f"| `reversal` | Reverse granted | {constants.REVERSE_COST} |\n"
        f"| `unfinished` | Train unfinished at shift end | {constants.UNFINISHED_COST} plus "
        "destination lateness at shift end |"
    )
    possession_row = (
        f"| Possession | Blocks entry to a section for its half-open time window; conflicting authorities are cut back "
        f"| {constants.POSSESSION_LOOKAHEAD[0]}–{constants.POSSESSION_LOOKAHEAD[1]} ticks |"
    )
    restriction_row = (
        f"| Speed restriction | Lowers speed on listed sections for trains entering during the window "
        f"| {constants.RESTRICTION_LOOKAHEAD[0]}–{constants.RESTRICTION_LOOKAHEAD[1]} ticks |"
    )
    stringline_row = (
        f"| `stringline` | `from_tick`, `to_tick` | ASCII time-distance diagram; at most "
        f"{constants.STRINGLINE_WINDOW} ticks and {constants.STRINGLINE_MAX_CHARS} characters |"
    )
    return f"""# Dispatcher manual

You dispatch trains on a single-track railway. Your goal is to route them safely and
on time. The simulator advances between decision points and reports the visible state.

## Time and turns

One tick is {constants.TICK_MINUTES} minute. Play is turn based: actions at the current
tick take effect before time advances. `advance_to_next_event` stops at the earliest
trigger or after {constants.TIME_CAP} ticks.

## Network

The network has west yard YW, east yard YE, ordered stations, a main track at each
station, optional numbered loops, and sections joining adjacent locations. Loop lengths
and uses are listed by `network()`. Only stations marked as crew stations support crew
relief; yards are not crew stations.

## Train types

| Type | Length | Maximum speed | Restart ticks | Delay weight | Scheduled stops |
|---|---:|---:|---:|---:|---|
{chr(10).join(train_rows)}

## Movement authority

1. A train moves only under authority. An authority reserves every section, each
   intermediate station's main, and the requested destination track; a section and a
   main are released when the train leaves them.
2. Grant checks run in order: arguments and train state; resources in path order;
   destination track occupancy, length and permitted loop kind; then revealed
   possession timing using projected section occupancy.
3. A grant must go strictly forward from the train or its current authority end. Sections
   and station mains must be free or already held by this train.
4. Extending authority releases its old destination track, reserves that station's main
   as a pass-through, and means the train does not stop at the old destination.
5. Scheduled stops inside an authority happen automatically on main. A train dwells at
   least `min_dwell`, waits until `dep`, then continues; at the destination it uses the
   requested track.
6. Departure waits for entry time, scheduled departure, dwell, hold, relief, timeout and
   breakdown timers. The train must have authority and cannot depart while reversing.
7. Section travel is ceil(length × 60 / (minimum of line, train and active restriction
   speeds × 1000)), with a minimum of one tick. Restrictions apply when entering within
   `[start, end)`. F5 adds the type's restart ticks after an unscheduled stop.
8. Passing a station takes zero ticks. A stopped train on main blocks that station.
9. Each tick processes reveals and authority cuts; breakdown and timer changes; section
   exits and arrivals; departures and entries; crew time; decision triggers; then
   gridlock. Train-id order resolves ties. An action at a returned tick takes effect
   before time advances, and already reported edge triggers do not fire again.
10. Gridlock is a strongly connected component of at least two blocked trains in the
    wait-for graph. A blocked train is unfinished, stationary at a station, and has no
    authority. Timers do not affect this structural check.
11. `reverse(train)` is allowed only in gridlock. It moves to the previous station using
    a free, fitting, permitted loop or yard, occupies the section for
    {constants.REVERSE_TICKS} ticks, and costs {constants.REVERSE_COST}.
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

Factors enabled for this shift are listed by `network()`.

F1 defines the track network. F2 defines train types, lengths and speeds. F3 enforces
movement authority, resource occupancy and scheduled stops. F4 computes lateness and
unfinished-train costs. F5 adds restart time after an unscheduled stop. F6 enforces
service rotations and crew carryover. F7 reveals possessions, cuts conflicting
authorities and prevents entry during the active window. F8 applies a revealed speed
restriction when a train enters a covered section. F10 tracks crew duty, warns at
{constants.CREW_WARNING} minutes remaining, times out at {constants.CREW_LIMIT} minutes,
and permits {constants.RELIEF_TICKS}-tick relief at crew stations. F15 stops an
in-network train for the announced breakdown duration. F16 reveals a late entry and
updates its effective entry tick.

## Events and lookaheads

| Event | Effect | Reveal lookahead |
|---|---|---|
{possession_row}
{restriction_row}
| Late entry | Moves effective entry later than the original timetable | {constants.LATE_ENTRY_LOOKAHEAD} ticks |
| Breakdown | Stops an in-network train for its announced duration | Revealed when it starts |

## Decision points

`advance_to_next_event` stops when a trigger becomes true. Triggers are
`AUTHORITY_END`, `READY`, `BULLETIN`, `BREAKDOWN`, `CREW_WARNING`, `CREW_TIMEOUT`,
`GRIDLOCK`, `TIME_CAP`, and `SHIFT_END`. They fire on the transition into a condition,
not repeatedly while it remains true. The initial state at tick zero is a decision point.

## Cost

Lateness is `max(0, actual arrival − scheduled arrival)`; passenger grace is
{constants.PAX_GRACE} ticks. Costs are accumulated on arrival and rounded to six
decimals.

| Component | When charged | Formula |
|---|---|---|
{cost_rows}

The express passenger weight is {constants.W_EXPRESS}; local is {constants.W_LOCAL}.
Passenger lateness also multiplies by pax/100. Total is the sum of the five components.

## Score

Your score compares your total cost with a simple greedy dispatcher and with a strong
reference dispatcher: matching the reference scores 1, doing no better than greedy
scores 0.

## Budgets and ending

Each episode has a budget of {constants.TOOL_BUDGET} calls; every call, including
read-only and invalid calls, uses one. When the budget runs out the episode ends.
Stopping without `end_shift()` counts as calling `end_shift()` and finalizes
unfinished-train costs.

## Tools

Every tool name below is also its MCP name. Every call uses one budget unit.

| Tool | Arguments | Output |
|---|---|---|
| `manual` | none | This markdown manual |
| `network` | none | Shift, factors, rules version, yards, stations, loops, sections and crew stations |
| `timetable` | optional `train` | Original train schedule and any entry bulletins |
| `status` | `compact=false` | Full visible state; compact mode prints one line per train and a restrictions footer |
| `bulletins` | none | Revealed event list and reveal ticks |
| `kpis` | none | Component costs, gridlocks, timeouts, reversals, refusals and budget |
{stringline_row}
| `notes` | `text` | Replaces notes; maximum {constants.NOTES_MAX} characters |
| `authorize` | `train`, `station`, `track` | Granted authority or a rule refusal |
| `cancel_authority` | `train` | Cancels or truncates authority, or a refusal |
| `hold` | `train`, `until_tick` | Sets a hold or a refusal |
| `relieve_crew` | `train` | Starts crew relief or a refusal |
| `reverse` | `train` | Reverses a gridlocked train or a refusal |
| `advance_to_next_event` | none | Tick, stop reasons, changes since last advance and remaining budget |
| `end_shift` | none | Runs to shift end and returns final cost |

Compact train status includes type, position, authority, delay to the next timed stop,
optional crew remaining, and next scheduled stop. `advance_to_next_event` prints
`t=<tick> stop: <REASON> <train>`, then changes, then
`budget: <used>/{constants.TOOL_BUDGET}`.

## Public constants

| Constant | Current value |
|---|---|
{constants_rows}
"""
