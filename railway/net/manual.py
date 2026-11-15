"""Dispatcher manual for the v0.3 railway network (docs/network-v0.3.md)."""

from __future__ import annotations

from . import cost_constants as cc
from .engine import IDLE_MIN, LOOKAHEAD_MIN, TURNAROUND_MIN
from .formats import STRINGLINE_MAX_CHARS, STRINGLINE_WINDOW
from .rules import NetRule
from .trains import TYPES

TOOL_BUDGET = 600
NOTES_MAX = 1500

INSTRUCTION = f"""# Network railway dispatcher shift

You are the signaller and controller for a railway network over one shift. Automatic route
setting runs non-conflicting traffic; you settle conflicts, manage platforms and holds, and
handle disruptions through the tools provided.

1. Read `network()`, `timetable()` and `status(compact=True)` to inspect the layout and booked service.
2. Time advances only when you call `advance_to_next_event()`. Give your regulation commands
   (`order`, `hold`, `platform`, ...) at each decision point, then advance.
3. Reward: minus your total cost. Nothing else counts.

Ending: the shift ends when it reaches `shift_min`, when you call `end_shift()`, when you use up
the tool budget ({TOOL_BUDGET} calls), or when you stop. Stopping without `end_shift()` counts as
calling `end_shift()`.
"""

RULE_DESCRIPTIONS: dict[NetRule, str] = {
    NetRule.UNKNOWN_ELEMENT: "Unknown node, track, platform, move, area, corridor or resource key.",
    NetRule.BAD_ROUTE: "Route has fewer than two elements, repeats an element, or reverses inside a platform.",
    NetRule.NOT_CONNECTED: "Consecutive route elements are not joined at the node, or miss an intervening move.",
    NetRule.WRONG_DIRECTION: "One-way track traversed against its permitted travel direction.",
    NetRule.DIRECTION_LOCKED: "Track is already locked for traffic running in the opposite direction.",
    NetRule.BLOCK_OCCUPIED: "Entry block or required track block is currently held by another train.",
    NetRule.MOVE_LOCKED: "Point move inside a node is currently locked for another train.",
    NetRule.MOVE_CONFLICT: "Requested move shares points or crosses a diamond currently locked for another move.",
    NetRule.PLATFORM_OCCUPIED: "Target or starting platform road is currently occupied by another train.",
    NetRule.PLATFORM_LEN: "Train is longer than a platform or loop road on its path.",
    NetRule.PLATFORM_KIND: "Passenger train cannot terminate or call in a siding.",
    NetRule.NOT_ELECTRIFIED: "Electric train cannot use a track or road without overhead electrification.",
    NetRule.UNKNOWN_TRAIN: "Train identifier does not exist in this shift.",
    NetRule.ALREADY_RUNNING: "Action requires a berthed or offstage train with remaining plan, not running or done.",
    NetRule.HOLD_PAST: "Hold target tick must be strictly greater than the current tick.",
    NetRule.NO_ROUTE: (
        "No admissible physical route connects the train's position and facing to that platform, "
        "or the target platform is not at the station of the train's next booked call."
    ),
    NetRule.SHIFT_ENDED: "The shift has already ended; no further calls or actions can take effect.",
    NetRule.BAD_ARGS: "Unknown tool, disabled tool, or malformed/unexpected arguments.",
    NetRule.BUDGET: "Tool call budget reached; the shift closes immediately.",
    NetRule.NOTES_TOO_LONG: f"Scratchpad text exceeds {NOTES_MAX} characters.",
    NetRule.WINDOW: f"Stringline tick range must lie within [0, shift_min] and span at most {STRINGLINE_WINDOW} min.",
    NetRule.TRAIN_DONE: "Train has already completed its shift or been cancelled.",
    NetRule.NO_CALL: "Train has no remaining booked stop matching that station or action.",
    NetRule.SKIP_DESTINATION: "Cannot skip a terminating train's final stop; use short_turn or cancel instead.",
    NetRule.NO_RETURN: "Train has no return working (turnaround stop) left to short-turn onto.",
    NetRule.NOT_REVERSIBLE: "Track is not signalled for bidirectional single-line working.",
    NetRule.SINGLE_LINE: "Single-line working is already in that state, or a train is routed the wrong way over it.",
    NetRule.NOT_CREW_POINT: "Train is neither at nor booked to call at a crew-relief platform.",
    NetRule.CREW_RELIEF: "A crew relief is already scheduled or in progress for that train.",
    NetRule.POSSESSION: "Track is closed under an engineering possession.",
    NetRule.POINTS_FAILED: "Point move is closed due to a points failure.",
    NetRule.PLATFORM_CLOSED: "Platform road is temporarily closed to traffic.",
}


def _fmt_const(value: object) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items())
    if isinstance(value, tuple):
        return ", ".join(str(x) for x in value)
    return str(value)


def render_manual() -> str:
    """Render the deterministic v0.3 network dispatcher manual."""
    from .session import TOOLS, describe

    train_rows = []
    for code in ("IC", "RE", "S", "FI", "FB"):
        if code not in TYPES:
            continue
        t = TYPES[code]
        traction = "electric" if t.electric else "diesel"
        loads = f"{t.load[0]}–{t.load[1]}" if t.load[1] else "none"
        train_rows.append(
            f"| `{t.code}` ({t.name}) | {t.priority} | {t.length_m} m | {t.max_speed_kmh} km/h "
            f"| {t.restart_min} min | {traction} | {t.reward_class} | {loads} |"
        )

    tool_rows = []
    for name, (model, _) in TOOLS.items():
        fields = []
        for fname, finfo in model.model_fields.items():
            fields.append(fname if finfo.is_required() else f"{fname}?")
        arg_str = ", ".join(f"`{f}`" for f in fields) if fields else "none"
        tool_rows.append(f"| `{name}` | {arg_str} | {describe(name)} |")

    rule_rows = []
    for rule in NetRule:
        desc = RULE_DESCRIPTIONS.get(rule) or rule.value.removeprefix("R-").replace("-", " ").capitalize() + "."
        rule_rows.append(f"| `{rule.value}` | {desc} |")

    steps_formula = " + ".join(f"{step}·[L ≥ {at}]" for at, step in cc.LATE_STEPS)
    unplanned_str = ", ".join(f"{k} {v}" for k, v in cc.UNPLANNED_STOP_COST.items())
    val_express = cc.VALUE["express"]
    val_local = cc.VALUE["local"]

    const_names = (
        "PAX_GRACE",
        "LATE_STEPS",
        "VALUE",
        "MISSED_CALL_FLAT",
        "MISSED_CALL_HASSLE",
        "DEFAULT_HEADWAY",
        "MAX_HEADWAY",
        "UNFINISHED_COST",
        "UNDELIVERED_FREIGHT",
        "W_FREIGHT",
        "FREIGHT_LATE_STEP",
        "PORT_CUTOFF_COST",
        "GRIDLOCK_COST",
        "UNPLANNED_STOP_COST",
        "PLATFORM_CHANGE_MINUTES",
        "TIMEOUT_COST",
    )
    const_rows = "\n".join(f"| `{name}` | `{_fmt_const(getattr(cc, name))}` |" for name in const_names)
    cost_rows = "\n".join(
        [
            "| `pax_lateness` (C1) | At each served passenger stop: `v * (alight / 100) * f(L)` |",
            (
                f"| `missed_calls` (C2) | Unserved stop: `{cc.MISSED_CALL_FLAT} + v * ((alight + board) / 100) * "
                f"(f(H) + {cc.MISSED_CALL_HASSLE})`; missed connection of `n` transfers: "
                f"`v * (n / 100) * (f(H) + {cc.MISSED_CALL_HASSLE})` |"
            ),
            (
                f"| `unfinished` (C2) | Per unfinished train at shift end: `{cc.UNFINISHED_COST}`; "
                f"undelivered freight adds `{cc.UNDELIVERED_FREIGHT}` plus lateness at shift end |"
            ),
            (
                f"| `freight_lateness` (C3) | `{cc.W_FREIGHT} * max(0, L)`, plus `{cc.FREIGHT_LATE_STEP}` "
                f"once past its window; missed port cutoff adds `{cc.PORT_CUTOFF_COST}` |"
            ),
            f"| `crew_timeout` (C4) | `{cc.TIMEOUT_COST}` per crew duty timeout (plus knock-on delays) |",
            f"| `gridlock` (C5) | `{cc.GRIDLOCK_COST}` when a deadlock forms |",
            (
                f"| `operations` (C5) | Unplanned signal stop: `{unplanned_str}`. "
                f"Platform change: `v * ((alight + board) / 100) * {cc.PLATFORM_CHANGE_MINUTES}` |"
            ),
        ]
    )

    return f"""# Network dispatcher manual

You control a busy railway network for one shift. Automatic route setting (ARS) handles
routine uncontested movements; you resolve every conflict, regulate priorities at junctions
and single-track bottlenecks, allocate platforms, and manage connections and disruptions.

**Reward: minus your total cost. Nothing else counts.** (`reward = -(total cost)`.)

## Time, turns and budget

- Time is measured in whole minutes (`tick`, from `0` to `shift_min`).
- Time stands still while you inspect state or issue commands, and advances only when you
  call `advance_to_next_event()`.
- You have a budget of `{TOOL_BUDGET}` tool calls for the shift. Every call through the tool
  interface (read-only, action, refused or invalid) consumes `1` call.
- Reaching `{TOOL_BUDGET}` calls closes the shift (`{NetRule.BUDGET.value}`). Calling `end_shift()`,
  or stopping without calling `end_shift()`, runs the uncontested remainder of the shift and
  finalizes all shift-end charges (`{NetRule.SHIFT_ENDED.value}` refuses any call after closure).

## Network and track kinds

The railway is a directed/undirected multigraph returned by `network(area?)`:

- **Nodes:** `station` (through stations and hubs), `junction` (flat, single-lead or
  grade-separated convergence points), `terminal` (dead-end bay platforms), `yard`,
  `freight_terminal` (reception sidings / ports) and `portal` (entry/exit boundary).
- **Tracks:** physical running lines between two nodes (`a` and `b`), divided into signal
  blocks of `block_m` metres (`blocks = ceil(length_m / block_m)`):
  - **One-way double track (`direction = "ab"` or `"ba"`):** trains follow one block apart
    under automatic block signalling. Entering a track claims only its entry block; blocks
    release behind the train as it steps forward.
  - **Bidirectional / single track (`direction = "both"`):** usable either way; locking a
    train in one direction locks out opposing traffic until the track clears.
- **Platforms and roads (`Platform`):** standing places inside non-junction, non-portal nodes:
  - `platform`: passenger or freight platform face (`length_m`, `electrified`).
  - `through`: unplatformed main/through road for fast pass-through or long freight.
  - `siding`: goods loop, reception road or yard road (`{NetRule.PLATFORM_KIND.value}` bars
    passenger stops in sidings).
  - A train must fit (`length_m`) every platform/through/siding road it traverses, not only
    the one it stops at (`{NetRule.PLATFORM_LEN.value}`). Electric trains require
    `electrified = true` on every track and road (`{NetRule.NOT_ELECTRIFIED.value}`).
- **Moves and crossings (`Move`):** point paths inside a node linking a track and a platform
  (with `side` `"a"` or `"b"`) or two tracks. Moves sharing an endpoint or platform side
  conflict automatically; flat-junction and throat diamond crossings are listed in `crossings`.
  Clearing a move takes `1` min at turnout speeds `>= 80 km/h` and `2` min below `80 km/h`.
- **Turnarounds:** a train entering a platform via side `"a"` cannot leave via side `"a"`
  unless its booked `min_dwell >= {TURNAROUND_MIN}` minutes, allowing the driver to change ends.

## Train types

Published priorities guide default tie-breaking, not the reward. Passenger loads and lateness
steps determine true cost per minute:

| Type | Published priority | Length | Max speed | Restart penalty | Traction | Reward class | Typical load |
|---|---:|---:|---:|---:|---|---|---|
{chr(10).join(train_rows)}

Track run time is `max(blocks, ceil(length_m * 60 / (min(line_speed_kmh, max_speed_kmh) * 1000)))`
minutes. An unplanned signal stop adds `restart_min` minutes when the train resumes.

## Automatic route setting and conflicts

ARS grants authority one section at a time (the moves out to the next track or platform, and
the entry block of that track). When only one train wants a resource within the next
`{LOOKAHEAD_MIN}` minutes, ARS sets its route automatically as soon as the interlocking is free.

Whenever **two or more trains** want a overlapping or conflicting resource within
`{LOOKAHEAD_MIN}` minutes—and no active `order` ranks them—ARS stops instead of guessing and
raises a bundled `CONFLICT`. Resource keys have four forms:

- `<track>:<block>` (e.g. `A-JA.u:0` or `JA-N:0`): contention for a track's entry block.
- `<move1>/<move2>` (e.g. `JA.m2/JA.m3` or `A.e1/A.e3`): conflicting or crossing moves at a node.
- `<platform>` (e.g. `C1`): two trains routed to the same platform road.
- `<move>` (e.g. `JA.m1`): two trains needing the same point move.

You settle contention with:

1. `order(resource, trains)`: list the train IDs in the order they should take `resource`
   (listing the winner first is already enough; unlisted trains keep timetable order behind
   those listed). You may pass an exact conflict key (`JA.m2/JA.m3`, `A-JA.u:0`, `C1`), a
   track ID (`JA-N`), or a node ID (`JA`) to order every move and platform at that node.
2. `hold(train, until)`: hold an offstage or berthed train until `t = until` (e.g. to protect
   a passenger connection or keep a slow train in a loop while a fast service overtakes).
3. `platform(train, station, n)`: re-allocate the train's next booked call to another
   platform `n` at that same station (e.g. `station="C", n=2` or `n="C2"` for a call booked
   at `C`), freeing a busy bay or clearing an `UNROUTABLE` train. It never moves a call to a
   different station: a platform at any other station is refused (`{NetRule.NO_ROUTE.value}`).
4. Service and crew regulation during disruptions: `route(train, via)` diverts a train's next
   leg through ordered track or platform IDs (after `single_line(segment, on=True)` if using a
   reversible track the wrong way); `skip(train, station)` runs past a booked stop;
   `short_turn(train, station)` turns a train back early onto its return leg; `cancel(train)`
   withdraws a service; and `relieve_crew(train)` relieves a crew at a crew-relief platform
   (8 min stop) before duty runs out.

## Decision points

`advance_to_next_event()` runs the network forward and stops at the earliest of:

- `CONFLICT`: one or more unresolved resource contentions within `{LOOKAHEAD_MIN}` minutes
  (all open conflicts are bundled together).
- `DISRUPTION`: a disruption is announced or begins.
- `CREW`: a crew reaches its warning margin (30 min left) or times out (240 min duty limit).
- `UNROUTABLE`: a due train has no valid physical path to its next booked platform (for
  example, booked onto a road shorter than the train); reported once per newly stranded set.
- `GRIDLOCK` / `BLOCKED`: remaining active trains are deadlocked and cannot move; permanent
  gridlocks incur the gridlock charge.
- `IDLE`: `{IDLE_MIN}` minutes elapsed without another trigger, letting you inspect the shift.
- `SHIFT_END`: `tick` reached `shift_min`.

When disruptions, crew duty warnings, or connection risks are active in the shift, `bulletins()`,
`conflicts()` and `advance_to_next_event()` surface them alongside the stop reason.

## Tools

Use area-scoped `network(area)` and `status(area, compact=True)` on large layouts so context
stays compact (`area` accepts a node ID like `"JA"`, a track ID, or a node range like `"PW-C"`).
Every call returns a status in `GRANTED`, `OK`, `REFUSED`, `INVALID` or `INFO`.

| Tool | Arguments | Description |
|---|---|---|
{chr(10).join(tool_rows)}

`stringline(corridor, from_tick, to_tick)` accepts a node range (`"PW-C"`, `"JA-PN"`), a track
ID, or a train ID as `corridor`, with `0 <= from_tick <= to_tick <= shift_min`, window at most
`{STRINGLINE_WINDOW}` minutes, and output capped at `{STRINGLINE_MAX_CHARS}` characters.

## Rules

| Rule ID | Meaning |
|---|---|
{chr(10).join(rule_rows)}

## Cost and reward

One cost unit equals `100` passenger-minutes late. For passenger lateness `L = actual - booked`
minutes at a call, the per-passenger curve has a `{cc.PAX_GRACE}`-minute grace and Delay-Repay
steps:

`f(L) = max(0, L - {cc.PAX_GRACE}) + {steps_formula}`

Value of time `v` is `{val_express}` for `express` (`IC`) and `{val_local}` for `local`
(`RE`, `S`). Headway `H` is the wait until the next service at that stop (default
`{cc.DEFAULT_HEADWAY}` min, capped at `{cc.MAX_HEADWAY}` min).

| Component | Charge |
|---|---|
{cost_rows}

### Cost constants

| Constant | Value |
|---|---|
{const_rows}

### Objective

**Reward: minus your total cost. Nothing else counts.**
The reward is `-(total cost)`. Minimize total cost across the entire shift.
"""
