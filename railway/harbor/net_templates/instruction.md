# Network railway dispatcher shift

You are the signaller and controller for a railway network over one shift: stations, junctions,
terminals, double-track main lines, bidirectional branches and freight spurs. Automatic route
setting (ARS) runs uncontested movements; you resolve resource conflicts, manage holds and
platforms, and handle disruptions through the tools of the `railway` MCP server.

1. Call `manual()` first. It contains every rule, cost formula and tool argument format.
2. Inspect `network()`, `timetable()` and `status(compact=True)` to understand the layout and booked service.
3. Time advances only when you call `advance_to_next_event()`. Settle resource conflicts and give
   your regulation commands (`order`, `hold`, `platform`, ...) at each decision point, then advance.
4. Goal: minimize total cost across the shift (passenger lateness, missed station calls and
   connections, unfinished trains and undelivered freight, freight lateness, crew timeouts,
   gridlocks, unplanned signal stops and platform changes).

Reward: minus your total cost (`reward = -(total cost)`). Nothing else counts.

Ending: the shift ends when it reaches `shift_min`, when you call `end_shift()`, when you use up
the tool budget ({TOOL_BUDGET} calls), or when you stop. Stopping without calling `end_shift()`
counts as calling `end_shift()`: the uncontested remainder of the shift runs and all shift-end
charges are finalized.

Everything you need is in the tools. There are no files to read. Play the whole shift without
asking for confirmation; nobody will answer questions.
