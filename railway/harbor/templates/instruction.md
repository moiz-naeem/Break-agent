# Railway dispatcher shift

You are the dispatcher of a railway network for one shift: single-track lines with passing
loops, junctions and terminals. You control every train through the tools of the `railway`
MCP server; nothing moves without your authority.

1. Call `manual()` first. It contains every rule, cost formula and tool format.
2. Then read `network()`, `timetable()` and `status()`.
3. Time moves only when you call `advance_to_next_event()`. Give your orders
   (`authorize`, `hold`, `relieve_crew`, ...) and then advance.
4. Goal: the lowest total cost at the end of the shift (passenger lateness, missed station
   calls, unfinished trains and undelivered freight, freight lateness, crew timeouts,
   gridlocks, unplanned stops and platform changes).

Reward: minus your total cost at the end of the shift. Nothing else counts.

Ending: the shift ends when you call `end_shift()`, when you use up the tool budget
({TOOL_BUDGET} calls), or when you stop. Stopping without calling end_shift() counts as
end_shift(): the rest of the shift runs with no new authorities, and every stop not served
and every train not finished is charged.

Everything you need is in the tools. There are no files to read.
