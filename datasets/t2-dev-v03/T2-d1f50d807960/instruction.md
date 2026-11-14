# Railway dispatcher shift

You are the dispatcher of a single-track railway for one shift. You control every train
through the tools of the `railway` MCP server; nothing moves without your authority.

1. Call `manual()` first. It contains every rule, cost formula and tool format.
2. Then read `network()`, `timetable()` and `status()`.
3. Time moves only when you call `advance_to_next_event()`. Give your orders
   (`authorize`, `hold`, `relieve_crew`, ...) and then advance.
4. Goal: the lowest total cost at the end of the shift (passenger and freight lateness,
   crew timeouts, gridlock reversals, unfinished trains).

Scoring: your total cost is compared with a simple greedy dispatcher (score 0) and a
strong reference dispatcher (score 1).

Ending: the shift ends when you call `end_shift()`, when you use up the tool budget
(600 calls), or when you stop. Stopping without calling end_shift() counts as
end_shift(): the rest of the shift runs with no new authorities and unfinished trains
are charged.

Everything you need is in the tools. There are no files to read.
