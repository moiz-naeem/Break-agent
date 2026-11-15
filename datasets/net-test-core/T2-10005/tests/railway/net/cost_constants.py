"""v0.3 cost weights for the network engine (DESIGN.md §6.6 as amended in docs/design-changes-v0.3.md §6).
1 unit = 100 passenger-minutes late. The reward is minus the total cost."""

PAX_GRACE = 2
LATE_STEPS = ((15, 5), (30, 10), (60, 20))  # f(L) = max(0, L - 2) + 5[L>=15] + 10[L>=30] + 20[L>=60]
VALUE = {"express": 1.5, "local": 1.0}  # value of time per passenger, by reward class
MISSED_CALL_FLAT = 500
MISSED_CALL_HASSLE = 10
DEFAULT_HEADWAY = 60
MAX_HEADWAY = 120
UNFINISHED_COST = 200
UNDELIVERED_FREIGHT = 1500
W_FREIGHT = 0.5
FREIGHT_LATE_STEP = 50  # added once freight is later than its window
PORT_CUTOFF_COST = 400
GRIDLOCK_COST = 1000
UNPLANNED_STOP_COST = {"passenger": 1, "freight": 4}
PLATFORM_CHANGE_MINUTES = 2
TIMEOUT_COST = 40
