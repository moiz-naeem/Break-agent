RULES_VERSION = "0.3-provisional"
TICK_MINUTES = 1
TOOL_BUDGET = 600  # calibrated (T05-35): greedy needs up to ~370 calls on T2; DESIGN said 400
TIME_CAP = 60
# Cost (docs/design-changes-v0.3.md §3, DESIGN.md §6.6). 1 unit = 100 passenger-minutes late.
PAX_GRACE = 2
LATE_STEPS = ((15, 5), (30, 10), (60, 20))  # Delay Repay style steps added to the lateness curve f(L)
VALUE = {"express": 1.5, "local": 1.0}  # value of time per passenger, by train type
MISSED_CALL_FLAT = 500  # per scheduled passenger stop not served
MISSED_CALL_HASSLE = 10  # minutes added to the wait for the next service
DEFAULT_HEADWAY = 60  # ticks to the next service, when a stop does not state its own
UNFINISHED_COST = 200
UNDELIVERED_FREIGHT = 1500  # a freight train not delivered by shift end: the cargo's missed delivery
W_FREIGHT = 0.5
FREIGHT_LATE_STEP = (30, 50)  # freight more than 30 ticks late pays 50 more
GRIDLOCK_COST = 1000
UNPLANNED_STOP_COST = {"express": 1, "local": 1, "freight": 4, "engineering": 0}
PLATFORM_CHANGE_MINUTES = 2
TIMEOUT_COST = 40  # v0.3 scale: 300 made crew charges 12-49% of a T2 episode's cost (2026-11-14)
RESTART_TICKS = {"express": 1, "local": 1, "freight": 4, "engineering": 1}
TURNAROUND = 10
CREW_LIMIT = 240
CREW_WARNING = 30
RELIEF_TICKS = 8
TIMEOUT_BLOCK = 45
POSSESSION_LOOKAHEAD = (45, 60)
RESTRICTION_LOOKAHEAD = (30, 60)
LATE_ENTRY_LOOKAHEAD = 20
NOTES_MAX = 1500
STRINGLINE_WINDOW = 60
STRINGLINE_MAX_CHARS = 2000
TRAIN_TYPES = {
    "express": {"length_m": 200, "max_speed_kmh": 140},
    "local": {"length_m": 120, "max_speed_kmh": 100},
    "freight": {"length_m": 600, "max_speed_kmh": 70},
    "engineering": {"length_m": 100, "max_speed_kmh": 60},
}
