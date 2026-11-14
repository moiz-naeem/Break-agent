RULES_VERSION = "0.2-provisional"
TICK_MINUTES = 1
TOOL_BUDGET = 600  # calibrated (T05-35): greedy needs up to ~370 calls on T2; DESIGN said 400
TIME_CAP = 60
PAX_GRACE = 2
TERMINUS_MULT = 2
W_EXPRESS = 3.0
W_LOCAL = 1.0
W_FREIGHT = 0.5
W_ENGINEERING = 0.0
UNFINISHED_COST = 200
UNFINISHED_FREIGHT_EXTRA = 1000  # per UNFINISHED_FREIGHT_SHIFT ticks of shift; calibrated (T06-19) so holding
UNFINISHED_FREIGHT_SHIFT = 240  # freight costs more than greedy loses: T1 (240) +1000, T2 (720) +3000
RESTART_TICKS = {"express": 1, "local": 1, "freight": 4, "engineering": 1}
TURNAROUND = 10
CREW_LIMIT = 240
CREW_WARNING = 30
RELIEF_TICKS = 8
TIMEOUT_BLOCK = 45
TIMEOUT_COST = 300
REVERSE_TICKS = 15
REVERSE_COST = 50
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
