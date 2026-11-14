RULES_VERSION = "0.1-provisional"
TICK_MINUTES = 1
TOOL_BUDGET = 400
TIME_CAP = 60
PAX_GRACE = 2
TERMINUS_MULT = 2
W_EXPRESS = 3.0
W_LOCAL = 1.0
W_FREIGHT = 0.5
W_ENGINEERING = 0.0
UNFINISHED_COST = 200
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
