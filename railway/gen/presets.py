from __future__ import annotations

from .config import GenConfig

CORE_FACTORS = ["F1", "F2", "F3", "F4"]
ALL_FACTORS = sorted(["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F10", "F15", "F16"])

PRESETS = {
    "T1": GenConfig(
        preset="T1",
        stations=8,
        trains=8,
        mix={"express": 0.25, "local": 0.5, "freight": 0.25},
        factors=CORE_FACTORS,
        shift_ticks=240,  # calibrated (T05/T06): DESIGN's 120 left greedy unable to finish any seed
        end_margin=60,
    ),
    "T2": GenConfig(
        preset="T2",
        stations=12,
        trains=10,  # calibrated (T05/T06): DESIGN's 14 trains / 360 ticks left greedy unable to finish any seed
        mix={"express": 0.25, "local": 0.5, "freight": 0.25},
        factors=ALL_FACTORS,
        shift_ticks=720,
        end_margin=120,
        event_density=0.5,
    ),
    "T2-S14": GenConfig(
        preset="T2-S14",
        stations=14,
        trains=16,
        mix={"express": 0.25, "local": 0.5, "freight": 0.25},
        factors=ALL_FACTORS,
        shift_ticks=360,
    ),
    "T3": GenConfig(
        preset="T3",
        stations=16,
        trains=22,
        mix={"express": 0.25, "local": 0.5, "freight": 0.25},
        factors=ALL_FACTORS,
        shift_ticks=720,
    ),
}
