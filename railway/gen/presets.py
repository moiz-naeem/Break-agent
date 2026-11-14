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

GRAPH_FACTORS = sorted([*ALL_FACTORS, "F11"])
MIX = {"express": 0.25, "local": 0.5, "freight": 0.25}

# v0.3 graph tiers (docs/design-changes-v0.3.md §4.2). Sizes are starting values, calibrated by the seed filter.
GRAPH_PRESETS = {
    "G1": GenConfig(preset="G1", layout="graph", stations=10, branches=1, bypasses=0, trains=10, mix=MIX,
                    factors=CORE_FACTORS, shift_ticks=240, end_margin=60, od_pairs=4),
    "G2": GenConfig(preset="G2", layout="graph", stations=16, branches=2, bypasses=1, port=True, trains=14, mix=MIX,
                    factors=GRAPH_FACTORS, shift_ticks=720, end_margin=120, event_density=0.5, od_pairs=7,
                    connections=3),
    "G3": GenConfig(preset="G3", layout="graph", stations=28, branches=4, bypasses=2, port=True, trains=28, mix=MIX,
                    factors=GRAPH_FACTORS, shift_ticks=720, end_margin=120, event_density=0.5, od_pairs=11,
                    connections=6),
    # Structural holdout: a layout family never used for training (three branches, two bypasses).
    "G2-X": GenConfig(preset="G2-X", layout="graph", stations=24, branches=3, bypasses=2, port=True, trains=16,
                      mix=MIX, factors=GRAPH_FACTORS, shift_ticks=720, end_margin=120, event_density=0.5,
                      od_pairs=8, connections=3),
}  # fmt: skip
PRESETS.update(GRAPH_PRESETS)
