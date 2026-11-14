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
        shift_ticks=120,
    ),
    "T2": GenConfig(
        preset="T2",
        stations=12,
        trains=14,
        mix={"express": 0.25, "local": 0.5, "freight": 0.25},
        factors=ALL_FACTORS,
        shift_ticks=360,
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

# T09 factor ablation: T2 with the four core factors plus exactly one other.
for _factor in ("F5", "F6", "F7", "F8", "F10", "F15", "F16"):
    _name = f"T2-only-{_factor}"
    PRESETS[_name] = PRESETS["T2"].model_copy(update={"preset": _name, "factors": sorted([*CORE_FACTORS, _factor])})
