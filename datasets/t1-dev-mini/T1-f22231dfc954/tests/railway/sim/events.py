from __future__ import annotations

from collections.abc import Iterable


def due_events(events: Iterable, revealed: set[str], tick: int):
    """Return events that become public at or before this tick in stable order."""
    return sorted(
        (event for event in events if event.id not in revealed and event.reveal_tick <= tick),
        key=lambda event: (event.reveal_tick, event.id),
    )


def overlaps_window(intervals, section: str, start: int, end: int) -> bool:
    """Whether projected half-open section occupancy overlaps a possession window."""
    return any(
        occupied_section == section and occupied_start < end and occupied_end > start
        for occupied_section, occupied_start, occupied_end in intervals
    )
