"""v0.3 cost (docs/design-changes-v0.3.md §3). Every charge is rounded to six decimals."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .constants import (
    DEFAULT_HEADWAY,
    FREIGHT_LATE_STEP,
    LATE_STEPS,
    MISSED_CALL_FLAT,
    MISSED_CALL_HASSLE,
    PAX_GRACE,
    PLATFORM_CHANGE_MINUTES,
    UNDELIVERED_FREIGHT,
    UNFINISHED_COST,
    VALUE,
    W_FREIGHT,
)

COMPONENTS = (
    "pax_lateness",
    "missed_calls",
    "unfinished",
    "freight_lateness",
    "crew_timeout",
    "gridlock",
    "operations",
)


class CostBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pax_lateness: float = 0.0  # C1
    missed_calls: float = 0.0  # C2: unserved stops and missed connections
    unfinished: float = 0.0  # C2: trains that did not finish
    freight_lateness: float = 0.0  # C3
    crew_timeout: float = 0.0  # C4
    gridlock: float = 0.0  # C5
    operations: float = 0.0  # C5: unplanned stops, platform changes
    total: float = 0.0


def lateness_curve(late: int) -> int:
    """f(L) = max(0, L − 2) + 5·[L ≥ 15] + 10·[L ≥ 30] + 20·[L ≥ 60], per passenger."""
    return max(0, late - PAX_GRACE) + sum(step for at, step in LATE_STEPS if late >= at)


def value(train_type: str) -> float:
    return VALUE.get(train_type, 0.0)


def pax_cost(train_type: str, alight: int, late: int) -> float:
    """C1, at a timed stop: v × alight/100 × f(late)."""
    return round(value(train_type) * alight / 100 * lateness_curve(late), 6)


def stranded_cost(train_type: str, people: int, headway: int | None) -> float:
    """Passengers left behind: v × people/100 × (f(H) + hassle)."""
    wait = DEFAULT_HEADWAY if headway is None else headway
    return round(value(train_type) * people / 100 * (lateness_curve(wait) + MISSED_CALL_HASSLE), 6)


def missed_call_cost(train_type: str, alight: int, board: int, headway: int | None) -> float:
    """C2, a scheduled passenger stop not served: 500 + v × (alight + board)/100 × (f(H) + 10)."""
    return round(MISSED_CALL_FLAT + stranded_cost(train_type, alight + board, headway), 6)


def freight_cost(late: int) -> float:
    """C3, at the destination: 0.5 × late, plus 50 if more than 30 late."""
    at, step = FREIGHT_LATE_STEP
    return round(W_FREIGHT * max(0, late) + (step if late > at else 0), 6)


def platform_change_cost(train_type: str, alight: int, board: int) -> float:
    return round(value(train_type) * (alight + board) / 100 * PLATFORM_CHANGE_MINUTES, 6)


def unfinished_cost(train, shift_ticks: int) -> float:
    """The flat charge for a train that did not finish, plus freight lateness as if arriving at shift end.
    A passenger train's unserved stops are charged separately, as missed calls."""
    if train.type == "freight":  # its cargo is not delivered: charged like a missed delivery
        return round(UNFINISHED_COST + UNDELIVERED_FREIGHT + freight_cost(shift_ticks - train.stops[-1].arr), 6)
    return float(UNFINISHED_COST)


def update_total(cost: CostBreakdown) -> CostBreakdown:
    cost.total = round(sum(getattr(cost, k) for k in COMPONENTS), 6)
    return cost
