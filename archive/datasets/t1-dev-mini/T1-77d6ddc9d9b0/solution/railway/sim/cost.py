from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .constants import (
    PAX_GRACE,
    TERMINUS_MULT,
    UNFINISHED_COST,
    UNFINISHED_FREIGHT_EXTRA,
    UNFINISHED_FREIGHT_SHIFT,
    W_EXPRESS,
    W_FREIGHT,
    W_LOCAL,
)


class CostBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pax_lateness: float = 0.0
    freight_lateness: float = 0.0
    crew_timeout: float = 0.0
    reversal: float = 0.0
    unfinished: float = 0.0
    total: float = 0.0


def pax_cost(train_type: str, pax: int, late: int, terminus: bool = False) -> float:
    weight = W_EXPRESS if train_type == "express" else W_LOCAL
    return round(weight * pax / 100 * max(0, late - PAX_GRACE) * (TERMINUS_MULT if terminus else 1), 6)


def freight_cost(late: int) -> float:
    return round(W_FREIGHT * max(0, late), 6)


def unfinished_cost(train, shift_ticks: int) -> float:
    final = train.stops[-1]
    late = shift_ticks - final.arr
    if train.type in {"express", "local"}:
        return round(UNFINISHED_COST + pax_cost(train.type, train.pax, late, True), 6)
    if train.type == "freight":
        extra = UNFINISHED_FREIGHT_EXTRA * shift_ticks / UNFINISHED_FREIGHT_SHIFT
        return round(UNFINISHED_COST + extra + freight_cost(late), 6)
    return float(UNFINISHED_COST)


def update_total(cost: CostBreakdown) -> CostBreakdown:
    keys = ("pax_lateness", "freight_lateness", "crew_timeout", "reversal", "unfinished")
    cost.total = round(sum(getattr(cost, k) for k in keys), 6)
    return cost
