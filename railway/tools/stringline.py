"""ASCII time-distance diagram (T05 §2.2). Past rows from session history, future rows from a belief projection."""

from __future__ import annotations

import math

from railway.sim.constants import STRINGLINE_MAX_CHARS, STRINGLINE_WINDOW
from railway.sim.engine import Simulator
from railway.sim.rules import RuleId

YARDS = {"YW", "YE"}


def _place(change) -> str:
    # ENTRY "X enters at YW", ENTER "X enters S1-S2", ARRIVE "X at S2/L1": the place is last.
    return change.text.rsplit(" ", 1)[1].split("/")[0]


def _projection(session, to_tick: int) -> list:
    """Changes after now if every train just follows its current authority: no new actions, no hidden events."""
    belief = Simulator.from_observation(session.sim.static_info(), session.state())
    changes = []
    while not belief.done and belief.tick < to_tick:
        before = belief.tick
        result = belief.advance(max_tick=to_tick)
        changes += [c for c in result.changes if c.tick >= session.sim.tick]
        if result.tick == before and not result.changes:
            break
    return changes


def _occupancy(changes, tick: int) -> dict[str, list[str]]:
    where: dict[str, tuple[int, str, str]] = {}
    for c in changes:
        if c.tick <= tick and c.kind in {"ENTRY", "ENTER", "ARRIVE"}:
            where[c.train] = (c.tick, c.kind, _place(c))
    cells: dict[str, list[str]] = {}
    for train, (at, kind, place) in where.items():
        if kind == "ARRIVE" and place in YARDS and at < tick:
            continue  # finished: it left the network
        cells.setdefault(place, []).append(train)
    return cells


def render(session, from_tick: int, to_tick: int):
    shift = session.scenario.shift_ticks
    if not (0 <= from_tick <= to_tick <= shift and to_tick - from_tick <= STRINGLINE_WINDOW):
        detail = f"need 0 <= from_tick <= to_tick <= {shift} and a window of at most {STRINGLINE_WINDOW} ticks"
        return f"REFUSED {RuleId.WINDOW.value}: {detail}", "REFUSED", RuleId.WINDOW.value
    now = session.sim.tick
    places = ["YW", *[s.id for s in session.scenario.network.stations], "YE"]
    columns = [x for a, b in zip(places, places[1:], strict=False) for x in (a, f"{a}-{b}")] + ["YE"]
    projected = [*session.history, *(_projection(session, to_tick) if to_tick > now else [])]
    header = " " * 6 + "".join(f"{'-' if '-' in c else c:<4}" for c in columns)

    def row(tick: int) -> str:
        cells = _occupancy(session.history if tick <= now else projected, tick)
        text = "".join(
            f"{(min(cells[c]) + ('+' if len(cells[c]) > 1 else '')) if c in cells else '.':<4}" for c in columns
        )
        return f"{'>' if tick == now else ' '}t{tick:<3} {text}".rstrip()

    step = math.ceil((to_tick - from_tick + 1) / 30)
    while True:
        out = "\n".join([header.rstrip(), *(row(t) for t in range(from_tick, to_tick + 1, step))])
        if len(out) <= STRINGLINE_MAX_CHARS or step > to_tick - from_tick:
            return out, "INFO", None
        step += 1
