"""Replay (T05 §2.5): re-run a logged episode through a fresh Session; the verifier trusts this, not the live cost."""

from __future__ import annotations

from railway.sim.constants import TOOL_BUDGET
from railway.sim.model import Scenario
from railway.tools.session import FinalResult, Session


class ReplayMismatch(Exception):
    pass


def replay(scenario: Scenario, log: dict, *, budget: int = TOOL_BUDGET) -> FinalResult:
    """`log` is the PRD §7.9 episode log (parsed JSON)."""
    session = Session(scenario, budget=budget)
    for entry in log["entries"]:
        session.call(entry["tool"], entry["args"])
        got = session.log[-1]
        if (got["status"], got["rule"]) != (entry["status"], entry["rule"]):
            raise ReplayMismatch(
                f"seq {entry['seq']} {entry['tool']}: logged {entry['status']}/{entry['rule']}, "
                f"replayed {got['status']}/{got['rule']}"
            )
    return session.close()
