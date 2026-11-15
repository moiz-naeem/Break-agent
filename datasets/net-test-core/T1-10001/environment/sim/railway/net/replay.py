"""Replay a logged v0.3 network episode through a fresh Session and verify determinism."""

from __future__ import annotations

from railway.util import content_hash

from .manual import TOOL_BUDGET
from .model import Timetable
from .session import FinalResult, Session


class ReplayMismatch(Exception):
    pass


def replay(timetable: Timetable, log: dict, *, budget: int = TOOL_BUDGET) -> FinalResult:
    """Re-run `log["entries"]` on a fresh `Session(timetable)` and check status, rules and final state."""
    if "scenario_hash" in log:
        expected_hash = content_hash(timetable.model_dump(mode="json"))
        if log["scenario_hash"] != expected_hash:
            raise ReplayMismatch(f"timetable hash mismatch: logged {log['scenario_hash']}, got {expected_hash}")
    session = Session(timetable, budget=budget)
    for entry in log["entries"]:
        session.call(entry["tool"], entry["args"])
        got = session.log[-1]
        if (got["tick"], got["status"], got["rule"]) != (entry["tick"], entry["status"], entry["rule"]):
            raise ReplayMismatch(
                f"seq {entry['seq']} {entry['tool']}: logged t={entry['tick']} {entry['status']}/{entry['rule']}, "
                f"replayed t={got['tick']} {got['status']}/{got['rule']}"
            )
    final = session.close()
    if "final_tick" in log and final.tick != log["final_tick"]:
        raise ReplayMismatch(f"final_tick mismatch: logged {log['final_tick']}, replayed {final.tick}")
    if "final_cost" in log and final.cost.total != log["final_cost"]:
        raise ReplayMismatch(f"final_cost mismatch: logged {log['final_cost']}, replayed {final.cost.total}")
    if "final_state" in log and [list(r) for r in final.state] != [list(r) for r in log["final_state"]]:
        raise ReplayMismatch("final_state mismatch between logged episode and replay")
    return final
