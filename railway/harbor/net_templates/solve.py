"""Booked-order MCP client for `harbor run -a oracle` on v0.3 railway network tasks.

Connects to the `railway` MCP server over streamable HTTP (`http://sim:8000/mcp`) using only `mcp.Client`
(zero access to `timetable.json`, `token`, or simulator internals), resolves open resource conflicts in
booked schedule order, re-platforms unroutable trains when a valid alternate road exists at that station,
relieves warned crews at crew-relief platforms, and finishes the shift cleanly.

Everything it knows comes from public tool output: `timetable()`, `network()`, `status()`, `bulletins()`
and the text of `advance_to_next_event()`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys

from mcp import Client

MAX_STEPS = 590  # T2/T3 shifts have 100-250+ decision points; the 600-call budget is the real cap
STALL_ADVANCES = 4  # identical advances, with no action taken in between, before giving up on a stuck shift
DEFAULT_WARNING_LEFT = 30  # duty minutes a crew has left when warned, if bulletins() does not say
EXIT_RUN_MIN = 15  # allowance for running off the network after the last booked call

CONFLICT_RE = re.compile(r"^C\d+\s+(\S+)\s+\([^)]+\):\s+(.+)$", re.M)
UNROUTABLE_RE = re.compile(r"stop:\s+UNROUTABLE\s+(.+)$", re.M)
STOP_RE = re.compile(r"^t=(\d+)\s+stop:\s+(\S+)(?:\s+(.+))?$", re.M)
CREW_WARNING_RE = re.compile(r"^\s*t=(\d+)\s+(\S+)\s+crew_warning\b", re.M)
TAG_DUE_RE = re.compile(r"@(\d+)")
ENDED_MARKERS = ("stop: SHIFT_END", "SHIFT ENDED", "BUDGET EXHAUSTED", "R-SHIFT-ENDED")


async def _call(client: Client, tool: str, args: dict) -> str:
    res = await client.call_tool(tool, args)
    return "".join(getattr(c, "text", "") for c in res.content)


def _ended(text: str) -> bool:
    return any(marker in text for marker in ENDED_MARKERS)


async def _status(client: Client) -> dict | None:
    try:
        return json.loads(await _call(client, "status", {}))
    except ValueError:
        return None


async def _replatform_unroutable(client: Client, train_ids: list[str], net_data: dict, tt_by_id: dict) -> bool:
    st = await _status(client)
    if st is None:
        return False
    trains_st = {t["id"]: t for t in st.get("trains", [])}
    plats_by_id = {p["id"]: p for p in net_data.get("platforms", [])}
    changed = False
    for tid in train_ids:
        t_st = trains_st.get(tid)
        t_tt = tt_by_id.get(tid)
        if not t_st or not t_tt or not t_st.get("next_call"):
            continue
        cur_plat_id = t_st["next_call"]["platform"]
        cur_plat = plats_by_id.get(cur_plat_id)
        if not cur_plat:
            continue
        node = cur_plat["node"]
        for cand in net_data.get("platforms", []):
            if cand["node"] != node or cand["id"] == cur_plat_id:
                continue
            if cand["length_m"] < t_tt["length_m"]:
                continue
            if t_tt["electric"] and not cand["electrified"]:
                continue
            if t_tt["passenger"] and cand["kind"] == "siding":
                continue
            out = await _call(client, "platform", {"train": tid, "station": node, "n": cand["id"]})
            if out.startswith("OK"):
                changed = True
                break
    return changed


def crew_warnings(adv: str) -> set[tuple[str, int]]:
    """`(train, tick)` crew warnings listed in the changes of one `advance_to_next_event()` reply.

    A `stop: CREW` header alone is not enough: it also names crews that have just timed out.
    """
    return {(m.group(2), int(m.group(1))) for m in CREW_WARNING_RE.finditer(adv)}


def _remaining_calls(t_st: dict, t_tt: dict) -> list[dict]:
    """The calls a train still makes: its live next call, then the booked calls after it."""
    nxt = t_st.get("next_call")
    if not nxt:
        return []
    calls = t_tt.get("calls", [])
    start = None
    for i, c in enumerate(calls):
        if c.get("arrive") == nxt.get("arrive") and c.get("depart") == nxt.get("depart"):
            start = i
            break
    rest = list(calls[start + 1 :]) if start is not None else []
    return [nxt, *rest]


def _work_ends(t_st: dict, t_tt: dict, now: int) -> int:
    """When the train's booked work is expected to end: its last remaining booked time plus its current delay."""
    times = [t for c in _remaining_calls(t_st, t_tt) for t in (c.get("arrive"), c.get("depart")) if t is not None]
    end = max(times, default=now) + max(0, int(t_st.get("delay") or 0))
    if t_tt.get("exit_node"):
        end += EXIT_RUN_MIN
    return end


def relief_candidates(
    warnings: dict[str, tuple[int, int]], st: dict, tt_by_id: dict, relief_platforms: set[str]
) -> list[str]:
    """Warned trains that can be relieved and need it.

    `warnings` maps train -> (warning tick, duty minutes left). A train can be relieved when it is berthed at a
    crew-relief platform or still calls at one; it needs relief only if its booked work outlasts the crew's duty
    (an 8-minute relief hold for a crew that would finish anyway is pure delay).
    """
    trains_st = {t["id"]: t for t in st.get("trains", [])}
    out = []
    for tid, (tick, left) in sorted(warnings.items()):
        t_st = trains_st.get(tid)
        t_tt = tt_by_id.get(tid, {})
        if not t_st or t_st.get("phase") == "done":
            continue
        if _work_ends(t_st, t_tt, tick) <= tick + left:
            continue
        berthed_at_relief = t_st.get("phase") == "berthed" and t_st.get("position") in relief_platforms
        booked_relief = any(c.get("platform") in relief_platforms for c in _remaining_calls(t_st, t_tt))
        if berthed_at_relief or booked_relief:
            out.append(tid)
    return out


async def _relieve_crews(
    client: Client, new: dict[str, tuple[int, int]], tt_by_id: dict, relief_platforms: set[str]
) -> tuple[int, bool]:
    """Ask for relief for each newly warned crew that needs and can get one; returns (actions, shift ended)."""
    st = await _status(client)
    if st is None:
        return 0, False
    acted = 0
    for tid in relief_candidates(new, st, tt_by_id, relief_platforms):
        out = await _call(client, "relieve_crew", {"train": tid})
        if _ended(out):
            return acted, True
        if out.startswith("OK"):
            acted += 1
    return acted, False


async def play(url: str, max_steps: int = MAX_STEPS, stop_at: int | None = None) -> None:
    """Dispatch the whole shift; with `stop_at`, quit with `end_shift` at the first decision at/after that minute."""
    for attempt in range(15):
        try:
            async with Client(url) as client:
                tt = json.loads(await _call(client, "timetable", {}))
                tt_by_id = {t["id"]: t for t in tt.get("trains", [])}
                entry_by_id = {t["id"]: int(t.get("entry_tick") or 0) for t in tt.get("trains", [])}
                net_data = json.loads(await _call(client, "network", {}))
                relief_platforms = {p["id"] for p in net_data.get("platforms", []) if p.get("crew_relief")}

                seen: set[tuple[str, int]] = set()  # (train, tick) crew warnings already handled
                last_sig: str | None = None
                repeats = 0
                ended = False

                for _ in range(max_steps):
                    adv = await _call(client, "advance_to_next_event", {})
                    if _ended(adv):
                        ended = True
                        break
                    stop = STOP_RE.search(adv)
                    if stop_at is not None and stop and int(stop.group(1)) >= stop_at:
                        break
                    actions = 0

                    # Crew warnings: from the advance text, with each crew's duty left from bulletins().
                    warnings = {key: DEFAULT_WARNING_LEFT for key in crew_warnings(adv)}
                    if relief_platforms and (warnings or (stop and stop.group(2) == "CREW")):
                        try:
                            items = json.loads(await _call(client, "bulletins", {}))
                        except ValueError:
                            items = []
                        for item in items if isinstance(items, list) else []:
                            if item.get("kind") == "CREW_WARNING" and item.get("train"):
                                key = (str(item["train"]), int(item.get("tick") or 0))
                                left = (item.get("detail") or {}).get("remaining_min", DEFAULT_WARNING_LEFT)
                                warnings[key] = int(left)
                    new = {tid: (tick, left) for (tid, tick), left in warnings.items() if (tid, tick) not in seen}
                    seen |= set(warnings)
                    if new:
                        acted, ended = await _relieve_crews(client, new, tt_by_id, relief_platforms)
                        actions += acted
                        if ended:
                            break

                    unroutable = UNROUTABLE_RE.search(adv)
                    if unroutable:
                        who = [x.strip() for x in unroutable.group(1).split(",") if x.strip()]
                        if await _replatform_unroutable(client, who, net_data, tt_by_id):
                            actions += 1

                    for m in CONFLICT_RE.finditer(adv):
                        resource = m.group(1)
                        parts = [p.strip() for p in m.group(2).split("|") if p.strip()]
                        ranked: list[tuple[int, int, str]] = []
                        for part in parts:
                            tid = part.split()[0]
                            due_m = TAG_DUE_RE.search(part)
                            due = int(due_m.group(1)) if due_m else entry_by_id.get(tid, 0)
                            ranked.append((due, entry_by_id.get(tid, 0), tid))
                        ranked.sort()
                        trains = [tid for _, _, tid in ranked]
                        out = await _call(client, "order", {"resource": resource, "trains": trains})
                        if _ended(out):
                            ended = True
                            break
                        actions += 1
                    if ended:
                        break

                    # Several decision points can share one minute; only a run of identical advances with
                    # nothing done in between means the shift is stuck.
                    head = adv.split("changes since", 1)[0].strip()
                    if head == last_sig and actions == 0:
                        repeats += 1
                    else:
                        repeats = 0
                    last_sig = head
                    if repeats + 1 >= STALL_ADVANCES:
                        break

                if not ended:
                    await _call(client, "end_shift", {})
                return
        except OSError:
            if attempt == 14:
                raise
            await asyncio.sleep(1.0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Play a booked-order v0.3 network shift over MCP.")
    p.add_argument("--url", default="http://sim:8000/mcp")
    p.add_argument(
        "--stop-at", type=int, default=None, help="stop-early baseline: end the shift at this minute (default: never)"
    )
    args = p.parse_args(argv)
    asyncio.run(play(args.url, stop_at=args.stop_at))
    return 0


if __name__ == "__main__":
    sys.exit(main())
