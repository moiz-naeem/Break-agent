"""Booked-order MCP client for `harbor run -a oracle` on v0.3 railway network tasks.

Connects to the `railway` MCP server over streamable HTTP (`http://sim:8000/mcp`) using only `mcp.Client`
(zero access to `timetable.json`, `token`, or simulator internals), resolves open resource conflicts in
booked schedule order, re-platforms unroutable trains when a valid alternate road exists at that station,
and finishes the shift cleanly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys

from mcp import Client

CONFLICT_RE = re.compile(r"^C\d+\s+(\S+)\s+\([^)]+\):\s+(.+)$", re.M)
UNROUTABLE_RE = re.compile(r"stop:\s+UNROUTABLE\s+(.+)$", re.M)
TAG_DUE_RE = re.compile(r"@(\d+)")


async def _call(client: Client, tool: str, args: dict) -> str:
    res = await client.call_tool(tool, args)
    return "".join(getattr(c, "text", "") for c in res.content)


async def _replatform_unroutable(client: Client, train_ids: list[str], net_data: dict, tt_by_id: dict) -> bool:
    raw_status = await _call(client, "status", {})
    try:
        st = json.loads(raw_status)
    except ValueError:
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


async def play(url: str, max_steps: int = 240) -> None:
    for attempt in range(15):
        try:
            async with Client(url) as client:
                tt_raw = await _call(client, "timetable", {})
                tt = json.loads(tt_raw)
                tt_by_id = {t["id"]: t for t in tt.get("trains", [])}
                entry_by_id = {t["id"]: int(t.get("entry_tick") or 0) for t in tt.get("trains", [])}
                net_data: dict | None = None

                last_sig: str | None = None
                repeats = 0
                ended = False

                for _ in range(max_steps):
                    adv = await _call(client, "advance_to_next_event", {})
                    if "stop: SHIFT_END" in adv or "SHIFT ENDED" in adv or "BUDGET EXHAUSTED" in adv:
                        ended = True
                        break

                    head = adv.split("changes since", 1)[0].strip()
                    if head == last_sig:
                        repeats += 1
                    else:
                        last_sig = head
                        repeats = 0
                    if repeats >= 2:
                        break

                    unroutable = UNROUTABLE_RE.search(adv)
                    if unroutable:
                        if net_data is None:
                            net_data = json.loads(await _call(client, "network", {}))
                        who = [x.strip() for x in unroutable.group(1).split(",") if x.strip()]
                        await _replatform_unroutable(client, who, net_data, tt_by_id)

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
                        if "BUDGET EXHAUSTED" in out or "R-SHIFT-ENDED" in out:
                            ended = True
                            break
                    if ended:
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
    args = p.parse_args(argv)
    asyncio.run(play(args.url))
    return 0


if __name__ == "__main__":
    sys.exit(main())
