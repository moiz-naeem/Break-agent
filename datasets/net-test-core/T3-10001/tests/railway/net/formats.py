"""Formatting, area/corridor scoping and resource-key resolution for v0.3 network tools."""

from __future__ import annotations

import copy
import math
import re
from collections import deque
from typing import TYPE_CHECKING

from railway.util import canonical_json

from .engine import IDLE_MIN, LOOKAHEAD_MIN, Conflict, Decision, Engine, Key, Movement
from .interlocking import Refusal
from .model import Move, Network, Platform, Timetable, Track, TrainSpec, Trip
from .rules import NetRule
from .timetable import legs
from .trains import TYPES

if TYPE_CHECKING:
    from .session import CostBreakdown, Session

NET_RULES_VERSION = "0.3-net"
STRINGLINE_WINDOW = 60
STRINGLINE_MAX_CHARS = 2000

_TYPE_BY_SIG = {(t.passenger, t.length_m, t.max_speed_kmh): t.code for t in TYPES.values()}


def train_type_code(spec: TrainSpec) -> str:
    """Short type code (IC, RE, S, FI, FB) from a TrainSpec, with PAX/FRT fallback for custom test specs."""
    for prefix in ("IC", "RE", "S", "FI", "FB"):
        if spec.id.startswith(prefix) and prefix in TYPES and TYPES[prefix].passenger == spec.passenger:
            return prefix
    return _TYPE_BY_SIG.get((spec.passenger, spec.length_m, spec.max_speed_kmh), "PAX" if spec.passenger else "FRT")


def onboard_pax(mv: Movement) -> int:
    """Passengers currently on board (or boarding at origin when offstage)."""
    if not mv.train.passenger or not mv.trip.calls:
        return 0
    done = len(mv.trip.calls) - len(mv.plan)
    if done <= 0:
        first = mv.trip.calls[0]
        return first.alight if first.alight > 0 else (first.board if mv.phase == "offstage" else 0)
    return max(0, sum(c.board - c.alight for c in mv.trip.calls[:done]))


# ------------------------------------------------------------------ resource & platform resolution


def format_key(key: Key) -> str:
    """Canonical string form of an Engine conflict/order key."""
    if isinstance(key, str):
        return key
    if isinstance(key[1], int):
        return f"{key[0]}:{key[1]}"
    a, b = sorted((str(key[0]), str(key[1])))
    return f"{a}/{b}"


def key_label(net: Network, key: Key) -> str:
    """Short human label for a conflict key: block, single, platform, move, crossing or points."""
    if isinstance(key, tuple):
        if isinstance(key[1], int):
            el = net.element(key[0])
            return "single" if isinstance(el, Track) and el.direction == "both" else "block"
        pair = (key[0], key[1])
        kind = "crossing" if pair in net.crossings or (pair[1], pair[0]) in net.crossings else "points"
        m = net.element(key[0])
        return f"{kind} at {m.node}" if isinstance(m, Move) else kind
    el = net.element(key)
    if isinstance(el, Platform):
        return f"{el.kind} at {el.node}"
    if isinstance(el, Move):
        return f"move at {el.node}"
    return "resource"


def key_nodes(net: Network, key: Key) -> set[str]:
    """Nodes associated with a conflict/resource key, used for area filtering."""
    if isinstance(key, tuple):
        if isinstance(key[1], int):
            el = net.element(key[0])
            return {el.a, el.b} if isinstance(el, Track) else set()
        m = net.element(key[0])
        return {m.node} if isinstance(m, Move) else set()
    el = net.element(key)
    if isinstance(el, Track):
        return {el.a, el.b}
    if isinstance(el, Platform | Move):
        return {el.node}
    return set()


def resolve_order_keys(net: Network, resource: str) -> list[Key] | Refusal:
    """Parse a tool `resource` string into one or more Engine Keys, or refuse with R-UNKNOWN-ELEMENT."""
    if not resource:
        return Refusal(NetRule.UNKNOWN_ELEMENT, "empty resource")
    if ":" in resource:
        track_id, _, raw_idx = resource.partition(":")
        el = net.element(track_id)
        if isinstance(el, Track) and raw_idx.isdigit() and 0 <= int(raw_idx) < el.blocks:
            return [(track_id, int(raw_idx))]
        return Refusal(NetRule.UNKNOWN_ELEMENT, f"no block resource {resource!r}")
    if "/" in resource:
        parts = resource.split("/")
        if len(parts) == 2:
            m1, m2 = net.element(parts[0]), net.element(parts[1])
            if isinstance(m1, Move) and isinstance(m2, Move) and m2.id in net.conflicts(m1.id):
                return [tuple(sorted((m1.id, m2.id)))]
        return Refusal(NetRule.UNKNOWN_ELEMENT, f"no conflicting move pair {resource!r}")
    el = net.element(resource)
    if isinstance(el, Platform | Move):
        return [el.id]
    if isinstance(el, Track):
        return [el.id, *((el.id, b) for b in range(el.blocks))]
    node = net.node(resource)
    if node is not None:
        keys: list[Key] = [p.id for p in net.platforms if p.node == resource]
        moves = [m.id for m in net.moves if m.node == resource]
        keys += moves
        keys += sorted({tuple(sorted((m, c))) for m in moves for c in net.conflicts(m)})
        if not keys:
            tracks = [t for t in net.tracks if resource in (t.a, t.b)]
            keys = [k for t in tracks for k in (t.id, *((t.id, b) for b in range(t.blocks)))]
        if keys:
            return keys
    return Refusal(NetRule.UNKNOWN_ELEMENT, f"no resource {resource!r}")


def resolve_platform(net: Network, station: str, n: int | str) -> str | Refusal:
    """Resolve `(station, n)` to a platform id at `station`."""
    if net.node(station) is None:
        return Refusal(NetRule.UNKNOWN_ELEMENT, f"no station {station!r}")
    plats = [p for p in net.platforms if p.node == station]
    raw = str(n).strip()
    if any(p.id == raw for p in plats):
        return raw
    for candidate in (f"{station}{raw}", f"{station}.{raw}"):
        if any(p.id == candidate for p in plats):
            return candidate
    if raw:
        alpha_matches = [p.id for p in plats if p.id.endswith(raw) and p.id[: -len(raw)].isalpha()]
        if len(alpha_matches) == 1:
            return alpha_matches[0]
    return Refusal(NetRule.NO_ROUTE, f"no platform {raw!r} at {station}")


# ------------------------------------------------------------------ area & corridor scoping


def _bfs_nodes(net: Network, start: str, goal: str) -> list[str] | None:
    if start == goal:
        return [start] if net.node(start) is not None else None
    if net.node(start) is None or net.node(goal) is None:
        return None
    adj: dict[str, list[str]] = {n.id: [] for n in net.nodes}
    for t in sorted(net.tracks, key=lambda x: x.id):
        if t.b not in adj[t.a]:
            adj[t.a].append(t.b)
        if t.a not in adj[t.b]:
            adj[t.b].append(t.a)
    paths: dict[str, list[str]] = {start: [start]}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == goal:
            return paths[cur]
        for nxt in adj[cur]:
            if nxt not in paths:
                paths[nxt] = [*paths[cur], nxt]
                queue.append(nxt)
    return None


def _chain_waypoints(net: Network, waypoints: list[str]) -> list[str] | None:
    if len(waypoints) < 2 or any(net.node(w) is None for w in waypoints):
        return None
    full = [waypoints[0]]
    for a, b in zip(waypoints, waypoints[1:], strict=False):
        seg = _bfs_nodes(net, a, b)
        if seg is None or len(seg) < 2:
            return None
        full.extend(seg[1:])
    return full


def resolve_corridor(timetable: Timetable, corridor: str) -> list[str] | Refusal:
    """Resolve a corridor string (node range `PW-C`, waypoints, track id or train id) to ordered nodes."""
    net = timetable.net
    raw = corridor.strip()
    if not raw:
        return Refusal(NetRule.UNKNOWN_ELEMENT, "empty corridor")
    el = net.element(raw)
    if isinstance(el, Track):
        return [el.a, el.b]
    trip_by_id = {t.train.id: t for t in timetable.trips}
    if raw in trip_by_id:
        trip = trip_by_id[raw]
        trip_legs = legs(net, trip)
        if trip_legs:
            nodes: list[str] = []
            cur = trip.entry_node or net.element(trip.calls[0].platform).node  # type: ignore[union-attr]
            nodes.append(cur)
            for path in trip_legs:
                for eid in path:
                    item = net.element(eid)
                    if isinstance(item, Track):
                        cur = item.other(cur)
                        if nodes[-1] != cur:
                            nodes.append(cur)
            if len(nodes) >= 2:
                return nodes
    if "-" in raw and "->" not in raw:
        parts = [p.strip() for p in raw.split("-") if p.strip()]
        if chained := _chain_waypoints(net, parts):
            return chained
    tokens = [p for p in re.split(r"\s*(?:->|,|\s)\s*", raw) if p]
    if chained := _chain_waypoints(net, tokens):
        return chained
    return Refusal(NetRule.UNKNOWN_ELEMENT, f"unknown corridor {corridor!r}")


def resolve_area(timetable: Timetable, area: str | None) -> tuple[set[str], set[str]] | Refusal:
    """Return `(node_ids, element_ids)` covered by `area`, or Refusal if `area` is unknown."""
    net = timetable.net
    if area is None:
        return {n.id for n in net.nodes}, {x.id for x in (*net.tracks, *net.platforms, *net.moves)}
    raw = area.strip()
    if not raw:
        return Refusal(NetRule.UNKNOWN_ELEMENT, "empty area")
    if net.node(raw) is not None:
        tracks = {t.id for t in net.tracks if raw in (t.a, t.b)}
        nodes = {raw} | {t.other(raw) for t in net.tracks if raw in (t.a, t.b)}
        plats = {p.id for p in net.platforms if p.node == raw}
        moves = {m.id for m in net.moves if m.node == raw}
        return nodes, tracks | plats | moves
    el = net.element(raw)
    if isinstance(el, Track):
        nodes = {el.a, el.b}
        plats = {p.id for p in net.platforms if p.node in nodes}
        moves = {m.id for m in net.moves if m.node in nodes}
        return nodes, {el.id} | plats | moves
    if isinstance(el, Platform | Move):
        return resolve_area(timetable, el.node)
    corridor_nodes = resolve_corridor(timetable, raw)
    if not isinstance(corridor_nodes, Refusal):
        nodes = set(corridor_nodes)
        tracks = {t.id for t in net.tracks if t.a in nodes and t.b in nodes}
        plats = {p.id for p in net.platforms if p.node in nodes}
        moves = {m.id for m in net.moves if m.node in nodes}
        return nodes, tracks | plats | moves
    return Refusal(NetRule.UNKNOWN_ELEMENT, f"unknown area {area!r}")


def _train_in_area(mv: Movement, nodes: set[str], elements: set[str]) -> bool:
    if mv.position in elements or mv.position in nodes:
        return True
    if mv.phase == "offstage" and mv.trip.entry_node in nodes:
        return True
    return False


# ------------------------------------------------------------------ open conflicts & connections


def open_conflicts(engine: Engine) -> tuple[Conflict, ...]:
    if engine.tick >= engine.shift_min:
        return ()
    return engine._conflicts(engine._wanting(LOOKAHEAD_MIN))


def at_risk_connections(timetable: Timetable, engine: Engine) -> list[dict]:
    """Connections where the connector has not left `node` yet and feeder delay threatens the transfer."""
    if not timetable.connections or engine.tick >= engine.shift_min:
        return []
    net = timetable.net
    out: list[dict] = []
    for conn in timetable.connections:
        feeder = engine.trains.get(conn.feeder)
        connector = engine.trains.get(conn.connector)
        if feeder is None or connector is None or connector.phase == "done":
            continue
        conn_call = next((c for c in connector.plan if net.element(c.platform).node == conn.node), None)  # type: ignore[union-attr]
        if conn_call is None and connector.at is not None and net.element(connector.at.platform).node == conn.node:  # type: ignore[union-attr]
            conn_call = connector.at
        if conn_call is None or conn_call.depart is None:
            continue
        feed_booked = next(
            (c.arrive for c in feeder.trip.calls if net.element(c.platform).node == conn.node and c.arrive is not None),  # type: ignore[union-attr]
            None,
        )
        if feed_booked is None:
            continue
        slack = conn_call.depart - feed_booked - conn.min_transfer_min
        delay = feeder.lateness(engine.tick)
        if delay > slack and conn_call.depart <= engine.tick + IDLE_MIN:
            out.append(
                {
                    "node": conn.node,
                    "feeder": conn.feeder,
                    "connector": conn.connector,
                    "transfers": conn.transfers,
                    "slack_min": slack,
                    "feeder_delay": delay,
                    "connector_depart": conn_call.depart,
                }
            )
    return out


# ------------------------------------------------------------------ tool formatters


def network(timetable: Timetable, area: str | None = None) -> str | Refusal:
    resolved = resolve_area(timetable, area)
    if isinstance(resolved, Refusal):
        return resolved
    node_ids, element_ids = resolved
    net = timetable.net
    return canonical_json(
        {
            "rules_version": NET_RULES_VERSION,
            "tier": timetable.tier,
            "shift_min": timetable.shift_min,
            "area": area or "all",
            "nodes": [n.model_dump(mode="json") for n in net.nodes if n.id in node_ids],
            "tracks": [{**t.model_dump(mode="json"), "blocks": t.blocks} for t in net.tracks if t.id in element_ids],
            "platforms": [p.model_dump(mode="json") for p in net.platforms if p.id in element_ids],
            "moves": [m.model_dump(mode="json") for m in net.moves if m.id in element_ids],
            "crossings": [list(c) for c in net.crossings if c[0] in element_ids or c[1] in element_ids],
        }
    )


def timetable_json(timetable: Timetable, trips: list[Trip]) -> str:
    net = timetable.net
    ids = {t.train.id for t in trips}
    trains = []
    for trip in trips:
        spec = trip.train
        trains.append(
            {
                "id": spec.id,
                "type": train_type_code(spec),
                "length_m": spec.length_m,
                "max_speed_kmh": spec.max_speed_kmh,
                "electric": spec.electric,
                "passenger": spec.passenger,
                "restart_min": spec.restart_min,
                "entry_node": trip.entry_node,
                "entry_tick": trip.entry_tick,
                "exit_node": trip.exit_node,
                "calls": [
                    {
                        "platform": c.platform,
                        "node": net.element(c.platform).node,  # type: ignore[union-attr]
                        "arrive": c.arrive,
                        "depart": c.depart,
                        "min_dwell": c.min_dwell,
                        "alight": c.alight,
                        "board": c.board,
                    }
                    for c in trip.calls
                ],
            }
        )
    conns = [
        c.model_dump(mode="json") for c in timetable.connections if c.feeder in ids or c.connector in ids
    ]
    return canonical_json({"trains": trains, "connections": conns})


def _train_conflict_tag(engine: Engine, train_id: str) -> str:
    mv = engine.trains[train_id]
    tcode = train_type_code(mv.train)
    late = mv.lateness(engine.tick)
    late_str = f"+{late}" if late > 0 else str(late)
    pax = onboard_pax(mv)
    pax_part = f" {pax}p" if mv.train.passenger else ""
    due = mv.ready_at()
    due_part = " wait" if mv.waiting else (f" @{due}" if due is not None else "")
    return f"{train_id} {tcode}{pax_part} {late_str}{due_part}"


def format_conflict_lines(timetable: Timetable, engine: Engine, conflicts: tuple[Conflict, ...]) -> list[str]:
    lines = []
    for idx, c in enumerate(conflicts, start=1):
        res = format_key(c.key)
        label = key_label(timetable.net, c.key)
        who = " | ".join(_train_conflict_tag(engine, t) for t in c.trains)
        lines.append(f"C{idx} {res} ({label}): {who}")
    return lines


def conflicts_text(session: Session) -> str:
    eng = session.engine
    found = open_conflicts(eng)
    risks = at_risk_connections(session.timetable, eng)
    blocked = ",".join(eng.reported_blocked or ()) or "none"
    header = f"t={eng.tick} | {len(found)} conflicts | blocked: {blocked}"
    lines = [header, *format_conflict_lines(session.timetable, eng, found)]
    for r in risks:
        lines.append(
            f"CONN {r['node']}: {r['feeder']}->{r['connector']} ({r['transfers']}p, "
            f"slack {r['slack_min']}m, {r['feeder']} +{r['feeder_delay']}, dep@{r['connector_depart']})"
        )
    if eng.stranded:
        lines.append(f"UNROUTABLE: {', '.join(sorted(eng.stranded))}")
    return "\n".join(lines)


def full_status(session: Session, area: str | None = None) -> str | Refusal:
    resolved = resolve_area(session.timetable, area)
    if isinstance(resolved, Refusal):
        return resolved
    nodes, elements = resolved
    eng = session.engine
    net = session.timetable.net
    found = [c for c in open_conflicts(eng) if area is None or (key_nodes(net, c.key) & nodes)]
    trains = []
    for mv in sorted(eng.trains.values(), key=lambda m: m.train.id):
        if area is not None and not _train_in_area(mv, nodes, elements):
            continue
        el = net.element(mv.position)
        nxt = mv.plan[0] if mv.plan else None
        finished_tick = None
        if mv.phase == "done":
            finished_tick = mv.visits[-1].arrived if mv.visits and mv.trip.exit_node is None else eng.tick
        trains.append(
            {
                "id": mv.train.id,
                "type": train_type_code(mv.train),
                "phase": mv.phase,
                "position": mv.position,
                "facing": mv.facing(net) if mv.phase != "done" else None,
                "block": mv.block if mv.phase == "running" and isinstance(el, Track) else None,
                "blocks": el.blocks if mv.phase == "running" and isinstance(el, Track) else None,
                "waiting": mv.waiting,
                "held_until": mv.held_until,
                "delay": mv.lateness(eng.tick),
                "onboard_pax": onboard_pax(mv),
                "next_call": (
                    {"platform": nxt.platform, "arrive": nxt.arrive, "depart": nxt.depart} if nxt is not None else None
                ),
                "exit_node": mv.trip.exit_node if not mv.plan and mv.working else None,
                "finished_tick": finished_tick,
            }
        )
    holders = {
        format_key(r): t
        for r, t in sorted(eng.il.holder.items(), key=lambda kv: format_key(kv[0]))
        if area is None or (key_nodes(net, r) & nodes)
    }
    orders = {format_key(k): v for k, v in sorted(eng.orders.items(), key=lambda kv: format_key(kv[0]))}
    return canonical_json(
        {
            "tick": eng.tick,
            "area": area or "all",
            "notes": session.notes,
            "blocked": list(eng.reported_blocked or ()),
            "unroutable": sorted(eng.stranded),
            "conflicts": [{"resource": format_key(c.key), "trains": list(c.trains)} for c in found],
            "orders": orders,
            "holders": holders,
            "trains": trains,
        }
    )


def _compact_next(mv: Movement) -> str:
    if mv.plan:
        c = mv.plan[0]
        return f"{c.platform}@{c.arrive}" if c.arrive is not None else c.platform
    if mv.working and mv.trip.exit_node:
        return mv.trip.exit_node
    return "-"


def compact_status(session: Session, area: str | None = None) -> str | Refusal:
    resolved = resolve_area(session.timetable, area)
    if isinstance(resolved, Refusal):
        return resolved
    nodes, elements = resolved
    eng = session.engine
    net = session.timetable.net
    found = [c for c in open_conflicts(eng) if area is None or (key_nodes(net, c.key) & nodes)]
    blocked = ",".join(eng.reported_blocked or ()) or "none"
    later = done = 0
    rows: list[str] = []
    for mv in sorted(eng.trains.values(), key=lambda m: m.train.id):
        if area is not None and not _train_in_area(mv, nodes, elements):
            continue
        if mv.phase == "done":
            done += 1
            continue
        due = mv.ready_at()
        if mv.phase == "offstage" and (due or 0) > eng.tick + IDLE_MIN:
            later += 1
            continue
        tid = mv.train.id
        tcode = train_type_code(mv.train)
        late = mv.lateness(eng.tick)
        dstr = f"+{late}" if late > 0 else "0"
        nxt = _compact_next(mv)
        if mv.phase == "offstage":
            rows.append(f"{tid:<4} {tcode:<2} in {mv.trip.entry_node}@{due} ->{nxt}")
        elif mv.phase == "berthed":
            hold = f" hold@{mv.held_until}" if mv.held_until and mv.held_until > eng.tick else ""
            rows.append(f"{tid:<4} {tcode:<2} at {mv.position} {dstr} dep@{due}{hold} ->{nxt}")
        else:
            el = net.element(mv.position)
            pos = f"{mv.position}[{mv.block + 1}/{el.blocks}]" if isinstance(el, Track) else mv.position
            wait = " WAIT" if mv.waiting else ""
            rows.append(f"{tid:<4} {tcode:<2} {pos}->{mv.facing(net)}{wait} {dstr} ->{nxt}")
    area_tag = f" area={area}" if area else ""
    counts_tag = f" | later:{later} done:{done}" if (later or done) else ""
    header = (
        f't={eng.tick}{area_tag} | conflicts:{len(found)} | blocked:{blocked}{counts_tag} | notes:"{session.notes}"'
    )
    lines = [header, *rows]
    if found:
        lines += format_conflict_lines(session.timetable, eng, tuple(found))
    if eng.stranded:
        stranded = [t for t in sorted(eng.stranded) if area is None or _train_in_area(eng.trains[t], nodes, elements)]
        if stranded:
            lines.append(f"UNROUTABLE: {','.join(stranded)}")
    return "\n".join(lines)


def final_line(tick: int, cost: CostBreakdown, has_cost_module: bool = True) -> str:
    if not has_cost_module:
        return f"SHIFT ENDED at t={tick}. Final cost {cost.total}"
    return (
        f"SHIFT ENDED at t={tick}. Final cost {cost.total} (pax {cost.pax_lateness}, missed {cost.missed_calls}, "
        f"unfinished {cost.unfinished}, freight {cost.freight_lateness}, crew {cost.crew_timeout}, "
        f"gridlock {cost.gridlock}, ops {cost.operations})"
    )


def advance_text(
    session: Session,
    decision: Decision,
    since: int,
    changes: list[tuple[int, str]],
    final: str | None,
) -> str:
    who = f" {', '.join(decision.trains)}" if decision.trains else ""
    lines = [f"t={decision.tick} stop: {decision.kind}{who}"]
    if final:
        lines.append(final)
    if decision.conflicts:
        lines += format_conflict_lines(session.timetable, session.engine, decision.conflicts)
    for r in at_risk_connections(session.timetable, session.engine):
        lines.append(
            f"CONN {r['node']}: {r['feeder']}->{r['connector']} ({r['transfers']}p, "
            f"slack {r['slack_min']}m, {r['feeder']} +{r['feeder_delay']}, dep@{r['connector_depart']})"
        )
    lines.append(f"changes since t={since}:")
    if changes:
        lines += [f"  t={t} {msg}" for t, msg in changes]
    else:
        lines.append("  (none)")
    lines.append(f"budget: {session.used}/{session.budget}")
    return "\n".join(lines)


# ------------------------------------------------------------------ stringline


def _corridor_place(net: Network, corridor_nodes: list[str], mv: Movement) -> str | None:
    if mv.phase == "offstage":
        return None
    node_set = set(corridor_nodes)
    seg_by_pair = {
        frozenset((a, b)): f"{a}-{b}" for a, b in zip(corridor_nodes, corridor_nodes[1:], strict=False)
    }
    if mv.position in node_set:
        return mv.position
    el = net.element(mv.position)
    if isinstance(el, Platform | Move):
        return el.node if el.node in node_set else None
    if isinstance(el, Track):
        return seg_by_pair.get(frozenset((el.a, el.b)))
    return None


def _clone_engine(eng: Engine) -> Engine:
    """Fast copy of an Engine sharing the frozen Network and Trip models."""
    memo: dict[int, object] = {id(eng.net): eng.net}
    for mv in eng.trains.values():
        memo[id(mv.trip)] = mv.trip
    return copy.deepcopy(eng, memo)


def _extend_ready(eng: Engine, auto_order: bool) -> None:
    eng._resume()
    soon = eng._wanting(LOOKAHEAD_MIN)
    conflicts = eng._conflicts(soon)
    if conflicts and auto_order:
        for c in conflicts:
            eng.order(c.key, sorted(c.trains, key=lambda t: (eng.trains[t].booked(), t)))
        conflicts = eng._conflicts(soon)
    contested = {t for c in conflicts for t in c.trains}
    now = {t: s for t, s in eng._wanting(0).items() if t in soon and t not in contested}
    for tid in eng._priority(now):
        eng._extend(eng.trains[tid], now[tid])
    eng.orders = {
        k: o for k, o in eng.orders.items() if any(eng.trains[t].working for t in o if t in eng.trains)
    }


def _step_one_tick_towards(eng: Engine, target_tick: int, auto_order: bool) -> None:
    """Advance `eng` settling un-ordered conflicts in timetable order when `auto_order` is True."""
    while eng.tick < target_tick:
        _extend_ready(eng, auto_order=auto_order)
        nxt = eng._next_event()
        if nxt is None or nxt > target_tick:
            eng.tick = target_tick
            return
        eng.tick = nxt
        eng._run_transitions()


def _corridor_cells(
    net: Network, corridor_nodes: list[str], eng: Engine, tick: int
) -> dict[str, list[str]]:
    cells: dict[str, list[str]] = {}
    portals = {n for n in corridor_nodes if (nd := net.node(n)) is not None and nd.kind == "portal"}
    for tid, mv in sorted(eng.trains.items()):
        if mv.phase == "done" and mv.position in portals:
            # Show on the exit tick only, not parked at the portal forever after leaving the area.
            continue
        place = _corridor_place(net, corridor_nodes, mv)
        if place is not None:
            cells.setdefault(place, []).append(tid)
    return cells


def render_stringline(
    session: Session, corridor: str, from_tick: int, to_tick: int
) -> tuple[str, str, str | None]:
    shift = session.timetable.shift_min
    nodes_or_refusal = resolve_corridor(session.timetable, corridor)
    if isinstance(nodes_or_refusal, Refusal):
        r = nodes_or_refusal
        return str(r), "INVALID", r.rule.value
    if not (0 <= from_tick <= to_tick <= shift and to_tick - from_tick <= STRINGLINE_WINDOW):
        detail = f"need 0 <= from_tick <= to_tick <= {shift} and a window of at most {STRINGLINE_WINDOW} ticks"
        return f"REFUSED {NetRule.WINDOW.value}: {detail}", "REFUSED", NetRule.WINDOW.value

    corridor_nodes = nodes_or_refusal
    net = session.timetable.net
    now = session.engine.tick
    columns = [x for a, b in zip(corridor_nodes, corridor_nodes[1:], strict=False) for x in (a, f"{a}-{b}")]
    columns.append(corridor_nodes[-1])

    # Replay past actions on a clone of the initial engine up to min(now, to_tick), and project from live engine after.
    occupancy_by_tick: dict[int, dict[str, list[str]]] = {}
    if from_tick <= now:
        past = _clone_engine(session._initial_engine)
        actions_by_tick: dict[int, list[tuple[str, tuple]]] = {}
        for act_tick, fn_name, fn_args in session._applied_actions:
            actions_by_tick.setdefault(act_tick, []).append((fn_name, fn_args))
        for t in range(0, min(now, to_tick) + 1):
            if t == now:
                if t >= from_tick:
                    occupancy_by_tick[t] = _corridor_cells(net, corridor_nodes, session.engine, t)
                break
            if t > past.tick:
                _step_one_tick_towards(past, t, auto_order=False)
            for fn_name, fn_args in actions_by_tick.get(t, ()):
                getattr(past, fn_name)(*fn_args)
            _extend_ready(past, auto_order=False)
            if t >= from_tick:
                occupancy_by_tick[t] = _corridor_cells(net, corridor_nodes, past, t)

    if to_tick > now:
        proj = _clone_engine(session.engine)
        for t in range(max(from_tick, now + 1), to_tick + 1):
            _step_one_tick_towards(proj, t, auto_order=True)
            occupancy_by_tick[t] = _corridor_cells(net, corridor_nodes, proj, t)

    title = f"corridor {' '.join(corridor_nodes)}"
    header = " " * 6 + "".join(f"{'-' if '-' in c else c:<5}" for c in columns)

    def row(tick: int) -> str:
        cells = occupancy_by_tick.get(tick, {})
        text = "".join(
            f"{(cells[c][0] + ('+' if len(cells[c]) > 1 else '')) if c in cells else '.':<5}" for c in columns
        )
        return f"{'>' if tick == now else ' '}t{tick:<3} {text}".rstrip()

    step = max(1, math.ceil((to_tick - from_tick + 1) / 30))
    while True:
        out = "\n".join([title, header.rstrip(), *(row(t) for t in range(from_tick, to_tick + 1, step))])
        if len(out) <= STRINGLINE_MAX_CHARS or step > max(1, to_tick - from_tick):
            return out, "INFO", None
        step += 1
