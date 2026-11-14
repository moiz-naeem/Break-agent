"""v0.3 scenario generator: an irregular network of a trunk, branches and bypasses (docs/design-changes-v0.3.md §4).

Layout template, perturbed per seed:

    TW - S1 - S2 - ... - Sk - TE          the trunk, between two trunk-end terminals
              |                           a branch leaves at a junction station: A1 - A2 - ... - TA
              +- X1 - X2 -+               a bypass rejoins the trunk 2-3 stations on, 20-40% slower

Every train has an origin and a destination terminal and a timetabled route (the quickest path). Most trains
cross the busiest trunk section, so the trunk stays the bottleneck however the branches fall. Passenger stops
carry alight/board counts from a load model, a headway from the timetable, and at hubs a published platform.
"""

from __future__ import annotations

import heapq
import math
import random

from railway.sim.constants import (
    CREW_LIMIT,
    LATE_ENTRY_LOOKAHEAD,
    POSSESSION_LOOKAHEAD,
    RESTRICTION_LOOKAHEAD,
    RULES_VERSION,
    TRAIN_TYPES,
    TURNAROUND,
)
from railway.sim.model import Scenario
from railway.sim.timing import travel_ticks
from railway.util import content_hash

from .config import GenConfig

BRANCH_LETTERS = "ABCD"
BYPASS_LETTERS = "XY"
PLATFORM = {"id": "P1", "length_m": 300, "kind": "platform"}
MAX_HEADWAY = 120


def _rng(seed: int, stage: str) -> random.Random:
    return random.Random(f"graph:{seed}:{stage}")


# -- network -------------------------------------------------------------------------------------------


def _loops(rng: random.Random, config: GenConfig) -> list[dict]:
    count = rng.choices(list(config.loops), weights=list(config.loops.values()), k=1)[0]
    loops = []
    for j in range(count):
        length = rng.choices([150, 300, 700], weights=[0.25, 0.45, 0.30], k=1)[0]
        kind = "goods" if length == 700 and rng.random() < 0.5 else "platform"
        loops.append({"id": f"L{j + 1}", "length_m": length, "kind": kind})
    return loops


def _ensure_loop(station: dict, length: int = 700, kind: str = "goods"):
    if any(loop["length_m"] >= length and (kind == "goods" or loop["kind"] == kind) for loop in station["loops"]):
        return
    if len(station["loops"]) < 3:
        station["loops"].append({"id": f"L{len(station['loops']) + 1}", "length_m": length, "kind": kind})
    else:
        station["loops"][-1].update(length_m=length, kind=kind)


def _layout(config: GenConfig, rng: random.Random):
    """Choose trunk length, branch and bypass sizes and the trunk stations they attach to."""
    need = 2 + config.branches + 3 * config.bypasses + 2  # trunk stations: ends, junctions, bypass spans, spacing
    spare = config.stations - need
    if spare < 2 * (config.branches + config.bypasses):
        raise ValueError(f"{config.preset}: {config.stations} stations are too few for the layout (trunk needs {need})")
    branch_lens = [rng.randint(2, 4) for _ in range(config.branches)]
    bypass_lens = [rng.randint(2, 3) for _ in range(config.bypasses)]
    while sum(branch_lens) + sum(bypass_lens) > spare:  # shorten the longest arm until the layout fits
        arms = [(n, "b", i) for i, n in enumerate(branch_lens) if n > 2] + [(n, "x", i) for i, n in enumerate(bypass_lens) if n > 2]
        _, kind, i = max(arms)
        (branch_lens if kind == "b" else bypass_lens)[i] -= 1
    trunk = config.stations - sum(branch_lens) - sum(bypass_lens)
    for _ in range(200):
        taken: set[int] = set()
        bypasses = []
        for length in bypass_lens:
            span = rng.randint(2, 3)  # rejoins 2-3 trunk stations on: skips 1-2 stations
            starts = [a for a in range(2, trunk - span) if not taken & set(range(a - 1, a + span + 2))]
            if not starts:
                break
            a = rng.choice(starts)
            bypasses.append((a, a + span, length))
            taken |= set(range(a, a + span + 1))
        else:
            free = [i for i in range(2, trunk) if i not in taken]
            if len(free) < config.branches:
                continue
            junctions = sorted(rng.sample(free, config.branches))
            return trunk, list(zip(junctions, branch_lens, strict=True)), bypasses
    raise ValueError(f"{config.preset}: could not place {config.bypasses} bypasses and {config.branches} branches")


def _network(config: GenConfig, seed: int):
    rng = _rng(seed, "network")
    trunk, branches, bypasses = _layout(config, rng)
    stations: dict[str, dict] = {}
    sections: list[tuple[str, str]] = []

    def station(sid: str) -> dict:
        stations[sid] = {"id": sid, "major": False, "crew_station": False, "loops": _loops(rng, config)}
        return stations[sid]

    trunk_ids = [f"S{i}" for i in range(1, trunk + 1)]
    for sid in trunk_ids:
        station(sid)
    nodes = ["TW", *trunk_ids, "TE"]
    sections += list(zip(nodes, nodes[1:], strict=False))
    terminals = {"TW": "trunk", "TE": "trunk"}
    for b, (junction, length) in enumerate(branches):
        letter = BRANCH_LETTERS[b]
        ids = [f"{letter}{i}" for i in range(1, length + 1)]
        for sid in ids:
            station(sid)
        path = [f"S{junction}", *ids, f"T{letter}"]
        sections += list(zip(path, path[1:], strict=False))
        terminals[f"T{letter}"] = "branch"
    if config.port and branches:
        terminals[f"T{BRANCH_LETTERS[len(branches) - 1]}"] = "port"
    bypass_routes = []
    for k, (a, b, length) in enumerate(bypasses):
        letter = BYPASS_LETTERS[k]
        ids = [f"{letter}{i}" for i in range(1, length + 1)]
        for sid in ids:
            station(sid)
            _ensure_loop(stations[sid])
        path = [f"S{a}", *ids, f"S{b}"]
        sections += list(zip(path, path[1:], strict=False))
        bypass_routes.append((f"S{a}", f"S{b}", path))

    degree: dict[str, int] = {}
    for a, b in sections:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    junctions = sorted(s for s in stations if degree[s] == 3)
    for sid in junctions:  # a junction is a hub: a long goods loop and a platform loop
        _ensure_loop(stations[sid])
        _ensure_loop(stations[sid], 300, "platform")
        stations[sid]["major"] = True
    # A 600 m freight can only wait in a 700 m loop: never leave 3 trunk stations in a row without one.
    gap = 0
    for sid in trunk_ids:
        gap = 0 if any(x["length_m"] == 700 for x in stations[sid]["loops"]) else gap + 1
        if gap == 3:
            _ensure_loop(stations[sid])
            gap = 0
    for a, b in sections:  # no two loopless stations side by side
        if a in stations and b in stations and not stations[a]["loops"] and not stations[b]["loops"]:
            stations[b]["loops"].append({"id": "L1", "length_m": 150, "kind": "platform"})
    plain = sorted(s for s in stations if s not in junctions)
    for sid in rng.sample(plain, max(0, round(len(stations) * config.major_fraction) - len(junctions))):
        stations[sid]["major"] = True
    if "F10" in config.factors:
        candidates = [s for s in [*junctions, *trunk_ids] if stations[s]["loops"]]
        chosen = []
        for position in (0.2, 0.5, 0.8):
            pick = candidates[min(len(candidates) - 1, round(position * (len(candidates) - 1)))]
            if pick not in chosen:
                chosen.append(pick)
        for sid in chosen[: max(2, len(junctions))]:
            stations[sid]["crew_station"] = True

    section_rng = _rng(seed, "sections")
    out_sections = {}
    for a, b in sections:
        out_sections[(a, b)] = {
            "id": f"{a}-{b}",
            "west": a,
            "east": b,
            "length_m": section_rng.choice(range(4000, 9001, 500)),
            "line_speed_kmh": section_rng.choice([80, 100, 120, 140]),
        }
    # A bypass is 20-40% slower than the trunk stretch it avoids, so the timetable keeps to the trunk.
    for a, b, path in bypass_routes:
        i, j = trunk_ids.index(a), trunk_ids.index(b)
        trunk_minutes = sum(
            out_sections[(x, y)]["length_m"] / out_sections[(x, y)]["line_speed_kmh"]
            for x, y in zip(trunk_ids[i:j], trunk_ids[i + 1 : j + 1], strict=False)
        )
        target = trunk_minutes * section_rng.uniform(1.2, 1.4)
        hops = list(zip(path, path[1:], strict=False))
        speed = 100
        for x, y in hops:
            out_sections[(x, y)]["line_speed_kmh"] = speed
            out_sections[(x, y)]["length_m"] = max(2000, int(round(target * speed / len(hops) / 500)) * 500)

    tracks = {
        "trunk": [PLATFORM, {"id": "P2", "length_m": 300, "kind": "platform"}, {"id": "P3", "length_m": 700, "kind": "goods"}],
        "branch": [PLATFORM, {"id": "P2", "length_m": 700, "kind": "goods"}],
        "port": [{"id": "P1", "length_m": 700, "kind": "goods"}, {"id": "P2", "length_m": 700, "kind": "goods"}],
    }  # fmt: skip
    network = {
        "terminals": [{"id": t, "tracks": [dict(x) for x in tracks[kind]]} for t, kind in terminals.items()],
        "stations": [stations[s] for s in sorted(stations, key=_node_order)],
        "sections": [out_sections[k] for k in sections],
    }
    return network, terminals, trunk_ids


def _node_order(node: str):
    return (node[0] != "S", node[0], int("".join(c for c in node if c.isdigit()) or 0))


# -- routing ---------------------------------------------------------------------------------------------


def _quickest(network: dict, kind: str, origin: str, dest: str) -> list[str] | None:
    """The quickest route for a train type (Dijkstra on running time; ties broken by node order)."""
    speed = TRAIN_TYPES[kind]["max_speed_kmh"]
    adj: dict[str, list[tuple[str, int]]] = {}
    for s in network["sections"]:
        t = travel_ticks(s["length_m"], s["line_speed_kmh"], speed)
        adj.setdefault(s["west"], []).append((s["east"], t))
        adj.setdefault(s["east"], []).append((s["west"], t))
    terminals = {t["id"] for t in network["terminals"]}
    best = {origin: (0, [origin])}
    queue = [(0, [origin])]
    while queue:
        cost, path = heapq.heappop(queue)
        node = path[-1]
        if node == dest:
            return path
        if best.get(node, (math.inf,))[0] < cost or (node in terminals and node != origin):
            continue
        for nxt, t in sorted(adj[node]):
            if nxt in path:
                continue
            cand = (cost + t, [*path, nxt])
            if cand < best.get(nxt, (math.inf, [])):
                best[nxt] = cand
                heapq.heappush(queue, cand)
    return None


def _od_pairs(config: GenConfig, network, terminals: dict, trunk_ids: list[str], rng: random.Random):
    """Directed origin-destination pairs. Passenger trains never use the port; freight always does if there
    is one. Pairs crossing the middle trunk section come first, so they carry most trains."""
    middle = {trunk_ids[len(trunk_ids) // 2 - 1], trunk_ids[len(trunk_ids) // 2]}
    names = sorted(terminals)
    pax_pairs, freight_pairs = [], []
    for a in names:
        for b in names:
            if a == b:
                continue
            route = _quickest(network, "local", a, b)
            if route is None:
                continue
            crosses = middle <= set(route)
            port = "port" in (terminals[a], terminals[b])
            if not port:
                pax_pairs.append((not crosses, a, b))
            if port or not any(v == "port" for v in terminals.values()):
                freight_pairs.append((not crosses, a, b))
    rng.shuffle(pax_pairs)
    rng.shuffle(freight_pairs)
    pax_pairs.sort(key=lambda p: p[0])  # crossing pairs first (stable: shuffled within each group)
    freight_pairs.sort(key=lambda p: p[0])
    return [(a, b, not c) for c, a, b in pax_pairs], [(a, b, not c) for c, a, b in freight_pairs]


def _assign(pairs, count: int, wanted_pairs: int, trunk_share: float, rng: random.Random):
    """Spread `count` trains over `wanted_pairs` pairs, with at least `trunk_share` of them on pairs whose
    route crosses the middle of the trunk."""
    if count == 0:
        return []
    crossing = [p for p in pairs if p[2]]
    other = [p for p in pairs if not p[2]]
    n_other_pairs = min(len(other), round(wanted_pairs * (1 - trunk_share)))
    n_cross_pairs = max(1, min(len(crossing), wanted_pairs - n_other_pairs))
    cross_pairs, other_pairs = crossing[:n_cross_pairs], other[:n_other_pairs]
    if not cross_pairs:
        cross_pairs, other_pairs = other[: max(1, wanted_pairs)], []
    n_cross = count if not other_pairs else min(count, math.ceil(count * trunk_share))
    out = [cross_pairs[i % len(cross_pairs)] for i in range(n_cross)]
    out += [other_pairs[i % len(other_pairs)] for i in range(count - n_cross)]
    rng.shuffle(out)
    return [(a, b) for a, b, _ in out]


# -- timetable ---------------------------------------------------------------------------------------------


def _type_counts(config: GenConfig):
    exact = {name: config.trains * weight for name, weight in config.mix.items()}
    counts = {name: math.floor(value) for name, value in exact.items()}
    remainder = config.trains - sum(counts.values())
    order = sorted(exact, key=lambda name: (-(exact[name] - counts[name]), name))
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def _flows(rng: random.Random, pax: int, stops: list[dict], hubs: set[str]):
    """Load model: `pax` board at the origin; at each stop some get off and others on; all get off at the end."""
    load = pax
    for i, stop in enumerate(stops):
        if i == len(stops) - 1:
            stop["alight"], stop["board"] = load, 0
            continue
        weight = 2.0 if stop["station"] in hubs else 1.0
        alight = min(load, round(load * rng.uniform(0.05, 0.2) * weight))
        board = round(pax * rng.uniform(0.03, 0.15) * weight)
        stop["alight"], stop["board"] = alight, board
        load = load - alight + board


def _schedule(config, network, kind, route, majors, rng):
    """Stops and their times relative to entry, along `route`."""
    sec = {}
    for s in network["sections"]:
        sec[(s["west"], s["east"])] = sec[(s["east"], s["west"])] = s
    stops, tick = [], 0
    for a, b in zip(route, route[1:], strict=False):
        s = sec[(a, b)]
        run = travel_ticks(s["length_m"], s["line_speed_kmh"], TRAIN_TYPES[kind]["max_speed_kmh"])
        tick += run + math.ceil(run * config.slack)
        terminal = b == route[-1]
        call = terminal or (kind == "local") or (kind == "express" and b in majors)
        if call:
            dwell = 0 if terminal else (1 if kind == "express" else rng.randint(1, 2))
            stops.append({"station": b, "arr": tick, "dep": None if terminal else tick + dwell,
                          "min_dwell": 0 if terminal else dwell})  # fmt: skip
            tick += dwell
    return stops


def _terminal_peak(trains, terminal: str) -> int:
    events = []
    for t in trains:
        if t["origin"] == terminal:
            events += [(t["entry_tick"], 1), (t["entry_tick"] + 1, -1)]
        if t["destination"] == terminal:
            arr = t["stops"][-1]["arr"]
            events += [(arr, 1), (arr + TURNAROUND, -1)]
    peak = level = 0
    for _, delta in sorted(events, key=lambda e: (e[0], e[1])):
        level += delta
        peak = max(peak, level)
    return peak


def _trains(config: GenConfig, network: dict, terminals: dict, trunk_ids: list[str], seed: int):
    rng = _rng(seed, "trains")
    tt = _rng(seed, "timetable")
    majors = {s["id"] for s in network["stations"] if s["major"]}
    hubs = {s["id"] for s in network["stations"] if sum(s["id"] in (x["west"], x["east"]) for x in network["sections"]) == 3}
    pax_pairs, freight_pairs = _od_pairs(config, network, terminals, trunk_ids, rng)
    counts = _type_counts(config)
    n_pax = counts["express"] + counts["local"]
    pax_od = _assign(pax_pairs, n_pax, max(1, config.od_pairs - 1), config.trunk_share, rng)
    freight_od = _assign(freight_pairs, counts["freight"], max(1, min(2, config.od_pairs // 3)), config.trunk_share, rng)
    kinds = ["express"] * counts["express"] + ["local"] * counts["local"]
    rng.shuffle(kinds)
    prefix = {"express": "E", "local": "L", "freight": "F"}
    numbers = dict.fromkeys(prefix, 0)
    output = []
    for kind, (origin, dest) in [*zip(kinds, pax_od, strict=True), *(("freight", od) for od in freight_od)]:
        numbers[kind] += 1
        route = _quickest(network, kind, origin, dest)
        stops = _schedule(config, network, kind, route, majors, tt)
        pax = tt.randrange(150, 401, 10) if kind == "express" else tt.randrange(50, 201, 10) if kind == "local" else 0
        if kind != "freight":
            _flows(tt, pax, stops, hubs)
        output.append({"id": f"{prefix[kind]}{numbers[kind]}", "type": kind, "origin": origin, "destination": dest,
                       "route": route, "pax": pax, "stops": stops, "next_service": None,
                       "crew_used_at_entry": 0})  # fmt: skip
    # Entry ticks: uniform in the shift, resampled until every terminal has a spare track at its peak.
    capacity = {t["id"]: len(t["tracks"]) for t in network["terminals"]}
    relative = {t["id"]: [dict(s) for s in t["stops"]] for t in output}
    for _ in range(500):
        for t in output:
            run = relative[t["id"]][-1]["arr"]
            latest = config.shift_ticks - run - config.end_margin
            if latest < 0:
                raise ValueError(f"shift too short for {config.preset} run length {run}")
            entry = tt.randint(0, latest)
            t["entry_tick"] = entry
            t["stops"] = [
                {**s, "arr": s["arr"] + entry, "dep": None if s["dep"] is None else s["dep"] + entry}
                for s in relative[t["id"]]
            ]
        if all(_terminal_peak(output, term) <= capacity[term] - 1 for term in capacity):
            break
    else:
        raise ValueError(f"{config.preset}: no timetable fits the terminal tracks")
    return output, hubs


def _headways(trains: list[dict]):
    """Headway at a stop: minutes until the next passenger train calling there and going the same way."""
    calls: dict[tuple[str, str], list[int]] = {}
    for t in trains:
        if t["type"] == "freight":
            continue
        for i, s in enumerate(t["stops"]):
            onward = t["route"][t["route"].index(s["station"]) + 1] if s["dep"] is not None else "end"
            calls.setdefault((s["station"], onward), []).append(s["arr"])
            s["_key"] = (s["station"], onward)
    for t in trains:
        for s in t["stops"]:
            key = s.pop("_key", None)
            if key is None:
                continue
            later = [a for a in calls[key] if a > s["arr"]]
            s["headway"] = min(MAX_HEADWAY, min(later) - s["arr"]) if later else MAX_HEADWAY  # last one of the day


def _platforms(trains: list[dict], hubs: set[str], network: dict, rng: random.Random):
    """At a hub, a passenger stop has a published platform: the through main or the platform loop."""
    loops = {s["id"]: s["loops"] for s in network["stations"]}
    for t in trains:
        if t["type"] == "freight":
            continue
        length = TRAIN_TYPES[t["type"]]["length_m"]
        for s in t["stops"][:-1]:
            if s["station"] not in hubs:
                continue
            fitting = [x["id"] for x in loops[s["station"]] if x["kind"] == "platform" and x["length_m"] >= length]
            s["platform"] = rng.choice(["main", *fitting])


def _connections(trains: list[dict], hubs: set[str], rng: random.Random, wanted: int):
    """F11: at a hub, passengers from a feeder change to a connector leaving 3-15 minutes after it arrives."""
    out = []
    by_id = {t["id"]: t for t in trains if t["type"] != "freight"}
    options = []
    for f in by_id.values():
        for fs in f["stops"]:
            if fs["station"] not in hubs:
                continue
            for c in by_id.values():
                if c["id"] == f["id"]:
                    continue
                cs = next((s for s in c["stops"][:-1] if s["station"] == fs["station"]), None)
                if cs and 3 <= cs["dep"] - fs["arr"] <= 15:
                    options.append((f["id"], c["id"], fs["station"], fs["alight"]))
    rng.shuffle(options)
    used = set()
    for feeder, connector, station, alight in sorted(options[: wanted * 3], key=lambda o: o[:3]):
        if len(out) >= wanted or (feeder, station) in used or alight < 5:
            continue
        used.add((feeder, station))
        out.append({"feeder": feeder, "connector": connector, "station": station,
                    "pax": max(1, round(alight * rng.uniform(0.2, 0.5))), "min_transfer": rng.randint(2, 4)})  # fmt: skip
    return out


def _rotations(trains: list[dict], fraction: float):
    """F6: the same stock runs a later service of the same type from the terminal it arrived at."""
    count = round(len(trains) * fraction)
    used = set()
    links = 0
    for before in sorted(trains, key=lambda t: (t["stops"][-1]["arr"], t["id"])):
        if links >= count:
            break
        ready = before["stops"][-1]["arr"] + TURNAROUND
        after = next(
            (
                c
                for c in sorted(trains, key=lambda t: (t["entry_tick"], t["id"]))
                if c["id"] not in used
                and c["id"] != before["id"]
                and c["type"] == before["type"]
                and c["origin"] == before["destination"]
                and c["entry_tick"] >= ready
            ),
            None,
        )
        if after:
            before["next_service"] = after["id"]
            used.add(after["id"])
            links += 1


def _events(config: GenConfig, network: dict, trains: list[dict], seed: int):
    output = []
    used_late = set()
    sections = network["sections"]
    for factor, kind in (("F7", "possession"), ("F8", "speed_restriction"), ("F15", "breakdown"), ("F16", "late_entry")):
        if factor not in config.factors:
            continue
        rng = _rng(seed, f"events-{kind}")
        for _ in range(max(1, round(config.event_density * config.shift_ticks / 120))):
            event = {"kind": kind}
            if kind == "possession":
                section = rng.choice(sections)
                start = rng.randint(60, config.shift_ticks - 40)
                event.update(section=section["id"], start=start, end=start + rng.randint(15, min(40, config.shift_ticks - start)),
                             reveal_tick=max(0, start - rng.randint(*POSSESSION_LOOKAHEAD)))  # fmt: skip
                target = section["id"]
            elif kind == "speed_restriction":
                start = rng.randint(30, config.shift_ticks - 30)
                first = rng.choice(sections)
                covered = [first["id"]]
                touching = [s["id"] for s in sections if s["id"] != first["id"] and {s["west"], s["east"]} & {first["west"], first["east"]}]
                if touching and rng.random() < 0.5:
                    covered.append(rng.choice(sorted(touching)))
                event.update(sections=covered, speed_kmh=rng.choice([40, 60]), start=start, end=start + rng.randint(30, 90),
                             reveal_tick=max(0, start - rng.randint(*RESTRICTION_LOOKAHEAD)))  # fmt: skip
                target = covered[0]
            elif kind == "breakdown":
                train = rng.choice(trains)
                lo, hi = train["entry_tick"] + 5, train["stops"][-1]["arr"] - 5
                if hi < lo:
                    continue
                tick = rng.randint(lo, hi)
                event.update(train=train["id"], tick=tick, reveal_tick=tick, duration=rng.randint(5, 25))
                target = train["id"]
            else:
                eligible = [t for t in trains if t["entry_tick"] >= LATE_ENTRY_LOOKAHEAD and t["id"] not in used_late]
                if not eligible:
                    continue
                train = rng.choice(eligible)
                used_late.add(train["id"])
                event.update(train=train["id"], new_entry_tick=train["entry_tick"] + rng.randint(10, 40),
                             reveal_tick=train["entry_tick"] - LATE_ENTRY_LOOKAHEAD)  # fmt: skip
                target = train["id"]
            event["_sort"] = (event["reveal_tick"], kind, target)
            output.append(event)
    output.sort(key=lambda e: e["_sort"])
    for i, event in enumerate(output, start=1):
        event["id"] = f"EV{i}"
        del event["_sort"]
    output.sort(key=lambda e: (e["reveal_tick"], e["id"]))
    return output


def generate_graph(config: GenConfig, seed: int) -> Scenario:
    network, terminals, trunk_ids = _network(config, seed)
    trains, hubs = _trains(config, network, terminals, trunk_ids, seed)
    _headways(trains)
    _platforms(trains, hubs, network, _rng(seed, "platforms"))
    connections = []
    if "F11" in config.factors:
        connections = _connections(trains, hubs, _rng(seed, "connections"), config.connections)
    if "F6" in config.factors:
        _rotations(trains, config.rotation_fraction)
    if "F10" in config.factors:
        rng = _rng(seed, "crew")
        for t in trains:
            run = t["stops"][-1]["arr"] - t["entry_tick"]
            if t["next_service"]:
                t["crew_used_at_entry"] = 0
            elif rng.random() < config.tight_crew_fraction:
                t["crew_used_at_entry"] = max(0, CREW_LIMIT - int(run * rng.uniform(0.4, 0.8)))
            else:
                t["crew_used_at_entry"] = rng.randint(0, 100)
    trains.sort(key=lambda t: t["id"])
    events = _events(config, network, trains, seed)
    payload = {
        "schema_version": 2,
        "rules_version": RULES_VERSION,
        "scenario_id": "",
        "seed": seed,
        "preset": config.preset,
        "shift_ticks": config.shift_ticks,
        "factors": list(config.factors),
        "network": network,
        "trains": trains,
        "events": events,
        "connections": connections,
    }
    payload["scenario_id"] = f"{config.preset}-{content_hash({k: v for k, v in payload.items() if k != 'scenario_id'})[:12]}"
    return Scenario.model_validate(payload)
