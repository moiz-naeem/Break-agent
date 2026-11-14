from __future__ import annotations

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


def _rng(seed: int, stage: str) -> random.Random:
    return random.Random(f"{seed}:{stage}")


def _network(config: GenConfig, seed: int):
    rng = _rng(seed, "network")
    stations = []
    for i in range(1, config.stations + 1):
        count = rng.choices(list(config.loops), weights=list(config.loops.values()), k=1)[0]
        loops = []
        for j in range(count):
            length = rng.choices([150, 300, 700], weights=[0.25, 0.45, 0.30], k=1)[0]
            kind = "goods" if length == 700 and rng.random() < 0.5 else "platform"
            loops.append({"id": f"L{j + 1}", "length_m": length, "kind": kind})
        stations.append({"id": f"S{i}", "major": False, "crew_station": False, "loops": loops})

    def add_long_loop(index: int):
        station = stations[index]
        existing = next((loop for loop in station["loops"] if loop["length_m"] == 700), None)
        if existing:
            return
        if len(station["loops"]) < 3:
            loop_id = f"L{len(station['loops']) + 1}"
            station["loops"].append({"id": loop_id, "length_m": 700, "kind": "goods"})
        else:
            station["loops"][-1].update(length_m=700, kind="goods")

    for third in range(3):
        lo = third * config.stations // 3
        hi = max(lo + 1, (third + 1) * config.stations // 3)
        if not any(any(loop["length_m"] == 700 for loop in stations[i]["loops"]) for i in range(lo, hi)):
            add_long_loop(lo)
    # A 600 m freight can only wait in a 700 m loop, and the reference (T06 §2.2) only sends a train somewhere
    # with another fitting loop at most 3 stations on: so never leave 3 stations in a row without one.
    gap = 0
    for i, station in enumerate(stations):
        gap = 0 if any(loop["length_m"] == 700 for loop in station["loops"]) else gap + 1
        if gap == 3:
            add_long_loop(i)
            gap = 0
    long_stations = [i for i, st in enumerate(stations) if any(x["length_m"] == 700 for x in st["loops"])]
    for i in range(config.stations):
        if len(long_stations) >= math.ceil(config.stations / 3):
            break
        if i not in long_stations:
            add_long_loop(i)
            long_stations.append(i)
    for i in range(1, config.stations):
        if not stations[i - 1]["loops"] and not stations[i]["loops"]:
            stations[i]["loops"].append({"id": "L1", "length_m": 150, "kind": "platform"})

    major_count = max(2, round(config.stations * config.major_fraction))
    selected = set(rng.sample(range(config.stations), min(major_count, config.stations))) | {0, config.stations - 1}
    for i, station in enumerate(stations):
        station["major"] = i in selected
    if "F10" in config.factors:
        eligible = [i for i, station in enumerate(stations) if station["loops"]]
        count = min(len(eligible), rng.randint(2, 3))
        chosen = []
        for position in (0.2, 0.5, 0.8):
            if len(chosen) >= count:
                break
            idx = eligible[min(len(eligible) - 1, round(position * (len(eligible) - 1)))]
            if idx not in chosen:
                chosen.append(idx)
        for idx in eligible:
            if len(chosen) >= count:
                break
            if idx not in chosen:
                chosen.append(idx)
        for idx in chosen:
            stations[idx]["crew_station"] = True

    section_rng = _rng(seed, "sections")
    sections = []
    locations = ["YW", *[station["id"] for station in stations], "YE"]
    for west, east in zip(locations, locations[1:], strict=False):
        sections.append(
            {
                "id": f"{west}-{east}",
                "west": west,
                "east": east,
                "length_m": section_rng.choice(range(4000, 9001, 500)),
                "line_speed_kmh": section_rng.choice([80, 100, 120, 140]),
            }
        )
    return {"yards": {"west": "YW", "east": "YE"}, "stations": stations, "sections": sections}


def _type_counts(config: GenConfig):
    exact = {name: config.trains * weight for name, weight in config.mix.items()}
    counts = {name: math.floor(value) for name, value in exact.items()}
    remainder = config.trains - sum(counts.values())
    order = sorted(exact, key=lambda name: (-(exact[name] - counts[name]), name))
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def _trains(config: GenConfig, network: dict, seed: int):
    rng = _rng(seed, "trains")
    timetable_rng = _rng(seed, "timetable")
    types = []
    prefix = {"express": "E", "local": "L", "freight": "F"}
    for kind, count in _type_counts(config).items():
        types.extend((kind, f"{prefix[kind]}{i}") for i in range(1, count + 1))
    shuffled = list(types)
    rng.shuffle(shuffled)
    first_direction = rng.choice(["east", "west"])
    directions = {
        tid: (first_direction if idx % 2 == 0 else ("west" if first_direction == "east" else "east"))
        for idx, (_, tid) in enumerate(shuffled)
    }
    sections = network["sections"]
    stations = network["stations"]
    output = []
    for kind, tid in types:
        direction = directions[tid]
        eastbound = direction == "east"
        route_sections = sections if eastbound else list(reversed(sections))
        stops = []
        relative_tick = 0
        destination = "YE" if eastbound else "YW"
        pax = (
            timetable_rng.randrange(150, 401, 10)
            if kind == "express"
            else timetable_rng.randrange(50, 201, 10)
            if kind == "local"
            else 0
        )
        for section in route_sections:
            relative_tick += travel_ticks(
                section["length_m"], section["line_speed_kmh"], TRAIN_TYPES[kind]["max_speed_kmh"]
            )
            # Schedule each section with the configured slack.
            relative_tick += math.ceil(
                travel_ticks(section["length_m"], section["line_speed_kmh"], TRAIN_TYPES[kind]["max_speed_kmh"])
                * config.slack
            )
            next_location = section["east"] if eastbound else section["west"]
            terminal = next_location == destination
            should_stop = (
                terminal
                or (kind == "local" and next_location.startswith("S"))
                or (
                    kind == "express"
                    and next_location.startswith("S")
                    and stations[int(next_location[1:]) - 1]["major"]
                )
            )
            if should_stop:
                dwell = 0 if terminal else (1 if kind == "express" else timetable_rng.randint(1, 2))
                arr = relative_tick
                stops.append(
                    {
                        "station": next_location,
                        "arr": arr,
                        "dep": None if terminal else arr + dwell,
                        "min_dwell": 0 if terminal else dwell,
                    }
                )
                relative_tick += dwell
        run_length = stops[-1]["arr"]
        latest_entry = config.shift_ticks - run_length - config.end_margin
        if latest_entry < 0:
            raise ValueError(f"shift too short for {config.preset} run length {run_length}")
        entry = timetable_rng.randint(0, latest_entry)
        # Stop times are absolute ticks (the simulator's convention), offset from the entry tick.
        for stop in stops:
            stop["arr"] += entry
            if stop["dep"] is not None:
                stop["dep"] += entry
        output.append(
            {
                "id": tid,
                "type": kind,
                "direction": direction,
                "pax": pax,
                "entry_tick": entry,
                "stops": stops,
                "next_service": None,
                "crew_used_at_entry": 0,
            }
        )

    return output


def run_length(train: dict) -> int:
    """Scheduled run length; stop times are absolute, so subtract the entry tick."""
    return train["stops"][-1]["arr"] - train["entry_tick"]


def _rotations(trains: list[dict], fraction: float):
    count = round(len(trains) * fraction)
    links = 0
    used_successors = set()
    arrivals = sorted(trains, key=lambda train: train["stops"][-1]["arr"])
    for before in arrivals:
        if links >= count:
            break
        if before["next_service"] is not None:
            continue
        yard = before["stops"][-1]["station"]
        min_entry = before["stops"][-1]["arr"] + TURNAROUND
        after = next(
            (
                candidate
                for candidate in sorted(trains, key=lambda t: (t["entry_tick"], t["id"]))
                if candidate["id"] not in used_successors
                and candidate["id"] != before["id"]
                and candidate["entry_tick"] >= min_entry
                and (
                    (candidate["direction"] == "west" and yard == "YE")
                    or (candidate["direction"] == "east" and yard == "YW")
                )
            ),
            None,
        )
        if after:
            before["next_service"] = after["id"]
            used_successors.add(after["id"])
            links += 1


def _events(config: GenConfig, network: dict, trains: list[dict], seed: int):
    output = []
    used_late_entry_trains = set()
    kinds = [("F7", "possession"), ("F8", "speed_restriction"), ("F15", "breakdown"), ("F16", "late_entry")]
    for factor, kind in kinds:
        if factor not in config.factors:
            continue
        rng = _rng(seed, f"events-{kind}")
        count = max(1, round(config.event_density * config.shift_ticks / 120))
        for _ in range(count):
            event = {"kind": kind}
            if kind == "possession":
                section = rng.choice(network["sections"])
                start = rng.randint(60, config.shift_ticks - 40)
                duration = rng.randint(15, min(40, config.shift_ticks - start))
                event.update(
                    section=section["id"],
                    start=start,
                    end=start + duration,
                    reveal_tick=max(0, start - rng.randint(*POSSESSION_LOOKAHEAD)),
                )
                target = section["id"]
            elif kind == "speed_restriction":
                start = rng.randint(30, config.shift_ticks - 30)
                sections = network["sections"]
                index = rng.randrange(len(sections))
                covered = [sections[index]["id"]]
                if index + 1 < len(sections) and rng.random() < 0.5:
                    covered.append(sections[index + 1]["id"])
                event.update(
                    sections=covered,
                    speed_kmh=rng.choice([40, 60]),
                    start=start,
                    end=start + rng.randint(30, 90),
                    reveal_tick=max(0, start - rng.randint(*RESTRICTION_LOOKAHEAD)),
                )
                target = covered[0]
            elif kind == "breakdown":
                train = rng.choice(trains)
                start = train["entry_tick"] + 5
                end = train["stops"][-1]["arr"] - 5
                if end < start:
                    continue
                tick = rng.randint(start, end)
                event.update(train=train["id"], tick=tick, reveal_tick=tick, duration=rng.randint(5, 25))
                target = train["id"]
            else:
                eligible = [
                    train
                    for train in trains
                    if train["entry_tick"] >= LATE_ENTRY_LOOKAHEAD and train["id"] not in used_late_entry_trains
                ]
                if not eligible:
                    continue
                train = rng.choice(eligible)
                used_late_entry_trains.add(train["id"])
                event.update(
                    train=train["id"],
                    new_entry_tick=train["entry_tick"] + rng.randint(10, 40),
                    reveal_tick=train["entry_tick"] - LATE_ENTRY_LOOKAHEAD,
                )
                target = train["id"]
            event["_sort"] = (event["reveal_tick"], kind, target)
            output.append(event)
    output.sort(key=lambda event: event["_sort"])
    for i, event in enumerate(output, start=1):
        event["id"] = f"EV{i}"
        del event["_sort"]
    output.sort(key=lambda event: (event["reveal_tick"], event["id"]))
    return output


def generate(config: GenConfig, seed: int) -> Scenario:
    network = _network(config, seed)
    trains = _trains(config, network, seed)
    if "F6" in config.factors:
        _rotations(trains, config.rotation_fraction)
    if "F10" in config.factors:
        rng = _rng(seed, "crew")
        for train in trains:
            if train["next_service"]:
                train["crew_used_at_entry"] = 0
            elif rng.random() < config.tight_crew_fraction:
                train["crew_used_at_entry"] = max(0, CREW_LIMIT - int(run_length(train) * rng.uniform(0.4, 0.8)))
            else:
                train["crew_used_at_entry"] = rng.randint(0, 100)
    trains.sort(key=lambda train: train["id"])
    events = _events(config, network, trains, seed)
    payload = {
        "schema_version": 1,
        "rules_version": RULES_VERSION,
        "scenario_id": "",
        "seed": seed,
        "preset": config.preset,
        "shift_ticks": config.shift_ticks,
        "factors": list(config.factors),
        "network": network,
        "trains": trains,
        "events": events,
    }
    payload["scenario_id"] = (
        f"{config.preset}-{content_hash({k: v for k, v in payload.items() if k != 'scenario_id'})[:12]}"
    )
    return Scenario.model_validate(payload)
