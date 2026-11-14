"""The three generator presets (docs/network-v0.3.md §6).

T1 must come out byte for byte as it did before the generator was sped up. T2 and T3 must be valid layouts with
crew-relief points, a way onto and off the wrong road wherever single-line working is signalled, and (T3) a
bypass that can carry trains round the trunk both ways. Every tier must generate, on every seed, a shift that
the planner's own timetable-order run delivers in full with nobody late.
"""

from __future__ import annotations

import hashlib
from functools import cache

import pytest

from railway.net.engine import TURNAROUND_MIN
from railway.net.model import Network, Timetable, Track
from railway.net.routing import find_route, portal_track
from railway.net.templates import TIERS, t1_services, trunk_and_branches, two_hubs_network
from railway.net.timetable import HIGH, generate, infeasible, peak_utilisation
from railway.net.trains import TYPES

IC = TYPES["IC"].spec("X1")
FB = TYPES["FB"].spec("X2")

# sha256 of Timetable.model_dump_json() for T1 at 180 min, generated at 40b861a, before the speed-up.
T1_BEFORE_SPEED_UP = {
    1: "b5684ba0ea0938f8b37a4e82968e8016426cbbc2ee50fc313e1977aa736f4720",
    2: "0a962e1ab58cc9fba62f7b64dd0958babf053a8e36e882a7cf1fb4573fa28ca8",
    3: "cef4044e2c056e7291f3839b8cd14307f1c98cbb97c5cc39ff770fbce05c089d",
    4: "092a2fbd02fe1bf129deade8e9e8bf6d5ae3929c64fea65cb677704381b2a500",
}

# Trains per shift at the default shift: §6's bands, widened at the bottom to what the layouts really carry
# (T1 16-19, T2 56-59, T3 108-118 over seeds 1-8).
SIZE = {"T1": (12, 35), "T2": (50, 80), "T3": (100, 160)}
HUBS = {"T2": {"H"}, "T3": {"HA", "HB"}}
TERMINALS = {"T2": {"TE", "TS"}, "T3": {"TA", "TB"}}

# Where single_line() can work each tier's reversible stretch: (stretch, up train from/facing/to, down likewise).
REVERSIBLE = {
    "T1": ("JA-B", ("A-JA.u", "JA", "B-JB.u"), ("B-JB.d", "B", "A-JA.d")),
    "T2": ("XW-JS", ("W-XW.u", "XW", "JS-H.u"), ("JS-H.d", "JS", "W-XW.d")),
    "T3": ("M-JB", ("JA-M.u", "M", "JB-HB.u"), ("JB-HB.d", "JB", "JA-M.d")),
}


@cache
def build(tier: str, seed: int) -> Timetable:
    return TIERS[tier].timetable(seed)


def _rebuilt(net: Network, **changes: dict) -> Network:
    """The same layout with some tracks changed, validated afresh."""
    tracks = [Track(**{**t.model_dump(), **changes.get(t.id, {})}) for t in net.tracks]
    return Network(nodes=net.nodes, tracks=tracks, platforms=net.platforms, moves=net.moves, crossings=net.crossings)


def _closed(net: Network, *track_ids: str) -> Network:
    """Tracks an electric train cannot use: how the tests take a track out of service without a disruption."""
    return _rebuilt(net, **{t: {"electrified": False} for t in track_ids})


# ---------------------------------------------------------------- the registry and T1


def test_the_tier_registry_names_each_preset_and_its_shift():
    assert {name: tier.shift_min for name, tier in TIERS.items()} == {"T1": 180, "T2": 240, "T3": 360}
    assert all(name == tier.name for name, tier in TIERS.items())
    assert TIERS["T1"].network() == trunk_and_branches()
    assert TIERS["T1"].services() == t1_services()


@pytest.mark.parametrize("seed", sorted(T1_BEFORE_SPEED_UP))
def test_t1_is_byte_identical_to_the_generator_before_the_speed_up(seed):
    timetable = TIERS["T1"].timetable(seed)
    assert hashlib.sha256(timetable.model_dump_json().encode()).hexdigest() == T1_BEFORE_SPEED_UP[seed]


def test_the_route_cache_follows_the_network_it_was_asked_about():
    """Routes are cached per network: a network with a track out of service must not be handed the old route."""
    net = two_hubs_network()
    trunk = find_route(net, "HA1", "HA", "HB1", IC, "a")
    assert "JA-M.u" in trunk
    diverted = find_route(_closed(net, "JA-M.u"), "HA1", "HA", "HB1", IC, "a")
    assert "JA-M.u" not in diverted
    assert find_route(net, "HA1", "HA", "HB1", IC, "a") == trunk
    trunk.append("tampered")  # callers get their own list, never the cached one
    assert find_route(net, "HA1", "HA", "HB1", IC, "a")[-1] == "HB1"


# ---------------------------------------------------------------- T2 and T3 layouts


@pytest.mark.parametrize("tier", ["T2", "T3"])
def test_the_layout_has_the_size_and_parts_section_6_asks_for(tier):
    net = TIERS[tier].network()  # building it runs every validator in railway.net.model
    kinds = {n.id: n.kind for n in net.nodes}
    per_hub = {hub: sum(p.node == hub for p in net.platforms) for hub in HUBS[tier]}
    low, high = (8, 10) if tier == "T2" else (6, 8)
    assert all(low <= n <= high for n in per_hub.values()), per_hub
    assert {n for n, k in kinds.items() if k == "terminal"} == TERMINALS[tier]
    assert sum(k == "portal" for k in kinds.values()) >= 3
    throats = 2 * len(HUBS[tier])  # §6 counts junctions with the hub throats: "5 + hub throat", "9+"
    assert sum(k == "junction" for k in kinds.values()) + throats >= (6 if tier == "T2" else 9)
    assert any(t.direction == "both" for t in net.tracks), "a single-track branch"
    assert any(n.id in {"RN", "NA", "NB"} and n.kind == "station" for n in net.nodes), "its crossing station"
    long_roads = [p for p in net.platforms if p.node in HUBS[tier] and p.length_m >= 750]
    assert long_roads, "a freight road through the hub that a 740 m train fits"
    assert net.crossings, "conflicting moves that share no points"


@pytest.mark.parametrize("tier", ["T2", "T3"])
def test_the_port_is_diesel_only(tier):
    net = TIERS[tier].network()
    port = [p for p in net.platforms if p.node == "PORT"]
    assert port and not any(p.electrified for p in port)
    assert all(not t.electrified for t in net.tracks if "PORT" in (t.a, t.b))
    entry = portal_track(net, "PW", inbound=True)
    facing = net.element(entry).other("PW")
    assert find_route(net, entry, facing, "P1", IC) is None
    assert find_route(net, entry, facing, "P1", FB) is not None


@pytest.mark.parametrize("tier", ["T2", "T3"])
def test_crews_are_relieved_at_every_hub_and_terminal_platform_and_nowhere_else(tier):
    net = TIERS[tier].network()
    relief_nodes = HUBS[tier] | TERMINALS[tier]
    for platform in net.platforms:
        assert platform.crew_relief == (platform.node in relief_nodes), platform.id


@pytest.mark.parametrize("tier", sorted(REVERSIBLE))
def test_single_line_working_has_a_way_onto_and_off_the_wrong_road(tier):
    """With one track of the reversible stretch out, trains both ways run over the other, through crossovers.

    Opening a track to both directions is what single_line() does; until then nothing runs the wrong way.
    """
    stretch, up, down = REVERSIBLE[tier]
    net = TIERS[tier].network()
    assert {t.id for t in net.tracks if t.reversible} == {f"{stretch}.u", f"{stretch}.d"}
    for (start, facing, target), normal, other in ((up, "u", "d"), (down, "d", "u")):
        route = find_route(net, start, facing, target, IC)
        assert f"{stretch}.{normal}" in route and f"{stretch}.{other}" not in route
        assert find_route(_closed(net, f"{stretch}.{normal}"), start, facing, target, IC) is None
        single = _rebuilt(net, **{f"{stretch}.{normal}": {"electrified": False},
                                  f"{stretch}.{other}": {"direction": "both", "reversible": False}})  # fmt: skip
        wrong_road = find_route(single, start, facing, target, IC)
        assert wrong_road is not None and f"{stretch}.{other}" in wrong_road, (tier, start, target)


def test_t3_trains_can_be_diverted_over_the_bypass_both_ways():
    """The trunk and the bypass are two distinct ways between the hubs, so route(train, via=[...]) has a choice."""
    net = two_hubs_network()
    trunk = {t.id for t in net.tracks if t.id.startswith(("JA-M.", "M-JB."))}
    bypass = {t.id for t in net.tracks if t.id.startswith(("JA-BYP.", "BYP-JP.", "JP-JB."))}
    closed = _closed(net, *trunk)
    for start, side, target, direction in (("HA1", "a", "HB1", "u"), ("HB3", "b", "HA3", "d")):
        node = start[:2]
        usual = set(find_route(net, start, node, target, IC, side))
        assert usual & trunk and not usual & bypass
        diverted = set(find_route(closed, start, node, target, IC, side))
        assert diverted & bypass == {f"{t}.{direction}" for t in ("JA-BYP", "BYP-JP", "JP-JB")}
        assert not diverted & trunk
    to_byp = find_route(net, "HA1", "HA", "BYP1", IC, "a")  # via the bypass platform, as route(via=) asks
    onward = find_route(net, "BYP1", "BYP", "HB1", IC, net.element(to_byp[-2]).side)
    assert to_byp is not None and onward is not None and not set(to_byp + onward) & trunk


@pytest.mark.parametrize("tier", ["T2", "T3"])
def test_every_service_group_can_be_routed(tier):
    """Each group's portals and stops form a route the engine can find, whatever the seed does to its times."""
    net = TIERS[tier].network()
    core, fill = TIERS[tier].services()
    for group in core + fill:
        spec = TYPES[group.type].spec(f"{group.code}01")
        position = portal_track(net, group.entry_node, inbound=True)
        facing, side = net.element(position).other(group.entry_node), None
        for stop, dwell in zip(group.stops, group.dwell, strict=True):
            path = find_route(net, position, facing, stop, spec, side)
            assert path is not None, (group.code, position, stop)
            position, facing = stop, net.element(stop).node
            side = None if dwell >= TURNAROUND_MIN else net.element(path[-2]).side
        if group.exit_node:
            out = portal_track(net, group.exit_node, inbound=False)
            assert find_route(net, position, facing, out, spec, side) is not None, (group.code, "exit")


# ---------------------------------------------------------------- generated shifts


GENERATED = [(tier, seed) for tier in ("T1", "T2", "T3") for seed in range(1, 9)]


@pytest.mark.slow
@pytest.mark.parametrize(("tier", "seed"), GENERATED, ids=[f"{t}-{s}" for t, s in GENERATED])
def test_every_seed_generates_a_shift_the_planner_runs_clean(tier, seed):
    """Undisturbed and settled in timetable order, every train finishes its work with nobody late (§4.1)."""
    timetable = build(tier, seed)
    assert (timetable.tier, timetable.seed, timetable.shift_min) == (tier, seed, TIERS[tier].shift_min)
    assert infeasible(timetable) == []
    low, high = SIZE[tier]
    assert low <= len(timetable.trips) <= high
    assert 0.5 <= peak_utilisation(timetable.net, timetable.trips, timetable.shift_min) <= HIGH
    assert timetable.connections
    if tier != "T1":  # T1's branch fills up before every seed fits a freight path
        assert any(not t.train.passenger for t in timetable.trips), "freight fills the gaps"
        assert len(timetable.connections) >= len(timetable.trips), "the hubs are where passengers change"


@pytest.mark.slow
@pytest.mark.parametrize("tier", ["T2", "T3"])
def test_a_tier_generates_the_same_shift_from_the_same_seed(tier):
    net = TIERS[tier].network()
    core, fill = TIERS[tier].services()
    again = generate(net, core, fill, TIERS[tier].shift_min, 1, tier)
    assert again.model_dump_json() == build(tier, 1).model_dump_json()
    assert again.model_dump_json() != build(tier, 2).model_dump_json()
