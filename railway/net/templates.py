"""Layout templates (docs/network-v0.3.md §6). The generator will perturb these; here they are fixed."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .model import Call, Move, Network, Node, Platform, ServiceGroup, Timetable, Track, Trip
from .trains import TYPES


def block_m(line_speed_kmh: int) -> int:
    """Signal blocks are spaced by braking distance, so a fast line is divided into longer blocks."""
    return 3000 if line_speed_kmh >= 140 else 2000 if line_speed_kmh >= 100 else 1500


CREW_RELIEF = frozenset({"A1", "A2", "A3", "B3", "C1", "C2", "C3"})


def trunk_and_branches() -> Network:
    """T1: double-track trunk PW-A-JA-B-JB-C, terminal C, single-track branch JA-N-PN, diesel-only port spur at JB.

    N is the branch crossing station: trains going opposite ways over the single track pass each other there.

    A3 is the through road: the long freight that does not fit a platform runs past the station on it.

    JA is a flat junction: up trains turning onto the branch cross the down main. JB is a single lead: the spur's
    two directions share one turnout. B has a 750 m loop where slow trains step aside.

    JA-B, the longest stretch of double track, is signalled both ways, with crossovers at JA and B, so single_line()
    can work both directions over one track while the other is blocked. Nothing can run the wrong way until then.
    An up train on the wrong road can still call at B1; a down train on it runs through B without calling, because
    a move from B2 onto the up line would let up trains into the down platform.
    Crews are relieved at A, at the freight loop B3 and at the terminal C.
    """
    nodes = [
        Node(id=i, kind=k)
        for i, k in [
            ("PW", "portal"), ("A", "station"), ("JA", "junction"), ("B", "station"), ("JB", "junction"),
            ("C", "terminal"), ("N", "station"), ("PN", "portal"), ("PORT", "freight_terminal"),
        ]
    ]  # fmt: skip
    tracks = [
        Track(id=f"{a}-{b}.{s}", a=a, b=b, length_m=km * 1000, line_speed_kmh=v, electrified=True, direction=d,
              block_m=block_m(v), reversible=(a, b) == ("JA", "B"))
        for a, b, km, v in [("PW", "A", 12, 160), ("A", "JA", 4, 160), ("JA", "B", 15, 160), ("B", "JB", 9, 160),
                            ("JB", "C", 5, 120)]
        for s, d in (("u", "ab"), ("d", "ba"))
    ]  # fmt: skip
    tracks += [
        Track(id=i, a=a, b=b, length_m=km * 1000, line_speed_kmh=v, electrified=e, direction="both", block_m=block_m(v))
        for i, a, b, km, v, e in [("JA-N", "JA", "N", 10, 100, True), ("N-PN", "N", "PN", 8, 100, True),
                                  ("JB-PORT", "JB", "PORT", 6, 60, False)]
    ]  # fmt: skip
    platforms = [
        Platform(id=i, node=n, length_m=m, kind=k, electrified=n != "PORT", crew_relief=i in CREW_RELIEF)
        for i, n, m, k in [
            ("A1", "A", 270, "platform"), ("A2", "A", 270, "platform"), ("A3", "A", 800, "through"),
            ("B1", "B", 270, "platform"), ("B2", "B", 270, "platform"), ("B3", "B", 750, "siding"),
            ("C1", "C", 260, "platform"), ("C2", "C", 260, "platform"), ("C3", "C", 180, "platform"),
            ("N1", "N", 200, "platform"), ("N2", "N", 750, "platform"),  # N2 is the loop they cross in
            ("P1", "PORT", 750, "siding"), ("P2", "PORT", 750, "siding"),
        ]
    ]  # fmt: skip
    moves = [  # id, node, ends, turnout speed (km/h), platform side ("a" west/approach, "b" east)
        ("A.w1", "A", "PW-A.u", "A1", 60, "a"), ("A.w2", "A", "A2", "PW-A.d", 60, "a"),
        ("A.w3", "A", "PW-A.u", "A2", 40, "a"),
        ("A.e1", "A", "A1", "A-JA.u", 60, "b"), ("A.e2", "A", "A-JA.d", "A2", 60, "b"),
        ("A.e3", "A", "A2", "A-JA.u", 40, "b"),
        ("A.w4", "A", "PW-A.u", "A3", 60, "a"), ("A.w5", "A", "A3", "PW-A.d", 60, "a"),
        ("A.e4", "A", "A3", "A-JA.u", 60, "b"), ("A.e5", "A", "A-JA.d", "A3", 60, "b"),
        ("JA.m1", "JA", "A-JA.u", "JA-B.u", 160, None), ("JA.m2", "JA", "JA-B.d", "A-JA.d", 160, None),
        ("JA.m3", "JA", "A-JA.u", "JA-N", 80, None), ("JA.m4", "JA", "JA-N", "A-JA.d", 80, None),
        ("JA.x1", "JA", "JA-B.u", "A-JA.d", 40, None), ("JA.x2", "JA", "A-JA.u", "JA-B.d", 40, None),  # crossovers
        ("B.w1", "B", "JA-B.u", "B1", 60, "a"), ("B.w2", "B", "JA-B.u", "B3", 40, "a"),
        ("B.w3", "B", "B2", "JA-B.d", 60, "a"),
        ("B.e1", "B", "B1", "B-JB.u", 60, "b"), ("B.e2", "B", "B3", "B-JB.u", 40, "b"),
        ("B.e3", "B", "B-JB.d", "B2", 60, "b"),
        ("B.w4", "B", "JA-B.d", "B1", 40, "a"), ("B.x1", "B", "B-JB.d", "JA-B.u", 40, None),  # wrong-road
        ("JB.m1", "JB", "B-JB.u", "JB-C.u", 160, None), ("JB.m2", "JB", "JB-C.d", "B-JB.d", 160, None),
        ("JB.m3", "JB", "B-JB.u", "JB-PORT", 40, None), ("JB.m4", "JB", "JB-PORT", "B-JB.d", 40, None),
        ("C.a1", "C", "JB-C.u", "C1", 40, "a"), ("C.a2", "C", "JB-C.u", "C2", 40, "a"),
        ("C.a3", "C", "JB-C.u", "C3", 40, "a"),
        ("C.d1", "C", "C1", "JB-C.d", 40, "a"), ("C.d2", "C", "C2", "JB-C.d", 40, "a"),
        ("C.d3", "C", "C3", "JB-C.d", 40, "a"),
        ("N.w1", "N", "JA-N", "N1", 60, "a"), ("N.w2", "N", "JA-N", "N2", 40, "a"),
        ("N.e1", "N", "N1", "N-PN", 60, "b"), ("N.e2", "N", "N2", "N-PN", 40, "b"),
        ("P.1", "PORT", "JB-PORT", "P1", 25, "a"), ("P.2", "PORT", "JB-PORT", "P2", 25, "a"),
    ]  # fmt: skip
    # Moves sharing points conflict automatically; these cross without sharing points.
    crossings = [
        ("JA.m2", "JA.m3"),  # flat junction: up trains turning onto the branch cross the down main
        ("JA.x1", "JA.x2"),  # the two crossovers form a scissors
        ("JB.m2", "JB.m3"),  # same at the port spur
        ("C.a2", "C.d1"), ("C.a3", "C.d1"), ("C.a3", "C.d2"),  # terminal throat: arrivals cross departures
    ]  # fmt: skip
    return Network(
        nodes=nodes,
        tracks=tracks,
        platforms=platforms,
        moves=[Move(id=i, node=n, ends=(e1, e2), speed_kmh=v, side=s) for i, n, e1, e2, v, s in moves],
        crossings=crossings,
    )


def t1_services() -> tuple[list[ServiceGroup], list[ServiceGroup]]:
    """The T1 service (§6): the core clockface lines, and the paths the generator fills the gaps with.

    The intercity and the regional express turn at the terminal C and run back west, so each is a rotation: a
    late arrival at C carries straight into the return service. The two commuter lines share the flat junction
    JA with the trunk in both directions, which is where most of the agent's decisions come from.
    """
    core = [
        ServiceGroup(code="IC", type="IC", entry_node="PW", exit_node="PW", first_min=5, every_min=25,
                     stops=("A1", "C1", "B2"), dwell=(2, 12, 2)),
        ServiceGroup(code="SE", type="S", entry_node="PW", exit_node="PN", first_min=0, every_min=16,
                     stops=("A1", "N1"), dwell=(2, 2)),
        ServiceGroup(code="SW", type="S", entry_node="PN", exit_node="PW", first_min=8, every_min=16,
                     stops=("N2",), dwell=(2,)),
        ServiceGroup(code="RE", type="RE", entry_node="PW", exit_node="PW", first_min=20, every_min=60,
                     stops=("A1", "B1", "C3", "B2"), dwell=(2, 2, 10, 2)),
    ]  # fmt: skip
    fill = [
        ServiceGroup(code="FB", type="FB", entry_node="PW", first_min=12, every_min=55,
                     stops=("B3", "P1"), dwell=(4, 0)),
        ServiceGroup(code="FI", type="FI", entry_node="PW", first_min=40, every_min=70,
                     stops=("B3", "P2"), dwell=(4, 0)),
        ServiceGroup(code="FN", type="FI", entry_node="PW", exit_node="PN", first_min=25, every_min=50,
                     stops=("N2",), dwell=(3,)),
        ServiceGroup(code="FW", type="FB", entry_node="PN", exit_node="PW", first_min=35, every_min=50,
                     stops=("N2",), dwell=(3,)),
    ]  # fmt: skip
    return core, fill


def sample_trips() -> list[Trip]:
    """A small T1 service for tests: an eastbound intercity, a branch commuter and a port freight.

    IC1 and L4 leave A at the same minute and both need the up line A-JA.u, where IC1 carries straight on and L4
    turns across the flat junction JA. ARS cannot choose between them, so the agent must.
    """
    return [
        Trip(
            train=TYPES["IC"].spec("IC1"),
            entry_node="PW",
            entry_tick=0,
            calls=(Call(platform="A1", arrive=8, depart=10), Call(platform="C1", arrive=40)),
        ),
        Trip(
            train=TYPES["S"].spec("L4"),
            calls=(Call(platform="A2", depart=10), Call(platform="N1", arrive=30)),
        ),
        Trip(
            train=TYPES["FB"].spec("F3"),
            calls=(Call(platform="B3", depart=12), Call(platform="P1", arrive=60)),
        ),
    ]


# ---------------------------------------------------------------------- building blocks for T2 and T3
#
# T1 is written out move by move above. The larger tiers are built from the same pieces, so these helpers spell
# the conventions out once. West is the "a" end of a node and of a platform, and "u" (up) runs west to east.
# Wherever tracks and platforms are listed in order, the order is north to south, with the down line north of
# the up line. A move row is (id, node, end, end, turnout speed km/h, platform side).
#
# Tracks from a portal to its first station are kept short (3 km): a train entering the area is given its route
# all the way into the first platform, so a long approach would hold that platform for the whole run in and cap
# how many trains a line can feed in.

MoveRow = tuple[str, str, str, str, int, str | None]
Crossing = tuple[str, str]


def _double(
    a: str, b: str, km: float, v: int, *, reversible: bool = False, electrified: bool = True, pair: str = ""
) -> list[Track]:
    """Double track: `a-b.u` runs a to b, `a-b.d` runs b to a. `pair` names a second pair on the same route."""
    return [
        Track(id=f"{a}-{b}.{pair}{s}", a=a, b=b, length_m=round(km * 1000), line_speed_kmh=v,
              electrified=electrified, direction=d, block_m=block_m(v), reversible=reversible)
        for s, d in (("u", "ab"), ("d", "ba"))
    ]  # fmt: skip


def _single(a: str, b: str, km: float, v: int, *, electrified: bool = True) -> Track:
    return Track(id=f"{a}-{b}", a=a, b=b, length_m=round(km * 1000), line_speed_kmh=v, electrified=electrified,
                 direction="both", block_m=block_m(v))  # fmt: skip


def _throat(
    node: str,
    side: str,
    tracks: list[str],
    platforms: list[str],
    links: list[tuple[str, str]],
    fast: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[list[MoveRow], list[Crossing]]:
    """One end of a station: a move for each (platform, track) link, and the crossings their geometry implies.

    `tracks` and `platforms` are each in north-to-south order. Two moves whose track and platform orders are
    reversed cross on the ground (a diamond): a train running from the south track to a northern platform cuts
    across one running from a southern platform to a more northern track. Moves sharing a track or a platform end
    already conflict through their points. A link in `fast` is the straight road (80 km/h), the rest are 40.
    """
    letter = "w" if side == "a" else "e"
    rows: list[MoveRow] = []
    at: list[tuple[str, int, int]] = []
    for k, (platform, track) in enumerate(links, start=1):
        move_id = f"{node}.{letter}{k}"
        rows.append((move_id, node, track, platform, 80 if (platform, track) in fast else 40, side))
        at.append((move_id, tracks.index(track), platforms.index(platform)))
    crossings = [
        (m1, m2) for i, (m1, t1, p1) in enumerate(at) for m2, t2, p2 in at[i + 1 :] if (t1 - t2) * (p1 - p2) < 0
    ]
    return rows, crossings


def _station(
    node: str,
    west: tuple[str, str],
    east: tuple[str, str],
    up: str,
    down: str,
    through: str | None = None,
) -> tuple[list[MoveRow], list[Crossing]]:
    """A through station on double track: an up and a down platform with straight roads, and a through road for
    long trains used both ways. `west` and `east` are each (down track, up track)."""
    (w_down, w_up), (e_down, e_up) = west, east
    order = [down, *([through] if through else []), up]
    road = [] if through is None else [(through, w_down), (through, w_up)]
    w, w_x = _throat(node, "a", [w_down, w_up], order, [(down, w_down), *road, (up, w_up)],
                     frozenset({(down, w_down), (up, w_up)}))  # fmt: skip
    road = [] if through is None else [(through, e_down), (through, e_up)]
    e, e_x = _throat(node, "b", [e_down, e_up], order, [(down, e_down), *road, (up, e_up)],
                     frozenset({(down, e_down), (up, e_up)}))  # fmt: skip
    return w + e, w_x + e_x


def _terminal(node: str, arrive: str, depart: str, platforms: tuple[str, ...]) -> tuple[list[MoveRow], list[Crossing]]:
    """Bay platforms at the end of a line, all reached from one throat, where each arrival crosses the departures
    from the platforms before it (as at T1's terminal C)."""
    rows: list[MoveRow] = [(f"{node}.a{i}", node, arrive, p, 40, "a") for i, p in enumerate(platforms, start=1)]
    rows += [(f"{node}.d{i}", node, p, depart, 40, "a") for i, p in enumerate(platforms, start=1)]
    crossings = [(f"{node}.a{i}", f"{node}.d{j}") for i in range(2, len(platforms) + 1) for j in range(1, i)]
    return rows, crossings


def _loop_station(node: str, west: str, east: str, platforms: tuple[str, ...]) -> list[MoveRow]:
    """A crossing station on single track: every platform reachable from both ends."""
    rows: list[MoveRow] = []
    for i, p in enumerate(platforms, start=1):
        v = 60 if i == 1 else 40
        rows += [(f"{node}.w{i}", node, west, p, v, "a"), (f"{node}.e{i}", node, p, east, v, "b")]
    return rows


def _crossovers(node: str, west: str, east: str) -> tuple[list[MoveRow], Crossing]:
    """A scissors crossover between double track `west` (ending here) and `east` (starting here).

    Unused while both tracks run their normal way; once single_line() opens a reversible track to both directions,
    these are how a train gets onto and off the wrong road.
    """
    rows: list[MoveRow] = [
        (f"{node}.x1", node, f"{west}.u", f"{east}.d", 40, None),
        (f"{node}.x2", node, f"{east}.u", f"{west}.d", 40, None),
    ]
    return rows, (f"{node}.x1", f"{node}.x2")


def _network(
    nodes: list[tuple[str, str]],
    tracks: list[Track],
    platforms: list[tuple[str, str, int, str]],
    moves: list[MoveRow],
    crossings: list[Crossing],
    *,
    relief: frozenset[str],
    diesel: frozenset[str],
) -> Network:
    """Assemble a layout. A platform is electrified unless its node is in `diesel`."""
    return Network(
        nodes=[Node(id=i, kind=k) for i, k in nodes],
        tracks=tracks,
        platforms=[
            Platform(id=i, node=n, length_m=m, kind=k, electrified=n not in diesel, crew_relief=i in relief)
            for i, n, m, k in platforms
        ],
        moves=[Move(id=i, node=n, ends=(e1, e2), speed_kmh=v, side=s) for i, n, e1, e2, v, s in moves],
        crossings=crossings,
    )


# ---------------------------------------------------------------------- T2: hub


def hub_network() -> Network:
    """T2: a 10-platform hub H with four lines converging on its two throats (docs/network-v0.3.md §6).

    West throat: the fast main from PW (PW-W-XW-JS-H, 160 km/h) and the double-track line from the terminal TS,
    each on its own pair of tracks. XW-JS is signalled both ways, with scissors crossovers at XW and JS, so
    single_line() can work it over one track. At JS, a single-lead junction, the diesel-only port branch
    JS-YD-PORT leaves; YD is a yard with two 750 m roads where port trains cross.

    East throat: the main to E, and a second pair that carries the single-track regional branch. They part at
    JH, a flat junction: trains running onto the branch cross the down main at grade (trains off it pass under).
    RN is the branch's crossing station, with a 750 m loop. Beyond E, JE (flat) splits the main for the terminal
    TE and the east portal PE.

    The hub is laid out line by line, north to south: the TS line and branch half (west bay H5, east bay H8, H4
    down, H2 up), then the main half (H3 down, west bay H6, east bays H7 and H10, H9 the 800 m freight road, H1
    up). Each
    half is fed by its own pair of tracks at each throat, so trains on one line rarely cross another's path:
    the branch is worked through to TS, the main through to the west. Crews are relieved at every hub platform
    and at both terminals.
    """
    nodes = [
        ("PW", "portal"), ("W", "station"), ("XW", "junction"), ("JS", "junction"), ("H", "station"),
        ("JH", "junction"), ("E", "station"), ("JE", "junction"), ("TE", "terminal"), ("PE", "portal"),
        ("RN", "station"), ("PN", "portal"), ("YD", "yard"), ("PORT", "freight_terminal"), ("TS", "terminal"),
    ]  # fmt: skip
    tracks = [
        *_double("PW", "W", 3, 160), *_double("W", "XW", 12, 160), *_double("XW", "JS", 14, 160, reversible=True),
        *_double("JS", "H", 4, 120), *_double("TS", "H", 6, 120),
        *_double("H", "JH", 2, 80), *_double("H", "JH", 2, 80, pair="n"),
        *_double("JH", "E", 10, 140), *_double("E", "JE", 9, 140), *_double("JE", "TE", 3, 100),
        *_double("JE", "PE", 3, 160),
        _single("JH", "RN", 9, 100), _single("RN", "PN", 3, 100),
        _single("JS", "YD", 5, 60, electrified=False), _single("YD", "PORT", 4, 40, electrified=False),
    ]  # fmt: skip
    platforms = [
        ("W1", "W", 280, "platform"), ("W2", "W", 280, "platform"), ("W3", "W", 800, "through"),
        ("H1", "H", 320, "platform"), ("H2", "H", 300, "platform"), ("H3", "H", 320, "platform"),
        ("H4", "H", 300, "platform"), ("H5", "H", 220, "platform"), ("H6", "H", 220, "platform"),
        ("H7", "H", 220, "platform"), ("H8", "H", 220, "platform"), ("H9", "H", 800, "through"),
        ("H10", "H", 220, "platform"),
        ("E1", "E", 280, "platform"), ("E2", "E", 280, "platform"), ("E3", "E", 800, "through"),
        ("TE1", "TE", 280, "platform"), ("TE2", "TE", 270, "platform"), ("TE3", "TE", 220, "platform"),
        ("TE4", "TE", 180, "platform"),
        ("TS1", "TS", 220, "platform"), ("TS2", "TS", 220, "platform"), ("TS3", "TS", 180, "platform"),
        ("RN1", "RN", 220, "platform"), ("RN2", "RN", 750, "platform"),  # RN2 is the loop trains cross in
        ("Y1", "YD", 750, "siding"), ("Y2", "YD", 750, "siding"),
        ("P1", "PORT", 750, "siding"), ("P2", "PORT", 750, "siding"), ("P3", "PORT", 750, "siding"),
    ]  # fmt: skip
    hub_order = ["H5", "H8", "H4", "H2", "H3", "H6", "H7", "H10", "H9", "H1"]
    hub_w, hub_wx = _throat(
        "H", "a", ["TS-H.d", "TS-H.u", "JS-H.d", "JS-H.u"], hub_order,
        [("H6", "JS-H.d"), ("H6", "JS-H.u"), ("H3", "JS-H.d"), ("H9", "JS-H.d"), ("H9", "JS-H.u"),
         ("H1", "JS-H.u"), ("H4", "TS-H.d"), ("H2", "TS-H.u"), ("H5", "TS-H.d"), ("H5", "TS-H.u")],
        frozenset({("H3", "JS-H.d"), ("H1", "JS-H.u"), ("H4", "TS-H.d"), ("H2", "TS-H.u")}),
    )  # fmt: skip
    hub_e, hub_ex = _throat(
        "H", "b", ["H-JH.nd", "H-JH.nu", "H-JH.d", "H-JH.u"], hub_order,
        [("H7", "H-JH.d"), ("H7", "H-JH.u"), ("H10", "H-JH.d"), ("H10", "H-JH.u"), ("H3", "H-JH.d"),
         ("H9", "H-JH.d"), ("H9", "H-JH.u"),
         ("H1", "H-JH.u"), ("H4", "H-JH.nd"), ("H2", "H-JH.nu"), ("H8", "H-JH.nd"), ("H8", "H-JH.nu")],
        frozenset({("H3", "H-JH.d"), ("H1", "H-JH.u"), ("H4", "H-JH.nd"), ("H2", "H-JH.nu")}),
    )  # fmt: skip
    w, w_x = _station("W", ("PW-W.d", "PW-W.u"), ("W-XW.d", "W-XW.u"), up="W1", down="W2", through="W3")
    e, e_x = _station("E", ("JH-E.d", "JH-E.u"), ("E-JE.d", "E-JE.u"), up="E1", down="E2", through="E3")
    te, te_x = _terminal("TE", "JE-TE.u", "JE-TE.d", ("TE1", "TE2", "TE3", "TE4"))
    ts, ts_x = _terminal("TS", "TS-H.d", "TS-H.u", ("TS1", "TS2", "TS3"))
    xw, xw_x = _crossovers("XW", "W-XW", "XW-JS")
    moves: list[MoveRow] = [
        *w, *e, *hub_w, *hub_e, *te, *ts, *xw,
        ("XW.m1", "XW", "W-XW.u", "XW-JS.u", 160, None), ("XW.m2", "XW", "XW-JS.d", "W-XW.d", 160, None),
        ("JS.m1", "JS", "XW-JS.u", "JS-H.u", 160, None), ("JS.m2", "JS", "JS-H.d", "XW-JS.d", 160, None),
        ("JS.m3", "JS", "XW-JS.u", "JS-YD", 40, None), ("JS.m4", "JS", "JS-YD", "XW-JS.d", 40, None),  # single lead
        ("JS.x1", "JS", "XW-JS.d", "JS-H.u", 40, None), ("JS.x2", "JS", "JS-H.d", "XW-JS.u", 40, None),
        ("JH.m1", "JH", "H-JH.u", "JH-E.u", 140, None), ("JH.m2", "JH", "JH-E.d", "H-JH.d", 140, None),
        ("JH.m3", "JH", "H-JH.nu", "JH-RN", 80, None), ("JH.m4", "JH", "JH-RN", "H-JH.nd", 80, None),
        ("JE.m1", "JE", "E-JE.u", "JE-PE.u", 160, None), ("JE.m2", "JE", "JE-PE.d", "E-JE.d", 160, None),
        ("JE.m3", "JE", "E-JE.u", "JE-TE.u", 80, None), ("JE.m4", "JE", "JE-TE.d", "E-JE.d", 80, None),
        *_loop_station("RN", "JH-RN", "RN-PN", ("RN1", "RN2")),
        *_loop_station("YD", "JS-YD", "YD-PORT", ("Y1", "Y2")),
        *[(f"PORT.{i}", "PORT", "YD-PORT", f"P{i}", 25, "a") for i in (1, 2, 3)],
    ]  # fmt: skip
    crossings = [
        xw_x, ("JS.x1", "JS.x2"),  # scissors at both ends of the reversible stretch
        ("JS.m2", "JS.m3"),  # port trains off the up main cross the down main
        ("JH.m2", "JH.m3"),  # flat: trains onto the branch cross the down main
        ("JE.m2", "JE.m3"),  # up trains for the terminal cross the down main from PE
        *hub_wx, *hub_ex, *w_x, *e_x, *te_x, *ts_x,
    ]  # fmt: skip
    relief = frozenset({f"H{i}" for i in range(1, 11)} | {"TE1", "TE2", "TE3", "TE4", "TS1", "TS2", "TS3"})
    return _network(nodes, tracks, platforms, moves, crossings, relief=relief, diesel=frozenset({"YD", "PORT"}))


def t2_services() -> tuple[list[ServiceGroup], list[ServiceGroup]]:
    """The T2 service (§6), 4 hours: clockface lines through and into the hub, freight filling the gaps.

    The intercity IC turns at TE and the regional express RE, off the branch, turns at TS: both are rotations.
    IX and IY cross the area through the hub, and so do the commuters SX and SY; the other commuters turn in the
    hub's bays (SW from the west, SE and SF from the east, SN off the branch) or run through it to turn at TS
    (SR, off the branch), so the throats carry trains every way; most connections are made at H. Freight runs
    between the portals on the hub's freight road, and out to the port, where it turns.
    """
    core = [
        ServiceGroup(code="IC", type="IC", entry_node="PW", exit_node="PW", first_min=5, every_min=30,
                     stops=("H1", "TE1", "H3"), dwell=(2, 12, 2)),
        ServiceGroup(code="IX", type="IC", entry_node="PW", exit_node="PE", first_min=20, every_min=60,
                     stops=("H1",), dwell=(3,)),
        ServiceGroup(code="IY", type="IC", entry_node="PE", exit_node="PW", first_min=0, every_min=60,
                     stops=("H3",), dwell=(3,)),
        ServiceGroup(code="RE", type="RE", entry_node="PN", exit_node="PN", first_min=10, every_min=60,
                     stops=("RN1", "H4", "TS1", "H2", "RN2"), dwell=(1, 2, 10, 2, 1)),
        ServiceGroup(code="SW", type="S", entry_node="PW", exit_node="PW", first_min=0, every_min=15,
                     stops=("W1", "H6", "W2"), dwell=(1, 8, 1)),
        ServiceGroup(code="SE", type="S", entry_node="PE", exit_node="PE", first_min=7, every_min=30,
                     stops=("E2", "H7", "E1"), dwell=(1, 8, 1)),
        ServiceGroup(code="SF", type="S", entry_node="PE", exit_node="PE", first_min=22, every_min=30,
                     stops=("E2", "H10", "E1"), dwell=(1, 8, 1)),
        ServiceGroup(code="SN", type="S", entry_node="PN", exit_node="PN", first_min=25, every_min=30,
                     stops=("RN1", "H8", "RN2"), dwell=(1, 8, 1)),
        ServiceGroup(code="SX", type="S", entry_node="PW", exit_node="PE", first_min=12, every_min=20,
                     stops=("W1", "H1", "E1"), dwell=(1, 2, 1)),
        ServiceGroup(code="SY", type="S", entry_node="PE", exit_node="PW", first_min=18, every_min=20,
                     stops=("E2", "H3", "W2"), dwell=(1, 2, 1)),
        ServiceGroup(code="SR", type="S", entry_node="PN", exit_node="PN", first_min=40, every_min=60,
                     stops=("RN1", "H4", "TS3", "H2", "RN2"), dwell=(1, 2, 8, 2, 1)),
    ]  # fmt: skip
    fill = [
        ServiceGroup(code="FT", type="FI", entry_node="PW", exit_node="PE", first_min=8, every_min=30,
                     stops=("H9",), dwell=(2,)),
        ServiceGroup(code="FU", type="FI", entry_node="PE", exit_node="PW", first_min=22, every_min=30,
                     stops=("H9",), dwell=(2,)),
        ServiceGroup(code="FP", type="FB", entry_node="PW", exit_node="PW", first_min=15, every_min=60,
                     stops=("P1",), dwell=(12,)),
        ServiceGroup(code="FQ", type="FI", entry_node="PW", exit_node="PW", first_min=45, every_min=60,
                     stops=("P2",), dwell=(10,)),
        ServiceGroup(code="FX", type="FB", entry_node="PW", exit_node="PE", first_min=38, every_min=60,
                     stops=("H9",), dwell=(2,)),
        ServiceGroup(code="FY", type="FB", entry_node="PE", exit_node="PW", first_min=52, every_min=60,
                     stops=("H9",), dwell=(2,)),
    ]  # fmt: skip
    return core, fill


# ---------------------------------------------------------------------- T3: two hubs, trunk and bypass


def two_hubs_network() -> Network:
    """T3: two 8-platform hubs HA and HB joined by a fast trunk and a slower bypass (docs/network-v0.3.md §6).

    The trunk HA-JA-M-JB-HB runs at 160 km/h; M-JB is signalled both ways, with crossovers at M and a scissors
    at JB, for single_line(). The bypass JA-BYP-JP-JB (110 km/h, electrified) has its own pair of tracks into
    each hub beside the trunk's, and is linked to the trunk at JA, a flat junction (trains from the trunk onto
    the bypass cross the down trunk), and at JB, which is grade-separated, so a train can be diverted round
    anything on the trunk in either direction. JP is a single-lead junction for the diesel-only port spur
    JP-PORT.

    West of HA: the main PW-WA-JT, and the suburban line PS-WS-JT. At JT, grade-separated, the suburban line and
    the line from the terminal TA merge into a slow pair of tracks into HA, beside the fast pair from the main.
    East of HB: a main pair to JN-EB-JE-PE and a branch pair to JN, a flat junction where the single-track
    branch JN-NA-NB-PN leaves: trains onto it cross the down main at grade. NA and NB are its crossing stations.
    The suburban line JN-ES-PF flies over into the main pair at JN. The terminal TB is off JE, grade-separated.
    PW, PE and PN are the outer portals; PS and PF feed the suburban lines.

    Each hub is laid out line by line: a fast group (up, down, an 800 m freight road and a bay) on one pair of
    tracks and a slow group (up, down and bays) on the other, so the two rarely cross. Crews are relieved at
    every hub platform and at both terminals.
    """
    nodes = [
        ("PW", "portal"), ("WA", "station"), ("JT", "junction"), ("TA", "terminal"), ("HA", "station"),
        ("JA", "junction"), ("M", "station"), ("JB", "junction"), ("BYP", "station"), ("JP", "junction"),
        ("PORT", "freight_terminal"), ("HB", "station"), ("JN", "junction"), ("NA", "station"), ("NB", "station"),
        ("PN", "portal"), ("EB", "station"), ("JE", "junction"), ("TB", "terminal"), ("PE", "portal"),
        ("PS", "portal"), ("WS", "station"), ("ES", "station"), ("PF", "portal"),
    ]  # fmt: skip
    tracks = [
        *_double("PW", "WA", 3, 160), *_double("WA", "JT", 13, 160), *_double("TA", "JT", 5, 120),
        *_double("PS", "WS", 3, 120), *_double("WS", "JT", 10, 120),
        *_double("JN", "ES", 8, 120), *_double("ES", "PF", 3, 120),
        *_double("JT", "HA", 3, 120), *_double("JT", "HA", 3, 100, pair="s"),
        *_double("HA", "JA", 3, 120), *_double("HA", "JA", 3, 100, pair="b"),
        *_double("JA", "M", 14, 160), *_double("M", "JB", 16, 160, reversible=True),
        *_double("JB", "HB", 3, 120), *_double("JB", "HB", 3, 100, pair="b"),
        *_double("JA", "BYP", 12, 110), *_double("BYP", "JP", 10, 110), *_double("JP", "JB", 9, 110),
        _single("JP", "PORT", 5, 50, electrified=False),
        *_double("HB", "JN", 3, 120), *_double("HB", "JN", 3, 100, pair="n"), *_double("JN", "EB", 9, 140),
        *_double("EB", "JE", 11, 140), *_double("JE", "PE", 3, 160), *_double("JE", "TB", 4, 100),
        _single("JN", "NA", 9, 100), _single("NA", "NB", 9, 100), _single("NB", "PN", 3, 100),
    ]  # fmt: skip
    platforms = [
        ("WA1", "WA", 280, "platform"), ("WA2", "WA", 280, "platform"), ("WA3", "WA", 800, "through"),
        ("WS1", "WS", 220, "platform"), ("WS2", "WS", 220, "platform"),
        ("ES1", "ES", 220, "platform"), ("ES2", "ES", 220, "platform"),
        ("TA1", "TA", 280, "platform"), ("TA2", "TA", 260, "platform"), ("TA3", "TA", 220, "platform"),
        ("HA1", "HA", 320, "platform"), ("HA2", "HA", 300, "platform"), ("HA3", "HA", 320, "platform"),
        ("HA4", "HA", 300, "platform"), ("HA5", "HA", 220, "platform"), ("HA6", "HA", 220, "platform"),
        ("HA7", "HA", 220, "platform"), ("HA8", "HA", 800, "through"),
        ("M1", "M", 280, "platform"), ("M2", "M", 280, "platform"), ("M3", "M", 800, "through"),
        ("BYP1", "BYP", 280, "platform"), ("BYP2", "BYP", 280, "platform"), ("BYP3", "BYP", 800, "through"),
        ("P1", "PORT", 750, "siding"), ("P2", "PORT", 750, "siding"), ("P3", "PORT", 750, "siding"),
        ("HB1", "HB", 320, "platform"), ("HB2", "HB", 300, "platform"), ("HB3", "HB", 320, "platform"),
        ("HB4", "HB", 300, "platform"), ("HB5", "HB", 220, "platform"), ("HB6", "HB", 220, "platform"),
        ("HB7", "HB", 220, "platform"), ("HB8", "HB", 800, "through"),
        ("NA1", "NA", 220, "platform"), ("NA2", "NA", 750, "platform"),
        ("NB1", "NB", 220, "platform"), ("NB2", "NB", 750, "platform"),
        ("EB1", "EB", 280, "platform"), ("EB2", "EB", 280, "platform"), ("EB3", "EB", 800, "through"),
        ("TB1", "TB", 280, "platform"), ("TB2", "TB", 270, "platform"), ("TB3", "TB", 220, "platform"),
        ("TB4", "TB", 220, "platform"),
    ]  # fmt: skip
    # HA, north to south: the bypass pair's east bay HA7, the slow group (west bay HA5, HA4 down, HA2 up) off the
    # slow pair from JT, then the fast group (HA3 down, west bay HA6, HA8 freight road, HA1 up) off the fast pair.
    ha_order = ["HA5", "HA7", "HA4", "HA2", "HA3", "HA6", "HA8", "HA1"]
    ha_w, ha_wx = _throat(
        "HA", "a", ["JT-HA.sd", "JT-HA.su", "JT-HA.d", "JT-HA.u"], ha_order,
        [("HA5", "JT-HA.sd"), ("HA5", "JT-HA.su"), ("HA4", "JT-HA.sd"), ("HA2", "JT-HA.su"), ("HA3", "JT-HA.d"),
         ("HA6", "JT-HA.d"), ("HA6", "JT-HA.u"), ("HA8", "JT-HA.d"), ("HA8", "JT-HA.u"), ("HA1", "JT-HA.u")],
        frozenset({("HA4", "JT-HA.sd"), ("HA2", "JT-HA.su"), ("HA3", "JT-HA.d"), ("HA1", "JT-HA.u")}),
    )  # fmt: skip
    ha_e, ha_ex = _throat(
        "HA", "b", ["HA-JA.bd", "HA-JA.bu", "HA-JA.d", "HA-JA.u"], ha_order,
        [("HA7", "HA-JA.bd"), ("HA7", "HA-JA.bu"), ("HA4", "HA-JA.d"), ("HA2", "HA-JA.u"), ("HA3", "HA-JA.d"),
         ("HA8", "HA-JA.d"), ("HA8", "HA-JA.u"), ("HA1", "HA-JA.u")],
        frozenset({("HA3", "HA-JA.d"), ("HA1", "HA-JA.u")}),
    )  # fmt: skip
    # HB, north to south: the branch half (HB4 down, east bay HB7, HB2 up) fed by the bypass pair from the west
    # and the branch pair from the east, then the main half (HB3 down, west bay HB5, east bay HB6, HB8 freight
    # road, HB1 up) fed by the trunk pair and the main pair.
    hb_order = ["HB4", "HB7", "HB2", "HB3", "HB5", "HB6", "HB8", "HB1"]
    hb_w, hb_wx = _throat(
        "HB", "a", ["JB-HB.bd", "JB-HB.bu", "JB-HB.d", "JB-HB.u"], hb_order,
        [("HB4", "JB-HB.bd"), ("HB2", "JB-HB.bu"), ("HB5", "JB-HB.d"), ("HB5", "JB-HB.u"), ("HB3", "JB-HB.d"),
         ("HB8", "JB-HB.d"), ("HB8", "JB-HB.u"), ("HB1", "JB-HB.u")],
        frozenset({("HB4", "JB-HB.bd"), ("HB2", "JB-HB.bu"), ("HB3", "JB-HB.d"), ("HB1", "JB-HB.u")}),
    )  # fmt: skip
    hb_e, hb_ex = _throat(
        "HB", "b", ["HB-JN.nd", "HB-JN.nu", "HB-JN.d", "HB-JN.u"], hb_order,
        [("HB6", "HB-JN.d"), ("HB6", "HB-JN.u"), ("HB3", "HB-JN.d"), ("HB8", "HB-JN.d"), ("HB8", "HB-JN.u"),
         ("HB1", "HB-JN.u"), ("HB4", "HB-JN.nd"), ("HB2", "HB-JN.nu"), ("HB7", "HB-JN.nd"), ("HB7", "HB-JN.nu")],
        frozenset({("HB3", "HB-JN.d"), ("HB1", "HB-JN.u"), ("HB4", "HB-JN.nd"), ("HB2", "HB-JN.nu")}),
    )  # fmt: skip
    wa, wa_x = _station("WA", ("PW-WA.d", "PW-WA.u"), ("WA-JT.d", "WA-JT.u"), up="WA1", down="WA2", through="WA3")
    ws, ws_x = _station("WS", ("PS-WS.d", "PS-WS.u"), ("WS-JT.d", "WS-JT.u"), up="WS1", down="WS2")
    es, es_x = _station("ES", ("JN-ES.d", "JN-ES.u"), ("ES-PF.d", "ES-PF.u"), up="ES1", down="ES2")
    m, m_x = _throat("M", "a", ["JA-M.d", "JA-M.u"], ["M2", "M3", "M1"],
                     [("M2", "JA-M.d"), ("M3", "JA-M.d"), ("M3", "JA-M.u"), ("M1", "JA-M.u")],
                     frozenset({("M2", "JA-M.d"), ("M1", "JA-M.u")}))  # fmt: skip
    m_e, m_ex = _throat("M", "b", ["M-JB.d", "M-JB.u"], ["M2", "M3", "M1"],
                        [("M2", "M-JB.d"), ("M3", "M-JB.d"), ("M3", "M-JB.u"), ("M1", "M-JB.u"),
                         ("M1", "M-JB.d"), ("M2", "M-JB.u")],  # the last two are the wrong-road crossovers
                        frozenset({("M2", "M-JB.d"), ("M1", "M-JB.u")}))  # fmt: skip
    byp, byp_x = _station("BYP", ("JA-BYP.d", "JA-BYP.u"), ("BYP-JP.d", "BYP-JP.u"), up="BYP1", down="BYP2",
                          through="BYP3")  # fmt: skip
    eb, eb_x = _station("EB", ("JN-EB.d", "JN-EB.u"), ("EB-JE.d", "EB-JE.u"), up="EB1", down="EB2", through="EB3")
    ta, ta_x = _terminal("TA", "TA-JT.d", "TA-JT.u", ("TA1", "TA2", "TA3"))
    tb, tb_x = _terminal("TB", "JE-TB.u", "JE-TB.d", ("TB1", "TB2", "TB3", "TB4"))
    jb_x, jb_xx = _crossovers("JB", "M-JB", "JB-HB")
    moves: list[MoveRow] = [
        *wa, *ws, *es, *ha_w, *ha_e, *m, *m_e, *byp, *hb_w, *hb_e, *eb, *ta, *tb, *jb_x,
        ("JT.m1", "JT", "WA-JT.u", "JT-HA.u", 160, None), ("JT.m2", "JT", "JT-HA.d", "WA-JT.d", 160, None),
        ("JT.m3", "JT", "TA-JT.u", "JT-HA.su", 120, None), ("JT.m4", "JT", "JT-HA.sd", "TA-JT.d", 120, None),
        ("JT.m5", "JT", "WS-JT.u", "JT-HA.su", 120, None), ("JT.m6", "JT", "JT-HA.sd", "WS-JT.d", 120, None),
        ("JA.m1", "JA", "HA-JA.u", "JA-M.u", 160, None), ("JA.m2", "JA", "JA-M.d", "HA-JA.d", 160, None),
        ("JA.m3", "JA", "HA-JA.u", "JA-BYP.u", 80, None), ("JA.m4", "JA", "JA-BYP.d", "HA-JA.d", 80, None),
        ("JA.m5", "JA", "HA-JA.bu", "JA-BYP.u", 100, None), ("JA.m6", "JA", "JA-BYP.d", "HA-JA.bd", 100, None),
        ("JB.m1", "JB", "M-JB.u", "JB-HB.u", 160, None), ("JB.m2", "JB", "JB-HB.d", "M-JB.d", 160, None),
        ("JB.m3", "JB", "JP-JB.u", "JB-HB.u", 80, None), ("JB.m4", "JB", "JB-HB.d", "JP-JB.d", 80, None),
        ("JB.m5", "JB", "JP-JB.u", "JB-HB.bu", 100, None), ("JB.m6", "JB", "JB-HB.bd", "JP-JB.d", 100, None),
        ("JP.m1", "JP", "BYP-JP.u", "JP-JB.u", 110, None), ("JP.m2", "JP", "JP-JB.d", "BYP-JP.d", 110, None),
        ("JP.m3", "JP", "BYP-JP.u", "JP-PORT", 40, None), ("JP.m4", "JP", "JP-PORT", "BYP-JP.d", 40, None),
        ("JN.m1", "JN", "HB-JN.u", "JN-EB.u", 140, None), ("JN.m2", "JN", "JN-EB.d", "HB-JN.d", 140, None),
        ("JN.m3", "JN", "HB-JN.nu", "JN-NA", 80, None), ("JN.m4", "JN", "JN-NA", "HB-JN.nd", 80, None),
        ("JN.m5", "JN", "HB-JN.u", "JN-ES.u", 120, None), ("JN.m6", "JN", "JN-ES.d", "HB-JN.d", 120, None),
        ("JE.m1", "JE", "EB-JE.u", "JE-PE.u", 160, None), ("JE.m2", "JE", "JE-PE.d", "EB-JE.d", 160, None),
        ("JE.m3", "JE", "EB-JE.u", "JE-TB.u", 80, None), ("JE.m4", "JE", "JE-TB.d", "EB-JE.d", 80, None),
        *[(f"PORT.{i}", "PORT", "JP-PORT", f"P{i}", 25, "a") for i in (1, 2, 3)],
        *_loop_station("NA", "JN-NA", "NA-NB", ("NA1", "NA2")),
        *_loop_station("NB", "NA-NB", "NB-PN", ("NB1", "NB2")),
    ]  # fmt: skip
    crossings = [
        ("JA.m2", "JA.m3"),  # flat: trains from the trunk onto the bypass cross the down trunk; JT, JB, JE fly
        jb_xx,  # scissors at the east end of the reversible stretch
        ("JP.m1", "JP.m4"),  # trains off the port spur cross the up bypass
        ("JN.m2", "JN.m3"),  # flat: trains onto the branch cross the down main
        *ha_wx, *ha_ex, *hb_wx, *hb_ex, *wa_x, *ws_x, *es_x, *m_x, *m_ex, *byp_x, *eb_x, *ta_x, *tb_x,
    ]  # fmt: skip
    relief = frozenset(
        {f"HA{i}" for i in range(1, 9)} | {f"HB{i}" for i in range(1, 9)} | {"TA1", "TA2", "TA3"}
        | {"TB1", "TB2", "TB3", "TB4"}
    )  # fmt: skip
    return _network(nodes, tracks, platforms, moves, crossings, relief=relief, diesel=frozenset({"PORT"}))


def t3_services() -> tuple[list[ServiceGroup], list[ServiceGroup]]:
    """The T3 service (§6), 6 hours.

    Intercities cross the area both ways over the trunk (IC east, ID west) and IT turns at TB; the regional
    express RE runs from PE to TA and back on the hubs' slow groups, RX and RY right across. Commuters turn in
    the hub bays: SA off the west suburban line and SC off the main at HA, SB off the east suburban line and SN
    off the branch at HB, SM from the west suburban line over the trunk, and SY (from PE) and SQ (off the branch)
    over the bypass, so the bypass is in daily use before anything is diverted onto it. Freight runs hub to hub
    over the trunk and over the bypass, and out to the port, where it turns.
    """
    core = [
        ServiceGroup(code="IC", type="IC", entry_node="PW", exit_node="PE", first_min=4, every_min=30,
                     stops=("HA1", "HB1"), dwell=(3, 3)),
        ServiceGroup(code="ID", type="IC", entry_node="PE", exit_node="PW", first_min=11, every_min=30,
                     stops=("HB3", "HA3"), dwell=(3, 3)),
        ServiceGroup(code="IT", type="IC", entry_node="PW", exit_node="PW", first_min=20, every_min=60,
                     stops=("HA1", "HB1", "TB1", "HB3", "HA3"), dwell=(2, 2, 12, 2, 2)),
        ServiceGroup(code="RE", type="RE", entry_node="PE", exit_node="PE", first_min=0, every_min=60,
                     stops=("EB2", "HB3", "HA4", "TA1", "HA2", "HB1", "EB1"), dwell=(1, 2, 2, 10, 2, 2, 1)),
        ServiceGroup(code="SA", type="S", entry_node="PS", exit_node="PS", first_min=0, every_min=15,
                     stops=("WS1", "HA5", "WS2"), dwell=(1, 8, 1)),
        ServiceGroup(code="SB", type="S", entry_node="PF", exit_node="PF", first_min=6, every_min=15,
                     stops=("ES2", "HB6", "ES1"), dwell=(1, 8, 1)),
        ServiceGroup(code="SN", type="S", entry_node="PN", exit_node="PN", first_min=14, every_min=30,
                     stops=("NB1", "NA1", "HB7", "NA2", "NB2"), dwell=(1, 1, 8, 1, 1)),
        ServiceGroup(code="SM", type="S", entry_node="PS", exit_node="PS", first_min=9, every_min=20,
                     stops=("WS1", "HA2", "M1", "HB5", "M2", "HA4", "WS2"), dwell=(1, 2, 1, 8, 1, 2, 1)),
        ServiceGroup(code="SC", type="S", entry_node="PW", exit_node="PW", first_min=12, every_min=20,
                     stops=("WA1", "HA6", "WA2"), dwell=(1, 8, 1)),
        ServiceGroup(code="RX", type="RE", entry_node="PW", exit_node="PE", first_min=25, every_min=60,
                     stops=("WA1", "HA1", "HB1", "EB1"), dwell=(1, 2, 2, 1)),
        ServiceGroup(code="RY", type="RE", entry_node="PE", exit_node="PW", first_min=40, every_min=60,
                     stops=("EB2", "HB3", "HA3", "WA2"), dwell=(1, 2, 2, 1)),
        ServiceGroup(code="SY", type="S", entry_node="PE", exit_node="PE", first_min=30, every_min=30,
                     stops=("EB2", "HB3", "BYP2", "HA7", "BYP1", "HB1", "EB1"), dwell=(1, 2, 1, 8, 1, 2, 1)),
        ServiceGroup(code="SQ", type="S", entry_node="PN", exit_node="PN", first_min=20, every_min=60,
                     stops=("NB1", "NA1", "HB4", "BYP2", "HA7", "BYP1", "HB2", "NA2", "NB2"),
                     dwell=(1, 1, 2, 1, 8, 1, 2, 1, 1)),
    ]  # fmt: skip
    fill = [
        ServiceGroup(code="FW", type="FI", entry_node="PW", exit_node="PE", first_min=7, every_min=30,
                     stops=("HA8", "HB8"), dwell=(2, 2)),
        ServiceGroup(code="FE", type="FI", entry_node="PE", exit_node="PW", first_min=19, every_min=30,
                     stops=("HB8", "HA8"), dwell=(2, 2)),
        ServiceGroup(code="FU", type="FI", entry_node="PW", exit_node="PE", first_min=37, every_min=60,
                     stops=("HA8", "BYP3", "HB8"), dwell=(2, 2, 2)),
        ServiceGroup(code="FV", type="FI", entry_node="PE", exit_node="PW", first_min=49, every_min=60,
                     stops=("HB8", "BYP3", "HA8"), dwell=(2, 2, 2)),
        ServiceGroup(code="FP", type="FB", entry_node="PW", exit_node="PW", first_min=15, every_min=60,
                     stops=("P1",), dwell=(12,)),
        ServiceGroup(code="FQ", type="FI", entry_node="PW", exit_node="PW", first_min=45, every_min=60,
                     stops=("P2",), dwell=(10,)),
        ServiceGroup(code="FX", type="FB", entry_node="PW", exit_node="PE", first_min=27, every_min=60,
                     stops=("HA8", "HB8"), dwell=(2, 2)),
        ServiceGroup(code="FY", type="FB", entry_node="PE", exit_node="PW", first_min=55, every_min=60,
                     stops=("HB8", "HA8"), dwell=(2, 2)),
    ]  # fmt: skip
    return core, fill


# ---------------------------------------------------------------------- the tiers


@dataclass(frozen=True)
class Tier:
    """One generator preset (§6): a layout, its service, and the default shift length in minutes."""

    name: str
    network: Callable[[], Network]
    services: Callable[[], tuple[list[ServiceGroup], list[ServiceGroup]]]
    shift_min: int

    def timetable(self, seed: int, shift_min: int | None = None) -> Timetable:
        from .timetable import generate  # the generator runs the engine; templates stay importable without it

        core, fill = self.services()
        return generate(self.network(), core, fill, shift_min or self.shift_min, seed, self.name)


TIERS: dict[str, Tier] = {
    "T1": Tier("T1", trunk_and_branches, t1_services, 180),
    "T2": Tier("T2", hub_network, t2_services, 240),
    "T3": Tier("T3", two_hubs_network, t3_services, 360),
}
