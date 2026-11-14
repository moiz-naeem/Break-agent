"""Layout templates (docs/network-v0.3.md §6). The generator will perturb these; here they are fixed."""

from __future__ import annotations

from .model import Call, Move, Network, Node, Platform, Track, TrainSpec, Trip

INTERCITY = {"length_m": 260, "electric": True, "passenger": True, "max_speed_kmh": 200, "restart_min": 1}
COMMUTER = {"length_m": 160, "electric": True, "passenger": True, "max_speed_kmh": 120, "restart_min": 1}
FREIGHT = {"length_m": 740, "electric": False, "passenger": False, "max_speed_kmh": 80, "restart_min": 5}


def block_m(line_speed_kmh: int) -> int:
    """Signal blocks are spaced by braking distance, so a fast line is divided into longer blocks."""
    return 3000 if line_speed_kmh >= 140 else 2000 if line_speed_kmh >= 100 else 1500


def trunk_and_branches() -> Network:
    """T1: double-track trunk PW-A-JA-B-JB-C, terminal C, single-track branch JA-N-PN, diesel-only port spur at JB.

    JA is a flat junction: up trains turning onto the branch cross the down main. JB is a single lead: the spur's
    two directions share one turnout. B has a 750 m loop where slow trains step aside.
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
              block_m=block_m(v))
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
        Platform(id=i, node=n, length_m=m, kind=k, electrified=n != "PORT")
        for i, n, m, k in [
            ("A1", "A", 270, "platform"), ("A2", "A", 270, "platform"),
            ("B1", "B", 270, "platform"), ("B2", "B", 270, "platform"), ("B3", "B", 750, "siding"),
            ("C1", "C", 260, "platform"), ("C2", "C", 260, "platform"), ("C3", "C", 180, "platform"),
            ("N1", "N", 200, "platform"), ("N2", "N", 750, "siding"),
            ("P1", "PORT", 750, "siding"), ("P2", "PORT", 750, "siding"),
        ]
    ]  # fmt: skip
    moves = [  # id, node, ends, turnout speed (km/h), platform side ("a" west/approach, "b" east)
        ("A.w1", "A", "PW-A.u", "A1", 60, "a"), ("A.w2", "A", "A2", "PW-A.d", 60, "a"),
        ("A.w3", "A", "PW-A.u", "A2", 40, "a"),
        ("A.e1", "A", "A1", "A-JA.u", 60, "b"), ("A.e2", "A", "A-JA.d", "A2", 60, "b"),
        ("A.e3", "A", "A2", "A-JA.u", 40, "b"),
        ("JA.m1", "JA", "A-JA.u", "JA-B.u", 160, None), ("JA.m2", "JA", "JA-B.d", "A-JA.d", 160, None),
        ("JA.m3", "JA", "A-JA.u", "JA-N", 80, None), ("JA.m4", "JA", "JA-N", "A-JA.d", 80, None),
        ("B.w1", "B", "JA-B.u", "B1", 60, "a"), ("B.w2", "B", "JA-B.u", "B3", 40, "a"),
        ("B.w3", "B", "B2", "JA-B.d", 60, "a"),
        ("B.e1", "B", "B1", "B-JB.u", 60, "b"), ("B.e2", "B", "B3", "B-JB.u", 40, "b"),
        ("B.e3", "B", "B-JB.d", "B2", 60, "b"),
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


def sample_trips() -> list[Trip]:
    """A small T1 service for tests: an eastbound intercity, a branch commuter and a port freight.

    IC1 and L4 leave A at the same minute and both need the up line A-JA.u, where IC1 carries straight on and L4
    turns across the flat junction JA. ARS cannot choose between them, so the agent must.
    """
    return [
        Trip(
            train=TrainSpec(id="IC1", **INTERCITY),
            entry_node="PW",
            entry_tick=0,
            calls=(Call(platform="A1", arrive=8, depart=10), Call(platform="C1", arrive=40)),
        ),
        Trip(
            train=TrainSpec(id="L4", **COMMUTER),
            calls=(Call(platform="A2", depart=10), Call(platform="N1", arrive=30)),
        ),
        Trip(
            train=TrainSpec(id="F3", **FREIGHT),
            calls=(Call(platform="B3", depart=12), Call(platform="P1", arrive=60)),
        ),
    ]
