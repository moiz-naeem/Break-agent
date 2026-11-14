import pytest
from pydantic import ValidationError

from railway.net.model import Network, TrainSpec
from railway.net.templates import trunk_and_branches

NET = trunk_and_branches()


def mutated(change):
    data = {k: list(v) for k, v in NET.model_dump().items()}
    change(data)
    return data


def by_id(items, item_id):
    return next(x for x in items if x["id"] == item_id)


def test_t1_template_is_valid():
    assert (len(NET.nodes), len(NET.tracks), len(NET.platforms), len(NET.moves)) == (9, 13, 13, 40)
    assert NET.element("JA-B.u").blocks == 5  # 15 km in 3 km blocks on a 160 km/h line
    assert NET.element("A-JA.u").blocks == 2  # 4 km in 3 km blocks
    assert not NET.element("JB-PORT").electrified
    assert NET.element("nope") is None and NET.node("PW").kind == "portal"


def test_conflicts_are_derived_from_shared_points_and_crossings():
    # shared points with m1, m4 and the crossover x2; crossing with m2
    assert NET.conflicts("JA.m3") == ["JA.m1", "JA.m2", "JA.m4", "JA.x2"]
    assert NET.conflicts("C.a1") == ["C.a2", "C.a3", "C.d1"]  # same throat, and the same end of C1 as C.d1
    assert "A.e1" not in NET.conflicts("A.w1")  # opposite ends of platform A1


def test_network_and_trains_are_frozen_and_validated():
    with pytest.raises(ValidationError):
        NET.nodes[0].kind = "station"
    assert isinstance(NET.moves, tuple)
    with pytest.raises(ValidationError):
        TrainSpec(id="X", length_m=-5, electric=True, passenger=True)


def _drop_moves(d, ids):
    d["moves"] = [m for m in d["moves"] if m["id"] not in ids]


def _bridge_lines_through_portal(d):
    d["nodes"].append({"id": "Q", "kind": "station"})
    d["tracks"].append(
        {"id": "PW-Q", "a": "PW", "b": "Q", "length_m": 1000, "line_speed_kmh": 60, "electrified": True,
         "direction": "both"}
    )  # fmt: skip
    d["platforms"].append({"id": "Q1", "node": "Q", "length_m": 200, "kind": "platform"})
    d["moves"].append({"id": "Q.1", "node": "Q", "ends": ("PW-Q", "Q1"), "speed_kmh": 40, "side": "a"})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: by_id(d["moves"], "A.w1").update(id="A1"), "duplicate ids"),
        (lambda d: by_id(d["tracks"], "PW-A.u").update(a="ZZ"), "two different known nodes"),
        (lambda d: by_id(d["platforms"], "A1").update(node="JA"), "must be in a station"),
        (lambda d: by_id(d["moves"], "A.w1").update(ends=("PW-A.u", "B1")), "B1 is not a track or platform at A"),
        (lambda d: by_id(d["moves"], "A.w1").update(side=None), "needs a side"),
        (lambda d: by_id(d["moves"], "JA.m1").update(side="a"), "needs a side"),
        (lambda d: d["crossings"].append(("A.w1", "B.w1")), "two different moves at one node"),
        (lambda d: _drop_moves(d, {"N.e1", "N.e2"}), "N-PN dead-ends at N"),
        (lambda d: d["platforms"].append({"id": "A9", "node": "A", "length_m": 100, "kind": "platform"}), "A9 has no"),
        (lambda d: by_id(d["nodes"], "JA").update(kind="portal"), "portal JA must have no moves"),
        (_bridge_lines_through_portal, "portal PW must have no moves and lead to exactly one node"),
        (lambda d: d["nodes"].append({"id": "X", "kind": "station"}), r"\['X'\] unreachable"),
        (lambda d: d.update(nodes=[]), "at least 1"),
    ],
)
def test_invalid_networks_are_rejected(change, message):
    with pytest.raises(ValidationError, match=message):
        Network.model_validate(mutated(change))
