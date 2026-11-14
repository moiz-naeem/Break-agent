import pytest

from railway.net.model import TrainSpec
from railway.net.routing import find_route
from railway.net.templates import COMMUTER, FREIGHT, INTERCITY, trunk_and_branches

NET = trunk_and_branches()
IC = TrainSpec(id="IC1", **INTERCITY)
L = TrainSpec(id="L2", **COMMUTER)
F = TrainSpec(id="F3", **FREIGHT)


def test_route_across_the_network():
    assert find_route(NET, "A1", "A", "C1", IC) == [
        "A1", "A.e1", "A-JA.u", "JA.m1", "JA-B.u", "B.w1", "B1", "B.e1", "B-JB.u", "JB.m1", "JB-C.u", "C.a1", "C1",
    ]  # fmt: skip
    assert find_route(NET, "N1", "N", "A2", L) == ["N1", "N.w1", "JA-N", "JA.m4", "A-JA.d", "A.e2", "A2"]
    assert find_route(NET, "PW-A.u", "A", "A1", IC) == ["PW-A.u", "A.w1", "A1"]  # entering from the portal


@pytest.mark.parametrize(
    ("train", "start", "facing", "target", "why"),
    [
        (IC, "A1", "A", "N1", "N1 is 200 m and the intercity is 260 m"),
        (L, "B1", "B", "B3", "B3 is a siding, so a passenger train may not terminate there"),
        (IC, "B1", "B", "P1", "the port spur has no overhead line for an electric train"),
        (IC, "A1", "A", "nope", "no such platform"),
        (IC, "A1", "A", "A-JA.u", "a track is not a destination"),
        (L, "A1", "A", "B2", "B2 faces the down line, which cannot be reached going east"),
    ],
)
def test_impossible_routes(train, start, facing, target, why):
    assert find_route(NET, start, facing, target, train) is None, why


def test_freight_reaches_the_port_but_the_electric_one_does_not():
    diesel = find_route(NET, "B3", "B", "P1", F)
    assert diesel == ["B3", "B.e2", "B-JB.u", "JB.m3", "JB-PORT", "P.1", "P1"]
    electric = TrainSpec(id="F9", **{**FREIGHT, "electric": True})
    assert find_route(NET, "B3", "B", "P1", electric) is None


def test_route_never_reverses_in_a_platform():
    route = find_route(NET, "A2", "A", "N1", L)
    assert route[:3] == ["A2", "A.e3", "A-JA.u"]  # leaves by the east side, not back the way it came
    assert len(set(route)) == len(route)
