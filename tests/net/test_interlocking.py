import pytest
from hypothesis import given
from hypothesis import strategies as st

from railway.net.interlocking import Interlocking, Refusal
from railway.net.model import Move, Track
from railway.net.rules import NetRule
from railway.net.templates import trunk_and_branches
from railway.net.trains import TYPES

NET = trunk_and_branches()
IC = TYPES["IC"].spec("IC1")
S = TYPES["S"].spec("S1")
S2 = TYPES["S"].spec("S2")
FD = TYPES["FB"].spec("F1")
FE = TYPES["FI"].spec("F2").model_copy(update={"electric": True})


def rule(result):
    assert isinstance(result, Refusal), result
    return result.rule


def test_entry_into_platform_and_conflicting_crossover():
    il = Interlocking(NET)
    assert il.lock(IC, "PW", ["PW-A.u", "A.w1", "A1"]) is None
    assert il.holder["A1"] == "IC1" and il.holder[("PW-A.u", 0)] == "IC1"
    assert ("PW-A.u", 3) not in il.holder  # entering takes one block, as any other train does
    assert rule(il.lock(S, "PW", ["PW-A.u", "A.w3", "A2"])) == NetRule.MOVE_CONFLICT
    il.release("IC1", "A.w1")
    assert rule(il.lock(S, "PW", ["PW-A.u", "A.w3", "A2"])) == NetRule.BLOCK_OCCUPIED  # IC1 still in the entry block
    il.release("IC1", "PW-A.u")  # IC1 has run on up the line
    assert il.lock(S, "PW", ["PW-A.u", "A.w3", "A2"]) is None
    for passed in ("PW-A.u", "A.w3"):
        il.release("S1", passed)
    assert rule(il.lock(S2, "PW", ["PW-A.u", "A.w1", "A1"])) == NetRule.PLATFORM_OCCUPIED


def test_portal_entry_cannot_meet_a_train_head_on():
    il = Interlocking(NET)
    assert il.lock(S, "N", ["N1", "N.e1", "N-PN"]) is None  # S1 heads out to PN on single track
    for passed in ("N.e1", "N1"):
        il.release("S1", passed)
    assert rule(il.lock(S2, "PN", ["N-PN", "N.e1", "N1"])) == NetRule.DIRECTION_LOCKED


def test_flat_junction_turn_blocks_the_opposite_main():
    il = Interlocking(NET)
    assert il.lock(IC, "A", ["A1", "A.e1", "A-JA.u", "JA.m3", "JA-N"]) is None
    assert rule(il.lock(S, "B", ["JA-B.d", "JA.m2", "A-JA.d"])) == NetRule.MOVE_CONFLICT
    il.release("IC1", "JA.m3")
    assert il.lock(S, "B", ["JA-B.d", "JA.m2", "A-JA.d"]) is None


def test_single_track_direction_lock_and_refusal_changes_nothing():
    il = Interlocking(NET)
    assert il.lock(IC, "A", ["A1", "A.e1", "A-JA.u", "JA.m3", "JA-N"]) is None
    before = snapshot(il)
    assert rule(il.lock(FE, "N", ["N2", "N.w2", "JA-N", "JA.m4", "A-JA.d"])) == NetRule.DIRECTION_LOCKED
    assert snapshot(il) == before  # N.w2 was free but is not taken
    assert rule(il.lock(IC, "N", ["N2", "N.w2", "JA-N", "JA.m4", "A-JA.d"])) == NetRule.DIRECTION_LOCKED  # own, too


def test_following_train_waits_for_the_entry_block():
    il = Interlocking(NET)
    assert il.lock(IC, "A", ["A1", "A.e1", "A-JA.u", "JA.m1", "JA-B.u"]) is None
    assert il.holder[("JA-B.u", 0)] == "IC1" and ("JA-B.u", 1) not in il.holder
    for passed in ("A.e1", "A-JA.u", "JA.m1"):
        il.release("IC1", passed)
    assert rule(il.lock(S, "A", ["A2", "A.e3", "A-JA.u", "JA.m1", "JA-B.u"])) == NetRule.BLOCK_OCCUPIED
    del il.holder[("JA-B.u", 0)]  # IC1 runs on into block 1 (block movement arrives in step 2)
    il.holder[("JA-B.u", 1)] = "IC1"
    assert il.lock(S, "A", ["A2", "A.e3", "A-JA.u", "JA.m1", "JA-B.u"]) is None
    assert il.directions["JA-B.u"] == {"IC1": "ab", "S1": "ab"}


def test_position_held_by_another_train_is_refused():
    il = Interlocking(NET)
    assert il.lock(IC, "PW", ["PW-A.u", "A.w1", "A1"]) is None
    assert rule(il.lock(S, "A", ["A1", "A.e1", "A-JA.u"])) == NetRule.PLATFORM_OCCUPIED


def test_electrification_platform_length_and_kind():
    to_port = ["B3", "B.e2", "B-JB.u", "JB.m3", "JB-PORT", "P.1", "P1"]
    assert rule(Interlocking(NET).lock(FE, "B", to_port)) == NetRule.NOT_ELECTRIFIED
    assert Interlocking(NET).lock(FD, "B", to_port) is None
    assert rule(Interlocking(NET).lock(FD, "PW", ["PW-A.u", "A.w1", "A1"])) == NetRule.PLATFORM_LEN
    assert Interlocking(NET).lock(FD, "JA", ["JA-B.u", "B.w2", "B3"]) is None
    assert rule(Interlocking(NET).lock(S, "JA", ["JA-B.u", "B.w2", "B3"])) == NetRule.PLATFORM_KIND
    through_loop = ["JA-B.u", "B.w2", "B3", "B.e2", "B-JB.u"]  # running through a siding is fine
    assert Interlocking(NET).lock(S, "JA", through_loop) is None


def test_a_route_that_can_never_work_is_not_told_to_wait():
    il = Interlocking(NET)
    assert il.lock(IC, "B", ["B1", "B.e1", "B-JB.u"]) is None  # B.e1 conflicts with B.e2
    assert rule(il.lock(FE, "B", ["B3", "B.e2", "B-JB.u", "JB.m3", "JB-PORT", "P.1", "P1"])) == NetRule.NOT_ELECTRIFIED


@pytest.mark.parametrize(
    ("train", "start", "elements", "why"),
    [
        (FD, "JA", ["A-JA.d", "A.e2", "A2", "A.e3", "A-JA.u"], "reverse in a platform"),  # 740 m in a 270 m platform
        (IC, "JB", ["JB-C.u", "C.a3", "C3", "C.d3", "JB-C.d"], "reverse in a platform"),  # "through" a dead-end bay
        (S, "JA", ["JA-N", "N.w2", "N2", "N.w2", "JA-N"], "twice"),  # reverse in the branch loop
        (IC, "A", ["A1", "A.e1", "A-JA.u", "JA.m3", "JA-N", "N.w1", "N1", "N.w1", "JA-N"], "visits JA-N twice"),
    ],
)
def test_routes_may_not_reverse_or_repeat(train, start, elements, why):
    refusal = Interlocking(NET).lock(train, start, elements)
    assert rule(refusal) == NetRule.BAD_ROUTE and why in refusal.detail


@pytest.mark.parametrize(
    ("start", "elements", "expected"),
    [
        ("A", ["A1"], NetRule.BAD_ROUTE),
        ("A", ["A1", "zzz"], NetRule.UNKNOWN_ELEMENT),
        ("JA", ["A-JA.u", "A.e1", "A1"], NetRule.WRONG_DIRECTION),
        ("A", ["PW-A.u", "A.w1", "A1"], NetRule.WRONG_DIRECTION),  # start_node is where the train came from
        ("A", ["A1", "A-JA.u"], NetRule.NOT_CONNECTED),
        ("A", ["A1", "A.e2", "A-JA.d"], NetRule.NOT_CONNECTED),
        ("B", ["A1", "A.e1", "A-JA.u"], NetRule.NOT_CONNECTED),
        ("A", ["A.e1", "A-JA.u"], NetRule.NOT_CONNECTED),
        ("PW", ["PW-A.u", "A.w1"], NetRule.NOT_CONNECTED),
    ],
)
def test_malformed_routes(start, elements, expected):
    assert rule(Interlocking(NET).lock(IC, start, elements)) == expected


def test_relock_own_route_and_release_train():
    il = Interlocking(NET)
    route = ["A1", "A.e1", "A-JA.u", "JA.m3", "JA-N"]
    assert il.lock(IC, "A", route) is None
    assert il.lock(IC, "A", route) is None
    il.release("IC1", "JA-N")
    il.holder[("JA-N", 1)] = "IC1"  # a block with no direction entry must not leak
    il.release_train("IC1")
    assert (il.holder, il.directions) == ({}, {})


# Property: whatever trains ask for, the interlocking never grants two trains the same resource, never holds two
# conflicting moves, never sets one track both ways, keeps blocks and directions in step, never takes a resource
# from its holder, and a refusal changes nothing.

STARTS = [(p.node, p.id) for p in NET.platforms] + [("PW", "PW-A.u"), ("PN", "N-PN")]
TRAINS = [IC, S, FD, FE]


@st.composite
def routes(draw):
    start, first = draw(st.sampled_from(STARTS))
    el = NET.element(first)
    node = el.other(start) if isinstance(el, Track) else start
    elements = [first]
    for _ in range(draw(st.integers(1, 4))):
        options = [m for m in NET.moves if m.node == node and elements[-1] in m.ends]
        if not options:
            break
        move = draw(st.sampled_from(options))
        nxt = NET.element(move.other(elements[-1]))
        elements += [move.id, nxt.id]
        if isinstance(nxt, Track):
            node = nxt.other(node)
    return start, elements


lock_op = st.tuples(st.just("lock"), st.sampled_from(TRAINS), routes())
ALL_ELEMENTS = [e.id for e in (*NET.tracks, *NET.moves, *NET.platforms)]
ops = st.one_of(  # weighted towards locks so trains actually compete for moves, blocks and directions
    lock_op,
    lock_op,
    lock_op,
    st.tuples(st.just("release"), st.sampled_from(TRAINS), st.sampled_from(ALL_ELEMENTS)),
    st.tuples(st.just("release_train"), st.sampled_from(TRAINS), st.none()),
)


def snapshot(il):
    return dict(il.holder), {k: dict(v) for k, v in il.directions.items()}


@given(st.lists(ops, min_size=20, max_size=60))
def test_interlocking_safety_invariants(sequence):
    il = Interlocking(NET)
    for op, train, arg in sequence:
        before = snapshot(il)
        if op == "lock":
            if il.lock(train, *arg) is not None:
                assert snapshot(il) == before
            else:
                assert all(il.holder[k] == t for k, t in before[0].items())
        elif op == "release":
            il.release(train.id, arg)
        else:
            il.release_train(train.id)
        for resource in il.holder:
            if isinstance(NET.element(resource if isinstance(resource, str) else ""), Move):
                assert not any(c in il.holder for c in NET.conflicts(resource))
            if isinstance(resource, tuple):
                assert il.holder[resource] in il.directions.get(resource[0], {})
        for holders in il.directions.values():
            assert len(set(holders.values())) == 1
