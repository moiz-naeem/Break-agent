import math

import pytest

from railway.train.advantages import assign_advantages, group_advantages, normalise
from railway.train.samples import Trajectory, TurnSample


def traj(eid, group, reward, status="ok", turns=2):
    return Trajectory(
        episode_id=eid,
        env="x",
        seed=1,
        group=group,
        model="m",
        mode="windowed",
        status=status,
        cost=None if reward is None else -reward,
        reward=reward,
        turns=[TurnSample(turn=i + 1, messages=[], tools=[]) for i in range(turns)],
    )


def test_normalise_std():
    a = normalise([-10.0, -20.0, -30.0, -40.0])
    assert sum(a) == pytest.approx(0.0, abs=1e-9)
    assert math.sqrt(sum(x * x for x in a) / 4) == pytest.approx(1.0, rel=1e-4)
    assert a[0] > a[1] > a[2] > a[3]


def test_normalise_none_is_mean_centred():
    assert normalise([1.0, 3.0], scale="none") == [-1.0, 1.0]


def test_constant_group_gives_zero():
    assert normalise([-5.0, -5.0, -5.0]) == [0.0, 0.0, 0.0]


def test_bad_scale():
    with pytest.raises(ValueError):
        normalise([1.0], scale="rank")


def test_groups_are_separate_and_failed_episodes_dropped():
    ts = [
        traj("a0", "s1", -100.0),
        traj("a1", "s1", -200.0),
        traj("b0", "s2", -1000.0),
        traj("b1", "s2", -3000.0),
        traj("b2", "s2", None, status="error"),
    ]
    adv = group_advantages(ts)
    assert set(adv) == {"a0", "a1", "b0", "b1"}
    # scale-free: both groups give +1 / -1 although their cost scales differ by 10x
    assert adv["a0"] == pytest.approx(1.0, rel=1e-4) and adv["b0"] == pytest.approx(1.0, rel=1e-4)
    assert adv["a1"] == pytest.approx(-1.0, rel=1e-4)


def test_assign_puts_episode_advantage_on_every_turn():
    ts = [traj("a0", "s1", -1.0, turns=3), traj("a1", "s1", -3.0, turns=1), traj("x", "s1", None, status="error")]
    kept = assign_advantages(ts)
    assert [t.episode_id for t in kept] == ["a0", "a1"]
    assert all(s.advantage == pytest.approx(1.0, rel=1e-4) for s in kept[0].turns)
    assert kept[1].turns[0].advantage == pytest.approx(-1.0, rel=1e-4)
    assert ts[2].turns[0].advantage is None
