"""Group-normalised episode advantages for multi-turn GRPO (pure; no torch).

A group is the G episodes of one seed sampled with the same policy. Each episode's advantage
(reward - group mean) / (group std + eps) applies to every assistant turn of that episode.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

from railway.train.samples import Trajectory

EPS = 1e-6


def normalise(rewards: list[float], *, scale: str = "std", eps: float = EPS) -> list[float]:
    """`std`: (r - mean) / (std + eps); `none`: r - mean (Dr. GRPO). A constant group gives zeros."""
    if scale not in ("std", "none"):
        raise ValueError(f"scale must be 'std' or 'none', not {scale!r}")
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    centred = [r - mean for r in rewards]
    if scale == "none":
        return centred
    std = math.sqrt(sum(c * c for c in centred) / len(centred))
    if std < eps:
        return [0.0] * len(rewards)
    return [c / (std + eps) for c in centred]


def group_advantages(trajectories: Iterable[Trajectory], *, scale: str = "std") -> dict[str, float]:
    """episode_id -> advantage. Failed episodes (status != ok) get none and are left out of their group."""
    groups: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        if t.ok:
            groups[t.group].append(t)
    out: dict[str, float] = {}
    for members in groups.values():
        for t, a in zip(members, normalise([t.reward for t in members], scale=scale), strict=True):
            out[t.episode_id] = a
    return out


def assign_advantages(trajectories: list[Trajectory], *, scale: str = "std") -> list[Trajectory]:
    """Write each episode's advantage onto all of its turns; return the episodes that got one."""
    adv = group_advantages(trajectories, scale=scale)
    kept = []
    for t in trajectories:
        if t.episode_id not in adv:
            continue
        for turn in t.turns:
            turn.advantage = adv[t.episode_id]
        kept.append(t)
    return kept
