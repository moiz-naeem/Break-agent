"""The training environment interface: reset(seed) -> Episode; step(tool, args) -> text; reward = -cost.

An adapter turns a seed into a (scenario, session) pair that `railway.loop.runner.play_episode` can drive
and that `Episode.step` can drive directly. Two adapters:

- `StandInEnv`: the v0.2 single-track game (`railway.tools.session.Session` on T1). Used for the smoke
  test until the network session lands.
- `NetEnv`: ADAPTER SLOT for the network game (`railway.net.session`, lane D). Same contract:
  `call(tool, args) -> str`, a tool budget, `closed`, `close() -> result with cost.total`.

Training seeds come only from the `train` range in `seeds/ranges.json`; `reset` refuses others unless
`allow_any_seed=True` (evaluation on dev or held-out seeds goes through the benchmark, not here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from railway.gen.seeds import ranges


class EnvSession(Protocol):
    closed: bool

    def call(self, tool_name: str, args: dict) -> str: ...

    def close(self) -> Any: ...


def train_range() -> tuple[int, int]:
    return ranges()["train"]


def is_train_seed(seed: int) -> bool:
    lo, hi = train_range()
    return lo <= seed < hi


def final_cost(final) -> float:
    """`cost.total` from a FinalResult-like object or mapping."""
    cost = final["cost"] if isinstance(final, dict) else final.cost
    return float(cost["total"] if isinstance(cost, dict) else cost.total)


@dataclass
class Episode:
    seed: int
    scenario: Any
    session: EnvSession
    tools: list[dict]
    meta: dict = field(default_factory=dict)
    _cost: float | None = None

    @property
    def done(self) -> bool:
        return bool(self.session.closed)

    def step(self, tool_name: str, args: dict) -> str:
        """Run one tool call and return its observation text (a call after the end is refused, not raised)."""
        return self.session.call(tool_name, args)

    def close(self) -> float:
        """End the episode (stopping counts as end_shift) and return the reward, -cost."""
        if self._cost is None:
            self._cost = final_cost(self.session.close())
        return -self._cost

    @property
    def cost(self) -> float | None:
        return self._cost

    @property
    def reward(self) -> float:
        return self.close()


class Env(Protocol):
    name: str

    def reset(self, seed: int) -> Episode: ...

    def greedy_cost(self, seed: int) -> float: ...


def _check_seed(seed: int, allow_any_seed: bool) -> None:
    if not allow_any_seed and not is_train_seed(seed):
        lo, hi = train_range()
        raise ValueError(f"seed {seed} is not a training seed ({lo} <= seed < {hi})")


class StandInEnv:
    """The v0.2 single-track game on a generator preset (T1 by default)."""

    name = "standin-v02"

    def __init__(self, preset: str = "T1", *, budget: int | None = None, allow_any_seed: bool = False):
        self.preset = preset
        self.budget = budget
        self.allow_any_seed = allow_any_seed

    def scenario(self, seed: int):
        from railway.gen.generator import generate
        from railway.gen.presets import PRESETS

        return generate(PRESETS[self.preset], seed)

    def reset(self, seed: int) -> Episode:
        from railway.tools.schemas import tool_schemas
        from railway.tools.session import Session

        _check_seed(seed, self.allow_any_seed)
        scenario = self.scenario(seed)
        session = Session(scenario) if self.budget is None else Session(scenario, budget=self.budget)
        return Episode(seed, scenario, session, tool_schemas(), {"env": self.name, "preset": self.preset})

    def greedy_cost(self, seed: int) -> float:
        from railway.policies.baselines import Greedy
        from railway.policies.runner import run_policy

        return float(run_policy(Greedy(), self.scenario(seed)).final.cost.total)


# -- the v0.3 network game ---------------------------------------------------------------------------
#
# Seeds come from the network's own disjoint splits (`railway.net.exploits.SPLITS`: dev, test, train),
# not from `seeds/ranges.json` (v0.2). Inside `train`, every seed divisible by VAL_MODULUS is a
# validation seed: used to pick training settings, never trained on. `test` is never touched here.
VAL_MODULUS = 10


def net_split(seed: int) -> str:
    from railway.net.exploits import split_of

    return split_of(seed)


def is_val_seed(seed: int) -> bool:
    return net_split(seed) == "train" and seed % VAL_MODULUS == 0


def is_net_train_seed(seed: int) -> bool:
    """A network seed we may train on: in the `train` split and not a validation seed."""
    return net_split(seed) == "train" and not is_val_seed(seed)


def net_seed_list(tier: str, split: str) -> list[dict]:
    """Accepted seeds of `seeds/net-<tier>-<split>.json` (written by infra/modal/net_seeds.py)."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "seeds" / f"net-{tier}-{split}.json"
    return json.loads(path.read_text(encoding="utf-8"))["accepted"]


class NetEnv:
    """The network game: `build_timetable(tier, seed)` (the seed-list scenario, with disruptions) and
    `railway.net.session.Session`. There is no greedy baseline (decisions.md #4): `greedy_cost` is None."""

    name = "net-v03"

    def __init__(self, preset: str = "T1", *, allow_any_seed: bool = False, allow_val: bool = False):
        self.preset = preset
        self.allow_any_seed = allow_any_seed
        self.allow_val = allow_val

    def check_seed(self, seed: int) -> None:
        if self.allow_any_seed:
            return
        if net_split(seed) != "train":
            raise ValueError(f"seed {seed} is in the {net_split(seed)!r} split, not 'train'")
        if is_val_seed(seed) and not self.allow_val:
            raise ValueError(f"seed {seed} is a validation seed (seed % {VAL_MODULUS} == 0)")

    def scenario(self, seed: int):
        from railway.net.exploits import build_timetable

        return build_timetable(self.preset, seed)

    def reset(self, seed: int) -> Episode:
        from railway.loop.runner import _tool_specs
        from railway.net.session import Session

        self.check_seed(seed)
        timetable = self.scenario(seed)
        session = Session(timetable)
        return Episode(seed, timetable, session, _tool_specs(session, ()), {"env": self.name, "preset": self.preset})

    def greedy_cost(self, seed: int) -> None:
        return None

    def train_seeds(self, count: int, rng_seed: int = 0) -> list[int]:
        """`count` trainable seeds from the accepted train seed list (validation seeds excluded)."""
        import random

        pool = [int(r["seed"]) for r in net_seed_list(self.preset, "train") if is_net_train_seed(int(r["seed"]))]
        if count > len(pool):
            raise ValueError(f"asked for {count} train seeds, the seed list has {len(pool)}; generate more")
        return random.Random(rng_seed).sample(pool, count)

    def val_seeds(self) -> list[int]:
        return [int(r["seed"]) for r in net_seed_list(self.preset, "train") if is_val_seed(int(r["seed"]))]


ENVS = {"standin": StandInEnv, "net": NetEnv}


def make_env(name: str, **kwargs) -> Env:
    if name not in ENVS:
        raise ValueError(f"unknown env {name!r}; choose one of {sorted(ENVS)}")
    return ENVS[name](**kwargs)
