"""Shared T06 helpers: fixtures, dev scenarios and a reference driven to its k-th decision point."""

from railway.gen.generator import generate
from railway.gen.presets import PRESETS
from railway.gen.seeds import candidate_seeds
from railway.policies.runner import run_policy
from railway.ref.reference import Reference
from railway.sim.model import Scenario
from railway.tools.observe import Observer
from railway.tools.session import Session
from tests.helpers import load_fixture


def fixture(name: str, **patch) -> Scenario:
    return Scenario.model_validate(load_fixture(name) | patch)


def dev(preset: str, n: int, offset: int = 0) -> list[Scenario]:
    seeds = list(candidate_seeds("dev", count=n + offset))[offset:]
    return [generate(PRESETS[preset], seed) for seed in seeds]


def cost(policy, scenario) -> float:
    return run_policy(policy, scenario).final.cost.total


def at_decision_point(scenario: Scenario, k: int):
    """(static, state) seen by the reference at its k-th decision point (k=1 is the start)."""
    session = Session(scenario)
    observer, reference = Observer(session), Reference()
    for _ in range(k - 1):
        static, state = observer.observe()
        for call in reference.decide(static, state):
            session.call(call.tool, call.args)
    return observer.observe()
