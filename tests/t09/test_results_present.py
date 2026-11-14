"""T09 §5.5 / §5.6 outcome checks. They need the real runs, datasets and README; they skip
(with the reason) until those exist. T09-20's text checks also run on the README draft."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
import yaml

from tests.helpers import REPO, harbor_run

README_SECTIONS = (
    "What this is",
    "Quickstart",
    "Run the judges' checks",
    "How scoring works",
    "Why it's hard",
    "Exploit analysis",
    "Reproduce our numbers",
    "Repo map",
    "Limits and honest caveats",
)
README_COMMANDS = (
    "uv sync && make ci",
    "harbor run -p datasets/t2-heldout -a oracle",
    "railway-eval run",
    "make report",
)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_T09_17_heldout_results():
    """T09-17: each Harbor held-out run has 60 rows, ≥ 90% ok, rules_version 1.0 on every row."""
    manifests = sorted((REPO / "runs").glob("heldout-*.yaml"))
    harbor = [m for m in manifests if yaml.safe_load(m.read_text(encoding="utf-8")).get("harness") == "harbor"]
    if not harbor:
        pytest.skip("waiting for the held-out Harbor runs (blocking input: models)")
    for m in harbor:
        results = REPO / "results" / f"{m.stem}.jsonl"
        assert results.exists(), results
        rows = _rows(results)
        expected = 30 if m.stem.endswith("-s14") else 60  # S14: 10 tasks × 3 reps
        assert len(rows) == expected, m.stem
        assert sum(r["status"] == "ok" for r in rows) >= 0.9 * len(rows), m.stem
        assert all(r["rules_version"] == "1.0" for r in rows), m.stem


def test_T09_18_baselines():
    """T09-18: greedy unclipped 0, reference 1, do-nothing and stop-early clipped 0, on all 20 tasks."""
    path = REPO / "results" / "heldout-baselines.jsonl"
    if not path.exists():
        pytest.skip("waiting for `railway-eval baselines` on datasets/t2-heldout (T05/T06/T08 + freeze)")
    by_model: dict[str, list[dict]] = {}
    for r in _rows(path):
        by_model.setdefault(r["model"], []).append(r)
    for model, check in (
        ("greedy", lambda r: r["score_unclipped"] == 0),
        ("reference", lambda r: r["score_unclipped"] == 1),
        ("do-nothing", lambda r: r["score_clipped"] == 0),
        ("stop-early", lambda r: r["score_clipped"] == 0),
    ):
        rows = by_model.get(model, [])
        assert len({r["scenario_id"] for r in rows}) == 20, model
        assert all(r["harness"] == "script" and check(r) for r in rows), model


def check_readme(text: str, *, full: bool) -> None:
    for command in README_COMMANDS:
        assert command in text, command
    assert not EMAIL.search(text), "no email addresses in the README"
    if full:
        headings = [h.strip() for h in re.findall(r"^##\s+(?:\d+\.\s*)?(.+)$", text, flags=re.M)]
        for section in README_SECTIONS:
            assert any(h.startswith(section) for h in headings), section


def test_T09_20_readme():
    """T09-20: README.md has the 9 sections of §2.7, the key commands and no email address."""
    readme = REPO / "README.md"
    if not readme.exists():
        pytest.skip("README.md not written yet (draft: docs/README-evaluation-draft.md)")
    check_readme(readme.read_text(encoding="utf-8"), full=True)


def test_T09_20_readme_draft():
    """T09-20 (draft): the evaluation draft has the commands, no email, and no score numbers yet."""
    text = (REPO / "docs" / "README-evaluation-draft.md").read_text(encoding="utf-8")
    check_readme(text, full=False)
    assert not re.search(r"score_ros\s*[=:]\s*-?\d", text), "no result numbers before the real runs"


def _dataset(name: str) -> Path:
    path = REPO / "datasets" / name
    if not path.exists():
        pytest.skip(f"waiting for datasets/{name} (post-freeze export)")
    return path


@pytest.mark.harbor
def test_T09_H1_oracle_heldout():
    trials = harbor_run(_dataset("t2-heldout"), "oracle")
    assert len(trials) == 20 and all(not t.errored and t.reward == 1.0 for t in trials)


@pytest.mark.harbor
def test_T09_H2_nop_heldout():
    trials = harbor_run(_dataset("t2-heldout"), os.environ.get("HARBOR_NOP_AGENT", "nop"))
    assert len(trials) == 20 and all(not t.errored and t.reward == 0.0 for t in trials)


@pytest.mark.harbor
def test_T09_H3_stop_early_heldout():
    pytest.importorskip("railway.harbor.agents", reason="StopEarlyAgent (T07 §2.9) not present")
    trials = harbor_run(_dataset("t2-heldout"), "railway.harbor.agents:StopEarlyAgent")
    assert len(trials) == 20 and all(not t.errored and t.reward == 0.0 for t in trials)


@pytest.mark.harbor
def test_T09_H4_oracle_s14():
    trials = harbor_run(_dataset("t2s14-heldout"), "oracle")
    assert len(trials) == 10 and all(not t.errored and t.reward == 1.0 for t in trials)
