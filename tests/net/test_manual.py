"""Tests for `railway/net/manual.py` and `railway/net/cost_constants.py`."""

from __future__ import annotations

import re

from railway.net import cost_constants as cc
from railway.net.manual import INSTRUCTION, render_manual
from railway.net.rules import NetRule
from railway.net.session import TOOLS


def test_every_rule_id_in_rules_appears_in_manual():
    manual = render_manual()
    missing = [rule.value for rule in NetRule if rule.value not in manual]
    assert missing == []


def test_manual_sections_size_determinism_and_no_forbidden_terms():
    manual = render_manual()
    assert manual == render_manual()
    assert len(manual) <= 16_000

    headings = [
        "# Network dispatcher manual",
        "## Time, turns and budget",
        "## Network and track kinds",
        "## Train types",
        "## Automatic route setting and conflicts",
        "## Decision points",
        "## Tools",
        "## Rules",
        "## Cost and reward",
    ]
    positions = [manual.index(h) for h in headings]
    assert positions == sorted(positions)

    # No mention of seeds, generators, files or timestamps.
    assert not re.search(r"seed|generator|file|heldout", manual, re.IGNORECASE)
    assert not re.search(r"seed|generator|file|heldout", INSTRUCTION, re.IGNORECASE)
    assert not re.search(r"\b(19|20)\d\d-\d\d-\d\d|\d\d:\d\d:\d\d", manual)


def test_manual_states_exact_reward_and_imports_cost_constants(monkeypatch):
    manual = render_manual()
    assert "Reward: minus your total cost. Nothing else counts." in manual

    expected_constants = (
        "PAX_GRACE",
        "LATE_STEPS",
        "VALUE",
        "MISSED_CALL_FLAT",
        "MISSED_CALL_HASSLE",
        "DEFAULT_HEADWAY",
        "MAX_HEADWAY",
        "UNFINISHED_COST",
        "UNDELIVERED_FREIGHT",
        "W_FREIGHT",
        "FREIGHT_LATE_STEP",
        "PORT_CUTOFF_COST",
        "GRIDLOCK_COST",
        "UNPLANNED_STOP_COST",
        "PLATFORM_CHANGE_MINUTES",
        "TIMEOUT_COST",
    )
    for name in expected_constants:
        assert hasattr(cc, name), name
        assert f"| `{name}` |" in manual

    # Confirm render_manual interpolates live values from cost_constants rather than hardcoding.
    monkeypatch.setattr(cc, "GRIDLOCK_COST", 7654)
    assert "7654" in render_manual()


def test_manual_lists_every_registered_tool_and_decision_kind():
    manual = render_manual()
    tools_section = manual.split("## Tools", 1)[1].split("\n## ", 1)[0]
    for name in TOOLS:
        assert f"| `{name}` |" in tools_section, name
    for kind in ("CONFLICT", "UNROUTABLE", "BLOCKED", "IDLE", "SHIFT_END"):
        assert f"`{kind}`" in manual
