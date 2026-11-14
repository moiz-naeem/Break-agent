import re
from pathlib import Path

from railway.sim import constants
from railway.sim.cost import CostBreakdown
from railway.sim.manual import _value, render_manual
from railway.sim.rules import RuleId


def test_T04_01_manual_headings_rules_and_size():
    manual = render_manual()
    headings = [
        "# Dispatcher manual",
        "## Time and turns",
        "## Network",
        "## Train types",
        "## Movement authority",
        "## Rules",
        "## Factors",
        "## Events and lookaheads",
        "## Decision points",
        "## Cost",
        "## Score",
        "## Budgets and ending",
        "## Tools",
    ]
    positions = [manual.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert all(rule.value in manual for rule in RuleId)
    assert len(manual) <= 16_000
    assert not re.search(r"seed|generator|scenario\.json|heldout", manual, re.IGNORECASE)


def test_T04_02_manual_is_deterministic_and_lists_tools():
    first = render_manual()
    assert first == render_manual()
    for tool in (
        "manual",
        "network",
        "timetable",
        "status",
        "bulletins",
        "kpis",
        "stringline",
        "notes",
        "authorize",
        "cancel_authority",
        "hold",
        "relieve_crew",
        "advance_to_next_event",
        "end_shift",
    ):
        assert tool in first


def test_T04_09_manual_interpolates_constants(monkeypatch):
    from railway.sim import constants

    monkeypatch.setattr(constants, "TOOL_BUDGET", 123)
    assert "123" in render_manual()


def test_T04_03_code_rule_references_are_known():
    root = Path(__file__).parents[2] / "railway"
    rule_names = {rule.name for rule in RuleId}
    rule_values = {rule.value for rule in RuleId}
    for source in root.rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert all(name in rule_names for name in re.findall(r"RuleId\.([A-Z_]+)", text))
        assert all(value in rule_values for value in re.findall(r"R-[A-Z][A-Z-]+", text))


def test_T04_04_manual_lists_all_components_and_public_constants():
    manual = render_manual()
    assert all(name in manual for name in CostBreakdown.model_fields)
    for name, value in vars(constants).items():
        if name.isupper() and not name.startswith("__"):
            assert name in manual
            assert _value(value) in manual


def test_T04_05_manual_lists_every_decision_reason_kind():
    manual = render_manual()
    expected = {
        "AUTHORITY_END",
        "READY",
        "BULLETIN",
        "BREAKDOWN",
        "CREW_WARNING",
        "CREW_TIMEOUT",
        "GRIDLOCK",
        "TIME_CAP",
        "SHIFT_END",
    }
    assert all(kind in manual for kind in expected)


TOOLS = (
    "manual",
    "network",
    "timetable",
    "status",
    "bulletins",
    "kpis",
    "stringline",
    "notes",
    "authorize",
    "cancel_authority",
    "hold",
    "relieve_crew",
    "advance_to_next_event",
    "end_shift",
)


def test_T04_06_every_tool_in_tools_section():
    tools_section = render_manual().split("## Tools", 1)[1].split("\n## ", 1)[0]
    for tool in TOOLS:
        assert f"| `{tool}` |" in tools_section, tool


def test_T04_08_deterministic_without_timestamp():
    manual = render_manual()
    assert manual == render_manual()
    assert not re.search(r"\b(19|20)\d\d-\d\d-\d\d|\d\d:\d\d:\d\d", manual)
