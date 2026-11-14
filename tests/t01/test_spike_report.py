import re

from tests.helpers import REPO

FACTS = ["F-REACH", "F-ALLOW", "F-NONROOT", "F-SMALL", "F-CUSTOM", "F-NOP", "F-SECRET", "F-LAYOUT"]


def test_T01_20_report_complete():
    text = (REPO / "docs" / "spike-report.md").read_text(encoding="utf-8")
    for fact in FACTS:
        assert re.search(rf"{fact}\b.*\b(PASS|FAIL|BLOCKED)\b", text), fact
    assert re.search(r"^Decision: (SIDECAR|FALLBACK)", text, re.M)
    assert "Harbor version:" in text
