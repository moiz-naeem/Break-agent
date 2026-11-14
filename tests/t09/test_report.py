"""T09 §5.3 report tests on SYNTHETIC fixture result files (no models).

`summary_test_only` stands in for T08's stats.py so the rendering can be tested now;
T09-08 proper runs against the real stats.py and skips until T08 lands.
"""

from __future__ import annotations

import hashlib
import html
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from railway.eval.report import LEGEND, build_report, embedded_hashes
from tests.helpers import REPO
from tests.t09.synthetic import make_row, model_rows, summary_test_only, write_jsonl

SEEDS = range(1, 21)


def _harbor(model, score):
    rows = []
    for rep in range(3):
        rows += model_rows(model, score + 0.01 * rep, seeds=SEEDS, rep=rep, run=f"heldout-{model}")
    return rows


def _fixture(tmp_path: Path, *, ablation=False, training=False) -> Path:
    small = _harbor("small", 0.2)
    small[0]["status"], small[0]["error"] = "error", "synthetic infra error"
    big = _harbor("big", 0.6)
    big[5].update(score_unclipped=1.2, cost_total=1000 - 1.2 * 400, score_clipped=1.0)  # flagged: seed 6, rep 0
    loop = [
        make_row(
            model="big",
            harness="loop",
            mode="windowed",
            seed=s,
            score=0.7,
            run="heldout-big-loop",
            format_errors=2,
            tokens_in=50_000,
            tokens_out=2_000,
            invalid_calls=3,
            gridlocks=1,
        )
        for s in SEEDS
    ]
    base = []
    for model, score in (("greedy", 0.0), ("reference", 1.0), ("do-nothing", -3.0)):
        base += [make_row(model=model, harness="script", seed=s, score=score, run="heldout-baselines") for s in SEEDS]
    files = {
        "results/heldout-small.jsonl": small,
        "results/heldout-big.jsonl": big,
        "results/heldout-big-loop.jsonl": loop,
        "results/heldout-baselines.jsonl": base,
    }
    for rel, rows in files.items():
        write_jsonl(tmp_path / rel, rows)
    spec = {
        "name": "synthetic",
        "dataset": "datasets/t2-heldout",
        "results_date": "2026-01-01",
        "results": list(files),
    }
    if ablation:
        abl = []
        for f, (s, b) in {"F5": (0.1, 0.5), "F10": (0.0, 0.6)}.items():
            for model, score in (("small", s), ("big", b)):
                abl += model_rows(
                    model, score, seeds=range(1, 6), preset=f"T2-only-{f}", harness="loop", mode="windowed"
                )
        write_jsonl(tmp_path / "results/ablation.jsonl", abl)
        spec["ablation"] = ["results/ablation.jsonl"]
    if training:
        write_jsonl(tmp_path / "results/before.jsonl", model_rows("small", 0.2, seeds=SEEDS, harness="loop"))
        write_jsonl(tmp_path / "results/after.jsonl", model_rows("small-trained", 0.35, seeds=SEEDS, harness="loop"))
        spec["training"] = {"before": "results/before.jsonl", "after": "results/after.jsonl"}
    path = tmp_path / "runs" / "report-synthetic.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return path


def _render(tmp_path, summarize=summary_test_only, **kw):
    return build_report(_fixture(tmp_path, **kw), root=tmp_path, summarize=summarize)


def _charts(doc: str) -> list[str]:
    return re.findall(r'<section class="chart"[^>]*>(.*?)</section>', doc, flags=re.S)


@pytest.mark.parametrize("full", [False, True])
def test_T09_07_self_contained(tmp_path, full):
    """T09-07: no remote src/href, no <script src>, exactly 3 charts, each an inline svg or a placeholder."""
    doc = _render(tmp_path, ablation=full, training=full)
    assert not re.search(r"""(?:src|href)\s*=\s*["']?\s*https?://""", doc, flags=re.I)
    assert not re.search(r"<script[^>]*\bsrc\s*=", doc, flags=re.I)
    assert not re.search(r"url\(\s*['\"]?https?://|@import", doc, flags=re.I)
    assert not re.search(r"<link\b", doc, flags=re.I)
    charts = _charts(doc)
    assert len(charts) == 3
    for body in charts:
        assert "<svg" in body or 'class="placeholder"' in body
    if full:
        assert all("<svg" in body for body in charts)


def _table_cells(doc: str) -> dict[tuple[str, str], tuple[str, str]]:
    """(group, key) → (data-value, shown text) for every numeric cell of the main table."""
    table = re.search(r'<table id="results">(.*?)</table>', doc, flags=re.S).group(1)
    cells = {}
    for row in re.findall(r'<tr data-group="([^"]*)">(.*?)</tr>', table, flags=re.S):
        group, body = html.unescape(row[0]), row[1]
        for key, value, text in re.findall(r'<td data-key="([^"]+)" data-value="([^"]*)">(.*?)</td>', body):
            cells[(group, key)] = (value, html.unescape(text))
    return cells


def _check_table(doc: str, tmp_path: Path, summarize) -> None:
    import json

    cells = _table_cells(doc)
    groups = {}
    for f in sorted((tmp_path / "results").glob("heldout-*.jsonl")):
        for r in map(json.loads, f.read_text(encoding="utf-8").splitlines()):
            groups.setdefault(f"{r['model']}|{r['harness']}|{r['mode'] or ''}", []).append(r)
    assert {g for g, _ in cells} == set(groups)
    keys = ("score_ros", "reward_mean", "unclipped_mean", "cost_ratio", "gridlocks", "crew_timeouts", "invalid_calls")
    for group, rows in groups.items():
        expected = summarize(rows)
        for key in keys:
            value, text = cells[(group, key)]
            assert float(value) == expected[key], (group, key)
            assert text == f"{expected[key]:.3f}"
        lo, hi = expected["ci"]
        assert cells[(group, "ci")][0] == f"{lo!r},{hi!r}"
        assert cells[(group, "episodes")][1] == f"{expected['episodes']} / {expected['errors']}"
        fe = cells[(group, "format_errors")]
        if group.split("|")[1] == "loop":
            assert float(fe[0]) == expected["format_errors"]
        else:
            assert fe == ("", "–")


def test_T09_08_table_numbers_rendering(tmp_path):
    """T09-08 (rendering, with the test-local summary): every numeric cell equals the summary."""
    doc = _render(tmp_path)
    _check_table(doc, tmp_path, summary_test_only)
    big = _table_cells(doc)[("big|harbor|", "flagged")][1]
    assert big == "T1-synth0000006"


def test_T09_08_table_numbers(tmp_path):
    """T09-08: every numeric cell equals stats.py recomputed from the fixture rows."""
    pytest.importorskip("railway.eval.stats")
    from railway.eval._stats import summarize

    doc = build_report(_fixture(tmp_path), root=tmp_path)
    _check_table(doc, tmp_path, summarize)


def test_T09_08_baselines_after_models(tmp_path):
    """T09-08 / §2.4: one row per model and harness in display order, then the baselines."""
    table = re.search(r'<table id="results">(.*?)</table>', _render(tmp_path), flags=re.S).group(1)
    groups = re.findall(r'<tr data-group="([^"]*)">', table)
    assert groups == [
        "small|harbor|",
        "big|harbor|",
        "big|loop|windowed",
        "greedy|script|",
        "reference|script|",
        "do-nothing|script|",
    ]


def test_T09_08_failure_breakdown(tmp_path):
    """T09-08 / §2.4 item 6: mean cost by component per model, from ok rows."""
    doc = _render(tmp_path)
    table = re.search(r'<table id="failures">(.*?)</table>', doc, flags=re.S).group(1)
    row = re.search(r'<tr data-group="big\|loop\|windowed">(.*?)</tr>', table, flags=re.S).group(1)
    pax = re.search(r'data-key="pax_lateness" data-value="([^"]*)"', row).group(1)
    assert float(pax) == pytest.approx(0.75 * (1000 - 0.7 * 400))


def test_T09_09_deterministic(tmp_path):
    """T09-09: two renders of the same input → identical bytes, also across processes."""
    spec = _fixture(tmp_path, ablation=True, training=True)
    a = build_report(spec, root=tmp_path, summarize=summary_test_only)
    b = build_report(spec, root=tmp_path, summarize=summary_test_only)
    assert a.encode() == b.encode()
    code = (
        "import sys; from pathlib import Path; from railway.eval.report import build_report;"
        "from tests.t09.synthetic import summary_test_only as s;"
        "sys.stdout.buffer.write(build_report(Path(sys.argv[1]), root=Path(sys.argv[2]), summarize=s).encode())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(spec), str(tmp_path)], cwd=REPO, capture_output=True, check=True
    ).stdout
    assert out == a.encode()


def test_T09_10_placeholders(tmp_path):
    """T09-10: without ablation or training data → `ablation not run` and `pending T10`."""
    charts = _charts(_render(tmp_path))
    assert "<svg" in charts[0]
    assert "ablation not run" in charts[1] and "<svg" not in charts[1]
    assert "pending T10" in charts[2] and "<svg" not in charts[2]


def test_T09_10_charts_when_data_present(tmp_path):
    """T09-10: with ablation and training data the placeholders are gone."""
    doc = _render(tmp_path, ablation=True, training=True)
    assert "ablation not run" not in doc and "pending T10" not in doc


def test_T09_11_legend(tmp_path):
    """T09-11: the headline/diagnostic legend sentence of §2.4."""
    assert LEGEND == (
        "Harbor + OpenCode numbers are headline. Our-loop numbers are diagnostic: the loop shows the state for free."
    )
    doc = _render(tmp_path)
    assert html.escape(LEGEND) in doc
    assert "2026-01-01" in doc  # date of the results files, from the manifest
    assert "datasets/t2-heldout" in doc
    assert "0.1-provisional" in doc


def test_T09_11_escapes_names(tmp_path):
    """Model names are escaped, never injected as markup."""
    spec = _fixture(tmp_path)
    write_jsonl(tmp_path / "results/heldout-small.jsonl", model_rows("<b>x</b>", 0.2, seeds=SEEDS))
    doc = build_report(spec, root=tmp_path, summarize=summary_test_only)
    assert "<b>x</b>" not in doc and "&lt;b&gt;x&lt;/b&gt;" in doc


def test_T09_19_hashes_embedded(tmp_path):
    """T09-19 (mechanism): the report embeds each results file's sha256; recomputing matches."""
    doc = _render(tmp_path)
    hashes = embedded_hashes(doc)
    assert set(hashes) == {
        "results/heldout-small.jsonl",
        "results/heldout-big.jsonl",
        "results/heldout-big-loop.jsonl",
        "results/heldout-baselines.jsonl",
    }
    for rel, digest in hashes.items():
        assert digest == hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest()


def test_T09_19_report_built():
    """T09-19: reports/heldout.html exists and was built from the current results files."""
    report = REPO / "reports" / "heldout.html"
    if not report.exists():
        pytest.skip("waiting for the held-out runs and the report build (blocking input: models)")
    hashes = embedded_hashes(report.read_text(encoding="utf-8"))
    assert hashes
    for rel, digest in hashes.items():
        assert digest == hashlib.sha256((REPO / rel).read_bytes()).hexdigest(), rel


def test_T09_07_cli_writes_file(tmp_path):
    """The `report` entry point writes the HTML file (T08 wires it into `railway-eval report`)."""
    from railway.eval.report import main

    spec = _fixture(tmp_path)
    out = tmp_path / "reports" / "synthetic.html"
    # The CLI uses the real stats.py; without it, it must fail loudly rather than invent numbers.
    try:
        import railway.eval.stats  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            main([str(spec), "--out", str(out), "--root", str(tmp_path)])
        assert not out.exists()
        return
    assert main([str(spec), "--out", str(out), "--root", str(tmp_path)]) == 0
    assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
