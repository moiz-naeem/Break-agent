"""One self-contained HTML report from results JSONL files (DESIGN.md §9.6, T09 §2.4).

    python -m railway.eval.report runs/report-<name>.yaml --out reports/<name>.html

(T08 owns the `railway-eval` CLI; its `report` subcommand should call `main`.)

Report manifest (`runs/report-<name>.yaml`; paths relative to the repo root):

    name: heldout
    dataset: datasets/t2-heldout
    results:                       # display order; harness "script" rows go last as baselines
      - results/heldout-small.jsonl
      - results/heldout-big.jsonl
      - results/heldout-baselines.jsonl
    ablation: []                   # optional: rows with presets T2-only-<F> (chart 2)
    training: null                 # optional: {before: <jsonl>, after: <jsonl>} (chart 3, T10)
    results_date: null             # optional override; default = newest results file's mtime (UTC date)

Output: inline CSS, charts as inline matplotlib SVG (fixed hashsalt, no date), no
scripts, no external URLs; byte-deterministic for the same inputs. Each input file's
sha256 is embedded in a comment (`embedded_hashes`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import io
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from railway.eval._stats import SummarizeFn, Summary
from railway.eval.diagnostics import COMPONENTS, ablation_gaps, failure_breakdown, group_rows

LEGEND = "Harbor + OpenCode numbers are headline. Our-loop numbers are diagnostic: the loop shows the state for free."
ABLATION_PLACEHOLDER = "ablation not run"
TRAINING_PLACEHOLDER = "pending T10"

# Reference categorical palette (dataviz skill), fixed order: small = slot 1, big = slot 2.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd5", "#fcfcfb"


@dataclass(frozen=True)
class ReportSpec:
    name: str
    dataset: str
    results: list[str]
    ablation: list[str] = field(default_factory=list)
    training: dict[str, str] | None = None
    results_date: str | None = None


def load_spec(path: Path) -> ReportSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    allowed = {"name", "dataset", "results", "ablation", "training", "results_date"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown report manifest keys: {sorted(unknown)}")
    if not raw.get("results"):
        raise ValueError("report manifest needs a non-empty `results` list")
    training = raw.get("training")
    if training is not None and set(training) != {"before", "after"}:
        raise ValueError("`training` needs exactly `before` and `after`")
    return ReportSpec(
        name=str(raw["name"]),
        dataset=str(raw.get("dataset", "")),
        results=[str(p) for p in raw["results"]],
        ablation=[str(p) for p in raw.get("ablation") or []],
        training={k: str(v) for k, v in training.items()} if training else None,
        results_date=str(raw["results_date"]) if raw.get("results_date") else None,
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _default_summarize(rows: list[dict]) -> Summary:
    from railway.eval._stats import summarize

    return summarize(rows)


# ---- formatting ---------------------------------------------------------------------------

_E = html.escape


def _cell(key: str, value, text: str | None = None) -> str:
    if value is None:
        return f'<td data-key="{key}" data-value="">–</td>'
    shown = text if text is not None else f"{value:.3f}"
    raw = repr(value) if isinstance(value, float) else str(value)
    return f'<td data-key="{key}" data-value="{_E(raw)}">{_E(shown)}</td>'


def _label(group: str) -> tuple[str, str, str]:
    model, harness, mode = group.split("|")
    return model, harness, mode


def _results_row(group: str, s: Summary | None, rows: list[dict]) -> str:
    model, harness, mode = _label(group)
    head = f"<td>{_E(model)}</td><td>{_E(harness)}</td><td>{_E(mode or '–')}</td>"
    errors = sum(r["status"] != "ok" for r in rows)
    if s is None:
        empty = "".join(
            _cell(k, None)
            for k in (
                "score_ros",
                "ci",
                "reward_mean",
                "unclipped_mean",
                "cost_ratio",
                "gridlocks",
                "crew_timeouts",
                "invalid_calls",
                "format_errors",
                "tokens",
                "flagged",
            )
        )
        episodes = _cell("episodes", f"{len(rows)},{errors}", f"{len(rows)} / {errors}")
        return f'<tr data-group="{_E(group)}">{head}{empty}{episodes}</tr>'
    ci = s.get("ci")
    cells = [
        _cell("score_ros", s["score_ros"]),
        _cell("ci", None) if ci is None else _cell("ci", f"{ci[0]!r},{ci[1]!r}", f"[{ci[0]:.3f}, {ci[1]:.3f}]"),
        _cell("reward_mean", s["reward_mean"]),
        _cell("unclipped_mean", s["unclipped_mean"]),
        _cell("cost_ratio", s["cost_ratio"]),
        _cell("gridlocks", s["gridlocks"]),
        _cell("crew_timeouts", s["crew_timeouts"]),
        _cell("invalid_calls", s["invalid_calls"]),
        _cell("format_errors", s["format_errors"] if harness == "loop" else None),
        _cell("tokens", s["tokens"], None if s["tokens"] is None else f"{s['tokens']:.0f}"),
        _cell("episodes", f"{s['episodes']},{s['errors']}", f"{s['episodes']} / {s['errors']}"),
        _cell("flagged", ",".join(s["flagged"]), ", ".join(s["flagged"]) or "none"),
    ]
    return f'<tr data-group="{_E(group)}">{head}{"".join(cells)}</tr>'


def _failure_row(group: str, rows: list[dict]) -> str:
    model, harness, mode = _label(group)
    fb = failure_breakdown(rows)
    head = f"<td>{_E(model)}</td><td>{_E(harness)}</td><td>{_E(mode or '–')}</td>"
    cells = "".join(
        _cell(k, None if fb is None else fb[k], None if fb is None else f"{fb[k]:.1f}") for k in (*COMPONENTS, "total")
    )
    return f'<tr data-group="{_E(group)}">{head}{cells}</tr>'


# ---- charts ----------------------------------------------------------------------------------


def _svg(fig, salt: str) -> str:
    import matplotlib

    buf = io.StringIO()
    with matplotlib.rc_context({"svg.hashsalt": salt, "svg.fonttype": "path"}):
        fig.savefig(buf, format="svg", metadata={"Date": None}, facecolor=SURFACE)
    import matplotlib.pyplot as plt

    plt.close(fig)
    text = buf.getvalue()
    return text[text.index("<svg") :]


def _axes(width=7.0, height=3.2):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height), dpi=100)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _anchor_lines(ax, right: float) -> None:
    for y, text in ((1.0, "reference = 1"), (0.0, "greedy = 0")):
        ax.axhline(y, color=MUTED, linewidth=1, linestyle="--")
        ax.text(right, y, f" {text}", color=MUTED, fontsize=8, va="center", ha="left")


def _chart_models(groups: list[tuple[str, Summary]]) -> str:
    fig, ax = _axes()
    xs = list(range(len(groups)))
    for x, (group, s) in zip(xs, groups, strict=True):
        model, harness, mode = _label(group)
        color = SERIES[0] if harness == "harbor" else SERIES[2]
        ci = s.get("ci")
        yerr = None if ci is None else [[s["score_ros"] - ci[0]], [ci[1] - s["score_ros"]]]
        ax.bar(x, s["score_ros"], width=0.6, color=color, yerr=yerr, ecolor=INK, capsize=4)
        top = max(s["score_ros"], ci[1] if ci is not None else s["score_ros"], 0)
        ax.text(x, top + 0.02, f"{s['score_ros']:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(
        xs, [f"{_label(g)[0]}\n{_label(g)[1]}" + (f" {_label(g)[2]}" if _label(g)[2] else "") for g, _ in groups]
    )
    _anchor_lines(ax, len(groups) - 0.5)
    ax.set_ylabel("score_ros (95% CI)", color=MUTED, fontsize=9)
    ax.set_title("Score by model (blue = Harbor + OpenCode, green = our loop)", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    return _svg(fig, "railway-chart-1")


def _chart_ablation(gaps: dict[str, dict]) -> str:
    fig, ax = _axes()
    factors = list(gaps)
    xs = list(range(len(factors)))
    for offset, (role, color) in ((-0.2, ("small", SERIES[0])), (0.2, ("big", SERIES[1]))):
        vals = [gaps[f][role] for f in factors]
        ax.bar(
            [x + offset for x, v in zip(xs, vals, strict=True) if v is not None],
            [v for v in vals if v is not None],
            width=0.38,
            color=color,
            label=role,
        )
    ax.set_xticks(xs, factors)
    _anchor_lines(ax, len(factors) - 0.5)
    ax.set_ylabel("score_ros on T2-only-<factor>", color=MUTED, fontsize=9)
    ax.set_title("Score by factor (ablation)", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return _svg(fig, "railway-chart-2")


def _chart_training(before: float, after: float) -> str:
    fig, ax = _axes(width=4.5)
    ax.bar([0, 1], [before, after], width=0.55, color=[SERIES[0], SERIES[2]])
    for x, v in ((0, before), (1, after)):
        ax.text(x, max(v, 0), f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks([0, 1], ["before training", "after training"])
    _anchor_lines(ax, 1.5)
    ax.set_ylabel("score_ros", color=MUTED, fontsize=9)
    ax.set_title("Small model before vs after training", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    return _svg(fig, "railway-chart-3")


def _chart(n: int, title: str, body: str | None, placeholder: str) -> str:
    inner = body if body is not None else f'<p class="placeholder">{_E(placeholder)}</p>'
    return f'<section class="chart" id="chart-{n}"><h3>{_E(title)}</h3>{inner}</section>'


# ---- assembly ------------------------------------------------------------------------------

_CSS = """
:root{color-scheme:light}
body{background:#fcfcfb;color:#0b0b0b;font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;
margin:0 auto;max-width:1100px;padding:16px}
h1{font-size:22px;margin:8px 0}h2{font-size:17px;margin:28px 0 8px}h3{font-size:14px;margin:0 0 6px}
.meta{color:#52514e}.legend{border-left:3px solid #2a78d6;padding:6px 10px;background:#f1f0ec}
.scroll{overflow-x:auto}table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}
th,td{padding:4px 8px;border-bottom:1px solid #dcdbd5;text-align:right;white-space:nowrap}
th:nth-child(-n+3),td:nth-child(-n+3){text-align:left}
.chart{margin:18px 0}.chart svg{max-width:100%;height:auto}
.placeholder{color:#52514e;font-style:italic;border:1px dashed #dcdbd5;padding:24px;text-align:center}
""".strip()

_RESULT_HEAD = (
    "<tr><th>model</th><th>harness</th><th>mode</th><th>score_ros</th><th>95% CI</th><th>reward_mean</th>"
    "<th>unclipped_mean</th><th>cost_ratio</th><th>gridlocks/ep</th><th>crew timeouts/ep</th>"
    "<th>invalid calls/ep</th><th>format errors/ep (loop)</th><th>mean tokens</th><th>episodes / errors</th>"
    "<th>flagged seeds</th></tr>"
)


def render(spec: ReportSpec, root: Path, summarize: SummarizeFn) -> str:
    inputs = [*spec.results, *spec.ablation, *(spec.training.values() if spec.training else ())]
    files = {rel: root / rel for rel in dict.fromkeys(inputs)}
    rows_by_file = {rel: _read_jsonl(path) for rel, path in files.items()}

    main_rows = [r for rel in spec.results for r in rows_by_file[rel]]
    groups = group_rows(main_rows)
    order = [g for g in groups if _label(g)[1] != "script"] + [g for g in groups if _label(g)[1] == "script"]
    summaries = {g: summarize(groups[g]) if any(r["status"] == "ok" for r in groups[g]) else None for g in order}

    def score(rows: list[dict]) -> float:
        return summarize(rows)["score_ros"]

    ablation_rows = [r for rel in spec.ablation for r in rows_by_file[rel]]
    gaps = ablation_gaps(ablation_rows, score_fn=score) if ablation_rows else {}
    chart2 = _chart_ablation(gaps) if gaps else None
    chart3 = None
    if spec.training:
        before, after = rows_by_file[spec.training["before"]], rows_by_file[spec.training["after"]]
        if before and after:
            chart3 = _chart_training(score(before), score(after))
    model_groups = [(g, summaries[g]) for g in order if _label(g)[1] != "script" and summaries[g] is not None]
    chart1 = _chart_models(model_groups) if model_groups else None

    all_rows = [r for rows in rows_by_file.values() for r in rows]
    versions = ", ".join(sorted({str(r["rules_version"]) for r in all_rows})) or "–"
    date = (
        spec.results_date
        or dt.datetime.fromtimestamp(max(p.stat().st_mtime for p in files.values()), tz=dt.UTC).date().isoformat()
    )
    hashes = "\n".join(
        f"<!-- results-sha256 {rel} {hashlib.sha256(path.read_bytes()).hexdigest()} -->" for rel, path in files.items()
    )
    fail_head = "".join(f"<th>{_E(c)}</th>" for c in (*COMPONENTS, "total"))
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_E(spec.name)} report</title>",
        f"<style>{_CSS}</style>",
        "</head><body>",
        hashes,
        f"<h1>Railway dispatcher evaluation: {_E(spec.name)}</h1>",
        f'<p class="meta">Rules version: {_E(versions)} · Dataset: {_E(spec.dataset or "–")} · '
        f"Results files dated: {_E(date)}</p>",
        f'<p class="legend">{_E(LEGEND)}</p>',
        "<h2>Results</h2>",
        '<p class="meta">score_ros = Σ(C_greedy − C_agent) / Σ(C_greedy − C_ref) over seeds (reference = 1, '
        "greedy = 0). Flagged seeds: unclipped score above 1.0 on some rep.</p>",
        f'<div class="scroll"><table id="results"><thead>{_RESULT_HEAD}</thead><tbody>',
        *(_results_row(g, summaries[g], groups[g]) for g in order),
        "</tbody></table></div>",
        _chart(1, "Chart 1: score by model", chart1, "no ok rows"),
        _chart(2, "Chart 2: score by factor (ablation)", chart2, ABLATION_PLACEHOLDER),
        _chart(3, "Chart 3: before vs after training", chart3, TRAINING_PLACEHOLDER),
        "<h2>Failure breakdown: mean cost by component per episode</h2>",
        '<div class="scroll"><table id="failures"><thead><tr><th>model</th><th>harness</th><th>mode</th>'
        f"{fail_head}</tr></thead><tbody>",
        *(_failure_row(g, groups[g]) for g in order),
        "</tbody></table></div>",
        "</body></html>",
        "",
    ]
    return "\n".join(parts)


def build_report(spec_path: Path, *, root: Path | None = None, summarize: SummarizeFn | None = None) -> str:
    """Render the report for a report manifest. `summarize` defaults to T08's stats.py."""
    return render(load_spec(Path(spec_path)), Path(root) if root else Path.cwd(), summarize or _default_summarize)


def embedded_hashes(text: str) -> dict[str, str]:
    return dict(re.findall(r"<!-- results-sha256 (\S+) ([0-9a-f]{64}) -->", text))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="railway-eval report", description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    doc = build_report(args.manifest, root=args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(doc.encode("utf-8"))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
