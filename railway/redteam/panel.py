"""Red-team panel: run every local check and write one scoreboard.

    python -m railway.redteam.panel --preset T1 --seeds 5

Runs the strategic battery (`strategic.py`) and the verifier tamper suite (`tamper.py`) in this process,
collects their reports, and writes `results/redteam/scoreboard.md`. The Harbor-side runs (oracle,
nop, stop-early, agent trials, isolation probe) need Docker and the Harbor CLI; `harbor_protocol.py`
runs those, and this scoreboard links to its output when it is present.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from railway.gen.presets import PRESETS
from railway.redteam import strategic, tamper

ROOT = Path(__file__).resolve().parents[2]


def section(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}\n"


def _tamper_row(path: Path, code: int) -> str:
    """A win the agent cannot reach is not a pass: say so, so the row is never read as 'nothing found'."""
    if code:
        return "FAIL: an agent-reachable attack won"
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "pass"
    unreachable = [r["case"] for r in rows if r["verdict"] == "WIN"]
    if unreachable:
        return f"pass, with {len(unreachable)} finding(s) no agent can reach: {', '.join(unreachable)}"
    return "pass"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--preset", default="T1", choices=sorted(PRESETS))
    p.add_argument("--set", default="dev", dest="set_name")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", default=str(ROOT / "results" / "redteam"))
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print("== strategic battery ==", flush=True)
    strat_code = strategic.main(
        ["--preset", a.preset, "--set", a.set_name, "--seeds", str(a.seeds), "--out", str(out)]
    )
    print("\n== verifier tamper suite ==", flush=True)
    tamper_code = tamper.main(["--preset", a.preset, "--out", str(out)])

    strat_md = (out / f"strategic-{a.preset}-{a.set_name}.md").read_text(encoding="utf-8")
    tamper_md = (out / "tamper.md").read_text(encoding="utf-8")
    harbor_path = out / "harbor-protocol.md"

    strat_row = "FAIL" if strat_code else "pass"
    tamper_row = _tamper_row(out / "tamper.json", tamper_code)
    harbor_row = "see below" if harbor_path.exists() else "not run (needs Docker + Harbor)"
    lines = [
        f"# Red-team scoreboard ({a.preset}, {a.seeds} seeds)",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Strategic battery (non-dispatching policies, stop-early curve) | {strat_row} |",
        f"| Verifier tamper suite (reward files, forged logs, patched tests) | {tamper_row} |",
        f"| Harbor judge protocol | {harbor_row} |",
        "",
        "The LLM attacker briefs in `railway/redteam/prompts/` are run by hand against a live task; see",
        "`docs/redteam-playbook.md` for the commands and for how to record their findings here.",
        "",
    ]
    parts = [
        section("Strategic battery", strat_md.split("\n", 1)[1]),
        section("Verifier tamper suite", tamper_md.split("\n", 1)[1]),
    ]
    if harbor_path.exists():
        parts.append(section("Harbor judge protocol", harbor_path.read_text(encoding="utf-8").split("\n", 1)[1]))
    body = "\n".join(lines) + "\n".join(parts)
    (out / "scoreboard.md").write_text(body, encoding="utf-8")
    print(f"\nscoreboard: {out / 'scoreboard.md'}")
    return 1 if strat_code else 0


if __name__ == "__main__":
    sys.exit(main())
