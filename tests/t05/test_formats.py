import json
import re

from railway.gen.generator import generate
from railway.gen.presets import PRESETS
from railway.gen.seeds import candidate_seeds
from railway.sim.constants import TOOL_BUDGET
from railway.sim.observe import State, StaticInfo
from railway.tools.session import Session
from tests.t03.common import F22
from tests.t05.common import EPISODE_A, play, session

TRAIN_LINE = re.compile(
    r"^(?P<id>\S+) +(?P<type>\S+) +"
    r"(?P<pos>at \S+|\S+ (eastbound|westbound)|waiting \S+ (entry@\d+|after \S+))"
    r"( auth:(?P<auth>\S+) delay:(\+\d+|0)( crew:\d+)?( next:\S+@\d+)?)?$"
)


def test_T05_08_compact_status_grammar():
    s = session()
    play(s, EPISODE_A[:4])  # t=6
    lines = s.call("status", {"compact": True}).splitlines()
    assert re.fullmatch(r't=6 \| gridlock: none \| notes: ""', lines[0])
    assert lines[-1] == "restrictions: none"
    by_id = {}
    for line in lines[1:-1]:
        m = TRAIN_LINE.match(line)
        assert m, line
        by_id[m["id"]] = line
    assert set(by_id) == {"E1", "F21", "L5"}
    assert "at S2/L1-platform" in by_id["E1"] and "auth:-" in by_id["E1"]
    assert "at S3/main " in by_id["L5"] and "auth:S2/main" in by_id["L5"]
    assert "at YW " in by_id["F21"] and "auth:-" in by_id["F21"]
    assert by_id["E1"].startswith("E1   express   ")  # id padded to 4, type to 9


def test_T05_09_full_status_roundtrip():
    s = session()
    play(s, EPISODE_A[:4])
    s.call("notes", {"text": "pass L5 at S2"})
    expected = s.sim.state().model_copy(update={"notes": "pass L5 at S2"})
    assert State.model_validate_json(s.call("status", {})) == expected


def test_T05_10_static_from_tool_json():
    scenarios = [session().scenario] + [generate(PRESETS["T2"], seed) for seed in candidate_seeds("dev", count=5)]
    for scenario in scenarios:
        s = Session(scenario)
        rebuilt = StaticInfo.from_tool_json(s.call("network", {}), s.call("timetable", {}))
        assert rebuilt == s.sim.static_info(), scenario.scenario_id


def test_T05_11_advance_changes_only():
    s = session()
    outs = play(s, EPISODE_A[:-1])
    advances = [o for (tool, *_), o in zip(EPISODE_A[:-1], outs, strict=True) if tool == "advance_to_next_event"]
    for out in advances:
        since = int(re.search(r"^changes since t=(\d+):$", out, re.M)[1])
        ticks = [int(t) for t in re.findall(r"^  t=(\d+) ", out, re.M)]
        assert all(t >= since for t in ticks)
        assert re.search(rf"budget: \d+/{TOOL_BUDGET}$", out)
    assert re.search(r"^  t=\d+ E1 at S2/L1$", advances[1], re.M)  # E1's arrival is reported once
    assert "E1 at S2/L1" not in advances[2]


def test_T05_12_kpis_shape():
    s = session()
    play(s, EPISODE_A[:4])
    k = json.loads(s.call("kpis", {}))
    assert set(k) == {"tick", "cost", "counts", "budget"}
    assert set(k["cost"]) == {"pax_lateness", "freight_lateness", "crew_timeout", "unfinished", "total"}
    assert set(k["counts"]) == {
        "gridlocks", "crew_timeouts", "refusals", "invalid_calls", "decision_points"
    }  # fmt: skip
    assert set(k["budget"]) == {"used", "left", "total"}
    assert k["cost"]["total"] == sum(v for key, v in k["cost"].items() if key != "total")
    assert k["budget"] == {"used": 5, "left": TOOL_BUDGET - 5, "total": TOOL_BUDGET}


def test_T05_13_bulletins():
    late = {"id": "EV1", "kind": "late_entry", "reveal_tick": 10, "train": "F22", "new_entry_tick": 45}
    s = session("F16", extra=(F22,), shift=90, events=[late])
    s.call("advance_to_next_event", {})  # t=2
    assert json.loads(s.call("bulletins", {})) == []
    s.call("advance_to_next_event", {})  # t=10
    assert json.loads(s.call("bulletins", {})) == [late]

    skipped = {"id": "EV1", "kind": "breakdown", "reveal_tick": 1, "train": "F21", "tick": 1, "duration": 5}
    s = session("F15", events=[skipped])
    while not s.closed:
        assert json.loads(s.call("bulletins", {})) == []
        s.call("advance_to_next_event", {})


def _grid(out: str, scenario) -> tuple[list[str], dict[int, dict[str, str]], list[str]]:
    """Parse stringline output into column names, {tick: {column: cell}} and the raw row lines."""
    places = ["YW", *[st.id for st in scenario.network.stations], "YE"]
    columns = [x for a, b in zip(places, places[1:], strict=False) for x in (a, f"{a}-{b}")] + ["YE"]
    lines = out.splitlines()
    rows = [line for line in lines if re.match(r"^[ >]t\d+", line)]
    grid = {}
    for row in rows:
        tick = int(re.match(r"^[ >]t(\d+)", row)[1])
        grid[tick] = {col: row[6 + 4 * i : 10 + 4 * i].strip() for i, col in enumerate(columns)}
    return columns, grid, rows


def test_T05_14_stringline_limits():
    s = session()
    for window in ((0, 61), (10, 5), (-1, 3), (0, 10_000)):
        out = s.call("stringline", {"from_tick": window[0], "to_tick": window[1]})
        assert out.startswith("REFUSED R-WINDOW"), (window, out)
        assert s.log[-1]["status"] == "REFUSED"
    _, grid, rows = _grid(s.call("stringline", {"from_tick": 0, "to_tick": 10}), s.scenario)
    assert sorted(grid) == list(range(11)) and len(rows) == 11  # step 1
    assert grid[0]["YW"] == "E1" and grid[0]["YE"] == "L5"  # in the network from load, before any advance
    t2 = Session(generate(PRESETS["T2"], next(iter(candidate_seeds("dev", count=1)))))
    out = t2.call("stringline", {"from_tick": 0, "to_tick": 60})
    assert t2.log[-1]["status"] == "INFO"
    assert len(out) <= 2000
    _, grid, _ = _grid(out, t2.scenario)
    ticks = sorted(grid)
    assert ticks[0] == 0 and len({b - a for a, b in zip(ticks, ticks[1:], strict=False)}) == 1  # one step


def test_T05_15_stringline_content():
    s = session()
    play(s, EPISODE_A[:8])  # t=11, E1 just granted to YE
    assert s.sim.tick == 11
    out = s.call("stringline", {"from_tick": 0, "to_tick": 20})
    columns, grid, rows = _grid(out, s.scenario)
    assert out.splitlines()[0].split() == [c if "-" not in c else "-" for c in columns]
    assert sorted(grid) == list(range(21))
    assert grid[3]["S1-S2"] == "E1"  # history: E1 enters S1-S2 at 3
    assert grid[6]["S2"] == "E1"  # history: arrives S2/L1 at 6
    assert grid[11]["S2"] == "E1+"  # present row from history; L5 is also at S2 (main), lowest id + "+"
    assert grid[1]["YW"] == "." and grid[2]["YW"] == grid[10]["YW"] == "F21"  # F21 enters at its entry tick 2
    assert grid[12]["S2-S3"] == "E1" and grid[15]["S3-YE"] == "E1"  # projection of the current authority
    assert grid[17]["YE"] == "E1" and "E1" not in grid[18].values()  # arrives at 17, then leaves the network
    assert [r for r in rows if r.startswith(">")] == [r for r in rows if r.startswith(">t11 ")]
    cells = [cell for row in grid.values() for cell in row.values()]
    assert all(cell in {"", "."} or re.fullmatch(r"[A-Z]\d+\+?", cell) for cell in cells)
