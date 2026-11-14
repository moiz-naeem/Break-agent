"""T03-36, the core fairness property: an unrevealed event cannot be observed before its reveal tick."""

from hypothesis import given, settings
from hypothesis import strategies as st

from railway.util import canonical_json
from tests.t03.common import ALL_FACTORS, F22, L6, dump, load, run_step, steps, vf


@st.composite
def hidden_event(draw):
    kind = draw(st.sampled_from(["possession", "speed_restriction", "late_entry", "breakdown"]))
    if kind == "late_entry":
        return {
            "id": "EV1",
            "kind": kind,
            "reveal_tick": draw(st.integers(1, 29)),
            "train": "F22",
            "new_entry_tick": draw(st.integers(31, 70)),
        }
    r = draw(st.integers(1, 40))
    if kind == "breakdown":
        return {
            "id": "EV1",
            "kind": kind,
            "reveal_tick": r,
            "train": draw(st.sampled_from(["E1", "F21", "L5"])),
            "tick": r,
            "duration": draw(st.integers(1, 20)),
        }
    start = draw(st.integers(r, r + 20))
    end = draw(st.integers(start + 1, 90))
    if kind == "possession":
        return {
            "id": "EV1",
            "kind": kind,
            "reveal_tick": r,
            "section": draw(st.sampled_from(["YW-S1", "S1-S2", "S2-S3", "S3-YE"])),
            "start": start,
            "end": end,
        }
    return {
        "id": "EV1",
        "kind": kind,
        "reveal_tick": r,
        "sections": ["S1-S2"],
        "speed_kmh": 40,
        "start": start,
        "end": end,
    }


def pair(event):
    common = {"extra": (L6, F22), "shift": 90, "L5": {"next_service": "L6"}}
    return load(vf(*ALL_FACTORS, events=[event], **common)), load(vf(*ALL_FACTORS, **common))


def same(a, b):
    assert canonical_json(dump(a)) == canonical_json(dump(b))


@settings(max_examples=150)
@given(hidden_event(), steps)
def test_T03_36_unrevealed_event_invisible(event, plan):
    x, y = pair(event)
    reveal = event["reveal_tick"]
    same(x.state(), y.state())
    for acts, jump in plan:
        if x.tick >= reveal or x.done:
            break
        x_results, x_adv = run_step(x, acts, jump)
        y_results, y_adv = run_step(y, acts, jump)
        for a, b in zip(x_results, y_results, strict=True):
            same(a, b)
        if min(x_adv.tick, y_adv.tick) < reveal:
            same(x_adv, y_adv)
            same(x.state(), y.state())
