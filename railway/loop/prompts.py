"""Prompt building for our own loop (T08 §2.3; DESIGN.md §9.4).

`build_messages` is a pure function of (system, manual, turns, current, mode), so every model sees
an identical prompt for an identical episode. Messages are rebuilt every turn.
"""

from __future__ import annotations

from pathlib import Path

from railway.sim.constants import TOOL_BUDGET

SYSTEM_SUFFIX = "Reply with exactly one tool call per message. Do not write prose outside tool calls."
FORMAT_ERROR = "ERROR: reply with exactly one tool call."
WINDOW_TURNS = 5
MODES = ("windowed", "full")

# Where T07 keeps the instruction.md template (T07 §2.4). Owned by T07; read here, never copied.
INSTRUCTION_TEMPLATE = Path(__file__).resolve().parent.parent / "harbor" / "templates" / "instruction.md"


def load_instruction(path: Path = INSTRUCTION_TEMPLATE) -> str:
    """The T07 instruction.md text with the budget interpolated (the same text Harbor agents get)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: the instruction template comes from T07")
    return path.read_text(encoding="utf-8").replace("{TOOL_BUDGET}", str(TOOL_BUDGET)).rstrip("\n")


def system_message(instruction: str) -> str:
    return f"{instruction}\n\n{SYSTEM_SUFFIX}"


def render_current(status_compact: str, bulletins: str, notes: str, used: int, total: int, error: bool = False) -> str:
    body = (
        f"CURRENT STATE\n{status_compact}\nBULLETINS\n{bulletins}\nNOTES\n{notes}\n"
        f"Budget used: {used}/{total}. Call exactly one tool."
    )
    return f"{FORMAT_ERROR}\n{body}" if error else body


def turn_messages(index: int, name: str, arguments_raw: str, output: str) -> tuple[dict, dict]:
    """One turn = the assistant tool-call message + its tool-result message.

    Call ids are renumbered `call_<index>` so the prompt doesn't depend on the provider's ids.
    """
    call_id = f"call_{index}"
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments_raw}}],
    }
    return assistant, {"role": "tool", "tool_call_id": call_id, "content": output}


def build_messages(system: str, manual: str, turns: list[tuple[dict, dict]], current: str, mode: str) -> list[dict]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    shown = turns[-WINDOW_TURNS:] if mode == "windowed" else turns
    messages = [{"role": "system", "content": system}, {"role": "user", "content": f"MANUAL\n{manual}"}]
    for assistant, tool in shown:
        messages += [assistant, tool]
    messages.append({"role": "user", "content": current})
    return messages
