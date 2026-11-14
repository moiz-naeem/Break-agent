"""Episode transcript bundles, SFT/GRPO training-sample reader, and Modal volume storage (Lane F).

Every episode writes a self-contained directory:

    <root>/<run>/<model>/<seed-set>/<scenario_id>/<rep>/
        transcript.jsonl   # 1 header line + N turn lines + 1 footer line
        actions.json       # Session.episode_log() (action/tool audit log)
        events.jsonl       # Engine EventLog (when the session/engine exposes one)
        result.json        # EpisodeRecord summary row

`load_training_samples(path)` reconstructs exact `(prompt, reasoning, completion)` samples per turn
from either a single `transcript.jsonl`, an episode directory, or a tree of episodes (such as a
mounted `railway-transcripts` Modal volume at `/transcripts`).
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORMAT_VERSION = "1.0"
TRANSCRIPTS_VOLUME_NAME = "railway-transcripts"
TRANSCRIPTS_MOUNT_PATH = "/transcripts"

_UNSAFE_SEGMENT_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def utc_iso_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def safe_segment(value: str | int | None, fallback: str = "unspecified") -> str:
    """Sanitize a single path segment for both Windows and POSIX / Modal volumes."""
    if value is None:
        return fallback
    cleaned = _UNSAFE_SEGMENT_RE.sub("_", str(value).strip()).strip(". ")
    return cleaned or fallback


def default_transcript_root() -> Path:
    """Resolve where episode bundles land when no explicit `transcript_root` is passed."""
    env_dir = os.environ.get("RAILWAY_TRANSCRIPTS_DIR")
    if env_dir:
        return Path(env_dir)
    if sys.platform != "win32" and Path(TRANSCRIPTS_MOUNT_PATH).is_dir():
        return Path(TRANSCRIPTS_MOUNT_PATH)
    return Path("artifacts") / "transcripts"


def episode_bundle_dir(
    root: str | Path | None = None,
    *,
    run: str | None = None,
    model: str | None = None,
    seed_set: str | None = None,
    scenario_id: str | None = None,
    rep: int = 0,
) -> Path:
    """Canonical `<root>/<run>/<model>/<seed-set>/<scenario_id>/<rep>` directory."""
    base = Path(root) if root is not None else default_transcript_root()
    return (
        base
        / safe_segment(run, "adhoc")
        / safe_segment(model, "unknown")
        / safe_segment(seed_set, "unspecified")
        / safe_segment(scenario_id, "unknown")
        / safe_segment(rep, "0")
    )


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, str | int | float | bool):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_jsonable(x) for x in obj]
    if hasattr(obj, "__dict__"):
        return {str(k): _to_jsonable(v) for k, v in vars(obj).items()}
    return str(obj)


def extract_action_log(session: Any, rec: Any, final: Any) -> dict:
    """Extract the session's action log (`episode_log()` on real sessions, fallback for fakes)."""
    if hasattr(session, "episode_log") and callable(session.episode_log):
        try:
            log = session.episode_log()
            if isinstance(log, dict):
                return _to_jsonable(log)
        except Exception:  # noqa: BLE001
            pass
    entries = getattr(session, "log", [])
    final_tick = getattr(final, "tick", None) if final is not None else None
    if isinstance(final, dict):
        final_tick = final.get("tick", final_tick)
    return {
        "scenario_id": getattr(rec, "scenario_id", None),
        "scenario_hash": getattr(rec, "scenario_hash", None),
        "rules_version": getattr(rec, "rules_version", None),
        "entries": _to_jsonable(list(entries)),
        "closed": bool(getattr(session, "closed", True)),
        "final_tick": final_tick,
        "final_cost": getattr(rec, "cost_total", None),
        "final_cost_by_component": _to_jsonable(getattr(rec, "cost_by_component", None)),
    }


def extract_event_log(session: Any) -> list[dict] | None:
    """Extract the v0.3 engine `EventLog` (`railway/net/events.py`) if the session exposes one.

    Checks `session.event_log`, `session.events`, `session.engine.log` (Lane B's canonical attribute),
    `session.engine.event_log`, `session.sim.event_log`, and `session.sim.log`.
    Returns `None` if the session/engine does not expose an event log (e.g. v0.2 single-track `Simulator`),
    or a `list[dict]` (possibly empty) when an event log is exposed.
    """
    candidates: list[Any] = [
        getattr(session, "event_log", None),
        getattr(session, "events", None),
    ]
    for attr in ("engine", "sim"):
        eng = getattr(session, attr, None)
        if eng is not None:
            candidates.extend(
                [
                    getattr(eng, "log", None),
                    getattr(eng, "event_log", None),
                    getattr(eng, "events", None),
                ]
            )

    for cand in candidates:
        if cand is None:
            continue
        if callable(cand):
            try:
                cand = cand()
            except TypeError:
                continue
        if hasattr(cand, "events") and isinstance(cand.events, list | tuple):
            return [_to_jsonable(e) for e in cand.events]
        if isinstance(cand, list | tuple):
            return [_to_jsonable(e) for e in cand]
    return None


def reconstruct_turn_messages(header: dict, turns_by_num: dict[int, dict], turn_entry: dict) -> list[dict]:
    """Reconstruct the exact OpenAI-shaped `messages` list sent to the model on `turn_entry`."""
    if "messages" in turn_entry and isinstance(turn_entry["messages"], list):
        return copy.deepcopy(turn_entry["messages"])
    ref = turn_entry.get("prompt_ref")
    if not isinstance(ref, dict):
        raise ValueError(f"turn {turn_entry.get('turn')} has neither `messages` nor `prompt_ref`")

    messages: list[dict] = []
    if ref.get("from_header", True):
        messages.append({"role": "system", "content": header["system_prompt"]})
        messages.append({"role": "user", "content": f"MANUAL\n{header['manual']}"})

    for prev_num in ref.get("history_turns") or []:
        prev = turns_by_num[int(prev_num)]
        for delta_msg in prev.get("delta_messages") or []:
            messages.append(copy.deepcopy(delta_msg))

    if "current" in ref and ref["current"] is not None:
        messages.append({"role": "user", "content": ref["current"]})
    for extra_msg in ref.get("extra_messages") or []:
        messages.append(copy.deepcopy(extra_msg))
    return messages


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def write_episode_bundle(
    bundle_dir: Path,
    *,
    header: dict,
    turns: list[dict],
    footer: dict,
    action_log: dict,
    event_log: list[dict] | None,
    result_row: dict,
    extra_transcript_path: Path | None = None,
) -> Path:
    """Write `{transcript.jsonl, actions.json, [events.jsonl], result.json}` into `bundle_dir`."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    records = [header, *turns, footer]
    transcript_file = bundle_dir / "transcript.jsonl"
    write_jsonl(transcript_file, records)

    (bundle_dir / "actions.json").write_text(
        json.dumps(action_log, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if event_log is not None:
        write_jsonl(bundle_dir / "events.jsonl", event_log)

    (bundle_dir / "result.json").write_text(
        json.dumps(result_row, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    if extra_transcript_path is not None:
        extra = Path(extra_transcript_path)
        if extra.resolve() != transcript_file.resolve():
            write_jsonl(extra, records)

    return bundle_dir


# --- Transcript Reader & Training Sample Loader (for Lane E SFT / GRPO) ----------------------------


@dataclass
class LoadedTranscript:
    header: dict
    turns: list[dict]
    footer: dict | None
    path: Path | None = None

    @property
    def turns_by_num(self) -> dict[int, dict]:
        return {int(t["turn"]): t for t in self.turns}

    def reconstruct_messages(self, turn: int | dict) -> list[dict]:
        entry = self.turns_by_num[turn] if isinstance(turn, int) else turn
        return reconstruct_turn_messages(self.header, self.turns_by_num, entry)


@dataclass(frozen=True)
class TrainingSample:
    """One SFT / RL training sample `(prompt, reasoning, completion)` reconstructed from a turn.

    Supports both tuple unpacking (`prompt, reasoning, completion = sample`) and attribute access
    (`sample.prompt`, `sample.messages`, `sample.tools`, `sample.reasoning`, `sample.completion`, `sample.meta`).
    """

    prompt: dict[str, Any]
    reasoning: str | None
    completion: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Any]:
        yield self.prompt
        yield self.reasoning
        yield self.completion

    @property
    def messages(self) -> list[dict]:
        return self.prompt["messages"]

    @property
    def tools(self) -> list[dict]:
        return self.prompt["tools"]

    def to_chat_messages(
        self,
        *,
        include_reasoning: bool = True,
        reasoning_key: str = "reasoning_content",
    ) -> list[dict]:
        """Return `messages + [assistant_completion]` ready for `tokenizer.apply_chat_template`."""
        assistant = copy.deepcopy(self.completion)
        if include_reasoning and self.reasoning is not None:
            assistant[reasoning_key] = self.reasoning
        return [*copy.deepcopy(self.messages), assistant]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": copy.deepcopy(self.prompt),
            "reasoning": self.reasoning,
            "completion": copy.deepcopy(self.completion),
            "meta": copy.deepcopy(self.meta),
        }


def load_transcript(path: str | Path) -> LoadedTranscript:
    """Load `transcript.jsonl` from a file path or an episode bundle directory."""
    p = Path(path)
    if p.is_dir():
        p = p / "transcript.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"transcript not found: {p}")

    header: dict | None = None
    turns: list[dict] = []
    footer: dict | None = None

    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        rec = json.loads(line)
        rtype = rec.get("type")
        if rtype == "header":
            header = rec
        elif rtype == "turn" or ("turn" in rec and rtype is None):
            turns.append(rec)
        elif rtype == "footer":
            footer = rec
        else:
            raise ValueError(f"{p}:{line_no}: unknown transcript record type {rtype!r}")

    if header is None:
        raise ValueError(f"{p}: missing transcript header record")
    return LoadedTranscript(header=header, turns=turns, footer=footer, path=p)


def iter_transcript_files(path: str | Path) -> Iterator[Path]:
    """Yield one or more `transcript.jsonl` files from a file, episode dir, or root directory."""
    p = Path(path)
    if p.is_file():
        yield p
        return
    direct = p / "transcript.jsonl"
    if direct.is_file():
        yield direct
        return
    if p.is_dir():
        yield from sorted(p.rglob("transcript.jsonl"))
        return
    raise FileNotFoundError(f"no transcript path at {p}")


def _turn_completion(turn_entry: dict) -> dict[str, Any]:
    executed = turn_entry.get("executed")
    if isinstance(executed, dict) and executed.get("tool"):
        call_id = str(executed.get("call_id") or f"call_{turn_entry['turn']}")
        args_raw = executed.get("arguments_raw")
        if args_raw is None:
            args_raw = json.dumps(executed.get("args") or {}, sort_keys=True)
        return {
            "role": "assistant",
            "content": turn_entry.get("content"),
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": str(executed["tool"]), "arguments": str(args_raw)},
                }
            ],
        }
    tcs = turn_entry.get("tool_calls") or []
    msg: dict[str, Any] = {"role": "assistant", "content": turn_entry.get("content")}
    if tcs:
        msg["tool_calls"] = [
            {
                "id": str(tc.get("id") or f"call_{i}"),
                "type": "function",
                "function": {"name": str(tc.get("name") or ""), "arguments": str(tc.get("arguments_raw") or "")},
            }
            for i, tc in enumerate(tcs)
        ]
    return msg


def load_training_samples(
    path: str | Path,
    *,
    skip_format_errors: bool = True,
    require_reasoning: bool = False,
    only_ok_episodes: bool = True,
    beat_greedy_only: bool = False,
) -> list[TrainingSample]:
    """Load `(prompt, reasoning, completion)` training samples from one transcript or a directory tree.

    - `skip_format_errors`: exclude turns where the model emitted 0 or >1 tool calls (default True).
    - `require_reasoning`: exclude turns without non-empty reasoning text (default False).
    - `only_ok_episodes`: exclude transcripts whose footer status is not `"ok"` (default True).
    - `beat_greedy_only`: keep only episodes with `cost_total < c_greedy` (warm-start SFT filter, decision #12).
    """
    samples: list[TrainingSample] = []
    for tfile in iter_transcript_files(path):
        loaded = load_transcript(tfile)
        hdr = loaded.header
        ftr = loaded.footer or {}

        if only_ok_episodes and ftr.get("status", "ok") != "ok":
            continue
        if beat_greedy_only:
            c_tot = ftr.get("cost_total")
            c_grd = ftr.get("c_greedy")
            if c_tot is None or c_grd is None or float(c_tot) >= float(c_grd):
                continue

        tools = copy.deepcopy(hdr.get("tools") or [])
        turns_by_num = loaded.turns_by_num

        for turn_entry in loaded.turns:
            if skip_format_errors and turn_entry.get("format_error", False):
                continue
            reasoning = turn_entry.get("reasoning")
            if require_reasoning and not (isinstance(reasoning, str) and reasoning.strip()):
                continue

            messages = reconstruct_turn_messages(hdr, turns_by_num, turn_entry)
            completion = _turn_completion(turn_entry)
            meta = {
                "run": hdr.get("run"),
                "model": hdr.get("model"),
                "model_name": hdr.get("model_name"),
                "provider": hdr.get("provider"),
                "seed_set": hdr.get("seed_set"),
                "preset": hdr.get("preset"),
                "seed": hdr.get("seed"),
                "scenario_id": hdr.get("scenario_id"),
                "scenario_hash": hdr.get("scenario_hash"),
                "rep": hdr.get("rep"),
                "rules_version": hdr.get("rules_version"),
                "manual_hash": hdr.get("manual_hash"),
                "turn": turn_entry.get("turn"),
                "tick": turn_entry.get("tick"),
                "format_error": bool(turn_entry.get("format_error", False)),
                "turn_cost_usd": turn_entry.get("cost_usd", 0.0),
                "cost_total": ftr.get("cost_total"),
                "reward": ftr.get("reward"),
                "c_greedy": ftr.get("c_greedy"),
                "c_ref": ftr.get("c_ref"),
                "transcript_path": str(tfile),
            }
            samples.append(
                TrainingSample(
                    prompt={"messages": messages, "tools": copy.deepcopy(tools)},
                    reasoning=reasoning,
                    completion=completion,
                    meta=meta,
                )
            )
    return samples


# --- Modal Volume Helpers (`railway-transcripts`) --------------------------------------------------


def get_transcripts_volume(volume_name: str = TRANSCRIPTS_VOLUME_NAME):
    """Look up or create the Modal volume holding episode transcripts."""
    import modal

    return modal.Volume.from_name(volume_name, create_if_missing=True)


def upload_episode_to_volume(
    episode_dir: str | Path,
    *,
    root: str | Path | None = None,
    remote_rel: str | None = None,
    volume_name: str = TRANSCRIPTS_VOLUME_NAME,
) -> str:
    """Upload a local episode bundle directory to the `railway-transcripts` Modal volume.

    Returns the remote path (e.g. `/<run>/<model>/<seed-set>/<scenario_id>/<rep>`).
    """
    ep = Path(episode_dir).resolve()
    if not (ep / "transcript.jsonl").is_file():
        raise FileNotFoundError(f"{ep} is missing transcript.jsonl")

    if remote_rel is None:
        if root is not None:
            rel = ep.relative_to(Path(root).resolve()).as_posix()
        else:
            # Take the last 5 path segments: <run>/<model>/<seed-set>/<scenario_id>/<rep>
            if len(ep.parts) < 5:
                raise ValueError(f"cannot infer 5-level remote path from {ep}; pass `root` or `remote_rel`")
            rel = "/".join(ep.parts[-5:])
    else:
        rel = remote_rel.strip("/")

    remote_dir = f"/{rel}"
    vol = get_transcripts_volume(volume_name)
    with vol.batch_upload(force=True) as batch:
        batch.put_directory(str(ep), remote_dir)
    return remote_dir


def upload_tree_to_volume(
    local_root: str | Path,
    *,
    volume_name: str = TRANSCRIPTS_VOLUME_NAME,
) -> list[str]:
    """Upload every episode bundle under `local_root` to the `railway-transcripts` Modal volume."""
    root = Path(local_root).resolve()
    ep_dirs = sorted({p.parent for p in root.rglob("transcript.jsonl")})
    if not ep_dirs:
        return []
    vol = get_transcripts_volume(volume_name)
    uploaded: list[str] = []
    with vol.batch_upload(force=True) as batch:
        for ep in ep_dirs:
            rel = ep.relative_to(root).as_posix().strip("/")
            remote_dir = f"/{rel}"
            batch.put_directory(str(ep), remote_dir)
            uploaded.append(remote_dir)
    return uploaded
