# Episode Transcripts & SFT/GRPO Training Data Format

Every episode played through `railway.loop.runner.play_episode` unconditionally writes a complete
episode bundle, whether or not `transcript_path` is supplied. Episode bundles are the primary
training data source for Lane E (warm-start SFT on Claude Fable 5.2 episodes that beat greedy, and
audit/inspection for GRPO).

---

## 1. Storage Layout

Both locally (`artifacts/transcripts/` by default, or `$RAILWAY_TRANSCRIPTS_DIR`) and on the Modal
volume **`railway-transcripts`** (mounted at `/transcripts`), bundles use the 5-level hierarchy:

```text
<root>/<run>/<model>/<seed-set>/<scenario_id>/<rep>/
    transcript.jsonl   # 1 header line + N turn lines + 1 footer line
    actions.json       # Session.episode_log() action & refusal audit log
    events.jsonl       # Engine EventLog (one Event JSON per line, when exposed)
    result.json        # EpisodeRecord summary row (same schema as results/<run>.jsonl)
```

Path segments are sanitized with `railway.loop.transcripts.safe_segment` so model IDs containing
slashes or colons are safe on both Windows and Linux/Modal.

### Modal Volume Helpers (`railway.loop.transcripts`)

- **Direct write from Modal functions:** mount `modal.Volume.from_name("railway-transcripts", create_if_missing=True)` at `/transcripts` (or set `RAILWAY_TRANSCRIPTS_DIR=/transcripts`), run `play_episode(...)`, and call `volume.commit()`.
- **Local write + upload:** set `RAILWAY_UPLOAD_TRANSCRIPTS=1` or call:
  ```python
  from railway.loop.transcripts import upload_episode_to_volume, upload_tree_to_volume

  remote_path = upload_episode_to_volume(rec.transcript_dir)
  # or upload an entire local tree:
  uploaded = upload_tree_to_volume("artifacts/transcripts")
  ```

---

## 2. `transcript.jsonl` Schema (`format_version = "1.0"`)

Each `transcript.jsonl` file contains `N + 2` UTF-8 JSON lines:

1. **Header (`type: "header"`)** — static prompt context & episode metadata (stored once):
   ```json
   {
     "type": "header",
     "format_version": "1.0",
     "run": "warmstart-train",
     "model": "fable-high",
     "model_name": "claude-fable-5-2",
     "provider": "anthropic",
     "settings": {
       "provider": "anthropic",
       "model": "claude-fable-5-2",
       "base_url": "https://api.anthropic.com/v1/messages",
       "thinking_mode": "adaptive",
       "display": "summarized",
       "effort": "high",
       "max_tokens": 32768,
       "mode": "windowed",
       "max_turns": 620,
       "disabled_tools": [],
       "diagnostics": false
     },
     "seed_set": "train",
     "preset": "T1",
     "seed": 5,
     "scenario_id": "T1-2f06dfb68904",
     "scenario_hash": "89d67021...",
     "rep": 0,
     "rules_version": "0.3",
     "manual_hash": "a1b2c3...",
     "manual": "# Dispatcher manual\n...",
     "system_prompt": "...\n\nReply with exactly one tool call per message. Do not write prose outside tool calls.",
     "tools": [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}],
     "started_at": "2026-11-14T21:30:00.123Z"
   }
   ```
   API keys and authorization headers are **never** stored anywhere in `settings` or transcripts.

2. **Turns (`type: "turn"`, lines `2 .. N+1`)** — one record per model call:
   To keep transcript files compact without losing a single byte of prompt context, `system_prompt`,
   `manual`, and `tools` are not repeated on every turn. Instead, each turn stores:
   - `prompt_ref`: `{"from_header": true, "history_turns": [1, 2, ...], "current": "CURRENT STATE\n..."}`, referencing the header plus the `delta_messages` of whichever prior turns were in the context window (`turns[-5:]` in `"windowed"` mode, all prior turns in `"full"` mode), plus this turn's `CURRENT` block. (If a custom caller ever sends messages that do not match `prompt_ref`, the runner falls back to storing inline `"messages"` so reconstruction is always byte-exact.)
   - `delta_messages`: `[assistant_tool_call_msg, tool_result_msg]` appended to history if a tool ran (`[]` on zero-call format-error turns).
   - `reasoning`: `str | null` — reasoning text or thought summary returned by the provider.
   - `content`: `str | null` — visible assistant text outside tool calls (`null` on clean turns).
   - `tool_calls`: `[{"id": "...", "name": "...", "arguments_raw": "..."}]` — raw calls emitted by the model.
   - `provider_blocks`: `list[dict] | null` — raw provider blocks needed to continue multi-turn conversations (e.g. Claude's signed `thinking` / `redacted_thinking` / `tool_use` blocks, or Gemini's `parts` with `thoughtSignature`).
   - `executed`: `{"tool": "...", "args": {...}, "arguments_raw": "...", "call_id": "call_1"} | null`.
   - `output`: `str | null` — output string from `Session.call`.
   - `tool_outputs`: `[{"call_id": "call_1", "tool": "...", "output": "..."}]`.
   - `usage`: `{"prompt_tokens", "completion_tokens", "input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens", "cache_write_tokens", "cost_usd"}`.
   - `cost_usd`: `float` — USD cost of this turn from `PRICE_TABLE`.
   - `tick`: `int | null`, `started_at`: ISO-8601 UTC, `ended_at`: ISO-8601 UTC, `latency_s`: `float`.
   - `format_error`: `bool`, `format_error_kind`: `null | "no_tool_call" | "multiple_tool_calls"`, `stop`: `null | "three_strikes"`.
   - `finish_reason`: `str | null` — the provider's raw stop reason (`"tool_calls"` / `"length"` for OpenAI-compatible servers, `"tool_use"` / `"max_tokens"` for Anthropic, `"STOP"` / `"MAX_TOKENS"` for Gemini). A `no_tool_call` turn with `"length"`, `"max_tokens"` or `"MAX_TOKENS"` ran out of output budget, usually all of it spent thinking, rather than declining to call a tool. Gemini returns no thought summary on such a turn, so `reasoning` is `null`.

3. **Footer (`type: "footer"`, last line)** — final outcome, KPIs, counts, and cumulative token/USD totals:
   ```json
   {
     "type": "footer",
     "status": "ok",
     "error": null,
     "ended_at": "2026-11-14T21:32:10.456Z",
     "wall_s": 130.333,
     "turns": 42,
     "tool_calls": 42,
     "format_errors": 0,
     "cost_total": 185.0,
     "reward": -185.0,
     "cost_by_component": {...},
     "c_greedy": 410.0,
     "c_ref": 150.0,
     "score_clipped": 0.8654,
     "score_unclipped": 0.8654,
     "kpis": {...},
     "counts": {
       "gridlocks": 0,
       "crew_timeouts": 0,
       "reversals": 0,
       "refusals": 1,
       "invalid_calls": 0,
       "decision_points": 14,
       "unfinished": 0
     },
     "usage_total": {
       "prompt_tokens": 295000,
       "completion_tokens": 18400,
       "input_tokens": 22000,
       "output_tokens": 18400,
       "reasoning_tokens": 0,
       "cached_tokens": 266300,
       "cache_write_tokens": 6700,
       "cost_usd": 0.9892
     }
   }
   ```
   Per Decision #2 (`docs/decisions.md`), `reward` is `-cost_total` (`-(C1 + C2 + C3 + C4 + C5)`).

---

## 3. Companion Files

- **`actions.json`**: the session's `episode_log()` (every `Session.call` with `seq`, `tick`, `tool`, `args`, `status`, `rule`, plus `final_tick`, `final_cost`, and `final_cost_by_component`).
- **`events.jsonl`**: written whenever the session or its engine exposes a `railway.net.events.EventLog` (e.g. `session.engine.log` from Lane B / `session.event_log` from Lane D). Each line is one serialized `Event` (`tick`, `kind`, `train`, `element`, `node`, `call`, `booked`, `detail`).
- **`result.json`**: `EpisodeRecord.to_row()` formatted as JSON.

---

## 4. Provider Reasoning & Continuation Details (`railway.loop.providers`)

| Benchmark Model | Client | How Reasoning Is Requested & Returned | Multi-Turn Continuation |
|---|---|---|---|
| **Claude Fable 5.2** (`claude-fable-5-2`, high) | `AnthropicClient` (`/v1/messages`) | `thinking={"type": "adaptive", "display": "summarized"}`, `output_config={"effort": "high"}`, `tool_choice={"type": "auto", "disable_parallel_tool_use": True}`. Summarized reasoning arrives in `content[i].thinking` (`type == "thinking"`) and exact thinking token count in `usage.output_tokens_details.thinking_tokens` (billed inside `output_tokens`). Ephemeral cache breakpoint on `MANUAL` caches `tools + system + MANUAL` at 97.5% discount. | Complete `content` array (`thinking` with `signature`, `redacted_thinking` with `data`, `tool_use`) is stored in `provider_blocks` and passed back unchanged. `thinking-binding-controls-2026-08-01` (`drop_block`) prevents 400 prefix-check failures when windowed `CURRENT` updates omit prior `CURRENT` blocks. |
| **Qwen 3.8 27B** (`qwen38`, thinking on) | `OpenAICompatClient` (`ModelClient`) | Served by vLLM (`infra/modal/serve_qwen.py`) with `--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder`. Full `<think>` trace arrives in `choices[0].message.reasoning` (with fallback to `reasoning_content` or `<think>` tags). | Standard OpenAI `assistant` tool-call messages (`provider_blocks` stripped). |
| **Gemini 4 Carbon** (`gemini-4-carbon`, high) | `GeminiClient` (`:generateContent`) | `generationConfig.thinkingConfig={"thinkingLevel": "high", "includeThoughts": True}`. Thought summaries arrive in `candidates[0].content.parts` with `"thought": True`; exact thinking token count arrives in `usageMetadata.thoughtsTokenCount` and implicit cache hits in `cachedContentTokenCount`. | Raw `candidates[0].content.parts` (including `thoughtSignature` on `functionCall` parts) are stored in `provider_blocks` and passed back unchanged. |
| **Nemotron 3.5 Super VL** (`nemotron-3.5-super-vl`) | `OpenAICompatClient` (`ModelClient`) | Served on Verda's vLLM endpoint; reasoning trace arrives in `choices[0].message.reasoning` / `reasoning_content`. | Standard OpenAI `assistant` tool-call messages. |

---

## 5. Loading Training Samples for SFT & GRPO (`railway.loop.transcripts`)

Lane E's SFT trainer loads `(prompt, reasoning, completion)` samples directly from a single transcript
file, an episode bundle directory, or a root directory (such as `/transcripts/warmstart/fable-high/train`):

```python
from railway.loop.transcripts import load_training_samples, load_transcript

# Load all clean turns from Fable episodes on training seeds that beat greedy (Decision #12):
samples = load_training_samples(
    "/transcripts/warmstart/fable-high/train",
    skip_format_errors=True,   # exclude 0-call or multi-call format errors
    require_reasoning=False,   # set True to drop turns where the provider returned no reasoning text
    only_ok_episodes=True,     # exclude infra-error episodes
    beat_greedy_only=True,     # keep only episodes with cost_total < c_greedy
)

for sample in samples:
    # Tuple unpacking works directly:
    prompt, reasoning, completion = sample
    # prompt == {"messages": [...OpenAI messages sent on this turn...], "tools": [...tool schemas...]}
    # reasoning == "..." (or None)
    # completion == {"role": "assistant", "content": None, "tool_calls": [{"id": "call_k", "type": "function", "function": {"name": "...", "arguments": "..."}}]}

    # Ready for Qwen 3.8's HuggingFace chat template (`tokenizer.apply_chat_template(..., tools=sample.tools)`):
    chat_messages = sample.to_chat_messages(include_reasoning=True, reasoning_key="reasoning_content")
```
