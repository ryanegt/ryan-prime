#!/usr/bin/env python3
"""
Build a JSONL training set from the corpus.

Emits one record per entry with:
  - messages: [{role:user, content:user_prompt}, {role:assistant, content:content_or_text}]
  - metadata: useful provenance (source/date/id/title/topics)

Skips entries missing user_prompt or missing assistant text.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


JsonDict = Dict[str, Any]


def load_json(path: Path) -> Optional[Union[JsonDict, List[Any]]]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def is_entry_dict(d: Any) -> bool:
    return isinstance(d, dict) and any(k in d for k in ("source", "date", "content", "text", "context", "summary", "title"))


def as_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def extract_entries(data: Any) -> List[JsonDict]:
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return [e for e in data["items"] if isinstance(e, dict)]
    if is_entry_dict(data):
        return [data]  # type: ignore[list-item]
    if isinstance(data, list) and all(isinstance(e, dict) for e in data):
        return [e for e in data if isinstance(e, dict)]
    return []


def assistant_text(entry: JsonDict) -> Optional[str]:
    for k in ("text", "content"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def sanitize_user_prompt(prompt: str) -> str:
    """
    Remove explicit 'Ryan's voice' instruction from prompts.
    We want identity/voice to emerge implicitly from corpus rather than being
    repeatedly commanded in prompts.
    """
    s = prompt.strip()

    # Normalize curly apostrophes for matching.
    s = s.replace("Ryan’s", "Ryan's")

    # Common phrasings to strip.
    replacements = [
        " in Ryan's voice",
        " in Ryan's voice.",
        " in Ryan's voice:",
        " with Ryan's voice",
        " with Ryan's voice.",
        " with Ryan's voice:",
        " Ryan's voice",
        " Ryan's voice.",
        " Ryan's voice:",
    ]
    for r in replacements:
        s = s.replace(r, "")

    # Clean up leftover double spaces and awkward punctuation spacing.
    while "  " in s:
        s = s.replace("  ", " ")
    s = s.replace(" .", ".").replace(" :", ":").strip()
    return s


def is_transcript_entry(entry: JsonDict) -> bool:
    """
    Returns True if this entry looks like a conversation transcript (e.g. chat/SMS),
    where `text` or `content` is a list of turns rather than a single assistant blob.
    """
    for k in ("text", "content", "messages"):
        v = entry.get(k)
        if isinstance(v, list) and v and all(isinstance(t, dict) for t in v):
            return True
    return False


def transcript_turns(entry: JsonDict) -> List[JsonDict]:
    """
    Normalize transcript turns to a list of dicts; prefers `messages`, then `text`, then `content`.
    Expected turn formats:
      - { "from": "...", "msg": "..." }
      - { "speaker": "...", "content": "..." }
      - { "role": "user"/"assistant", "content": "..." } (already chat format)
    """
    for k in ("messages", "text", "content"):
        v = entry.get(k)
        if isinstance(v, list) and all(isinstance(t, dict) for t in v):
            return v  # type: ignore[return-value]
    return []


def normalize_chat_messages(
    turns: List[JsonDict],
    *,
    self_speaker: str = "Ryan",
) -> List[JsonDict]:
    """
    Convert transcript turns into OpenAI-style chat messages:
      [{role: "user"|"assistant", content: "..."}]
    """
    out: List[JsonDict] = []
    for t in turns:
        # Already in chat form?
        role = t.get("role")
        content = t.get("content")
        if isinstance(role, str) and isinstance(content, str) and content.strip():
            r = role.strip().lower()
            if r in ("user", "assistant", "system"):
                out.append({"role": r, "content": content.strip()})
                continue

        speaker = t.get("from") or t.get("speaker") or t.get("author") or ""
        if not isinstance(speaker, str):
            speaker = str(speaker)
        speaker = speaker.strip()

        msg = t.get("msg") if "msg" in t else t.get("text")
        if msg is None:
            msg = t.get("content")
        if not isinstance(msg, str) or not msg.strip():
            continue

        is_self = speaker.lower() == self_speaker.lower()
        out.append({"role": "assistant" if is_self else "user", "content": msg.strip()})
    return out


def chat_windows(
    messages: List[JsonDict],
    *,
    max_messages: int = 12,
    require_last_role: str = "assistant",
) -> List[List[JsonDict]]:
    """
    Produce training windows that end on an assistant message, so the model is
    always trained to produce the assistant continuation.

    We generate one window per assistant message (sliding), capped to the last `max_messages`.
    """
    windows: List[List[JsonDict]] = []
    for i, m in enumerate(messages):
        if m.get("role") != require_last_role:
            continue
        start = max(0, i - max_messages + 1)
        w = messages[start : i + 1]
        # Prefer windows that start with a user turn (more natural SFT).
        while w and w[0].get("role") == "assistant":
            w = w[1:]
        if len(w) < 2:
            continue
        if w[-1].get("role") != require_last_role:
            continue
        windows.append(w)
    return windows


def build_gentle_system_message(
    entry: JsonDict,
    *,
    mode: str,
    max_len: int = 320,
) -> Optional[str]:
    """
    Create a lightweight system message that sets context without explicit identity instructions.

    mode:
      - "none"
      - "context"
      - "tone"
      - "context+tone"
      - "channel+context"
      - "channel+context+tone"
    """
    mode = (mode or "none").strip().lower()
    if mode == "none":
        return None

    parts: List[str] = []

    def _fallback_context_from_metadata(e: JsonDict) -> Optional[str]:
        """
        Use topic/channel to produce a neutral context when the authored `context`
        contains third-person possessives like "Ryan's ...".
        """
        channel = e.get("channel")
        channel_str = channel.strip() if isinstance(channel, str) else ""

        topic_val = e.get("topic") if e.get("topic") is not None else e.get("topics")
        topics: List[str] = []
        if isinstance(topic_val, list):
            topics = [str(x).strip() for x in topic_val if str(x).strip()]
        elif isinstance(topic_val, str) and topic_val.strip():
            topics = [topic_val.strip()]

        if channel_str.lower() == "professional_dialogue":
            base = "Professional chat"
        elif channel_str.lower() == "external_transcript":
            base = "Personal text exchange"
        elif channel_str:
            base = f"Conversation ({channel_str})"
        else:
            base = "Conversation"

        if topics:
            return f"{base} about: {', '.join(topics)}."
        return f"{base}."

    def _neutralize_context(ctx: str) -> str:
        """
        Heuristic rewrite to avoid third-person identity narration in system messages,
        e.g. 'Ryan discusses X' -> 'Discussion of X'.
        """
        s = ctx.strip()
        # Common patterns in this repo's chat contexts.
        mappings = {
            "discusses": "Discussion of",
            "explains": "Explanation of",
            "clarifies": "Clarification of",
            "describes": "Description of",
            "states": "Statement about",
            "notes": "Notes on",
            "recounts": "Recounting of",
            "contrasts": "Contrast of",
            "models": "Example response about",
            "gives": "Advice on",
            "rejects": "Rejection of",
            "talks about": "Discussion of",
            "talks": "Discussion of",
        }

        lower = s.lower()
        if lower.startswith("ryan "):
            # Try single-token verbs first.
            for verb, prefix in mappings.items():
                needle = f"ryan {verb} "
                if lower.startswith(needle):
                    rest = s[len(needle) :].lstrip()
                    return f"{prefix} {rest}".strip()

            # Also catch "Ryan and X ..." as a generic transcript descriptor.
            if lower.startswith("ryan and "):
                return s.replace("Ryan and", "Conversation between", 1).strip()

            # Fallback: just drop leading "Ryan " if present.
            return s[5:].lstrip()

        return s

    channel = entry.get("channel")
    if "channel" in mode and isinstance(channel, str) and channel.strip():
        parts.append(f"Channel: {channel.strip()}.")

    ctx = entry.get("context")
    if "context" in mode and isinstance(ctx, str) and ctx.strip():
        ctx_norm = _neutralize_context(ctx)

        # If the normalized context still contains third-person possessives like "Ryan's",
        # prefer a neutral, topic-derived context rather than telling the model about "Ryan".
        ctx_norm_check = ctx_norm.replace("Ryan’s", "Ryan's").lower()
        if "ryan's" in ctx_norm_check:
            fb = _fallback_context_from_metadata(entry)
            if fb:
                ctx_norm = fb
            else:
                # As a last resort, strip possessives (best-effort).
                ctx_norm = re.sub(r"\\bRyan['’]s\\b\\s*", "", ctx_norm).strip()

        parts.append(f"Context: {ctx_norm}")

    tone = entry.get("tone")
    if "tone" in mode:
        if isinstance(tone, list):
            tone_list = [str(x).strip() for x in tone if str(x).strip()]
            if tone_list:
                parts.append("Tone: " + ", ".join(tone_list) + ".")
        elif isinstance(tone, str) and tone.strip():
            parts.append(f"Tone: {tone.strip()}.")

    if not parts:
        return None

    msg = " ".join(parts).strip()
    if len(msg) <= max_len:
        return msg
    return msg[: max_len - 1].rstrip() + "…"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "corpus"),
        help="Path to corpus directory",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "training" / "v1_dataset.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Omit top-level `metadata` fields (recommended for platform.openai.com UI uploads).",
    )
    parser.add_argument(
        "--transcript-system",
        type=str,
        default="none",
        choices=[
            "none",
            "context",
            "tone",
            "context+tone",
            "channel+context",
            "channel+context+tone",
        ],
        help="Optional gentle system message for transcript rows (no explicit identity instructions).",
    )
    args = parser.parse_args()

    corpus_dir = Path(args.corpus).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    json_files = sorted(corpus_dir.rglob("*.json"))

    rows: List[str] = []
    skipped_missing_prompt = 0
    skipped_missing_text = 0
    invalid_files = 0
    transcript_rows = 0
    transcript_system_rows = 0

    for p in json_files:
        data = load_json(p)
        if data is None:
            invalid_files += 1
            continue

        for entry in extract_entries(data):
            # Transcript-style entries: emit multi-turn chat windows.
            if isinstance(entry, dict) and is_transcript_entry(entry):
                turns = transcript_turns(entry)
                msgs = normalize_chat_messages(turns, self_speaker="Ryan")
                sys_msg = build_gentle_system_message(entry, mode=args.transcript_system)
                for w in chat_windows(msgs, max_messages=12, require_last_role="assistant"):
                    if sys_msg:
                        w = [{"role": "system", "content": sys_msg}] + w
                        transcript_system_rows += 1
                    row: JsonDict = {"messages": w}
                    if not args.no_metadata:
                        row["metadata"] = {
                            "path": str(p.relative_to(corpus_dir)),
                            "source": entry.get("source"),
                            "date": entry.get("date"),
                            "id": entry.get("id"),
                            "title": entry.get("title"),
                            "topic": entry.get("topic"),
                            "topics": entry.get("topics"),
                            "tone": entry.get("tone"),
                            "channel": entry.get("channel"),
                            "kind": "transcript",
                        }
                    rows.append(json.dumps(row, ensure_ascii=False))
                    transcript_rows += 1
                continue

            prompt = entry.get("user_prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                skipped_missing_prompt += 1
                continue

            atext = assistant_text(entry)
            if not atext:
                skipped_missing_text += 1
                continue

            prompt_out = sanitize_user_prompt(prompt)

            row: JsonDict = {
                "messages": [
                    {"role": "user", "content": prompt_out},
                    {"role": "assistant", "content": atext},
                ],
            }
            if not args.no_metadata:
                row["metadata"] = {
                    "path": str(p.relative_to(corpus_dir)),
                    "source": entry.get("source"),
                    "date": entry.get("date"),
                    "id": entry.get("id"),
                    "title": entry.get("title"),
                    "topic": entry.get("topic"),
                    "topics": entry.get("topics"),
                    "tone": entry.get("tone"),
                    "channel": entry.get("channel"),
                }
            rows.append(json.dumps(row, ensure_ascii=False))

    out_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    print(f"Corpus: {corpus_dir}")
    print(f"Output: {out_path}")
    print(f"JSON files scanned: {len(json_files)} (invalid/empty: {invalid_files})")
    print(f"Rows written: {len(rows)}")
    print(f"Skipped (missing user_prompt): {skipped_missing_prompt}")
    print(f"Skipped (missing assistant text/content): {skipped_missing_text}")
    print(f"Transcript rows written: {transcript_rows}")
    print(f"Transcript rows with system message: {transcript_system_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


