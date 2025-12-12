#!/usr/bin/env python3
"""
Bulk-inject `user_prompt` into corpus JSON entries.

Supports two shapes:
  1) { "items": [ { ... }, ... ] }
  2) { ...single entry... }

Behavior:
  - Adds `user_prompt` only when missing or blank.
  - Does NOT overwrite existing non-empty `user_prompt`.
  - Skips files that are empty or not valid JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


JsonDict = Dict[str, Any]


def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []
    return [str(v).strip()] if str(v).strip() else []


def _pick(*vals: Any) -> Optional[str]:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _join_csv(items: List[str]) -> Optional[str]:
    items = [x.strip() for x in items if x.strip()]
    if not items:
        return None
    return ", ".join(items)


def generate_user_prompt(entry: JsonDict, file_path: Path) -> str:
    source = str(entry.get("source") or "").strip().lower()
    title = _pick(entry.get("title"), entry.get("id"))
    context = _pick(entry.get("context"), entry.get("summary"))
    tone_csv = _join_csv(_as_list(entry.get("tone")))
    topic_csv = _join_csv(_as_list(entry.get("topic") or entry.get("topics")))

    # Special-case: personal QA tends to already include question prompts.
    # If it's missing, prefer summary/context as a stand-in.
    if source == "personal_qa":
        if context:
            return f"Answer in first person with Ryan's voice. Prompt: {context}"
        if topic_csv:
            return f"Answer in first person with Ryan's voice about: {topic_csv}"
        return "Answer in first person with Ryan's voice."

    if source == "email":
        parts = ["Draft an email in Ryan's voice."]
        if context:
            parts.append(f"Context: {context}")
        if topic_csv:
            parts.append(f"Topic: {topic_csv}")
        if tone_csv:
            parts.append(f"Tone: {tone_csv}")
        return " ".join(parts)

    if source == "instagram":
        parts = ["Write an Instagram caption in Ryan's voice."]
        if title and entry.get("title"):
            parts.append(f"Post title: {title}")
        if context:
            parts.append(f"Context: {context}")
        if topic_csv:
            parts.append(f"Topics: {topic_csv}")
        if tone_csv:
            parts.append(f"Tone: {tone_csv}")
        return " ".join(parts)

    if source == "blog":
        parts = []
        if entry.get("title"):
            parts.append(f"Write a first-person blog post titled \"{entry['title']}\" in Ryan's voice.")
        else:
            parts.append("Write a first-person blog post in Ryan's voice.")
        if context:
            parts.append(f"Context: {context}")
        if topic_csv:
            parts.append(f"Topics: {topic_csv}")
        if tone_csv:
            parts.append(f"Tone: {tone_csv}")
        return " ".join(parts)

    if source == "whitepaper":
        parts = []
        if entry.get("title"):
            parts.append(f"Write a technical memo/whitepaper section titled \"{entry['title']}\" in Ryan's voice.")
        else:
            parts.append("Write a technical memo/whitepaper section in Ryan's voice.")
        if context:
            parts.append(f"Context: {context}")
        if topic_csv:
            parts.append(f"Topics: {topic_csv}")
        if tone_csv:
            parts.append(f"Tone: {tone_csv}")
        return " ".join(parts)

    # Fallback: infer from whatever we have.
    parts = ["Write in Ryan's voice."]
    if title and entry.get("title"):
        parts.append(f"Title: {title}")
    if context:
        parts.append(f"Context: {context}")
    if topic_csv:
        parts.append(f"Topics: {topic_csv}")
    if tone_csv:
        parts.append(f"Tone: {tone_csv}")

    # If we somehow have nothing, at least provide a stable prompt.
    if len(parts) == 1:
        parts.append(f"Source file: {file_path.name}")

    return " ".join(parts)


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


def update_file(path: Path, *, dry_run: bool) -> Tuple[int, int, str]:
    """
    Returns (entries_seen, entries_updated, status).
    status in {"updated","skipped","invalid"}.
    """
    data = load_json(path)
    if data is None:
        return (0, 0, "invalid")

    entries: List[JsonDict] = []
    container_kind: str = "unknown"

    if isinstance(data, dict) and isinstance(data.get("items"), list):
        container_kind = "items"
        entries = [e for e in data["items"] if isinstance(e, dict)]
    elif is_entry_dict(data):
        container_kind = "root"
        entries = [data]  # type: ignore[list-item]
    else:
        return (0, 0, "skipped")

    seen = 0
    updated = 0
    for entry in entries:
        seen += 1
        existing = entry.get("user_prompt")
        if isinstance(existing, str) and existing.strip():
            continue
        entry["user_prompt"] = generate_user_prompt(entry, path)
        updated += 1

    if updated == 0:
        return (seen, 0, "skipped")

    if not dry_run:
        if container_kind == "items":
            path.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
        elif container_kind == "root":
            path.write_text(json.dumps(entries[0], ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

    return (seen, updated, "updated")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "corpus"),
        help="Path to corpus directory",
    )
    parser.add_argument("--dry-run", action="store_true", help="Analyze and report without writing changes")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus).expanduser().resolve()
    if not corpus_dir.exists():
        raise SystemExit(f"Corpus directory not found: {corpus_dir}")

    json_files = sorted(corpus_dir.rglob("*.json"))

    total_seen = 0
    total_updated = 0
    invalid_files: List[Path] = []
    skipped_files: List[Path] = []
    updated_files: List[Tuple[Path, int]] = []

    for p in json_files:
        seen, updated, status = update_file(p, dry_run=args.dry_run)
        total_seen += seen
        total_updated += updated
        if status == "invalid":
            invalid_files.append(p)
        elif status == "skipped":
            skipped_files.append(p)
        elif status == "updated":
            updated_files.append((p, updated))

    print(f"Corpus: {corpus_dir}")
    print(f"JSON files: {len(json_files)}")
    print(f"Entries seen: {total_seen}")
    print(f"Entries updated (added user_prompt): {total_updated}")
    print(f"Updated files: {len(updated_files)}")
    print(f"Skipped files: {len(skipped_files)}")
    print(f"Invalid/empty JSON files: {len(invalid_files)}")
    if invalid_files:
        print("Invalid/empty:")
        for p in invalid_files:
            print(f"  - {p.relative_to(corpus_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


