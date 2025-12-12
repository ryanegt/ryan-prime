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
    return []


def assistant_text(entry: JsonDict) -> Optional[str]:
    for k in ("text", "content"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


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
    args = parser.parse_args()

    corpus_dir = Path(args.corpus).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    json_files = sorted(corpus_dir.rglob("*.json"))

    rows: List[str] = []
    skipped_missing_prompt = 0
    skipped_missing_text = 0
    invalid_files = 0

    for p in json_files:
        data = load_json(p)
        if data is None:
            invalid_files += 1
            continue

        for entry in extract_entries(data):
            prompt = entry.get("user_prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                skipped_missing_prompt += 1
                continue

            atext = assistant_text(entry)
            if not atext:
                skipped_missing_text += 1
                continue

            md = {
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

            row = {
                "messages": [
                    {"role": "user", "content": prompt.strip()},
                    {"role": "assistant", "content": atext},
                ],
                "metadata": md,
            }
            rows.append(json.dumps(row, ensure_ascii=False))

    out_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    print(f"Corpus: {corpus_dir}")
    print(f"Output: {out_path}")
    print(f"JSON files scanned: {len(json_files)} (invalid/empty: {invalid_files})")
    print(f"Rows written: {len(rows)}")
    print(f"Skipped (missing user_prompt): {skipped_missing_prompt}")
    print(f"Skipped (missing assistant text/content): {skipped_missing_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


