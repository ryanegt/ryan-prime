#!/usr/bin/env python3
"""
Run an eval set (e.g. eval/basic.json) against a model and write results to /runs.

Outputs:
  runs/<run_id>/
    - config.json
    - results.jsonl   (one record per prompt per model)
    - summary.md      (quick human-readable summary)

Usage (dry-run, no API calls):
  python3 scripts/run_eval.py --eval eval/basic.json --model gpt-4.1 --dry-run

Usage (real):
  export OPENAI_API_KEY=...
  python3 scripts/run_eval.py --eval eval/basic.json --model <YOUR_FINE_TUNED_MODEL_ID> --baseline gpt-4.1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class ModelResult:
    model: str
    output_text: str
    latency_s: float
    usage: Optional[JsonDict]


def utc_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}Z"


def load_eval_items(path: Path) -> List[JsonDict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return [x for x in data["items"] if isinstance(x, dict)]
    raise ValueError(f"Unsupported eval schema in {path}. Expected list or {{items: [...]}}.")


def build_messages(prompt: str, system: Optional[str]) -> List[JsonDict]:
    msgs: List[JsonDict] = []
    if system and system.strip():
        msgs.append({"role": "system", "content": system.strip()})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def openai_client():
    # Lazy import so --dry-run works without the SDK installed.
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing OpenAI SDK. Install with: `pip install openai` "
            "or run with --dry-run."
        ) from e
    return OpenAI()


def load_conf(path: Path) -> JsonDict:
    """
    Load a simple JSON config file (typically gitignored).
    Example:
      {
        "api_key": "sk-...",
        "ft-model": "ft:gpt-4.1:..."
      }
    """
    raw = path.read_text(encoding="utf-8").strip()
    data = json.loads(raw) if raw else {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid conf format (expected JSON object): {path}")
    return data  # type: ignore[return-value]


def first_present(d: JsonDict, keys: List[str]) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def run_one(
    *,
    model: str,
    messages: List[JsonDict],
    temperature: float,
    max_output_tokens: int,
    dry_run: bool,
) -> ModelResult:
    if dry_run:
        return ModelResult(
            model=model,
            output_text="",
            latency_s=0.0,
            usage=None,
        )

    client = openai_client()
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
    dt = time.time() - t0

    out = ""
    if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
        out = resp.choices[0].message.content

    usage = None
    # The SDK returns a pydantic-ish object; convert best-effort.
    if getattr(resp, "usage", None) is not None:
        u = resp.usage
        usage = {
            "prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None),
            "total_tokens": getattr(u, "total_tokens", None),
        }

    return ModelResult(model=model, output_text=out, latency_s=dt, usage=usage)


def write_summary_md(run_dir: Path, config: JsonDict, records: List[JsonDict]) -> None:
    by_model: Dict[str, int] = {}
    for r in records:
        by_model[r["model"]] = by_model.get(r["model"], 0) + 1

    lines: List[str] = []
    lines.append(f"# Eval run: `{run_dir.name}`")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("```")
    lines.append(json.dumps(config, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    for m, n in sorted(by_model.items(), key=lambda x: x[0]):
        lines.append(f"- **{m}**: {n} prompts")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `results.jsonl` contains one JSON object per prompt per model.")
    lines.append("- For scoring, you can eyeball against `notes`, `good_signals`, and `red_flags`.")
    lines.append("")
    run_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=str, default="eval/basic.json", help="Path to eval JSON")
    parser.add_argument("--model", type=str, default=None, help="Model id to test (e.g. your fine-tuned model)")
    parser.add_argument("--baseline", type=str, default=None, help="Optional baseline model to compare against")
    parser.add_argument("--system", type=str, default=None, help="Optional system message applied to all prompts")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-output-tokens", type=int, default=400)
    parser.add_argument("--runs-dir", type=str, default="runs", help="Directory to place run outputs")
    parser.add_argument("--run-id", type=str, default=None, help="Override run id folder name")
    parser.add_argument("--dry-run", action="store_true", help="Do not call the API; just write files")
    parser.add_argument(
        "--conf",
        type=str,
        default=".conf",
        help="Optional JSON config file (gitignored) containing api_key and default model ids.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    eval_path = (repo_root / args.eval).resolve() if not os.path.isabs(args.eval) else Path(args.eval)
    runs_root = (repo_root / args.runs_dir).resolve() if not os.path.isabs(args.runs_dir) else Path(args.runs_dir)
    runs_root.mkdir(parents=True, exist_ok=True)

    # Load config defaults (if present).
    conf_path = (repo_root / args.conf).resolve() if not os.path.isabs(args.conf) else Path(args.conf)
    conf: JsonDict = {}
    if conf_path.exists():
        try:
            conf = load_conf(conf_path)
        except Exception:
            conf = {}

    # Allow .conf to supply OPENAI_API_KEY if not set.
    if not os.environ.get("OPENAI_API_KEY"):
        api_key = first_present(conf, ["api_key", "OPENAI_API_KEY", "openai_api_key"])
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    # Allow .conf to supply default fine-tuned model id.
    model = args.model or first_present(conf, ["ft-model", "ft_model", "model", "fine_tuned_model"])
    if not model:
        raise SystemExit("Missing --model and no model found in .conf (expected key like 'ft-model').")

    # Optional baseline can come from .conf too.
    baseline = args.baseline or first_present(conf, ["baseline", "baseline_model"])

    run_id = args.run_id or utc_run_id("eval")
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    items = load_eval_items(eval_path)
    models = [model] + ([baseline] if baseline else [])

    config: JsonDict = {
        "eval": str(eval_path),
        "models": models,
        "system": args.system,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "dry_run": args.dry_run,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "conf_path": str(conf_path) if conf_path.exists() else None,
    }
    run_dir.joinpath("config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    records: List[JsonDict] = []
    out_lines: List[str] = []

    for item in items:
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue

        messages = build_messages(prompt.strip(), args.system)
        for model in models:
            if not model:
                continue
            r = run_one(
                model=model,
                messages=messages,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                dry_run=args.dry_run,
            )
            rec: JsonDict = {
                "id": item.get("id"),
                "bucket": item.get("bucket"),
                "prompt": prompt.strip(),
                "notes": item.get("notes"),
                "good_signals": item.get("good_signals"),
                "red_flags": item.get("red_flags"),
                "model": r.model,
                "output": r.output_text,
                "latency_s": r.latency_s,
                "usage": r.usage,
            }
            records.append(rec)
            out_lines.append(json.dumps(rec, ensure_ascii=False))

    run_dir.joinpath("results.jsonl").write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    write_summary_md(run_dir, config, records)

    print(f"Wrote run: {run_dir}")
    print(f"Records: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


