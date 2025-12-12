#!/usr/bin/env python3
"""
Render static HTML reports for eval runs created by scripts/run_eval.py.

Input (per run):
  - runs/<run_id>/results.jsonl
  - runs/<run_id>/config.json

Output (per run):
  - runs/<run_id>/report.html

Optionally generate an index:
  - runs/index.html
"""

from __future__ import annotations

import argparse
import html
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class EvalRow:
    eval_id: str
    bucket: str
    prompt: str
    notes: Optional[str]
    good_signals: List[str]
    red_flags: List[str]
    model: str
    output: str
    latency_s: Optional[float]
    usage: Optional[JsonDict]


def read_json(path: Path) -> JsonDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def read_results_jsonl(path: Path) -> List[EvalRow]:
    rows: List[EvalRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            continue
        rows.append(
            EvalRow(
                eval_id=str(obj.get("id") or ""),
                bucket=str(obj.get("bucket") or ""),
                prompt=str(obj.get("prompt") or ""),
                notes=(obj.get("notes") if isinstance(obj.get("notes"), str) else None),
                good_signals=[str(x) for x in (obj.get("good_signals") or [])] if isinstance(obj.get("good_signals"), list) else [],
                red_flags=[str(x) for x in (obj.get("red_flags") or [])] if isinstance(obj.get("red_flags"), list) else [],
                model=str(obj.get("model") or ""),
                output=str(obj.get("output") or ""),
                latency_s=(obj.get("latency_s") if isinstance(obj.get("latency_s"), (int, float)) else None),
                usage=(obj.get("usage") if isinstance(obj.get("usage"), dict) else None),
            )
        )
    return rows


def group_for_comparison(rows: List[EvalRow]) -> Tuple[List[str], Dict[str, Dict[str, List[EvalRow]]]]:
    """
    Returns:
      - models (sorted by first seen)
      - grouped[bucket][eval_id] -> list of rows (different models)
    """
    models: List[str] = []
    grouped: Dict[str, Dict[str, List[EvalRow]]] = {}
    for r in rows:
        if r.model and r.model not in models:
            models.append(r.model)
        grouped.setdefault(r.bucket or "unknown", {}).setdefault(r.eval_id or "unknown", []).append(r)
    return models, grouped


def esc(s: str) -> str:
    return html.escape(s or "", quote=False)


def render_report_html(config: JsonDict, rows: List[EvalRow], run_dir: Path) -> str:
    models, grouped = group_for_comparison(rows)
    model_a = models[0] if models else "model"
    model_b = models[1] if len(models) > 1 else None
    label_a = "fine-tuned" if "ft:" in (model_a or "") else "primary"
    label_b = "baseline" if model_b else None
    label_b_str = label_b or "baseline"

    buckets = sorted(grouped.keys())
    created = config.get("created_at_utc") or config.get("created_at") or ""

    # Build prompt cards HTML.
    cards_html: List[str] = []
    for bucket in buckets:
        cards_html.append(f'<section class="bucket" data-bucket="{esc(bucket)}">')
        cards_html.append(f"<h2>{esc(bucket)}</h2>")
        for eval_id, items in sorted(grouped[bucket].items(), key=lambda kv: kv[0]):
            # index rows by model
            by_model: Dict[str, EvalRow] = {r.model: r for r in items if r.model}

            # choose any row for prompt/meta (should be same across models)
            any_row = items[0]
            prompt = any_row.prompt
            notes = any_row.notes or ""
            good = any_row.good_signals
            bad = any_row.red_flags

            out_a = by_model.get(model_a).output if model_a in by_model else ""
            out_b = by_model.get(model_b).output if model_b and model_b in by_model else ""

            # allow search across prompt + outputs
            search_blob = " ".join([prompt, out_a, out_b, notes, " ".join(good), " ".join(bad)])

            cards_html.append(
                f'<article class="card" data-id="{esc(eval_id)}" data-search="{esc(search_blob.lower())}">'
            )
            cards_html.append('<div class="cardHeader">')
            cards_html.append(f'<div class="evalId">{esc(eval_id)}</div>')
            cards_html.append("</div>")

            cards_html.append(f'<div class="prompt"><div class="label">Prompt</div><pre>{esc(prompt)}</pre></div>')

            cards_html.append('<div class="meta">')
            cards_html.append(f'<div class="notes"><div class="label">Notes</div><div class="metaText">{esc(notes)}</div></div>')
            cards_html.append(
                '<div class="signals">'
                f'<div class="good"><div class="label">Good signals</div>{render_list(good)}</div>'
                f'<div class="bad"><div class="label">Red flags</div>{render_list(bad)}</div>'
                "</div>"
            )
            cards_html.append("</div>")

            cards_html.append('<div class="outputs">')
            cards_html.append(
                f'<div class="col"><div class="colTitle"><span class="tag">{esc(label_a)}</span> {esc(model_a)}</div><pre>{esc(out_a)}</pre></div>'
            )
            if model_b:
                cards_html.append(
                    f'<div class="col"><div class="colTitle"><span class="tag">{esc(label_b_str)}</span> {esc(model_b)}</div><pre>{esc(out_b)}</pre></div>'
                )
            cards_html.append("</div>")

            cards_html.append("</article>")
        cards_html.append("</section>")

    # Basic, dependency-free HTML with embedded JS.
    # Note: the run output is gitignored; this is meant for local review.
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Ryan Prime Eval Report — {esc(run_dir.name)}</title>
  <style>
    :root {{
      --bg: #0b0f16;
      --panel: #121a27;
      --panel2: #0f1623;
      --text: #e8eef7;
      --muted: #a5b4c7;
      --border: rgba(255,255,255,0.12);
      --accent: #76b7ff;
      --good: #7ee787;
      --bad: #ff7b72;
    }}
    @media (prefers-color-scheme: light) {{
      :root {{
        --bg: #f6f8fb;
        --panel: #ffffff;
        --panel2: #f2f5fa;
        --text: #0b1220;
        --muted: #4b5567;
        --border: rgba(10,20,40,0.15);
        --accent: #1f6feb;
        --good: #116329;
        --bad: #cf222e;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: linear-gradient(180deg, var(--bg), rgba(0,0,0,0));
      padding: 16px 16px 10px 16px;
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(8px);
    }}
    .titleRow {{
      display: flex;
      gap: 12px;
      align-items: baseline;
      flex-wrap: wrap;
    }}
    h1 {{
      font-size: 18px;
      margin: 0;
    }}
    .sub {{
      color: var(--muted);
      font-size: 12px;
    }}
    .controls {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 10px;
    }}
    input[type="text"], select {{
      background: var(--panel);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 13px;
      min-width: 240px;
    }}
    .toggles {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    label.chk {{
      display: inline-flex;
      gap: 8px;
      align-items: center;
      padding: 8px 10px;
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 10px;
      font-size: 12px;
      color: var(--muted);
      user-select: none;
    }}
    main {{
      padding: 16px;
      max-width: 1200px;
      margin: 0 auto;
    }}
    .bucket h2 {{
      margin: 18px 0 10px 0;
      font-size: 14px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .card {{
      background: var(--panel2);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      margin-bottom: 12px;
    }}
    .cardHeader {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }}
    .evalId {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      color: var(--muted);
      font-size: 12px;
    }}
    .label {{
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    pre {{
      white-space: pre-wrap;
      word-wrap: break-word;
      margin: 0;
      line-height: 1.35;
      font-size: 13px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 12px;
    }}
    .meta {{
      margin-top: 10px;
    }}
    .metaText {{
      color: var(--muted);
      font-size: 13px;
      padding: 0 2px;
    }}
    .signals {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 10px;
    }}
    @media (max-width: 900px) {{
      .signals {{ grid-template-columns: 1fr; }}
    }}
    .signals ul {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      font-size: 13px;
    }}
    .outputs {{
      display: grid;
      grid-template-columns: {"minmax(0,1fr) minmax(0,1fr)" if model_b else "minmax(0,1fr)"};
      gap: 12px;
      margin-top: 12px;
    }}
    /* Keep side-by-side visible for comparison; only stack on very small screens. */
    @media (max-width: 560px) {{
      .outputs {{ grid-template-columns: 1fr; }}
    }}
    .colTitle {{
      font-size: 12px;
      color: var(--accent);
      margin: 4px 0 8px 2px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }}
    .tag {{
      display: inline-block;
      padding: 2px 8px;
      margin-right: 6px;
      border: 1px solid var(--border);
      border-radius: 999px;
      color: var(--muted);
      font-size: 11px;
      letter-spacing: 0.02em;
      text-transform: lowercase;
      vertical-align: middle;
      background: var(--panel);
    }}
    .hidden {{ display: none !important; }}
  </style>
</head>
<body>
  <header>
    <div class="titleRow">
      <h1>Eval report</h1>
      <div class="sub">{esc(run_dir.name)} • {esc(str(created))}</div>
      <div class="sub">Models: {esc(model_a)}{(" vs " + esc(model_b)) if model_b else ""}</div>
    </div>
    <div class="controls">
      <input id="search" type="text" placeholder="Search prompts + outputs..." />
      <select id="bucketSelect">
        <option value="">All buckets</option>
        {''.join([f'<option value="{esc(b)}">{esc(b)}</option>' for b in buckets])}
      </select>
      <div class="toggles">
        <label class="chk"><input id="toggleMeta" type="checkbox" checked /> show notes/signals</label>
      </div>
    </div>
  </header>
  <main>
    {''.join(cards_html)}
  </main>
  <script>
    const searchEl = document.getElementById('search');
    const bucketEl = document.getElementById('bucketSelect');
    const toggleMetaEl = document.getElementById('toggleMeta');

    function applyFilters() {{
      const q = (searchEl.value || '').trim().toLowerCase();
      const b = bucketEl.value || '';

      document.querySelectorAll('section.bucket').forEach(sec => {{
        const secBucket = sec.getAttribute('data-bucket') || '';
        const bucketOk = (!b) || (b === secBucket);
        let anyVisible = false;

        sec.querySelectorAll('article.card').forEach(card => {{
          const hay = (card.getAttribute('data-search') || '');
          const searchOk = (!q) || hay.includes(q);
          const show = bucketOk && searchOk;
          card.classList.toggle('hidden', !show);
          if (show) anyVisible = true;
        }});

        sec.classList.toggle('hidden', !anyVisible);
      }});

      const showMeta = toggleMetaEl.checked;
      document.querySelectorAll('.meta').forEach(m => m.classList.toggle('hidden', !showMeta));
    }}

    searchEl.addEventListener('input', applyFilters);
    bucketEl.addEventListener('change', applyFilters);
    toggleMetaEl.addEventListener('change', applyFilters);
    applyFilters();
  </script>
</body>
</html>
"""
    return html_doc


def render_list(items: List[str]) -> str:
    if not items:
        return "<div class='metaText'>(none)</div>"
    lis = "".join([f"<li>{esc(x)}</li>" for x in items])
    return f"<ul>{lis}</ul>"


def write_report(run_dir: Path, *, out_name: str = "report.html") -> Path:
    config_path = run_dir / "config.json"
    results_path = run_dir / "results.jsonl"
    if not config_path.exists() or not results_path.exists():
        raise FileNotFoundError(f"Missing config.json or results.jsonl in {run_dir}")

    config = read_json(config_path)
    rows = read_results_jsonl(results_path)
    html_doc = render_report_html(config, rows, run_dir)
    out_path = run_dir / out_name
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path


def try_parse_run_time(run_name: str) -> Optional[datetime]:
    # matches eval_YYYYMMDD_HHMMSSZ
    try:
        parts = run_name.split("_")
        if len(parts) < 3:
            return None
        date = parts[1]
        timez = parts[2]
        if not timez.endswith("Z"):
            return None
        t = timez[:-1]
        return datetime.strptime(date + t, "%Y%m%d%H%M%S")
    except Exception:
        return None


def generate_index(runs_dir: Path, run_reports: List[Tuple[str, str, Optional[str]]]) -> Path:
    # run_reports: (run_name, report_rel_path, models_str)
    rows = []
    for run_name, report_rel, models_str in run_reports:
        m = esc(models_str or "")
        rows.append(
            f"<tr><td><a href='{esc(report_rel)}'>{esc(run_name)}</a></td><td>{m}</td></tr>"
        )

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Eval runs</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; padding: 18px; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 900px; }}
    th, td {{ border: 1px solid rgba(0,0,0,0.12); padding: 10px 12px; text-align: left; }}
    th {{ background: rgba(0,0,0,0.04); }}
    .muted {{ color: rgba(0,0,0,0.6); font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Eval runs</h1>
  <div class="muted">{esc(str(runs_dir))}</div>
  <table>
    <thead>
      <tr><th>Run</th><th>Models</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    out = runs_dir / "index.html"
    out.write_text(doc, encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None, help="Path to a single run dir (e.g. runs/eval_...Z)")
    parser.add_argument("--runs", type=str, default=None, help="Path to runs dir (e.g. runs/) to render all runs + index")
    parser.add_argument("--out-name", type=str, default="report.html", help="Report filename within run dir")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    # Default to the repo's runs/ folder if neither is provided.
    if not args.run and not args.runs:
        args.runs = "runs"

    if args.run:
        run_dir = (repo_root / args.run).resolve() if not os.path.isabs(args.run) else Path(args.run)
        out = write_report(run_dir, out_name=args.out_name)
        print(f"Wrote: {out}")
        return 0

    runs_dir = (repo_root / args.runs).resolve() if not os.path.isabs(args.runs) else Path(args.runs)
    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    # newest-ish first
    run_dirs.sort(key=lambda p: try_parse_run_time(p.name) or datetime.min, reverse=True)

    run_reports: List[Tuple[str, str, Optional[str]]] = []
    for rd in run_dirs:
        try:
            out = write_report(rd, out_name=args.out_name)
        except Exception:
            continue
        # relative link from index
        report_rel = f"{rd.name}/{out.name}"
        models_str = None
        cfg_path = rd / "config.json"
        if cfg_path.exists():
            cfg = read_json(cfg_path)
            models = cfg.get("models")
            if isinstance(models, list):
                models_str = " vs ".join([str(x) for x in models if str(x)])
        run_reports.append((rd.name, report_rel, models_str))

    idx = generate_index(runs_dir, run_reports)
    print(f"Wrote: {idx}")
    print(f"Rendered runs: {len(run_reports)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


