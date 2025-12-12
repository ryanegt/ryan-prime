Prompts to evaluate quality of model responses.
vibe.json tests 6 categories:
- voice + vibe (5)
- epistemics / reasoning (5) 
- boundaries / refusals (5)
- conflict / being right (5)
- attachment / relationships (5)
- work / thinking style (5)

## Viewing results (static HTML, no backend)

After you run `scripts/run_eval.py` it will write a folder under `runs/` containing `results.jsonl`.

Generate HTML reports:

```bash
python3 scripts/render_eval_html.py --runs runs
```

This writes:
- `runs/index.html` (links to each run)
- `runs/<run_id>/report.html` (filter/search + side-by-side comparison if you ran a baseline)

Open `runs/index.html` in your browser.