# RSS AI Summary

This repository turns a NetNewsWire OPML export into a daily GitHub Pages digest.

## What it does

- Reads `subscriptions.opml`
- Fetches each RSS or Atom feed
- Keeps a persistent backlog of retained feed items in `data/state.json`
- Pulls full article text when the feed content is too short
- Updates per-feed TLDRs with the configured LLM provider
- Publishes a static dashboard to `site/index.html`

## Files

- `subscriptions.opml`: your NetNewsWire export
- `src/main.py`: CLI entrypoint
- `src/rss_digest.py`: feed ingestion, summarization, and HTML rendering
- `data/state.json`: persistent backlog state committed back to the repo
- `data/latest.json`: latest digest data for debugging or reuse
- `.github/workflows/daily-summary.yml`: scheduled generation and Pages deployment

## GitHub setup

1. Add the repository secret `OPENROUTER_API_KEY`.
2. Optional: add the repository secret `GEMINI_API_KEY` if you want to switch providers later.
3. In GitHub Pages settings, set the source to `GitHub Actions`.
4. Commit and push this repository.
5. Run the `Daily Feed Summary` workflow once with `workflow_dispatch` to bootstrap the first digest.

## Optional repository variables

- `LLM_PROVIDER`: defaults to `openrouter`
- `OPENROUTER_MODEL`: defaults to `qwen/qwen3.6-plus:free`
- `GEMINI_MODEL`: defaults to `gemini-2.0-flash-lite`
- `SITE_TITLE`: defaults to `Daily Feed TLDR`
- `SUMMARY_LANGUAGE`: defaults to `English`

## Local run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/main.py --mock-summary
```

`--mock-summary` skips the LLM provider entirely and generates synthetic summaries so you can verify the pipeline and the site locally without an API key.

For a real local run with the default provider:

```bash
OPENROUTER_API_KEY=your_key_here .venv/bin/python src/main.py
```

To force Google AI Studio instead:

```bash
LLM_PROVIDER=gemini GEMINI_API_KEY=your_key_here .venv/bin/python src/main.py
```

## Notes

- The workflow commits `data/state.json`, `data/latest.json`, and `site/index.html` so the next run stays incremental.
- The script keeps retained items on the page across workflow runs instead of dropping them after a single digest.
- New runs append newly discovered items to each feed backlog and refresh the per-feed TLDRs.
- Some feeds only expose excerpts. In those cases the pipeline falls back to feed-provided text if article extraction fails.
