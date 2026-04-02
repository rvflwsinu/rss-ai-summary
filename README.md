# RSS AI Summary

This repository turns a NetNewsWire OPML export into a daily GitHub Pages digest.

## What it does

- Reads `subscriptions.opml`
- Fetches each RSS or Atom feed
- Keeps only unseen entries with `data/state.json`
- Pulls full article text when the feed content is too short
- Summarizes each updated feed with Gemini
- Publishes a static dashboard to `site/index.html`

## Files

- `subscriptions.opml`: your NetNewsWire export
- `src/main.py`: CLI entrypoint
- `src/rss_digest.py`: feed ingestion, summarization, and HTML rendering
- `data/state.json`: dedupe state committed back to the repo
- `data/latest.json`: latest digest data for debugging or reuse
- `.github/workflows/daily-summary.yml`: scheduled generation and Pages deployment

## GitHub setup

1. Add the repository secret `GEMINI_API_KEY`.
2. In GitHub Pages settings, set the source to `GitHub Actions`.
3. Commit and push this repository.
4. Run the `Daily Feed Summary` workflow once with `workflow_dispatch` to bootstrap the first digest.

## Optional repository variables

- `GEMINI_MODEL`: defaults to `gemini-2.0-flash-lite`
- `SITE_TITLE`: defaults to `Daily Feed TLDR`
- `SUMMARY_LANGUAGE`: defaults to `English`
- `MAX_ITEMS_PER_FEED`: defaults to `4`

## Local run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/main.py --mock-summary
```

`--mock-summary` skips Gemini and generates synthetic summaries so you can verify the pipeline and the site locally without an API key.

## Notes

- The workflow commits `data/state.json`, `data/latest.json`, and `site/index.html` so the next run stays incremental.
- The script marks all unseen entries as seen, but only summarizes the newest `MAX_ITEMS_PER_FEED` items for each feed. This keeps the daily digest small instead of building a backlog.
- Some feeds only expose excerpts. In those cases the pipeline falls back to feed-provided text if article extraction fails.
