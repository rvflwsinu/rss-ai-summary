from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape, unescape
from pathlib import Path
from typing import Any

import feedparser
import httpx
import trafilatura

USER_AGENT = "rss-ai-summary-bot/0.1"
STATE_VERSION = 1
DEFAULT_MODEL = "gemini-2.0-flash-lite"
DEFAULT_SUMMARY_LANGUAGE = "English"


@dataclass
class FeedConfig:
    category: str
    title: str
    xml_url: str
    html_url: str | None = None


def main() -> int:
    args = parse_args()
    config = load_runtime_config(args)
    report = build_digest_report(config)
    write_outputs(config, report)
    print(
        f"Generated digest for {report['feeds_with_updates']} feed(s) "
        f"with {report['total_items']} item(s)."
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an AI summarized RSS digest from an OPML file."
    )
    parser.add_argument(
        "--opml-path",
        default="subscriptions.opml",
        help="Path to the NetNewsWire OPML export.",
    )
    parser.add_argument(
        "--state-path",
        default="data/state.json",
        help="Path to the persisted dedupe state JSON file.",
    )
    parser.add_argument(
        "--latest-path",
        default="data/latest.json",
        help="Path to the latest digest JSON output.",
    )
    parser.add_argument(
        "--site-dir",
        default="site",
        help="Directory where the generated static site is written.",
    )
    parser.add_argument(
        "--mock-summary",
        action="store_true",
        help="Skip Gemini and synthesize summaries from the feed titles for local testing.",
    )
    return parser.parse_args()


def load_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "opml_path": Path(args.opml_path),
        "state_path": Path(args.state_path),
        "latest_path": Path(args.latest_path),
        "site_dir": Path(args.site_dir),
        "mock_summary": args.mock_summary,
        "gemini_api_key": os.getenv("GEMINI_API_KEY", "").strip(),
        "gemini_model": os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        "summary_language": os.getenv("SUMMARY_LANGUAGE", DEFAULT_SUMMARY_LANGUAGE).strip()
        or DEFAULT_SUMMARY_LANGUAGE,
        "site_title": os.getenv("SITE_TITLE", "Daily Feed TLDR").strip() or "Daily Feed TLDR",
        "max_items_per_feed": positive_int_env("MAX_ITEMS_PER_FEED", 4),
        "max_seen_ids_per_feed": positive_int_env("MAX_SEEN_IDS_PER_FEED", 500),
        "min_feed_content_chars": positive_int_env("MIN_FEED_CONTENT_CHARS", 700),
        "max_item_chars": positive_int_env("MAX_ITEM_CHARS", 4000),
        "max_prompt_chars": positive_int_env("MAX_PROMPT_CHARS", 18000),
    }


def positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {name} must be greater than zero.")
    return value


def build_digest_report(config: dict[str, Any]) -> dict[str, Any]:
    if not config["opml_path"].exists():
        raise FileNotFoundError(f"OPML file not found: {config['opml_path']}")
    if not config["mock_summary"] and not config["gemini_api_key"]:
        raise ValueError(
            "GEMINI_API_KEY is required unless --mock-summary is enabled."
        )

    feeds = parse_opml(config["opml_path"])
    state = load_state(config["state_path"])
    generated_at = utc_now_iso()

    report: dict[str, Any] = {
        "site_title": config["site_title"],
        "generated_at": generated_at,
        "generated_at_human": human_timestamp(generated_at),
        "model": "mock-summary" if config["mock_summary"] else config["gemini_model"],
        "summary_language": config["summary_language"],
        "total_feeds": len(feeds),
        "feeds_with_updates": 0,
        "total_items": 0,
        "feeds": [],
        "errors": [],
    }

    transport = httpx.HTTPTransport(retries=2)
    timeout = httpx.Timeout(20.0, connect=10.0)
    with httpx.Client(
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        transport=transport,
    ) as client:
        for feed in feeds:
            digest = process_feed(client, config, state, feed)
            if digest["error"]:
                report["errors"].append(
                    {
                        "feed": feed.title,
                        "category": feed.category,
                        "message": digest["error"],
                    }
                )
            if digest["summary"]:
                report["feeds"].append(digest["summary"])
                report["feeds_with_updates"] += 1
                report["total_items"] += digest["summary"]["item_count"]

    state["version"] = STATE_VERSION
    state["updated_at"] = generated_at
    state["feed_count"] = len(feeds)
    report["categories"] = ordered_categories(report["feeds"])
    report["state"] = state
    return report


def parse_opml(opml_path: Path) -> list[FeedConfig]:
    tree = ET.parse(opml_path)
    root = tree.getroot()
    body = root.find("body")
    if body is None:
        raise ValueError("The OPML file does not have a <body> section.")

    feeds: list[FeedConfig] = []

    def visit(node: ET.Element, category: str) -> None:
        for outline in node.findall("outline"):
            xml_url = outline.attrib.get("xmlUrl")
            title = (
                outline.attrib.get("title")
                or outline.attrib.get("text")
                or xml_url
                or "Untitled feed"
            )
            if xml_url:
                feeds.append(
                    FeedConfig(
                        category=category,
                        title=title,
                        xml_url=xml_url,
                        html_url=outline.attrib.get("htmlUrl"),
                    )
                )
                continue
            nested_category = outline.attrib.get("title") or outline.attrib.get("text") or category
            visit(outline, nested_category)

    visit(body, "Ungrouped")
    return feeds


def process_feed(
    client: httpx.Client,
    config: dict[str, Any],
    state: dict[str, Any],
    feed: FeedConfig,
) -> dict[str, Any]:
    try:
        parsed_feed = fetch_feed(client, feed.xml_url)
    except Exception as exc:
        return {"summary": None, "error": f"Feed fetch failed: {exc}"}

    feed_state = state.setdefault("feeds", {}).setdefault(feed.xml_url, {})
    seen_ids = list(feed_state.get("seen_ids", []))
    seen_lookup = set(seen_ids)
    normalized_entries = normalize_entries(parsed_feed, feed)

    unseen_entries = [entry for entry in normalized_entries if entry["id"] not in seen_lookup]
    selected_entries = unseen_entries[: config["max_items_per_feed"]]

    all_unseen_ids = [entry["id"] for entry in unseen_entries]
    feed_title = parsed_feed.feed.get("title") or feed.title
    display_feed = FeedConfig(
        category=feed.category,
        title=feed_title,
        xml_url=feed.xml_url,
        html_url=feed.html_url,
    )

    feed_state["seen_ids"] = trim_unique(all_unseen_ids + seen_ids, config["max_seen_ids_per_feed"])
    feed_state["last_checked_at"] = utc_now_iso()
    feed_state["feed_title"] = feed_title

    if not selected_entries:
        return {"summary": None, "error": None}

    summary_inputs = []
    output_items = []
    for entry in selected_entries:
        source_text, source_kind = resolve_entry_text(
            client,
            entry,
            min_feed_content_chars=config["min_feed_content_chars"],
            max_item_chars=config["max_item_chars"],
        )
        summary_inputs.append(
            {
                "title": entry["title"],
                "link": entry["link"],
                "published": entry["published"],
                "text": source_text,
            }
        )
        output_items.append(
            {
                "title": entry["title"],
                "link": entry["link"],
                "published": entry["published"],
                "source_kind": source_kind,
            }
        )

    summary_error = None
    try:
        summary = summarize_feed(config, display_feed, summary_inputs)
    except Exception as exc:
        summary = fallback_summary(display_feed, output_items)
        summary_error = f"Gemini summary failed, used fallback: {exc}"

    digest = {
        "category": feed.category,
        "title": feed_title,
        "site_url": feed.html_url or parsed_feed.feed.get("link") or feed.xml_url,
        "feed_url": feed.xml_url,
        "item_count": len(output_items),
        "skipped_count": max(0, len(unseen_entries) - len(output_items)),
        "tldr": summary["tldr"],
        "highlights": summary["highlights"],
        "items": output_items,
    }
    return {"summary": digest, "error": summary_error}


def fetch_feed(client: httpx.Client, url: str) -> feedparser.FeedParserDict:
    response = client.get(url)
    response.raise_for_status()
    parsed = feedparser.parse(response.content)
    if parsed.bozo and not parsed.entries:
        raise ValueError(str(parsed.bozo_exception))
    return parsed


def normalize_entries(
    parsed_feed: feedparser.FeedParserDict,
    feed: FeedConfig,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for entry in parsed_feed.entries:
        timestamp = entry_timestamp(entry)
        entries.append(
            {
                "id": entry_fingerprint(feed.xml_url, entry),
                "title": normalize_text(entry.get("title") or "Untitled item"),
                "link": entry.get("link"),
                "published": timestamp.strftime("%Y-%m-%d %H:%M UTC") if timestamp else None,
                "sort_key": timestamp.timestamp() if timestamp else 0,
                "feed_text": feed_entry_text(entry),
            }
        )
    entries.sort(key=lambda item: item["sort_key"], reverse=True)
    return entries


def entry_timestamp(entry: feedparser.FeedParserDict) -> datetime | None:
    for field_name in ("published_parsed", "updated_parsed", "created_parsed"):
        value = entry.get(field_name)
        if not value:
            continue
        try:
            return datetime(*value[:6], tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def entry_fingerprint(feed_url: str, entry: feedparser.FeedParserDict) -> str:
    basis = "|".join(
        [
            feed_url,
            entry.get("id", ""),
            entry.get("link", ""),
            entry.get("title", ""),
            entry.get("published", ""),
        ]
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def feed_entry_text(entry: feedparser.FeedParserDict) -> str:
    chunks: list[str] = []
    for item in entry.get("content", []):
        value = item.get("value")
        if value:
            chunks.append(value)
    if entry.get("summary"):
        chunks.append(entry["summary"])
    if entry.get("description"):
        chunks.append(entry["description"])
    return normalize_text("\n\n".join(chunks))


def resolve_entry_text(
    client: httpx.Client,
    entry: dict[str, Any],
    *,
    min_feed_content_chars: int,
    max_item_chars: int,
) -> tuple[str, str]:
    feed_text = clip_text(entry["feed_text"], max_item_chars)
    if len(feed_text) >= min_feed_content_chars or not entry.get("link"):
        return feed_text or entry["title"], "feed"

    try:
        article_text = extract_article_text(client, entry["link"])
    except Exception:
        article_text = ""

    article_text = clip_text(article_text, max_item_chars)
    if len(article_text) > len(feed_text):
        return article_text, "article"
    return feed_text or entry["title"], "feed"


def extract_article_text(client: httpx.Client, url: str) -> str:
    response = client.get(url)
    response.raise_for_status()
    extracted = trafilatura.extract(
        response.text,
        url=url,
        favor_precision=True,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
    )
    return normalize_text(extracted or "")


def summarize_feed(
    config: dict[str, Any],
    feed: FeedConfig,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    if config["mock_summary"]:
        return fallback_summary(feed, items)

    payload_items = []
    remaining_chars = config["max_prompt_chars"]
    for item in items:
        clipped_text = clip_text(item["text"], min(len(item["text"]), remaining_chars))
        if not clipped_text:
            continue
        payload_items.append(
            {
                "title": item["title"],
                "published": item["published"],
                "link": item["link"],
                "text": clipped_text,
            }
        )
        remaining_chars -= len(clipped_text)
        if remaining_chars <= 0:
            break

    if not payload_items:
        return fallback_summary(feed, items)

    prompt = textwrap.dedent(
        f"""
        You are creating a daily digest for a single RSS feed.

        Rules:
        - Reply with valid JSON only.
        - Write in {config['summary_language']}.
        - Use only the supplied source material.
        - Be concise and avoid hype.
        - Capture the common themes or the most important changes across the feed.
        - Do not fabricate details that are not present in the sources.

        Return this exact JSON shape:
        {{
          "tldr": "2-4 sentence feed-level summary",
          "highlights": ["short bullet", "short bullet", "short bullet"]
        }}

        Feed:
        {json.dumps({"title": feed.title, "category": feed.category}, ensure_ascii=False, indent=2)}

        Items:
        {json.dumps(payload_items, ensure_ascii=False, indent=2)}
        """
    ).strip()

    response_data = call_gemini_json(
        api_key=config["gemini_api_key"],
        model=config["gemini_model"],
        prompt=prompt,
    )
    tldr = normalize_text(str(response_data.get("tldr", "")))
    highlights = [
        normalize_text(str(item))
        for item in response_data.get("highlights", [])
        if normalize_text(str(item))
    ]
    if not tldr:
        raise ValueError("Gemini response did not contain a TLDR.")
    if not highlights:
        highlights = [item["title"] for item in items[:3]]
    return {"tldr": tldr, "highlights": highlights[:5]}


def call_gemini_json(*, api_key: str, model: str, prompt: str) -> dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }
    response = httpx.post(
        url,
        params={"key": api_key},
        json=payload,
        headers={"User-Agent": USER_AGENT},
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini returned no candidates.")
    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        raise ValueError("Gemini returned no content parts.")
    text = parts[0].get("text", "")
    if not text:
        raise ValueError("Gemini returned an empty response body.")
    return json.loads(strip_code_fences(text))


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def fallback_summary(feed: FeedConfig, items: list[dict[str, Any]]) -> dict[str, Any]:
    highlight_titles = [item["title"] for item in items[:3] if item.get("title")]
    if not highlight_titles:
        highlight_titles = [f"New items detected in {feed.title}."]
    tldr = (
        f"{feed.title} has {len(items)} new item(s) in this run. "
        f"Key items include {join_titles(highlight_titles)}."
    )
    return {"tldr": tldr, "highlights": highlight_titles}


def join_titles(titles: list[str]) -> str:
    if len(titles) == 1:
        return f'"{titles[0]}"'
    if len(titles) == 2:
        return f'"{titles[0]}" and "{titles[1]}"'
    return ", ".join(f'"{title}"' for title in titles[:-1]) + f', and "{titles[-1]}"'


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": STATE_VERSION, "feeds": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def write_outputs(config: dict[str, Any], report: dict[str, Any]) -> None:
    state = report.pop("state")
    ensure_parent(config["state_path"])
    ensure_parent(config["latest_path"])
    config["site_dir"].mkdir(parents=True, exist_ok=True)

    config["state_path"].write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config["latest_path"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (config["site_dir"] / ".nojekyll").write_text("\n", encoding="utf-8")
    (config["site_dir"] / "index.html").write_text(
        render_site(report),
        encoding="utf-8",
    )


def render_site(report: dict[str, Any]) -> str:
    cards = []
    if report["feeds"]:
        for category in report["categories"]:
            category_cards = []
            for feed in report["feeds"]:
                if feed["category"] != category:
                    continue
                item_list = "".join(
                    render_item(item) for item in feed["items"]
                )
                skipped_note = ""
                if feed["skipped_count"]:
                    skipped_note = (
                        f"<p class=\"meta-note\">"
                        f"{feed['skipped_count']} more new item(s) were skipped "
                        f"by the per-feed cap.</p>"
                    )
                category_cards.append(
                    f"""
                    <article class="feed-card">
                      <div class="feed-head">
                        <div>
                          <h3><a href="{escape(feed['site_url'])}">{escape(feed['title'])}</a></h3>
                          <p class="meta">{escape(feed['item_count'].__str__())} new item(s)</p>
                        </div>
                        <a class="rss-link" href="{escape(feed['feed_url'])}">RSS</a>
                      </div>
                      <p class="tldr">{escape(feed['tldr'])}</p>
                      <ul class="highlights">{''.join(f'<li>{escape(point)}</li>' for point in feed['highlights'])}</ul>
                      {skipped_note}
                      <ul class="items">{item_list}</ul>
                    </article>
                    """
                )
            cards.append(
                f"""
                <section class="category-block">
                  <h2>{escape(category)}</h2>
                  <div class="feed-grid">{''.join(category_cards)}</div>
                </section>
                """
            )
    else:
        cards.append(
            """
            <section class="empty-state">
              <h2>No new items this run</h2>
              <p>The workflow completed, but every feed was already up to date.</p>
            </section>
            """
        )

    errors_html = ""
    if report["errors"]:
        errors_html = (
            "<section class=\"errors\"><h2>Feed Errors</h2><ul>"
            + "".join(
                f"<li><strong>{escape(error['feed'])}</strong>: {escape(error['message'])}</li>"
                for error in report["errors"]
            )
            + "</ul></section>"
        )

    return textwrap.dedent(
        f"""
        <!DOCTYPE html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>{escape(report['site_title'])}</title>
            <style>
              :root {{
                color-scheme: light dark;
                --bg: #0b1020;
                --panel: rgba(20, 28, 52, 0.88);
                --panel-soft: rgba(255, 255, 255, 0.06);
                --text: #edf2ff;
                --muted: #a7b2d1;
                --accent: #7dd3fc;
                --accent-2: #c084fc;
                --border: rgba(255, 255, 255, 0.08);
              }}
              * {{ box-sizing: border-box; }}
              body {{
                margin: 0;
                font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
                background:
                  radial-gradient(circle at top left, rgba(125, 211, 252, 0.18), transparent 28%),
                  radial-gradient(circle at top right, rgba(192, 132, 252, 0.14), transparent 22%),
                  var(--bg);
                color: var(--text);
              }}
              a {{ color: inherit; }}
              .shell {{ max-width: 1200px; margin: 0 auto; padding: 32px 20px 64px; }}
              .hero {{ margin-bottom: 28px; }}
              .hero h1 {{ margin: 0 0 10px; font-size: clamp(2rem, 5vw, 3.4rem); line-height: 1.05; }}
              .hero p {{ margin: 0; color: var(--muted); max-width: 720px; }}
              .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px;
                margin: 24px 0 32px;
              }}
              .stat-card, .feed-card, .errors, .empty-state {{
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 18px;
                backdrop-filter: blur(16px);
                box-shadow: 0 18px 42px rgba(0, 0, 0, 0.24);
              }}
              .stat-card {{ padding: 16px 18px; }}
              .stat-label {{ margin: 0 0 8px; color: var(--muted); font-size: 0.92rem; }}
              .stat-value {{ margin: 0; font-size: 1.6rem; font-weight: 700; }}
              .category-block {{ margin-top: 30px; }}
              .category-block h2 {{ margin: 0 0 14px; font-size: 1.1rem; letter-spacing: 0.04em; text-transform: uppercase; color: var(--accent); }}
              .feed-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 16px; }}
              .feed-card {{ padding: 18px; }}
              .feed-head {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }}
              .feed-head h3 {{ margin: 0 0 6px; font-size: 1.15rem; }}
              .meta, .meta-note {{ margin: 0; color: var(--muted); font-size: 0.9rem; }}
              .rss-link {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 8px 10px;
                border-radius: 999px;
                background: var(--panel-soft);
                text-decoration: none;
                font-size: 0.88rem;
              }}
              .tldr {{ margin: 16px 0 14px; line-height: 1.6; }}
              .highlights, .items, .errors ul {{ margin: 0; padding-left: 20px; }}
              .highlights li, .items li, .errors li {{ margin: 0 0 10px; color: var(--muted); }}
              .items a {{ color: var(--text); text-decoration: none; }}
              .items a:hover, .feed-head a:hover {{ color: var(--accent); }}
              .item-meta {{ display: block; color: var(--muted); font-size: 0.82rem; margin-top: 4px; }}
              .errors, .empty-state {{ margin-top: 28px; padding: 18px; }}
              @media (max-width: 720px) {{
                .shell {{ padding: 20px 14px 48px; }}
                .feed-head {{ flex-direction: column; }}
              }}
            </style>
          </head>
          <body>
            <main class="shell">
              <section class="hero">
                <h1>{escape(report['site_title'])}</h1>
                <p>Daily AI TLDRs generated from your NetNewsWire OPML export and published with GitHub Actions.</p>
              </section>

              <section class="stats">
                <article class="stat-card">
                  <p class="stat-label">Updated</p>
                  <p class="stat-value">{escape(report['generated_at_human'])}</p>
                </article>
                <article class="stat-card">
                  <p class="stat-label">Feeds With Updates</p>
                  <p class="stat-value">{report['feeds_with_updates']}</p>
                </article>
                <article class="stat-card">
                  <p class="stat-label">Items Summarized</p>
                  <p class="stat-value">{report['total_items']}</p>
                </article>
                <article class="stat-card">
                  <p class="stat-label">Model</p>
                  <p class="stat-value">{escape(report['model'])}</p>
                </article>
              </section>

              {''.join(cards)}
              {errors_html}
            </main>
          </body>
        </html>
        """
    ).strip() + "\n"


def render_item(item: dict[str, Any]) -> str:
    meta_bits = []
    if item.get("published"):
        meta_bits.append(item["published"])
    if item.get("source_kind"):
        meta_bits.append(item["source_kind"])
    meta = " | ".join(meta_bits)
    title = escape(item["title"])
    if item.get("link"):
        label = f'<a href="{escape(item["link"])}">{title}</a>'
    else:
        label = title
    meta_html = f'<span class="item-meta">{escape(meta)}</span>' if meta else ""
    return f"<li>{label}{meta_html}</li>"


def ordered_categories(feeds: list[dict[str, Any]]) -> list[str]:
    categories: list[str] = []
    for feed in feeds:
        if feed["category"] not in categories:
            categories.append(feed["category"])
    return categories


def trim_unique(values: list[str], limit: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
        if len(output) >= limit:
            break
    return output


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def human_timestamp(raw_timestamp: str) -> str:
    return datetime.fromisoformat(raw_timestamp).strftime("%Y-%m-%d %H:%M UTC")


def clip_text(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    normalized = normalize_text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "…"


def normalize_text(text: str) -> str:
    no_tags = re.sub(r"<[^>]+>", " ", text or "")
    compact = re.sub(r"\s+", " ", unescape(no_tags))
    return compact.strip()
