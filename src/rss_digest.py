from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import textwrap
import time
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
DEFAULT_LLM_PROVIDER = "openrouter"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash-lite"
DEFAULT_OPENROUTER_MODEL = "qwen/qwen3.6-plus:free"
DEFAULT_SUMMARY_LANGUAGE = "English"
LLM_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
LLM_MAX_ATTEMPTS = 4


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
        help="Path to the persisted backlog state JSON file.",
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
        help="Skip the configured LLM and synthesize summaries from the feed titles for local testing.",
    )
    return parser.parse_args()


def load_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "opml_path": Path(args.opml_path),
        "state_path": Path(args.state_path),
        "latest_path": Path(args.latest_path),
        "site_dir": Path(args.site_dir),
        "mock_summary": args.mock_summary,
        "llm_provider": os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()
        or DEFAULT_LLM_PROVIDER,
        "gemini_api_key": os.getenv("GEMINI_API_KEY", "").strip(),
        "gemini_model": os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
        or DEFAULT_GEMINI_MODEL,
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", "").strip(),
        "openrouter_model": os.getenv(
            "OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL
        ).strip()
        or DEFAULT_OPENROUTER_MODEL,
        "summary_language": os.getenv("SUMMARY_LANGUAGE", DEFAULT_SUMMARY_LANGUAGE).strip()
        or DEFAULT_SUMMARY_LANGUAGE,
        "site_title": os.getenv("SITE_TITLE", "Daily Feed TLDR").strip() or "Daily Feed TLDR",
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
    validate_llm_config(config)

    feeds = parse_opml(config["opml_path"])
    state = load_state(config["state_path"])
    migrate_state_from_latest(state, config["latest_path"])
    generated_at = utc_now_iso()

    report: dict[str, Any] = {
        "site_title": config["site_title"],
        "generated_at": generated_at,
        "generated_at_human": human_timestamp(generated_at),
        "model": active_model_name(config),
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
    feed_state = state.setdefault("feeds", {}).setdefault(feed.xml_url, {})
    pending_items = load_pending_items(feed_state, feed.xml_url)
    stored_title = feed_state.get("feed_title") or feed.title
    stored_site_url = feed_state.get("site_url") or feed.html_url or feed.xml_url
    stored_feed = FeedConfig(
        category=feed.category,
        title=stored_title,
        xml_url=feed.xml_url,
        html_url=feed.html_url,
    )
    try:
        parsed_feed = fetch_feed(client, feed.xml_url)
    except Exception as exc:
        if not pending_items:
            return {"summary": None, "error": f"Feed fetch failed: {safe_error_message(exc)}"}
        stored_summary = summary_from_state(feed_state, stored_feed, pending_items)
        return {
            "summary": build_feed_digest(
                feed=stored_feed,
                site_url=stored_site_url,
                items=pending_items,
                summary=stored_summary,
            ),
            "error": f"Feed fetch failed: {safe_error_message(exc)}",
        }

    seen_ids = list(feed_state.get("seen_ids", []))
    seen_lookup = set(seen_ids)
    normalized_entries = normalize_entries(parsed_feed, feed)
    pending_signatures = {item["signature"] for item in pending_items}

    new_entries = [
        entry
        for entry in normalized_entries
        if entry["id"] not in seen_lookup and entry["signature"] not in pending_signatures
    ]
    feed_title = parsed_feed.feed.get("title") or feed.title
    display_feed = FeedConfig(
        category=feed.category,
        title=feed_title,
        xml_url=feed.xml_url,
        html_url=feed.html_url,
    )
    site_url = feed.html_url or parsed_feed.feed.get("link") or feed.xml_url

    new_pending_items = []
    for entry in new_entries:
        source_text, source_kind = resolve_entry_text(
            client,
            entry,
            min_feed_content_chars=config["min_feed_content_chars"],
            max_item_chars=config["max_item_chars"],
        )
        new_pending_items.append(
            {
                "id": entry["id"],
                "signature": entry["signature"],
                "title": entry["title"],
                "link": entry["link"],
                "published": entry["published"],
                "sort_key": entry["sort_key"],
                "source_kind": source_kind,
                "text": source_text,
            }
        )

    pending_items = merge_pending_items(feed.xml_url, pending_items, new_pending_items)

    feed_state["pending_items"] = pending_items
    feed_state["seen_ids"] = unique_values([entry["id"] for entry in new_entries] + seen_ids)
    feed_state["last_checked_at"] = utc_now_iso()
    feed_state["feed_title"] = feed_title
    feed_state["site_url"] = site_url

    if not pending_items:
        return {"summary": None, "error": None}

    summary_inputs = [pending_item_summary_input(item) for item in pending_items]

    summary_error = None
    try:
        summary = summarize_feed(config, display_feed, summary_inputs)
    except Exception as exc:
        summary = fallback_summary(display_feed, pending_items)
        summary_error = f"LLM summary failed, used fallback: {safe_error_message(exc)}"

    feed_state["summary"] = summary

    return {
        "summary": build_feed_digest(
            feed=display_feed,
            site_url=site_url,
            items=pending_items,
            summary=summary,
        ),
        "error": summary_error,
    }


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
                "signature": item_signature(
                    feed.xml_url,
                    title=normalize_text(entry.get("title") or "Untitled item"),
                    link=entry.get("link"),
                    published=timestamp.strftime("%Y-%m-%d %H:%M UTC") if timestamp else None,
                ),
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


def item_signature(
    feed_url: str,
    *,
    title: str,
    link: str | None,
    published: str | None,
) -> str:
    basis = "|".join([feed_url, link or "", title or "", published or ""])
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

    response_data = call_llm_json(config=config, prompt=prompt)
    tldr = normalize_text(str(response_data.get("tldr", "")))
    highlights = [
        normalize_text(str(item))
        for item in response_data.get("highlights", [])
        if normalize_text(str(item))
    ]
    if not tldr:
        raise ValueError("LLM response did not contain a TLDR.")
    if not highlights:
        highlights = [item["title"] for item in items[:3]]
    return {"tldr": tldr, "highlights": highlights[:5]}


def validate_llm_config(config: dict[str, Any]) -> None:
    if config["mock_summary"]:
        return

    provider = config["llm_provider"]
    if provider == "gemini":
        if not config["gemini_api_key"]:
            raise ValueError(
                "GEMINI_API_KEY is required when LLM_PROVIDER=gemini unless --mock-summary is enabled."
            )
        return
    if provider == "openrouter":
        if not config["openrouter_api_key"]:
            raise ValueError(
                "OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter unless --mock-summary is enabled."
            )
        return
    raise ValueError(
        f"LLM_PROVIDER must be one of: gemini, openrouter. Got: {provider!r}"
    )


def migrate_state_from_latest(state: dict[str, Any], latest_path: Path) -> None:
    if not latest_path.exists():
        return

    try:
        latest_report = json.loads(latest_path.read_text(encoding="utf-8"))
    except Exception:
        return

    for feed in latest_report.get("feeds", []):
        feed_url = feed.get("feed_url")
        if not feed_url:
            continue

        feed_state = state.setdefault("feeds", {}).setdefault(feed_url, {})
        if feed_state.get("pending_items"):
            continue

        pending_items = [
            migrated_pending_item(feed_url, item) for item in feed.get("items", [])
        ]
        if not pending_items:
            continue

        feed_state["pending_items"] = pending_items
        if feed.get("title"):
            feed_state["feed_title"] = feed["title"]
        if feed.get("site_url"):
            feed_state["site_url"] = feed["site_url"]

        tldr = normalize_text(str(feed.get("tldr", "")))
        highlights = [
            normalize_text(str(item))
            for item in feed.get("highlights", [])
            if normalize_text(str(item))
        ]
        if tldr:
            feed_state["summary"] = {
                "tldr": tldr,
                "highlights": highlights or [item["title"] for item in pending_items[:3]],
            }


def migrated_pending_item(feed_url: str, item: dict[str, Any]) -> dict[str, Any]:
    title = normalize_text(str(item.get("title") or "Untitled item"))
    published = item.get("published")
    signature = item_signature(
        feed_url,
        title=title,
        link=item.get("link"),
        published=published,
    )
    return {
        "id": f"migrated:{signature}",
        "signature": signature,
        "title": title,
        "link": item.get("link"),
        "published": published,
        "sort_key": rendered_timestamp_sort_key(published),
        "source_kind": item.get("source_kind"),
        "text": title,
    }


def load_pending_items(feed_state: dict[str, Any], feed_url: str) -> list[dict[str, Any]]:
    pending_items = [
        normalize_pending_item(feed_url, item)
        for item in feed_state.get("pending_items", [])
        if isinstance(item, dict)
    ]
    pending_items.sort(key=lambda item: item["sort_key"], reverse=True)
    return pending_items


def normalize_pending_item(feed_url: str, item: dict[str, Any]) -> dict[str, Any]:
    title = normalize_text(str(item.get("title") or "Untitled item"))
    published = item.get("published")
    sort_key = item.get("sort_key")
    if not isinstance(sort_key, (int, float)):
        sort_key = rendered_timestamp_sort_key(published)

    signature = item.get("signature")
    if not signature:
        signature = item_signature(
            feed_url,
            title=title,
            link=item.get("link"),
            published=published,
        )

    item_id = item.get("id") or f"pending:{signature}"
    return {
        "id": str(item_id),
        "signature": str(signature),
        "title": title,
        "link": item.get("link"),
        "published": published,
        "sort_key": float(sort_key),
        "source_kind": item.get("source_kind"),
        "text": normalize_text(str(item.get("text") or title)),
    }


def merge_pending_items(
    feed_url: str,
    existing_items: list[dict[str, Any]],
    new_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in new_items + existing_items:
        normalized = normalize_pending_item(feed_url, item)
        merged.setdefault(normalized["signature"], normalized)
    return sorted(merged.values(), key=lambda item: item["sort_key"], reverse=True)


def pending_item_summary_input(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": item["title"],
        "link": item.get("link"),
        "published": item.get("published"),
        "text": item.get("text") or item["title"],
    }


def pending_item_output(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": item["title"],
        "link": item.get("link"),
        "published": item.get("published"),
        "source_kind": item.get("source_kind"),
    }


def summary_from_state(
    feed_state: dict[str, Any],
    feed: FeedConfig,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    stored_summary = feed_state.get("summary", {})
    tldr = normalize_text(str(stored_summary.get("tldr", "")))
    highlights = [
        normalize_text(str(item))
        for item in stored_summary.get("highlights", [])
        if normalize_text(str(item))
    ]
    if tldr:
        return {
            "tldr": tldr,
            "highlights": highlights or [item["title"] for item in items[:3]],
        }
    return fallback_summary(feed, items)


def build_feed_digest(
    *,
    feed: FeedConfig,
    site_url: str,
    items: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    output_items = [pending_item_output(item) for item in items]
    return {
        "category": feed.category,
        "title": feed.title,
        "site_url": site_url,
        "feed_url": feed.xml_url,
        "item_count": len(output_items),
        "skipped_count": 0,
        "tldr": summary["tldr"],
        "highlights": summary["highlights"],
        "items": output_items,
    }


def active_model_name(config: dict[str, Any]) -> str:
    if config["mock_summary"]:
        return "mock-summary"
    if config["llm_provider"] == "gemini":
        return config["gemini_model"]
    return config["openrouter_model"]


def call_llm_json(*, config: dict[str, Any], prompt: str) -> dict[str, Any]:
    if config["llm_provider"] == "gemini":
        return call_gemini_json(
            api_key=config["gemini_api_key"],
            model=config["gemini_model"],
            prompt=prompt,
        )
    if config["llm_provider"] == "openrouter":
        return call_openrouter_json(
            api_key=config["openrouter_api_key"],
            model=config["openrouter_model"],
            prompt=prompt,
        )
    raise ValueError(f"Unsupported LLM provider: {config['llm_provider']}")


def call_gemini_json(*, api_key: str, model: str, prompt: str) -> dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        try:
            response = httpx.post(
                url,
                params={"key": api_key},
                json=payload,
                headers={"User-Agent": USER_AGENT},
                timeout=60.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if (
                status_code not in LLM_RETRYABLE_STATUS_CODES
                or attempt == LLM_MAX_ATTEMPTS
            ):
                raise
        except httpx.HTTPError:
            if attempt == LLM_MAX_ATTEMPTS:
                raise
        else:
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

        # Back off on rate limits and transient upstream errors.
        time.sleep(min(20, 5 * attempt))

    raise RuntimeError("Gemini request retries exhausted unexpectedly.")


def call_openrouter_json(*, api_key: str, model: str, prompt: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/zenityann-dot/rss-ai-summary",
        "X-OpenRouter-Title": "rss-ai-summary",
        "User-Agent": USER_AGENT,
    }
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        try:
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=60.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if (
                status_code not in LLM_RETRYABLE_STATUS_CODES
                or attempt == LLM_MAX_ATTEMPTS
            ):
                raise
        except httpx.HTTPError:
            if attempt == LLM_MAX_ATTEMPTS:
                raise
        else:
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise ValueError("OpenRouter returned no choices.")
            text = openrouter_message_text(choices[0].get("message", {}).get("content"))
            if not text:
                raise ValueError("OpenRouter returned an empty response body.")
            return json.loads(strip_code_fences(text))

        time.sleep(min(20, 5 * attempt))

    raise RuntimeError("OpenRouter request retries exhausted unexpectedly.")


def openrouter_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if isinstance(content, list):
        chunks = []
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if not text:
                continue
            if part.get("type") in {None, "text", "output_text"}:
                chunks.append(str(text))
        return "\n".join(chunks).strip()
    return ""


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def safe_error_message(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code} {exc.response.reason_phrase}"
    return redact_sensitive_text(str(exc))


def redact_sensitive_text(text: str) -> str:
    return re.sub(r"([?&]key=)[^&\s'\"]+", r"\1[REDACTED]", text or "")


def fallback_summary(feed: FeedConfig, items: list[dict[str, Any]]) -> dict[str, Any]:
    highlight_titles = [item["title"] for item in items[:3] if item.get("title")]
    if not highlight_titles:
        highlight_titles = [f"Tracked items are available in {feed.title}."]
    tldr = (
        f"{feed.title} currently has {len(items)} retained item(s) on the page. "
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
                category_cards.append(
                    f"""
                    <article class="feed-card">
                      <div class="feed-head">
                        <div>
                          <h3><a href="{escape(feed['site_url'])}">{escape(feed['title'])}</a></h3>
                          <p class="meta">{escape(feed['item_count'].__str__())} retained item(s)</p>
                        </div>
                        <a class="rss-link" href="{escape(feed['feed_url'])}">RSS</a>
                      </div>
                      <p class="tldr">{escape(feed['tldr'])}</p>
                      <ul class="highlights">{''.join(f'<li>{escape(point)}</li>' for point in feed['highlights'])}</ul>
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
              <h2>No retained items</h2>
              <p>The workflow completed, but there are no items currently being kept on the page.</p>
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
                  <p class="stat-label">Feeds On Page</p>
                  <p class="stat-value">{report['feeds_with_updates']}</p>
                </article>
                <article class="stat-card">
                  <p class="stat-label">Items On Page</p>
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


def unique_values(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def rendered_timestamp_sort_key(raw_timestamp: str | None) -> float:
    if not raw_timestamp:
        return 0
    try:
        return datetime.strptime(raw_timestamp, "%Y-%m-%d %H:%M UTC").replace(
            tzinfo=timezone.utc
        ).timestamp()
    except ValueError:
        return 0


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
