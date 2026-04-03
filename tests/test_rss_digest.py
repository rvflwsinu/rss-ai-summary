from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import feedparser

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rss_digest


class TestStateMigration(unittest.TestCase):
    def test_migrate_state_from_latest_backfills_pending_items(self) -> None:
        state = {
            "version": 1,
            "feeds": {
                "https://example.com/feed.xml": {
                    "seen_ids": ["legacy-seen-id"],
                }
            },
        }
        latest_report = {
            "feeds": [
                {
                    "title": "Example Feed",
                    "site_url": "https://example.com",
                    "feed_url": "https://example.com/feed.xml",
                    "tldr": "Existing summary",
                    "highlights": ["Hello world"],
                    "items": [
                        {
                            "title": "Hello world",
                            "link": "https://example.com/hello",
                            "published": "2026-04-02 13:00 UTC",
                            "source_kind": "feed",
                        }
                    ],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            latest_path = Path(tmpdir) / "latest.json"
            latest_path.write_text(
                json.dumps(latest_report, ensure_ascii=False),
                encoding="utf-8",
            )

            rss_digest.migrate_state_from_latest(state, latest_path)

        feed_state = state["feeds"]["https://example.com/feed.xml"]
        self.assertEqual(feed_state["feed_title"], "Example Feed")
        self.assertEqual(feed_state["site_url"], "https://example.com")
        self.assertEqual(feed_state["summary"]["tldr"], "Existing summary")
        self.assertEqual(len(feed_state["pending_items"]), 1)
        self.assertEqual(feed_state["pending_items"][0]["title"], "Hello world")
        self.assertEqual(feed_state["pending_items"][0]["text"], "Hello world")


class TestPersistentBacklog(unittest.TestCase):
    def setUp(self) -> None:
        self.feed = rss_digest.FeedConfig(
            category="Tech",
            title="Example Feed",
            xml_url="https://example.com/feed.xml",
            html_url="https://example.com",
        )
        self.config = {
            "min_feed_content_chars": 50,
            "max_item_chars": 4000,
            "mock_summary": False,
        }

    def test_process_feed_appends_new_items_without_dropping_existing_backlog(self) -> None:
        existing_item = {
            "id": "existing-item-id",
            "signature": rss_digest.item_signature(
                self.feed.xml_url,
                title="Existing item",
                link="https://example.com/existing",
                published="2026-04-01 00:00 UTC",
            ),
            "title": "Existing item",
            "link": "https://example.com/existing",
            "published": "2026-04-01 00:00 UTC",
            "sort_key": rss_digest.rendered_timestamp_sort_key("2026-04-01 00:00 UTC"),
            "source_kind": "feed",
            "text": "Existing item body",
        }
        state = {
            "feeds": {
                self.feed.xml_url: {
                    "pending_items": [existing_item],
                    "summary": {
                        "tldr": "Old summary",
                        "highlights": ["Existing item"],
                    },
                    "seen_ids": [],
                }
            }
        }
        parsed_feed = feedparser.FeedParserDict(
            {
                "feed": {"title": "Example Feed", "link": "https://example.com"},
                "entries": [
                    feedparser.FeedParserDict(
                        {
                            "id": "new-entry-id",
                            "title": "New item",
                            "link": "https://example.com/new",
                            "published_parsed": time.strptime(
                                "2026-04-03 00:00:00",
                                "%Y-%m-%d %H:%M:%S",
                            ),
                            "summary": "New item body",
                        }
                    )
                ],
            }
        )

        with (
            mock.patch.object(rss_digest, "fetch_feed", return_value=parsed_feed),
            mock.patch.object(
                rss_digest,
                "resolve_entry_text",
                return_value=("New item body", "feed"),
            ),
            mock.patch.object(
                rss_digest,
                "summarize_feed",
                return_value={
                    "tldr": "Updated summary",
                    "highlights": ["New item", "Existing item"],
                },
            ),
        ):
            result = rss_digest.process_feed(mock.Mock(), self.config, state, self.feed)

        self.assertIsNone(result["error"])
        self.assertEqual(result["summary"]["item_count"], 2)
        self.assertEqual(
            [item["title"] for item in result["summary"]["items"]],
            ["New item", "Existing item"],
        )
        feed_state = state["feeds"][self.feed.xml_url]
        self.assertEqual(len(feed_state["pending_items"]), 2)
        self.assertEqual(feed_state["summary"]["tldr"], "Updated summary")

    def test_process_feed_keeps_existing_backlog_when_feed_fetch_fails(self) -> None:
        state = {
            "feeds": {
                self.feed.xml_url: {
                    "pending_items": [
                        {
                            "id": "existing-item-id",
                            "signature": rss_digest.item_signature(
                                self.feed.xml_url,
                                title="Existing item",
                                link="https://example.com/existing",
                                published="2026-04-01 00:00 UTC",
                            ),
                            "title": "Existing item",
                            "link": "https://example.com/existing",
                            "published": "2026-04-01 00:00 UTC",
                            "sort_key": rss_digest.rendered_timestamp_sort_key(
                                "2026-04-01 00:00 UTC"
                            ),
                            "source_kind": "feed",
                            "text": "Existing item body",
                        }
                    ],
                    "summary": {
                        "tldr": "Stored summary",
                        "highlights": ["Existing item"],
                    },
                    "feed_title": "Example Feed",
                    "site_url": "https://example.com",
                }
            }
        }

        with mock.patch.object(rss_digest, "fetch_feed", side_effect=ValueError("boom")):
            result = rss_digest.process_feed(mock.Mock(), self.config, state, self.feed)

        self.assertIsNotNone(result["summary"])
        self.assertIn("Feed fetch failed", result["error"])
        self.assertEqual(result["summary"]["item_count"], 1)
        self.assertEqual(result["summary"]["tldr"], "Stored summary")


class TestOpenRouterIntegration(unittest.TestCase):
    def test_load_runtime_config_defaults_to_qwen_openrouter_model(self) -> None:
        args = argparse.Namespace(
            opml_path="subscriptions.opml",
            state_path="data/state.json",
            latest_path="data/latest.json",
            site_dir="site",
            mock_summary=False,
        )

        with mock.patch.dict(rss_digest.os.environ, {}, clear=True):
            config = rss_digest.load_runtime_config(args)

        self.assertEqual(config["llm_provider"], "openrouter")
        self.assertEqual(config["openrouter_model"], "qwen/qwen3.6-plus:free")

    def test_call_llm_json_uses_openrouter_provider(self) -> None:
        config = {
            "llm_provider": "openrouter",
            "openrouter_api_key": "router-key",
            "openrouter_model": "qwen/qwen3.6-plus:free",
        }

        with mock.patch.object(
            rss_digest,
            "call_openrouter_json",
            return_value={"tldr": "Summary", "highlights": ["One"]},
        ) as call_openrouter:
            result = rss_digest.call_llm_json(config=config, prompt="Summarize this feed")

        self.assertEqual(result, {"tldr": "Summary", "highlights": ["One"]})
        call_openrouter.assert_called_once_with(
            api_key="router-key",
            model="qwen/qwen3.6-plus:free",
            prompt="Summarize this feed",
        )

    def test_call_openrouter_json_requests_json_mode_and_parses_output_text(self) -> None:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "reasoning", "text": "hidden reasoning"},
                            {
                                "type": "output_text",
                                "text": '{"tldr": "Summary", "highlights": ["One"]}',
                            },
                        ]
                    }
                }
            ]
        }

        with mock.patch.object(rss_digest.httpx, "post", return_value=response) as post:
            result = rss_digest.call_openrouter_json(
                api_key="test-key",
                model="qwen/qwen3.6-plus:free",
                prompt="Summarize this feed",
            )

        self.assertEqual(result, {"tldr": "Summary", "highlights": ["One"]})
        self.assertEqual(
            post.call_args.kwargs["json"]["response_format"],
            {"type": "json_object"},
        )
        self.assertEqual(
            post.call_args.kwargs["headers"]["X-OpenRouter-Title"],
            "rss-ai-summary",
        )

    def test_process_feed_raises_on_openrouter_auth_error(self) -> None:
        feed = rss_digest.FeedConfig(
            category="Research",
            title="Paper Feed",
            xml_url="https://example.com/feed.xml",
            html_url="https://example.com",
        )
        config = {
            "min_feed_content_chars": 50,
            "max_item_chars": 4000,
            "mock_summary": False,
            "llm_provider": "openrouter",
        }
        state = {"feeds": {feed.xml_url: {"pending_items": [], "seen_ids": []}}}
        parsed_feed = feedparser.FeedParserDict(
            {
                "feed": {"title": "Paper Feed", "link": "https://example.com"},
                "entries": [
                    feedparser.FeedParserDict(
                        {
                            "id": "new-entry-id",
                            "title": "New paper",
                            "link": "https://example.com/paper",
                            "published_parsed": time.strptime(
                                "2026-04-03 00:00:00",
                                "%Y-%m-%d %H:%M:%S",
                            ),
                            "summary": "Paper body",
                        }
                    )
                ],
            }
        )
        response = mock.Mock(status_code=401, reason_phrase="Unauthorized")
        auth_error = rss_digest.httpx.HTTPStatusError(
            "401 Unauthorized",
            request=mock.Mock(),
            response=response,
        )

        with (
            mock.patch.object(rss_digest, "fetch_feed", return_value=parsed_feed),
            mock.patch.object(
                rss_digest,
                "resolve_entry_text",
                return_value=("Paper body", "feed"),
            ),
            mock.patch.object(rss_digest, "summarize_feed", side_effect=auth_error),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "openrouter authentication failed: HTTP 401 Unauthorized",
            ):
                rss_digest.process_feed(mock.Mock(), config, state, feed)


class TestRenderSite(unittest.TestCase):
    def test_render_site_shows_linked_title_followed_by_tldr(self) -> None:
        report = {
            "site_title": "Daily Feed TLDR",
            "generated_at_human": "2026-04-03 12:00 UTC",
            "model": "qwen/qwen3.6-plus:free",
            "feeds_with_updates": 1,
            "total_items": 2,
            "categories": ["Research"],
            "feeds": [
                {
                    "category": "Research",
                    "title": "Paper Feed",
                    "site_url": "https://example.com/papers",
                    "feed_url": "https://example.com/feed.xml",
                    "item_count": 2,
                    "tldr": "A short summary of the latest papers.",
                    "highlights": ["Paper A", "Paper B"],
                    "items": [
                        {"title": "Paper A", "link": "https://example.com/a"},
                        {"title": "Paper B", "link": "https://example.com/b"},
                    ],
                }
            ],
            "errors": [],
        }

        html = rss_digest.render_site(report)

        self.assertIn(
            '<div class="feed-title"><a href="https://example.com/papers">Paper Feed</a></div>',
            html,
        )
        self.assertIn('<p class="tldr">A short summary of the latest papers.</p>', html)
        self.assertIn('<div class="feed-list">', html)
        self.assertNotIn("<details>", html)


if __name__ == "__main__":
    unittest.main()
