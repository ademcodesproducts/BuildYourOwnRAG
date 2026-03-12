"""
Tests for crawler/fetcher.py — uses `responses` library to mock HTTP.
"""

import time
import pytest
import responses as responses_lib
from unittest.mock import MagicMock

from crawler.fetcher import Fetcher, FetchResult
from crawler.robots import RobotsCache


def _make_robots(allowed: bool = True, crawl_delay: float | None = None) -> RobotsCache:
    """Return a RobotsCache mock that allows or disallows all URLs."""
    robots = MagicMock(spec=RobotsCache)
    robots.is_allowed.return_value = allowed
    robots.get_crawl_delay.return_value = crawl_delay
    return robots


@responses_lib.activate
def test_fetch_html_success():
    responses_lib.add(
        responses_lib.GET,
        "https://eecs.berkeley.edu/page",
        body="<html><body>Hello</body></html>",
        status=200,
        content_type="text/html; charset=utf-8",
    )
    fetcher = Fetcher(crawl_delay=0.0)
    result = fetcher.fetch("https://eecs.berkeley.edu/page", _make_robots())
    assert result.error is None
    assert result.html == "<html><body>Hello</body></html>"
    assert result.status_code == 200


@responses_lib.activate
def test_fetch_non_html_returns_error():
    responses_lib.add(
        responses_lib.GET,
        "https://eecs.berkeley.edu/doc.pdf",
        body=b"%PDF-1.4",
        status=200,
        content_type="application/pdf",
    )
    fetcher = Fetcher(crawl_delay=0.0)
    result = fetcher.fetch("https://eecs.berkeley.edu/doc.pdf", _make_robots())
    assert result.html is None
    assert "Non-HTML" in result.error


def test_fetch_disallowed_by_robots():
    fetcher = Fetcher(crawl_delay=0.0)
    result = fetcher.fetch("https://eecs.berkeley.edu/private", _make_robots(allowed=False))
    assert result.html is None
    assert "robots" in result.error.lower()


@responses_lib.activate
def test_fetch_404():
    responses_lib.add(
        responses_lib.GET,
        "https://eecs.berkeley.edu/missing",
        body="<html>Not Found</html>",
        status=404,
        content_type="text/html",
    )
    fetcher = Fetcher(crawl_delay=0.0)
    result = fetcher.fetch("https://eecs.berkeley.edu/missing", _make_robots())
    # 404 with text/html body is still returned as HTML (let caller decide)
    assert result.status_code == 404
    assert result.html is not None


@responses_lib.activate
def test_fetch_timeout_exhausts_retries():
    import requests as req
    responses_lib.add(
        responses_lib.GET,
        "https://eecs.berkeley.edu/slow",
        body=req.Timeout(),
    )
    responses_lib.add(
        responses_lib.GET,
        "https://eecs.berkeley.edu/slow",
        body=req.Timeout(),
    )
    responses_lib.add(
        responses_lib.GET,
        "https://eecs.berkeley.edu/slow",
        body=req.Timeout(),
    )
    fetcher = Fetcher(crawl_delay=0.0, max_retries=3, backoff_base=0.0)
    result = fetcher.fetch("https://eecs.berkeley.edu/slow", _make_robots())
    assert result.html is None
    assert result.error is not None


@responses_lib.activate
def test_rate_limit_enforced_per_domain():
    """Two requests to the same domain should be separated by at least crawl_delay."""
    for _ in range(2):
        responses_lib.add(
            responses_lib.GET,
            "https://eecs.berkeley.edu/page",
            body="<html></html>",
            status=200,
            content_type="text/html",
        )
    delay = 0.1
    fetcher = Fetcher(crawl_delay=delay)
    robots = _make_robots()

    t0 = time.monotonic()
    fetcher.fetch("https://eecs.berkeley.edu/page", robots)
    fetcher.fetch("https://eecs.berkeley.edu/page", robots)
    elapsed = time.monotonic() - t0

    assert elapsed >= delay, f"Rate limit not enforced: elapsed={elapsed:.3f}s < delay={delay}s"


@responses_lib.activate
def test_redirect_url_captured():
    """The final post-redirect URL should be stored in result.url."""
    responses_lib.add(
        responses_lib.GET,
        "https://eecs.berkeley.edu/old",
        status=301,
        headers={"Location": "https://eecs.berkeley.edu/new"},
    )
    responses_lib.add(
        responses_lib.GET,
        "https://eecs.berkeley.edu/new",
        body="<html>New page</html>",
        status=200,
        content_type="text/html",
    )
    fetcher = Fetcher(crawl_delay=0.0)
    result = fetcher.fetch("https://eecs.berkeley.edu/old", _make_robots())
    assert result.original_url == "https://eecs.berkeley.edu/old"
    assert "new" in result.url
