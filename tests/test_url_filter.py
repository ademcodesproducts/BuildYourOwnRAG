"""Tests for crawler/url_filter.py — pure logic, no I/O."""

import pytest
from crawler.url_filter import (
    normalize_url,
    is_allowed_domain,
    is_html_url,
    should_crawl,
)

ALLOWED = {"eecs.berkeley.edu", "www2.eecs.berkeley.edu", "cs.berkeley.edu"}


# ---------------------------------------------------------------------------
# normalize_url
# ---------------------------------------------------------------------------

class TestNormalizeUrl:
    def test_strips_fragment(self):
        assert normalize_url("https://eecs.berkeley.edu/page#section") == \
               normalize_url("https://eecs.berkeley.edu/page")

    def test_lowercases_scheme_and_host(self):
        assert normalize_url("HTTPS://EECS.BERKELEY.EDU/page") == \
               normalize_url("https://eecs.berkeley.edu/page")

    def test_strips_trailing_slash(self):
        assert normalize_url("https://eecs.berkeley.edu/research/") == \
               normalize_url("https://eecs.berkeley.edu/research")

    def test_root_path_preserved(self):
        # Root "/" should not be stripped to ""
        norm = normalize_url("https://eecs.berkeley.edu/")
        assert norm == "https://eecs.berkeley.edu/"

    def test_sorts_query_params(self):
        a = normalize_url("https://eecs.berkeley.edu/search?b=2&a=1")
        b = normalize_url("https://eecs.berkeley.edu/search?a=1&b=2")
        assert a == b

    def test_different_paths_not_equal(self):
        assert normalize_url("https://eecs.berkeley.edu/foo") != \
               normalize_url("https://eecs.berkeley.edu/bar")


# ---------------------------------------------------------------------------
# is_allowed_domain
# ---------------------------------------------------------------------------

class TestIsAllowedDomain:
    def test_exact_match(self):
        assert is_allowed_domain("https://eecs.berkeley.edu/page", ALLOWED)

    def test_www_prefix_ignored(self):
        assert is_allowed_domain("https://www.eecs.berkeley.edu/page", ALLOWED)

    def test_subdomain_of_allowed(self):
        # inst.eecs.berkeley.edu is a subdomain of eecs.berkeley.edu
        assert is_allowed_domain("https://inst.eecs.berkeley.edu/~cs189", ALLOWED)

    def test_out_of_domain_rejected(self):
        assert not is_allowed_domain("https://berkeley.edu/page", ALLOWED)

    def test_similar_domain_rejected(self):
        assert not is_allowed_domain("https://noeecs.berkeley.edu/page", ALLOWED)

    def test_external_site_rejected(self):
        assert not is_allowed_domain("https://google.com/", ALLOWED)


# ---------------------------------------------------------------------------
# is_html_url
# ---------------------------------------------------------------------------

class TestIsHtmlUrl:
    @pytest.mark.parametrize("url", [
        "https://eecs.berkeley.edu/research/areas/",
        "https://eecs.berkeley.edu/people/faculty",
        "https://www2.eecs.berkeley.edu/Courses/CS189/",
        "https://eecs.berkeley.edu/page.php",
        "https://eecs.berkeley.edu/page.html",
    ])
    def test_html_urls_accepted(self, url):
        assert is_html_url(url)

    @pytest.mark.parametrize("url", [
        "https://eecs.berkeley.edu/doc.pdf",
        "https://eecs.berkeley.edu/photo.jpg",
        "https://eecs.berkeley.edu/photo.JPEG",  # uppercase extension
        "https://eecs.berkeley.edu/style.css",
        "https://eecs.berkeley.edu/app.js",
        "https://eecs.berkeley.edu/data.zip",
        "https://eecs.berkeley.edu/slides.pptx",
    ])
    def test_non_html_urls_rejected(self, url):
        assert not is_html_url(url)


# ---------------------------------------------------------------------------
# should_crawl
# ---------------------------------------------------------------------------

class TestShouldCrawl:
    def test_fresh_allowed_url_accepted(self):
        seen = set()
        assert should_crawl("https://eecs.berkeley.edu/page", ALLOWED, seen)

    def test_already_seen_rejected(self):
        from crawler.url_filter import normalize_url
        url = "https://eecs.berkeley.edu/page"
        seen = {normalize_url(url)}
        assert not should_crawl(url, ALLOWED, seen)

    def test_fragment_variant_of_seen_rejected(self):
        from crawler.url_filter import normalize_url
        url = "https://eecs.berkeley.edu/page"
        seen = {normalize_url(url)}
        assert not should_crawl(url + "#section", ALLOWED, seen)

    def test_out_of_domain_rejected(self):
        seen = set()
        assert not should_crawl("https://google.com/", ALLOWED, seen)

    def test_pdf_url_rejected(self):
        seen = set()
        assert not should_crawl("https://eecs.berkeley.edu/doc.pdf", ALLOWED, seen)
