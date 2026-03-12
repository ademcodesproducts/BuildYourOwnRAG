"""Tests for cleaner/bs4_extractor.py."""

import pytest
from cleaner.bs4_extractor import extract_with_bs4

# ---------------------------------------------------------------------------
# Fixture HTML snippets
# ---------------------------------------------------------------------------

FULL_PAGE_HTML = """
<html>
<head><title>Research Areas | EECS at UC Berkeley</title></head>
<body>
  <header id="site-header"><a href="/">Home</a></header>
  <nav class="main-nav"><ul><li><a href="/about">About</a></li></ul></nav>
  <main id="content">
    <h1>Research Areas</h1>
    <p>The EECS department covers a broad range of research topics.</p>
    <p>These include computer vision, NLP, and systems.</p>
  </main>
  <aside class="sidebar"><p>Related links</p></aside>
  <footer id="site-footer"><p>Copyright UC Berkeley</p></footer>
  <script>alert("hello")</script>
  <style>body { color: red; }</style>
</body>
</html>
"""

MINIMAL_HTML = """
<html>
<head><title>Simple Page</title></head>
<body>
  <p>This is the only content on the page.</p>
</body>
</html>
"""

ENCODING_HTML = """
<html>
<head><title>Unicode Page</title></head>
<body>
  <main>
    <p>Ünïcödé chäracters: 你好世界 — café résumé naïve</p>
  </main>
</body>
</html>
"""

NO_TITLE_HTML = """
<html>
<body><p>Some content without a title tag.</p></body>
</html>
"""

EMPTY_HTML = "<html><body></body></html>"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBs4Extractor:
    def test_title_extracted(self):
        result = extract_with_bs4(FULL_PAGE_HTML)
        assert result["title"] == "Research Areas | EECS at UC Berkeley"

    def test_main_content_present(self):
        result = extract_with_bs4(FULL_PAGE_HTML)
        assert "EECS department" in result["text"]
        assert "computer vision" in result["text"]

    def test_nav_excluded(self):
        result = extract_with_bs4(FULL_PAGE_HTML)
        assert "main-nav" not in result["text"]
        # The nav link text itself should not appear in isolation
        # (it may appear in page content, so check the nav marker)

    def test_footer_excluded(self):
        result = extract_with_bs4(FULL_PAGE_HTML)
        assert "Copyright UC Berkeley" not in result["text"]

    def test_script_excluded(self):
        result = extract_with_bs4(FULL_PAGE_HTML)
        assert "alert" not in result["text"]

    def test_style_excluded(self):
        result = extract_with_bs4(FULL_PAGE_HTML)
        assert "color: red" not in result["text"]

    def test_sidebar_excluded(self):
        result = extract_with_bs4(FULL_PAGE_HTML)
        assert "Related links" not in result["text"]

    def test_unicode_preserved(self):
        result = extract_with_bs4(ENCODING_HTML)
        assert "你好世界" in result["text"]
        assert "café" in result["text"]

    def test_no_title_returns_empty_string(self):
        result = extract_with_bs4(NO_TITLE_HTML)
        assert result["title"] == ""
        assert "Some content" in result["text"]

    def test_empty_body_returns_empty_text(self):
        result = extract_with_bs4(EMPTY_HTML)
        assert result["text"] == ""

    def test_extractor_key_is_bs4(self):
        result = extract_with_bs4(MINIMAL_HTML)
        assert result["extractor"] == "bs4"

    def test_minimal_page_content_extracted(self):
        result = extract_with_bs4(MINIMAL_HTML)
        assert "only content" in result["text"]
