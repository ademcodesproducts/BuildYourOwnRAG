"""Tests for cleaner/resiliparse_extractor.py."""

import pytest
from cleaner.resiliparse_extractor import extract_with_resiliparse, is_available

FULL_PAGE_HTML = """
<html>
<head><title>Faculty | EECS at UC Berkeley</title></head>
<body>
  <nav><a href="/">Home</a></nav>
  <main>
    <h1>Faculty</h1>
    <p>Our faculty work on cutting-edge research in computer science and engineering.</p>
    <p>Areas include machine learning, security, and theoretical CS.</p>
  </main>
  <footer><p>UC Berkeley EECS</p></footer>
  <script>console.log("x")</script>
</body>
</html>
"""

MINIMAL_HTML = """
<html>
<head><title>Minimal</title></head>
<body><p>Just a short paragraph of content.</p></body>
</html>
"""


class TestResiliparseExtractor:
    def test_title_extracted(self):
        result = extract_with_resiliparse(FULL_PAGE_HTML)
        assert result["title"] == "Faculty | EECS at UC Berkeley"

    @pytest.mark.skipif(not is_available(), reason="resiliparse not installed")
    def test_main_content_present(self):
        result = extract_with_resiliparse(FULL_PAGE_HTML)
        assert "cutting-edge research" in result["text"]

    @pytest.mark.skipif(not is_available(), reason="resiliparse not installed")
    def test_script_excluded(self):
        result = extract_with_resiliparse(FULL_PAGE_HTML)
        assert 'console.log' not in result["text"]

    def test_extractor_key_is_resiliparse(self):
        result = extract_with_resiliparse(FULL_PAGE_HTML)
        assert result["extractor"] == "resiliparse"

    def test_empty_html_returns_empty_text(self):
        result = extract_with_resiliparse("<html><body></body></html>")
        assert result["text"] == ""

    def test_no_title_tag_returns_empty_title(self):
        html = "<html><body><p>Content without title.</p></body></html>"
        result = extract_with_resiliparse(html)
        assert result["title"] == ""

    @pytest.mark.skipif(not is_available(), reason="resiliparse not installed")
    def test_text_nonempty_when_installed(self):
        result = extract_with_resiliparse(FULL_PAGE_HTML)
        assert len(result["text"]) > 10
