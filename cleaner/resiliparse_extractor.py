"""
resiliparse-based HTML content extractor (primary extractor).

resiliparse uses structural DOM analysis (similar to Mozilla Readability)
rather than CSS class heuristics, making it more robust across the varied
markup generations of the UCB EECS websites.

Falls back gracefully to an empty result if resiliparse is not installed,
so the bs4 fallback in cleaner.py will handle it.
"""

from bs4 import BeautifulSoup

try:
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.html import HTMLTree
    _RESILIPARSE_AVAILABLE = True
except ImportError:
    _RESILIPARSE_AVAILABLE = False


def is_available() -> bool:
    return _RESILIPARSE_AVAILABLE


def extract_with_resiliparse(html: str) -> dict:
    """
    Extract title and main text content from raw HTML using resiliparse.

    Returns:
        {"title": str, "text": str, "extractor": "resiliparse"}
        On import error or extraction failure, text will be "".
    """
    title = _extract_title(html)

    if not _RESILIPARSE_AVAILABLE:
        return {"title": title, "text": "", "extractor": "resiliparse"}

    try:
        tree = HTMLTree.parse(html)
        text = extract_plain_text(
            tree,
            main_content=True,   # Focus on the main content block
            alt_texts=False,     # Skip image alt text (noisy)
            links=False,         # Skip raw link URLs
            form_fields=False,   # Skip form input labels
            noscript=False,      # Skip <noscript> blocks
        )
        return {"title": title, "text": (text or "").strip(), "extractor": "resiliparse"}

    except Exception:
        # If resiliparse fails on a malformed page, return empty so bs4 fallback kicks in
        return {"title": title, "text": "", "extractor": "resiliparse"}


def _extract_title(html: str) -> str:
    """Extract <title> text via BeautifulSoup (resiliparse doesn't expose it easily)."""
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return ""
