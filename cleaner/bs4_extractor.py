"""
BeautifulSoup4-based HTML content extractor.

Strategy:
1. Remove comments, scripts, styles, and known boilerplate tags.
2. Heuristically remove nav/footer elements by class/id patterns.
3. Find the most likely main content container.
4. Extract plain text with whitespace normalization.
"""

import re
from bs4 import BeautifulSoup, Comment, Tag

# Tags whose entire subtree is boilerplate — always removed.
_REMOVE_TAGS = {
    "script", "style", "noscript", "nav", "footer", "header",
    "aside", "form", "button", "iframe", "figure", "figcaption",
    "menu", "menuitem", "template",
}

# Substrings in class/id values that strongly indicate boilerplate.
_BOILERPLATE_PATTERNS = [
    "nav", "footer", "header", "sidebar", "side-bar",
    "menu", "breadcrumb", "cookie", "banner",
    "advertisement", "ads", "ad-", "social", "share",
    "popup", "modal", "overlay", "toolbar",
]


def _is_boilerplate_element(tag: Tag) -> bool:
    if tag.attrs is None:
        return False
    for attr in ("class", "id"):
        values = tag.get(attr, [])
        if isinstance(values, str):
            values = [values]
        for val in values:
            val_lower = val.lower()
            if any(pat in val_lower for pat in _BOILERPLATE_PATTERNS):
                return True
    return False


def extract_with_bs4(html: str) -> dict:
    """
    Extract title and main text content from raw HTML.

    Returns:
        {"title": str, "text": str, "extractor": "bs4"}
    """
    soup = BeautifulSoup(html, "html.parser")

    # Title before any destructive mutations
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Remove known boilerplate tag subtrees
    for tag_name in _REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove boilerplate by class/id heuristic
    # Work on a snapshot list — decompose() modifies the tree during iteration
    for tag in list(soup.find_all(True)):
        if isinstance(tag, Tag) and _is_boilerplate_element(tag):
            tag.decompose()

    # Find main content container (most specific first)
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile(r"\bcontent\b|\bmain\b", re.I))
        or soup.find(class_=re.compile(r"\bcontent\b|\bmain\b", re.I))
        or soup.find("div", id="page")
        or soup.body
    )

    if main is None:
        return {"title": title, "text": "", "extractor": "bs4"}

    text = main.get_text(separator="\n", strip=True)
    # Collapse 3+ consecutive newlines to two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse 3+ spaces to one
    text = re.sub(r" {3,}", " ", text)

    return {"title": title, "text": text.strip(), "extractor": "bs4"}
