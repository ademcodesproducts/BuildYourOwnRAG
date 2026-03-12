"""
Orchestrates the cleaning pipeline.

Extraction order:
1. resiliparse (primary) — structural DOM analysis
2. bs4 (fallback)       — tag/class heuristic removal

A page is discarded (returns None) if both extractors produce fewer than
MIN_CONTENT_LENGTH characters of text.
"""

import logging

import config
from cleaner.resiliparse_extractor import extract_with_resiliparse
from cleaner.bs4_extractor import extract_with_bs4

logger = logging.getLogger(__name__)


def clean_page(html: str, page_id: str = "") -> dict | None:
    """
    Clean raw HTML and return a dict with title, text, and extractor name.

    Returns None if the page has insufficient content (login walls,
    empty pages, custom error pages, etc.).

    Args:
        html:     Raw HTML string.
        page_id:  Optional identifier used in log messages.

    Returns:
        {"title": str, "text": str, "extractor": str} or None.
    """
    result = extract_with_resiliparse(html)

    if len(result["text"]) < config.MIN_CONTENT_LENGTH:
        logger.debug(
            "[%s] resiliparse short (%d chars), trying bs4 fallback",
            page_id, len(result["text"]),
        )
        result = extract_with_bs4(html)

    if len(result["text"]) < config.MIN_CONTENT_LENGTH:
        logger.debug(
            "[%s] Both extractors below MIN_CONTENT_LENGTH (%d) — discarding",
            page_id, config.MIN_CONTENT_LENGTH,
        )
        return None

    return result
