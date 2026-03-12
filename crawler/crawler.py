"""
BFS web crawler for UCB EECS sites.

Frontier: deque (BFS order — breadth-first explores the site structure evenly).
Deduplication: normalized URL set updated before enqueue (not after fetch) to
avoid fetching the same URL twice even if it appears in many link lists.
"""

import logging
from collections import deque
from urllib.parse import urljoin

from bs4 import BeautifulSoup

import config
from crawler.fetcher import Fetcher
from crawler.robots import RobotsCache
from crawler.storage import save_raw_page
from crawler.url_filter import normalize_url, should_crawl

logger = logging.getLogger(__name__)


def _extract_links(html: str, base_url: str) -> list[str]:
    """Return all absolute hrefs found in <a> tags."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        absolute = urljoin(base_url, href)
        links.append(absolute)
    return links


def crawl(
    seed_urls: list[str] = config.SEED_URLS,
    allowed_domains: set = config.ALLOWED_DOMAINS,
    max_pages: int = config.MAX_PAGES,
    raw_dir: str = config.RAW_PAGES_DIR,
) -> int:
    """
    Crawl UCB EECS pages via BFS, saving raw HTML to raw_dir.

    Returns the number of pages successfully crawled.
    """
    fetcher = Fetcher()
    robots_cache = RobotsCache(user_agent=config.USER_AGENT)

    seen: set[str] = set()
    frontier: deque[str] = deque()

    for seed in seed_urls:
        norm = normalize_url(seed)
        if norm not in seen:
            seen.add(norm)
            frontier.append(seed)

    pages_crawled = 0

    while frontier and pages_crawled < max_pages:
        url = frontier.popleft()
        result = fetcher.fetch(url, robots_cache)

        if result.error or result.html is None:
            logger.info("[SKIP] %s — %s", url, result.error)
            continue

        # Post-redirect deduplication: the final URL may differ from requested
        final_norm = normalize_url(result.url)
        if final_norm in seen and final_norm != normalize_url(url):
            logger.debug("[REDIRECT-DUP] %s -> %s", url, result.url)
            continue
        seen.add(final_norm)

        page_id = save_raw_page(result, raw_dir)
        pages_crawled += 1
        logger.info("[OK] (%d/%d) %s [id=%s]", pages_crawled, max_pages, result.url, page_id)

        # Extract and enqueue links
        for link in _extract_links(result.html, result.url):
            if should_crawl(link, allowed_domains, seen):
                seen.add(normalize_url(link))
                frontier.append(link)

    logger.info("Crawl complete. Pages saved: %d", pages_crawled)
    return pages_crawled
