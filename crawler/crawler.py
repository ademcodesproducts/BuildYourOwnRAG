"""
BFS web crawler for UCB EECS sites.

Frontier: deque (BFS order — breadth-first explores the site structure evenly).
Deduplication: normalized URL set updated before enqueue (not after fetch) to
avoid fetching the same URL twice even if it appears in many link lists.
"""

import logging
import os
import signal
from collections import deque
from urllib.parse import urljoin

from bs4 import BeautifulSoup

import config
from crawler.fetcher import Fetcher
from crawler.robots import RobotsCache
from crawler.storage import save_raw_page, load_all_meta, load_html
from crawler.url_filter import normalize_url, should_crawl

logger = logging.getLogger(__name__)


def _frontier_path(raw_dir: str) -> str:
    return raw_dir.rstrip("/") + ".frontier"


def _save_frontier(frontier: deque, raw_dir: str) -> None:
    path = _frontier_path(raw_dir)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(frontier))
    logger.info("Frontier saved: %d URLs → %s", len(frontier), path)


def _load_frontier(raw_dir: str) -> deque | None:
    path = _frontier_path(raw_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    logger.info("Frontier loaded: %d URLs from %s", len(urls), path)
    return deque(urls)


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

    existing = load_all_meta(raw_dir)
    for meta in existing:
        seen.add(normalize_url(meta["url"]))
    pages_crawled = len(existing)

    if pages_crawled:
        saved_frontier = _load_frontier(raw_dir)
        if saved_frontier is not None:
            # Fast path: frontier was checkpointed — re-validate each URL through
            # should_crawl so stale entries (wrong domain, filtered extensions) are dropped.
            for url in saved_frontier:
                if should_crawl(url, allowed_domains, seen):
                    seen.add(normalize_url(url))
                    frontier.append(url)
            logger.info("Resuming crawl: %d pages done, %d URLs in frontier", pages_crawled, len(frontier))
        else:
            # Fallback: reconstruct frontier by re-parsing saved HTML files
            logger.info("Resuming crawl: %d pages done — rebuilding frontier from saved HTML", pages_crawled)
            for meta in existing:
                html = load_html(meta["page_id"], raw_dir)
                if html is None:
                    continue
                for link in _extract_links(html, meta["url"]):
                    if should_crawl(link, allowed_domains, seen):
                        seen.add(normalize_url(link))
                        frontier.append(link)
            logger.info("Frontier rebuilt: %d URLs queued", len(frontier))
    else:
        for seed in seed_urls:
            norm = normalize_url(seed)
            if norm not in seen:
                seen.add(norm)
                frontier.append(seed)

    def _handle_signal(signum, frame):
        logger.info("Signal %d received — saving frontier before exit", signum)
        _save_frontier(frontier, raw_dir)
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

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

        if pages_crawled % 100 == 0:
            _save_frontier(frontier, raw_dir)

        # Extract and enqueue links
        for link in _extract_links(result.html, result.url):
            if should_crawl(link, allowed_domains, seen):
                seen.add(normalize_url(link))
                frontier.append(link)

    frontier_file = _frontier_path(raw_dir)
    if os.path.exists(frontier_file):
        os.remove(frontier_file)
        logger.info("Frontier file removed (crawl complete)")

    logger.info("Crawl complete. Pages saved: %d", pages_crawled)
    return pages_crawled
