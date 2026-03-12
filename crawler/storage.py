"""
Persist raw crawled pages to disk as HTML + JSON metadata sidecar pairs.

File layout per page:
    data/raw_pages/{page_id}.html        — raw HTML bytes
    data/raw_pages/{page_id}.meta.json   — metadata (URL, timestamp, etc.)
"""

import hashlib
import json
import os
from datetime import datetime, timezone

from crawler.fetcher import FetchResult


def _page_id(url: str) -> str:
    """Stable, filesystem-safe 16-char ID derived from the canonical URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def save_raw_page(result: FetchResult, raw_dir: str) -> str:
    """
    Write the HTML and a metadata sidecar to raw_dir.

    Returns the page_id so the caller can log or track it.
    Raises if result.html is None (caller should check before calling).
    """
    if result.html is None:
        raise ValueError(f"Cannot save page with no HTML content: {result.url}")

    os.makedirs(raw_dir, exist_ok=True)
    page_id = _page_id(result.url)

    html_path = os.path.join(raw_dir, f"{page_id}.html")
    meta_path = os.path.join(raw_dir, f"{page_id}.meta.json")

    with open(html_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(result.html)

    meta = {
        "page_id": page_id,
        "url": result.url,
        "original_url": result.original_url,
        "status_code": result.status_code,
        "content_type": result.content_type,
        "crawl_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return page_id


def load_all_meta(raw_dir: str) -> list[dict]:
    """
    Load all metadata sidecar files from raw_dir.
    Returns a list of metadata dicts sorted by page_id for deterministic ordering.
    """
    if not os.path.isdir(raw_dir):
        return []

    meta_files = sorted(
        f for f in os.listdir(raw_dir) if f.endswith(".meta.json")
    )
    results = []
    for fname in meta_files:
        path = os.path.join(raw_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results


def load_html(page_id: str, raw_dir: str) -> str | None:
    """
    Load the raw HTML for a given page_id. Returns None if the file is missing.
    """
    html_path = os.path.join(raw_dir, f"{page_id}.html")
    if not os.path.exists(html_path):
        return None
    with open(html_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()
