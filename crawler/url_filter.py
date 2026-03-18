"""
URL filtering, normalization, and deduplication logic.
All URL decisions flow through this module — no URL logic elsewhere.
"""

from urllib.parse import urlparse, urlunparse, urlencode, parse_qsl


# File extensions that are definitively not HTML — skip before any network request.
NON_HTML_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".mp4", ".mp3", ".wav", ".ogg", ".wmv", ".avi", ".mov", ".mkv", ".flv",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".css", ".js", ".json", ".xml",
    ".woff", ".woff2", ".ttf", ".eot",
    ".eps", ".ps",
}


def normalize_url(url: str) -> str:
    """
    Canonical form of a URL for deduplication purposes:
    - Lowercase scheme and host
    - Strip fragment (anchors don't change page content)
    - Strip trailing slash from path (treat /foo/ == /foo)
    - Sort query parameters for stability
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    # Sort query params to treat ?b=2&a=1 == ?a=1&b=2
    query = urlencode(sorted(parse_qsl(parsed.query)))
    # Fragment is always dropped
    return urlunparse((scheme, netloc, path, parsed.params, query, ""))


def _bare_domain(netloc: str) -> str:
    """Strip port and leading 'www.' for comparison."""
    host = netloc.split(":")[0].lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def is_allowed_domain(url: str, allowed_domains: set) -> bool:
    """Return True if the URL's host is within the allowed domain set."""
    netloc = urlparse(url).netloc
    host = _bare_domain(netloc)
    for domain in allowed_domains:
        bare = _bare_domain(domain)
        if host == bare or host.endswith("." + bare):
            return True
    return False


def is_html_url(url: str) -> bool:
    """
    Return False if the URL path ends with a known non-HTML extension.
    This is a pre-fetch heuristic; Content-Type is the authoritative check.
    """
    path = urlparse(url).path.lower()
    for ext in NON_HTML_EXTENSIONS:
        if path.endswith(ext):
            return False
    return True


def should_crawl(url: str, allowed_domains: set, seen: set) -> bool:
    """
    Return True if this URL should be added to the crawl frontier.
    Checks: not seen, allowed domain, HTML-like URL.
    """
    norm = normalize_url(url)
    if norm in seen:
        return False
    if not is_allowed_domain(url, allowed_domains):
        return False
    if not is_html_url(url):
        return False
    return True
