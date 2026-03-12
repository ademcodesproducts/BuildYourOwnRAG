# Plan: UCB EECS RAG Data Preparation Pipeline

## Context
ANLP Assignment 3 requires building a RAG model. The first step is compiling a retrieval corpus by
crawling UCB EECS websites, cleaning the HTML, and producing a structured file format for
downstream model development.

---

## Module Structure

```
Assn3/
├── config.py                          # All tunables (seed URLs, delays, paths, etc.)
├── crawler/
│   ├── crawler.py                     # BFS crawl loop + frontier management
│   ├── fetcher.py                     # HTTP fetch, rate limiting, retries, redirect handling
│   ├── robots.py                      # robots.txt parsing + per-domain caching
│   ├── url_filter.py                  # Domain scoping, HTML-only filter, URL normalization, dedup
│   └── storage.py                     # Write raw HTML + JSON sidecar to data/raw_pages/
├── cleaner/
│   ├── cleaner.py                     # Orchestrates extraction with resiliparse→bs4 fallback
│   ├── bs4_extractor.py               # BeautifulSoup4-based content extraction
│   └── resiliparse_extractor.py       # resiliparse-based extraction (primary)
├── exporter/
│   └── to_jsonl.py                    # Reads raw_pages/ + cleans → writes corpus.jsonl
├── tests/
│   ├── test_url_filter.py
│   ├── test_fetcher.py
│   ├── test_bs4_extractor.py
│   ├── test_resiliparse_extractor.py
│   └── test_to_jsonl.py
├── data/
│   ├── raw_pages/                     # {page_id}.html + {page_id}.meta.json (crawler output)
│   └── corpus.jsonl                   # Final RAG corpus (exporter output)
├── run_crawler.py                     # CLI: python run_crawler.py
├── run_exporter.py                    # CLI: python run_exporter.py
└── requirements.txt
```

---

## Data Flow

```
[Seed URLs] → crawler (BFS, rate-limited, robots-aware)
                  ↓
            data/raw_pages/   ← {id}.html + {id}.meta.json per page
                  ↓
            cleaner (resiliparse primary, bs4 fallback)
                  ↓
            data/corpus.jsonl ← one JSON record per cleaned page
```

Each stage is independently re-runnable.

---

## Key Design Decisions

### Crawling
- **Seed URLs**: `eecs.berkeley.edu`, `www2.eecs.berkeley.edu`, `cs.berkeley.edu`
- **Domain scoping**: strict allowlist in `ALLOWED_DOMAINS`
- **HTML-only filter**: URL extension check (pre-fetch) + `Content-Type: text/html` (post-fetch)
- **Politeness**: per-domain rate limiting (1s default); respects `robots.txt` `Crawl-delay`
- **robots.txt**: fetched once per domain, cached; unreachable → allow-all
- **Deduplication**: normalized URL (fragment stripped) in `seen` set; post-redirect check too
- **MAX_PAGES** hard cap to bound crawl size

### Cleaning
- **Primary**: `resiliparse` with `main_content=True` — structural DOM analysis
- **Fallback**: BeautifulSoup4 — tag/class heuristic removal
- **Discard threshold**: < `MIN_CONTENT_LENGTH` chars after extraction → skip page

### Output Format: JSONL
```json
{
  "id": "a3f9c2d1b4e87f60",
  "url": "https://eecs.berkeley.edu/research/areas/",
  "title": "Research Areas | EECS at UC Berkeley",
  "text": "EECS at UC Berkeley covers research in...",
  "crawl_timestamp": "2026-03-11T14:23:05+00:00",
  "content_length": 3412,
  "extractor": "resiliparse"
}
```

---

## Edge Cases

| Case | Mitigation |
|---|---|
| Infinite crawl loops | `seen` set; BFS won't re-enqueue |
| Redirect to out-of-domain | Post-redirect URL domain-checked before saving |
| PDF/JPG/binary URLs | Extension check before any network request |
| Encoding issues | `requests` uses Content-Type charset; `errors="replace"` fallback |
| Empty/login-wall pages | `MIN_CONTENT_LENGTH` threshold |
| robots.txt unreachable | Default to allow-all |
| 429 Too Many Requests | `Retry-After` header + exponential backoff |

---

## Verification Steps
1. `python run_crawler.py` with `MAX_PAGES=10` → check `data/raw_pages/` has `.html` + `.meta.json`
2. `python run_exporter.py` → check `data/corpus.jsonl` schema, no empty `text` fields
3. Spot-check 5–10 records vs. live pages
4. `pytest tests/` → all pass
5. Scale to full crawl
