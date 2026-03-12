"""
CLI entry point for the crawler stage.

Usage:
    python run_crawler.py
    python run_crawler.py --max-pages 50
"""

import argparse
import logging
import sys

import config
from crawler.crawler import crawl


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl UCB EECS websites.")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=config.MAX_PAGES,
        help=f"Maximum pages to crawl (default: {config.MAX_PAGES})",
    )
    parser.add_argument(
        "--raw-dir",
        default=config.RAW_PAGES_DIR,
        help=f"Output directory for raw pages (default: {config.RAW_PAGES_DIR})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    count = crawl(max_pages=args.max_pages, raw_dir=args.raw_dir)
    print(f"\nDone. {count} pages saved to '{args.raw_dir}'.")


if __name__ == "__main__":
    main()
