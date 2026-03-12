"""
CLI entry point for the export stage.

Usage:
    python run_exporter.py
    python run_exporter.py --raw-dir data/raw_pages --output data/corpus.jsonl
    python run_exporter.py --raw-dir data/raw_pages data/raw_pages_www --output data/corpus_combined.jsonl
"""

import argparse
import logging
import sys

import config
from exporter.to_jsonl import export_to_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean raw pages and export to JSONL corpus."
    )
    parser.add_argument(
        "--raw-dir",
        nargs="+",
        default=[config.RAW_PAGES_DIR],
        help=f"One or more directories of raw crawled pages (default: {config.RAW_PAGES_DIR})",
    )
    parser.add_argument(
        "--output",
        default=config.CORPUS_JSONL_PATH,
        help=f"Output JSONL path (default: {config.CORPUS_JSONL_PATH})",
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

    written, skipped = export_to_jsonl(raw_dirs=args.raw_dir, output_path=args.output)
    print(f"\nDone. {written} records written, {skipped} skipped → '{args.output}'.")


if __name__ == "__main__":
    main()
