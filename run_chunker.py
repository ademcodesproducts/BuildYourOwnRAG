"""CLI entry point for chunking the corpus."""

import argparse
import logging
import sys

from chunker.chunker import chunk_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk corpus JSONL into fixed-size passages.")
    parser.add_argument("--input", default="data/corpus_combined.jsonl", help="Input corpus JSONL path")
    parser.add_argument("--output", default="data/chunks.jsonl", help="Output chunks JSONL path")
    parser.add_argument("--chunk-size", type=int, default=200, help="Words per chunk (default: 200)")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap words between chunks (default: 50)")
    parser.add_argument("--min-doc-words", type=int, default=200, help="Docs shorter than this stay whole (default: 200)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    total_chunks, total_docs = chunk_corpus(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_doc_words=args.min_doc_words,
    )
    print(f"Done: {total_docs} documents → {total_chunks} chunks → {args.output}")


if __name__ == "__main__":
    main()
